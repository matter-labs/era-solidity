// This file is part of solidity.

// solidity is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// solidity is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with solidity.  If not, see <http://www.gnu.org/licenses/>.

// SPDX-License-Identifier: GPL-3.0

//
// Solidity to MLIR pass
//

#include "libevmasm/GasMeter.h"
#include "liblangutil/CharStream.h"
#include "liblangutil/EVMVersion.h"
#include "liblangutil/Exceptions.h"
#include "liblangutil/SourceLocation.h"
#include "libsolidity/ast/AST.h"
#include "libsolidity/ast/ASTEnums.h"
#include "libsolidity/ast/ASTForward.h"
#include "libsolidity/ast/ASTUtils.h"
#include "libsolidity/ast/Types.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Sol/Sol.h"
#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolidity/codegen/mlir/Yul/Yul.h"
#include "libsolidity/interface/CompilerStack.h"
#include "libsolutil/CommonIO.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "range/v3/view/zip.hpp"
#include "llvm-c/Core.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ThreadPool.h"
#include <mutex>
#include <string>

using namespace solidity::langutil;
using namespace solidity::frontend;

namespace solidity::frontend {

class SolidityToMLIRPass {
public:
  explicit SolidityToMLIRPass(mlir::MLIRContext &ctx, EVMVersion evmVersion,
                              bool genUnkLoc)
      : b(&ctx), evmVersion(evmVersion), genUnkLoc(genUnkLoc) {}

  /// Lowers the free functions in the source unit.
  void lowerFreeFuncs(SourceUnit const &);

  /// Lowers the contract.
  void lower(ContractDefinition const &);

  /// Initializes (or resets) the module and the insertion-point.
  void init(std::shared_ptr<CharStream> s) {
    stream = std::move(s);
    mod = mlir::ModuleOp::create(b.getUnknownLoc());
    mod->setAttr("sol.evm_version",
                 mlir::sol::EvmVersionAttr::get(
                     b.getContext(), *mlir::sol::symbolizeEvmVersion(
                                         evmVersion.getVersionAsInt())));
    b.setInsertionPointToEnd(mod.getBody());
  }

  /// Returns the ModuleOp
  mlir::ModuleOp getModule() { return mod; }

private:
  mlir::OpBuilder b;
  std::shared_ptr<CharStream> stream;
  EVMVersion evmVersion;
  mlir::ModuleOp mod;

  /// The contract being lowered.
  ContractDefinition const *currContract = nullptr;

  /// Maps a local variable to its address.
  std::map<VariableDeclaration const *, mlir::Value> localVarAddrMap;

  /// Maps an interface function or state variable getter to its selector.
  std::map<Declaration const *, util::FixedHash<4>> selectorMap;

  /// True if the current block is unchecked.
  bool inUnchecked = false;

  /// Tracks if the codegen is generating a constructor.
  bool inCtor = false;

  /// Forces generated locations to be unknown. FIXME: This is to avoid the slow
  /// translatePositionToLineColumn
  bool genUnkLoc;

  /// Returns the mlir location for the solidity source location `loc`
  mlir::Location getLoc(SourceLocation const &loc) {
    if (genUnkLoc)
      return b.getUnknownLoc();
    // TODO: Cache the translatePositionToLineColumn results. (Ideally, the
    // lexer + parser should record this instead-of/along-with the existing
    // linear offset)
    //
    // FIXME: Track loc.end as well.
    LineColumn lineCol = stream->translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream->name()),
                                     lineCol.line, lineCol.column);
  }

  mlir::Location getLoc(ASTNode const &ast) { return getLoc(ast.location()); }

  /// Returns the corresponding mlir type for the solidity type `ty`.
  mlir::Type getType(Type const *ty);

  /// Tracks the address of the local variable.
  void trackLocalVarAddr(VariableDeclaration const &decl, mlir::Value addr) {
    localVarAddrMap[&decl] = addr;
  }

  /// Returns the address of the local variable.
  mlir::Value getLocalVarAddr(VariableDeclaration const &decl) {
    auto it = localVarAddrMap.find(&decl);
    assert(it != localVarAddrMap.end());
    return it->second;
  }

  /// Returns the mangled name of the declaration composed of its name and its
  /// AST ID.
  std::string getMangledName(Declaration const &decl) {
    return decl.name() + "_" + std::to_string(decl.id());
  }

  mlir::Value genStateVarRef(VariableDeclaration const &var,
                             bool inCreationContext) {
    auto currContr =
        b.getBlock()->getParentOp()->getParentOfType<mlir::sol::ContractOp>();
    assert(currContr);

    if (var.isConstant()) {
      // TODO: Should we track the state variable name in the sol.constant?
      return genRValExpr(*var.value(), getType(var.type()));
    }

    if (var.immutable()) {
      auto immOp =
          currContr.lookupSymbol<mlir::sol::ImmutableOp>(getMangledName(var));
      assert(immOp);
      assert(!mlir::sol::isNonPtrRefType(immOp.getType()));
      if (!inCreationContext)
        return b.create<mlir::sol::LoadImmutableOp>(
            immOp.getLoc(), immOp.getType(), immOp.getName());
      mlir::Type addrTy = mlir::sol::PointerType::get(
          b.getContext(), immOp.getType(), mlir::sol::DataLocation::Immutable);
      return b.create<mlir::sol::AddrOfOp>(immOp.getLoc(), addrTy,
                                           immOp.getName());
    }

    auto stateVarOp =
        currContr.lookupSymbol<mlir::sol::StateVarOp>(getMangledName(var));
    assert(stateVarOp);
    mlir::Type addrTy;
    if (mlir::sol::isNonPtrRefType(stateVarOp.getType()))
      addrTy = stateVarOp.getType();
    else
      addrTy = mlir::sol::PointerType::get(b.getContext(), stateVarOp.getType(),
                                           mlir::sol::DataLocation::Storage);
    // TODO: Should we use the state variable's location here?
    return b.create<mlir::sol::AddrOfOp>(stateVarOp.getLoc(), addrTy,
                                         stateVarOp.getName());
  }

  mlir::sol::FuncOp genGetter(VariableDeclaration const &stateVar) {
    assert(stateVar.isStateVariable());
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    mlir::Location loc = getLoc(stateVar);

    // Create the function.
    auto astFnTy = FunctionType(stateVar);
    auto fnTy = cast<mlir::FunctionType>(getType(&astFnTy));
    auto fn = b.create<mlir::sol::FuncOp>(
        loc, "get_" + getMangledName(stateVar), fnTy);
    assert(selectorMap.find(&stateVar) != selectorMap.end());
    fn.setSelectorAttr(
        b.getIntegerAttr(b.getIntegerType(32),
                         mlir::APInt(32, selectorMap[&stateVar].hex(), 16)));
    fn.setOrigFnTypeAttr(mlir::TypeAttr::get(fnTy));
    fn.setStateMutability(mlir::sol::StateMutability::NonPayable);

    mlir::Block *entryBlk = b.createBlock(&fn.getRegion());
    b.setInsertionPointToStart(entryBlk);

    // Load the state variable.
    mlir::Value stateVarLd;
    if (stateVar.isConstant()) {
      stateVarLd = genRValExpr(*stateVar.value(), getType(stateVar.type()));
    } else {
      mlir::Value stateVarRef =
          genStateVarRef(stateVar, /*inCreationContext=*/false);
      stateVarLd = genRValExpr(stateVarRef, stateVarRef.getLoc());
    }

    // Array type
    if (isa<mlir::sol::ArrayType>(stateVarLd.getType())) {
      mlir::Value ret = stateVarLd;
      for (auto inpTy : fnTy.getInputs()) {
        mlir::BlockArgument blkArg = entryBlk->addArgument(inpTy, loc);
        auto gep = b.create<mlir::sol::GepOp>(loc, ret, blkArg);
        ret = genRValExpr(gep, loc);
      }
      b.create<mlir::sol::ReturnOp>(loc, ret);

      // Mapping type
    } else if (auto mappingTy =
                   dyn_cast<mlir::sol::MappingType>(stateVarLd.getType())) {
      mlir::Value lastMap = stateVarLd;
      for (auto inpTy : fnTy.getInputs()) {
        auto lastMapTy = cast<mlir::sol::MappingType>(lastMap.getType());
        mlir::Type addrTy = lastMapTy.getValType();
        if (!mlir::sol::isNonPtrRefType(lastMapTy.getValType()))
          addrTy = mlir::sol::PointerType::get(
              b.getContext(), lastMapTy.getValType(),
              mlir::sol::DataLocation::Storage);
        mlir::BlockArgument blkArg = entryBlk->addArgument(inpTy, loc);
        auto map = b.create<mlir::sol::MapOp>(loc, addrTy, lastMap, blkArg);
        lastMap = genRValExpr(map, loc);
      }
      b.create<mlir::sol::ReturnOp>(loc, genRValExpr(lastMap, loc));

      // Struct type
    } else if (auto structTy =
                   dyn_cast<mlir::sol::StructType>(stateVarLd.getType())) {
      mlir::SmallVector<mlir::Value, 4> tuple;
      int64_t i = 0;
      for (auto memTy : structTy.getMemberTypes()) {
        if (mlir::sol::isNonPtrRefType(memTy))
          llvm_unreachable("NYI");
        auto gep = b.create<mlir::sol::GepOp>(
            loc, stateVarLd, genUnsignedConst(i++, /*numBits=*/64, loc));
        tuple.push_back(genRValExpr(gep, loc));
      }

      b.create<mlir::sol::ReturnOp>(loc, tuple);

      // Scalar, string types etc.
    } else {
      b.create<mlir::sol::ReturnOp>(loc, stateVarLd);
    }

    return fn;
  }

  /// Generates the ir to zero the allocation.
  void genZeroedVal(mlir::sol::AllocaOp addr);

  /// Generates a integeral constant op.
  mlir::Value genUnsignedConst(uint64_t val, unsigned numBits,
                               mlir::Location loc) {
    return b.create<mlir::sol::ConstantOp>(
        loc,
        b.getIntegerAttr(b.getIntegerType(numBits, /*isSigned=*/false), val));
  }

  /// Generates type cast expression.
  mlir::Value genCast(mlir::Value val, mlir::Type dstTy);

  /// Returns the mlir expression for the literal.
  mlir::Value genExpr(Literal const &lit);

  /// Returns the mlir expression for the identifier in an l-value context.
  mlir::Value genExpr(Identifier const &ident);

  /// Returns the mlir expression for the index access in an l-value context.
  mlir::Value genExpr(IndexAccess const &idxAcc);

  /// Returns the mlir expression for the member access in an r-value context.
  mlir::Value genExpr(MemberAccess const &memberAcc);

  /// Returns the mlir expression for the binary operation.
  mlir::Value genBinExpr(Token op, mlir::Value lhs, mlir::Value rhs,
                         mlir::Location loc);

  /// Returns the mlir expression for the unary operation.
  mlir::Value genExpr(UnaryOperation const &unaryOp);

  /// Returns the mlir expression for the binary operation.
  mlir::Value genExpr(BinaryOperation const &binOp);

  /// Returns the mlir expression for the call.
  mlir::SmallVector<mlir::Value> genExprs(FunctionCall const &call);

  /// Returns the mlir expression for the tuple.
  mlir::SmallVector<mlir::Value> genExprs(TupleExpression const &tuple);

  // We can't completely rely on ExpressionAnnotation::isLValue here since the
  // TypeChecker doesn't, for instance, tag RHS expression of an assignment as
  // an r-value.

  /// Returns the mlir expression in an l-value context.
  mlir::Value genLValExpr(Expression const &expr);
  mlir::SmallVector<mlir::Value> genLValExprs(Expression const &expr);

  /// Returns the mlir expression in an r-value context and optionally casts it
  /// to the corresponding mlir type of `resTy`.
  mlir::Value genRValExpr(Expression const &expr,
                          std::optional<mlir::Type> resTy = std::nullopt);
  mlir::Value genRValExpr(mlir::Value val, mlir::Location loc);
  mlir::SmallVector<mlir::Value> genRValExprs(Expression const &expr);

  /// Generates an ir that assigns `rhs` to `lhs`.
  void genAssign(mlir::Value lhs, mlir::Value rhs, mlir::Location loc);

  /// Lowers the expression statement.
  void lower(ExpressionStatement const &);

  /// Lowers the emit statement.
  void lower(EmitStatement const &);

  /// Lowers the break statement.
  void lower(Break const &);

  /// Lowers the continue statement.
  void lower(Continue const &);

  /// Lowers the placeholder statement.
  void lower(PlaceholderStatement const &);

  /// Lowers the return statement.
  void lower(Return const &);

  /// Lowers the assignment statement.
  void lower(Assignment const &);

  /// Lowers the variable declaration statement.
  void lower(VariableDeclarationStatement const &);

  /// Lowers the if-then-else statement.
  void lower(IfStatement const &);

  /// Lowers the while/do-while statement.
  void lower(WhileStatement const &);

  /// Lowers the for statement.
  void lower(ForStatement const &);

  /// Lowers the try statement.
  void lower(TryStatement const &);

  /// Lowers the inline asm statement.
  void lower(InlineAssembly const &);

  /// Lower the statement.
  void lower(Statement const &);

  /// Lowers the block.
  void lower(Block const &);

  /// Lowers the modifier definition.
  void lower(ModifierDefinition const &);

  /// Lowers the function definition.
  mlir::sol::FuncOp lower(FunctionDefinition const &);
};

} // namespace solidity::frontend

/// Returns the mlir::sol::DataLocation of the type
static mlir::sol::DataLocation getDataLocation(ReferenceType const *ty) {
  switch (ty->location()) {
  case DataLocation::CallData:
    return mlir::sol::DataLocation::CallData;
  case DataLocation::Storage:
    return mlir::sol::DataLocation::Storage;
  case DataLocation::Memory:
    return mlir::sol::DataLocation::Memory;
  case DataLocation::Transient:
    llvm_unreachable("NYI");
  }
}

mlir::Type SolidityToMLIRPass::getType(Type const *ty) {
  switch (ty->category()) {
  case Type::Category::Bool:
    return b.getIntegerType(/*width=*/1);

  case Type::Category::Integer: {
    const auto *intTy = static_cast<IntegerType const *>(ty);
    return b.getIntegerType(intTy->numBits(), intTy->isSigned());
  }
  case Type::Category::RationalNumber: {
    const auto *ratNumTy = static_cast<RationalNumberType const *>(ty);
    if (ratNumTy->isFractional())
      llvm_unreachable("NYI: Fractional type");
    const IntegerType *intTy = ratNumTy->integerType();
    return b.getIntegerType(intTy->numBits(), intTy->isSigned());
  }
  case Type::Category::Address:
    // FIXME: 256 -> 160
    return b.getIntegerType(256, /*isSigned=*/false);

  case Type::Category::FixedBytes: {
    const auto *fixedBytesTy = static_cast<FixedBytesType const *>(ty);
    return mlir::sol::BytesType::get(b.getContext(), fixedBytesTy->numBytes());
  }
  case Type::Category::Mapping: {
    auto *mappingTy = static_cast<MappingType const *>(ty);
    return mlir::sol::MappingType::get(b.getContext(),
                                       getType(mappingTy->keyType()),
                                       getType(mappingTy->valueType()));
  }
  case Type::Category::Array: {
    // Array or string type
    const auto *arrTy = static_cast<ArrayType const *>(ty);
    if (arrTy->isByteArrayOrString())
      return mlir::sol::StringType::get(b.getContext(), getDataLocation(arrTy));
    mlir::Type eltTy = getType(arrTy->baseType());

    // TODO: Does convert_to alreay do this?
    assert(arrTy->length() <= INT64_MAX);
    int64_t size = arrTy->isDynamicallySized()
                       ? -1
                       : arrTy->length().convert_to<int64_t>();
    return mlir::sol::ArrayType::get(b.getContext(), size, eltTy,
                                     getDataLocation(arrTy));
  }
  case Type::Category::Struct: {
    const auto *structTy = static_cast<StructType const *>(ty);
    std::vector<mlir::Type> memberTys;
    for (const auto &mem : structTy->nativeMembers(nullptr)) {
      memberTys.push_back(getType(mem.type));
    }

    return mlir::sol::StructType::get(b.getContext(), memberTys,
                                      getDataLocation(structTy));
  }
  case Type::Category::Function: {
    const auto *fnTy = static_cast<FunctionType const *>(ty);
    std::vector<mlir::Type> inTys, outTys;

    inTys.reserve(fnTy->parameterTypes().size());
    for (Type const *inTy : fnTy->parameterTypes())
      inTys.push_back(getType(inTy));

    outTys.reserve(fnTy->returnParameterTypes().size());
    for (Type const *outTy : fnTy->returnParameterTypes())
      outTys.push_back(getType(outTy));

    return b.getFunctionType(inTys, outTys);
  }
  case Type::Category::Contract:
    // FIXME: 256 -> 160
    return b.getIntegerType(256, /*isSigned=*/false);
  default:
    break;
  }

  llvm_unreachable("NYI");
}

mlir::Value SolidityToMLIRPass::genExpr(Identifier const &id) {
  Declaration const *decl = id.annotation().referencedDeclaration;

  if (MagicVariableDeclaration const *magicVar =
          dynamic_cast<MagicVariableDeclaration const *>(decl)) {
    switch (magicVar->type()->category()) {
    case Type::Category::Contract:
      assert(id.name() == "this");
      return b.create<mlir::sol::ThisOp>(getLoc(id));
    default:
      break;
    }
    llvm_unreachable("NYI");
  }

  if (const auto *var = dynamic_cast<VariableDeclaration const *>(decl)) {
    if (var->isStateVariable())
      return genStateVarRef(*var, inCtor);
    return getLocalVarAddr(*var);
  }

  if (const auto *contr = dynamic_cast<ContractDefinition const *>(decl)) {
    assert(contr->isLibrary() && "NYI");
    return b.create<mlir::sol::LibAddrOp>(getLoc(id),
                                          contr->fullyQualifiedName());
  }

  llvm_unreachable("NYI");
}

void SolidityToMLIRPass::genZeroedVal(mlir::sol::AllocaOp addr) {
  mlir::Location loc = addr.getLoc();

  auto pointeeTy =
      mlir::cast<mlir::sol::PointerType>(addr.getType()).getPointeeType();

  mlir::Value val;
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(pointeeTy)) {
    val = b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(intTy, llvm::APInt(intTy.getWidth(), 0)));

  } else if (auto arrTy = mlir::dyn_cast<mlir::sol::ArrayType>(pointeeTy)) {
    val = b.create<mlir::sol::MallocOp>(loc, arrTy, /*zeroInit=*/true,
                                        /*size=*/mlir::Value{});

  } else if (auto structTy = mlir::dyn_cast<mlir::sol::StructType>(pointeeTy)) {
    val = b.create<mlir::sol::MallocOp>(loc, structTy, /*zeroInit=*/true,
                                        /*size=*/mlir::Value{});

  } else if (auto stringTy = mlir::dyn_cast<mlir::sol::StringType>(pointeeTy)) {
    // TODO: Do we need to zero-init here?
    val = b.create<mlir::sol::MallocOp>(loc, stringTy, /*zeroInit=*/false,
                                        /*size=*/mlir::Value{});
  }
  assert(val);

  b.create<mlir::sol::StoreOp>(loc, val, addr);
}

mlir::Value SolidityToMLIRPass::genCast(mlir::Value val, mlir::Type dstTy) {
  mlir::Location loc = val.getLoc();
  mlir::Type srcTy = val.getType();

  // Don't cast if we're casting to the same type.
  if (srcTy == dstTy)
    return val;

  if (mlir::isa<mlir::sol::BytesType>(srcTy) ||
      mlir::isa<mlir::sol::BytesType>(dstTy))
    return b.create<mlir::sol::BytesCastOp>(loc, dstTy, val);

  // Casting to integer type.
  if (mlir::isa<mlir::IntegerType>(dstTy))
    return b.create<mlir::sol::CastOp>(loc, dstTy, val);

  // Casting between reference types (excluding pointer types).
  if (mlir::sol::isNonPtrRefType(dstTy)) {
    assert(mlir::sol::isNonPtrRefType(srcTy));
    return b.create<mlir::sol::DataLocCastOp>(loc, dstTy, val);
  }

  llvm_unreachable("NYI or invalid cast");
}

mlir::Value SolidityToMLIRPass::genExpr(Literal const &lit) {
  mlir::Location loc = getLoc(lit);
  Type const *ty = lit.annotation().type;

  // Bool literal
  if (dynamic_cast<BoolType const *>(ty))
    return b.create<mlir::sol::ConstantOp>(
        loc, b.getBoolAttr(lit.token() == Token::TrueLiteral));

  // Rational number literal
  if (const auto *ratNumTy = dynamic_cast<RationalNumberType const *>(ty)) {
    if (ratNumTy->isFractional())
      llvm_unreachable("NYI: Fractional literal");

    auto *intTy = ratNumTy->integerType();
    u256 val = ty->literalValue(nullptr);
    // TODO: Is there a faster way to convert boost::multiprecision::number to
    // llvm::APInt?
    return b.create<mlir::sol::ConstantOp>(
        loc,
        b.getIntegerAttr(getType(ty), llvm::APInt(intTy->numBits(), val.str(),
                                                  /*radix=*/10)));
  }

  llvm_unreachable("NYI: Literal");
}

mlir::Value SolidityToMLIRPass::genBinExpr(Token op, mlir::Value lhs,
                                           mlir::Value rhs,
                                           mlir::Location loc) {
  switch (op) {
  case Token::Add:
    if (inUnchecked)
      return b.create<mlir::sol::AddOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CAddOp>(loc, lhs, rhs);
  case Token::Sub:
    if (inUnchecked)
      return b.create<mlir::sol::SubOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CSubOp>(loc, lhs, rhs);
  case Token::Mul:
    if (inUnchecked)
      return b.create<mlir::sol::MulOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CMulOp>(loc, lhs, rhs);
  case Token::Div:
    if (inUnchecked)
      return b.create<mlir::sol::DivOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CDivOp>(loc, lhs, rhs);
  case Token::Mod:
    return b.create<mlir::sol::ModOp>(loc, lhs, rhs);
  case Token::Exp:
    if (inUnchecked)
      return b.create<mlir::sol::ExpOp>(loc, lhs, rhs);
    break;
  case Token::Equal:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::eq, lhs,
                                      rhs);
  case Token::NotEqual:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::ne, lhs,
                                      rhs);
  case Token::LessThan:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::lt, lhs,
                                      rhs);
  case Token::LessThanOrEqual:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::le, lhs,
                                      rhs);
  case Token::GreaterThan:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::gt, lhs,
                                      rhs);
  case Token::GreaterThanOrEqual:
    return b.create<mlir::sol::CmpOp>(loc, mlir::sol::CmpPredicate::ge, lhs,
                                      rhs);
  default:
    break;
  }
  llvm_unreachable("NYI: Binary operator");
}

mlir::Value SolidityToMLIRPass::genExpr(UnaryOperation const &unaryOp) {
  mlir::Location loc = getLoc(unaryOp);

  assert(!*unaryOp.annotation().userDefinedFunction && "NYI");

  Type const *ty = unaryOp.annotation().type;
  mlir::Type mlirTy = getType(ty);

  // Negative constant
  if (ty->category() == Type::Category::RationalNumber) {
    auto intTy = mlir::cast<mlir::IntegerType>(mlirTy);
    u256 val = ty->literalValue(nullptr);
    return b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(intTy, mlirgen::getAPInt(val, intTy.getWidth())));
  }

  switch (unaryOp.getOperator()) {
  // Increment and decrement
  case Token::Inc:
  case Token::Dec: {
    mlir::Value lValExpr = genLValExpr(unaryOp.subExpression());
    mlir::Value rValExpr = genRValExpr(lValExpr, lValExpr.getLoc());
    mlir::Value one =
        b.create<mlir::sol::ConstantOp>(loc, b.getIntegerAttr(mlirTy, 1));
    mlir::Value newVal = genBinExpr(
        unaryOp.getOperator() == Token::Inc ? Token::Add : Token::Sub, rValExpr,
        one, loc);
    b.create<mlir::sol::StoreOp>(loc, newVal, lValExpr);
    return unaryOp.isPrefixOperation() ? newVal : rValExpr;
  }
  // Negation
  case Token::Sub: {
    mlir::Value expr = genRValExpr(unaryOp.subExpression());
    mlir::Value zero =
        b.create<mlir::sol::ConstantOp>(loc, b.getIntegerAttr(mlirTy, 0));
    return genBinExpr(Token::Sub, zero, expr, loc);
  }
  default:
    break;
  }

  llvm_unreachable("NYI");
}

mlir::Value SolidityToMLIRPass::genExpr(BinaryOperation const &binOp) {
  mlir::Type argTy = getType(binOp.annotation().commonType);
  auto loc = getLoc(binOp);

  mlir::Value lhs = genRValExpr(binOp.leftExpression(), argTy);
  mlir::Value rhs = genRValExpr(binOp.rightExpression(), argTy);

  return genBinExpr(binOp.getOperator(), lhs, rhs, loc);
}

mlir::Value SolidityToMLIRPass::genExpr(IndexAccess const &idxAcc) {
  mlir::Location loc = getLoc(idxAcc);

  mlir::Value baseExpr = genRValExpr(idxAcc.baseExpression());
  mlir::Value idxExpr = genRValExpr(*idxAcc.indexExpression());

  // Mapping
  if (auto mappingTy =
          mlir::dyn_cast<mlir::sol::MappingType>(baseExpr.getType())) {
    mlir::Type addrTy;
    if (mlir::sol::isNonPtrRefType(mappingTy.getValType()))
      addrTy = mappingTy.getValType();
    else
      addrTy =
          mlir::sol::PointerType::get(b.getContext(), mappingTy.getValType(),
                                      mlir::sol::DataLocation::Storage);
    return b.create<mlir::sol::MapOp>(loc, addrTy, baseExpr, idxExpr);
  }

  // Bytes/array indexing
  if (mlir::isa<mlir::sol::ArrayType>(baseExpr.getType()) ||
      mlir::isa<mlir::sol::StringType>(baseExpr.getType()))
    return b.create<mlir::sol::GepOp>(loc, baseExpr, idxExpr);

  llvm_unreachable("Invalid IndexAccess");
}

mlir::Value SolidityToMLIRPass::genExpr(MemberAccess const &memberAcc) {
  mlir::Location loc = getLoc(memberAcc);

  const Type *memberAccTy = memberAcc.expression().annotation().type;
  const ASTString &memberName = memberAcc.memberName();
  switch (memberAccTy->category()) {
  case Type::Category::Magic:
    if (memberName == "sender") {
      // FIXME: sol.caller yields an i256 instead of an address.
      auto callerOp = b.create<mlir::yul::CallerOp>(loc);
      return b.create<mlir::sol::ConvCastOp>(
          loc, b.getIntegerType(256, /*isSigned=*/false), callerOp);
    }
    if (memberName == "data")
      return b.create<mlir::sol::GetCallDataOp>(loc);
    break;

  case Type::Category::Contract:
    return {};

  case Type::Category::Array:
    if (memberName == "length")
      return b.create<mlir::sol::LengthOp>(loc,
                                           genRValExpr(memberAcc.expression()));
    break;
  case Type::Category::Struct: {
    const auto *structTy = dynamic_cast<StructType const *>(memberAccTy);
    auto memberIdx = genUnsignedConst(structTy->index(memberAcc.memberName()),
                                      /*numBits=*/64, loc);
    return b.create<mlir::sol::GepOp>(loc, genRValExpr(memberAcc.expression()),
                                      memberIdx);
  }
  default:
    break;
  case Type::Category::Address: {
    if (memberName == "code") {
      return b.create<mlir::sol::CodeOp>(loc,
                                         genRValExpr(memberAcc.expression()));
    }
  }
  }

  llvm_unreachable("NYI");
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genExprs(FunctionCall const &call) {
  assert(*call.annotation().kind != FunctionCallKind::StructConstructorCall &&
         "NYI");
  mlir::SmallVector<mlir::Value, 2> resVals;

  // Type conversion
  if (*call.annotation().kind == FunctionCallKind::TypeConversion) {
    resVals.push_back(genRValExpr(*call.arguments().front(),
                                  getType(call.annotation().type)));
    return resVals;
  }

  const auto *calleeTy =
      dynamic_cast<FunctionType const *>(call.expression().annotation().type);
  assert(calleeTy);

  std::vector<Type const *> argTys = calleeTy->parameterTypes();
  std::vector<ASTPointer<Expression const>> const &astArgs =
      call.sortedArguments();

  mlir::Location loc = getLoc(call);
  switch (calleeTy->kind()) {
  // Internal call
  case FunctionType::Kind::Internal: {
    // Get callee.
    FunctionDefinition const *callee = nullptr;
    if (currContract)
      callee = ASTNode::resolveFunctionCall(call, currContract);
    else
      callee = dynamic_cast<FunctionDefinition const *>(
          ASTNode::referencedDeclaration(call.expression()));
    assert(callee && "NYI: Internal function dispatch");
    bool calleeInLib = callee->annotation().contract &&
                       callee->annotation().contract->isLibrary();
    bool currContrNotInLib = !currContract || !currContract->isLibrary();
    bool freeCallee = !callee->annotation().contract;
    if ((calleeInLib && currContrNotInLib) || (freeCallee && currContract)) {
      mlir::OpBuilder::InsertionGuard insertGuard(b);
      auto *parentOp = b.getInsertionBlock()->getParentOp();
      if (!mlir::isa<mlir::sol::FuncOp>(parentOp))
        parentOp = parentOp->getParentOfType<mlir::sol::FuncOp>();
      assert(parentOp);
      b.setInsertionPoint(parentOp);
      lower(*callee);
    }

    // Lower args.
    std::vector<mlir::Value> args;
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes())) {
      args.push_back(genRValExpr(*arg, getType(dstTy)));
    }

    // Collect return types.
    std::vector<mlir::Type> resTys;
    for (Type const *ty : calleeTy->returnParameterTypes()) {
      resTys.push_back(getType(ty));
    }

    // Generate the call op.
    auto callOp =
        b.create<mlir::sol::CallOp>(loc, getMangledName(*callee), resTys, args);
    for (mlir::Value val : callOp.getResults())
      resVals.push_back(val);
    return resVals;
  }

  // External call
  case FunctionType::Kind::External:
  case FunctionType::Kind::DelegateCall: {
    // Handle the FunctionCallOptions case.
    Expression const *callExpr = &call.expression();
    mlir::Value gas, value;
    mlir::Type ui256Ty = b.getIntegerType(256, /*isSigned=*/false);
    if (const auto *fnCallOpt =
            dynamic_cast<FunctionCallOptions const *>(&call.expression())) {
      for (const auto &[namePtr, exprPtr] :
           llvm::zip(fnCallOpt->names(), fnCallOpt->options())) {
        ASTString const &name = *namePtr;
        Expression const &expr = *exprPtr;
        mlir::Value loweredExpr = genRValExpr(expr, ui256Ty);
        if (name == "gas")
          gas = loweredExpr;
        else if (name == "value")
          value = loweredExpr;
      }
      callExpr = &fnCallOpt->expression();
    }

    // Generate the address.
    const auto *memberAcc = dynamic_cast<MemberAccess const *>(callExpr);
    assert(memberAcc);
    mlir::Value addr = genLValExpr(memberAcc->expression());

    // Get the callee and the selector.
    const auto *callee = dynamic_cast<FunctionDefinition const *>(
        memberAcc->annotation().referencedDeclaration);
    assert(callee && "NYI: State variable getters");
    auto selector =
        FunctionType(*callee).externalIdentifier().convert_to<uint32_t>();

    // Lower the args.
    std::vector<mlir::Value> args;
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes())) {
      args.push_back(genRValExpr(*arg, getType(dstTy)));
    }

    // Collect the return types.
    // TODO: The builder should prepend the bool type for the status flag here.
    std::vector<mlir::Type> resTys{b.getI1Type()};
    for (Type const *ty : calleeTy->returnParameterTypes()) {
      resTys.push_back(getType(ty));
    }

    // FIXME: Don't use signless int operands.
    mlirgen::BuilderExt bExt(b, loc);

    // Generate gas.
    if (!gas) {
      if (evmVersion.canOverchargeGasForCall()) {
        gas = b.create<mlir::yul::GasOp>(loc);
      } else {
        u256 gasNeededByCaller = evmasm::GasCosts::callGas(evmVersion) + 10;
        size_t encodedHeadSize = 0;
        for (Type const *ty : calleeTy->returnParameterTypes())
          encodedHeadSize += ty->decodingType()->calldataHeadSize();
        if (encodedHeadSize == 0 || !evmVersion.supportsReturndata())
          gasNeededByCaller += evmasm::GasCosts::callNewAccountGas;
        gas =
            b.create<mlir::arith::SubIOp>(loc, b.create<mlir::yul::GasOp>(loc),
                                          bExt.genI256Const(gasNeededByCaller));
      }
    }
    gas = b.create<mlir::sol::ConvCastOp>(loc, ui256Ty, gas);

    // Generate value.
    if (!value) {
      value = genUnsignedConst(0, /*numBits=*/256, loc);
    }

    // Generate the external call.
    auto callOp = b.create<mlir::sol::ExtCallOp>(
        loc, resTys, getMangledName(*callee), args, addr, gas, value,
        /*tryCall=*/call.annotation().tryCall,
        /*staticCall=*/calleeTy->stateMutability() <= StateMutability::View,
        /*delegateCall=*/calleeTy->kind() == FunctionType::Kind::DelegateCall,
        selector,
        /*calleeType=*/mlir::cast<mlir::FunctionType>(getType(calleeTy)));
    for (mlir::Value val : llvm::drop_begin(callOp.getResults()))
      resVals.push_back(val);
    return resVals;
  }

  case FunctionType::Kind::Creation: {
    ContractDefinition const &cont =
        dynamic_cast<ContractType const &>(
            *calleeTy->returnParameterTypes().front())
            .contractDefinition();
    // FIXME: We assert that the creation object's name is contract op's name.
    std::string objName = getMangledName(cont);

    // Lower args.
    std::vector<mlir::Value> args;
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes()))
      args.push_back(genRValExpr(*arg, getType(dstTy)));

    mlir::Value salt, value;
    mlir::Type ui256Ty = b.getIntegerType(256, /*isSigned=*/false);
    if (const auto *fnCallOpt =
            dynamic_cast<FunctionCallOptions const *>(&call.expression())) {
      for (const auto &[namePtr, exprPtr] :
           llvm::zip(fnCallOpt->names(), fnCallOpt->options())) {
        ASTString const &name = *namePtr;
        Expression const &expr = *exprPtr;
        mlir::Value loweredExpr = genRValExpr(expr, ui256Ty);
        if (name == "salt")
          salt = loweredExpr;
        else if (name == "value")
          value = loweredExpr;
      }
    }
    if (!value)
      value = genUnsignedConst(0, /*numBits=*/256, loc);

    resVals.push_back(
        b.create<mlir::sol::NewOp>(loc, objName, value, salt, args));
    return resVals;
  }

  case FunctionType::Kind::ObjectCreation: {
    mlir::Type ty =
        getType(dynamic_cast<ArrayType const *>(call.annotation().type));
    assert(astArgs.size() == 1);
    resVals.push_back(b.create<mlir::sol::MallocOp>(
        loc, ty, /*zeroInit=*/true, genRValExpr(*astArgs.front())));
    return resVals;
  }

  // Event invocation
  case FunctionType::Kind::Event: {
    const auto &event =
        dynamic_cast<EventDefinition const &>(calleeTy->declaration());

    // Lower and track the indexed and non-indexed args.
    std::vector<mlir::Value> indexedArgs, nonIndexedArgs;
    for (size_t i = 0; i < event.parameters().size(); ++i) {
      assert(dynamic_cast<IntegerType const *>(calleeTy->parameterTypes()[i]) ||
             dynamic_cast<AddressType const *>(calleeTy->parameterTypes()[i]));

      // TODO? YulUtilFunctions::conversionFunction
      mlir::Value arg =
          genRValExpr(*astArgs[i], getType(calleeTy->parameterTypes()[i]));

      if (event.parameters()[i]->isIndexed()) {
        indexedArgs.push_back(arg);
      } else {
        nonIndexedArgs.push_back(arg);
      }
    }

    // Generate sol.emit (with signature for non-anonymous events).
    if (event.isAnonymous()) {
      b.create<mlir::sol::EmitOp>(loc, indexedArgs, nonIndexedArgs);
    } else {
      b.create<mlir::sol::EmitOp>(loc, indexedArgs, nonIndexedArgs,
                                  calleeTy->externalSignature());
    }

    return {};
  }

  // Require statement
  case FunctionType::Kind::Require: {
    if (call.arguments().size() == 2) {
      const auto *msg = dynamic_cast<Literal const *>(astArgs[1].get());
      assert(msg && "NYI: Magic vars");
      b.create<mlir::sol::RequireOp>(loc, genRValExpr(*astArgs[0]),
                                     b.getStringAttr(msg->value()));
    } else {
      b.create<mlir::sol::RequireOp>(loc, genRValExpr(*astArgs[0]));
    }
    return {};
  }

  // ABI encode
  case FunctionType::Kind::ABIEncode: {
    mlir::SmallVector<mlir::Value, 4> args;
    for (const auto &arg : astArgs)
      args.push_back(genRValExpr(*arg));
    resVals.push_back(b.create<mlir::sol::EncodeOp>(
        loc, /*res=*/
        mlir::sol::StringType::get(b.getContext(),
                                   mlir::sol::DataLocation::Memory),
        args));
    return resVals;
  }

  // ABI decode
  case FunctionType::Kind::ABIDecode: {
    TypePointers astTys;
    if (TupleType const *tupleTy =
            dynamic_cast<TupleType const *>(call.annotation().type))
      astTys = tupleTy->components();
    else
      astTys = TypePointers{call.annotation().type};
    mlir::SmallVector<mlir::Type, 4> resTys;
    for (const Type *astTy : astTys)
      resTys.push_back(getType(astTy));

    auto decodeOp =
        b.create<mlir::sol::DecodeOp>(loc, resTys, genRValExpr(*astArgs[0]));
    for (mlir::Value res : decodeOp.getResults())
      resVals.push_back(res);
    return resVals;
  }

  case FunctionType::Kind::ArrayPush:
  case FunctionType::Kind::ArrayPop: {
    const auto *memberAcc =
        dynamic_cast<MemberAccess const *>(&call.expression());
    solAssert(memberAcc);

    // Lower `pop`
    if (calleeTy->kind() == FunctionType::Kind::ArrayPop) {
      b.create<mlir::sol::PopOp>(loc, genRValExpr(memberAcc->expression()));
      return resVals;
    }

    // Lower `push`
    auto newAddr =
        b.create<mlir::sol::PushOp>(loc, genRValExpr(memberAcc->expression()));
    if (!astArgs.empty())
      b.create<mlir::sol::StoreOp>(loc, genRValExpr(*astArgs[0]), newAddr);
    resVals.push_back(newAddr);
    return resVals;
  }

  default:
    break;
  }

  llvm_unreachable("NYI");
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genExprs(TupleExpression const &tuple) {
  mlir::SmallVector<mlir::Value, 2> vals;

  // Array literal
  if (tuple.isInlineArray()) {
    const auto *const arrTy =
        dynamic_cast<ArrayType const *>(tuple.annotation().type);
    for (const ASTPointer<Expression> &subExpr : tuple.components())
      vals.push_back(genRValExpr(*subExpr, getType(arrTy->baseType())));
    mlir::SmallVector<mlir::Value, 1> res;
    res.push_back(
        b.create<mlir::sol::ArrayLitOp>(getLoc(tuple), getType(arrTy), vals));
    return res;
  }

  for (const ASTPointer<Expression> &subExpr : tuple.components())
    vals.push_back(genLValExpr(*subExpr));
  return vals;
}

void SolidityToMLIRPass::genAssign(mlir::Value lhs, mlir::Value rhs,
                                   mlir::Location loc) {
  // Generate copy for assignment to storage reference types.
  if (mlir::sol::isNonPtrRefType(rhs.getType()) &&
      mlir::sol::getDataLocation(lhs.getType()) ==
          mlir::sol::DataLocation::Storage) {
    b.create<mlir::sol::CopyOp>(loc, rhs, lhs);
  } else {
    mlir::Value castedRhs = rhs;
    if (mlir::isa<mlir::sol::PointerType>(lhs.getType()))
      castedRhs = genCast(rhs, mlir::sol::getEltType(lhs.getType()));
    b.create<mlir::sol::StoreOp>(loc, castedRhs, lhs);
  }
}

void SolidityToMLIRPass::lower(Assignment const &asgnStmt) {
  mlir::Location loc = getLoc(asgnStmt);

  mlir::SmallVector<mlir::Value> lhsVals =
      genLValExprs(asgnStmt.leftHandSide());
  mlir::SmallVector<mlir::Value> rhsVals =
      genRValExprs(asgnStmt.rightHandSide());
  assert(lhsVals.size() == rhsVals.size());

  if (asgnStmt.assignmentOperator() == Token::Assign) {
    for (auto [lhsVal, rhsVal] : llvm::zip(lhsVals, rhsVals))
      genAssign(lhsVal, rhsVal, loc);

    // Compound assignment statement
  } else {
    assert(lhsVals.size() == 1);
    mlir::Value lhs = lhsVals.front();
    mlir::Value rhs = rhsVals.front();
    mlir::Value lhsAsRVal = genRValExpr(lhs, loc);
    Token binOp =
        TokenTraits::AssignmentToBinaryOp(asgnStmt.assignmentOperator());
    b.create<mlir::sol::StoreOp>(
        loc,
        genBinExpr(binOp, lhsAsRVal, genCast(rhs, lhsAsRVal.getType()), loc),
        lhs);
  }
}

mlir::Value SolidityToMLIRPass::genLValExpr(Expression const &expr) {
  // TODO: We should do a faster dispatch here. We could:
  // (a) Get frontend::ASTConstVisitor and ASTNode::accept to be able to return
  // mlir::Value(s).
  // (b) Adopt llvm's rtti in the ast so that we can switch over the enum that
  // discriminates the derived ast's.

  // Literal
  if (const auto *lit = dynamic_cast<Literal const *>(&expr))
    return genExpr(*lit);

  // Identifier
  if (const auto *ident = dynamic_cast<Identifier const *>(&expr))
    return genExpr(*ident);

  // Index access
  if (const auto *idxAcc = dynamic_cast<IndexAccess const *>(&expr))
    return genExpr(*idxAcc);

  // Member access
  if (const auto *memAcc = dynamic_cast<MemberAccess const *>(&expr))
    return genExpr(*memAcc);

  // (Compound) Assignment statement
  if (const auto *asgnStmt = dynamic_cast<Assignment const *>(&expr)) {
    lower(*asgnStmt);
    return {};
  }

  // Unary operation
  if (const auto *unaryOp = dynamic_cast<UnaryOperation const *>(&expr))
    return genExpr(*unaryOp);

  // Binary operation
  if (const auto *binOp = dynamic_cast<BinaryOperation const *>(&expr))
    return genExpr(*binOp);

  // Tuple
  if (const auto *tuple = dynamic_cast<TupleExpression const *>(&expr)) {
    mlir::SmallVector<mlir::Value, 1> res = genExprs(*tuple);
    assert(res.size() == 1);
    return res.front();
  }

  // Function call
  if (const auto *call = dynamic_cast<FunctionCall const *>(&expr)) {
    mlir::SmallVector<mlir::Value> exprs = genExprs(*call);
    assert(exprs.size() < 2);
    if (exprs.size() == 1)
      return exprs[0];
    return {};
  }

  llvm_unreachable("NYI");
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genLValExprs(Expression const &expr) {
  // Tuple
  if (const auto *tuple = dynamic_cast<TupleExpression const *>(&expr))
    return genExprs(*tuple);

  // Function call
  if (const auto *call = dynamic_cast<FunctionCall const *>(&expr))
    return genExprs(*call);

  mlir::SmallVector<mlir::Value, 1> vals;
  vals.push_back(genLValExpr(expr));
  return vals;
}

mlir::Value SolidityToMLIRPass::genRValExpr(mlir::Value val,
                                            mlir::Location loc) {
  if (mlir::isa<mlir::sol::PointerType>(val.getType()))
    return b.create<mlir::sol::LoadOp>(loc, val);
  return val;
}

mlir::Value SolidityToMLIRPass::genRValExpr(Expression const &expr,
                                            std::optional<mlir::Type> resTy) {
  mlir::Value lVal = genLValExpr(expr);
  assert(lVal);

  mlir::Value val = genRValExpr(lVal, getLoc(expr));
  // Generate cast (optional).
  if (resTy)
    return genCast(val, *resTy);
  return val;
}

mlir::SmallVector<mlir::Value>
SolidityToMLIRPass::genRValExprs(Expression const &expr) {
  mlir::SmallVector<mlir::Value> lVals = genLValExprs(expr);
  assert(!lVals.empty());

  mlir::SmallVector<mlir::Value, 2> rVals;
  for (mlir::Value lVal : lVals)
    rVals.push_back(genRValExpr(lVal, getLoc(expr)));

  return rVals;
}

void SolidityToMLIRPass::lower(ExpressionStatement const &exprStmt) {
  genLValExpr(exprStmt.expression());
}

void SolidityToMLIRPass::lower(
    VariableDeclarationStatement const &varDeclStmt) {
  mlir::Location loc = getLoc(varDeclStmt);

  mlir::SmallVector<mlir::Value> initExprs(varDeclStmt.declarations().size());
  if (Expression const *initExpr = varDeclStmt.initialValue())
    initExprs = genRValExprs(*initExpr);

  for (auto [varDeclPtr, initExpr] :
       llvm::zip(varDeclStmt.declarations(), initExprs)) {
    VariableDeclaration const &varDecl = *varDeclPtr;

    mlir::Type varTy = getType(varDecl.type());
    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), varTy, mlir::sol::DataLocation::Stack);

    auto addr = b.create<mlir::sol::AllocaOp>(loc, allocTy);
    trackLocalVarAddr(varDecl, addr);
    if (initExpr)
      b.create<mlir::sol::StoreOp>(loc, genCast(initExpr, varTy), addr);
    else
      genZeroedVal(addr);
  }
}

void SolidityToMLIRPass::lower(EmitStatement const &emit) {
  genLValExprs(emit.eventCall());
}

void SolidityToMLIRPass::lower(Break const &brkStmt) {
  b.create<mlir::sol::BreakOp>(getLoc(brkStmt));
  mlir::Block *newBlock = b.getBlock()->splitBlock(b.getInsertionPoint());
  b.setInsertionPointToStart(newBlock);
}

void SolidityToMLIRPass::lower(Continue const &contStmt) {
  b.create<mlir::sol::ContinueOp>(getLoc(contStmt));
  mlir::Block *newBlock = b.getBlock()->splitBlock(b.getInsertionPoint());
  b.setInsertionPointToStart(newBlock);
}

void SolidityToMLIRPass::lower(PlaceholderStatement const &placeholder) {
  b.create<mlir::sol::PlaceholderOp>(getLoc(placeholder));
}

void SolidityToMLIRPass::lower(Return const &ret) {
  TypePointers fnResTys;
  for (ASTPointer<VariableDeclaration> const &retParam :
       ret.annotation().function->returnParameters())
    fnResTys.push_back(retParam->type());

  Expression const *astExpr = ret.expression();
  if (astExpr) {
    mlir::SmallVector<mlir::Value> exprs = genRValExprs(*astExpr);
    mlir::SmallVector<mlir::Value> castedExprs;
    for (auto [expr, dstTy] : llvm::zip(exprs, fnResTys)) {
      castedExprs.push_back(genCast(expr, getType(dstTy)));
    }
    b.create<mlir::sol::ReturnOp>(getLoc(ret), castedExprs);
  } else {
    b.create<mlir::sol::ReturnOp>(getLoc(ret));
  }
  b.setInsertionPointToStart(b.createBlock(b.getBlock()->getParent()));
}

void SolidityToMLIRPass::lower(IfStatement const &ifStmt) {
  mlir::Value cond = genRValExpr(ifStmt.condition());
  auto ifOp = b.create<mlir::sol::IfOp>(getLoc(ifStmt), cond);
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
  lower(ifStmt.trueStatement());
  b.create<mlir::sol::YieldOp>(ifOp.getLoc());
  if (ifStmt.falseStatement()) {
    b.setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    lower(*ifStmt.falseStatement());
    b.create<mlir::sol::YieldOp>(ifOp.getLoc());
  }
}

void SolidityToMLIRPass::lower(WhileStatement const &whileStmt) {
  mlir::sol::LoopOpInterface whileOp;
  if (whileStmt.isDoWhile())
    whileOp = b.create<mlir::sol::DoWhileOp>(getLoc(whileStmt));
  else
    whileOp = b.create<mlir::sol::WhileOp>(getLoc(whileStmt));
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  // Lower condition.
  b.setInsertionPointToStart(&whileOp.getCond().emplaceBlock());
  mlir::Value cond = genRValExpr(whileStmt.condition());
  b.create<mlir::sol::ConditionOp>(getLoc(whileStmt.condition()), cond);

  // Lower body.
  b.setInsertionPointToStart(&whileOp.getBody().emplaceBlock());
  lower(whileStmt.body());
  b.create<mlir::sol::YieldOp>(whileOp.getLoc());
}

void SolidityToMLIRPass::lower(ForStatement const &forStmt) {
  mlirgen::BuilderExt bExt(b);

  // Lower init expression.
  if (forStmt.initializationExpression())
    lower(*forStmt.initializationExpression());

  auto forOp = b.create<mlir::sol::ForOp>(getLoc(forStmt));
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  // Lower condition.
  b.setInsertionPointToStart(&forOp.getCond().emplaceBlock());
  mlir::Value cond = forStmt.condition() ? genRValExpr(*forStmt.condition())
                                         : bExt.genBool(true, forOp.getLoc());
  b.create<mlir::sol::ConditionOp>(cond.getLoc(), cond);

  // Lower body.
  b.setInsertionPointToStart(&forOp.getBody().emplaceBlock());
  lower(forStmt.body());
  b.create<mlir::sol::YieldOp>(forOp.getLoc());

  // Lower loop expression.
  if (forStmt.loopExpression()) {
    b.setInsertionPointToStart(&forOp.getStep().emplaceBlock());
    llvm::SaveAndRestore<bool> g(inUnchecked, true);
    lower(*forStmt.loopExpression());
    b.create<mlir::sol::YieldOp>(forOp.getLoc());
  }
}

void SolidityToMLIRPass::lower(TryStatement const &tryStmt) {
  mlir::Location loc = getLoc(tryStmt);

  // TODO: sol.new
  auto externalCall = mlir::cast<mlir::sol::ExtCallOp>(
      genRValExpr(tryStmt.externalCall()).getDefiningOp());
  auto status = externalCall.getResult(0);
  auto tryOp = b.create<mlir::sol::TryOp>(loc, status);

  // Lower success clause.
  if (TryCatchClause const *successClause = tryStmt.successClause()) {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(&tryOp.getSuccessRegion().emplaceBlock());

    // Lower parameters.
    if (successClause->parameters()) {
      unsigned i = 1;
      for (ASTPointer<VariableDeclaration> const &param :
           successClause->parameters()->parameters()) {
        mlir::Location loc = getLoc(*param);
        mlir::Type allocTy =
            mlir::sol::PointerType::get(b.getContext(), getType(param->type()),
                                        mlir::sol::DataLocation::Stack);
        auto addr = b.create<mlir::sol::AllocaOp>(loc, allocTy);
        trackLocalVarAddr(*param, addr);
        b.create<mlir::sol::StoreOp>(loc, externalCall.getResult(i), addr);
      }
    }

    lower(successClause->block());
    b.create<mlir::sol::YieldOp>(loc);
  }

  // Lower panic clause.
  if (TryCatchClause const *panicClause = tryStmt.panicClause()) {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    mlir::Block *blk = &tryOp.getPanicRegion().emplaceBlock();
    b.setInsertionPointToStart(blk);

    // Add block argument for the error code which is expected to be replaced by
    // the error code from the external call by the sol.try lowering.
    assert(panicClause->parameters() &&
           panicClause->parameters()->parameters().size() == 1);
    auto ui256 = b.getIntegerType(256, /*isSigned=*/false);
    ASTPointer<VariableDeclaration> const &codeParam =
        panicClause->parameters()->parameters()[0];
    mlir::Location codeParamLoc = getLoc(*codeParam);
    mlir::BlockArgument codeParamBlkArg = blk->addArgument(ui256, codeParamLoc);

    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), ui256, mlir::sol::DataLocation::Stack);
    auto codeParamAddr = b.create<mlir::sol::AllocaOp>(codeParamLoc, allocTy);
    trackLocalVarAddr(*codeParam, codeParamAddr);
    b.create<mlir::sol::StoreOp>(loc, codeParamBlkArg, codeParamAddr);

    lower(panicClause->block());
    b.create<mlir::sol::YieldOp>(loc);
  }

  // Lower panic clause.
  if (TryCatchClause const *errorClause = tryStmt.errorClause()) {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    mlir::Block *blk = &tryOp.getErrorRegion().emplaceBlock();
    b.setInsertionPointToStart(blk);

    // Add a block argument for the error message which is expected to be
    // replaced by the error message from the external call by the sol.try
    // lowering.
    assert(errorClause->parameters() &&
           errorClause->parameters()->parameters().size() == 1);
    auto memStrTy = mlir::sol::StringType::get(b.getContext(),
                                               mlir::sol::DataLocation::Memory);
    ASTPointer<VariableDeclaration> const &msgParam =
        errorClause->parameters()->parameters()[0];
    mlir::Location msgParamLoc = getLoc(*msgParam);
    mlir::BlockArgument msgParamBlkArg =
        blk->addArgument(memStrTy, msgParamLoc);

    mlir::Type allocTy = mlir::sol::PointerType::get(
        b.getContext(), memStrTy, mlir::sol::DataLocation::Stack);
    auto msgParamAddr = b.create<mlir::sol::AllocaOp>(msgParamLoc, allocTy);
    trackLocalVarAddr(*msgParam, msgParamAddr);
    b.create<mlir::sol::StoreOp>(loc, msgParamBlkArg, msgParamAddr);
    lower(errorClause->block());
    b.create<mlir::sol::YieldOp>(loc);
  }

  // Lower fallback clause.
  if (TryCatchClause const *fallbackClause = tryStmt.fallbackClause()) {
    assert(!fallbackClause->parameters() && "NYI");
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(&tryOp.getFallbackRegion().emplaceBlock());
    lower(fallbackClause->block());
    b.create<mlir::sol::YieldOp>(loc);
  }
}

void SolidityToMLIRPass::lower(InlineAssembly const &inAsm) {
  mlir::Location loc = getLoc(inAsm.location());
  std::function<mlir::Value(yul::Identifier const *)> externalRefResolver =
      [&](yul::Identifier const *id) -> mlir::Value {
    auto it = inAsm.annotation().externalReferences.find(id);
    if (it == inAsm.annotation().externalReferences.end())
      return {};
    auto const *decl =
        dynamic_cast<VariableDeclaration const *>(it->second.declaration);
    assert(decl);
    mlir::Value localVarAddr = getLocalVarAddr(*decl);
    return b.create<mlir::sol::ConvCastOp>(
        loc, mlir::LLVM::LLVMPointerType::get(b.getContext()), localVarAddr);
  };

  // TODO: YulToMLIRPass has an expensive ctor (Due to things like
  // populateBuiltinGenMap() etc.). Can we ctor once?
  mlirgen::runYulToMLIRPass(inAsm.operations(), *stream, externalRefResolver,
                            b);
}

void SolidityToMLIRPass::lower(Statement const &stmt) {
  // Expression
  if (const auto *exprStmt = dynamic_cast<ExpressionStatement const *>(&stmt))
    lower(*exprStmt);

  // Variable declaration
  else if (const auto *varDeclStmt =
               dynamic_cast<VariableDeclarationStatement const *>(&stmt))
    lower(*varDeclStmt);

  // Emit
  else if (const auto *emitStmt = dynamic_cast<EmitStatement const *>(&stmt))
    lower(*emitStmt);

  // Placeholder
  else if (const auto *placeholderStmt =
               dynamic_cast<PlaceholderStatement const *>(&stmt))
    lower(*placeholderStmt);

  // Return
  else if (const auto *retStmt = dynamic_cast<Return const *>(&stmt))
    lower(*retStmt);

  // Break
  else if (const auto *brkStmt = dynamic_cast<Break const *>(&stmt))
    lower(*brkStmt);

  // Continue
  else if (const auto *contStmt = dynamic_cast<Continue const *>(&stmt))
    lower(*contStmt);

  // If-then-else
  else if (const auto *ifStmt = dynamic_cast<IfStatement const *>(&stmt))
    lower(*ifStmt);

  // While and do-while
  else if (const auto *whileStmt = dynamic_cast<WhileStatement const *>(&stmt))
    lower(*whileStmt);

  // For
  else if (const auto *forStmt = dynamic_cast<ForStatement const *>(&stmt))
    lower(*forStmt);

  // Try
  else if (const auto *tryStmt = dynamic_cast<TryStatement const *>(&stmt))
    lower(*tryStmt);

  // Inline assembly
  else if (const auto *inAsm = dynamic_cast<InlineAssembly const *>(&stmt))
    lower(*inAsm);

  // Block
  else if (const auto *blk = dynamic_cast<Block const *>(&stmt))
    lower(*blk);

  else
    llvm_unreachable("NYI");
}

void SolidityToMLIRPass::lower(Block const &blk) {
  llvm::SaveAndRestore<bool> g(inUnchecked, blk.unchecked());
  for (const ASTPointer<Statement> &stmt : blk.statements())
    lower(*stmt);
}

/// Returns the mlir::sol::StateMutability of the function
static mlir::sol::StateMutability
getStateMutability(FunctionDefinition const &fn) {
  switch (fn.stateMutability()) {
  case StateMutability::Pure:
    return mlir::sol::StateMutability::Pure;
  case StateMutability::View:
    return mlir::sol::StateMutability::View;
  case StateMutability::NonPayable:
    return mlir::sol::StateMutability::NonPayable;
  case StateMutability::Payable:
    return mlir::sol::StateMutability::Payable;
  }
}

void SolidityToMLIRPass::lower(ModifierDefinition const &modifier) {
  std::vector<mlir::Type> inpTys;
  std::vector<mlir::Location> inpLocs;

  for (const auto &param : modifier.parameters()) {
    inpTys.push_back(getType(param->annotation().type));
    inpLocs.push_back(getLoc(*param));
  }
  auto funcType = b.getFunctionType(inpTys, {});
  auto op = b.create<mlir::sol::ModifierOp>(getLoc(modifier),
                                            getMangledName(modifier), funcType);

  mlir::Block *entryBlk = b.createBlock(&op.getRegion());
  b.setInsertionPointToStart(entryBlk);
  for (auto &&[inpTy, inpLoc, param] :
       ranges::views::zip(inpTys, inpLocs, modifier.parameters())) {
    mlir::Value arg = entryBlk->addArgument(inpTy, inpLoc);
    auto addr = b.create<mlir::sol::AllocaOp>(
        inpLoc, mlir::sol::PointerType::get(b.getContext(), inpTy,
                                            mlir::sol::DataLocation::Stack));
    trackLocalVarAddr(*param, addr);
    b.create<mlir::sol::StoreOp>(inpLoc, arg, addr);
  }

  lower(modifier.body());
  b.create<mlir::sol::ReturnOp>(getLoc(modifier));

  b.setInsertionPointAfter(op);
}

mlir::sol::FuncOp SolidityToMLIRPass::lower(FunctionDefinition const &fn) {
  assert(fn.isImplemented());
  // Create the function type.
  std::vector<mlir::Type> inpTys, outTys;
  std::vector<mlir::Location> inpLocs;
  for (const auto &param : fn.parameters()) {
    inpTys.push_back(getType(param->annotation().type));
    inpLocs.push_back(getLoc(*param));
  }
  for (const auto &param : fn.returnParameters())
    outTys.push_back(getType(param->annotation().type));
  auto fnTy = b.getFunctionType(inpTys, outTys);

  // Generate sol.func.
  mlir::Location fnLoc = getLoc(fn);
  auto op = b.create<mlir::sol::FuncOp>(fnLoc, getMangledName(fn), fnTy,
                                        getStateMutability(fn));

  if (fn.isPartOfExternalInterface()) {
    assert(selectorMap.find(&fn) != selectorMap.end());
    op.setSelectorAttr(b.getIntegerAttr(
        b.getIntegerType(32), mlir::APInt(32, selectorMap[&fn].hex(), 16)));
    op.setOrigFnType(fnTy);
  }

  // Set function kind.
  if (fn.isReceive()) {
    op.setKind(mlir::sol::FunctionKind::Receive);
  } else if (fn.isFallback()) {
    op.setKind(mlir::sol::FunctionKind::Fallback);
  }

  mlir::Block *entryBlk = b.createBlock(&op.getRegion());
  b.setInsertionPointToStart(entryBlk);

  // Lower modifier invocations.
  for (const ASTPointer<ModifierInvocation> &modifier : fn.modifiers()) {
    // FIXME: Handle lookup!
    ModifierDefinition const *modifierDef =
        dynamic_cast<ModifierDefinition const *>(
            modifier->name().annotation().referencedDeclaration);
    if (!modifierDef) {
      assert(dynamic_cast<ContractDefinition const *>(
          modifier->name().annotation().referencedDeclaration));
      continue;
    }

    mlir::Location loc = getLoc(*modifier);

    auto modifierCallBlk = b.create<mlir::sol::ModifierCallBlkOp>(loc);
    mlir::OpBuilder::InsertionGuard insertGuard(b);

    // sol.modifier_call_blk's block args should match that of the function. We
    // don't need to generate stack allocations since the block args are
    // "forwarded" in the call chain of modifiers and the modified function.
    for (auto &&[inpTy, inpLoc, param] :
         ranges::views::zip(inpTys, inpLocs, fn.parameters()))
      trackLocalVarAddr(*param,
                        modifierCallBlk.getBody()->addArgument(inpTy, inpLoc));

    b.setInsertionPointToStart(modifierCallBlk.getBody());

    std::vector<mlir::Value> loweredArgs;
    if (modifier->arguments()) {
      loweredArgs.reserve(modifier->arguments()->size());
      unsigned i = 0;
      for (const ASTPointer<Expression> &arg : *modifier->arguments()) {
        mlir::Type reqTy = getType(modifierDef->parameters()[i++]->type());
        loweredArgs.push_back(genRValExpr(*arg, reqTy));
      }
    }
    b.create<mlir::sol::CallOp>(loc, getMangledName(*modifierDef),
                                /*results=*/mlir::TypeRange{}, loweredArgs);
  }

  // Lower the args.
  for (auto &&[inpTy, inpLoc, param] :
       ranges::views::zip(inpTys, inpLocs, fn.parameters())) {
    mlir::Value arg = entryBlk->addArgument(inpTy, inpLoc);
    auto addr = b.create<mlir::sol::AllocaOp>(
        inpLoc, mlir::sol::PointerType::get(b.getContext(), inpTy,
                                            mlir::sol::DataLocation::Stack));
    trackLocalVarAddr(*param, addr);
    b.create<mlir::sol::StoreOp>(inpLoc, arg, addr);
  }

  // Generate the call to the next ctor (if any) if `fn` is a ctor.
  if (fn.isConstructor()) {
    // Get base contract of `currContract`
    auto const &baseCont =
        dynamic_cast<ContractDefinition const &>(*fn.scope());

    if (FunctionDefinition const *nextCtor =
            baseCont.nextConstructor(*currContract)) {
      auto nextCtorArgsFound =
          currContract->annotation().baseConstructorArguments.find(nextCtor);
      mlir::SmallVector<mlir::Value> loweredArgs;
      if (nextCtorArgsFound !=
          currContract->annotation().baseConstructorArguments.end()) {
        std::vector<ASTPointer<Expression>> const *nextCtorArgs = nullptr;
        ASTNode const *argsNode =
            currContract->annotation().baseConstructorArguments.at(nextCtor);
        if (const auto *inheritanceSpec =
                dynamic_cast<InheritanceSpecifier const *>(argsNode))
          nextCtorArgs = inheritanceSpec->arguments();
        else if (const auto *modifierInvoc =
                     dynamic_cast<ModifierInvocation const *>(argsNode))
          nextCtorArgs = modifierInvoc->arguments();
        assert(nextCtorArgs);
        for (ASTPointer<Expression> const &arg : *nextCtorArgs)
          loweredArgs.push_back(
              genRValExpr(*arg, getType(arg->annotation().type)));
      }
      b.create<mlir::sol::CallOp>(fnLoc, getMangledName(*nextCtor),
                                  /*resTys=*/mlir::TypeRange{}, loweredArgs);
    }
  }

  // Lower the body.
  lower(fn.body());

  // Return stmt lowering generates an empty block that might be empty at this
  // stage.
  if (outTys.empty())
    b.create<mlir::sol::ReturnOp>(getLoc(fn));
  mlir::Block *currBlk = b.getBlock();
  if (currBlk->empty()) {
    op.getBody().back().erase();
  } else if (!currBlk->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    // FIXME: Generate "default" return statement for non-empty return types
    // (zero/zero-pointer).
    llvm_unreachable("NYI");
  }

  b.setInsertionPointAfter(op);
  return op;
}

/// Returns the mlir::sol::ContractKind of the contract
static mlir::sol::ContractKind getContractKind(ContractDefinition const &cont) {
  switch (cont.contractKind()) {
  case ContractKind::Interface:
    return mlir::sol::ContractKind::Interface;
  case ContractKind::Contract:
    return mlir::sol::ContractKind::Contract;
  case ContractKind::Library:
    return mlir::sol::ContractKind::Library;
  }
}

void SolidityToMLIRPass::lower(ContractDefinition const &cont) {
  currContract = &cont;
  mlir::Location loc = getLoc(cont);

  // This function works on the full inheritance tree from `cont` but we only
  // generate the sol.contract op for `cont`.

  // Track selectors of interface functions.
  const auto &interfaceFnInfos = cont.interfaceFunctions();
  for (const auto &i : interfaceFnInfos)
    selectorMap[&i.second->declaration()] = i.first;

  // Create the contract op.
  mlir::sol::ContractOp contOp = b.create<mlir::sol::ContractOp>(
      loc, getMangledName(cont), getContractKind(cont));
  b.setInsertionPointToStart(&contOp.getBodyRegion().emplaceBlock());

  // Lower immutables and state variables; Generate getters
  for (ContractDefinition const *baseCont :
       cont.annotation().linearizedBaseContracts) {
    for (VariableDeclaration const *stateVar : baseCont->stateVariables()) {
      if (stateVar->immutable())
        b.create<mlir::sol::ImmutableOp>(getLoc(*stateVar),
                                         getMangledName(*stateVar),
                                         getType(stateVar->type()));
      else if (!stateVar->isConstant())
        b.create<mlir::sol::StateVarOp>(getLoc(*stateVar),
                                        getMangledName(*stateVar),
                                        getType(stateVar->type()));

      if (stateVar->isPartOfExternalInterface())
        genGetter(*stateVar);
    }
  }

  // Lower/generate ctor if not library. Note that lower() of functions
  // generates the call to the next ctor.
  if (!cont.isLibrary()) {
    mlir::sol::FuncOp ctorFn;
    inCtor = true;
    if (FunctionDefinition const *ctor = cont.constructor()) {
      ctorFn = lower(*ctor);
    } else {
      mlir::OpBuilder::InsertionGuard insertGuard(b);
      ctorFn = b.create<mlir::sol::FuncOp>(
          loc, contOp.getName(), b.getFunctionType({}, {}),
          mlir::sol::StateMutability::NonPayable);
      b.setInsertionPointToStart(b.createBlock(&ctorFn.getRegion()));
      FunctionDefinition const *nextCtor = cont.nextConstructor(cont);
      if (nextCtor)
        b.create<mlir::sol::CallOp>(loc, getMangledName(*nextCtor),
                                    /*resTys=*/mlir::TypeRange{});
      b.create<mlir::sol::ReturnOp>(loc);
    }
    ctorFn.setKind(mlir::sol::FunctionKind::Constructor);
    ctorFn.setOrigFnType(ctorFn.getFunctionType());

    // Generate state variable init in the ctor.
    b.setInsertionPointToStart(&ctorFn.getBody().front());
    for (auto &var :
         ContractType(cont).linearizedStateVariables(DataLocation::Storage)) {
      VariableDeclaration const *stateVar = std::get<0>(var);
      if (!stateVar->isConstant() && stateVar->value()) {
        genAssign(genStateVarRef(*stateVar, /*inCreationContext=*/true),
                  genRValExpr(*stateVar->value()), getLoc(*stateVar));
      }
    }
    b.setInsertionPointAfter(ctorFn);
    inCtor = false;
  }

  // Lower all other functions and modifiers.
  auto lowerFnsAndMods = [&](ContractDefinition const &baseCont) {
    // Lower functions.
    for (auto *f : baseCont.definedFunctions()) {
      // Skip the current contract's ctor since it is already lowered.
      if (baseCont == *currContract && f->isConstructor())
        continue;
      if (f->isImplemented())
        lower(*f);
    }

    // Lower modifiers.
    for (auto *modifier : baseCont.functionModifiers()) {
      lower(*modifier);
    }
  };
  for (ContractDefinition const *baseCont :
       cont.annotation().linearizedBaseContracts) {
    lowerFnsAndMods(*baseCont);
  }
  currContract = nullptr;
}

void SolidityToMLIRPass::lowerFreeFuncs(SourceUnit const &srcUnit) {
  for (const auto *func :
       ASTNode::filteredNodes<FunctionDefinition>(srcUnit.nodes())) {
    lower(*func);
  }
}

static void loadDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::sol::SolDialect>();
  ctx.getOrLoadDialect<mlir::yul::YulDialect>();
  // For lowering yul in inline-asm.
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}

bool CompilerStack::runMlirPipeline() {
  llvm::StdThreadPool threadPool;
  // For sync'ing the output collection.
  std::mutex outMtx;
  // For sync'ing the error printing.
  std::mutex errMtx;
  // Tracks if any thread had an error.
  std::atomic<bool> hadError{false};

  // Maps contract ast to requested output. Updates are sync'ed by `outMtx`.
  std::map<ContractDefinition const *, std::string, ASTNode::CompareByID>
      outputMap;
  std::map<ContractDefinition const *, evm::UnlinkedObj, ASTNode::CompareByID>
      unlinkedObjMap;

  for (Source const *src : m_sourceOrder) {
    // Lower requested contracts per thread.
    bool hasContract = false;
    for (const auto *contr :
         ASTNode::filteredNodes<ContractDefinition>(src->ast->nodes())) {
      if (!contr->canBeDeployed())
        continue;
      hasContract = true;
      if (isRequestedContract(*contr)) {
        threadPool.async([&, src, contr]() {
          // Create the mlir context.
          mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
          loadDialects(ctx);

          // Run the ast lowering pass.
          SolidityToMLIRPass gen(ctx, m_evmVersion,
                                 /*genUnkLoc=*/m_mlirGenJob.action ==
                                     mlirgen::Action::GenObj);
          gen.init(src->charStream);
          // Lower free functions.
          gen.lowerFreeFuncs(*src->ast);
          gen.lower(*contr);
          mlir::ModuleOp mod = gen.getModule();

          // Verify the module.
          if (failed(mlir::verify(mod))) {
            hadError.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> g(errMtx);
            mod.dump();
            mod.emitError("Module verification error");
            return;
          }

          if (m_mlirGenJob.action == mlirgen::Action::GenObj) {
            // Create the llvm target machine.
            std::unique_ptr<llvm::TargetMachine> tgtMach =
                createTargetMachine(m_mlirGenJob.tgt);
            mlirgen::setTgtMachOpt(tgtMach.get(), m_mlirGenJob.optLevel);

            // Generate the object.
            evm::UnlinkedObj obj =
                mlirgen::genEvmObj(mod, m_mlirGenJob.optLevel, *tgtMach);
            std::lock_guard<std::mutex> g(outMtx);
            unlinkedObjMap[contr] = obj;

          } else {
            // Generate the print output.
            std::string out = mlirgen::printJob(m_mlirGenJob, mod);
            std::lock_guard<std::mutex> g(outMtx);
            outputMap[contr] = out;
          }
        });
      }
    }

    if (!hasContract) {
      mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
      loadDialects(ctx);

      SolidityToMLIRPass gen(ctx, m_evmVersion,
                             /*genUnkLoc=*/m_mlirGenJob.action ==
                                 mlirgen::Action::GenObj);
      // Then lower free functions. This is handy in testing.
      gen.init(src->charStream);
      gen.lowerFreeFuncs(*src->ast);

      mlir::ModuleOp mod = gen.getModule();
      if (failed(mlir::verify(mod))) {
        std::lock_guard<std::mutex> g(errMtx);
        mod.dump();
        mod.emitError("Module verification error");
        return false;
      }

      if (m_mlirGenJob.action != mlirgen::Action::GenObj) {
        std::lock_guard<std::mutex> g(outMtx);
        llvm::outs() << mlirgen::printJob(m_mlirGenJob, mod);
      }
    }
  }

  threadPool.wait();
  if (hadError)
    return false;

  // Combine all the outputs.
  if (m_mlirGenJob.action != mlirgen::Action::GenObj) {
    for (auto const &i : outputMap)
      llvm::outs() << i.second;
  } else {
    evm::BytecodeGen bcGen(unlinkedObjMap, m_libraries);
    for (auto const &i : unlinkedObjMap) {
      ContractDefinition const *cont = i.first;
      m_contracts.at(cont->fullyQualifiedName()).mlirPipeline =
          bcGen.genEvmBytecode(i.first);
    }
    for (auto const &i : unlinkedObjMap) {
      evm::UnlinkedObj obj = i.second;
      if (obj.creationPart)
        LLVMDisposeMemoryBuffer(obj.creationPart);
      if (obj.runtimePart)
        LLVMDisposeMemoryBuffer(obj.runtimePart);
    }
  }

  return true;
}

// TODO: Move the following functions somewhere else.

void solidity::mlirgen::registerMLIRCLOpts() {
  // FIXME: Verifier's InFlightDiagnostic doesn't work with --mmlir
  // -mlir-print-op-on-diagnostic!
  mlir::registerMLIRContextCLOptions();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();
}

bool solidity::mlirgen::parseMLIROpts(std::vector<const char *> &argv) {
  // ParseCommandLineOptions() expects argv[0] to be the name of a program
  std::vector<const char *> fooArgv{"foo"};
  for (const char *arg : argv) {
    fooArgv.push_back(arg);
  }

  return llvm::cl::ParseCommandLineOptions(fooArgv.size(), fooArgv.data(),
                                           "Generic MLIR flags\n");
}

solidity::mlirgen::Target
solidity::mlirgen::strToTarget(std::string const &str) {
  std::string inLowerCase = str;
  std::transform(inLowerCase.begin(), inLowerCase.end(), inLowerCase.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (inLowerCase == "evm")
    return Target::EVM;
  if (inLowerCase == "eravm")
    return Target::EraVM;
  return Target::Undefined;
}
