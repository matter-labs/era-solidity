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

#include "Sol/SolOps.h"
#include "liblangutil/CharStream.h"
#include "liblangutil/Exceptions.h"
#include "liblangutil/SourceLocation.h"
#include "libsolidity/ast/AST.h"
#include "libsolidity/ast/ASTEnums.h"
#include "libsolidity/ast/ASTVisitor.h"
#include "libsolidity/ast/Types.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolidity/interface/CompilerStack.h"
#include "libsolutil/CommonIO.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "range/v3/view/zip.hpp"
#include "llvm/Support/CommandLine.h"
#include <string>

using namespace solidity::langutil;
using namespace solidity::frontend;

namespace solidity::frontend {

class SolidityToMLIRPass : public ASTConstVisitor {
public:
  explicit SolidityToMLIRPass(mlir::MLIRContext &ctx, CharStream const &stream)
      : b(&ctx), stream(stream) {
    mod = mlir::ModuleOp::create(b.getUnknownLoc());
    b.setInsertionPointToEnd(mod.getBody());
  }

  void run(ContractDefinition const &);

  /// Returns the ModuleOp
  mlir::ModuleOp getModule() { return mod; }

private:
  mlir::OpBuilder b;
  CharStream const &stream;
  mlir::ModuleOp mod;

  /// The block being lowered.
  Block const *currBlk;

  /// The function being lowered.
  FunctionDefinition const *currFunc;

  /// The contract being lowered.
  ContractDefinition const *currContract;

  /// Maps a local variable to its address.
  std::map<VariableDeclaration const *, mlir::Value> localVarAddrMap;

  /// Returns true if the current block is unchecked.
  bool inUncheckedBlk() {
    assert(currBlk);
    return currBlk->unchecked();
  }

  /// Returns the mlir location for the solidity source location `loc`
  mlir::Location getLoc(SourceLocation const &loc) {
    // FIXME: Track loc.end as well
    LineColumn lineCol = stream.translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream.name()),
                                     lineCol.line, lineCol.column);
  }

  /// Returns the corresponding mlir type for the solidity type `ty`.
  mlir::Type getType(Type const *ty);

  /// Tracks the address of a local variable.
  void trackLocalVarAddr(VariableDeclaration const *decl, mlir::Value addr) {
    localVarAddrMap[decl] = addr;
  }

  /// Returns the mangled name of the declaration composed of its name and its
  /// AST ID.
  std::string getMangledName(Declaration const &decl) {
    return decl.name() + "_" + std::to_string(decl.id());
  }

  /// Returns the array attribute for tracking interface functions (symbol, type
  /// and selector) in the contract.
  mlir::ArrayAttr getInterfaceFnsAttr(ContractDefinition const &cont);

  /// Generates the ir to zero the allocation.
  void genZeroedVal(mlir::sol::AllocaOp addr);

  /// Generates a integeral constant op.
  mlir::Value genConst(uint64_t val, mlir::Location loc) {
    return b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/false), val));
  }

  /// Generates type cast expression.
  mlir::Value genCast(mlir::Value val, mlir::Type dstTy);

  // We can't completely rely on ExpressionAnnotation::isLValue here since the
  // TypeChecker doesn't, for instance, tag RHS expression of an assignment as
  // an r-value.

  // FIXME: The genRValExpr's look almost the same. Unify them!

  /// Returns the mlir expression for the literal.
  mlir::Value genExpr(Literal const *lit);

  /// Returns the mlir expression for the identifier in an l-value context.
  mlir::Value genLValExpr(Identifier const *ident);

  /// Returns the mlir expression for the identifier in an r-value context.
  mlir::Value genRValExpr(Identifier const *ident);

  /// Returns the mlir expression for the index access in an l-value context.
  mlir::Value genLValExpr(IndexAccess const *idxAcc);

  /// Returns the mlir expression for the index access in an r-value context.
  mlir::Value genRValExpr(IndexAccess const *idxAcc);

  /// Returns the mlir expression for the member access in an r-value context.
  mlir::Value genLValExpr(MemberAccess const *memberAcc);

  /// Returns the mlir expression for the member access in an r-value context.
  mlir::Value genRValExpr(MemberAccess const *memberAcc);

  /// Returns the mlir expression for the binary operation.
  mlir::Value genBinExpr(Token op, mlir::Value lhs, mlir::Value rhs,
                         mlir::Location loc);

  /// Returns the mlir expression for the binary operation.
  mlir::Value genExpr(BinaryOperation const *binOp);

  /// Returns the mlir expression for the call.
  mlir::Value genExpr(FunctionCall const *call);

  /// Returns the mlir expression in an l-value context.
  mlir::Value genLValExpr(Expression const *expr);

  /// Returns the mlir expression in an r-value context and optionally casts it
  /// to the corresponding mlir type of `resTy`.
  mlir::Value genRValExpr(Expression const *expr,
                          std::optional<mlir::Type> resTy = std::nullopt);

  bool visit(ExpressionStatement const &) override;
  bool visit(VariableDeclarationStatement const &) override;
  bool visit(EmitStatement const &) override;
  bool visit(Return const &) override;
  bool visit(Block const &) override;
  void run(FunctionDefinition const &);
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
  // Bool type
  if (const auto *boolTy = dynamic_cast<BoolType const *>(ty)) {
    return b.getIntegerType(/*width=*/1);
  }

  // Integer type
  if (const auto *intTy = dynamic_cast<IntegerType const *>(ty))
    return b.getIntegerType(intTy->numBits(), intTy->isSigned());

  // Rational number type
  if (const auto *ratNumTy = dynamic_cast<RationalNumberType const *>(ty)) {
    if (ratNumTy->isFractional())
      llvm_unreachable("NYI: Fractional type");

    // Integral rational number type
    const IntegerType *intTy = ratNumTy->integerType();
    return b.getIntegerType(intTy->numBits(), intTy->isSigned());
  }

  // Address type
  if (const auto *addrTy = dynamic_cast<AddressType const *>(ty))
    // FIXME: 256 -> 160
    return b.getIntegerType(256, /*isSigned=*/false);

  // Mapping type
  if (const auto *mappingTy = dynamic_cast<MappingType const *>(ty)) {
    return mlir::sol::MappingType::get(b.getContext(),
                                       getType(mappingTy->keyType()),
                                       getType(mappingTy->valueType()));
  }

  // Array or string type
  if (const auto *arrTy = dynamic_cast<ArrayType const *>(ty)) {
    if (arrTy->isString())
      return mlir::sol::StringType::get(b.getContext(), getDataLocation(arrTy));
    mlir::Type eltTy = getType(arrTy->baseType());

    // TODO: Does convert_to alreay do this?
    assert(arrTy->length() <= INT64_MAX);
    return mlir::sol::ArrayType::get(b.getContext(),
                                     arrTy->length().convert_to<int64_t>(),
                                     eltTy, getDataLocation(arrTy));
  }

  // Struct type
  if (const auto *structTy = dynamic_cast<StructType const *>(ty)) {
    std::vector<mlir::Type> memberTys;
    for (auto const &mem : structTy->nativeMembers(nullptr)) {
      memberTys.push_back(getType(mem.type));
    }

    return mlir::sol::StructType::get(b.getContext(), memberTys,
                                      getDataLocation(structTy));
  }

  // Function type
  if (const auto *fnTy = dynamic_cast<FunctionType const *>(ty)) {
    std::vector<mlir::Type> inTys, outTys;

    inTys.reserve(fnTy->parameterTypes().size());
    for (Type const *inTy : fnTy->parameterTypes())
      inTys.push_back(getType(inTy));

    outTys.reserve(fnTy->returnParameterTypes().size());
    for (Type const *outTy : fnTy->returnParameterTypes())
      outTys.push_back(getType(outTy));

    return b.getFunctionType(inTys, outTys);
  }

  llvm_unreachable("NYI: Unknown type");
}

mlir::Value SolidityToMLIRPass::genLValExpr(Identifier const *id) {
  Declaration const *decl = id->annotation().referencedDeclaration;

  if (auto *var = dynamic_cast<VariableDeclaration const *>(decl)) {
    // State variable.
    if (var->isStateVariable()) {
      auto currContr =
          b.getBlock()->getParentOp()->getParentOfType<mlir::sol::ContractOp>();
      assert(currContr);
      auto stateVarOp =
          currContr.lookupSymbol<mlir::sol::StateVarOp>(var->name());
      assert(stateVarOp);
      mlir::Type addrTy;
      if (mlir::sol::isNonPtrRefType(stateVarOp.getType()))
        addrTy = stateVarOp.getType();
      else
        addrTy =
            mlir::sol::PointerType::get(b.getContext(), stateVarOp.getType(),
                                        mlir::sol::DataLocation::Storage);
      // TODO: Should we use the state variable's location here?
      return b.create<mlir::sol::AddrOfOp>(stateVarOp.getLoc(), addrTy,
                                           stateVarOp.getSymName());
    }

    // Local variable.
    auto it = localVarAddrMap.find(var);
    assert(it != localVarAddrMap.end());
    return it->second;
  }

  llvm_unreachable("NYI");
}

mlir::Value SolidityToMLIRPass::genRValExpr(Identifier const *id) {
  auto addr = genLValExpr(id);

  // Don't load non pointer ref types.
  if (mlir::sol::isNonPtrRefType(addr.getType()))
    return addr;

  return b.create<mlir::sol::LoadOp>(getLoc(id->location()), addr);
}

mlir::ArrayAttr
SolidityToMLIRPass::getInterfaceFnsAttr(ContractDefinition const &cont) {
  auto const &interfaceFnInfos = cont.interfaceFunctions();
  std::vector<mlir::Attribute> interfaceFnAttrs;
  interfaceFnAttrs.reserve(interfaceFnInfos.size());
  for (auto const &i : interfaceFnInfos) {
    auto fnSymAttr = mlir::SymbolRefAttr::get(
        b.getContext(), getMangledName(i.second->declaration()));

    mlir::FunctionType fnTy = mlir::cast<mlir::FunctionType>(getType(i.second));
    auto fnTyAttr = mlir::TypeAttr::get(fnTy);

    auto selectorAttr = b.getIntegerAttr(b.getIntegerType(32),
                                         mlir::APInt(32, i.first.hex(), 16));

    interfaceFnAttrs.push_back(b.getDictionaryAttr(
        {b.getNamedAttr("sym", fnSymAttr), b.getNamedAttr("type", fnTyAttr),
         b.getNamedAttr("selector", selectorAttr)}));
  }
  return b.getArrayAttr(interfaceFnAttrs);
}

void SolidityToMLIRPass::genZeroedVal(mlir::sol::AllocaOp addr) {
  mlir::Location loc = addr.getLoc();

  auto pointeeTy =
      mlir::cast<mlir::sol::PointerType>(addr.getType()).getPointeeType();

  mlir::Value val;
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(pointeeTy)) {
    val = b.create<mlir::sol::ConstantOp>(
        loc, b.getIntegerAttr(intTy, llvm::APInt(intTy.getWidth(), 0)));
  } else if (auto stringTy = mlir::dyn_cast<mlir::sol::StringType>(pointeeTy)) {
    assert(stringTy.getDataLocation() == mlir::sol::DataLocation::Memory &&
           "NYI");
    val = b.create<mlir::sol::MallocOp>(loc, stringTy, stringTy,
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

  // Casting to integer type.
  if (auto dstIntTy = mlir::dyn_cast<mlir::IntegerType>(dstTy)) {
    auto srcIntTy = mlir::cast<mlir::IntegerType>(srcTy);

    if (dstIntTy.getWidth() > srcIntTy.getWidth())
      return b.create<mlir::sol::ExtOp>(loc, dstTy, val);
    assert(dstIntTy.getWidth() < srcIntTy.getWidth());
    return b.create<mlir::sol::TruncOp>(loc, dstTy, val);
  }

  // Casting between reference types (excluding pointer types).
  if (mlir::sol::isNonPtrRefType(dstTy)) {
    assert(mlir::sol::isNonPtrRefType(dstTy));
    return b.create<mlir::sol::DataLocCastOp>(loc, dstTy, val);
  }

  llvm_unreachable("NYI or invalid cast");
}

mlir::Value SolidityToMLIRPass::genExpr(Literal const *lit) {
  mlir::Location loc = getLoc(lit->location());
  Type const *ty = lit->annotation().type;

  // Bool literal
  if (auto *boolTy = dynamic_cast<BoolType const *>(ty))
    return b.create<mlir::sol::ConstantOp>(
        loc, b.getBoolAttr(lit->token() == Token::TrueLiteral));

  // Rational number literal
  if (auto *ratNumTy = dynamic_cast<RationalNumberType const *>(ty)) {
    if (ratNumTy->isFractional())
      llvm_unreachable("NYI: Fractional literal");

    auto *intTy = ratNumTy->integerType();
    u256 val = ty->literalValue(lit);
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
    if (inUncheckedBlk())
      return b.create<mlir::sol::AddOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CAddOp>(loc, lhs, rhs);
  case Token::Sub:
    if (inUncheckedBlk())
      return b.create<mlir::sol::SubOp>(loc, lhs, rhs);
    else
      return b.create<mlir::sol::CSubOp>(loc, lhs, rhs);
  case Token::Mul:
    assert(inUncheckedBlk() && "NYI");
    return b.create<mlir::sol::MulOp>(loc, lhs, rhs);
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

mlir::Value SolidityToMLIRPass::genExpr(BinaryOperation const *binOp) {
  mlir::Type resTy = getType(binOp->leftExpression().annotation().type);
  auto loc = getLoc(binOp->location());

  mlir::Value lhs = genRValExpr(&binOp->leftExpression(), resTy);
  mlir::Value rhs = genRValExpr(&binOp->rightExpression(), resTy);

  return genBinExpr(binOp->getOperator(), lhs, rhs, loc);
}

mlir::Value SolidityToMLIRPass::genLValExpr(IndexAccess const *idxAcc) {
  mlir::Location loc = getLoc(idxAcc->location());

  mlir::Value baseExpr = genRValExpr(&idxAcc->baseExpression());
  mlir::Value idxExpr = genRValExpr(idxAcc->indexExpression());

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

  // Array indexing
  if (auto arrTy = mlir::dyn_cast<mlir::sol::ArrayType>(baseExpr.getType())) {
    return b.create<mlir::sol::GepOp>(loc, baseExpr, idxExpr);
  }

  llvm_unreachable("Invalid IndexAccess");
}

mlir::Value SolidityToMLIRPass::genRValExpr(IndexAccess const *idxAcc) {
  mlir::Value addr = genLValExpr(idxAcc);

  // Don't load non pointer ref types.
  if (mlir::sol::isNonPtrRefType(addr.getType()))
    return addr;
  return b.create<mlir::sol::LoadOp>(getLoc(idxAcc->location()), addr);
}

mlir::Value SolidityToMLIRPass::genLValExpr(MemberAccess const *memberAcc) {
  mlir::Location loc = getLoc(memberAcc->location());

  const auto *memberAccTy = memberAcc->expression().annotation().type;
  switch (memberAccTy->category()) {
  case Type::Category::Magic:
    if (memberAcc->memberName() == "sender") {
      // FIXME: sol.caller yields an i256 instead of an address.
      auto callerOp = b.create<mlir::sol::CallerOp>(loc);
      return b.create<mlir::sol::ConvCastOp>(
          loc, b.getIntegerType(256, /*isSigned=*/false), callerOp);
    }
    break;
  case Type::Category::Struct: {
    const auto *structTy = dynamic_cast<StructType const *>(memberAccTy);
    auto memberIdx = genConst(structTy->index(memberAcc->memberName()), loc);
    return b.create<mlir::sol::GepOp>(
        loc, genRValExpr(&memberAcc->expression()), memberIdx);
    break;
  }
  default:
    break;
  }

  llvm_unreachable("NYI");
}

mlir::Value SolidityToMLIRPass::genRValExpr(MemberAccess const *memberAcc) {
  mlir::Value addr = genLValExpr(memberAcc);
  if (memberAcc->expression().annotation().type->category() ==
      Type::Category::Struct) {
    // Don't load non pointer ref types.
    if (mlir::sol::isNonPtrRefType(addr.getType()))
      return addr;

    return b.create<mlir::sol::LoadOp>(getLoc(memberAcc->location()), addr);
  }
  return addr;
}

mlir::Value SolidityToMLIRPass::genExpr(FunctionCall const *call) {
  assert(*call->annotation().kind != FunctionCallKind::StructConstructorCall &&
         "NYI");

  // Type conversion
  if (*call->annotation().kind == FunctionCallKind::TypeConversion) {
    return genRValExpr(call->arguments().front().get(),
                       getType(call->annotation().type));
  }

  auto const *calleeTy =
      dynamic_cast<FunctionType const *>(call->expression().annotation().type);
  assert(calleeTy);

  std::vector<Type const *> argTys = calleeTy->parameterTypes();
  std::vector<ASTPointer<Expression const>> const &astArgs =
      call->sortedArguments();

  mlir::Location loc = getLoc(call->location());
  switch (calleeTy->kind()) {
  // Internal call
  case FunctionType::Kind::Internal: {
    // Get callee.
    assert(currContract);
    FunctionDefinition const *callee =
        ASTNode::resolveFunctionCall(*call, currContract);
    assert(callee && "NYI: Internal function dispatch");

    // Lower args.
    std::vector<mlir::Value> args;
    for (auto [arg, dstTy] : llvm::zip(astArgs, calleeTy->parameterTypes())) {
      args.push_back(genRValExpr(arg.get(), getType(dstTy)));
    }

    // Collect return types.
    std::vector<mlir::Type> resTys;
    for (Type const *ty : calleeTy->returnParameterTypes()) {
      resTys.push_back(getType(ty));
    }

    // FIXME: To support multi-valued return args, genExpr should return
    // ValueRange.
    assert(resTys.size() <= 1);
    // Generate the call op.
    auto callOp = b.create<mlir::sol::CallOp>(
        getLoc(call->location()), getMangledName(*callee), resTys, args);
    return resTys.empty() ? mlir::Value{} : callOp.getResult(0);
  }

  // Event invocation
  case FunctionType::Kind::Event: {
    auto const &event =
        dynamic_cast<EventDefinition const &>(calleeTy->declaration());

    // Lower and track the indexed and non-indexed args.
    std::vector<mlir::Value> indexedArgs, nonIndexedArgs;
    for (size_t i = 0; i < event.parameters().size(); ++i) {
      assert(dynamic_cast<IntegerType const *>(calleeTy->parameterTypes()[i]) ||
             dynamic_cast<AddressType const *>(calleeTy->parameterTypes()[i]));

      // TODO? YulUtilFunctions::conversionFunction
      mlir::Value arg =
          genRValExpr(astArgs[i].get(), getType(calleeTy->parameterTypes()[i]));

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
    if (call->arguments().size() == 2) {
      const auto *msg = dynamic_cast<Literal const *>(astArgs[1].get());
      assert(msg && "NYI: Magic vars");
      b.create<mlir::sol::RequireOp>(loc, genRValExpr(astArgs[0].get()),
                                     b.getStringAttr(msg->value()));
    } else {
      b.create<mlir::sol::RequireOp>(loc, genRValExpr(astArgs[0].get()));
    }
    return {};
  }

  default:
    break;
  }

  llvm_unreachable("NYI");
}

mlir::Value SolidityToMLIRPass::genLValExpr(Expression const *expr) {
  // Variable access
  if (auto *ident = dynamic_cast<Identifier const *>(expr)) {
    auto addr = genLValExpr(ident);
    return addr;
  }

  // Index access
  if (auto *idxAcc = dynamic_cast<IndexAccess const *>(expr)) {
    return genLValExpr(idxAcc);
  }

  // (Compound) Assignment statement
  if (auto *asgnStmt = dynamic_cast<Assignment const *>(expr)) {
    mlir::Location loc = getLoc(asgnStmt->location());

    mlir::Value lhs = genLValExpr(&asgnStmt->leftHandSide());
    mlir::Value rhs = genRValExpr(&asgnStmt->rightHandSide());

    if (asgnStmt->assignmentOperator() == Token::Assign) {
      mlir::Type lhsTy = lhs.getType();
      mlir::Type rhsTy = rhs.getType();

      // Generate copy for assignment to storage reference types.
      if (mlir::sol::isNonPtrRefType(rhsTy) &&
          mlir::sol::getDataLocation(lhsTy) ==
              mlir::sol::DataLocation::Storage) {
        b.create<mlir::sol::CopyOp>(loc, rhs, lhs);
      } else {
        b.create<mlir::sol::StoreOp>(
            loc, genCast(rhs, mlir::sol::getEltType(lhs.getType())), lhs);
      }

      // Compound assignment statement
    } else {
      mlir::Value lhsAsRVal = genRValExpr(&asgnStmt->leftHandSide());
      Token binOp =
          TokenTraits::AssignmentToBinaryOp(asgnStmt->assignmentOperator());
      b.create<mlir::sol::StoreOp>(loc, genBinExpr(binOp, lhsAsRVal, rhs, loc),
                                   lhs);
    }

    return {};
  }

  return genRValExpr(expr);
}

mlir::Value SolidityToMLIRPass::genRValExpr(Expression const *expr,
                                            std::optional<mlir::Type> resTy) {
  mlir::Value val;

  // Literal
  if (auto *lit = dynamic_cast<Literal const *>(expr)) {
    val = genExpr(lit);
  }

  // Binary operation
  else if (auto *binOp = dynamic_cast<BinaryOperation const *>(expr)) {
    val = genExpr(binOp);
  }

  // Variable access
  else if (auto *id = dynamic_cast<Identifier const *>(expr)) {
    val = genRValExpr(id);

  }

  // Index access
  else if (auto *idxAcc = dynamic_cast<IndexAccess const *>(expr)) {
    val = genRValExpr(idxAcc);
  }

  // Member access
  else if (auto *memAcc = dynamic_cast<MemberAccess const *>(expr)) {
    val = genRValExpr(memAcc);
  }

  // Function call
  else if (auto *call = dynamic_cast<FunctionCall const *>(expr)) {
    val = genExpr(call);
  }

  else {
    llvm_unreachable("NYI");
  }

  // Generate cast (optional).
  if (resTy) {
    return genCast(val, *resTy);
  }

  return val;
}

bool SolidityToMLIRPass::visit(ExpressionStatement const &exprStmt) {
  genLValExpr(&exprStmt.expression());
  return true;
}

bool SolidityToMLIRPass::visit(
    VariableDeclarationStatement const &varDeclStmt) {
  assert(varDeclStmt.declarations().size() == 1 && "NYI");
  VariableDeclaration const *varDecl = varDeclStmt.declarations()[0].get();

  mlir::Location loc = getLoc(varDeclStmt.location());

  mlir::Type varTy = getType(varDecl->type());
  mlir::Type allocTy = mlir::sol::PointerType::get(
      b.getContext(), varTy, mlir::sol::DataLocation::Stack);

  auto addr = b.create<mlir::sol::AllocaOp>(loc, allocTy);
  trackLocalVarAddr(varDecl, addr);

  if (Expression const *initExpr = varDeclStmt.initialValue()) {
    b.create<mlir::sol::StoreOp>(loc, genRValExpr(initExpr, varTy), addr);
  } else {
    genZeroedVal(addr);
  }

  return true;
}

bool SolidityToMLIRPass::visit(EmitStatement const &emit) {
  genExpr(&emit.eventCall());
  return true;
}

bool SolidityToMLIRPass::visit(Return const &ret) {
  auto currFuncResTys =
      currFunc->functionType(/*FIXME*/ true)->returnParameterTypes();

  // The function generator emits `ReturnOp` for empty result
  if (currFuncResTys.empty())
    return true;

  assert(currFuncResTys.size() == 1 && "NYI: Multivalued return");

  Expression const *astExpr = ret.expression();
  if (astExpr) {
    mlir::Value expr =
        genRValExpr(ret.expression(), getType(currFuncResTys[0]));
    b.create<mlir::sol::ReturnOp>(getLoc(ret.location()), expr);
  } else {
    llvm_unreachable("NYI: Empty return");
  }

  return true;
}

bool SolidityToMLIRPass::visit(Block const &blk) {
  currBlk = &blk;
  return true;
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

void SolidityToMLIRPass::run(FunctionDefinition const &func) {
  currFunc = &func;
  std::vector<mlir::Type> inpTys, outTys;
  std::vector<mlir::Location> inpLocs;

  for (auto const &param : func.parameters()) {
    inpTys.push_back(getType(param->annotation().type));
    inpLocs.push_back(getLoc(param->location()));
  }

  for (auto const &param : func.returnParameters()) {
    outTys.push_back(getType(param->annotation().type));
  }

  assert(outTys.size() <= 1 && "NYI: Multivalued return");

  // TODO: Specify visibility
  auto funcType = b.getFunctionType(inpTys, outTys);
  auto op =
      b.create<mlir::sol::FuncOp>(getLoc(func.location()), getMangledName(func),
                                  funcType, getStateMutability(func));
  op.setCtor(func.isConstructor());
  if (func.isConstructor()) {
    auto currContr =
        mlir::cast<mlir::sol::ContractOp>(b.getBlock()->getParentOp());
    assert(currContr);
    currContr.setCtorFnType(op.getFunctionType());
  }

  mlir::Block *entryBlk = b.createBlock(&op.getRegion());
  b.setInsertionPointToStart(entryBlk);

  for (auto &&[inpTy, inpLoc, param] :
       ranges::views::zip(inpTys, inpLocs, func.parameters())) {
    mlir::Value arg = entryBlk->addArgument(inpTy, inpLoc);
    // TODO: Support non-scalars.
    auto addr = b.create<mlir::sol::AllocaOp>(
        inpLoc, mlir::sol::PointerType::get(b.getContext(), inpTy,
                                            mlir::sol::DataLocation::Stack));
    trackLocalVarAddr(param.get(), addr);
    b.create<mlir::sol::StoreOp>(inpLoc, arg, addr);
  }

  func.accept(*this);

  // Generate empty return
  if (outTys.empty())
    b.create<mlir::sol::ReturnOp>(getLoc(func.location()));

  b.setInsertionPointAfter(op);
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

void SolidityToMLIRPass::run(ContractDefinition const &cont) {
  currContract = &cont;

  // TODO: Set using ContractDefinition::receiveFunction and
  // ContractDefinition::fallbackFunction.
  mlir::FlatSymbolRefAttr fallbackFn, receiveFn;

  // Create the contract op.
  auto op = b.create<mlir::sol::ContractOp>(
      getLoc(cont.location()), cont.name() + "_" + util::toString(cont.id()),
      getContractKind(cont), getInterfaceFnsAttr(cont),
      /*ctorFnType=*/mlir::TypeAttr{}, fallbackFn, receiveFn);
  b.setInsertionPointToStart(&op.getBodyRegion().emplaceBlock());

  for (VariableDeclaration const *stateVar : cont.stateVariables()) {
    b.create<mlir::sol::StateVarOp>(getLoc(stateVar->location()),
                                    stateVar->name(),
                                    getType(stateVar->type()));
  }

  // Lower functions.
  for (auto *f : cont.definedFunctions()) {
    run(*f);
  }
  b.setInsertionPointAfter(op);
}

bool CompilerStack::runMlirPipeline(
    std::vector<ContractDefinition const *> const &contracts,
    CharStream const &stream, solidity::mlirgen::JobSpec const &job) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::sol::SolDialect>();

  SolidityToMLIRPass gen(ctx, stream);
  for (auto *contract : contracts) {
    gen.run(*contract);
  }
  mlir::ModuleOp mod = gen.getModule();

  if (failed(mlir::verify(mod))) {
    mod.emitError("Module verification error");
    return false;
  }

  return doJob(
      job, mod,
      m_contracts.at(contracts[0]->fullyQualifiedName()).mlirPipeline.bytecode);
}

void solidity::mlirgen::registerMLIRCLOpts() {
  mlir::registerAsmPrinterCLOptions();
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
