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
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolutil/CommonIO.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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
#include "range/v3/view/zip.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
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

  /// The function being lowered
  FunctionDefinition const *currFunc;

  // TODO: Use VariableDeclaration instead?
  // FIXME: Alignments are hardcoded.
  // FIXME: Should we specify alignment for the memory operations at this stage?
  /// Maps variables to its MemRef
  std::map<Declaration const *, mlir::Value> memRefMap;

  /// Returns the mlir location for the solidity source location `loc`
  mlir::Location getLoc(SourceLocation const &loc) {
    // FIXME: Track loc.end as well
    LineColumn lineCol = stream.translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream.name()),
                                     lineCol.line, lineCol.column);
  }

  /// Returns the corresponding mlir type for the solidity type `ty`.
  mlir::Type getType(Type const *ty);

  /// Returns the memory reference of the variable.
  mlir::Value getMemRef(Declaration const *decl);
  mlir::Value getMemRef(Identifier const *ident);

  /// Sets the memory reference of the variable.
  void setMemRef(Declaration const *decl, mlir::Value addr) {
    memRefMap[decl] = addr;
  }

  /// Returns the mangled name of the declaration composed of its name and its
  /// AST ID.
  std::string getMangledName(Declaration const &decl) {
    return decl.name() + "_" + std::to_string(decl.id());
  }

  /// Returns the array attribute for tracking interface functions (symbol, type
  /// and selector) in the contract.
  mlir::ArrayAttr getInterfaceFnsAttr(ContractDefinition const &cont);

  /// Returns the cast from `val` having the corresponding mlir type of
  /// `srcTy` to a value having the corresponding mlir type of `dstTy`
  mlir::Value genCast(mlir::Value val, Type const *srcTy, Type const *dstTy);

  /// Returns the mlir expression for the literal `lit`
  mlir::Value genExpr(Literal const *lit);

  /// Returns the mlir expression for the binary operation `binOp`
  mlir::Value genExpr(BinaryOperation const *binOp);

  /// Returns the mlir expression from `expr` and optionally casts it to the
  /// corresponding mlir type of `resTy`
  mlir::Value genExpr(Expression const *expr,
                      std::optional<Type const *> resTy = std::nullopt);

  bool visit(Return const &) override;
  void run(FunctionDefinition const &);
};

} // namespace solidity::frontend

mlir::Type SolidityToMLIRPass::getType(Type const *ty) {
  // Integer type
  if (auto *i = dynamic_cast<IntegerType const *>(ty)) {
    return b.getIntegerType(i->numBits());

    // Rational number type
  } else if (auto *ratNumTy = dynamic_cast<RationalNumberType const *>(ty)) {
    if (ratNumTy->isFractional())
      llvm_unreachable("NYI: Fractional type");

    // Integral rational number type
    const IntegerType *intTy = ratNumTy->integerType();
    return b.getIntegerType(intTy->numBits());

    // Function type
  } else if (auto *fnTy = dynamic_cast<FunctionType const *>(ty)) {
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

mlir::Value SolidityToMLIRPass::getMemRef(Declaration const *decl) {
  auto it = memRefMap.find(decl);
  if (it == memRefMap.end())
    return {};
  return it->second;
}

mlir::Value SolidityToMLIRPass::getMemRef(Identifier const *ident) {
  return getMemRef(ident->annotation().referencedDeclaration);
}

mlir::ArrayAttr
SolidityToMLIRPass::getInterfaceFnsAttr(ContractDefinition const &cont) {
  auto const &interfaceFnInfos = cont.interfaceFunctions();
  std::vector<mlir::Attribute> interfaceFnAttrs;
  interfaceFnAttrs.reserve(interfaceFnInfos.size());
  for (auto const &i : interfaceFnInfos) {
    auto fnSymAttr = mlir::SymbolRefAttr::get(
        b.getContext(), getMangledName(i.second->declaration()));

    mlir::FunctionType fnTy = getType(i.second).cast<mlir::FunctionType>();
    auto fnTyAttr = mlir::TypeAttr::get(fnTy);

    std::string selector = i.first.hex();
    auto selectorAttr = b.getStringAttr(selector);

    interfaceFnAttrs.push_back(b.getDictionaryAttr(
        {b.getNamedAttr("sym", fnSymAttr), b.getNamedAttr("type", fnTyAttr),
         b.getNamedAttr("selector", selectorAttr)}));
  }
  return b.getArrayAttr(interfaceFnAttrs);
}

mlir::Value SolidityToMLIRPass::genCast(mlir::Value val, Type const *srcTy,
                                        Type const *dstTy) {
  // Don't cast if we're casting to the same type
  if (srcTy == dstTy)
    return val;

  auto getAsIntTy = [](Type const *ty) -> IntegerType const * {
    auto intTy = dynamic_cast<IntegerType const *>(ty);
    if (!intTy) {
      if (auto *ratTy = dynamic_cast<RationalNumberType const *>(ty)) {
        if (auto *intRatTy = ratTy->integerType())
          return intRatTy;
      }
      return nullptr;
    }
    return intTy;
  };

  // We generate signless integral mlir::Types, so we must track the solidity
  // type to perform "sign aware lowering".
  //
  // Casting between integers
  auto srcIntTy = getAsIntTy(srcTy);
  auto dstIntTy = getAsIntTy(dstTy);

  if (srcIntTy && dstIntTy) {
    // Generate extends
    if (dstIntTy->numBits() > srcIntTy->numBits()) {
      return dstIntTy->isSigned()
                 ? b.create<mlir::arith::ExtSIOp>(val.getLoc(),
                                                  getType(dstIntTy), val)
                       ->getResult(0)
                 : b.create<mlir::arith::ExtUIOp>(val.getLoc(),
                                                  getType(dstIntTy), val)
                       ->getResult(0);
    } else {
      llvm_unreachable("NYI: Unknown cast");
    }
  }

  llvm_unreachable("NYI: Unknown cast");
}

mlir::Value SolidityToMLIRPass::genExpr(BinaryOperation const *binOp) {
  auto resTy = binOp->annotation().type;
  auto lc = getLoc(binOp->location());

  mlir::Value lhs = genExpr(&binOp->leftExpression(), resTy);
  mlir::Value rhs = genExpr(&binOp->rightExpression(), resTy);

  switch (binOp->getOperator()) {
  case Token::Add:
    return b.create<mlir::arith::AddIOp>(lc, lhs, rhs)->getResult(0);
  case Token::Mul:
    return b.create<mlir::arith::MulIOp>(lc, lhs, rhs)->getResult(0);
  default:
    break;
  }
  llvm_unreachable("NYI: Binary operator");
}

mlir::Value SolidityToMLIRPass::genExpr(Expression const *expr,
                                        std::optional<Type const *> resTy) {
  mlir::Value val;

  // Generate literals
  if (auto *lit = dynamic_cast<Literal const *>(expr)) {
    val = genExpr(lit);
  }
  // Generate variable access
  else if (auto *ident = dynamic_cast<Identifier const *>(expr)) {
    auto addr = getMemRef(ident);
    assert(addr);
    val = b.create<mlir::LLVM::LoadOp>(getLoc(expr->location()), addr,
                                       /*alignment=*/32);
  }
  // Generate binary operation
  else if (auto *binOp = dynamic_cast<BinaryOperation const *>(expr)) {
    val = genExpr(binOp);
  }

  // Generate cast (Optional)
  if (resTy) {
    return genCast(val, expr->annotation().type, *resTy);
  }

  return val;
}

mlir::Value SolidityToMLIRPass::genExpr(Literal const *lit) {
  mlir::Location lc = getLoc(lit->location());
  Type const *ty = lit->annotation().type;

  // Rational number literal
  if (auto *ratNumTy = dynamic_cast<RationalNumberType const *>(ty)) {
    if (ratNumTy->isFractional())
      llvm_unreachable("NYI: Fractional literal");

    auto *intTy = ratNumTy->integerType();
    u256 val = ty->literalValue(lit);
    // TODO: Is there a faster way to convert boost::multiprecision::number to
    // llvm::APInt?
    return b.create<mlir::arith::ConstantOp>(
        lc,
        b.getIntegerAttr(getType(ty), llvm::APInt(intTy->numBits(), val.str(),
                                                  /*radix=*/10)));
  } else {
    llvm_unreachable("NYI: Literal");
  }
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
    mlir::Value expr = genExpr(ret.expression(), currFuncResTys[0]);
    b.create<mlir::sol::ReturnOp>(getLoc(ret.location()), expr);
  } else {
    llvm_unreachable("NYI: Empty return");
  }

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

  mlir::Block *entryBlk = b.createBlock(&op.getRegion());
  b.setInsertionPointToStart(entryBlk);

  mlirgen::BuilderHelper h(b);
  for (auto &&[inpTy, inpLoc, param] :
       ranges::views::zip(inpTys, inpLocs, func.parameters())) {
    mlir::Value arg = entryBlk->addArgument(inpTy, inpLoc);
    // TODO: Support non-scalars.
    mlir::Value addr = b.create<mlir::LLVM::AllocaOp>(
                            inpLoc, mlir::LLVM::LLVMPointerType::get(inpTy),
                            h.getConst(1, 256, inpLoc), /*alignment=*/32)
                           .getResult();
    setMemRef(param.get(), addr);
    b.create<mlir::LLVM::StoreOp>(inpLoc, arg, addr, /*alignment=*/32);
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
  // TODO: Set using ContractDefinition::receiveFunction and
  // ContractDefinition::fallbackFunction.
  mlir::FlatSymbolRefAttr ctor, fallbackFn, receiveFn;

  // Create the contract op.
  auto op = b.create<mlir::sol::ContractOp>(
      getLoc(cont.location()), cont.name() + "_" + util::toString(cont.id()),
      getContractKind(cont), getInterfaceFnsAttr(cont), ctor, fallbackFn,
      receiveFn);
  b.setInsertionPointToStart(&op.getBodyRegion().emplaceBlock());

  // Lower functions.
  for (auto *f : cont.definedFunctions()) {
    run(*f);
  }
  b.setInsertionPointAfter(op);
}

bool solidity::mlirgen::runSolidityToMLIRPass(
    std::vector<ContractDefinition const *> const &contracts,
    CharStream const &stream, solidity::mlirgen::JobSpec const &job) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::sol::SolDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  SolidityToMLIRPass gen(ctx, stream);
  for (auto *contract : contracts) {
    gen.run(*contract);
  }
  mlir::ModuleOp mod = gen.getModule();

  if (failed(mlir::verify(mod))) {
    mod.emitError("Module verification error");
    return false;
  }

  return doJob(job, ctx, mod);
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
