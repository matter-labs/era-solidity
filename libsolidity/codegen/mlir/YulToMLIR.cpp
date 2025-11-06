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
// Yul to MLIR pass
//

#include "liblangutil/CharStream.h"
#include "liblangutil/EVMVersion.h"
#include "liblangutil/Exceptions.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Sol/Sol.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolidity/codegen/mlir/Yul/Yul.h"
#include "libsolutil/Visitor.h"
#include "libyul/AST.h"
#include "libyul/Dialect.h"
#include "libyul/Object.h"
#include "libyul/Utilities.h"
#include "libyul/optimiser/ASTWalker.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <optional>
#include <variant>

using namespace solidity::langutil;
using namespace solidity::yul;

namespace solidity::mlirgen {

class YulToMLIRPass : public ASTWalker {
  mlir::OpBuilder &b;
  CharStream const &stream;
  Dialect const &yulDialect;
  std::map<YulName, mlir::Value> localVarAddrMap;
  std::map<std::string,
           std::function<mlir::SmallVector<mlir::Value>(
               std::vector<Expression> const &args, mlir::Location loc)>>
      builtinGenMap;
  std::function<mlir::Value(Identifier const *)> externalRefResolver;

public:
  explicit YulToMLIRPass(
      mlir::OpBuilder &b, CharStream const &stream, Dialect const &yulDialect,
      std::function<mlir::Value(Identifier const *)> externalRefResolver = {})
      : b(b), stream(stream), yulDialect(yulDialect),
        externalRefResolver(std::move(externalRefResolver)) {
    populateBuiltinGenMap();
  }

  /// Lowers a block.
  void lowerBlk(Block const &);

  /// Lowers a subobject
  void lowerObj(Object const &obj);

  /// Lowers a top level object
  void lowerTopLevelObj(Object const &obj);

private:
  /// Returns the IntegerAttr for `num`
  mlir::IntegerAttr getIntAttr(LiteralValue const &num);

  /// Returns the mlir location for the solidity source location
  mlir::Location getLoc(SourceLocation const &loc) {
    // FIXME: Track loc.end as well
    LineColumn lineCol = stream.translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream.name()),
                                     lineCol.line, lineCol.column);
  }
  mlir::Location getLoc(std::shared_ptr<DebugData const> const &dbg) {
    return getLoc(dbg->nativeLocation);
  }

  /// Returns the default integral type
  mlir::IntegerType getDefIntTy() { return b.getIntegerType(256); }

  /// Returns the default alignment
  uint64_t getDefAlign() {
    uint64_t width = getDefIntTy().getWidth();
    assert(width >= 8 && llvm::isPowerOf2_64(width));
    return width / 8;
  }

  /// "Converts" `val` to boolean. Integral values are converted to the result
  /// of non-zero check
  mlir::Value convToBool(mlir::Value val);

  /// Tracks the address of the local variable.
  void trackLocalVarAddr(YulName var, mlir::Value addr) {
    localVarAddrMap[var] = addr;
  }

  /// Returns the address of the local variable.
  mlir::Value getLocalVarAddr(YulName var);
  mlir::Value getLocalVarAddr(Identifier const &var) {
    if (externalRefResolver) {
      mlir::Value externalRef = externalRefResolver(&var);
      if (externalRef)
        return externalRef;
    }
    return getLocalVarAddr(var.name);
  }

  template <typename T>
  T lookupSymbol(llvm::StringRef name) {
    return mlir::SymbolTable::lookupNearestSymbolFrom<T>(
        b.getBlock()->getParentOp(), b.getStringAttr(name));
  }

  /// Defines a simple builtin codegen map.
  template <typename OpT, bool reverseArgs = false, bool noRet = false>
  void defSimpleBuiltinGen(const char *str) {
    builtinGenMap[str] = [&](std::vector<Expression> const &args,
                             mlir::Location loc) {
      mlir::SmallVector<mlir::Value, 3> ins;

      if constexpr (reverseArgs) {
        for (size_t i = args.size(); i--;)
          ins.push_back(genDefTyExpr(args[i]));
      } else {
        for (const Expression &arg : args)
          ins.push_back(genDefTyExpr(arg));
      }

      mlir::SmallVector<mlir::Value, 2> resVals;
      if constexpr (noRet)
        b.create<OpT>(loc, mlir::TypeRange{}, ins);
      else
        resVals.push_back(
            b.create<OpT>(loc, mlir::TypeRange{getDefIntTy()}, ins));

      return resVals;
    };
  }
  template <typename OpT>
  void defSimpleBuiltinGenNoRet(const char *str) {
    defSimpleBuiltinGen<OpT, /*reverseArgs=*/false, /*noRet=*/true>(str);
  }

  /// Defines builtin codegen map for cmp builtins.
  template <mlir::arith::CmpIPredicate predicate>
  void defCmpBuiltinGen(const char *name) {
    builtinGenMap[name] = [&](std::vector<Expression> const &args,
                              mlir::Location loc) {
      mlir::SmallVector<mlir::Value, 2> resVals;
      resVals.push_back(b.create<mlir::arith::CmpIOp>(
          loc, predicate, genDefTyExpr(args[0]), genDefTyExpr(args[1])));
      return resVals;
    };
  }

  /// Populates builtinGenMap with codegen of all the builtins.
  void populateBuiltinGenMap();

  /// Returns a cast expression.
  mlir::Value genCast(mlir::Value val, mlir::IntegerType dstTy);

  /// Returns the mlir expression for the literal `lit`
  mlir::Value genExpr(Literal const &lit);

  /// Returns the mlir expression for the identifier `id`
  mlir::Value genExpr(Identifier const &id);

  /// Returns the mlir expression for the function call `call`
  mlir::Value genExpr(FunctionCall const &call);
  mlir::SmallVector<mlir::Value> genExprs(FunctionCall const &call);

  /// Returns the mlir expression (optionally casts it `resTy`) for the
  /// expression `expr`
  mlir::Value genExpr(Expression const &expr,
                      std::optional<mlir::IntegerType> resTy = std::nullopt);
  mlir::SmallVector<mlir::Value> genExprs(Expression const &expr);

  /// Returns the mlir expression cast'ed to the default type for the expression
  /// `expr`
  mlir::Value genDefTyExpr(Expression const &expr);
  mlir::SmallVector<mlir::Value> genDefTyExprs(Expression const &expr);

  /// Lowers an expression statement
  void operator()(ExpressionStatement const &expr) override;

  /// Lowers an assignment statement
  void operator()(Assignment const &asgn) override;

  /// Lowers a variable decl
  void operator()(VariableDeclaration const &decl) override;

  /// Lowers an if statement
  void operator()(If const &ifStmt) override;

  /// Lowers a switch statement
  void operator()(Switch const &switchStmt) override;

  /// Lowers the break statement.
  void operator()(Break const &) override;

  /// Lowers the continue statement.
  void operator()(Continue const &) override;

  /// Lowers the for statement.
  void operator()(ForLoop const &) override;

  /// Lowers a function
  void operator()(FunctionDefinition const &fn) override;

  /// Lowers a block
  void operator()(Block const &blk) override;
};

/// Returns the llvm::APInt of `num`
static llvm::APInt getAPInt(std::string const &num, unsigned width) {
  llvm::StringRef numStr = num;
  uint8_t radix = 10;
  if (numStr.consume_front("0x")) {
    radix = 16;
  }
  return llvm::APInt(width, numStr, radix);
}

mlir::IntegerAttr YulToMLIRPass::getIntAttr(LiteralValue const &num) {
  auto defTy = getDefIntTy();
  return b.getIntegerAttr(defTy, getAPInt(num.value().str(), defTy.getWidth()));
}

mlir::Value YulToMLIRPass::getLocalVarAddr(YulString var) {
  auto it = localVarAddrMap.find(var);
  assert(it != localVarAddrMap.end());
  return it->second;
}

mlir::Value YulToMLIRPass::convToBool(mlir::Value val) {
  mlir::Location loc = val.getLoc();
  mlirgen::BuilderExt bExt(b, loc);

  auto ty = mlir::cast<mlir::IntegerType>(val.getType());
  if (ty.getWidth() == 1)
    return val;
  if (ty == getDefIntTy())
    return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                         val, bExt.genI256Const(0));
  llvm_unreachable("Invalid type");
}

void YulToMLIRPass::populateBuiltinGenMap() {
  using namespace mlir;
  using namespace mlir::yul;
  defSimpleBuiltinGen<arith::AddIOp>("add");
  defSimpleBuiltinGen<arith::SubIOp>("sub");
  defSimpleBuiltinGen<arith::MulIOp>("mul");
  defSimpleBuiltinGen<arith::AndIOp>("and");
  defSimpleBuiltinGen<arith::OrIOp>("or");
  defSimpleBuiltinGen<arith::XOrIOp>("xor");
  defSimpleBuiltinGen<DivOp>("div");
  defSimpleBuiltinGen<SDivOp>("sdiv");
  defSimpleBuiltinGen<ModOp>("mod");
  defSimpleBuiltinGen<SModOp>("smod");
  defSimpleBuiltinGen<ShrOp, /*reverseArgs=*/true>("shr");
  defSimpleBuiltinGen<ShlOp, /*reverseArgs=*/true>("shl");
  defSimpleBuiltinGen<SarOp, /*reverseArgs=*/true>("sar");
  defCmpBuiltinGen<arith::CmpIPredicate::ult>("lt");
  defCmpBuiltinGen<arith::CmpIPredicate::slt>("slt");
  defCmpBuiltinGen<arith::CmpIPredicate::ugt>("gt");
  defCmpBuiltinGen<arith::CmpIPredicate::sgt>("sgt");
  defCmpBuiltinGen<arith::CmpIPredicate::eq>("eq");
  builtinGenMap["not"] = [&](std::vector<Expression> const &args,
                             mlir::Location loc) {
    mlir::SmallVector<mlir::Value, 1> resVals;
    mlirgen::BuilderExt bExt(b, loc);
    resVals.push_back(b.create<mlir::arith::XOrIOp>(loc, genDefTyExpr(args[0]),
                                                    bExt.genI256Const(-1)));
    return resVals;
  };
  builtinGenMap["iszero"] = [&](std::vector<Expression> const &args,
                                mlir::Location loc) {
    mlir::SmallVector<mlir::Value, 2> resVals;
    mlirgen::BuilderExt bExt(b, loc);
    resVals.push_back(b.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, genDefTyExpr(args[0]),
        bExt.genI256Const(0)));
    return resVals;
  };
  defSimpleBuiltinGen<AddModOp>("addmod");
  defSimpleBuiltinGen<MulModOp>("mulmod");
  defSimpleBuiltinGen<SignExtendOp>("signextend");
  defSimpleBuiltinGen<MLoadOp>("mload");
  defSimpleBuiltinGenNoRet<LoadImmutableOp>("loadimmutable");
  defSimpleBuiltinGenNoRet<MStoreOp>("mstore");
  defSimpleBuiltinGenNoRet<MStore8Op>("mstore8");
  defSimpleBuiltinGenNoRet<MCopyOp>("mcopy");
  defSimpleBuiltinGenNoRet<SetImmutableOp>("setimmutable");
  defSimpleBuiltinGen<MSizeOp>("msize");
  defSimpleBuiltinGen<ByteOp>("byte");
  defSimpleBuiltinGen<CallDataLoadOp>("calldataload");
  defSimpleBuiltinGenNoRet<CallDataCopyOp>("calldatacopy");
  defSimpleBuiltinGen<CallDataSizeOp>("calldatasize");
  defSimpleBuiltinGenNoRet<ReturnDataCopyOp>("returndatacopy");
  defSimpleBuiltinGen<ReturnDataSizeOp>("returndatasize");
  defSimpleBuiltinGenNoRet<CodeCopyOp>("codecopy");
  defSimpleBuiltinGen<CodeSizeOp>("codesize");
  defSimpleBuiltinGen<ExtCodeSizeOp>("extcodesize");
  defSimpleBuiltinGenNoRet<ExtCodeCopyOp>("extcodecopy");
  defSimpleBuiltinGen<CreateOp>("create");
  defSimpleBuiltinGen<Create2Op>("create2");
  defSimpleBuiltinGen<SLoadOp>("sload");
  defSimpleBuiltinGenNoRet<SStoreOp>("sstore");
  defSimpleBuiltinGen<TLoadOp>("tload");
  defSimpleBuiltinGenNoRet<TStoreOp>("tstore");
  defSimpleBuiltinGenNoRet<ReturnOp>("return");
  defSimpleBuiltinGenNoRet<RevertOp>("revert");
  defSimpleBuiltinGenNoRet<StopOp>("stop");
  defSimpleBuiltinGen<Keccak256Op>("keccak256");
  defSimpleBuiltinGen<ExpOp>("exp");
  defSimpleBuiltinGen<CallValOp>("callvalue");
  defSimpleBuiltinGen<AddressOp>("address");
  defSimpleBuiltinGen<CallerOp>("caller");
  defSimpleBuiltinGen<GasOp>("gas");
  defSimpleBuiltinGen<CallOp>("call");
  defSimpleBuiltinGen<StaticCallOp>("staticcall");
  defSimpleBuiltinGen<DelegateCallOp>("delegatecall");
  defSimpleBuiltinGenNoRet<LogOp>("log0");
  defSimpleBuiltinGenNoRet<LogOp>("log1");
  defSimpleBuiltinGenNoRet<LogOp>("log2");
  defSimpleBuiltinGenNoRet<LogOp>("log3");
  defSimpleBuiltinGenNoRet<LogOp>("log4");
  builtinGenMap["dataoffset"] = [&](std::vector<Expression> const &args,
                                    Location loc) {
    SmallVector<Value, 2> resVals;
    auto *objectName = std::get_if<Literal>(&args[0]);
    assert(objectName);
    assert(objectName->kind == LiteralKind::String);
    auto objectOp =
        lookupSymbol<ObjectOp>(objectName->value.builtinStringLiteralValue());
    assert(objectOp && "NYI: References to external object");
    resVals.push_back(
        b.create<DataOffsetOp>(loc, FlatSymbolRefAttr::get(objectOp)));
    return resVals;
  };
  builtinGenMap["datasize"] = [&](std::vector<Expression> const &args,
                                  Location loc) {
    SmallVector<Value, 2> resVals;
    auto *objectName = std::get_if<Literal>(&args[0]);
    assert(objectName);
    assert(objectName->kind == LiteralKind::String);
    auto objectOp =
        lookupSymbol<ObjectOp>(objectName->value.builtinStringLiteralValue());
    assert(objectOp && "NYI: References to external object");
    resVals.push_back(
        b.create<DataSizeOp>(loc, FlatSymbolRefAttr::get(objectOp)));
    return resVals;
  };
  builtinGenMap["memoryguard"] = [&](std::vector<Expression> const &args,
                                     Location loc) {
    SmallVector<Value, 2> resVals;
    auto *arg = std::get_if<Literal>(&args[0]);
    assert(arg);
    resVals.push_back(b.create<MemGuardOp>(loc, getIntAttr(arg->value)));
    return resVals;
  };
}

mlir::Value YulToMLIRPass::genExpr(Literal const &lit) {
  mlir::Location loc = this->getLoc(lit.debugData);

  // TODO: Do we need to represent constants as u256? Can we do that in
  // arith::ConstantOp?
  return b.create<mlir::arith::ConstantOp>(loc, getIntAttr(lit.value));
}

mlir::Value YulToMLIRPass::genExpr(Identifier const &id) {
  mlir::Value addr = getLocalVarAddr(id);
  return b.create<mlir::LLVM::LoadOp>(
      getLoc(id.debugData), /*resTy=*/getDefIntTy(), addr, getDefAlign());
}

mlir::SmallVector<mlir::Value>
YulToMLIRPass::genExprs(FunctionCall const &call) {
  BuiltinFunction const *builtin =
      yul::resolveBuiltinFunction(call.functionName, yulDialect);
  mlir::Location loc = getLoc(call.debugData);
  mlirgen::BuilderExt bExt(b, loc);

  mlir::SmallVector<mlir::Value> resVals;
  if (builtin) {
    assert(builtinGenMap.count(builtin->name));
    return builtinGenMap[builtin->name](call.arguments, loc);
  }

  mlir::sol::FuncOp callee = lookupSymbol<mlir::sol::FuncOp>(
      yul::resolveFunctionName(call.functionName, yulDialect));
  assert(callee);
  std::vector<mlir::Value> args;
  args.reserve(call.arguments.size());
  for (Expression const &arg : call.arguments) {
    args.push_back(genDefTyExpr(arg));
  }
  auto callOp = b.create<mlir::sol::CallOp>(loc, callee, args);
  for (mlir::Value res : callOp.getResults())
    resVals.push_back(res);
  return resVals;
}

mlir::Value YulToMLIRPass::genExpr(FunctionCall const &call) {
  mlir::SmallVector<mlir::Value> exprs = genExprs(call);
  assert(exprs.size() < 2);
  return exprs.empty() ? mlir::Value{} : exprs.front();
}

mlir::Value YulToMLIRPass::genCast(mlir::Value val, mlir::IntegerType dstTy) {
  mlir::IntegerType valTy = mlir::cast<mlir::IntegerType>(val.getType());
  if (valTy == dstTy)
    return val;
  assert(valTy.getWidth() > dstTy.getWidth());
  return b.create<mlir::arith::ExtUIOp>(val.getLoc(), dstTy, val);
}

mlir::Value YulToMLIRPass::genExpr(Expression const &expr,
                                   std::optional<mlir::IntegerType> resTy) {
  mlir::Value gen = std::visit(
      [&](auto &&resolvedExpr) { return this->genExpr(resolvedExpr); }, expr);

  if (resTy) {
    assert(gen);
    auto genTy = mlir::cast<mlir::IntegerType>(gen.getType());
    if (*resTy != genTy) {
      assert(resTy->getWidth() > genTy.getWidth());
      // Zero-extend the result to `resTy`
      return b.create<mlir::arith::ExtUIOp>(gen.getLoc(), *resTy, gen);
    }
  }

  return gen;
}

mlir::SmallVector<mlir::Value> YulToMLIRPass::genExprs(Expression const &expr) {
  mlir::SmallVector<mlir::Value, 2> resVals;
  std::visit(util::GenericVisitor{
                 [&](FunctionCall const &call) { resVals = genExprs(call); },
                 [&](auto &&resolvedExpr) {
                   resVals.push_back(this->genExpr(resolvedExpr));
                 }},
             expr);

  return resVals;
}

mlir::Value YulToMLIRPass::genDefTyExpr(Expression const &expr) {
  return genExpr(expr, getDefIntTy());
}

mlir::SmallVector<mlir::Value>
YulToMLIRPass::genDefTyExprs(Expression const &expr) {
  mlir::SmallVector<mlir::Value> mlirExprs = genExprs(expr);
  mlir::SmallVector<mlir::Value, 2> castedExprs;
  for (mlir::Value mlirExpr : mlirExprs)
    castedExprs.push_back(genCast(mlirExpr, getDefIntTy()));
  return castedExprs;
}

void YulToMLIRPass::operator()(ExpressionStatement const &expr) {
  genExpr(expr.expression);
}

void YulToMLIRPass::operator()(Assignment const &asgn) {
  mlir::Location loc = getLoc(asgn.debugData);
  mlir::SmallVector<mlir::Value> rhsExprs = genDefTyExprs(*asgn.value);
  assert(asgn.variableNames.size() == rhsExprs.size());

  for (auto [lhs, rhsExpr] : llvm::zip(asgn.variableNames, rhsExprs)) {
    b.create<mlir::LLVM::StoreOp>(loc, rhsExpr, getLocalVarAddr(lhs),
                                  getDefAlign());
  }
}

void YulToMLIRPass::operator()(VariableDeclaration const &decl) {
  mlir::Location loc = getLoc(decl.debugData);
  mlirgen::BuilderExt bExt(b, loc);

  mlir::SmallVector<mlir::Value> rhsExprs = genDefTyExprs(*decl.value);
  for (auto [var, rhsExpr] : llvm::zip(decl.variables, rhsExprs)) {
    auto addr = b.create<mlir::LLVM::AllocaOp>(
        getLoc(var.debugData),
        /*resTy=*/mlir::LLVM::LLVMPointerType::get(b.getContext()),
        /*eltTy=*/getDefIntTy(), bExt.genI256Const(1), getDefAlign());
    trackLocalVarAddr(var.name, addr);
    b.create<mlir::LLVM::StoreOp>(loc, rhsExpr, addr, getDefAlign());
  }
}

void YulToMLIRPass::operator()(If const &ifStmt) {
  mlir::Location loc = getLoc(ifStmt.debugData);

  // TODO: Should we expand here? Or is it beneficial to represent `if` with a
  // non-boolean condition in the IR?
  auto ifOp =
      b.create<mlir::sol::IfOp>(loc, convToBool(genExpr(*ifStmt.condition)));
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  b.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
  ASTWalker::operator()(ifStmt.body);
  b.create<mlir::sol::YieldOp>(ifOp.getLoc());
}

void YulToMLIRPass::operator()(Switch const &switchStmt) {
  mlir::Location loc = getLoc(switchStmt.debugData);

  // Create the mlir attribute for all the case values (excluding the default
  // case); Track the default case AST.
  Case const *defCaseAST = nullptr;
  std::vector<llvm::APInt> caseVals;
  caseVals.reserve(switchStmt.cases.size());
  std::vector<Case const *> caseASTs;
  for (Case const &caseAST : switchStmt.cases) {
    // If non default block
    if (caseAST.value) {
      caseASTs.push_back(&caseAST);
      // FIXME (libyul): Getting the literal value of a case statement AST
      // shouldn't look this ugly!
      caseVals.push_back(getAPInt(caseAST.value->value.value().str(),
                                  getDefIntTy().getWidth()));

    } else {
      // There should only be one default case.
      assert(!defCaseAST);
      defCaseAST = &caseAST;
    }
  }
  assert(defCaseAST && "NYI: Switch block without a default case");
  auto caseValsAttr = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(static_cast<int64_t>(caseVals.size()),
                                  getDefIntTy()),
      caseVals);

  // Lower the switch argument and generate the switch op.
  mlir::Value arg = genExpr(*switchStmt.expression);
  auto switchOp = b.create<mlir::sol::SwitchOp>(
      loc, /*resultTypes=*/std::nullopt, arg, caseValsAttr, caseVals.size());
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  // Create blocks for all the case values and the default case. Then, lower
  // their body.
  auto lowerBody = [&](mlir::Region &region, Case const &caseAST) {
    mlir::Block *blk = b.createBlock(&region);
    b.setInsertionPointToStart(blk);
    b.create<mlir::sol::YieldOp>(loc);
    b.setInsertionPointToStart(blk);
    ASTWalker::operator()(caseAST.body);
  };
  lowerBody(switchOp.getDefaultRegion(), *defCaseAST);
  assert(switchOp.getCaseRegions().size() == caseASTs.size());
  for (auto [region, caseAST] : llvm::zip(switchOp.getCaseRegions(), caseASTs))
    lowerBody(region, *caseAST);
}

void YulToMLIRPass::operator()(Break const &brkStmt) {
  b.create<mlir::sol::BreakOp>(getLoc(brkStmt.debugData));
  mlir::Block *newBlock = b.getBlock()->splitBlock(b.getInsertionPoint());
  b.setInsertionPointToStart(newBlock);
}

void YulToMLIRPass::operator()(Continue const &contStmt) {
  b.create<mlir::sol::ContinueOp>(getLoc(contStmt.debugData));
  mlir::Block *newBlock = b.getBlock()->splitBlock(b.getInsertionPoint());
  b.setInsertionPointToStart(newBlock);
}

void YulToMLIRPass::operator()(ForLoop const &forStmt) {
  mlirgen::BuilderExt bExt(b);

  // Lower pre block.
  ASTWalker::operator()(forStmt.pre);

  auto forOp = b.create<mlir::sol::ForOp>(getLoc(forStmt.debugData));
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  // Lower condition.
  b.setInsertionPointToStart(&forOp.getCond().emplaceBlock());
  mlir::Value cond = forStmt.condition ? convToBool(genExpr(*forStmt.condition))
                                       : bExt.genBool(true, forOp.getLoc());
  b.create<mlir::sol::ConditionOp>(cond.getLoc(), cond);

  // Lower body.
  b.setInsertionPointToStart(&forOp.getBody().emplaceBlock());
  ASTWalker::operator()(forStmt.body);
  b.create<mlir::sol::YieldOp>(forOp.getLoc());

  // Lower post block.
  b.setInsertionPointToStart(&forOp.getStep().emplaceBlock());
  ASTWalker::operator()(forStmt.post);
  b.create<mlir::sol::YieldOp>(forOp.getLoc());
}

void YulToMLIRPass::operator()(FunctionDefinition const &fn) {
  mlir::Location loc = getLoc(fn.debugData);
  mlirgen::BuilderExt bExt(b, loc);

  // Lookup FuncOp (should be declared by the yul block lowering).
  auto fnOp = lookupSymbol<mlir::sol::FuncOp>(fn.name.str());
  assert(fnOp);

  // Restore the insertion point after lowering the function definition.
  mlir::OpBuilder::InsertionGuard insertGuard(b);

  // Generate entry block and args.
  mlir::Block *entryBlk = b.createBlock(&fnOp.getRegion());
  std::vector<mlir::Location> inpLocs;
  unsigned i = 0;
  for (const NameWithDebugData &in : fn.parameters) {
    mlir::BlockArgument blkArg = entryBlk->addArgument(
        fnOp.getFunctionType().getInput(i++), getLoc(in.debugData));
    auto addr = b.create<mlir::LLVM::AllocaOp>(
        blkArg.getLoc(),
        /*resTy=*/mlir::LLVM::LLVMPointerType::get(b.getContext()),
        /*eltTy=*/getDefIntTy(), bExt.genI256Const(1), getDefAlign());
    trackLocalVarAddr(in.name, addr);
  }
  for (const NameWithDebugData &retVar : fn.returnVariables) {
    auto addr = b.create<mlir::LLVM::AllocaOp>(
        getLoc(retVar.debugData),
        /*resTy=*/mlir::LLVM::LLVMPointerType::get(b.getContext()),
        /*eltTy=*/getDefIntTy(), bExt.genI256Const(1), getDefAlign());
    trackLocalVarAddr(retVar.name, addr);
  }

  // Lower the body.
  ASTWalker::operator()(fn.body);

  mlir::SmallVector<mlir::Value> retVarLds;
  for (const NameWithDebugData &retVar : fn.returnVariables) {
    auto ld = b.create<mlir::LLVM::LoadOp>(
        getLoc(retVar.debugData), /*resTy=*/getDefIntTy(),
        getLocalVarAddr(retVar.name), getDefAlign());
    retVarLds.push_back(ld);
  }
  b.create<mlir::sol::ReturnOp>(loc, retVarLds);
}

void YulToMLIRPass::operator()(Block const &blk) { lowerBlk(blk); }

void YulToMLIRPass::lowerBlk(Block const &blk) {
  // "Forward declare" FuncOps (i.e. create them with an empty region) at this
  // block so that we can lower calls before lowering the functions. The
  // function lowering is expected to lookup the FuncOp without creating it.
  //
  // TODO: Stop relying on libyul's Disambiguator
  // We tried emitting a single block op for yul blocks with a symbol table
  // trait. We're able to define symbols with the same name in different blocks,
  // but ops like sol::CallOp works with a FlatSymbolRefAttr which needs the
  // symbol definition to be in the same symbol table
  for (Statement const &stmt : blk.statements) {
    if (const auto *fn = std::get_if<FunctionDefinition>(&stmt)) {
      std::vector<mlir::Type> inTys(fn->parameters.size(), getDefIntTy()),
          outTys(fn->returnVariables.size(), getDefIntTy());
      mlir::FunctionType funcTy = b.getFunctionType(inTys, outTys);
      b.create<mlir::sol::FuncOp>(getLoc(fn->debugData), fn->name.str(),
                                  funcTy);
    }
  }

  ASTWalker::operator()(blk);
}

void YulToMLIRPass::lowerObj(Object const &obj) {
  // Lookup ObjectOp (should be declared by the top level object lowering)
  auto op = lookupSymbol<mlir::yul::ObjectOp>(obj.name);
  assert(op);

  b.setInsertionPointToStart(op.getEntryBlock());
  // TODO? Do we need a separate op for the `code` block?
  lowerBlk(obj.code()->root());
}

void YulToMLIRPass::lowerTopLevelObj(Object const &obj) {
  // "Forward declare" ObjectOp for the top level object and its sub-objects so
  // that we can create symbol references to them (for builtins like dataoffset)
  //
  // TODO: Where is the source location info for Object? Do we need to track it?
  auto topLevelObj = b.create<mlir::yul::ObjectOp>(b.getUnknownLoc(), obj.name);
  {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToEnd(topLevelObj.getEntryBlock());
    for (auto const &subNode : obj.subObjects) {
      if (auto *subObj = dynamic_cast<Object const *>(subNode.get())) {
        b.create<mlir::yul::ObjectOp>(b.getUnknownLoc(), subObj->name);
      }
    }
  }

  lowerObj(obj);
  // TODO: Does it make sense to nest subobjects in the top level ObjectOp's
  // body?
  for (auto const &subNode : obj.subObjects) {
    if (auto *subObj = dynamic_cast<Object const *>(subNode.get())) {
      lowerObj(*subObj);
    } else {
      llvm_unreachable("NYI: Metadata");
    }
  }
}

} // namespace solidity::mlirgen

void solidity::mlirgen::runYulToMLIRPass(
    yul::AST const &ast, CharStream const &stream,
    std::function<mlir::Value(Identifier const *)> const &externalRefResolver,
    mlir::OpBuilder &b) {
  solidity::mlirgen::YulToMLIRPass yulToMLIR(b, stream, ast.dialect(),
                                             externalRefResolver);
  yulToMLIR.lowerBlk(ast.root());
}

bool solidity::mlirgen::runYulToMLIRPass(Object const &obj,
                                         CharStream const &stream,
                                         Dialect const &yulDialect,
                                         JobSpec const &job,
                                         EVMVersion evmVersion) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.getOrLoadDialect<mlir::sol::SolDialect>();
  ctx.getOrLoadDialect<mlir::yul::YulDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  mlir::OpBuilder b(&ctx);
  mlir::ModuleOp mod = mlir::ModuleOp::create(b.getUnknownLoc());
  mod->setAttr("sol.evm_version",
               mlir::sol::EvmVersionAttr::get(
                   b.getContext(), *mlir::sol::symbolizeEvmVersion(
                                       evmVersion.getVersionAsInt())));
  b.setInsertionPointToEnd(mod.getBody());
  solidity::mlirgen::YulToMLIRPass yulToMLIR(b, stream, yulDialect);
  yulToMLIR.lowerTopLevelObj(obj);

  if (failed(mlir::verify(mod))) {
    mod.print(llvm::errs());
    mod.emitError("Module verification error");
    return false;
  }

  // FIXME:
  assert(job.action != solidity::mlirgen::Action::GenObj);
  llvm::outs() << mlirgen::printJob(job, mod);
  return true;
}
