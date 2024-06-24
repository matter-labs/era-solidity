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

#include "Sol/SolOps.h"
#include "liblangutil/CharStream.h"
#include "liblangutil/Exceptions.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libyul/AST.h"
#include "libyul/Dialect.h"
#include "libyul/Object.h"
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
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>
#include <variant>

using namespace solidity::langutil;
using namespace solidity::yul;

namespace solidity::mlirgen {

class YulToMLIRPass : public ASTWalker {
  mlir::OpBuilder b;
  mlir::ModuleOp mod;
  mlir::sol::ObjectOp currObj;
  CharStream const &stream;
  Dialect const &yulDialect;

public:
  mlir::ModuleOp getModule() { return mod; }

  explicit YulToMLIRPass(mlir::MLIRContext &ctx, CharStream const &stream,
                         Dialect const &yulDialect)
      : b(&ctx), stream(stream), yulDialect(yulDialect) {
    mod = mlir::ModuleOp::create(b.getUnknownLoc());
    b.setInsertionPointToEnd(mod.getBody());
  }

  /// Lowers a subobject
  void lowerObj(Object const &obj);

  /// Lowers a top level object
  void lowerTopLevelObj(Object const &obj);

private:
  /// Maps a yul variable name to its MemRef
  std::map<YulString, mlir::Value> memRefMap;

  /// Returns the IntegerAttr for `num`
  mlir::IntegerAttr getIntAttr(YulString num);

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

  /// Sets the MemRef addr of yul variable
  void setMemRef(YulString var, mlir::Value addr) { memRefMap[var] = addr; }

  /// Returns the MemRef addr of yul variable
  mlir::Value getMemRef(YulString var);

  /// Returns the symbol of type `T` in the current scope
  template <typename T>
  T lookupSymbol(llvm::StringRef name) {
    if (!currObj) {
      assert(mod);
      return mod.lookupSymbol<T>(name);
    }
    // FIXME: We should lookup in the current block and its ancestors
    return currObj.lookupSymbol<T>(name);
  }

  /// Returns the mlir expression for the literal `lit`
  mlir::Value genExpr(Literal const &lit);

  /// Returns the mlir expression for the identifier `id`
  mlir::Value genExpr(Identifier const &id);

  /// Returns the mlir expression for the function call `call`
  mlir::Value genExpr(FunctionCall const &call);

  /// Returns the mlir expression (optionally casts it `resTy`) for the
  /// expression `expr`
  mlir::Value genExpr(Expression const &expr,
                      std::optional<mlir::IntegerType> resTy = std::nullopt);

  /// Returns the mlir expression cast'ed to the default type for the expression
  /// `expr`
  mlir::Value genDefTyExpr(Expression const &expr);

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

  /// Lowers a function
  void operator()(FunctionDefinition const &fn) override;

  /// Lowers a block
  void operator()(Block const &blk) override;
};

/// Returns the llvm::APInt of `num`
static llvm::APInt getAPInt(YulString num, unsigned width) {
  llvm::StringRef numStr = num.str();
  uint8_t radix = 10;
  if (numStr.consume_front("0x")) {
    radix = 16;
  }
  return llvm::APInt(width, numStr, radix);
}

mlir::IntegerAttr YulToMLIRPass::getIntAttr(YulString num) {
  auto defTy = getDefIntTy();
  return b.getIntegerAttr(defTy, getAPInt(num, defTy.getWidth()));
}

mlir::Value YulToMLIRPass::getMemRef(YulString var) {
  auto it = memRefMap.find(var);
  if (it == memRefMap.end())
    return {};
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

mlir::Value YulToMLIRPass::genExpr(Literal const &lit) {
  mlir::Location loc = this->getLoc(lit.debugData);

  // TODO: Do we need to represent constants as u256? Can we do that in
  // arith::ConstantOp?
  return b.create<mlir::arith::ConstantOp>(loc, getIntAttr(lit.value));
}

mlir::Value YulToMLIRPass::genExpr(Identifier const &id) {
  mlir::Value addr = getMemRef(id.name);
  assert(addr);
  return b.create<mlir::LLVM::LoadOp>(getLoc(id.debugData), addr,
                                      getDefAlign());
}

mlir::Value YulToMLIRPass::genExpr(FunctionCall const &call) {
  BuiltinFunction const *builtin = yulDialect.builtin(call.functionName.name);
  mlir::Location loc = getLoc(call.debugData);
  mlirgen::BuilderExt bExt(b, loc);
  if (builtin) {
    if (builtin->name.str() == "add") {
      return b.create<mlir::arith::AddIOp>(loc, genDefTyExpr(call.arguments[0]),
                                           genDefTyExpr(call.arguments[1]));
    }
    if (builtin->name.str() == "sub") {
      return b.create<mlir::arith::SubIOp>(loc, genDefTyExpr(call.arguments[0]),
                                           genDefTyExpr(call.arguments[1]));
    }
    if (builtin->name.str() == "lt") {
      return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult,
                                           genDefTyExpr(call.arguments[0]),
                                           genDefTyExpr(call.arguments[1]));
    }
    if (builtin->name.str() == "slt") {
      return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
                                           genDefTyExpr(call.arguments[0]),
                                           genDefTyExpr(call.arguments[1]));
    }
    if (builtin->name.str() == "iszero") {
      return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                           genDefTyExpr(call.arguments[0]),
                                           bExt.genI256Const(0));
    }
    if (builtin->name.str() == "shr") {
      return b.create<mlir::arith::ShRUIOp>(loc,
                                            genDefTyExpr(call.arguments[1]),
                                            genDefTyExpr(call.arguments[0]));
    }

    //
    // TODO: The lowering of builtin function should be auto generated from
    // evmasm::InstructionInfo and the corresponding mlir ops
    //

    if (builtin->name.str() == "return") {
      b.create<mlir::sol::BuiltinRetOp>(loc, genDefTyExpr(call.arguments[0]),
                                        genDefTyExpr(call.arguments[1]));
      return {};
    }
    if (builtin->name.str() == "revert") {
      b.create<mlir::sol::RevertOp>(loc, genDefTyExpr(call.arguments[0]),
                                    genDefTyExpr(call.arguments[1]));
      return {};
    }
    if (builtin->name.str() == "mload") {
      return b.create<mlir::sol::MLoadOp>(loc, genDefTyExpr(call.arguments[0]));
    }
    if (builtin->name.str() == "mstore") {
      b.create<mlir::sol::MStoreOp>(loc, genDefTyExpr(call.arguments[0]),
                                    genDefTyExpr(call.arguments[1]));
      return {};
    }
    if (builtin->name.str() == "mcopy") {
      b.create<mlir::sol::MCopyOp>(loc, genDefTyExpr(call.arguments[0]),
                                   genDefTyExpr(call.arguments[1]),
                                   genDefTyExpr(call.arguments[2]));
      return {};
    }
    if (builtin->name.str() == "msize") {
      return b.create<mlir::sol::MSizeOp>(loc);
    }
    if (builtin->name.str() == "callvalue") {
      return b.create<mlir::sol::CallValOp>(loc);
    }
    if (builtin->name.str() == "calldataload") {
      return b.create<mlir::sol::CallDataLoadOp>(
          loc, genDefTyExpr(call.arguments[0]));
    }
    if (builtin->name.str() == "calldatasize") {
      return b.create<mlir::sol::CallDataSizeOp>(loc);
    }
    if (builtin->name.str() == "calldatacopy") {
      mlir::Value dst = genDefTyExpr(call.arguments[0]);
      mlir::Value offset = genDefTyExpr(call.arguments[1]);
      mlir::Value size = genDefTyExpr(call.arguments[2]);
      b.create<mlir::sol::CallDataCopyOp>(loc, dst, offset, size);
      return {};
    }
    if (builtin->name.str() == "sload") {
      return b.create<mlir::sol::SLoadOp>(loc, genDefTyExpr(call.arguments[0]));
    }
    if (builtin->name.str() == "sstore") {
      b.create<mlir::sol::SStoreOp>(loc, genDefTyExpr(call.arguments[0]),
                                    genDefTyExpr(call.arguments[1]));
      return {};
    }
    if (builtin->name.str() == "dataoffset") {
      auto *objectName = std::get_if<Literal>(&call.arguments[0]);
      assert(objectName);
      assert(objectName->kind == LiteralKind::String);
      auto objectOp =
          lookupSymbol<mlir::sol::ObjectOp>(objectName->value.str());
      assert(objectOp && "NYI: References to external object");
      return b.create<mlir::sol::DataOffsetOp>(
          loc, mlir::FlatSymbolRefAttr::get(objectOp));
    }
    if (builtin->name.str() == "datasize") {
      auto *objectName = std::get_if<Literal>(&call.arguments[0]);
      assert(objectName);
      assert(objectName->kind == LiteralKind::String);
      auto objectOp =
          lookupSymbol<mlir::sol::ObjectOp>(objectName->value.str());
      assert(objectOp && "NYI: References to external object");
      return b.create<mlir::sol::DataSizeOp>(
          loc, mlir::FlatSymbolRefAttr::get(objectOp));
    }
    if (builtin->name.str() == "codesize") {
      return b.create<mlir::sol::CodeSizeOp>(loc);
    }
    if (builtin->name.str() == "codecopy") {
      mlir::Value dst = genDefTyExpr(call.arguments[0]);
      mlir::Value offset = genDefTyExpr(call.arguments[1]);
      mlir::Value size = genDefTyExpr(call.arguments[2]);
      b.create<mlir::sol::CodeCopyOp>(loc, dst, offset, size);
      return {};
    }
    if (builtin->name.str() == "memoryguard") {
      auto *arg = std::get_if<Literal>(&call.arguments[0]);
      assert(arg);
      return b.create<mlir::sol::MemGuardOp>(loc, getIntAttr(arg->value));
    }
    if (builtin->name.str() == "keccak256") {
      return b.create<mlir::sol::Keccak256Op>(loc,
                                              genDefTyExpr(call.arguments[0]),
                                              genDefTyExpr(call.arguments[1]));
    }
    if (builtin->name.str() == "log0") {
      b.create<mlir::sol::LogOp>(loc, genDefTyExpr(call.arguments[0]),
                                 genDefTyExpr(call.arguments[1]),
                                 mlir::ValueRange{});
      return {};
    }
    if (builtin->name.str() == "log1") {
      b.create<mlir::sol::LogOp>(loc, genDefTyExpr(call.arguments[0]),
                                 genDefTyExpr(call.arguments[1]),
                                 genDefTyExpr(call.arguments[2]));
      return {};
    }
    if (builtin->name.str() == "log2") {
      b.create<mlir::sol::LogOp>(
          loc, genDefTyExpr(call.arguments[0]), genDefTyExpr(call.arguments[1]),
          mlir::ValueRange{genDefTyExpr(call.arguments[2]),
                           genDefTyExpr(call.arguments[3])});
      return {};
    }
    if (builtin->name.str() == "log3") {
      b.create<mlir::sol::LogOp>(
          loc, genDefTyExpr(call.arguments[0]), genDefTyExpr(call.arguments[1]),
          mlir::ValueRange{genDefTyExpr(call.arguments[2]),
                           genDefTyExpr(call.arguments[3]),
                           genDefTyExpr(call.arguments[4])});
      return {};
    }
    if (builtin->name.str() == "log4") {
      b.create<mlir::sol::LogOp>(
          loc, genDefTyExpr(call.arguments[0]), genDefTyExpr(call.arguments[1]),
          mlir::ValueRange{genDefTyExpr(call.arguments[2]),
                           genDefTyExpr(call.arguments[3]),
                           genDefTyExpr(call.arguments[4]),
                           genDefTyExpr(call.arguments[5])});
      return {};
    }

    solUnimplementedAssert(false, "NYI: builtin " + builtin->name.str());
  }

  mlir::sol::FuncOp callee =
      lookupSymbol<mlir::sol::FuncOp>(call.functionName.name.str());
  assert(callee);
  std::vector<mlir::Value> args;
  args.reserve(call.arguments.size());
  for (Expression const &arg : call.arguments) {
    args.push_back(genDefTyExpr(arg));
  }
  auto callOp = b.create<mlir::sol::CallOp>(loc, callee, args);
  assert(callOp.getNumResults() == 1 && "NYI: multivalue return");
  return callOp.getResult(0);
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

mlir::Value YulToMLIRPass::genDefTyExpr(Expression const &expr) {
  return genExpr(expr, getDefIntTy());
}

void YulToMLIRPass::operator()(ExpressionStatement const &expr) {
  genExpr(expr.expression);
}

void YulToMLIRPass::operator()(Assignment const &asgn) {
  assert(asgn.variableNames.size() == 1 && "NYI: Multivalued assignment");

  mlir::Value addr = getMemRef(asgn.variableNames[0].name);
  assert(addr);
  b.create<mlir::LLVM::StoreOp>(getLoc(asgn.debugData),
                                genDefTyExpr(*asgn.value), addr, getDefAlign());
}

void YulToMLIRPass::operator()(VariableDeclaration const &decl) {
  mlir::Location loc = getLoc(decl.debugData);
  mlirgen::BuilderExt bExt(b, loc);

  assert(decl.variables.size() == 1 && "NYI: Multivalued assignment");
  TypedName const &var = decl.variables[0];
  auto addr = b.create<mlir::LLVM::AllocaOp>(
      getLoc(var.debugData), mlir::LLVM::LLVMPointerType::get(getDefIntTy()),
      bExt.genI256Const(1), getDefAlign());
  setMemRef(var.name, addr);
  b.create<mlir::LLVM::StoreOp>(loc, genDefTyExpr(*decl.value), addr,
                                getDefAlign());
}

void YulToMLIRPass::operator()(If const &ifStmt) {
  mlir::Location loc = getLoc(ifStmt.debugData);

  // TODO: Should we expand here? Or is it beneficial to represent `if` with a
  // non-boolean condition in the IR?
  auto ifOp = b.create<mlir::scf::IfOp>(
      loc, convToBool(genExpr(*ifStmt.condition)), /*withElseRegion=*/false);
  mlir::OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  ASTWalker::operator()(ifStmt.body);
}

void YulToMLIRPass::operator()(Switch const &switchStmt) {
  mlir::Location loc = getLoc(switchStmt.debugData);

  // Create the mlir attribute for all the case values (excluding the default
  // case); Track the default case AST
  Case const *defCaseAST = nullptr;
  std::vector<llvm::APInt> caseVals;
  caseVals.reserve(switchStmt.cases.size());
  std::vector<Case const *> caseASTs;
  for (Case const &caseAST : switchStmt.cases) {
    // If non default block
    if (caseAST.value) {
      caseASTs.push_back(&caseAST);
      caseVals.push_back(
          getAPInt(caseAST.value->value, getDefIntTy().getWidth()));

    } else {
      // There should only be one default case
      assert(!defCaseAST);
      defCaseAST = &caseAST;
    }
  }
  assert(defCaseAST && "NYI: Switch block without a default case");
  auto caseValsAttr = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(static_cast<int64_t>(caseVals.size()),
                                  getDefIntTy()),
      caseVals);

  // Lower the switch argument and generate the switch op
  mlir::Value arg = genExpr(*switchStmt.expression);
  auto switchOp = b.create<mlir::scf::IntSwitchOp>(
      loc, /*resultTypes=*/std::nullopt, arg, caseValsAttr, caseVals.size());

  mlir::OpBuilder::InsertionGuard insertGuard(b);

  // Create blocks for all the case values and the default case; Then lower
  // their body
  auto lowerBody = [&](mlir::Region &region, Case const &caseAST) {
    mlir::Block *blk = b.createBlock(&region);
    b.setInsertionPointToStart(blk);
    b.create<mlir::scf::YieldOp>(loc);
    b.setInsertionPointToStart(blk);
    ASTWalker::operator()(caseAST.body);
  };
  lowerBody(switchOp.getDefaultRegion(), *defCaseAST);
  assert(switchOp.getCaseRegions().size() == caseASTs.size());
  for (auto [region, caseAST] : llvm::zip(switchOp.getCaseRegions(), caseASTs))
    lowerBody(region, *caseAST);
}

void YulToMLIRPass::operator()(FunctionDefinition const &fn) {
  mlir::Location loc = getLoc(fn.debugData);
  mlirgen::BuilderExt bExt(b, loc);

  // Lookup FuncOp (should be declared by the yul block lowering)
  auto funcOp = lookupSymbol<mlir::sol::FuncOp>(fn.name.str());
  assert(funcOp);

  // Add entry block and forward input args
  mlir::Block *entryBlk = b.createBlock(&funcOp.getRegion());
  std::vector<mlir::Location> inLocs;
  for (TypedName const &in : fn.parameters) {
    inLocs.push_back(getLoc(in.debugData));
  }
  assert(funcOp.getFunctionType().getNumInputs() == inLocs.size());
  entryBlk->addArguments(funcOp.getFunctionType().getInputs(), inLocs);

  mlir::OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(entryBlk);

  assert(fn.returnVariables.size() == 1 && "NYI: multivalued return");
  TypedName const &retVar = fn.returnVariables[0];
  setMemRef(retVar.name, b.create<mlir::LLVM::AllocaOp>(
                             getLoc(retVar.debugData),
                             mlir::LLVM::LLVMPointerType::get(getDefIntTy()),
                             bExt.genI256Const(1), getDefAlign()));

  // Lower the body
  ASTWalker::operator()(fn.body);

  b.create<mlir::sol::ReturnOp>(
      loc,
      mlir::ValueRange{b.create<mlir::LLVM::LoadOp>(
          getLoc(retVar.debugData), getMemRef(retVar.name), getDefAlign())});
}

void YulToMLIRPass::operator()(Block const &blk) {
  mlirgen::BuilderExt bExt(b);

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
  currObj = lookupSymbol<mlir::sol::ObjectOp>(obj.name.str());
  assert(currObj);

  b.setInsertionPointToStart(currObj.getBody());
  // TODO? Do we need a separate op for the `code` block?
  operator()(*obj.code);
}

void YulToMLIRPass::lowerTopLevelObj(Object const &obj) {
  // "Forward declare" ObjectOp for the top level object and its sub-objects so
  // that we can create symbol references to them (for builtins like dataoffset)
  //
  // TODO: Where is the source location info for Object? Do we need to track it?
  auto topLevelObj =
      b.create<mlir::sol::ObjectOp>(b.getUnknownLoc(), obj.name.str());
  {
    mlir::OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToEnd(topLevelObj.getBody());
    for (auto const &subNode : obj.subObjects) {
      if (auto *subObj = dynamic_cast<Object const *>(subNode.get()))
        b.create<mlir::sol::ObjectOp>(b.getUnknownLoc(), subObj->name.str());
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

bool solidity::mlirgen::runYulToMLIRPass(Object const &obj,
                                         CharStream const &stream,
                                         Dialect const &yulDialect,
                                         JobSpec const &job) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::sol::SolDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  solidity::mlirgen::YulToMLIRPass yulToMLIR(ctx, stream, yulDialect);
  yulToMLIR.lowerTopLevelObj(obj);

  mlir::ModuleOp mod = yulToMLIR.getModule();
  if (failed(mlir::verify(mod))) {
    mod.print(llvm::errs());
    mod.emitError("Module verification error");
    return false;
  }

  return doJob(job, ctx, mod);
}
