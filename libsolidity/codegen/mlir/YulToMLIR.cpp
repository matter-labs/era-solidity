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
#include "liblangutil/Exceptions.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Solidity/SolidityOps.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libyul/AST.h"
#include "libyul/Dialect.h"
#include "libyul/Object.h"
#include "libyul/optimiser/ASTWalker.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
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

  /// Lowers a function
  void operator()(FunctionDefinition const &fn) override;

  /// Lowers a block
  void operator()(Block const &blk) override;
};

mlir::IntegerAttr YulToMLIRPass::getIntAttr(YulString num) {
  auto defTy = getDefIntTy();
  llvm::StringRef numStr = num.str();
  uint8_t radix = 10;
  if (numStr.startswith("0x")) {
    numStr = numStr.ltrim("0x");
    radix = 16;
  }
  return b.getIntegerAttr(defTy, llvm::APInt(defTy.getWidth(), numStr, radix));
}

mlir::Value YulToMLIRPass::getMemRef(YulString var) {
  auto it = memRefMap.find(var);
  if (it == memRefMap.end())
    return {};
  return it->second;
}

mlir::Value YulToMLIRPass::convToBool(mlir::Value val) {
  mlir::Location loc = val.getLoc();
  BuilderHelper h(b);

  auto ty = val.getType().cast<mlir::IntegerType>();
  if (ty.getWidth() == 1)
    return val;
  else if (ty == getDefIntTy())
    return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                         val, h.getConst(loc, 0));
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
  BuilderHelper h(b);
  BuiltinFunction const *builtin = yulDialect.builtin(call.functionName.name);
  mlir::Location loc = getLoc(call.debugData);
  if (builtin) {
    if (builtin->name.str() == "lt") {
      return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult,
                                           genDefTyExpr(call.arguments[0]),
                                           genDefTyExpr(call.arguments[1]));

    } else if (builtin->name.str() == "iszero") {
      return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                           genDefTyExpr(call.arguments[0]),
                                           h.getConst(loc, 0));

    } else if (builtin->name.str() == "shr") {
      return b.create<mlir::arith::ShRUIOp>(loc,
                                            genDefTyExpr(call.arguments[1]),
                                            genDefTyExpr(call.arguments[0]));

      // TODO: The lowering of builtin function should be auto generated from
      // evmasm::InstructionInfo and the corresponding mlir ops
    } else if (builtin->name.str() == "return") {
      b.create<mlir::sol::ReturnOp>(loc, genDefTyExpr(call.arguments[0]),
                                    genDefTyExpr(call.arguments[1]));
      return {};

    } else if (builtin->name.str() == "revert") {
      b.create<mlir::sol::RevertOp>(loc, genDefTyExpr(call.arguments[0]),
                                    genDefTyExpr(call.arguments[1]));
      return {};

    } else if (builtin->name.str() == "mload") {
      return b.create<mlir::sol::MLoadOp>(loc, genDefTyExpr(call.arguments[0]));

    } else if (builtin->name.str() == "mstore") {
      b.create<mlir::sol::MStoreOp>(loc, genDefTyExpr(call.arguments[0]),
                                    genDefTyExpr(call.arguments[1]));
      return {};

    } else if (builtin->name.str() == "msize") {
      return b.create<mlir::sol::MSizeOp>(loc);

    } else if (builtin->name.str() == "callvalue") {
      return b.create<mlir::sol::CallValOp>(loc);

    } else if (builtin->name.str() == "calldataload") {
      return b.create<mlir::sol::CallDataLoadOp>(
          loc, genDefTyExpr(call.arguments[0]));

    } else if (builtin->name.str() == "calldatasize") {
      return b.create<mlir::sol::CallDataSizeOp>(loc);

    } else if (builtin->name.str() == "dataoffset") {
      auto *objectName = std::get_if<Literal>(&call.arguments[0]);
      assert(objectName);
      assert(objectName->kind == LiteralKind::String);
      auto objectOp =
          lookupSymbol<mlir::sol::ObjectOp>(objectName->value.str());
      assert(objectOp && "NYI: References to external object");
      return b.create<mlir::sol::DataOffsetOp>(
          loc, mlir::FlatSymbolRefAttr::get(objectOp));

    } else if (builtin->name.str() == "datasize") {
      auto *objectName = std::get_if<Literal>(&call.arguments[0]);
      assert(objectName);
      assert(objectName->kind == LiteralKind::String);
      auto objectOp =
          lookupSymbol<mlir::sol::ObjectOp>(objectName->value.str());
      assert(objectOp && "NYI: References to external object");
      return b.create<mlir::sol::DataSizeOp>(
          loc, mlir::FlatSymbolRefAttr::get(objectOp));

    } else if (builtin->name.str() == "codecopy") {
      mlir::Value dst = genDefTyExpr(call.arguments[0]);
      mlir::Value offset = genDefTyExpr(call.arguments[1]);
      mlir::Value size = genDefTyExpr(call.arguments[2]);
      b.create<mlir::sol::CodeCopyOp>(loc, dst, offset, size);
      return {};

    } else if (builtin->name.str() == "memoryguard") {
      auto *arg = std::get_if<Literal>(&call.arguments[0]);
      assert(arg);
      return b.create<mlir::sol::MemGuardOp>(loc, getIntAttr(arg->value));

    } else {
      solUnimplementedAssert(false, "NYI: builtin " + builtin->name.str());
    }
  }

  mlir::func::FuncOp callee =
      lookupSymbol<mlir::func::FuncOp>(call.functionName.name.str());
  assert(callee);
  std::vector<mlir::Value> args;
  args.reserve(call.arguments.size());
  for (Expression const &arg : call.arguments) {
    args.push_back(genDefTyExpr(arg));
  }
  auto callOp = b.create<mlir::func::CallOp>(loc, callee, args);
  assert(callOp.getNumResults() == 1 && "NYI: multivalue return");
  return callOp.getResult(0);
}

mlir::Value YulToMLIRPass::genExpr(Expression const &expr,
                                   std::optional<mlir::IntegerType> resTy) {
  mlir::Value gen = std::visit(
      [&](auto &&resolvedExpr) { return this->genExpr(resolvedExpr); }, expr);

  if (resTy) {
    assert(gen);
    auto genTy = gen.getType().cast<mlir::IntegerType>();
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
  BuilderHelper h(b);
  mlir::Location loc = getLoc(decl.debugData);

  assert(decl.variables.size() == 1 && "NYI: Multivalued assignment");
  TypedName const &var = decl.variables[0];
  auto addr = b.create<mlir::LLVM::AllocaOp>(
      getLoc(var.debugData), mlir::LLVM::LLVMPointerType::get(getDefIntTy()),
      h.getConst(loc, 1), getDefAlign());
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

void YulToMLIRPass::operator()(FunctionDefinition const &fn) {
  BuilderHelper h(b);
  mlir::Location loc = getLoc(fn.debugData);

  // Lookup FuncOp (should be declared by the yul block lowering)
  auto funcOp = lookupSymbol<mlir::func::FuncOp>(fn.name.str());
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
                             h.getConst(loc, 1), getDefAlign()));

  // Lower the body
  ASTWalker::operator()(fn.body);

  b.create<mlir::func::ReturnOp>(
      loc,
      mlir::ValueRange{b.create<mlir::LLVM::LoadOp>(
          getLoc(retVar.debugData), getMemRef(retVar.name), getDefAlign())});
}

void YulToMLIRPass::operator()(Block const &blk) {
  BuilderHelper h(b);

  // "Forward declare" FuncOps (i.e. create them with an empty region) at this
  // block so that we can lower calls before lowering the functions. The
  // function lowering is expected to lookup the FuncOp without creating it.
  //
  // TODO: Stop relying on libyul's Disambiguator
  // We tried emitting a single block op for yul blocks with a symbol table
  // trait. We're able to define symbols with the same name in different blocks,
  // but ops like func::CallOp works with a FlatSymbolRefAttr which needs the
  // symbol definition to be in the same symbol table
  for (Statement const &stmt : blk.statements) {
    if (auto fn = std::get_if<FunctionDefinition>(&stmt)) {
      std::vector<mlir::Type> inTys(fn->parameters.size(), getDefIntTy()),
          outTys(fn->returnVariables.size(), getDefIntTy());
      mlir::FunctionType funcTy = b.getFunctionType(inTys, outTys);
      b.create<mlir::func::FuncOp>(getLoc(fn->debugData), fn->name.str(),
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
  ctx.getOrLoadDialect<mlir::sol::SolidityDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
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
