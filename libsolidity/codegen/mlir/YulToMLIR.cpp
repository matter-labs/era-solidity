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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>

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
  /// Returns the mlir location for the solidity source location `loc`
  mlir::Location getLoc(SourceLocation const &loc) {
    // FIXME: Track loc.end as well
    LineColumn lineCol = stream.translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream.name()),
                                     lineCol.line, lineCol.column);
  }

  /// Returns the symbol of type `T` in the current scope
  template <typename T>
  T lookupSymbol(llvm::StringRef name) {
    // FIXME: We should lookup in the current block and its ancestors
    return currObj.lookupSymbol<T>(name);
  }

  /// Returns the mlir expression for the literal `lit`
  mlir::Value genExpr(Literal const &lit);

  /// Returns the mlir expression for the identifier `id`
  mlir::Value genExpr(Identifier const &id);

  /// Returns the mlir expression for the function call `call`
  mlir::Value genExpr(FunctionCall const &call);

  /// Returns the mlir expression for the expression `expr`
  mlir::Value genExpr(Expression const &expr);

  /// Lowers an expression statement
  void operator()(ExpressionStatement const &expr) override;

  /// Lowers a function
  void operator()(FunctionDefinition const &fn) override;

  /// Lowers a block
  void operator()(Block const &blk) override;
};

mlir::Value YulToMLIRPass::genExpr(Literal const &lit) {
  mlir::Location loc = this->getLoc(lit.debugData->nativeLocation);

  // Do we need to represent constants as u256? Can we do that in
  // arith::ConstantOp?
  auto i256Ty = b.getIntegerType(256);
  return b.create<mlir::arith::ConstantOp>(
      loc, b.getIntegerAttr(i256Ty, llvm::APInt(256, lit.value.str(),
                                                /*radix=*/10)));
}

mlir::Value YulToMLIRPass::genExpr(Identifier const &id) {
  solUnimplementedAssert(false, "TODO: Lower identifier");
}

mlir::Value YulToMLIRPass::genExpr(FunctionCall const &call) {
  BuiltinFunction const *builtin = yulDialect.builtin(call.functionName.name);
  mlir::Location loc = getLoc(call.debugData->nativeLocation);
  if (builtin) {

    // TODO: The lowering of builtin function should be auto generated from
    // evmasm::InstructionInfo and the corresponding mlir ops
    if (builtin->name.str() == "return") {
      b.create<mlir::sol::ReturnOp>(loc, genExpr(call.arguments[0]),
                                    genExpr(call.arguments[1]));
      return {};

    } else if (builtin->name.str() == "mstore") {
      b.create<mlir::sol::MstoreOp>(loc, genExpr(call.arguments[0]),
                                    genExpr(call.arguments[1]));
      return {};
    } else {
      solUnimplementedAssert(false, "TODO: Lower other builtin function call");
    }
  } else {
    mlir::func::FuncOp callee =
        lookupSymbol<mlir::func::FuncOp>(call.functionName.name.str());
    assert(callee);
    std::vector<mlir::Value> args;
    args.reserve(call.arguments.size());
    for (Expression const &arg : call.arguments) {
      args.push_back(genExpr(arg));
    }
    auto callOp = b.create<mlir::func::CallOp>(loc, callee, args);
    solUnimplementedAssert(callOp.getNumResults() == 1,
                           "TODO: Support multivalue return");
    return callOp.getResult(0);
  }

  solAssert(false);
}

mlir::Value YulToMLIRPass::genExpr(Expression const &expr) {
  return std::visit(
      [&](auto &&resolvedExpr) { return this->genExpr(resolvedExpr); }, expr);
}

void YulToMLIRPass::operator()(ExpressionStatement const &expr) {
  genExpr(expr.expression);
}

void YulToMLIRPass::operator()(FunctionDefinition const &fn) {
  BuilderHelper h(b);
  mlir::Location loc = getLoc(fn.debugData->nativeLocation);

  // Lookup FuncOp (should be declared by the yul block lowering)
  auto funcOp = lookupSymbol<mlir::func::FuncOp>(fn.name.str());
  assert(funcOp);

  // Add entry block and forward input args
  mlir::Block *entryBlk = b.createBlock(&funcOp.getRegion());
  std::vector<mlir::Location> inLocs;
  for (TypedName const &in : fn.parameters) {
    inLocs.push_back(getLoc(in.debugData->nativeLocation));
  }
  assert(funcOp.getFunctionType().getNumInputs() == inLocs.size());
  entryBlk->addArguments(funcOp.getFunctionType().getInputs(), inLocs);

  // Lower the body
  mlir::OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(entryBlk);
  ASTWalker::operator()(fn.body);

  // FIXME: Implement return
  b.create<mlir::func::ReturnOp>(loc, h.getConst(loc, 0));
}

void YulToMLIRPass::operator()(Block const &blk) {
  BuilderHelper h(b);
  mlir::IntegerType i256Ty = b.getIntegerType(256);

  // "Declare" FuncOps (i.e. create them with an empty region) at this block so
  // that we can lower calls before lowering the functions. The function
  // lowering is expected to lookup the FuncOp without creating it.
  //
  // TODO: Stop relying on libyul's Disambiguator
  // We tried emitting a single block op for yul blocks with a symbol table
  // trait. We're able to define symbols with the same name in different blocks,
  // but ops like func::CallOp works with a FlatSymbolRefAttr which needs the
  // symbol definition to be in the same symbol table
  for (Statement const &stmt : blk.statements) {
    if (auto fn = std::get_if<FunctionDefinition>(&stmt)) {
      std::vector<mlir::Type> inTys(fn->parameters.size(), i256Ty),
          outTys(fn->returnVariables.size(), i256Ty);
      mlir::FunctionType funcTy = b.getFunctionType(inTys, outTys);
      b.create<mlir::func::FuncOp>(getLoc(fn->debugData->nativeLocation),
                                   fn->name.str(), funcTy);
    }
  }

  ASTWalker::operator()(blk);
}

void YulToMLIRPass::lowerObj(Object const &obj) {
  // TODO: Where is the source location info for Object? Do we need to track it?
  currObj = b.create<mlir::sol::ObjectOp>(b.getUnknownLoc(), obj.name.str());
  b.setInsertionPointToEnd(currObj.getBody());
  // TODO? Do we need a separate op for the `code` block?
  operator()(*obj.code);
}

void YulToMLIRPass::lowerTopLevelObj(Object const &obj) {
  lowerObj(obj);

  // TODO: Does it make sense to nest subobjects in the top level ObjectOp's
  // body?
  for (auto const &subNode : obj.subObjects) {
    if (auto *subObj = dynamic_cast<Object const *>(subNode.get())) {
      lowerObj(*subObj);
    } else {
      solUnimplementedAssert(false, "TODO: Metadata translation");
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
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  solidity::mlirgen::YulToMLIRPass yulToMLIR(ctx, stream, yulDialect);
  yulToMLIR.lowerTopLevelObj(obj);

  mlir::ModuleOp mod = yulToMLIR.getModule();
  if (failed(mlir::verify(mod))) {
    mod.print(llvm::errs());
    mod.emitError("Module verification error");
    return false;
  }

  mlir::PassManager passMgr(&ctx);
  llvm::LLVMContext llvmCtx;

  switch (job.action) {
  case Action::PrintInitStg:
    mod.print(llvm::outs());
    break;
  case Action::PrintPostSolcDialLowering:
    assert(job.tgt != Target::Undefined);
    llvm_unreachable(
        "TODO: Support dumping the IR after solc dialect lowering");
  case Action::PrintLLVMIR: {
    assert(job.tgt != Target::Undefined);
    addMLIRPassesForTgt(passMgr, job.tgt);
    if (mlir::failed(passMgr.run(yulToMLIR.getModule())))
      return false;
    mlir::registerLLVMDialectTranslation(ctx);
    std::unique_ptr<llvm::Module> llvmMod =
        mlir::translateModuleToLLVMIR(mod, llvmCtx);
    assert(llvmMod);
    llvm::outs() << *llvmMod;
    break;
  }
  case Action::PrintAsm: {
    assert(job.tgt != Target::Undefined);
    addMLIRPassesForTgt(passMgr, job.tgt);
    if (mlir::failed(passMgr.run(yulToMLIR.getModule())))
      return false;
    mlir::registerLLVMDialectTranslation(ctx);
    std::unique_ptr<llvm::Module> llvmMod =
        mlir::translateModuleToLLVMIR(mod, llvmCtx);
    assert(llvmMod);

    // Create TargetMachine from `tgt`
    std::unique_ptr<llvm::TargetMachine> tgtMach = createTargetMachine(job.tgt);
    assert(tgtMach);

    // Set target specfic info in `llvmMod`
    setTgtSpecificInfoInModule(job.tgt, *llvmMod, *tgtMach);

    // Set up and run the asm printer
    llvm::legacy::PassManager llvmPassMgr;
    tgtMach->addPassesToEmitFile(llvmPassMgr, llvm::outs(),
                                 /*DwoOut=*/nullptr,
                                 llvm::CodeGenFileType::CGFT_AssemblyFile);
    llvmPassMgr.run(*llvmMod);
    break;
  }
  case Action::Undefined:
    llvm_unreachable("Undefined action");
  }

  return true;
}
