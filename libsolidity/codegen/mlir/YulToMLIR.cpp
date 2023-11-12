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
#include "libyul/AST.h"
#include "libyul/Dialect.h"
#include "libyul/Object.h"
#include "libyul/optimiser/ASTWalker.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

using namespace solidity::langutil;
using namespace solidity::yul;

namespace solidity::mlirgen {

class YulToMLIRPass : public ASTWalker {
  mlir::OpBuilder b;
  mlir::ModuleOp mod;
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
  mlir::Location loc(SourceLocation const &loc) {
    // FIXME: Track loc.end as well
    LineColumn lineCol = stream.translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream.name()),
                                     lineCol.line, lineCol.column);
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

  /// Lowers a block
  void operator()(Block const &blk) override;
};

mlir::Value YulToMLIRPass::genExpr(Literal const &lit) {
  mlir::Location lc = this->loc(lit.debugData->nativeLocation);

  // Do we need to represent constants as u256? Can we do that in
  // arith::ConstantOp?
  auto i256Ty = b.getIntegerType(256);
  return b.create<mlir::arith::ConstantOp>(
      lc, b.getIntegerAttr(i256Ty, llvm::APInt(256, lit.value.str(),
                                               /*radix=*/10)));
}

mlir::Value YulToMLIRPass::genExpr(Identifier const &id) {
  solUnimplementedAssert(false, "TODO: Lower identifier");
}

mlir::Value YulToMLIRPass::genExpr(FunctionCall const &call) {
  BuiltinFunction const *builtin = yulDialect.builtin(call.functionName.name);
  if (builtin) {

    // TODO: The lowering of builtin function should be auto generated from
    // evmasm::InstructionInfo and the corresponding mlir ops
    if (builtin->name.str() == "return") {
      b.create<mlir::sol::ReturnOp>(loc(call.debugData->nativeLocation),
                                    genExpr(call.arguments[0]),
                                    genExpr(call.arguments[1]));
      return {};

    } else if (builtin->name.str() == "mstore") {
      b.create<mlir::sol::MstoreOp>(loc(call.debugData->nativeLocation),
                                    genExpr(call.arguments[0]),
                                    genExpr(call.arguments[1]));
      return {};
    } else {
      solUnimplementedAssert(false, "TODO: Lower other builtin function call");
    }
  } else {
    solUnimplementedAssert(false, "TODO: Lower non builtin function call");
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

void YulToMLIRPass::operator()(Block const &blk) { ASTWalker::operator()(blk); }

void YulToMLIRPass::lowerObj(Object const &obj) {
  // TODO: Where is the source location info for Object? Do we need to track it?
  auto objOp = b.create<mlir::sol::ObjectOp>(b.getUnknownLoc(), obj.name.str());
  b.setInsertionPointToEnd(objOp.getBody());
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

static std::unique_ptr<llvm::TargetMachine>
createTargetMachine(solidity::mlirgen::Target tgt) {
  static std::once_flag initTargetOnceFlag;

  switch (tgt) {
  case solidity::mlirgen::Target::EraVM: {
    // Initialize and register the target
    std::call_once(initTargetOnceFlag, []() {
      LLVMInitializeEraVMTarget();
      LLVMInitializeEraVMTargetInfo();
      LLVMInitializeEraVMTargetMC();
      LLVMInitializeEraVMAsmPrinter();
    });

    // Lookup llvm::Target
    std::string errMsg;
    llvm::Target const *llvmTgt =
        llvm::TargetRegistry::lookupTarget("eravm", errMsg);
    if (!llvmTgt)
      llvm_unreachable(errMsg.c_str());

    // Create and return the llvm::TargetMachine
    llvm::TargetOptions Options;
    return std::unique_ptr<llvm::TargetMachine>(llvmTgt->createTargetMachine(
        "eravm", /*CPU=*/"", /*Features=*/"", Options, llvm::None, llvm::None));

    // TODO: Set code-model?
    // tgtMach->setCodeModel(?);
  }
  }
}

bool solidity::mlirgen::runYulToMLIRPass(Object const &obj,
                                         CharStream const &stream,
                                         Dialect const &yulDialect,
                                         Action action,
                                         std::optional<Target> tgt) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::sol::SolidityDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
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

  switch (action) {
  case Action::PrintInitStg:
    mod.print(llvm::outs());
    break;
  case Action::PrintPostSolcDialLowering:
    assert(tgt);
    llvm_unreachable(
        "TODO: Support dumping the IR after solc dialect lowering");
  case Action::PrintLLVMIR: {
    assert(tgt);
    addPassesForTarget(passMgr, *tgt);
    if (mlir::failed(passMgr.run(yulToMLIR.getModule())))
      return false;
    mlir::registerLLVMDialectTranslation(ctx);
    std::unique_ptr<llvm::Module> llvmMod =
        mlir::translateModuleToLLVMIR(mod, llvmCtx);
    llvm::outs() << *llvmMod;
    break;
  }
  case Action::PrintAsm: {
    assert(tgt);
    addPassesForTarget(passMgr, *tgt);
    if (mlir::failed(passMgr.run(yulToMLIR.getModule())))
      return false;
    mlir::registerLLVMDialectTranslation(ctx);
    std::unique_ptr<llvm::Module> llvmMod =
        mlir::translateModuleToLLVMIR(mod, llvmCtx);

    // Set triple in llvm::Module
    llvmMod->setTargetTriple(llvm::Triple::normalize("eravm-unknown-unknown"));

    // Create TargetMachine
    std::unique_ptr<llvm::TargetMachine> tgtMach = createTargetMachine(*tgt);
    assert(tgtMach);

    // Set data-layout in llvm::Module from TargetMachine
    llvmMod->setDataLayout(tgtMach->createDataLayout());

    // Set up and run the asm printer
    llvm::legacy::PassManager llvmPassMgr;
    tgtMach->addPassesToEmitFile(llvmPassMgr, llvm::outs(),
                                 /*DwoOut=*/nullptr,
                                 llvm::CodeGenFileType::CGFT_AssemblyFile);
    llvmPassMgr.run(*llvmMod);
  }
  }

  return true;
}
