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

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"
#include <mutex>

void solidity::mlirgen::addConversionPasses(mlir::PassManager &passMgr,
                                            Target tgt) {
  passMgr.addPass(mlir::sol::createConvertSolToStandardPass(tgt));
  passMgr.addPass(mlir::createConvertFuncToLLVMPass());
  passMgr.addPass(mlir::createConvertSCFToCFPass());
  passMgr.addPass(mlir::createConvertControlFlowToLLVMPass());
  passMgr.addPass(mlir::createArithToLLVMConversionPass());
}

std::unique_ptr<llvm::TargetMachine>
solidity::mlirgen::createTargetMachine(Target tgt) {
  static std::once_flag initTargetOnceFlag;

  switch (tgt) {
  case Target::EraVM: {
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
    llvm::TargetOptions options;
    return std::unique_ptr<llvm::TargetMachine>(
        llvmTgt->createTargetMachine("eravm", /*CPU=*/"", /*Features=*/"",
                                     options, /*Reloc::Model=*/std::nullopt));

    // TODO: Set code-model?
    // tgtMach->setCodeModel(?);
    break;
  }
  case Target::Undefined:
    llvm_unreachable("Undefined target");
  }
}

void solidity::mlirgen::setTgtSpecificInfoInModule(
    Target tgt, llvm::Module &llvmMod, llvm::TargetMachine const &tgtMach) {
  std::string triple;
  switch (tgt) {
  case Target::EraVM:
    triple = "eravm-unknown-unknown";
    break;
  case Target::Undefined:
    llvm_unreachable("Undefined target");
  }

  llvmMod.setTargetTriple(llvm::Triple::normalize("eravm-unknown-unknown"));
  llvmMod.setDataLayout(tgtMach.createDataLayout());
}

bool solidity::mlirgen::doJob(JobSpec const &job, mlir::MLIRContext &ctx,
                              mlir::ModuleOp mod) {
  mlir::PassManager passMgr(&ctx);
  llvm::LLVMContext llvmCtx;

  switch (job.action) {
  case Action::PrintInitStg:
    mod.print(llvm::outs());
    break;
  case Action::PrintStandardMLIR:
    assert(job.tgt != Target::Undefined);
    passMgr.addPass(mlir::sol::createConvertSolToStandardPass(job.tgt));
    if (mlir::failed(passMgr.run(mod)))
      return false;
    mod.print(llvm::outs());
    break;
  case Action::PrintLLVMIR: {
    assert(job.tgt != Target::Undefined);
    addConversionPasses(passMgr, job.tgt);
    if (mlir::failed(passMgr.run(mod)))
      return false;
    mlir::registerLLVMDialectTranslation(ctx);
    mlir::registerBuiltinDialectTranslation(ctx);
    std::unique_ptr<llvm::Module> llvmMod =
        mlir::translateModuleToLLVMIR(mod, llvmCtx);
    assert(llvmMod);
    llvm::outs() << *llvmMod;
    break;
  }
  case Action::PrintAsm:
  case Action::GenObj: {
    assert(job.tgt != Target::Undefined);
    addConversionPasses(passMgr, job.tgt);
    if (mlir::failed(passMgr.run(mod)))
      return false;
    mlir::registerLLVMDialectTranslation(ctx);
    mlir::registerBuiltinDialectTranslation(ctx);
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
                                 job.action == Action::PrintAsm
                                     ? llvm::CodeGenFileType::CGFT_AssemblyFile
                                     : llvm::CodeGenFileType::CGFT_ObjectFile);
    llvmPassMgr.run(*llvmMod);
    break;
  }
  case Action::Undefined:
    llvm_unreachable("Undefined action");
  }

  return true;
}
