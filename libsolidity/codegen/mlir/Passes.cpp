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
#include "lld-c/LLDAsLibraryC.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm-c/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <mutex>

// FIXME: Define an interface for targets!

void solidity::mlirgen::addConversionPasses(mlir::PassManager &passMgr,
                                            Target tgt) {
  passMgr.addPass(mlir::sol::createConvertSolToStandardPass(tgt));

  // FIXME: Adding individual conversion passes for each dialects causes
  // unrealized_conversion_cast's with index types.
  //
  // FIXME: `Target` should track triple, index bitwidth and data-layout.

  switch (tgt) {
  case Target::EVM:
    passMgr.addPass(mlir::sol::createConvertStandardToLLVMPass(
        /*triple=*/"evm-unknown-unknown",
        /*indexBitwidth=*/256,
        /*dataLayout=*/"E-p:256:256-i256:256:256-S256-a:256:256"));
    break;
  case Target::EraVM:
    passMgr.addPass(mlir::sol::createConvertStandardToLLVMPass(
        /*triple=*/"eravm-unknown-unknown",
        /*indexBitwidth=*/256,
        /*dataLayout=*/"E-p:256:256-i256:256:256-S32-a:256:256"));
    break;
  default:
    llvm_unreachable("");
  }
}

std::unique_ptr<llvm::TargetMachine>
solidity::mlirgen::createTargetMachine(Target tgt) {
  static std::once_flag initTargetOnceFlag;

  switch (tgt) {
  case Target::EVM: {
    // Initialize and register the target.
    std::call_once(initTargetOnceFlag, []() {
      LLVMInitializeEVMTarget();
      LLVMInitializeEVMTargetInfo();
      LLVMInitializeEVMTargetMC();
      LLVMInitializeEVMAsmPrinter();
    });

    // Lookup llvm::Target.
    std::string errMsg;
    llvm::Target const *llvmTgt =
        llvm::TargetRegistry::lookupTarget("evm", errMsg);
    if (!llvmTgt)
      llvm_unreachable(errMsg.c_str());

    // Create and return the llvm::TargetMachine.
    llvm::TargetOptions options;
    return std::unique_ptr<llvm::TargetMachine>(
        llvmTgt->createTargetMachine("evm", /*CPU=*/"", /*Features=*/"",
                                     options, /*Reloc::Model=*/std::nullopt));

    // TODO: Set code-model?
    // tgtMach->setCodeModel(?);
  }
  case Target::EraVM: {
    // Initialize and register the target.
    std::call_once(initTargetOnceFlag, []() {
      LLVMInitializeEraVMTarget();
      LLVMInitializeEraVMTargetInfo();
      LLVMInitializeEraVMTargetMC();
      LLVMInitializeEraVMAsmPrinter();
    });

    // Lookup llvm::Target.
    std::string errMsg;
    llvm::Target const *llvmTgt =
        llvm::TargetRegistry::lookupTarget("eravm", errMsg);
    if (!llvmTgt)
      llvm_unreachable(errMsg.c_str());

    // Create and return the llvm::TargetMachine.
    llvm::TargetOptions options;
    return std::unique_ptr<llvm::TargetMachine>(
        llvmTgt->createTargetMachine("eravm", /*CPU=*/"", /*Features=*/"",
                                     options, /*Reloc::Model=*/std::nullopt));

    // TODO: Set code-model?
    // tgtMach->setCodeModel(?);
    break;
  }
  case Target::Undefined:
    llvm_unreachable("Invalid target");
  }
}

void solidity::mlirgen::setTgtSpecificInfoInModule(
    Target tgt, llvm::Module &llvmMod, llvm::TargetMachine const &tgtMach) {
  std::string triple;
  switch (tgt) {
  case Target::EVM:
    triple = "evm-unknown-unknown";
    break;
  case Target::EraVM:
    triple = "eravm-unknown-unknown";
    break;
  case Target::Undefined:
    llvm_unreachable("Undefined target");
  }

  llvmMod.setTargetTriple(llvm::Triple::normalize(triple));
  llvmMod.setDataLayout(tgtMach.createDataLayout());
}

/// Sets the optimization level of the llvm::TargetMachine.
static void setTgtMachOpt(llvm::TargetMachine *tgtMach, char levelChar) {
  // Set level to 2 if size optimization is specified.
  if (levelChar == 's' || levelChar == 'z')
    levelChar = '2';
  auto level = llvm::CodeGenOpt::parseLevel(levelChar);
  assert(level);
  tgtMach->setOptLevel(*level);
}

/// Runs the llvm optimization pipeline.
static void runLLVMOptPipeline(llvm::Module *mod, char level,
                               llvm::TargetMachine *tgtMach) {
  char pipeline[12];
  std::sprintf(pipeline, "default<O%c>", level);
  LLVMErrorRef status =
      LLVMRunPasses(reinterpret_cast<LLVMModuleRef>(mod), pipeline,
                    reinterpret_cast<LLVMTargetMachineRef>(tgtMach),
                    LLVMCreatePassBuilderOptions());
  if (status != LLVMErrorSuccess) {
    llvm_unreachable(LLVMGetErrorMessage(status));
  }
}

static std::unique_ptr<llvm::Module>
genLLVMIR(mlir::ModuleOp mod, solidity::mlirgen::Target tgt, char optLevel,
          llvm::TargetMachine &tgtMach, llvm::LLVMContext &llvmCtx) {
  // Translate the llvm dialect to llvm-ir.
  mlir::registerLLVMDialectTranslation(*mod.getContext());
  mlir::registerBuiltinDialectTranslation(*mod.getContext());
  std::unique_ptr<llvm::Module> llvmMod =
      mlir::translateModuleToLLVMIR(mod, llvmCtx);

  // Set target specfic info in the llvm module.
  setTgtSpecificInfoInModule(tgt, *llvmMod, tgtMach);

  runLLVMOptPipeline(llvmMod.get(), optLevel, &tgtMach);

  return llvmMod;
}

bool solidity::mlirgen::doJob(JobSpec const &job, mlir::ModuleOp mod,
                              std::string &bytecodeInHex) {
  mlir::PassManager passMgr(mod.getContext());
  llvm::LLVMContext llvmCtx;

  switch (job.action) {
  case Action::PrintInitStg:
    mod.print(llvm::outs());
    break;
  case Action::PrintStandardMLIR:
    assert(job.tgt != Target::Undefined);
    passMgr.addPass(mlir::sol::createConvertSolToStandardPass(job.tgt));
    if (mlir::failed(passMgr.run(mod)))
      llvm_unreachable("Conversion to standard dialects failed");
    mod.print(llvm::outs());
    break;
  case Action::PrintLLVMIR: {
    assert(job.tgt != Target::Undefined);

    // Convert the module's ir to llvm dialect.
    addConversionPasses(passMgr, job.tgt);
    if (mlir::failed(passMgr.run(mod)))
      llvm_unreachable("Conversion to llvm dialect failed");

    std::unique_ptr<llvm::TargetMachine> tgtMach = createTargetMachine(job.tgt);
    setTgtMachOpt(tgtMach.get(), job.optLevel);

    switch (job.tgt) {
    case Target::EVM: {
      auto creationMod = mod;
      mlir::ModuleOp runtimeMod;
      for (mlir::Operation &op : *creationMod.getBody()) {
        if (mlir::isa<mlir::ModuleOp>(op)) {
          runtimeMod = mlir::cast<mlir::ModuleOp>(op);
          // Remove the runtime module from the creation module.
          runtimeMod->remove();
          break;
        }
      }

      // TODO: Run in parallel?
      std::unique_ptr<llvm::Module> creationLlvmMod =
          genLLVMIR(creationMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);
      std::unique_ptr<llvm::Module> runtimeLlvmMod =
          genLLVMIR(runtimeMod, job.tgt, job.optLevel, *tgtMach, llvmCtx);

      llvm::outs() << *creationLlvmMod;
      llvm::outs() << *runtimeLlvmMod;
      break;
    }
    case Target::EraVM: {
      std::unique_ptr<llvm::Module> llvmMod =
          genLLVMIR(mod, job.tgt, job.optLevel, *tgtMach, llvmCtx);

      llvm::outs() << *llvmMod;
      break;
    }
    default:
      llvm_unreachable("Invalid target");
    }

    break;
  }
  case Action::PrintAsm:
  case Action::GenObj: {
    assert(job.tgt == Target::EraVM && "NYI");

    // Convert the module's ir to llvm dialect.
    addConversionPasses(passMgr, job.tgt);
    if (mlir::failed(passMgr.run(mod)))
      llvm_unreachable("Conversion to llvm dialect failed");

    // Create TargetMachine and generate llvm-ir.
    std::unique_ptr<llvm::TargetMachine> tgtMach = createTargetMachine(job.tgt);
    setTgtMachOpt(tgtMach.get(), job.optLevel);
    std::unique_ptr<llvm::Module> llvmMod =
        genLLVMIR(mod, job.tgt, job.optLevel, *tgtMach, llvmCtx);

    // Set up and run the asm printer.
    llvm::legacy::PassManager llvmPassMgr;
    llvm::SmallString<0> outStreamData;
    llvm::raw_svector_ostream outStream(outStreamData);
    tgtMach->addPassesToEmitFile(llvmPassMgr, outStream,
                                 /*DwoOut=*/nullptr,
                                 job.action == Action::PrintAsm
                                     ? llvm::CodeGenFileType::CGFT_AssemblyFile
                                     : llvm::CodeGenFileType::CGFT_ObjectFile);
    llvmPassMgr.run(*llvmMod);

    if (job.action == Action::PrintAsm) {
      llvm::outs() << outStream.str();

      // Return the bytecode in hex.
    } else {
      LLVMMemoryBufferRef obj = LLVMCreateMemoryBufferWithMemoryRange(
          outStream.str().data(), outStream.str().size(), "Input",
          /*RequiresNullTerminator=*/0);
      LLVMMemoryBufferRef bytecode = nullptr;
      char *errMsg = nullptr;
      if (LLVMLinkEraVM(obj, &bytecode, /*linkerSymbolNames=*/nullptr,
                        /*linkerSymbolValues=*/nullptr, /*numLinkerSymbols=*/0,
                        &errMsg))
        llvm_unreachable(errMsg);

      bytecodeInHex = llvm::toHex(llvm::unwrap(bytecode)->getBuffer(),
                                  /*LowerCase=*/true);

      LLVMDisposeMemoryBuffer(bytecode);
    }

    break;
  }
  case Action::Undefined:
    llvm_unreachable("Undefined action");
  }

  return true;
}
