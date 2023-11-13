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
#include "mlir/Pass/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"
#include <mutex>

void solidity::mlirgen::addMLIRPassesForTgt(mlir::PassManager &passMgr,
                                            Target tgt) {
  switch (tgt) {
  case Target::EraVM:
    passMgr.addPass(mlir::sol::createSolidityDialectLoweringPassForEraVM());
    break;
  case Target::Undefined:
    llvm_unreachable("Undefined target");
  }
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
    llvm::TargetOptions Options;
    return std::unique_ptr<llvm::TargetMachine>(
        llvmTgt->createTargetMachine("eravm", /*CPU=*/"", /*Features=*/"",
                                     Options, /*Reloc::Model=*/llvm::None));

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
