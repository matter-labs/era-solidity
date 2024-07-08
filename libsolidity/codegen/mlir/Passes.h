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
// MLIR Passes
//

#pragma once

#include "libsolidity/codegen/mlir/Interface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

namespace mlir {
class Pass;

namespace sol {

/// Creates a pass to lower sol dialect to standard dialects.
std::unique_ptr<Pass> createConvertSolToStandardPass();
std::unique_ptr<Pass>
createConvertSolToStandardPass(solidity::mlirgen::Target tgt);

// TODO: Is mlir::sol the right namespace for this?
/// Creates a pass to convert standard dialects to llvm dialect.
std::unique_ptr<Pass> createConvertStandardToLLVMPass(StringRef triple,
                                                      unsigned indexBitwidth,
                                                      StringRef dataLayout);

} // namespace sol

} // namespace mlir

namespace solidity::mlirgen {

/// Adds dialect conversion passes for the target.
void addConversionPasses(mlir::PassManager &, Target tgt);

/// Creates and return the llvm::TargetMachine for `tgt`
std::unique_ptr<llvm::TargetMachine> createTargetMachine(Target tgt);

/// Sets target specific info in `llvmMod` from `tgt`
void setTgtSpecificInfoInModule(Target tgt, llvm::Module &llvmMod,
                                llvm::TargetMachine const &tgtMach);

/// Performs the JobSpec
bool doJob(JobSpec const &, mlir::MLIRContext &, mlir::ModuleOp);

} // namespace solidity::mlirgen
