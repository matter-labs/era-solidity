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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

namespace mlir {
class Pass;

namespace sol {
std::unique_ptr<Pass> createSolidityDialectLoweringPassForEraVM();
}

} // namespace mlir

namespace solidity::mlirgen {

/// Adds `tgt` specfic MLIR passes (including the lowering passes)
void addMLIRPassesForTgt(mlir::PassManager &, Target tgt);

/// Creates and return the llvm::TargetMachine for `tgt`
std::unique_ptr<llvm::TargetMachine> createTargetMachine(Target tgt);

/// Sets target specific info in `llvmMod` from `tgt`
void setTgtSpecificInfoInModule(Target tgt, llvm::Module &llvmMod,
                                llvm::TargetMachine const &tgtMach);

} // namespace solidity::mlirgen
