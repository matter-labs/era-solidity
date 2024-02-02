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
// Main file of the sol-opt tool
//

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Solidity/SolidityOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include <memory>

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<BuiltinDialect, sol::SolidityDialect, func::FuncDialect,
                  arith::ArithmeticDialect, LLVM::LLVMDialect>();

  registerPass([](void) -> std::unique_ptr<Pass> {
    return sol::createSolConvertPassForEraVM();
  });
  registerPass([](void) -> std::unique_ptr<Pass> {
    return sol::createSolidityDialectLoweringPassForEraVM();
  });

  // TODO: registerTransformsPasses()

  return failed(
      MlirOptMain(argc, argv, "Sol dialect conversion tool\n", registry));
}
