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
// MLIR codegen interface
//

#pragma once

#include "libsolidity/ast/ASTForward.h"
#include "libyul/ASTForward.h"
#include <optional>
#include <vector>

namespace solidity::langutil {
class CharStream;
};

namespace solidity::yul {
struct Dialect;
struct Object;
}; // namespace solidity::yul

namespace solidity::mlirgen {

enum class Action {
  /// Print the MLIR generated from the AST
  PrintInitStg,

  /// Print the MLIR after lowering dialect(s) in solc
  PrintPostSolcDialLowering,

  /// Print the LLVM-IR
  PrintLLVMIR,

  /// Print the assembly
  PrintAsm,
};

enum class Target {
  EraVM,
};

/// Registers required command line options in the MLIR framework
extern void registerMLIRCLOpts();

/// Parses command line options in `argv` for the MLIR framework
extern bool parseMLIROpts(std::vector<const char *> &_argv);

extern bool runSolidityToMLIRPass(
    std::vector<frontend::ContractDefinition const *> const &contracts,
    langutil::CharStream const &stream, Action);

extern bool runYulToMLIRPass(yul::Object const &, langutil::CharStream const &,
                             yul::Dialect const &, Action,
                             std::optional<Target> = std::nullopt);

} // namespace solidity::mlirgen
