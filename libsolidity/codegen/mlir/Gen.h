/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0
/**
 * MLIR generator
 */

#pragma once

#include <liblangutil/Exceptions.h>
#include <string>
#include <vector>

namespace solidity::langutil
{
class CharStream;
}

namespace solidity::frontend
{
class ContractDefinition;

/// Registers required command line options in the MLIR framework
extern void registerMLIRCLOpts();

/// Parses command line options in `argv` for the MLIR framework
extern bool parseMLIROpts(std::vector<const char*>& _argv);

/// MLIRGen stages
enum class MLIRGenStage
{
	Init,
	LLVMIR
};

extern bool runMLIRGen(
	std::vector<ContractDefinition const*> const& _contracts,
	langutil::CharStream const& _stream,
	MLIRGenStage stage = MLIRGenStage::Init);
}
