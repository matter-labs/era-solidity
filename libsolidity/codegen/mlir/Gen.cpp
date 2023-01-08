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

#include <libsolidity/codegen/mlir/Gen.h>

using namespace solidity::frontend;

bool MLIRGen::visit(BinaryOperation const& _binOp) { return true; }

bool MLIRGen::visit(Block const& _block) { return true; }

bool MLIRGen::visit(Assignment const& _assignment) { return true; }

void MLIRGen::run(Block const& _block) { _block.accept(*this); }

void MLIRGen::run(FunctionDefinition const& _function) { run(_function.body()); }

void MLIRGen::run(ContractDefinition const& _contract)
{
	for (auto* f: _contract.definedFunctions())
	{
		run(*f);
	}
}
