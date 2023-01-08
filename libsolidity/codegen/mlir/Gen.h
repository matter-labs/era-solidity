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

#include <libsolidity/ast/ASTVisitor.h>

namespace solidity::frontend
{

class MLIRGen: public ASTConstVisitor
{
public:
	void run(ContractDefinition const& _contract);

private:
	void run(FunctionDefinition const& _function);
	void run(Block const& _block);

	bool visit(Block const& _block) override;
	bool visit(Assignment const& _assignment) override;
	bool visit(BinaryOperation const& _binOp) override;
};

}
