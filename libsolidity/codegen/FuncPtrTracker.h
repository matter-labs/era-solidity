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
 * Tracks function pointer references
 */

#pragma once

#include <libsolidity/ast/ASTVisitor.h>
#include <libsolidity/codegen/CompilerContext.h>

namespace solidity::frontend
{

/**
 * This class is used to add all the function pointer references in the contract
 * and its ancestor contracts to the CompilerContext. The visitor is copied from
 * the yul codegen pipeline's usage of
 * IRGeneratorForStatements::assignInternalFunctionIDIfNotCalledDirectly()
 */
class FuncPtrTracker: private ASTConstVisitor
{
public:
	FuncPtrTracker(ContractDefinition const& _contract, CompilerContext& _context)
		: m_contract(_contract), m_currContract(&_contract), m_context(_context)
	{
	}

	void run(ContractDefinition const& _contract)
	{
		for (auto& baseSpec: _contract.baseContracts())
		{
			ContractDefinition const* baseContr
				= dynamic_cast<ContractDefinition const*>(baseSpec->name().annotation().referencedDeclaration);
			// TODO: Will this fail?
			solAssert(baseContr, "");
			run(*baseContr);
		}

		m_currContract = &_contract;
		_contract.accept(*this);
	}

	void run() { run(m_contract); }

private:
	ContractDefinition const& m_contract;
	ContractDefinition const* m_currContract;
	CompilerContext& m_context;

	void endVisit(Identifier const& _identifier);
	void endVisit(MemberAccess const& _memberAccess);
};

}
