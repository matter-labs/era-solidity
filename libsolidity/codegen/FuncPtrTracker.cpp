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

#include <libsolidity/codegen/FuncPtrTracker.h>

#include <libsolidity/ast/CallGraph.h>

using namespace std;
using namespace solidity;
using namespace solidity::frontend;

void FuncPtrTracker::run()
{
	for (ContractDefinition const* base: m_contract.annotation().linearizedBaseContracts)
		base->accept(*this);

	// The call graph based visitor should track all the relevant indirect function references in the dependent
	// libraries that the above visitor missed. Note that the call graph visitor alone will miss indirect function
	// references in state variables.
	ContractDefinitionAnnotation const& contrAnnotation = m_contract.annotation();
	solAssert(contrAnnotation.creationCallGraph.set() && contrAnnotation.deployedCallGraph.set());
	std::set<CallableDeclaration const*> reachableFuncs = (*contrAnnotation.creationCallGraph)->getFuncs();
	reachableFuncs += (*contrAnnotation.deployedCallGraph)->getFuncs();
	for (auto* func: reachableFuncs)
		func->accept(*this);
}

void FuncPtrTracker::trackIfIndirect(Expression const& _expression, FunctionDefinition const& _referencedFunction)
{
	if (_expression.annotation().calledDirectly)
		return;
	m_contract.annotation().intFuncPtrRefs.insert(&_referencedFunction);
}

void FuncPtrTracker::endVisit(Identifier const& _identifier)
{
	Declaration const* declaration = _identifier.annotation().referencedDeclaration;
	FunctionDefinition const* functionDef = dynamic_cast<FunctionDefinition const*>(declaration);
	if (!functionDef)
		return;

	solAssert(*_identifier.annotation().requiredLookup == VirtualLookup::Virtual);
	FunctionDefinition const& resolvedFunctionDef = functionDef->resolveVirtual(m_contract);

	solAssert(resolvedFunctionDef.functionType(true));
	solAssert(resolvedFunctionDef.functionType(true)->kind() == FunctionType::Kind::Internal);
	trackIfIndirect(_identifier, resolvedFunctionDef);
}

void FuncPtrTracker::endVisit(MemberAccess const& _memberAccess)
{
	auto memberFunctionType = dynamic_cast<FunctionType const*>(_memberAccess.annotation().type);

	if (memberFunctionType && memberFunctionType->hasBoundFirstArgument())
	{
		solAssert(*_memberAccess.annotation().requiredLookup == VirtualLookup::Static);
		if (memberFunctionType->kind() == FunctionType::Kind::Internal)
			trackIfIndirect(_memberAccess, dynamic_cast<FunctionDefinition const&>(memberFunctionType->declaration()));
	}

	Type::Category objectCategory = _memberAccess.expression().annotation().type->category();
	switch (objectCategory)
	{
	case Type::Category::TypeType:
	{
		Type const& actualType
			= *dynamic_cast<TypeType const&>(*_memberAccess.expression().annotation().type).actualType();

		if (actualType.category() == Type::Category::Contract)
		{
			ContractType const& contractType = dynamic_cast<ContractType const&>(actualType);
			if (contractType.isSuper())
			{
				solAssert(!!_memberAccess.annotation().referencedDeclaration, "Referenced declaration not resolved.");
				ContractDefinition const* super = contractType.contractDefinition().superContract(m_contract);
				solAssert(super, "Super contract not available.");
				FunctionDefinition const& resolvedFunctionDef
					= dynamic_cast<FunctionDefinition const&>(*_memberAccess.annotation().referencedDeclaration)
						  .resolveVirtual(m_contract, super);

				solAssert(resolvedFunctionDef.functionType(true));
				solAssert(resolvedFunctionDef.functionType(true)->kind() == FunctionType::Kind::Internal);
				trackIfIndirect(_memberAccess, resolvedFunctionDef);
			}
			else if (memberFunctionType && memberFunctionType->kind() == FunctionType::Kind::Internal)
			{
				if (auto const* function
					= dynamic_cast<FunctionDefinition const*>(_memberAccess.annotation().referencedDeclaration))
					trackIfIndirect(_memberAccess, *function);
			}
		}
		break;
	}
	case Type::Category::Module:
	{
		if (auto const* function
			= dynamic_cast<FunctionDefinition const*>(_memberAccess.annotation().referencedDeclaration))
		{
			auto funType = dynamic_cast<FunctionType const*>(_memberAccess.annotation().type);
			solAssert(function && function->isFree());
			solAssert(function->functionType(true));
			solAssert(function->functionType(true)->kind() == FunctionType::Kind::Internal);
			solAssert(funType->kind() == FunctionType::Kind::Internal);
			solAssert(*_memberAccess.annotation().requiredLookup == VirtualLookup::Static);

			trackIfIndirect(_memberAccess, *function);
		}
		break;
	}
	default:
		break;
	}
}
