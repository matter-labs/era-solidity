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

using namespace std;
using namespace solidity;
using namespace solidity::frontend;

void LibraryDependencyTracker::run()
{
	std::vector<ContractDefinition const*> worklist;
	std::set<ContractDefinition const*> visited;

	for (ContractDefinition const* base: m_contract.annotation().linearizedBaseContracts)
		worklist.push_back(base);

	while (!worklist.empty())
	{
		ContractDefinition const* current = worklist.back();
		worklist.pop_back();

		if (visited.contains(current))
			continue;

		visited.insert(current);
		trackDeps(*current);

		for (ContractDefinition const* lib: libraryDependencies)
		{
			if (!visited.contains(lib))
				worklist.push_back(lib);
		}
	}
}

void LibraryDependencyTracker::trackDeps(ContractDefinition const& _contract)
{
	_contract.accept(*this);

	for (UsingForDirective const* usingFor: _contract.usingForDirectives())
	{
		for (ASTPointer<IdentifierPath> idPath: usingFor->functionsOrLibrary())
		{
			Declaration const* decl = idPath->annotation().referencedDeclaration;
			solAssert(decl);
			if (auto* func = dynamic_cast<FunctionDefinition const*>(decl))
			{
				solAssert(func->scope());
				auto* lib = dynamic_cast<ContractDefinition const*>(func->scope());
				if (lib && lib->isLibrary())
					libraryDependencies.insert(lib);
			}
			else if (auto lib = dynamic_cast<ContractDefinition const*>(decl))
				libraryDependencies.insert(lib);
			else
				solAssert(false);
		}
	}
}

void LibraryDependencyTracker::endVisit(Identifier const& _identifier)
{
	Declaration const* declaration = _identifier.annotation().referencedDeclaration;
	auto const* contr = dynamic_cast<ContractDefinition const*>(declaration);
	if (contr && contr->isLibrary())
		libraryDependencies.insert(contr);
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
	if (_identifier.annotation().calledDirectly)
		return;
	m_contract.annotation().intFuncPtrRefs.insert(&resolvedFunctionDef);
}

void FuncPtrTracker::endVisit(MemberAccess const& _memberAccess)
{
	auto memberFunctionType = dynamic_cast<FunctionType const*>(_memberAccess.annotation().type);

	if (memberFunctionType && memberFunctionType->hasBoundFirstArgument())
	{
		solAssert(*_memberAccess.annotation().requiredLookup == VirtualLookup::Static);
		if (memberFunctionType->kind() == FunctionType::Kind::Internal)
			m_contract.annotation().intFuncPtrRefs.insert(
				&dynamic_cast<FunctionDefinition const&>(memberFunctionType->declaration()));
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
				m_contract.annotation().intFuncPtrRefs.insert(&resolvedFunctionDef);
			}
			else if (memberFunctionType && memberFunctionType->kind() == FunctionType::Kind::Internal)
			{
				if (auto const* function
					= dynamic_cast<FunctionDefinition const*>(_memberAccess.annotation().referencedDeclaration))
					m_contract.annotation().intFuncPtrRefs.insert(function);
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

			m_contract.annotation().intFuncPtrRefs.insert(function);
		}
		break;
	}
	default:
		break;
	}
}
