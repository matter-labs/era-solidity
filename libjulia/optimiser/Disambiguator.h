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
/**
 * Optimiser component that makes all identifiers unique.
 */

#pragma once

#include <libjulia/ASTDataForward.h>

#include <libjulia/optimiser/ASTCopier.h>
#include <libjulia/optimiser/NameDispenser.h>

#include <libsolidity/inlineasm/AsmAnalysisInfo.h>

#include <boost/variant.hpp>
#include <boost/optional.hpp>

#include <set>

namespace dev
{
namespace julia
{

/**
 * Creates a copy of a iulia AST replacing all identifiers by unique names.
 */
class Disambiguator: public ASTCopier
{
public:
	explicit Disambiguator(
		solidity::assembly::AsmAnalysisInfo const& _analysisInfo,
		std::set<std::string> const& _externallyUsedIdentifiers = {}
	):
		m_info(_analysisInfo), m_externallyUsedIdentifiers(_externallyUsedIdentifiers)
	{
		m_nameDispenser.m_usedNames = m_externallyUsedIdentifiers;
	}

protected:
	virtual void enterScope(Block const& _block) override;
	virtual void leaveScope(Block const& _block) override;
	virtual void enterFunction(FunctionDefinition const& _function) override;
	virtual void leaveFunction(FunctionDefinition const& _function) override;
	virtual std::string translateIdentifier(std::string const& _name) override;

	void enterScopeInternal(solidity::assembly::Scope& _scope);
	void leaveScopeInternal(solidity::assembly::Scope& _scope);

	solidity::assembly::AsmAnalysisInfo const& m_info;
	std::set<std::string> const& m_externallyUsedIdentifiers;

	std::vector<solidity::assembly::Scope*> m_scopes;
	std::map<void const*, std::string> m_translations;
	NameDispenser m_nameDispenser;
};

}
}
