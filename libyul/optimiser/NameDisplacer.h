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
 * Optimiser component that renames identifiers to free up certain names.
 */

#pragma once

#include <libyul/optimiser/ASTWalker.h>
#include <libyul/optimiser/NameDispenser.h>

#include <set>
#include <map>

namespace solidity::yul
{
struct Dialect;

/**
 * Optimiser component that renames identifiers to free up certain names.
 *
 * Only replaces names that have been defined inside the code. If the code uses
 * names to be freed but does not define them, they remain unchanged.
 *
 * Prerequisites: Disambiguator
 */
class NameDisplacer: public ASTModifier
{
public:
	explicit NameDisplacer(
		NameDispenser& _dispenser,
		std::set<YulName> const& _namesToFree
	):
		m_nameDispenser(_dispenser),
		m_namesToFree(_namesToFree)
	{
		for (YulName n: _namesToFree)
			m_nameDispenser.markUsed(n);
	}

	using ASTModifier::operator();
	void operator()(Identifier& _identifier) override;
	void operator()(VariableDeclaration& _varDecl) override;
	void operator()(FunctionDefinition& _function) override;
	void operator()(FunctionCall& _funCall) override;
	void operator()(Block& _block) override;

	std::map<YulName, YulName> const& translations() const { return m_translations; }

protected:
	/// Check if the newly introduced identifier @a _name has to be replaced.
	void checkAndReplaceNew(YulName& _name);
	/// Replace the identifier @a _name if it is in the translation map.
	void checkAndReplace(YulName& _name) const;

	NameDispenser& m_nameDispenser;
	std::set<YulName> const& m_namesToFree;
	std::map<YulName, YulName> m_translations;
};

}
