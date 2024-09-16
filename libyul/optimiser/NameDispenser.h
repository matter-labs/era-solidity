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
 * Optimiser component that can create new unique names.
 */
#pragma once

#include <libyul/ASTForward.h>

#include <libyul/YulName.h>

#include <set>

namespace solidity::yul
{
struct Dialect;

/**
 * Optimizer component that can be used to generate new names that
 * do not conflict with existing names.
 *
 * Tries to keep names short and appends decimals to disambiguate.
 */
class NameDispenser
{
public:
	/// Initialize the name dispenser with all the names used in the given AST.
	explicit NameDispenser(Dialect const& _dialect, Block const& _ast, std::set<YulName> _reservedNames = {});
	/// Initialize the name dispenser with the given used names.
	explicit NameDispenser(Dialect const& _dialect, std::set<YulName> _usedNames);

	/// @returns a currently unused name that should be similar to _nameHint.
	YulName newName(YulName _nameHint);

	/// Mark @a _name as used, i.e. the dispenser's newName function will not
	/// return it.
	void markUsed(YulName _name) { m_usedNames.insert(_name); }

	std::set<YulName> const& usedNames() { return m_usedNames; }

	/// Returns true if `_name` is either used or is a restricted identifier.
	bool illegalName(YulName _name);

	/// Resets `m_usedNames` with *only* the names that are used in the AST. Also resets value of
	/// `m_counter` to zero.
	void reset(Block const& _ast);

private:
	Dialect const& m_dialect;
	std::set<YulName> m_usedNames;
	std::set<YulName> m_reservedNames;
	size_t m_counter = 0;
};

}
