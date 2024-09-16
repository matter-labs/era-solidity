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

#include <libyul/optimiser/NameDispenser.h>

#include <libyul/optimiser/NameCollector.h>
#include <libyul/optimiser/OptimizerUtilities.h>
#include <libyul/AST.h>
#include <libyul/Dialect.h>

#include <libsolutil/CommonData.h>

using namespace solidity;
using namespace solidity::yul;
using namespace solidity::util;

NameDispenser::NameDispenser(Dialect const& _dialect, Block const& _ast, std::set<YulName> _reservedNames):
	NameDispenser(_dialect, NameCollector(_ast).names() + _reservedNames)
{
	m_reservedNames = std::move(_reservedNames);
}

NameDispenser::NameDispenser(Dialect const& _dialect, std::set<YulName> _usedNames):
	m_dialect(_dialect),
	m_usedNames(std::move(_usedNames))
{
}

YulName NameDispenser::newName(YulName _nameHint)
{
	YulName name = _nameHint;
	while (illegalName(name))
	{
		m_counter++;
		name = YulName(_nameHint.str() + "_" + std::to_string(m_counter));
	}
	m_usedNames.emplace(name);
	return name;
}

bool NameDispenser::illegalName(YulName _name)
{
	return isRestrictedIdentifier(m_dialect, _name) || m_usedNames.count(_name);
}

void NameDispenser::reset(Block const& _ast)
{
	m_usedNames = NameCollector(_ast).names() + m_reservedNames;
	m_counter = 0;
}
