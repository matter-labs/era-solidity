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
 * @file PeepholeOptimiser.h
 * Performs local optimising code changes to assembly.
 */
#pragma once

#include <vector>
#include <cstddef>
#include <iterator>

#include <liblangutil/EVMVersion.h>

namespace solidity::evmasm
{
class AssemblyItem;
using AssemblyItems = std::vector<AssemblyItem>;

class PeepholeOptimisationMethod
{
public:
	virtual ~PeepholeOptimisationMethod() = default;
	virtual size_t windowSize() const;
	virtual bool apply(AssemblyItems::const_iterator _in, std::back_insert_iterator<AssemblyItems> _out);
};

class PeepholeOptimiser
{
public:
	explicit PeepholeOptimiser(AssemblyItems& _items, langutil::EVMVersion const _evmVersion):
	m_items(_items),
	m_evmVersion(_evmVersion)
	{
	}
	virtual ~PeepholeOptimiser() = default;

	bool optimise();

private:
	AssemblyItems& m_items;
	AssemblyItems m_optimisedItems;
	langutil::EVMVersion const m_evmVersion;
};

}
