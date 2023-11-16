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
/** @file LinkerObject.cpp
 * @author Christian R <c@ethdev.com>
 * @date 2015
 */

#include <libevmasm/LinkerObject.h>
#include <libsolutil/CommonData.h>
#include <libsolutil/Keccak256.h>

using namespace solidity;
using namespace solidity::util;
using namespace solidity::evmasm;

void LinkerObject::append(LinkerObject const& _other)
{
	for (auto const& ref: _other.linkReferences)
		linkReferences[ref.first + bytecode.size()] = ref.second;
	bytecode += _other.bytecode;
}

void LinkerObject::link(std::map<std::string, h160> const& _libraryAddresses)
{
	std::map<size_t, std::string> remainingRefs;
	for (auto const& linkRef: linkReferences)
		if (h160 const* address = matchLibrary(linkRef.second, _libraryAddresses))
			copy(address->data(), address->data() + 20, bytecode.begin() + std::vector<uint8_t>::difference_type(linkRef.first));
		else
			remainingRefs.insert(linkRef);
	linkReferences.swap(remainingRefs);
}

std::string LinkerObject::toHex() const
{
	return "The EVM bytecode is unavailable in the zkEVM edition of solc";
	std::string hex = solidity::util::toHex(bytecode);
	for (auto const& ref: linkReferences)
	{
		size_t pos = ref.first * 2;
		std::string hash = libraryPlaceholder(ref.second);
		hex[pos] = hex[pos + 1] = hex[pos + 38] = hex[pos + 39] = '_';
		for (size_t i = 0; i < 36; ++i)
			hex[pos + 2 + i] = hash.at(i);
	}
	return hex;
}

std::string LinkerObject::libraryPlaceholder(std::string const& _libraryName)
{
	return "$" + keccak256(_libraryName).hex().substr(0, 34) + "$";
}

h160 const*
LinkerObject::matchLibrary(
	std::string const& _linkRefName,
	std::map<std::string, h160> const& _libraryAddresses
)
{
	auto it = _libraryAddresses.find(_linkRefName);
	if (it != _libraryAddresses.end())
		return &it->second;
	return nullptr;
}

bool LinkerObject::operator<(LinkerObject const& _other) const
{
	return tie(this->bytecode, this->linkReferences, this->immutableReferences) <
			tie(_other.bytecode, _other.linkReferences, _other.immutableReferences);
}
