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
 * Generates the storage/transient storage layout of a contract.
 */

#pragma once

#include <libsolutil/JSON.h>
#include <libsolidity/ast/AST.h>
#include <libsolidity/ast/Types.h>

namespace solidity::frontend
{

class StorageLayout
{
public:
	/// Generates the storage layout of the contract
	/// @param _contractDef The contract definition
	/// @param _location The location (storage or transient storage) for which to generate the layout
	/// @return A JSON representation of the contract's storage layout.
	Json generate(ContractDefinition const& _contractDef, DataLocation const _location);

private:
	/// Generates the JSON information for a variable and its storage location.
	Json generate(VariableDeclaration const& _var, u256 const& _slot, unsigned _offset);

	/// Generates the JSON information for @param _type
	void generate(Type const* _type);

	/// The key for the JSON object describing a type.
	std::string typeKeyName(Type const* _type);

	Json m_types;

	/// Current analyzed contract
	ContractDefinition const* m_contract = nullptr;
};

}
