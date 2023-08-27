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
 * The extra metadata recorder
 */

#include <libsolidity/ast/ASTForward.h>
#include <libsolidity/codegen/CompilerContext.h>

#include <json/value.h>

#include <memory>

#pragma once

namespace solidity::frontend
{

class ExtraMetadataRecorder
{
	CompilerContext const& m_context;
	CompilerContext const& m_runtimeContext;
	/// The root JSON value of the metadata
	/// Current mappings:
	/// - "recursiveFunctions": array of functions involved in recursion
	Json::Value m_metadata;

public:
	ExtraMetadataRecorder(CompilerContext const& _context, CompilerContext const& _runtimeContext)
		: m_context(_context), m_runtimeContext(_runtimeContext)
	{
	}

	/// Stores the extra metadata of @a _contract in `metadata`
	Json::Value run(ContractDefinition const& _contract);
};

}
