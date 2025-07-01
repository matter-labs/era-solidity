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

#pragma once

#include <libsolidity/ast/ASTAnnotations.h>
#include <libsolidity/ast/ASTForward.h>
#include <libsolidity/ast/ASTVisitor.h>
#include <libsolidity/ast/Types.h>

namespace solidity::langutil
{
class ErrorReporter;
}

namespace solidity::frontend
{
struct OptimiserSettings;
}

namespace solidity::frontend
{

class SpillAreaSafetyChecker: ASTConstVisitor
{
public:
	SpillAreaSafetyChecker(OptimiserSettings const& _optConf, langutil::ErrorReporter& _errorReporter)
		: m_optimiserSettings(_optConf), m_errorReporter(_errorReporter)
	{
	}

	bool check(SourceUnit const& _source);

private:
	size_t m_spillAreaSize = 0;

	OptimiserSettings const& m_optimiserSettings;
	langutil::ErrorReporter& m_errorReporter;

	bool visit(ContractDefinition const& _contr);
	bool visit(InlineAssembly const& _inlineAsm);
};

}
