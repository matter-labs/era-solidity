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

#include <libsolidity/analysis/SpillAreaSafetyChecker.h>

#include <liblangutil/ErrorReporter.h>

#include <libsolidity/interface/OptimiserSettings.h>

#include <fmt/format.h>

using namespace solidity;
using namespace solidity::util;
using namespace solidity::langutil;
using namespace solidity::frontend;

bool SpillAreaSafetyChecker::check(SourceUnit const& _source)
{
	// Don't bother walking the ast if the spill area sizes are zero.
	bool foundNonEmptySpillArea = false;
	for (auto i: m_optimiserSettings.spillAreaSize)
		foundNonEmptySpillArea |= i.second.creation + i.second.runtime > 0;
	if (!foundNonEmptySpillArea)
		return true;

	_source.accept(*this);
	return !langutil::Error::containsErrors(m_errorReporter.errors());
}

bool SpillAreaSafetyChecker::visit(ContractDefinition const& _contr)
{
	// Reset spill area size.
	m_spillAreaSize = 0;

	auto found = m_optimiserSettings.spillAreaSize.find(_contr.fullyQualifiedName());
	if (found != m_optimiserSettings.spillAreaSize.end())
		// Consider both creation and runtime spill area size since the inline-asm might be reachable from both.
		m_spillAreaSize = found->second.creation + found->second.runtime;
	return true;
}

bool SpillAreaSafetyChecker::visit(InlineAssembly const& _inlineAsm)
{
	if (m_spillAreaSize == 0)
		return true;
	if (*_inlineAsm.annotation().hasMemoryEffects && !_inlineAsm.annotation().markedMemorySafe)
		m_errorReporter.typeError(
			5726_error,
			_inlineAsm.location(),
			"memory-unsafe assembly is not supported when spilling is enabled due to stack-too-deep. Make the "
			"assembly memory-safe to fix this. See "
			"https://docs.soliditylang.org/en/latest/assembly.html#memory-safety");
	return true;
}
