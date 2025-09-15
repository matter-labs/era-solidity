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

#include <libsolidity/analysis/UnsafeAsmChecker.h>

#include <liblangutil/ErrorReporter.h>

#include <libsolidity/interface/OptimiserSettings.h>

#include <fmt/format.h>

using namespace solidity;
using namespace solidity::util;
using namespace solidity::langutil;
using namespace solidity::frontend;

bool UnsafeAsmChecker::check(SourceUnit const& _source)
{
	_source.accept(*this);
	return !langutil::Error::containsErrors(m_errorReporter.errors());
}

bool UnsafeAsmChecker::visit(InlineAssembly const& _inlineAsm)
{
	if (*_inlineAsm.annotation().hasMemoryEffects && !_inlineAsm.annotation().markedMemorySafe)
		m_errorReporter.warning(
			5726_error,
			_inlineAsm.location(),
			"Performance of this contract can be compromised due to the presence of this memory-unsafe assembly block.\n"
			"Such assembly blocks hinder many compiler optimizations and also prevent the allocation of a spill area on heap\n"
			"that is required to offload the stack and resolve potential stack-too-deep errors.\n\n"
			"Please check if this assembly block is memory-safe according to the requirements at \n\n"
			"    https://docs.soliditylang.org/en/latest/assembly.html#memory-safety\n\n"
			"and then mark it with a memory-safe tag.\n"
			"Beware of the memory corruption risks described at the link above!\n\n"
			"Overall, using assembly for the sake of performance is now obsolete, as solx can optimize inefficient code with\n"
			"its powerful LLVM-based infrastructure.");
	return true;
}
