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
			"This contract cannot be compiled safely due to a combination of a memory-unsafe assembly block and a "
			"stack-too-deep error. "
			"The compiler can automatically fix the stack-too-deep error, but only in the absence of memory-unsafe "
			"assembly.\n"
			"To get rid of this error, please check if this assembly block is memory-safe according to "
			"the requirements at \n\n"
			"    https://docs.soliditylang.org/en/latest/assembly.html#memory-safety\n\n"
			"and then mark it with a memory-safe tag.\n"
			"Alternatively, if you feel confident, you may convert this error into a warning project-wide by "
			"setting the EVM_DISABLE_MEMORY_SAFE_ASM_CHECK environment variable:\n\n"
			"    EVM_DISABLE_MEMORY_SAFE_ASM_CHECK=1 <your build command>\n\n"
			"Please be aware of the memory corruption risks described at the link above!\n");
	return true;
}
