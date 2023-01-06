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

#include <test/libsolidity/ZKEVMExtTest.h>

#include <test/Common.h>

#include <libsolidity/interface/CompilerStack.h>

#include "libyul/Dialect.h"

using namespace std;
using namespace solidity;
using namespace solidity::util;
using namespace solidity::frontend;
using namespace solidity::frontend::test;

ZKEVMExtTest::ZKEVMExtTest(string const& _filename): TestCase(_filename)
{
	m_source = m_reader.source();
	m_expectation = m_reader.simpleExpectations();
}

TestCase::TestResult ZKEVMExtTest::run(ostream& _stream, string const& _linePrefix, bool _formatted)
{
	CompilerStack compiler;

	compiler.setSources({{"", "pragma solidity >=0.0;\n// SPDX-License-Identifier: GPL-3.0\n" + m_source}});
	compiler.setEVMVersion(solidity::test::CommonOptions::get().evmVersion());
	compiler.setOptimiserSettings(solidity::test::CommonOptions::get().optimize);
	compiler.enableEvmBytecodeGeneration(false);
	compiler.enableIRGeneration(true);
	compiler.setMetadataFormat(CompilerStack::MetadataFormat::NoMetadata);
	yul::g_useZKEVMExt = true;
	if (!compiler.compile())
		BOOST_THROW_EXCEPTION(runtime_error("Compilation failed"));

	m_obtainedResult.clear();
	for (string const& contractName: compiler.contractNames())
	{
		m_obtainedResult += compiler.yulIROptimized(contractName);
	}

	return checkResult(_stream, _linePrefix, _formatted);
}
