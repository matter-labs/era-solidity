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
 * Unit tests for the optimized IR caching in CompilerStack.
 */

#pragma once

#include <test/libsolidity/AnalysisFramework.h>
#include <test/TestCase.h>

#include <ostream>
#include <string>

namespace solidity::frontend::test
{

class OptimizedIRCachingTest: public AnalysisFramework, public EVMVersionRestrictedTestCase
{
public:
	OptimizedIRCachingTest(std::string const& _filename):
		EVMVersionRestrictedTestCase(_filename)
	{
		m_source = m_reader.source();
		m_expectation = m_reader.simpleExpectations();
	}

	static std::unique_ptr<TestCase> create(Config const& _config)
	{
		return std::make_unique<OptimizedIRCachingTest>(_config.filename);
	}

	TestResult run(std::ostream& _stream, std::string const& _linePrefix = "", bool _formatted = false) override;

protected:
	void setupCompiler(CompilerStack& _compiler) override;
};

}
