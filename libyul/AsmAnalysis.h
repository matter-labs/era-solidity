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
/**
 * Analysis part of inline assembly.
 */

#pragma once

#include <liblangutil/Exceptions.h>
#include <liblangutil/EVMVersion.h>

#include <libyul/Dialect.h>
#include <libyul/AsmScope.h>
#include <libyul/AsmDataForward.h>

#include <libyul/backends/evm/AbstractAssembly.h>
#include <libyul/backends/evm/EVMDialect.h>

#include <functional>
#include <list>
#include <memory>
#include <optional>

namespace langutil
{
class ErrorReporter;
struct SourceLocation;
}

namespace yul
{

struct AsmAnalysisInfo;

/**
 * Performs the full analysis stage, calls the ScopeFiller internally, then resolves
 * references and performs other checks.
 * If all these checks pass, code generation should not throw errors.
 */
class AsmAnalyzer
{
public:
	explicit AsmAnalyzer(
		AsmAnalysisInfo& _analysisInfo,
		langutil::ErrorReporter& _errorReporter,
		Dialect const& _dialect,
		ExternalIdentifierAccess::Resolver const& _resolver = ExternalIdentifierAccess::Resolver(),
		std::set<YulString> const& _dataNames = {}
	):
		m_resolver(_resolver),
		m_info(_analysisInfo),
		m_errorReporter(_errorReporter),
		m_dialect(_dialect),
		m_dataNames(_dataNames)
	{
		if (EVMDialect const* evmDialect = dynamic_cast<EVMDialect const*>(&m_dialect))
			m_evmVersion = evmDialect->evmVersion();
	}

	bool analyze(Block const& _block);

	/// Performs analysis on the outermost code of the given object and returns the analysis info.
	/// Asserts on failure.
	static AsmAnalysisInfo analyzeStrictAssertCorrect(Dialect const& _dialect, Object const& _object);
	static AsmAnalysisInfo analyzeStrictAssertCorrect(
		Dialect const& _dialect, Object const& _object, yul::ExternalIdentifierAccess::Resolver _resolver);

	bool operator()(Literal const& _literal);
	bool operator()(Identifier const&);
	bool operator()(ExpressionStatement const&);
	bool operator()(Assignment const& _assignment);
	bool operator()(VariableDeclaration const& _variableDeclaration);
	bool operator()(FunctionDefinition const& _functionDefinition);
	bool operator()(FunctionCall const& _functionCall);
	bool operator()(If const& _if);
	bool operator()(Switch const& _switch);
	bool operator()(ForLoop const& _forLoop);
	bool operator()(Break const&);
	bool operator()(Continue const&);
	bool operator()(Leave const&);
	bool operator()(Block const& _block);

private:
	/// Visits the statement and expects it to deposit one item onto the stack.
	bool expectExpression(Expression const& _expr);
	bool expectDeposit(int _deposit, int _oldHeight, langutil::SourceLocation const& _location);

	/// Verifies that a variable to be assigned to exists and has the same size
	/// as the value, @a _valueSize, unless that is equal to -1.
	bool checkAssignment(Identifier const& _assignment, size_t _valueSize = size_t(-1));

	Scope& scope(Block const* _block);
	void expectValidType(std::string const& type, langutil::SourceLocation const& _location);
	bool warnOnInstructions(dev::eth::Instruction _instr, langutil::SourceLocation const& _location);
	bool warnOnInstructions(std::string const& _instrIdentifier, langutil::SourceLocation const& _location);

	int m_stackHeight = 0;
	yul::ExternalIdentifierAccess::Resolver m_resolver;
	Scope* m_currentScope = nullptr;
	/// Variables that are active at the current point in assembly (as opposed to
	/// "part of the scope but not yet declared")
	std::set<Scope::Variable const*> m_activeVariables;
	AsmAnalysisInfo& m_info;
	langutil::ErrorReporter& m_errorReporter;
	langutil::EVMVersion m_evmVersion;
	Dialect const& m_dialect;
	/// Names of data objects to be referenced by builtin functions with literal arguments.
	std::set<YulString> m_dataNames;
	ForLoop const* m_currentForLoop = nullptr;
};

}
