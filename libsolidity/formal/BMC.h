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
 * Class that implements an SMT-based Bounded Model Checker (BMC).
 * Traverses the AST such that:
 * - Loops are unrolled
 * - Internal function calls are inlined
 * Creates verification targets for:
 * - Underflow/Overflow
 * - Constant conditions
 * - Assertions
 */

#pragma once


#include <libsolidity/formal/EncodingContext.h>
#include <libsolidity/formal/ModelCheckerSettings.h>
#include <libsolidity/formal/SMTEncoder.h>

#include <libsolidity/interface/ReadFile.h>

#include <libsmtutil/BMCSolverInterface.h>
#include <liblangutil/UniqueErrorReporter.h>

#include <set>
#include <string>
#include <vector>
#include <stack>

using solidity::util::h256;

namespace solidity::langutil
{
class ErrorReporter;
struct ErrorId;
struct SourceLocation;
}

namespace solidity::frontend
{

class BMC: public SMTEncoder
{
public:
	BMC(
		smt::EncodingContext& _context,
		langutil::UniqueErrorReporter& _errorReporter,
		langutil::UniqueErrorReporter& _unsupportedErrorReporter,
		langutil::ErrorReporter& _provedSafeReporter,
		std::map<h256, std::string> const& _smtlib2Responses,
		ReadCallback::Callback const& _smtCallback,
		ModelCheckerSettings _settings,
		langutil::CharStreamProvider const& _charStreamProvider
	);

	void analyze(SourceUnit const& _sources, std::map<ASTNode const*, std::set<VerificationTargetType>, smt::EncodingContext::IdCompare> _solvedTargets);

	/// This is used if the SMT solver is not directly linked into this binary.
	/// @returns a list of inputs to the SMT solver that were not part of the argument to
	/// the constructor.
	std::vector<std::string> unhandledQueries() { return m_interface->unhandledQueries(); }

	/// @returns true if _funCall should be inlined, otherwise false.
	/// @param _scopeContract The contract that contains the current function being analyzed.
	/// @param _contextContract The most derived contract, currently being analyzed.
	static bool shouldInlineFunctionCall(
		FunctionCall const& _funCall,
		ContractDefinition const* _scopeContract,
		ContractDefinition const* _contextContract
	);

private:
	/// AST visitors.
	/// Only nodes that lead to verification targets being built
	/// or checked are visited.
	//@{
	bool visit(ContractDefinition const& _node) override;
	void endVisit(ContractDefinition const& _node) override;
	bool visit(FunctionDefinition const& _node) override;
	void endVisit(FunctionDefinition const& _node) override;
	bool visit(IfStatement const& _node) override;
	bool visit(Conditional const& _node) override;
	bool visit(WhileStatement const& _node) override;
	bool visit(ForStatement const& _node) override;
	void endVisit(UnaryOperation const& _node) override;
	void endVisit(BinaryOperation const& _node) override;
	void endVisit(FunctionCall const& _node) override;
	void endVisit(Return const& _node) override;
	bool visit(TryStatement const& _node) override;
	bool visit(Break const& _node) override;
	bool visit(Continue const& _node) override;
	//@}

	/// Visitor helpers.
	//@{
	void visitAssert(FunctionCall const& _funCall);
	void visitRequire(FunctionCall const& _funCall);
	void visitAddMulMod(FunctionCall const& _funCall) override;
	void assignment(smt::SymbolicVariable& _symVar, smtutil::Expression const& _value) override;
	/// Visits the FunctionDefinition of the called function
	/// if available and inlines the return value.
	void inlineFunctionCall(FunctionCall const& _funCall);
	void inlineFunctionCall(
		FunctionDefinition const* _funDef,
		Expression const& _callStackExpr,
		std::optional<Expression const*> _calledExpr,
		std::vector<Expression const*> const& _arguments
	);
	/// Inlines if the function call is internal or external to `this`.
	/// Erases knowledge about state variables if external.
	void internalOrExternalFunctionCall(FunctionCall const& _funCall);

	/// Creates underflow/overflow verification targets.
	std::pair<smtutil::Expression, smtutil::Expression> arithmeticOperation(
		Token _op,
		smtutil::Expression const& _left,
		smtutil::Expression const& _right,
		Type const* _commonType,
		Expression const& _expression
	) override;

	void reset();

	std::pair<std::vector<smtutil::Expression>, std::vector<std::string>> modelExpressions();
	//@}

	/// Verification targets.
	//@{
	struct BMCVerificationTarget: VerificationTarget
	{
		Expression const* expression;
		std::vector<CallStackEntry> callStack;
		std::pair<std::vector<smtutil::Expression>, std::vector<std::string>> modelExpressions;

		friend bool operator<(BMCVerificationTarget const& _a, BMCVerificationTarget const& _b)
		{
			if (_a.expression->id() == _b.expression->id())
				return _a.type < _b.type;
			else
				return _a.expression->id() < _b.expression->id();
		}
	};

	std::string targetDescription(BMCVerificationTarget const& _target);

	void checkVerificationTargets();
	void checkVerificationTarget(BMCVerificationTarget& _target);
	void checkConstantCondition(BMCVerificationTarget& _target);
	void checkUnderflow(BMCVerificationTarget& _target);
	void checkOverflow(BMCVerificationTarget& _target);
	void checkDivByZero(BMCVerificationTarget& _target);
	void checkBalance(BMCVerificationTarget& _target);
	void checkAssert(BMCVerificationTarget& _target);
	void addVerificationTarget(
		VerificationTargetType _type,
		smtutil::Expression const& _value,
		Expression const* _expression
	);
	//@}

	/// Solver related.
	//@{
	/// Check that a condition can be satisfied.
	void checkCondition(
		BMCVerificationTarget const& _target,
		smtutil::Expression _condition,
		std::vector<CallStackEntry> const& _callStack,
		std::pair<std::vector<smtutil::Expression>, std::vector<std::string>> const& _modelExpressions,
		langutil::SourceLocation const& _location,
		langutil::ErrorId _errorHappens,
		langutil::ErrorId _errorMightHappen,
		std::string const& _additionalValueName = "",
		smtutil::Expression const* _additionalValue = nullptr
	);
	/// Checks that a boolean condition is not constant. Do not warn if the expression
	/// is a literal constant.
	void checkBooleanNotConstant(
		Expression const& _condition,
		smtutil::Expression const& _constraints,
		smtutil::Expression const& _value,
		std::vector<CallStackEntry> const& _callStack
	);
	std::pair<smtutil::CheckResult, std::vector<std::string>>
	checkSatisfiableAndGenerateModel(std::vector<smtutil::Expression> const& _expressionsToEvaluate);

	smtutil::CheckResult checkSatisfiable();
	//@}

	smtutil::Expression mergeVariablesFromLoopCheckpoints();
	bool isInsideLoop() const;

	std::unique_ptr<smtutil::BMCSolverInterface> m_interface;

	/// Flags used for better warning messages.
	bool m_loopExecutionHappened = false;
	bool m_externalFunctionCallHappened = false;

	std::vector<BMCVerificationTarget> m_verificationTargets;

	/// Targets proved safe by this engine.
	std::map<ASTNode const*, std::set<BMCVerificationTarget>, smt::EncodingContext::IdCompare> m_safeTargets;

	/// Targets that were already proven before this engine started.
	std::map<ASTNode const*, std::set<VerificationTargetType>, smt::EncodingContext::IdCompare> m_solvedTargets;

	/// Number of verification conditions that could not be proved.
	size_t m_unprovedAmt = 0;

	enum class LoopControlKind
	{
		Continue,
		Break
	};

	// Current path conditions and SSA indices for break or continue statement
	struct LoopControl {
		LoopControlKind kind;
		smtutil::Expression pathConditions;
		VariableIndices variableIndices;
	};

	// Loop control statements for every loop
	std::stack<std::vector<LoopControl>> m_loopCheckpoints;
};
}
