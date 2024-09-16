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

#include <libsmtutil/SMTPortfolio.h>

#include <libsmtutil/SMTLib2Interface.h>

using namespace solidity;
using namespace solidity::util;
using namespace solidity::frontend;
using namespace solidity::smtutil;

SMTPortfolio::SMTPortfolio(
	std::vector<std::unique_ptr<BMCSolverInterface>> _solvers,
	std::optional<unsigned> _queryTimeout
):
	BMCSolverInterface(_queryTimeout), m_solvers(std::move(_solvers))
{}


void SMTPortfolio::reset()
{
	for (auto const& s: m_solvers)
		s->reset();
}

void SMTPortfolio::push()
{
	for (auto const& s: m_solvers)
		s->push();
}

void SMTPortfolio::pop()
{
	for (auto const& s: m_solvers)
		s->pop();
}

void SMTPortfolio::declareVariable(std::string const& _name, SortPointer const& _sort)
{
	smtAssert(_sort, "");
	for (auto const& s: m_solvers)
		s->declareVariable(_name, _sort);
}

void SMTPortfolio::addAssertion(Expression const& _expr)
{
	for (auto const& s: m_solvers)
		s->addAssertion(_expr);
}

/*
 * Broadcasts the SMT query to all solvers and returns a single result.
 * This comment explains how this result is decided.
 *
 * When a solver is queried, there are four possible answers:
 *   SATISFIABLE (SAT), UNSATISFIABLE (UNSAT), UNKNOWN, CONFLICTING, ERROR
 * We say that a solver _answered_ the query if it returns either:
 *   SAT or UNSAT
 * A solver did not answer the query if it returns either:
 *   UNKNOWN (it tried but couldn't solve it) or ERROR (crash, internal error, API error, etc).
 *
 * Ideally all solvers answer the query and agree on what the answer is
 * (all say SAT or all say UNSAT).
 *
 * The actual logic as as follows:
 * 1) If at least one solver answers the query, all the non-answer results are ignored.
 *   Here SAT/UNSAT is preferred over UNKNOWN since it's an actual answer, and over ERROR
 *   because one buggy solver/integration shouldn't break the portfolio.
 *
 * 2) If at least one solver answers SAT and at least one answers UNSAT, at least one of them is buggy
 * and the result is CONFLICTING.
 *   In the future if we have more than 2 solvers enabled we could go with the majority.
 *
 * 3) If NO solver answers the query:
 *   If at least one solver returned UNKNOWN (where the rest returned ERROR), the result is UNKNOWN.
 *   This is preferred over ERROR since the SMTChecker might decide to abstract the query
 *   when it is told that this is a hard query to solve.
 *
 *   If all solvers return ERROR, the result is ERROR.
*/
std::pair<CheckResult, std::vector<std::string>> SMTPortfolio::check(std::vector<Expression> const& _expressionsToEvaluate)
{
	CheckResult lastResult = CheckResult::ERROR;
	std::vector<std::string> finalValues;
	for (auto const& s: m_solvers)
	{
		CheckResult result;
		std::vector<std::string> values;
		tie(result, values) = s->check(_expressionsToEvaluate);
		if (solverAnswered(result))
		{
			if (!solverAnswered(lastResult))
			{
				lastResult = result;
				finalValues = std::move(values);
			}
			else if (lastResult != result)
			{
				lastResult = CheckResult::CONFLICTING;
				break;
			}
		}
		else if (result == CheckResult::UNKNOWN && lastResult == CheckResult::ERROR)
			lastResult = result;
	}
	return std::make_pair(lastResult, finalValues);
}

std::vector<std::string> SMTPortfolio::unhandledQueries()
{
	// This code assumes that the constructor guarantees that
	// SmtLib2Interface is in position 0, if enabled.
	if (!m_solvers.empty())
		if (auto smtlib2 = dynamic_cast<SMTLib2Interface*>(m_solvers.front().get()))
			return smtlib2->unhandledQueries();
	return {};
}

bool SMTPortfolio::solverAnswered(CheckResult result)
{
	return result == CheckResult::SATISFIABLE || result == CheckResult::UNSATISFIABLE;
}

std::string SMTPortfolio::dumpQuery(std::vector<Expression> const& _expressionsToEvaluate)
{
	// This code assumes that the constructor guarantees that
	// SmtLib2Interface is in position 0, if enabled.
	auto smtlib2 = dynamic_cast<SMTLib2Interface*>(m_solvers.front().get());
	solAssert(smtlib2, "Must use SMTLib2 solver to dump queries");
	return smtlib2->dumpQuery(_expressionsToEvaluate);
}
