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
 * Optimisation stage that replaces variables by their most recently assigned expressions.
 */

#pragma once

#include <libyul/optimiser/DataFlowAnalyzer.h>
#include <libyul/optimiser/OptimiserStep.h>

namespace solidity::yul
{

/**
 * Optimisation stage that replaces variable references by those expressions
 * that are most recently assigned to the referenced variables,
 * but only if the expression is movable and one of the following holds:
 *  - the variable is referenced exactly once (and definition-to-reference does not cross a loop boundary)
 *  - the value is extremely cheap ("cost" of zero like ``caller()``)
 *  - the variable is referenced at most 5 times and the value is rather cheap
 *    ("cost" of at most 1 like a constant up to 0xff) and we are not in a loop
 *
 * Prerequisite: Disambiguator, ForLoopInitRewriter.
 */
class Rematerialiser: public DataFlowAnalyzer
{
public:
	static constexpr char const* name{"Rematerialiser"};
	static void run(
		OptimiserStepContext& _context,
		Block& _ast
	) { run(_context.dialect, _ast); }

	static void run(
		Dialect const& _dialect,
		Block& _ast,
		std::set<YulName> _varsToAlwaysRematerialize = {},
		bool _onlySelectedVariables = false
	);

protected:
	Rematerialiser(
		Dialect const& _dialect,
		Block& _ast,
		std::set<YulName> _varsToAlwaysRematerialize = {},
		bool _onlySelectedVariables = false
	);

	using DataFlowAnalyzer::operator();

	using ASTModifier::visit;
	void visit(Expression& _e) override;

	std::map<YulName, size_t> m_referenceCounts;
	std::set<YulName> m_varsToAlwaysRematerialize;
	bool m_onlySelectedVariables = false;
};

/**
 * If a variable is referenced that is known to have a literal
 * value at that point, replace it by a literal.
 *
 * This is mostly used so that other components do not have to rely
 * on the data flow analyzer.
 *
 * Prerequisite: Disambiguator, ForLoopInitRewriter.
 */
class LiteralRematerialiser: public DataFlowAnalyzer
{
public:
	static constexpr char const* name{"LiteralRematerialiser"};
	static void run(
		OptimiserStepContext& _context,
		Block& _ast
	) { LiteralRematerialiser{_context.dialect}(_ast); }

	using ASTModifier::visit;
	void visit(Expression& _e) override;

private:
	LiteralRematerialiser(Dialect const& _dialect):
		DataFlowAnalyzer(_dialect, MemoryAndStorage::Ignore)
	{}
};


}
