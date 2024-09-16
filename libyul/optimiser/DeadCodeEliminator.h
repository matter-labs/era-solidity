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
 * Optimisation stage that removes unused variables and functions.
 */

#pragma once

#include <libyul/optimiser/ASTWalker.h>
#include <libyul/YulName.h>
#include <libyul/ControlFlowSideEffects.h>

#include <map>
#include <set>

namespace solidity::yul
{
struct Dialect;
struct OptimiserStepContext;

/**
 * Optimisation stage that removes unreachable code
 *
 * Unreachable code is any code within a block which is preceded by a
 * leave, return, invalid, break, continue, selfdestruct or revert or
 * a call to a user-defined function that never returns (either due to
 * recursion or a call to return / revert / stop).
 *
 * Function definitions are retained as they might be called by earlier
 * code and thus are considered reachable.
 *
 * Because variables declared in a for loop's init block have their scope extended to the loop body,
 * we require ForLoopInitRewriter to run before this step.
 *
 * Prerequisite: ForLoopInitRewriter, Function Hoister, Function Grouper
 */
class DeadCodeEliminator: public ASTModifier
{
public:
	static constexpr char const* name{"DeadCodeEliminator"};
	static void run(OptimiserStepContext&, Block& _ast);

	using ASTModifier::operator();
	void operator()(ForLoop& _for) override;
	void operator()(Block& _block) override;

private:
	DeadCodeEliminator(
		Dialect const& _dialect,
		std::map<YulName, ControlFlowSideEffects> _sideEffects
	): m_dialect(_dialect), m_functionSideEffects(std::move(_sideEffects)) {}

	Dialect const& m_dialect;
	std::map<YulName, ControlFlowSideEffects> m_functionSideEffects;
};

}
