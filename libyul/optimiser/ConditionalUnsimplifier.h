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
#pragma once

#include <libyul/optimiser/ASTWalker.h>
#include <libyul/optimiser/OptimiserStep.h>
#include <libyul/Dialect.h>
#include <libsolutil/Common.h>

namespace solidity::yul
{

/**
 * Reverse of conditional simplifier.
 *
 */
class ConditionalUnsimplifier: public ASTModifier
{
public:
	static constexpr char const* name{"ConditionalUnsimplifier"};
	static void run(OptimiserStepContext& _context, Block& _ast);

	using ASTModifier::operator();
	void operator()(Switch& _switch) override;
	void operator()(Block& _block) override;

private:
	explicit ConditionalUnsimplifier(
		Dialect const& _dialect,
		std::map<YulName, ControlFlowSideEffects> const& _sideEffects
	):
		m_dialect(_dialect), m_functionSideEffects(_sideEffects)
	{}
	Dialect const& m_dialect;
	std::map<YulName, ControlFlowSideEffects> const& m_functionSideEffects;
};

}
