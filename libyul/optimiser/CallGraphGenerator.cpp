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
 * Specific AST walker that generates the call graph.
 */

#include <libyul/AST.h>
#include <libyul/optimiser/CallGraphGenerator.h>

#include <libsolutil/CommonData.h>
#include <stack>

using namespace solidity;
using namespace solidity::yul;
using namespace solidity::util;

namespace
{
// TODO: This algorithm is non-optimal.
struct CallGraphCycleFinder
{
	CallGraph const& callGraph;
	std::set<YulName> containedInCycle{};
	std::set<YulName> visited{};
	std::vector<YulName> currentPath{};

	void visit(YulName _function)
	{
		if (visited.count(_function))
			return;
		if (
			auto it = find(currentPath.begin(), currentPath.end(), _function);
			it != currentPath.end()
		)
			containedInCycle.insert(it, currentPath.end());
		else
		{
			currentPath.emplace_back(_function);
			if (callGraph.functionCalls.count(_function))
				for (auto const& child: callGraph.functionCalls.at(_function))
					visit(child);
			currentPath.pop_back();
			visited.insert(_function);
		}
	}
};
}

std::set<YulName> CallGraph::recursiveFunctions() const
{
	CallGraphCycleFinder cycleFinder{*this};
	// Visiting the root only is not enough, since there may be disconnected recursive functions.
	for (auto const& call: functionCalls)
		cycleFinder.visit(call.first);
	return cycleFinder.containedInCycle;
}

CallGraph CallGraphGenerator::callGraph(Block const& _ast)
{
	CallGraphGenerator gen;
	gen(_ast);
	return std::move(gen.m_callGraph);
}

void CallGraphGenerator::operator()(FunctionCall const& _functionCall)
{
	auto& functionCalls = m_callGraph.functionCalls[m_currentFunction];
	if (!util::contains(functionCalls, _functionCall.functionName.name))
		functionCalls.emplace_back(_functionCall.functionName.name);
	ASTWalker::operator()(_functionCall);
}

void CallGraphGenerator::operator()(ForLoop const& _forLoop)
{
	m_callGraph.functionsWithLoops.insert(m_currentFunction);
	ASTWalker::operator()(_forLoop);
}

void CallGraphGenerator::operator()(FunctionDefinition const& _functionDefinition)
{
	YulName previousFunction = m_currentFunction;
	m_currentFunction = _functionDefinition.name;
	yulAssert(m_callGraph.functionCalls.count(m_currentFunction) == 0, "");
	m_callGraph.functionCalls[m_currentFunction] = {};
	ASTWalker::operator()(_functionDefinition);
	m_currentFunction = previousFunction;
}

CallGraphGenerator::CallGraphGenerator()
{
	m_callGraph.functionCalls[YulName{}] = {};
}

