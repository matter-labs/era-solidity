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
 * Specific AST walker that generates the call graph.
 */

#include <libyul/AsmData.h>
#include <libyul/optimiser/CallGraphGenerator.h>

#include <libevmasm/Instruction.h>

#include <stack>

using namespace std;
using namespace dev;
using namespace yul;

namespace
{
// TODO: This algorithm is non-optimal.
struct CallGraphCycleFinder
{
	CallGraph const& callGraph;
	set<YulString> containedInCycle{};
	set<YulString> visited{};
	vector<YulString> currentPath{};

	void visit(YulString _function)
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

set<YulString> CallGraph::recursiveFunctions() const
{
	CallGraphCycleFinder cycleFinder{*this};
	// Visiting the root only is not enough, since there may be disconnected recursive functions.
	for (auto const& call: functionCalls)
		cycleFinder.visit(call.first);
	return cycleFinder.containedInCycle;
}

map<YulString, set<YulString>> CallGraphGenerator::callGraph(Block const& _ast)
{
	CallGraphGenerator gen;
	gen(_ast);
	return std::move(gen.m_callGraph);
}

CallGraph CallGraphGenerator::callGraph2(Block const& _ast)
{
	CallGraphGenerator gen;
	gen(_ast);
	return std::move(gen.m_callGraph2);
}

void CallGraphGenerator::operator()(FunctionalInstruction const& _functionalInstruction)
{
	string name = dev::solidity::instructionInfo(_functionalInstruction.instruction).name;
	std::transform(name.begin(), name.end(), name.begin(), [](unsigned char _c) { return tolower(_c); });
	m_callGraph[m_currentFunction].insert(YulString{name});
	m_callGraph2.functionCalls[m_currentFunction].insert(YulString{name});
	ASTWalker::operator()(_functionalInstruction);
}

void CallGraphGenerator::operator()(FunctionCall const& _functionCall)
{
	m_callGraph[m_currentFunction].insert(_functionCall.functionName.name);
	m_callGraph2.functionCalls[m_currentFunction].insert(_functionCall.functionName.name);
	ASTWalker::operator()(_functionCall);
}

void CallGraphGenerator::operator()(ForLoop const&)
{
	m_callGraph2.functionsWithLoops.insert(m_currentFunction);
}

void CallGraphGenerator::operator()(FunctionDefinition const& _functionDefinition)
{
	YulString previousFunction = m_currentFunction;
	m_currentFunction = _functionDefinition.name;
	yulAssert(m_callGraph.count(m_currentFunction) == 0, "");
	yulAssert(m_callGraph2.functionCalls.count(m_currentFunction) == 0, "");
	m_callGraph[m_currentFunction] = {};
	m_callGraph2.functionCalls[m_currentFunction] = {};
	ASTWalker::operator()(_functionDefinition);
	m_currentFunction = previousFunction;
}

CallGraphGenerator::CallGraphGenerator()
{
	m_callGraph[YulString{}] = {};
	m_callGraph2.functionCalls[YulString{}] = {};
}

