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

#include <libjulia/optimiser/CallGraphGenerator.h>

#include <libjulia/optimiser/Utilities.h>

#include <libsolidity/inlineasm/AsmData.h>

#include <libevmasm/Instruction.h>

#include <stack>

using namespace std;
using namespace dev;
using namespace dev::julia;
using namespace dev::solidity;

namespace
{
// TODO: This algorithm is non-optimal.
struct CallGraphCycleFinder
{
	CallGraph const& callGraph;
	set<string> containedInCycle{};
	set<string> visited{};
	vector<string> currentPath{};

	void visit(string _function)
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

set<string> CallGraph::recursiveFunctions() const
{
	CallGraphCycleFinder cycleFinder{*this};
	// Visiting the root only is not enough, since there may be disconnected recursive functions.
	for (auto const& call: functionCalls)
		cycleFinder.visit(call.first);
	return cycleFinder.containedInCycle;
}

map<string, set<string>> CallGraphGenerator::callGraph(Block const& _ast)
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
	string name = dev::solidity::instructionInfo(_functionalInstruction.instruction.instruction).name;
	std::transform(name.begin(), name.end(), name.begin(), [](unsigned char _c) { return tolower(_c); });
	m_callGraph[m_currentFunction].insert(string{name});
	m_callGraph2.functionCalls[m_currentFunction].insert(string{name});
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
	string previousFunction = m_currentFunction;
	m_currentFunction = _functionDefinition.name;
	assertThrow(m_callGraph.count(m_currentFunction) == 0, OptimizerException, "");
	assertThrow(m_callGraph2.functionCalls.count(m_currentFunction) == 0, OptimizerException, "");
	m_callGraph[m_currentFunction] = {};
	m_callGraph2.functionCalls[m_currentFunction] = {};
	ASTWalker::operator()(_functionDefinition);
	m_currentFunction = previousFunction;
}

CallGraphGenerator::CallGraphGenerator()
{
	m_callGraph[string{}] = {};
	m_callGraph2.functionCalls[string{}] = {};
}

