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

#include <libsolidity/codegen/ExtraMetadata.h>

#include <libsolidity/ast/CallGraph.h>
#include <libsolidity/codegen/FuncPtrTracker.h>

#include <libyul/optimiser/CallGraphGenerator.h>

using namespace std;
using namespace solidity;
using namespace solidity::frontend;

class InlineAsmRecursiveFuncReporter: public ASTConstVisitor
{
public:
	void run() { m_func.accept(*this); }

	InlineAsmRecursiveFuncReporter(
		CallableDeclaration const& _func,
		CompilerContext const& _context,
		CompilerContext const& _runtimeContext,
		Json::Value& _recFuncs)
		: m_func(_func), m_context(_context), m_runtimeContext(_runtimeContext), m_recFuncs(_recFuncs)
	{
	}

private:
	CallableDeclaration const& m_func;
	CompilerContext const& m_context;
	CompilerContext const& m_runtimeContext;
	Json::Value& m_recFuncs;

	// Report recursions in @_asm for the extra metadata
	void report(InlineAssembly const& _asm, CompilerContext const& _context)
	{
		auto findIt = _context.inlineAsmContextMap.find(&_asm);
		if (findIt == m_context.inlineAsmContextMap.end())
			return;
		yul::CodeTransformContext const& yulContext = *findIt->second;

		set<yul::YulString> recFuncs;
		if (_asm.annotation().optimizedOperations)
		{
			yul::Block const& code = *_asm.annotation().optimizedOperations;
			recFuncs = yul::CallGraphGenerator::callGraph(code).recursiveFunctions();
		}
		else
		{
			recFuncs = yul::CallGraphGenerator::callGraph(_asm.operations()).recursiveFunctions();
		}
		for (auto recFunc: recFuncs)
		{
			auto findIt = yulContext.functionInfoMap.find(recFunc);
			if (findIt == yulContext.functionInfoMap.end())
				continue;
			for (auto& func: findIt->second)
			{
				Json::Value record(Json::objectValue);
				record["name"] = recFunc.str();
				if (_context.runtimeContext())
					record["creationTag"] = func.label;
				else
					record["runtimeTag"] = func.label;
				record["totalParamSize"] = func.ast->parameters.size();
				record["totalRetParamSize"] = func.ast->returnVariables.size();
				m_recFuncs.append(record);
			}
		}
	}

	void endVisit(InlineAssembly const& _asm)
	{
		report(_asm, m_context);
		report(_asm, m_runtimeContext);
	}
};

void ExtraMetadataReporter::run(ContractDefinition const& _contract)
{
	metadata = make_shared<Json::Value>();

	// Set "recursiveFunctions"
	Json::Value recFuncs(Json::arrayValue);

	// Report recursions in low level calls
	auto reportRecursiveLowLevelFuncs = [&](CompilerContext const& _context)
	{
		for (auto fn: _context.recursiveLowLevelFuncs)
		{
			Json::Value func(Json::objectValue);
			func["name"] = fn.name;
			if (_context.runtimeContext())
				func["creationTag"] = fn.tag;
			else
				func["runtimeTag"] = fn.tag;
			func["totalParamSize"] = fn.ins;
			func["totalRetParamSize"] = fn.outs;
			recFuncs.append(func);
		}
	};
	reportRecursiveLowLevelFuncs(m_context);
	reportRecursiveLowLevelFuncs(m_runtimeContext);

	// Get reachable functions from the call-graphs; And get cycles in the call-graphs
	auto& creationCallGraph = _contract.annotation().creationCallGraph;
	auto& runtimeCallGraph = _contract.annotation().deployedCallGraph;

	set<CallableDeclaration const*> reachableCycleFuncs, reachableFuncs;

	for (FunctionDefinition const* fn: _contract.definedFunctions())
	{
		if (fn->isConstructor() && creationCallGraph.set())
		{
			reachableCycleFuncs += (*creationCallGraph)->getReachableCycleFuncs(fn);
			reachableFuncs += (*creationCallGraph)->getReachableFuncs(fn);
		}
		else if (runtimeCallGraph.set())
		{
			reachableCycleFuncs += (*runtimeCallGraph)->getReachableCycleFuncs(fn);
			reachableFuncs += (*runtimeCallGraph)->getReachableFuncs(fn);
		}
	}

	// Report recursions in inline assembly
	for (auto* fn: reachableFuncs)
	{
		InlineAsmRecursiveFuncReporter inAsmReporter{*fn, m_context, m_runtimeContext, recFuncs};
		inAsmReporter.run();
	}

	// Report recursions in the solidity source
	auto reportRecursiveSolFuncs = [&](CompilerContext const& _context)
	{
		for (auto* fn: reachableCycleFuncs)
		{
			evmasm::AssemblyItem const& tag = _context.functionEntryLabelIfExists(*fn);
			if (tag == evmasm::AssemblyItem(evmasm::UndefinedItem))
				continue;

			Json::Value func(Json::objectValue);
			func["name"] = fn->name();

			// Assembly::new[Push]Tag() asserts that the tag is 32 bits
			auto tagNum = tag.data().convert_to<uint32_t>();
			if (_context.runtimeContext())
				func["creationTag"] = tagNum;
			else
				func["runtimeTag"] = tagNum;

			unsigned totalParamSize = 0, totalRetParamSize = 0;
			for (auto& param: fn->parameters())
				totalParamSize += param->type()->sizeOnStack();
			func["totalParamSize"] = totalParamSize;
			for (auto& param: fn->returnParameters())
				totalRetParamSize += param->type()->sizeOnStack();
			func["totalRetParamSize"] = totalRetParamSize;

			recFuncs.append(func);
		}
	};
	reportRecursiveSolFuncs(m_context);
	reportRecursiveSolFuncs(m_runtimeContext);

	if (!recFuncs.empty())
		(*metadata)["recursiveFunctions"] = recFuncs;
}
