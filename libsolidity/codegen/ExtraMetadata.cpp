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

class InlineAsmRecursiveFuncRecorder: public ASTConstVisitor
{
public:
	void run() { m_func.accept(*this); }

	InlineAsmRecursiveFuncRecorder(
		CallableDeclaration const& _func,
		CompilerContext const& _context,
		CompilerContext const& _runtimeContext,
		Json& _recFuncs)
		: m_func(_func), m_context(_context), m_runtimeContext(_runtimeContext), m_recFuncs(_recFuncs)
	{
	}

private:
	CallableDeclaration const& m_func;
	CompilerContext const& m_context;
	CompilerContext const& m_runtimeContext;
	Json& m_recFuncs;

	// Record recursions in @_asm for the extra metadata
	void record(InlineAssembly const& _p_asm, CompilerContext const& _context)
	{
		auto findRes = _context.findInlineAsmContextMapping(&_p_asm);
		if (!findRes)
			return;
		yul::CodeTransformContext const& yulContext = *findRes;

		set<yul::YulString> recFuncs;
		if (_p_asm.annotation().optimizedOperations)
		{
			yul::Block const& code = *_p_asm.annotation().optimizedOperations;
			recFuncs = yul::CallGraphGenerator::callGraph(code).recursiveFunctions();
		}
		else
		{
			recFuncs = yul::CallGraphGenerator::callGraph(_p_asm.operations()).recursiveFunctions();
		}
		for (auto recFunc: recFuncs)
		{
			auto findIt = yulContext.functionInfoMap.find(recFunc);
			if (findIt == yulContext.functionInfoMap.end())
				continue;
			for (auto& func: findIt->second)
			{
				Json record = Json::object();
				record["name"] = recFunc.str();
				if (_context.runtimeContext())
					record["creationTag"] = Json(static_cast<Json::number_integer_t>(func.label));
				else
					record["runtimeTag"] = Json(static_cast<Json::number_integer_t>(func.label));
				record["totalParamSize"] = Json(static_cast<Json::number_integer_t>(func.ast->parameters.size()));
				record["totalRetParamSize"]
					= Json(static_cast<Json::number_integer_t>(func.ast->returnVariables.size()));
				m_recFuncs.push_back(record);
			}
		}
	}

	void endVisit(InlineAssembly const& _p_asm)
	{
		record(_p_asm, m_context);
		record(_p_asm, m_runtimeContext);
	}
};

Json ExtraMetadataRecorder::run(ContractDefinition const& _contract)
{
	// Set "recursiveFunctions"
	Json recFuncs = Json::array();

	// Record recursions in low level calls
	auto recordRecursiveLowLevelFuncs = [&](CompilerContext const& _context)
	{
		for (auto fn: _context.recursiveLowLevelFuncs())
		{
			Json func = Json::object();
			func["name"] = fn.name;
			if (_context.runtimeContext())
				func["creationTag"] = fn.tag;
			else
				func["runtimeTag"] = fn.tag;
			func["totalParamSize"] = fn.ins;
			func["totalRetParamSize"] = fn.outs;
			recFuncs.push_back(func);
		}
	};
	recordRecursiveLowLevelFuncs(m_context);
	recordRecursiveLowLevelFuncs(m_runtimeContext);

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

	// Record recursions in inline assembly
	for (auto* fn: reachableFuncs)
	{
		InlineAsmRecursiveFuncRecorder inAsmRecorder{*fn, m_context, m_runtimeContext, recFuncs};
		inAsmRecorder.run();
	}

	// Record recursions in the solidity source
	auto recordRecursiveSolFuncs = [&](CompilerContext const& _context)
	{
		for (auto* fn: reachableCycleFuncs)
		{
			evmasm::AssemblyItem const& tag = _context.functionEntryLabelIfExists(*fn);
			if (tag == evmasm::AssemblyItem(evmasm::UndefinedItem))
				continue;

			Json func = Json::object();
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

			recFuncs.push_back(func);
		}
	};
	recordRecursiveSolFuncs(m_context);
	recordRecursiveSolFuncs(m_runtimeContext);

	if (!recFuncs.empty())
		m_metadata["recursiveFunctions"] = recFuncs;
	return m_metadata;
}
