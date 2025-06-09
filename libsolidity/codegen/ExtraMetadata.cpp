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

class FuncFinder: public yul::ASTWalker
{
public:
	using ASTWalker::operator();

	vector<yul::FunctionDefinition const*> funcs;
	void operator()(yul::FunctionDefinition const& _func) { funcs.push_back(&_func); }
};

class InlineAsmFuncRecorder: public ASTConstVisitor
{
public:
	void run(CallableDeclaration const& _func) { _func.accept(*this); }

	InlineAsmFuncRecorder(CompilerContext const& _context, CompilerContext const& _runtimeContext, Json& _funcs)
		: m_context(_context), m_runtimeContext(_runtimeContext), m_funcs(_funcs)
	{
	}

private:
	CompilerContext const& m_context;
	CompilerContext const& m_runtimeContext;
	Json& m_funcs;

	// Record functions in @_asm for the extra metadata
	void record(InlineAssembly const& _p_asm, CompilerContext const& _context)
	{
		auto findRes = _context.findInlineAsmContextMapping(&_p_asm);
		if (!findRes)
			return;
		yul::CodeTransformContext const& yulContext = *findRes;

		FuncFinder funcFinder;
		if (_p_asm.annotation().optimizedOperations)
			funcFinder(_p_asm.annotation().optimizedOperations->root());
		else
			funcFinder(_p_asm.operations().root());
		for (auto fn: funcFinder.funcs)
		{
			auto findIt = yulContext.functionInfoMap.find(fn->name);
			if (findIt == yulContext.functionInfoMap.end())
				continue;
			for (auto& func: findIt->second)
			{
				Json record = Json::object();
				record["name"] = fn->name.str();
				if (_context.runtimeContext())
					record["creationTag"] = Json(static_cast<Json::number_integer_t>(func.label));
				else
					record["runtimeTag"] = Json(static_cast<Json::number_integer_t>(func.label));
				record["totalParamSize"] = Json(static_cast<Json::number_integer_t>(func.ast->parameters.size()));
				record["totalRetParamSize"]
					= Json(static_cast<Json::number_integer_t>(func.ast->returnVariables.size()));
				m_funcs.push_back(record);
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
	Json funcs = Json::array();

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
			funcs.push_back(func);
		}
	};
	recordRecursiveLowLevelFuncs(m_context);
	recordRecursiveLowLevelFuncs(m_runtimeContext);

	// Get reachable functions from the call-graphs
	auto& creationCallGraph = _contract.annotation().creationCallGraph;
	auto& runtimeCallGraph = _contract.annotation().deployedCallGraph;
	set<CallableDeclaration const*> reachableFuncs;
	reachableFuncs = (*creationCallGraph)->getFuncs();
	reachableFuncs += (*runtimeCallGraph)->getFuncs();

	// Record functions in inline assembly
	for (auto* fn: reachableFuncs)
	{
		InlineAsmFuncRecorder recorder{m_context, m_runtimeContext, funcs};
		recorder.run(*fn);
	}

	auto recordFuncs = [&](CompilerContext const& _context)
	{
		for (auto* fn: reachableFuncs)
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

			funcs.push_back(func);
		}
	};
	recordFuncs(m_context);
	recordFuncs(m_runtimeContext);

	if (!funcs.empty())
		m_metadata["recursiveFunctions"] = funcs;
	return m_metadata;
}
