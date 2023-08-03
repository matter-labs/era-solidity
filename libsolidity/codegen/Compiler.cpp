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
 * @author Christian <c@ethdev.com>
 * @date 2014
 * Solidity compiler.
 */

#include <libsolidity/codegen/Compiler.h>
#include <libsolidity/codegen/FuncPtrTracker.h>

#include <libsolidity/codegen/ContractCompiler.h>
#include <libsolidity/ast/CallGraph.h>
#include <libevmasm/Assembly.h>

#include "libyul/optimiser/CallGraphGenerator.h"

using namespace std;
using namespace solidity;
using namespace solidity::frontend;

class InlineAsmRecursiveFuncTracker: public ASTConstVisitor
{
public:
	void run() { m_func.accept(*this); }

	InlineAsmRecursiveFuncTracker(
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

	// Appends recursive function to `m_recFuncs` from @a _recFuncs
	void
	appendRecFuncs(InlineAssembly const& _asm, CompilerContext const& _context, set<yul::YulString> const& _recFuncs)
	{
		auto findIt = _context.inlineAsmContextMap.find(&_asm);
		if (findIt == m_context.inlineAsmContextMap.end())
			return;
		yul::CodeTransformContext const& yulContext = *findIt->second;

		for (auto recFunc: _recFuncs)
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

		appendRecFuncs(_asm, m_context, recFuncs);
		appendRecFuncs(_asm, m_runtimeContext, recFuncs);
	}
};

void Compiler::addExtraMetadata(ContractDefinition const& _contract)
{
	// Set "recursiveFunctions"
	Json::Value recFuncs(Json::arrayValue);

	auto& creationCallGraph = _contract.annotation().creationCallGraph;
	auto& runtimeCallGraph = _contract.annotation().deployedCallGraph;

	set<CallableDeclaration const*> reachableCycleFuncs, reachableFuncs;

	for (FunctionDefinition const* fn: _contract.definedFunctions())
	{
		if (fn->isConstructor() && creationCallGraph.set())
		{
			(*creationCallGraph)->getReachableCycleFuncs(fn, reachableCycleFuncs);
			(*creationCallGraph)->getReachableFuncs(fn, reachableFuncs);
		}
		else if (runtimeCallGraph.set())
		{
			(*runtimeCallGraph)->getReachableCycleFuncs(fn, reachableCycleFuncs);
			(*runtimeCallGraph)->getReachableFuncs(fn, reachableFuncs);
		}
	}

	// Report recursive low level calls
	for (auto fn: m_context.recursiveLowLevelFuncs)
	{
		Json::Value func(Json::objectValue);
		func["name"] = get<0>(fn);
		func["creationTag"] = get<1>(fn);
		func["totalParamSize"] = get<2>(fn);
		func["totalRetParamSize"] = get<3>(fn);
		recFuncs.append(func);
	}
	for (auto fn: m_runtimeContext.recursiveLowLevelFuncs)
	{
		Json::Value func(Json::objectValue);
		func["name"] = get<0>(fn);
		func["runtimeTag"] = get<1>(fn);
		func["totalParamSize"] = get<2>(fn);
		func["totalRetParamSize"] = get<3>(fn);
		recFuncs.append(func);
	}

	for (auto* fn: reachableFuncs)
	{
		InlineAsmRecursiveFuncTracker inAsmTracker{*fn, m_context, m_runtimeContext, recFuncs};
		inAsmTracker.run();
	}

	for (auto* fn: reachableCycleFuncs)
	{
		evmasm::AssemblyItem const& creationTag = m_context.functionEntryLabelIfExists(*fn);
		evmasm::AssemblyItem const& runtimeTag = m_runtimeContext.functionEntryLabelIfExists(*fn);
		if (creationTag == evmasm::AssemblyItem(evmasm::UndefinedItem)
			&& runtimeTag == evmasm::AssemblyItem(evmasm::UndefinedItem))
			continue;

		Json::Value func(Json::objectValue);
		func["name"] = fn->name();


		if (creationTag != evmasm::AssemblyItem(evmasm::UndefinedItem))
			// Assembly::new[Push]Tag() asserts that the tag is 32 bits
			func["creationTag"] = creationTag.data().convert_to<uint32_t>();
		if (runtimeTag != evmasm::AssemblyItem(evmasm::UndefinedItem))
			func["runtimeTag"] = runtimeTag.data().convert_to<uint32_t>();

		unsigned totalParamSize = 0, totalRetParamSize = 0;
		for (auto& param: fn->parameters())
			totalParamSize += param->type()->sizeOnStack();
		func["totalParamSize"] = totalParamSize;
		for (auto& param: fn->returnParameters())
			totalRetParamSize += param->type()->sizeOnStack();
		func["totalRetParamSize"] = totalRetParamSize;

		recFuncs.append(func);
	}

	if (!recFuncs.empty())
		m_runtimeContext.metadata["recursiveFunctions"] = recFuncs;
}

void Compiler::compileContract(
	ContractDefinition const& _contract,
	std::map<ContractDefinition const*, shared_ptr<Compiler const>> const& _otherCompilers,
	bytes const& _metadata
)
{
	ContractCompiler runtimeCompiler(nullptr, m_runtimeContext, m_optimiserSettings);
	runtimeCompiler.compileContract(_contract, _otherCompilers);
	m_runtimeContext.appendToAuxiliaryData(_metadata);

	// This might modify m_runtimeContext because it can access runtime functions at
	// creation time.
	OptimiserSettings creationSettings{m_optimiserSettings};
	// The creation code will be executed at most once, so we modify the optimizer
	// settings accordingly.
	creationSettings.expectedExecutionsPerDeployment = 1;
	ContractCompiler creationCompiler(&runtimeCompiler, m_context, creationSettings);
	m_runtimeSub = creationCompiler.compileConstructor(_contract, _otherCompilers);

	m_context.optimise(m_optimiserSettings);

	addExtraMetadata(_contract);

	solAssert(m_context.appendYulUtilityFunctionsRan(), "appendYulUtilityFunctions() was not called.");
	solAssert(m_runtimeContext.appendYulUtilityFunctionsRan(), "appendYulUtilityFunctions() was not called.");
}

std::shared_ptr<evmasm::Assembly> Compiler::runtimeAssemblyPtr() const
{
	solAssert(m_context.runtimeContext(), "");
	return m_context.runtimeContext()->assemblyPtr();
}
