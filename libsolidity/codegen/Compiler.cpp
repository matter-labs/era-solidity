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

	// FIXME: Add const to CompilerContext&
	InlineAsmRecursiveFuncTracker(
		CallableDeclaration const& _func,
		CompilerContext& _context,
		CompilerContext& _runtimeContext,
		Json::Value& _recFuncs)
		: m_func(_func), m_context(_context), m_runtimeContext(_runtimeContext), m_recFuncs(_recFuncs)
	{
	}

private:
	CallableDeclaration const& m_func;
	CompilerContext& m_context;
	CompilerContext& m_runtimeContext;
	Json::Value& m_recFuncs;
	void endVisit(InlineAssembly const& _asm)
	{
		yul::Block const& code = _asm.operations();
		set<yul::YulString> recFuncs = yul::CallGraphGenerator::callGraph(code).recursiveFunctions();
		shared_ptr<yul::CodeTransformContext> yulContext = m_context.inlineAsmContextMap[&_asm];
		shared_ptr<yul::CodeTransformContext> yulRuntimeContext = m_runtimeContext.inlineAsmContextMap[&_asm];
		for (auto recFunc: recFuncs)
		{
			if (yulContext)
			{
				// FIXME: How does the call-graph of strings represent yul's
				// scopes?
				for (auto& func: yulContext->functionInfoMap[recFunc])
				{
					Json::Value record(Json::objectValue);
					record["name"] = recFunc.str();
					record["creationTag"] = func.label;
					// FIXME: Will this work for custom yul types?
					record["totalParamSize"] = func.ast->parameters.size();
					record["totalRetParamSize"] = func.ast->returnVariables.size();
					m_recFuncs.append(record);
				}
			}

			if (yulRuntimeContext)
			{
				for (auto& func: yulRuntimeContext->functionInfoMap[recFunc])
				{
					Json::Value record(Json::objectValue);
					record["name"] = recFunc.str();
					record["runtimeTag"] = func.label;
					record["totalParamSize"] = func.ast->parameters.size();
					record["totalRetParamSize"] = func.ast->returnVariables.size();
					m_recFuncs.append(record);
				}
			}
		}
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
