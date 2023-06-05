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
				for (auto tag: yulContext->functionEntryIDsWithoutScope[recFunc])
				{
					Json::Value func(Json::objectValue);
					func["name"] = recFunc.str();
					func["creationTag"] = tag;
					// FIXME: How do we get the parameters?
					m_recFuncs.append(func);
				}
			}

			if (yulRuntimeContext)
			{
				for (auto tag: yulRuntimeContext->functionEntryIDsWithoutScope[recFunc])
				{
					Json::Value func(Json::objectValue);
					func["name"] = recFunc.str();
					func["runtimeTag"] = tag;
					m_recFuncs.append(func);
				}
			}
		}
	}
};

void Compiler::addExtraMetadata(ContractDefinition const& _contract)
{
	// Set "recursiveFunctions"
	Json::Value recFuncs(Json::arrayValue);

	// FIXME: Can we have cases where where a cycle in creationCallGraph is
	// absent in deployedCallGraph?

	auto& callGraphSetOnce = _contract.annotation().deployedCallGraph;
	if (!callGraphSetOnce.set())
		return;

	auto& callGraph = *callGraphSetOnce;
	set<CallableDeclaration const*> reachableCycleFuncs;
	set<CallableDeclaration const*> reachableFuncs;
	for (FunctionDefinition const* fn: _contract.definedFunctions())
	{
		callGraph->getReachableCycleFuncs(fn, reachableCycleFuncs);
		callGraph->getReachableFuncs(fn, reachableFuncs);
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
			func["creationTag"] = creationTag.data().str();
		if (runtimeTag != evmasm::AssemblyItem(evmasm::UndefinedItem))
			func["runtimeTag"] = runtimeTag.data().str();

		Json::Value paramTypes(Json::arrayValue), retParamTypes(Json::arrayValue);
		unsigned totalParamSize = 0, totalRetParamSize = 0;
		for (auto& param: fn->parameters())
		{
			auto* type = param->type();
			paramTypes.append(type->toString());
			totalParamSize += type->sizeOnStack();
		}
		func["paramTypes"] = paramTypes;
		func["totalParamSize"] = totalParamSize;

		for (auto& param: fn->returnParameters())
		{
			auto* type = param->type();
			retParamTypes.append(type->toString());
			totalRetParamSize += type->sizeOnStack();
		}
		func["retParamTypes"] = retParamTypes;
		func["totalRetParamSize"] = totalRetParamSize;

		recFuncs.append(func);
	}


	// cerr << recFuncs;
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
