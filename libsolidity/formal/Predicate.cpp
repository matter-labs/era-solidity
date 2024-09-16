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

#include <libsolidity/formal/Predicate.h>

#include <libsolidity/formal/ExpressionFormatter.h>
#include <libsolidity/formal/SMTEncoder.h>

#include <liblangutil/CharStreamProvider.h>
#include <liblangutil/CharStream.h>
#include <libsolidity/ast/AST.h>
#include <libsolidity/ast/TypeProvider.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>

#include <range/v3/view.hpp>
#include <utility>

using boost::algorithm::starts_with;
using namespace solidity;
using namespace solidity::smtutil;
using namespace solidity::frontend;
using namespace solidity::frontend::smt;

std::map<std::string, Predicate> Predicate::m_predicates;

Predicate const* Predicate::create(
	SortPointer _sort,
	std::string _name,
	PredicateType _type,
	EncodingContext& _context,
	ASTNode const* _node,
	ContractDefinition const* _contractContext,
	std::vector<ScopeOpener const*> _scopeStack
)
{
	solAssert(!m_predicates.count(_name), "");
	return &m_predicates.emplace(
		std::piecewise_construct,
		std::forward_as_tuple(_name),
		std::forward_as_tuple(_name, std::move(_sort), _type, _context.state().hasBytesConcatFunction(),
			_node, _contractContext, std::move(_scopeStack))
	).first->second;
}

Predicate::Predicate(
	std::string _name,
	SortPointer _sort,
	PredicateType _type,
	bool _bytesConcatFunctionInContext,
	ASTNode const* _node,
	ContractDefinition const* _contractContext,
	std::vector<ScopeOpener const*> _scopeStack
):
	m_functor(std::move(_name), {}, std::move(_sort)),
	m_type(_type),
	m_node(_node),
	m_contractContext(_contractContext),
	m_scopeStack(_scopeStack),
	m_bytesConcatFunctionInContext(_bytesConcatFunctionInContext)
{
}

Predicate const* Predicate::predicate(std::string const& _name)
{
	return &m_predicates.at(_name);
}

void Predicate::reset()
{
	m_predicates.clear();
}

smtutil::Expression Predicate::operator()(std::vector<smtutil::Expression> const& _args) const
{
	return smtutil::Expression(m_functor.name, _args, SortProvider::boolSort);
}

smtutil::Expression const& Predicate::functor() const
{
	return m_functor;
}

ASTNode const* Predicate::programNode() const
{
	return m_node;
}

ContractDefinition const* Predicate::contextContract() const
{
	return m_contractContext;
}

ContractDefinition const* Predicate::programContract() const
{
	if (auto const* contract = dynamic_cast<ContractDefinition const*>(m_node))
		if (!contract->constructor())
			return contract;

	return nullptr;
}

FunctionDefinition const* Predicate::programFunction() const
{
	if (auto const* contract = dynamic_cast<ContractDefinition const*>(m_node))
	{
		if (contract->constructor())
			return contract->constructor();
		return nullptr;
	}

	if (auto const* fun = dynamic_cast<FunctionDefinition const*>(m_node))
		return fun;

	return nullptr;
}

FunctionCall const* Predicate::programFunctionCall() const
{
	return dynamic_cast<FunctionCall const*>(m_node);
}

VariableDeclaration  const* Predicate::programVariable() const
{
	return dynamic_cast<VariableDeclaration const*>(m_node);
}

std::optional<std::vector<VariableDeclaration const*>> Predicate::stateVariables() const
{
	if (m_contractContext)
		return SMTEncoder::stateVariablesIncludingInheritedAndPrivate(*m_contractContext);

	return std::nullopt;
}

bool Predicate::isSummary() const
{
	return isFunctionSummary() ||
		isInternalCall() ||
		isExternalCallTrusted() ||
		isExternalCallUntrusted() ||
		isConstructorSummary();
}

bool Predicate::isFunctionSummary() const
{
	return m_type == PredicateType::FunctionSummary;
}

bool Predicate::isFunctionBlock() const
{
	return m_type == PredicateType::FunctionBlock;
}

bool Predicate::isFunctionErrorBlock() const
{
	return m_type == PredicateType::FunctionErrorBlock;
}

bool Predicate::isInternalCall() const
{
	return m_type == PredicateType::InternalCall;
}

bool Predicate::isExternalCallTrusted() const
{
	return m_type == PredicateType::ExternalCallTrusted;
}

bool Predicate::isExternalCallUntrusted() const
{
	return m_type == PredicateType::ExternalCallUntrusted;
}

bool Predicate::isConstructorSummary() const
{
	return m_type == PredicateType::ConstructorSummary;
}

bool Predicate::isInterface() const
{
	return m_type == PredicateType::Interface;
}

bool Predicate::isNondetInterface() const
{
	return m_type == PredicateType::NondetInterface;
}

std::string Predicate::formatSummaryCall(
	std::vector<smtutil::Expression> const& _args,
	langutil::CharStreamProvider const& _charStreamProvider,
	bool _appendTxVars
) const
{
	solAssert(isSummary(), "");

	if (programVariable())
		return {};

	if (auto funCall = programFunctionCall())
	{
		if (funCall->location().hasText())
			return std::string(_charStreamProvider.charStream(*funCall->location().sourceName).text(funCall->location()));
		else
			return {};
	}

	/// The signature of a function summary predicate is: summary(error, this, abiFunctions, (optionally) bytesConcatFunctions, cryptoFunctions, txData, preBlockChainState, preStateVars, preInputVars, postBlockchainState, postStateVars, postInputVars, outputVars).
	/// Here we are interested in preInputVars to format the function call.

	std::string txModel;

	if (_appendTxVars)
	{
		std::set<std::string> txVars;
		if (isFunctionSummary())
		{
			solAssert(programFunction(), "");
			if (programFunction()->isPayable())
				txVars.insert("msg.value");
		}
		else if (isConstructorSummary())
		{
			FunctionDefinition const* fun = programFunction();
			if (fun && fun->isPayable())
				txVars.insert("msg.value");
		}

		struct TxVarsVisitor: public ASTConstVisitor
		{
			bool visit(MemberAccess const& _memberAccess)
			{
				Expression const* memberExpr = SMTEncoder::innermostTuple(_memberAccess.expression());

				Type const* exprType = memberExpr->annotation().type;
				solAssert(exprType, "");
				if (exprType->category() == Type::Category::Magic)
					if (auto const* identifier = dynamic_cast<Identifier const*>(memberExpr))
					{
						ASTString const& name = identifier->name();
						auto memberName = _memberAccess.memberName();

						// TODO remove this for 0.9.0
						if (name == "block" && memberName == "difficulty")
							memberName = "prevrandao";

						if (name == "block" || name == "msg" || name == "tx")
							txVars.insert(name + "." + memberName);
					}

				return true;
			}

			std::set<std::string> txVars;
		} txVarsVisitor;

		if (auto fun = programFunction())
		{
			fun->accept(txVarsVisitor);
			txVars += txVarsVisitor.txVars;
		}

		// Here we are interested in txData from the summary predicate.
		auto txValues = readTxVars(_args.at(txValuesIndex()));
		std::vector<std::string> values;
		for (auto const& _var: txVars)
			if (auto v = txValues.at(_var))
				values.push_back(_var + ": " + *v);

		if (!values.empty())
			txModel = "{ " + boost::algorithm::join(values, ", ") + " }";
	}

	if (auto contract = programContract())
		return contract->name() + ".constructor()" + txModel;

	auto stateVars = stateVariables();
	solAssert(stateVars.has_value(), "");
	auto const* fun = programFunction();
	solAssert(fun, "");

	auto first = _args.begin() + static_cast<int>(firstArgIndex()) + static_cast<int>(stateVars->size());
	auto last = first + static_cast<int>(fun->parameters().size());
	solAssert(first >= _args.begin() && first <= _args.end(), "");
	solAssert(last >= _args.begin() && last <= _args.end(), "");
	auto inTypes = SMTEncoder::replaceUserTypes(FunctionType(*fun).parameterTypes());
	std::vector<std::optional<std::string>> functionArgsCex = formatExpressions(std::vector<smtutil::Expression>(first, last), inTypes);
	std::vector<std::string> functionArgs;

	auto const& params = fun->parameters();
	solAssert(params.size() == functionArgsCex.size(), "");
	bool paramNameInsteadOfValue = false;
	for (unsigned i = 0; i < params.size(); ++i)
		if (params.at(i) && functionArgsCex.at(i))
			functionArgs.emplace_back(*functionArgsCex.at(i));
		else {
			paramNameInsteadOfValue = true;
			functionArgs.emplace_back(params[i]->name());
		}

	std::string fName = fun->isConstructor() ? "constructor" :
		fun->isFallback() ? "fallback" :
		fun->isReceive() ? "receive" :
		fun->name();

	std::string prefix;
	if (fun->isFree())
		prefix = !fun->sourceUnitName().empty() ? (fun->sourceUnitName() + ":") : "";
	else
	{
		solAssert(fun->annotation().contract, "");
		prefix = fun->annotation().contract->name() + ".";
	}

	std::string summary = prefix + fName + "(" + boost::algorithm::join(functionArgs, ", ") + ")" + txModel;
	if (paramNameInsteadOfValue)
		summary += " -- counterexample incomplete; parameter name used instead of value";
	return summary;
}

std::vector<std::optional<std::string>> Predicate::summaryStateValues(std::vector<smtutil::Expression> const& _args) const
{
	/// The signature of a function summary predicate is: summary(error, this, abiFunctions, (optionally) bytesConcatFunctions, cryptoFunctions, txData, preBlockchainState, preStateVars, preInputVars, postBlockchainState, postStateVars, postInputVars, outputVars).
	/// The signature of the summary predicate of a contract without constructor is: summary(error, this, abiFunctions, (optionally) bytesConcatFunctions, cryptoFunctions, txData, preBlockchainState, postBlockchainState, preStateVars, postStateVars).
	/// Here we are interested in postStateVars.
	auto stateVars = stateVariables();
	solAssert(stateVars.has_value(), "");

	std::vector<smtutil::Expression>::const_iterator stateFirst;
	std::vector<smtutil::Expression>::const_iterator stateLast;
	if (auto const* function = programFunction())
	{
		stateFirst = _args.begin() + static_cast<int>(firstArgIndex()) + static_cast<int>(stateVars->size()) + static_cast<int>(function->parameters().size()) + 1;
		stateLast = stateFirst + static_cast<int>(stateVars->size());
	}
	else if (programContract())
	{
		stateFirst = _args.begin() + static_cast<int>(firstStateVarIndex()) + static_cast<int>(stateVars->size());
		stateLast = stateFirst + static_cast<int>(stateVars->size());
	}
	else if (programVariable())
		return {};
	else
		solAssert(false, "");

	solAssert(stateFirst >= _args.begin() && stateFirst <= _args.end(), "");
	solAssert(stateLast >= _args.begin() && stateLast <= _args.end(), "");

	std::vector<smtutil::Expression> stateArgs(stateFirst, stateLast);
	solAssert(stateArgs.size() == stateVars->size(), "");
	auto stateTypes = util::applyMap(*stateVars, [&](auto const& _var) { return _var->type(); });
	return formatExpressions(stateArgs, stateTypes);
}

std::vector<std::optional<std::string>> Predicate::summaryPostInputValues(std::vector<smtutil::Expression> const& _args) const
{
	/// The signature of a function summary predicate is: summary(error, this, abiFunctions, (optionally) bytesConcatFunctions, cryptoFunctions, txData, preBlockchainState, preStateVars, preInputVars, postBlockchainState, postStateVars, postInputVars, outputVars).
	/// Here we are interested in postInputVars.
	auto const* function = programFunction();
	solAssert(function, "");

	auto stateVars = stateVariables();
	solAssert(stateVars.has_value(), "");

	auto const& inParams = function->parameters();

	auto first = _args.begin() + static_cast<int>(firstArgIndex()) + static_cast<int>(stateVars->size()) * 2 + static_cast<int>(inParams.size()) + 1;
	auto last = first + static_cast<int>(inParams.size());

	solAssert(first >= _args.begin() && first <= _args.end(), "");
	solAssert(last >= _args.begin() && last <= _args.end(), "");

	std::vector<smtutil::Expression> inValues(first, last);
	solAssert(inValues.size() == inParams.size(), "");
	auto inTypes = SMTEncoder::replaceUserTypes(FunctionType(*function).parameterTypes());
	return formatExpressions(inValues, inTypes);
}

std::vector<std::optional<std::string>> Predicate::summaryPostOutputValues(std::vector<smtutil::Expression> const& _args) const
{
	/// The signature of a function summary predicate is: summary(error, this, abiFunctions, (optionally) bytesConcatFunctions, cryptoFunctions, txData, preBlockchainState, preStateVars, preInputVars, postBlockchainState, postStateVars, postInputVars, outputVars).
	/// Here we are interested in outputVars.
	auto const* function = programFunction();
	solAssert(function, "");

	auto stateVars = stateVariables();
	solAssert(stateVars.has_value(), "");

	auto const& inParams = function->parameters();

	auto first = _args.begin() + static_cast<int>(firstArgIndex()) + static_cast<int>(stateVars->size()) * 2 + static_cast<int>(inParams.size()) * 2 + 1;

	solAssert(first >= _args.begin() && first <= _args.end(), "");

	std::vector<smtutil::Expression> outValues(first, _args.end());
	solAssert(outValues.size() == function->returnParameters().size(), "");
	auto outTypes = SMTEncoder::replaceUserTypes(FunctionType(*function).returnParameterTypes());
	return formatExpressions(outValues, outTypes);
}

std::pair<std::vector<std::optional<std::string>>, std::vector<VariableDeclaration const*>> Predicate::localVariableValues(std::vector<smtutil::Expression> const& _args) const
{
	/// The signature of a local block predicate is:
	/// block(error, this, abiFunctions, (optionally) bytesConcatFunctions, cryptoFunctions, txData, preBlockchainState, preStateVars, preInputVars, postBlockchainState, postStateVars, postInputVars, outputVars, localVars).
	/// Here we are interested in localVars.
	auto const* function = programFunction();
	solAssert(function, "");

	auto const& localVars = SMTEncoder::localVariablesIncludingModifiers(*function, m_contractContext);
	auto first = _args.end() - static_cast<int>(localVars.size());
	std::vector<smtutil::Expression> outValues(first, _args.end());

	auto mask = util::applyMap(
		localVars,
		[this](auto _var) {
			auto varScope = dynamic_cast<ScopeOpener const*>(_var->scope());
			return find(begin(m_scopeStack), end(m_scopeStack), varScope) != end(m_scopeStack);
		}
	);
	auto localVarsInScope = util::filter(localVars, mask);
	auto outValuesInScope = util::filter(outValues, mask);

	auto outTypes = util::applyMap(localVarsInScope, [](auto _var) { return _var->type(); });
	return {formatExpressions(outValuesInScope, outTypes), localVarsInScope};
}

std::map<std::string, std::string> Predicate::expressionSubstitution(smtutil::Expression const& _predExpr) const
{
	std::map<std::string, std::string> subst;
	std::string predName = functor().name;

	solAssert(contextContract(), "");
	auto const& stateVars = SMTEncoder::stateVariablesIncludingInheritedAndPrivate(*contextContract());

	auto nArgs = _predExpr.arguments.size();

	// The signature of an interface predicate is
	// interface(this, abiFunctions, (optionally) bytesConcatFunctions, cryptoFunctions, blockchainState, stateVariables).
	// An invariant for an interface predicate is a contract
	// invariant over its state, for example `x <= 0`.
	if (isInterface())
	{
		size_t shift = txValuesIndex();
		solAssert(starts_with(predName, "interface"), "");
		subst[_predExpr.arguments.at(0).name] = "address(this)";
		solAssert(nArgs == stateVars.size() + shift, "");
		for (size_t i = nArgs - stateVars.size(); i < nArgs; ++i)
			subst[_predExpr.arguments.at(i).name] = stateVars.at(i - shift)->name();
	}
	// The signature of a nondet interface predicate is
	// nondet_interface(error, this, abiFunctions, (optionally) bytesConcatFunctions, cryptoFunctions, blockchainState, stateVariables, blockchainState', stateVariables').
	// An invariant for a nondet interface predicate is a reentrancy property
	// over the pre and post state variables of a contract, where pre state vars
	// are represented by the variable's name and post state vars are represented
	// by the primed variable's name, for example
	// `(x <= 0) => (x' <= 100)`.
	else if (isNondetInterface())
	{
		solAssert(starts_with(predName, "nondet_interface"), "");
		subst[_predExpr.arguments.at(0).name] = "<errorCode>";
		subst[_predExpr.arguments.at(1).name] = "address(this)";
		solAssert(nArgs == stateVars.size() * 2 + firstArgIndex(), "");
		for (size_t i = nArgs - stateVars.size(), s = 0; i < nArgs; ++i, ++s)
			subst[_predExpr.arguments.at(i).name] = stateVars.at(s)->name() + "'";
		for (size_t i = nArgs - (stateVars.size() * 2 + 1), s = 0; i < nArgs - (stateVars.size() + 1); ++i, ++s)
			subst[_predExpr.arguments.at(i).name] = stateVars.at(s)->name();
	}

	return subst;
}


std::map<std::string, std::optional<std::string>> Predicate::readTxVars(smtutil::Expression const& _tx) const
{
	std::map<std::string, Type const*> const txVars{
		{"block.basefee", TypeProvider::uint256()},
		{"block.chainid", TypeProvider::uint256()},
		{"block.coinbase", TypeProvider::address()},
		{"block.prevrandao", TypeProvider::uint256()},
		{"block.gaslimit", TypeProvider::uint256()},
		{"block.number", TypeProvider::uint256()},
		{"block.timestamp", TypeProvider::uint256()},
		{"blockhash", TypeProvider::array(DataLocation::Memory, TypeProvider::uint256())},
		{"msg.data", TypeProvider::array(DataLocation::CallData)},
		{"msg.sender", TypeProvider::address()},
		{"msg.sig", TypeProvider::fixedBytes(4)},
		{"msg.value", TypeProvider::uint256()},
		{"tx.gasprice", TypeProvider::uint256()},
		{"tx.origin", TypeProvider::address()}
	};
	std::map<std::string, std::optional<std::string>> vars;
	for (auto&& [i, v]: txVars | ranges::views::enumerate)
		vars.emplace(v.first, expressionToString(_tx.arguments.at(i), v.second));
	return vars;
}
