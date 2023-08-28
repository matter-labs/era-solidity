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
 * Analyzer part of inline assembly.
 */

#include <libyul/AsmAnalysis.h>

#include <libyul/AsmData.h>
#include <libyul/AsmScopeFiller.h>
#include <libyul/AsmScope.h>
#include <libyul/AsmAnalysisInfo.h>
#include <libyul/Utilities.h>
#include <libyul/Exceptions.h>
#include <libyul/Object.h>

#include <liblangutil/ErrorReporter.h>

#include <boost/range/adaptor/reversed.hpp>
#include <boost/algorithm/string.hpp>

#include <memory>
#include <functional>
#include <utility>

using namespace std;
using namespace solidity;
using namespace solidity::yul;
using namespace solidity::util;
using namespace solidity::langutil;

bool AsmAnalyzer::analyze(Block const& _block)
{
	auto watcher = m_errorReporter.errorWatcher();
	try
	{
		if (!(ScopeFiller(m_info, m_errorReporter))(_block))
			return false;

		(*this)(_block);
	}
	catch (FatalError const&)
	{
		// This FatalError con occur if the errorReporter has too many errors.
		yulAssert(!watcher.ok(), "Fatal error detected, but no error is reported.");
	}
	return watcher.ok();
}

AsmAnalysisInfo AsmAnalyzer::analyzeStrictAssertCorrect(Dialect const& _dialect, Object const& _object)
{
	ErrorList errorList;
	langutil::ErrorReporter errors(errorList);
	AsmAnalysisInfo analysisInfo;
	bool success = yul::AsmAnalyzer(
		analysisInfo,
		errors,
		_dialect,
		{},
		_object.dataNames()
	).analyze(*_object.code);
	yulAssert(success && !errors.hasErrors(), "Invalid assembly/yul code.");
	return analysisInfo;
}

AsmAnalysisInfo AsmAnalyzer::analyzeStrictAssertCorrect(
	Dialect const& _dialect, Object const& _object, yul::ExternalIdentifierAccess::Resolver _resolver)
{
	ErrorList errorList;
	langutil::ErrorReporter errors(errorList);
	AsmAnalysisInfo analysisInfo;
	bool success = yul::AsmAnalyzer(analysisInfo, errors, _dialect, _resolver, _object.dataNames())
					   .analyze(*_object.code);
	yulAssert(success && !errors.hasErrors(), "Invalid assembly/yul code.");
	return analysisInfo;
}

vector<YulString> AsmAnalyzer::operator()(Literal const& _literal)
{
	expectValidType(_literal.type, _literal.location);
	if (_literal.kind == LiteralKind::String && _literal.value.str().size() > 32)
		m_errorReporter.typeError(
			3069_error,
			_literal.location,
			"String literal too long (" + to_string(_literal.value.str().size()) + " > 32)"
		);
	else if (_literal.kind == LiteralKind::Number && bigint(_literal.value.str()) > u256(-1))
		m_errorReporter.typeError(6708_error, _literal.location, "Number literal too large (> 256 bits)");
	else if (_literal.kind == LiteralKind::Boolean)
		yulAssert(_literal.value == "true"_yulstring || _literal.value == "false"_yulstring, "");

	if (!m_dialect.validTypeForLiteral(_literal.kind, _literal.value, _literal.type))
		m_errorReporter.typeError(
			5170_error,
			_literal.location,
			"Invalid type \"" + _literal.type.str() + "\" for literal \"" + _literal.value.str() + "\"."
		);


	return {_literal.type};
}

vector<YulString> AsmAnalyzer::operator()(Identifier const& _identifier)
{
	yulAssert(!_identifier.name.empty(), "");
	auto watcher = m_errorReporter.errorWatcher();
	YulString type = m_dialect.defaultType;

	if (m_currentScope->lookup(_identifier.name, GenericVisitor{
		[&](Scope::Variable const& _var)
		{
			if (!m_activeVariables.count(&_var))
				m_errorReporter.declarationError(
					4990_error,
					_identifier.location,
					"Variable " + _identifier.name.str() + " used before it was declared."
				);
			type = _var.type;
		},
		[&](Scope::Function const&)
		{
			m_errorReporter.typeError(
				6041_error,
				_identifier.location,
				"Function " + _identifier.name.str() + " used without being called."
			);
		}
	}))
	{
	}
	else
	{
		bool found = m_resolver && m_resolver(
			_identifier,
			yul::IdentifierContext::RValue,
			m_currentScope->insideFunction()
		);
		if (!found && watcher.ok())
			// Only add an error message if the callback did not do it.
			m_errorReporter.declarationError(8198_error, _identifier.location, "Identifier not found.");
	}

	return {type};
}

void AsmAnalyzer::operator()(ExpressionStatement const& _statement)
{
	auto watcher = m_errorReporter.errorWatcher();
	vector<YulString> types = std::visit(*this, _statement.expression);
	if (watcher.ok() && !types.empty())
		m_errorReporter.typeError(
			3083_error,
			_statement.location,
			"Top-level expressions are not supposed to return values (this expression returns " +
			to_string(types.size()) +
			" value" +
			(types.size() == 1 ? "" : "s") +
			"). Use ``pop()`` or assign them."
		);
}

void AsmAnalyzer::operator()(Assignment const& _assignment)
{
	yulAssert(_assignment.value, "");
	size_t const numVariables = _assignment.variableNames.size();
	yulAssert(numVariables >= 1, "");

	set<YulString> variables;
	for (auto const& _variableName: _assignment.variableNames)
		if (!variables.insert(_variableName.name).second)
			m_errorReporter.declarationError(
				9005_error,
				_assignment.location,
				"Variable " +
				_variableName.name.str() +
				" occurs multiple times on the left-hand side of the assignment."
			);

	vector<YulString> types = std::visit(*this, *_assignment.value);

	if (types.size() != numVariables)
		m_errorReporter.declarationError(
			8678_error,
			_assignment.location,
			"Variable count does not match number of values (" +
			to_string(numVariables) +
			" vs. " +
			to_string(types.size()) +
			")"
		);

	for (size_t i = 0; i < numVariables; ++i)
		if (i < types.size())
			checkAssignment(_assignment.variableNames[i], types[i]);
}

void AsmAnalyzer::operator()(VariableDeclaration const& _varDecl)
{
	size_t const numVariables = _varDecl.variables.size();
	if (m_resolver)
		for (auto const& variable: _varDecl.variables)
			// Call the resolver for variable declarations to allow it to raise errors on shadowing.
			m_resolver(
				yul::Identifier{variable.location, variable.name},
				yul::IdentifierContext::VariableDeclaration,
				m_currentScope->insideFunction()
			);
	for (auto const& variable: _varDecl.variables)
		expectValidType(variable.type, variable.location);

	if (_varDecl.value)
	{
		vector<YulString> types = std::visit(*this, *_varDecl.value);
		if (types.size() != numVariables)
			m_errorReporter.declarationError(
				3812_error,
				_varDecl.location,
				"Variable count mismatch: " +
				to_string(numVariables) +
				" variables and " +
				to_string(types.size()) +
				" values."
			);

		for (size_t i = 0; i < _varDecl.variables.size(); ++i)
		{
			YulString givenType = m_dialect.defaultType;
			if (i < types.size())
				givenType = types[i];
			TypedName const& variable = _varDecl.variables[i];
			if (variable.type != givenType)
				m_errorReporter.typeError(
					3947_error,
					variable.location,
					"Assigning value of type \"" + givenType.str() + "\" to variable of type \"" + variable.type.str() + "."
				);
		}
	}

	for (TypedName const& variable: _varDecl.variables)
		m_activeVariables.insert(&std::get<Scope::Variable>(
			m_currentScope->identifiers.at(variable.name))
		);
}

void AsmAnalyzer::operator()(FunctionDefinition const& _funDef)
{
	yulAssert(!_funDef.name.empty(), "");
	Block const* virtualBlock = m_info.virtualBlocks.at(&_funDef).get();
	yulAssert(virtualBlock, "");
	Scope& varScope = scope(virtualBlock);
	for (auto const& var: _funDef.parameters + _funDef.returnVariables)
	{
		expectValidType(var.type, var.location);
		m_activeVariables.insert(&std::get<Scope::Variable>(varScope.identifiers.at(var.name)));
	}

	(*this)(_funDef.body);
}

vector<YulString> AsmAnalyzer::operator()(FunctionCall const& _funCall)
{
	yulAssert(!_funCall.functionName.name.empty(), "");
	auto watcher = m_errorReporter.errorWatcher();
	vector<YulString> const* parameterTypes = nullptr;
	vector<YulString> const* returnTypes = nullptr;
	vector<bool> const* needsLiteralArguments = nullptr;

	if (BuiltinFunction const* f = m_dialect.builtin(_funCall.functionName.name))
	{
		parameterTypes = &f->parameters;
		returnTypes = &f->returns;
		if (f->literalArguments)
			needsLiteralArguments = &f->literalArguments.value();

		validateInstructions(_funCall);
	}
	else if (!m_currentScope->lookup(_funCall.functionName.name, GenericVisitor{
		[&](Scope::Variable const&)
		{
			m_errorReporter.typeError(
				4202_error,
				_funCall.functionName.location,
				"Attempt to call variable instead of function."
			);
		},
		[&](Scope::Function const& _fun)
		{
			parameterTypes = &_fun.arguments;
			returnTypes = &_fun.returns;
		}
	}))
	{
		if (!validateInstructions(_funCall))
			m_errorReporter.declarationError(4619_error, _funCall.functionName.location, "Function not found.");
		yulAssert(!watcher.ok(), "Expected a reported error.");
	}

	if (parameterTypes && _funCall.arguments.size() != parameterTypes->size())
		m_errorReporter.typeError(
			7000_error,
			_funCall.functionName.location,
			"Function expects " +
			to_string(parameterTypes->size()) +
			" arguments but got " +
			to_string(_funCall.arguments.size()) + "."
		);

	vector<YulString> argTypes;
	for (size_t i = _funCall.arguments.size(); i > 0; i--)
	{
		Expression const& arg = _funCall.arguments[i - 1];
		bool isLiteralArgument = needsLiteralArguments && (*needsLiteralArguments)[i - 1];
		bool isStringLiteral = holds_alternative<Literal>(arg) && get<Literal>(arg).kind == LiteralKind::String;

		if (isLiteralArgument && isStringLiteral)
			argTypes.emplace_back(expectUnlimitedStringLiteral(get<Literal>(arg)));
		else
			argTypes.emplace_back(expectExpression(arg));

		if (isLiteralArgument)
		{
			if (!holds_alternative<Literal>(arg))
				m_errorReporter.typeError(
					9114_error,
					_funCall.functionName.location,
					"Function expects direct literals as arguments."
				);
			else if (
				_funCall.functionName.name.str() == "datasize" ||
				_funCall.functionName.name.str() == "dataoffset"
			)
				if (!m_dataNames.count(std::get<Literal>(arg).value))
					m_errorReporter.typeError(
						3517_error,
						_funCall.functionName.location,
						"Unknown data object \"" + std::get<Literal>(arg).value.str() + "\"."
					);
		}
	}
	std::reverse(argTypes.begin(), argTypes.end());

	if (parameterTypes && parameterTypes->size() == argTypes.size())
		for (size_t i = 0; i < parameterTypes->size(); ++i)
			expectType((*parameterTypes)[i], argTypes[i], locationOf(_funCall.arguments[i]));

	if (watcher.ok())
	{
		yulAssert(parameterTypes && parameterTypes->size() == argTypes.size(), "");
		yulAssert(returnTypes, "");
		return *returnTypes;
	}
	else if (returnTypes)
		return vector<YulString>(returnTypes->size(), m_dialect.defaultType);
	else
		return {};
}

void AsmAnalyzer::operator()(If const& _if)
{
	expectBoolExpression(*_if.condition);

	(*this)(_if.body);
}

void AsmAnalyzer::operator()(Switch const& _switch)
{
	yulAssert(_switch.expression, "");

	if (_switch.cases.size() == 1 && !_switch.cases[0].value)
		m_errorReporter.warning(
			9592_error,
			_switch.location,
			"\"switch\" statement with only a default case."
		);

	YulString valueType = expectExpression(*_switch.expression);

	set<u256> cases;
	for (auto const& _case: _switch.cases)
	{
		if (_case.value)
		{
			auto watcher = m_errorReporter.errorWatcher();

			expectType(valueType, _case.value->type, _case.value->location);

			// We cannot use "expectExpression" here because *_case.value is not an
			// Expression and would be converted to an Expression otherwise.
			(*this)(*_case.value);

			/// Note: the parser ensures there is only one default case
			if (watcher.ok() && !cases.insert(valueOfLiteral(*_case.value)).second)
				m_errorReporter.declarationError(6792_error, _case.location, "Duplicate case defined.");
		}

		(*this)(_case.body);
	}
}

void AsmAnalyzer::operator()(ForLoop const& _for)
{
	yulAssert(_for.condition, "");

	Scope* outerScope = m_currentScope;

	(*this)(_for.pre);

	// The block was closed already, but we re-open it again and stuff the
	// condition, the body and the post part inside.
	m_currentScope = &scope(&_for.pre);

	expectBoolExpression(*_for.condition);
	// backup outer for-loop & create new state
	auto outerForLoop = m_currentForLoop;
	m_currentForLoop = &_for;

	(*this)(_for.body);
	(*this)(_for.post);

	m_currentScope = outerScope;
	m_currentForLoop = outerForLoop;
}

void AsmAnalyzer::operator()(Block const& _block)
{
	auto previousScope = m_currentScope;
	m_currentScope = &scope(&_block);

	for (auto const& s: _block.statements)
		std::visit(*this, s);

	m_currentScope = previousScope;
}

YulString AsmAnalyzer::expectExpression(Expression const& _expr)
{
	vector<YulString> types = std::visit(*this, _expr);
	if (types.size() != 1)
		m_errorReporter.typeError(
			3950_error,
			locationOf(_expr),
			"Expected expression to evaluate to one value, but got " +
			to_string(types.size()) +
			" values instead."
		);
	return types.empty() ? m_dialect.defaultType : types.front();
}

YulString AsmAnalyzer::expectUnlimitedStringLiteral(Literal const& _literal)
{
	yulAssert(_literal.kind == LiteralKind::String, "");
	yulAssert(m_dialect.validTypeForLiteral(LiteralKind::String, _literal.value, _literal.type), "");

	return {_literal.type};
}

void AsmAnalyzer::expectBoolExpression(Expression const& _expr)
{
	YulString type = expectExpression(_expr);
	if (type != m_dialect.boolType)
		m_errorReporter.typeError(
			1733_error,
			locationOf(_expr),
			"Expected a value of boolean type \"" +
			m_dialect.boolType.str() +
			"\" but got \"" +
			type.str() +
			"\""
		);
}

void AsmAnalyzer::checkAssignment(Identifier const& _variable, YulString _valueType)
{
	yulAssert(!_variable.name.empty(), "");
	auto watcher = m_errorReporter.errorWatcher();
	YulString const* variableType = nullptr;
	bool found = false;
	if (Scope::Identifier const* var = m_currentScope->lookup(_variable.name))
	{
		// Check that it is a variable
		if (!holds_alternative<Scope::Variable>(*var))
			m_errorReporter.typeError(2657_error, _variable.location, "Assignment requires variable.");
		else if (!m_activeVariables.count(&std::get<Scope::Variable>(*var)))
			m_errorReporter.declarationError(
				1133_error,
				_variable.location,
				"Variable " + _variable.name.str() + " used before it was declared."
			);
		else
			variableType = &std::get<Scope::Variable>(*var).type;
		found = true;
	}
	else if (m_resolver)
	{
		bool insideFunction = m_currentScope->insideFunction();
		if (m_resolver(_variable, yul::IdentifierContext::LValue, insideFunction))
		{
			found = true;
			variableType = &m_dialect.defaultType;
		}
	}

	if (!found && watcher.ok())
		// Only add message if the callback did not.
		m_errorReporter.declarationError(4634_error, _variable.location, "Variable not found or variable not lvalue.");
	if (variableType && *variableType != _valueType)
		m_errorReporter.typeError(
			9547_error,
			_variable.location,
			"Assigning a value of type \"" +
			_valueType.str() +
			"\" to a variable of type \"" +
			variableType->str() +
			"\"."
		);

	yulAssert(!watcher.ok() || variableType, "");
}

Scope& AsmAnalyzer::scope(Block const* _block)
{
	yulAssert(m_info.scopes.count(_block) == 1, "Scope requested but not present.");
	auto scopePtr = m_info.scopes.at(_block);
	yulAssert(scopePtr, "Scope requested but not present.");
	return *scopePtr;
}

void AsmAnalyzer::expectValidType(YulString _type, SourceLocation const& _location)
{
	if (!m_dialect.types.count(_type))
		m_errorReporter.typeError(
			5473_error,
			_location,
			"\"" + _type.str() + "\" is not a valid type (user defined types are not yet supported)."
		);
}

void AsmAnalyzer::expectType(YulString _expectedType, YulString _givenType, SourceLocation const& _location)
{
	if (_expectedType != _givenType)
		m_errorReporter.typeError(
			3781_error,
			_location,
			"Expected a value of type \"" +
			_expectedType.str() +
			"\" but got \"" +
			_givenType.str() +
			"\""
		);
}

bool AsmAnalyzer::validateInstructions(std::string const& _instructionIdentifier, langutil::SourceLocation const& _location)
{
	auto const builtin = EVMDialect::strictAssemblyForEVM(EVMVersion{}).builtin(YulString(_instructionIdentifier));
	if (builtin && builtin->instruction.has_value())
		return validateInstructions(builtin->instruction.value(), _location);
	else
		return false;
}

bool AsmAnalyzer::validateInstructions(evmasm::Instruction _instr, SourceLocation const& _location)
{
	// We assume that returndatacopy, returndatasize and staticcall are either all available
	// or all not available.
	yulAssert(m_evmVersion.supportsReturndata() == m_evmVersion.hasStaticCall(), "");
	// Similarly we assume bitwise shifting and create2 go together.
	yulAssert(m_evmVersion.hasBitwiseShifting() == m_evmVersion.hasCreate2(), "");

	// These instructions are disabled in the dialect.
	yulAssert(
		_instr != evmasm::Instruction::JUMP &&
		_instr != evmasm::Instruction::JUMPI &&
		_instr != evmasm::Instruction::JUMPDEST,
	"");

	auto errorForVM = [&](ErrorId _errorId, string const& vmKindMessage) {
		m_errorReporter.typeError(
			_errorId,
			_location,
			"The \"" +
			boost::to_lower_copy(instructionInfo(_instr).name)
			+ "\" instruction is " +
			vmKindMessage +
			" VMs " +
			"(you are currently compiling for \"" +
			m_evmVersion.name() +
			"\")."
		);
	};

	if ((
		_instr == evmasm::Instruction::RETURNDATACOPY ||
		_instr == evmasm::Instruction::RETURNDATASIZE
	) && !m_evmVersion.supportsReturndata())
		errorForVM(7756_error, "only available for Byzantium-compatible");
	else if (_instr == evmasm::Instruction::STATICCALL && !m_evmVersion.hasStaticCall())
		errorForVM(1503_error, "only available for Byzantium-compatible");
	else if ((
		_instr == evmasm::Instruction::SHL ||
		_instr == evmasm::Instruction::SHR ||
		_instr == evmasm::Instruction::SAR
	) && !m_evmVersion.hasBitwiseShifting())
		errorForVM(6612_error, "only available for Constantinople-compatible");
	else if (_instr == evmasm::Instruction::CREATE2 && !m_evmVersion.hasCreate2())
		errorForVM(6166_error, "only available for Constantinople-compatible");
	else if (_instr == evmasm::Instruction::EXTCODEHASH && !m_evmVersion.hasExtCodeHash())
		errorForVM(7110_error, "only available for Constantinople-compatible");
	else if (_instr == evmasm::Instruction::CHAINID && !m_evmVersion.hasChainID())
		errorForVM(1561_error, "only available for Istanbul-compatible");
	else if (_instr == evmasm::Instruction::PC)
		m_errorReporter.warning(
			2450_error,
			_location,
			"The \"" +
			boost::to_lower_copy(instructionInfo(_instr).name) +
			"\" instruction is deprecated and will be removed in the next breaking release."
		);
	else if (_instr == evmasm::Instruction::SELFBALANCE && !m_evmVersion.hasSelfBalance())
		errorForVM(3672_error, "only available for Istanbul-compatible");
	else
		return false;

	return true;
}
