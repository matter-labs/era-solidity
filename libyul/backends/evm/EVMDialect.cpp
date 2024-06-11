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
 * Yul dialects for EVM.
 */

#include <libyul/backends/evm/EVMDialect.h>

#include <libevmasm/Instruction.h>
#include <libevmasm/SemanticInformation.h>
#include <liblangutil/Exceptions.h>
#include <libsolutil/StringUtils.h>
#include <libyul/AST.h>
#include <libyul/AsmAnalysisInfo.h>
#include <libyul/AsmParser.h>
#include <libyul/Exceptions.h>
#include <libyul/Object.h>
#include <libyul/Utilities.h>
#include <libyul/backends/evm/AbstractAssembly.h>
#include <libyul/backends/evm/ZKEVMIntrinsics.h>

#include <range/v3/view/reverse.hpp>
#include <range/v3/view/tail.hpp>
#include <range/v3/view/zip.hpp>

#include <regex>

using namespace std;
using namespace solidity;
using namespace solidity::yul;
using namespace solidity::util;

namespace
{

pair<YulString, BuiltinFunctionForEVM> createEVMFunction(
	langutil::EVMVersion _evmVersion,
	string const& _name,
	evmasm::Instruction _instruction
)
{
	evmasm::InstructionInfo info = evmasm::instructionInfo(_instruction, _evmVersion);
	BuiltinFunctionForEVM f;
	f.name = YulString{_name};
	f.parameters.resize(static_cast<size_t>(info.args));
	f.returns.resize(static_cast<size_t>(info.ret));
	f.sideEffects = EVMDialect::sideEffectsOfInstruction(_instruction);
	if (evmasm::SemanticInformation::terminatesControlFlow(_instruction))
	{
		f.controlFlowSideEffects.canContinue = false;
		if (evmasm::SemanticInformation::reverts(_instruction))
		{
			f.controlFlowSideEffects.canTerminate = false;
			f.controlFlowSideEffects.canRevert = true;
		}
		else
		{
			f.controlFlowSideEffects.canTerminate = true;
			f.controlFlowSideEffects.canRevert = false;
		}
	}
	f.isMSize = _instruction == evmasm::Instruction::MSIZE;
	f.literalArguments.clear();
	f.instruction = _instruction;
	f.generateCode = [_instruction](
		FunctionCall const&,
		AbstractAssembly& _assembly,
		BuiltinContext&
	) {
		_assembly.appendInstruction(_instruction);
	};

	YulString name = f.name;
	return {name, std::move(f)};
}

pair<YulString, BuiltinFunctionForEVM> createFunction(
	string _name,
	size_t _params,
	size_t _returns,
	SideEffects _sideEffects,
	vector<optional<LiteralKind>> _literalArguments,
	std::function<void(FunctionCall const&, AbstractAssembly&, BuiltinContext&)> _generateCode
)
{
	yulAssert(_literalArguments.size() == _params || _literalArguments.empty(), "");

	YulString name{std::move(_name)};
	BuiltinFunctionForEVM f;
	f.name = name;
	f.parameters.resize(_params);
	f.returns.resize(_returns);
	f.sideEffects = std::move(_sideEffects);
	f.literalArguments = std::move(_literalArguments);
	f.isMSize = false;
	f.instruction = {};
	f.generateCode = std::move(_generateCode);
	return {name, f};
}

set<YulString> createReservedIdentifiers(langutil::EVMVersion _evmVersion)
{
	// TODO remove this in 0.9.0. We allow creating functions or identifiers in Yul with the name
	// basefee for VMs before london.
	auto baseFeeException = [&](evmasm::Instruction _instr) -> bool
	{
		return _instr == evmasm::Instruction::BASEFEE && _evmVersion < langutil::EVMVersion::london();
	};

	// TODO remove this in 0.9.0. We allow creating functions or identifiers in Yul with the name
	// prevrandao for VMs before paris.
	auto prevRandaoException = [&](string const& _instrName) -> bool
	{
		// Using string comparison as the opcode is the same as for "difficulty"
		return _instrName == "prevrandao" && _evmVersion < langutil::EVMVersion::paris();
	};

	set<YulString> reserved;
	for (auto const& instr: evmasm::c_instructions)
	{
		string name = toLower(instr.first);
		if (!baseFeeException(instr.second) && !prevRandaoException(name))
			reserved.emplace(name);
	}
	for (auto const& intr: solidity::zkevm::intrInfos)
	{
		reserved.emplace(intr.name);
	}
	reserved += vector<YulString>{
		"linkersymbol"_yulstring,
		"datasize"_yulstring,
		"dataoffset"_yulstring,
		"datacopy"_yulstring,
		"setimmutable"_yulstring,
		"loadimmutable"_yulstring,
	};
	return reserved;
}

pair<YulString, BuiltinFunctionForEVM> createVerbatimWrapper(
	const std::string& _name,
	size_t _params,
	size_t _returns,
	bool _sideEffects,
	const vector<optional<LiteralKind>>& _literalKinds)
{
	SideEffects sideEffects{};
	if (_sideEffects)
	{
		sideEffects
			= {/*movable=*/false,
			   /*movableApartFromEffects=*/false,
			   /*canBeRemoved=*/false,
			   /*canBeRemovedIfNoMSize=*/false,
			   /*cannotLoop=*/true,
			   /*otherState=*/SideEffects::Effect::Write,
			   /*storage=*/SideEffects::Effect::Write,
			   /*memory=*/SideEffects::Effect::Write};
	}

	std::function<void(FunctionCall const&, AbstractAssembly&, BuiltinContext&)> genCode;
	if (!_literalKinds.empty())
	{
		genCode = [=](FunctionCall const& _call, AbstractAssembly& _assembly, BuiltinContext&)
		{
			yulAssert(_call.arguments.size() == _literalKinds.size(), "");
			size_t numLits = 0;
			for (const auto&& [arg, kind]: ranges::views::zip(_call.arguments, _literalKinds))
			{
				if (!kind)
					continue;

				yulAssert(holds_alternative<Literal>(arg), "Expected literal");
				yulAssert(get<Literal>(arg).kind == kind, "Unexpected literal kind");
				numLits++;
			}

			_assembly.appendVerbatim(asBytes(_name), _params - numLits, _returns);
		};
	}
	else
	{
		genCode = [=](FunctionCall const&, AbstractAssembly& _assembly, BuiltinContext&)
		{ _assembly.appendVerbatim(asBytes(_name), _params, _returns); };
	}

	return createFunction( _name, _params, _returns, sideEffects, _literalKinds, genCode);
}

map<YulString, BuiltinFunctionForEVM> createBuiltins(langutil::EVMVersion _evmVersion, bool _objectAccess)
{

	// Exclude prevrandao as builtin for VMs before paris and difficulty for VMs after paris.
	auto prevRandaoException = [&](string const& _instrName) -> bool
	{
		return (_instrName == "prevrandao" && _evmVersion < langutil::EVMVersion::paris()) || (_instrName == "difficulty" && _evmVersion >= langutil::EVMVersion::paris());
	};

	map<YulString, BuiltinFunctionForEVM> builtins;
	for (auto const& instr: evmasm::c_instructions)
	{
		string name = toLower(instr.first);
		auto const opcode = instr.second;

		if (
			!evmasm::isDupInstruction(opcode) &&
			!evmasm::isSwapInstruction(opcode) &&
			!evmasm::isPushInstruction(opcode) &&
			opcode != evmasm::Instruction::JUMP &&
			opcode != evmasm::Instruction::JUMPI &&
			opcode != evmasm::Instruction::JUMPDEST &&
			_evmVersion.hasOpcode(opcode) &&
			!prevRandaoException(name)
		)
			builtins.emplace(createEVMFunction(_evmVersion, name, opcode));
	}

	for (auto const& intr: solidity::zkevm::intrInfos)
	{
		builtins.emplace(
			createVerbatimWrapper(intr.name, intr.args, intr.ret, intr.sideEffects, intr.literalKinds));
	}

	if (_objectAccess)
	{
		builtins.emplace(createFunction("linkersymbol", 1, 1, SideEffects{}, {LiteralKind::String}, [](
			FunctionCall const& _call,
			AbstractAssembly& _assembly,
			BuiltinContext&
		) {
			yulAssert(_call.arguments.size() == 1, "");
			Expression const& arg = _call.arguments.front();
			_assembly.appendLinkerSymbol(std::get<Literal>(arg).value.str());
		}));

		builtins.emplace(createFunction(
			"memoryguard",
			1,
			1,
			SideEffects{},
			{LiteralKind::Number},
			[](
				FunctionCall const& _call,
				AbstractAssembly& _assembly,
				BuiltinContext&
			) {
				yulAssert(_call.arguments.size() == 1, "");
				Literal const* literal = get_if<Literal>(&_call.arguments.front());
				yulAssert(literal, "");
				_assembly.appendConstant(valueOfLiteral(*literal));
			})
		);

		builtins.emplace(createFunction("$zk_datasize", 1, 1, SideEffects{}, {LiteralKind::String}, [](
			FunctionCall const& _call,
			AbstractAssembly& _assembly,
			BuiltinContext& _context
		) {
			yulAssert(_context.currentObject, "No object available.");
			yulAssert(_call.arguments.size() == 1, "");
			Expression const& arg = _call.arguments.front();
			YulString dataName = std::get<Literal>(arg).value;
			if (_context.currentObject->name == dataName)
				_assembly.appendAssemblySize();
			else
			{
				vector<size_t> subIdPath =
					_context.subIDs.count(dataName) == 0 ?
						_context.currentObject->pathToSubObject(dataName) :
						vector<size_t>{_context.subIDs.at(dataName)};
				yulAssert(!subIdPath.empty(), "Could not find assembly object <" + dataName.str() + ">.");
				_assembly.appendZKEVMDataSize(subIdPath);
			}
		}));
		builtins.emplace(createFunction("$zk_dataoffset", 1, 1, SideEffects{}, {LiteralKind::String}, [](
			FunctionCall const& _call,
			AbstractAssembly& _assembly,
			BuiltinContext& _context
		) {
			yulAssert(_context.currentObject, "No object available.");
			yulAssert(_call.arguments.size() == 1, "");
			Expression const& arg = _call.arguments.front();
			YulString dataName = std::get<Literal>(arg).value;
			if (_context.currentObject->name == dataName)
				_assembly.appendConstant(0);
			else
			{
				vector<size_t> subIdPath =
					_context.subIDs.count(dataName) == 0 ?
						_context.currentObject->pathToSubObject(dataName) :
						vector<size_t>{_context.subIDs.at(dataName)};
				yulAssert(!subIdPath.empty(), "Could not find assembly object <" + dataName.str() + ">.");
				_assembly.appendZKEVMDataOffset(subIdPath);
			}
		}));

		builtins.emplace(createFunction("datasize", 1, 1, SideEffects{}, {LiteralKind::String}, [](
			FunctionCall const& _call,
			AbstractAssembly& _assembly,
			BuiltinContext& _context
		) {
			yulAssert(_context.currentObject, "No object available.");
			yulAssert(_call.arguments.size() == 1, "");
			Expression const& arg = _call.arguments.front();
			YulString dataName = std::get<Literal>(arg).value;
			if (_context.currentObject->name == dataName)
				_assembly.appendAssemblySize();
			else
			{
				vector<size_t> subIdPath =
					_context.subIDs.count(dataName) == 0 ?
						_context.currentObject->pathToSubObject(dataName) :
						vector<size_t>{_context.subIDs.at(dataName)};
				yulAssert(!subIdPath.empty(), "Could not find assembly object <" + dataName.str() + ">.");
				_assembly.appendDataSize(subIdPath);
			}
		}));
		builtins.emplace(createFunction("dataoffset", 1, 1, SideEffects{}, {LiteralKind::String}, [](
			FunctionCall const& _call,
			AbstractAssembly& _assembly,
			BuiltinContext& _context
		) {
			yulAssert(_context.currentObject, "No object available.");
			yulAssert(_call.arguments.size() == 1, "");
			Expression const& arg = _call.arguments.front();
			YulString dataName = std::get<Literal>(arg).value;
			if (_context.currentObject->name == dataName)
				_assembly.appendConstant(0);
			else
			{
				vector<size_t> subIdPath =
					_context.subIDs.count(dataName) == 0 ?
						_context.currentObject->pathToSubObject(dataName) :
						vector<size_t>{_context.subIDs.at(dataName)};
				yulAssert(!subIdPath.empty(), "Could not find assembly object <" + dataName.str() + ">.");
				_assembly.appendDataOffset(subIdPath);
			}
		}));
		builtins.emplace(createFunction(
			"datacopy",
			3,
			0,
			SideEffects{false, true, false, false, true, SideEffects::None, SideEffects::None, SideEffects::Write},
			{},
			[](
				FunctionCall const&,
				AbstractAssembly& _assembly,
				BuiltinContext&
			) {
				_assembly.appendInstruction(evmasm::Instruction::CODECOPY);
			}
		));
		builtins.emplace(createFunction(
			"setimmutable",
			3,
			0,
			SideEffects{false, false, false, false, true, SideEffects::None, SideEffects::None, SideEffects::Write},
			{std::nullopt, LiteralKind::String, std::nullopt},
			[](
				FunctionCall const& _call,
				AbstractAssembly& _assembly,
				BuiltinContext&
			) {
				yulAssert(_call.arguments.size() == 3, "");
				YulString identifier = std::get<Literal>(_call.arguments[1]).value;
				_assembly.appendImmutableAssignment(identifier.str());
			}
		));
		builtins.emplace(createFunction(
			"loadimmutable",
			1,
			1,
			SideEffects{},
			{LiteralKind::String},
			[](
				FunctionCall const& _call,
				AbstractAssembly& _assembly,
				BuiltinContext&
			) {
				yulAssert(_call.arguments.size() == 1, "");
				_assembly.appendImmutable(std::get<Literal>(_call.arguments.front()).value.str());
			}
		));
	}
	return builtins;
}

regex const& verbatimPattern()
{
	regex static const pattern{"verbatim_([1-9]?[0-9])i_([1-9]?[0-9])o"};
	return pattern;
}

}


EVMDialect::EVMDialect(langutil::EVMVersion _evmVersion, bool _objectAccess):
	m_objectAccess(_objectAccess),
	m_evmVersion(_evmVersion),
	m_functions(createBuiltins(_evmVersion, _objectAccess)),
	m_reserved(createReservedIdentifiers(_evmVersion))
{
}

BuiltinFunctionForEVM const* EVMDialect::builtin(YulString _name) const
{
	if (m_objectAccess)
	{
		smatch match;
		if (regex_match(_name.str(), match, verbatimPattern()))
			return verbatimFunction(stoul(match[1]), stoul(match[2]));
	}
	auto it = m_functions.find(_name);
	if (it != m_functions.end())
		return &it->second;
	else
		return nullptr;
}

bool EVMDialect::reservedIdentifier(YulString _name) const
{
	if (m_objectAccess)
		if (_name.str().substr(0, "verbatim"s.size()) == "verbatim")
			return true;
	return m_reserved.count(_name) != 0;
}

EVMDialect const& EVMDialect::strictAssemblyForEVM(langutil::EVMVersion _version)
{
	static map<langutil::EVMVersion, unique_ptr<EVMDialect const>> dialects;
	static YulStringRepository::ResetCallback callback{[&] { dialects.clear(); }};
	if (!dialects[_version])
		dialects[_version] = make_unique<EVMDialect>(_version, false);
	return *dialects[_version];
}

EVMDialect const& EVMDialect::strictAssemblyForEVMObjects(langutil::EVMVersion _version)
{
	static map<langutil::EVMVersion, unique_ptr<EVMDialect const>> dialects;
	static YulStringRepository::ResetCallback callback{[&] { dialects.clear(); }};
	if (!dialects[_version])
		dialects[_version] = make_unique<EVMDialect>(_version, true);
	return *dialects[_version];
}

SideEffects EVMDialect::sideEffectsOfInstruction(evmasm::Instruction _instruction)
{
	auto translate = [](evmasm::SemanticInformation::Effect _e) -> SideEffects::Effect
	{
		return static_cast<SideEffects::Effect>(_e);
	};

	return SideEffects{
		evmasm::SemanticInformation::movable(_instruction),
		evmasm::SemanticInformation::movableApartFromEffects(_instruction),
		evmasm::SemanticInformation::canBeRemoved(_instruction),
		evmasm::SemanticInformation::canBeRemovedIfNoMSize(_instruction),
		true, // cannotLoop
		translate(evmasm::SemanticInformation::otherState(_instruction)),
		translate(evmasm::SemanticInformation::storage(_instruction)),
		translate(evmasm::SemanticInformation::memory(_instruction)),
	};
}

BuiltinFunctionForEVM const* EVMDialect::verbatimFunction(size_t _arguments, size_t _returnVariables) const
{
	pair<size_t, size_t> key{_arguments, _returnVariables};
	shared_ptr<BuiltinFunctionForEVM const>& function = m_verbatimFunctions[key];
	if (!function)
	{
		BuiltinFunctionForEVM builtinFunction = createFunction(
			"verbatim_" + to_string(_arguments) + "i_" + to_string(_returnVariables) + "o",
			1 + _arguments,
			_returnVariables,
			SideEffects::worst(),
			vector<optional<LiteralKind>>{LiteralKind::String} + vector<optional<LiteralKind>>(_arguments),
			[=](
				FunctionCall const& _call,
				AbstractAssembly& _assembly,
				BuiltinContext&
			) {
				yulAssert(_call.arguments.size() == (1 + _arguments), "");
				Expression const& bytecode = _call.arguments.front();

				_assembly.appendVerbatim(
					asBytes(std::get<Literal>(bytecode).value.str()),
					_arguments,
					_returnVariables
				);
			}
		).second;
		builtinFunction.isMSize = true;
		function = make_shared<BuiltinFunctionForEVM const>(std::move(builtinFunction));
	}
	return function.get();
}

EVMDialectTyped::EVMDialectTyped(langutil::EVMVersion _evmVersion, bool _objectAccess):
	EVMDialect(_evmVersion, _objectAccess)
{
	defaultType = "u256"_yulstring;
	boolType = "bool"_yulstring;
	types = {defaultType, boolType};

	// Set all types to ``defaultType``
	for (auto& fun: m_functions)
	{
		for (auto& p: fun.second.parameters)
			p = defaultType;
		for (auto& r: fun.second.returns)
			r = defaultType;
	}

	m_functions["lt"_yulstring].returns = {"bool"_yulstring};
	m_functions["gt"_yulstring].returns = {"bool"_yulstring};
	m_functions["slt"_yulstring].returns = {"bool"_yulstring};
	m_functions["sgt"_yulstring].returns = {"bool"_yulstring};
	m_functions["eq"_yulstring].returns = {"bool"_yulstring};

	// "not" and "bitnot" replace "iszero" and "not"
	m_functions["bitnot"_yulstring] = m_functions["not"_yulstring];
	m_functions["bitnot"_yulstring].name = "bitnot"_yulstring;
	m_functions["not"_yulstring] = m_functions["iszero"_yulstring];
	m_functions["not"_yulstring].name = "not"_yulstring;
	m_functions["not"_yulstring].returns = {"bool"_yulstring};
	m_functions["not"_yulstring].parameters = {"bool"_yulstring};
	m_functions.erase("iszero"_yulstring);

	m_functions["bitand"_yulstring] = m_functions["and"_yulstring];
	m_functions["bitand"_yulstring].name = "bitand"_yulstring;
	m_functions["bitor"_yulstring] = m_functions["or"_yulstring];
	m_functions["bitor"_yulstring].name = "bitor"_yulstring;
	m_functions["bitxor"_yulstring] = m_functions["xor"_yulstring];
	m_functions["bitxor"_yulstring].name = "bitxor"_yulstring;
	m_functions["and"_yulstring].parameters = {"bool"_yulstring, "bool"_yulstring};
	m_functions["and"_yulstring].returns = {"bool"_yulstring};
	m_functions["or"_yulstring].parameters = {"bool"_yulstring, "bool"_yulstring};
	m_functions["or"_yulstring].returns = {"bool"_yulstring};
	m_functions["xor"_yulstring].parameters = {"bool"_yulstring, "bool"_yulstring};
	m_functions["xor"_yulstring].returns = {"bool"_yulstring};
	m_functions["popbool"_yulstring] = m_functions["pop"_yulstring];
	m_functions["popbool"_yulstring].name = "popbool"_yulstring;
	m_functions["popbool"_yulstring].parameters = {"bool"_yulstring};
	m_functions.insert(createFunction("bool_to_u256", 1, 1, {}, {}, [](
		FunctionCall const&,
		AbstractAssembly&,
		BuiltinContext&
	) {}));
	m_functions["bool_to_u256"_yulstring].parameters = {"bool"_yulstring};
	m_functions["bool_to_u256"_yulstring].returns = {"u256"_yulstring};
	m_functions.insert(createFunction("u256_to_bool", 1, 1, {}, {}, [](
		FunctionCall const&,
		AbstractAssembly& _assembly,
		BuiltinContext&
	) {
		// TODO this should use a Panic.
		// A value larger than 1 causes an invalid instruction.
		_assembly.appendConstant(2);
		_assembly.appendInstruction(evmasm::Instruction::DUP2);
		_assembly.appendInstruction(evmasm::Instruction::LT);
		AbstractAssembly::LabelID inRange = _assembly.newLabelId();
		_assembly.appendJumpToIf(inRange);
		_assembly.appendInstruction(evmasm::Instruction::INVALID);
		_assembly.appendLabel(inRange);
	}));
	m_functions["u256_to_bool"_yulstring].parameters = {"u256"_yulstring};
	m_functions["u256_to_bool"_yulstring].returns = {"bool"_yulstring};

	m_functions["$zk_to_l1"_yulstring].parameters = {boolType, defaultType, defaultType};
}

BuiltinFunctionForEVM const* EVMDialectTyped::discardFunction(YulString _type) const
{
	if (_type == "bool"_yulstring)
		return builtin("popbool"_yulstring);
	else
	{
		yulAssert(_type == defaultType, "");
		return builtin("pop"_yulstring);
	}
}

BuiltinFunctionForEVM const* EVMDialectTyped::equalityFunction(YulString _type) const
{
	if (_type == "bool"_yulstring)
		return nullptr;
	else
	{
		yulAssert(_type == defaultType, "");
		return builtin("eq"_yulstring);
	}
}

EVMDialectTyped const& EVMDialectTyped::instance(langutil::EVMVersion _version)
{
	static map<langutil::EVMVersion, unique_ptr<EVMDialectTyped const>> dialects;
	static YulStringRepository::ResetCallback callback{[&] { dialects.clear(); }};
	if (!dialects[_version])
		dialects[_version] = make_unique<EVMDialectTyped>(_version, true);
	return *dialects[_version];
}
