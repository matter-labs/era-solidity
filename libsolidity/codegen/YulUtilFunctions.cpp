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
 * Component that can generate various useful Yul functions.
 */

#include <libsolidity/codegen/YulUtilFunctions.h>

#include <libsolidity/codegen/MultiUseYulFunctionCollector.h>
#include <libsolidity/ast/AST.h>
#include <libsolidity/codegen/CompilerUtils.h>
#include <libsolidity/codegen/ir/IRVariable.h>

#include <libsolutil/CommonData.h>
#include <libsolutil/FunctionSelector.h>
#include <libsolutil/Whiskers.h>
#include <libsolutil/StringUtils.h>
#include <libsolidity/ast/TypeProvider.h>

#include <range/v3/algorithm/all_of.hpp>

using namespace solidity;
using namespace solidity::util;
using namespace solidity::frontend;
using namespace std::string_literals;

namespace
{

std::optional<size_t> staticEncodingSize(std::vector<Type const*> const& _parameterTypes)
{
	size_t encodedSize = 0;
	for (auto* type: _parameterTypes)
	{
		if (type->isDynamicallyEncoded())
			return std::nullopt;
		encodedSize += type->calldataHeadSize();
	}
	return encodedSize;
}

}

std::string YulUtilFunctions::identityFunction()
{
	std::string functionName = "identity";
	return m_functionCollector.createFunction("identity", [&](std::vector<std::string>& _args, std::vector<std::string>& _rets) {
		_args.push_back("value");
		_rets.push_back("ret");
		return "ret := value";
	});
}

std::string YulUtilFunctions::combineExternalFunctionIdFunction()
{
	std::string functionName = "combine_external_function_id";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(addr, selector) -> combined {
				combined := <shl64>(or(<shl32>(addr), and(selector, 0xffffffff)))
			}
		)")
		("functionName", functionName)
		("shl32", shiftLeftFunction(32))
		("shl64", shiftLeftFunction(64))
		.render();
	});
}

std::string YulUtilFunctions::splitExternalFunctionIdFunction()
{
	std::string functionName = "split_external_function_id";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(combined) -> addr, selector {
				combined := <shr64>(combined)
				selector := and(combined, 0xffffffff)
				addr := <shr32>(combined)
			}
		)")
		("functionName", functionName)
		("shr32", shiftRightFunction(32))
		("shr64", shiftRightFunction(64))
		.render();
	});
}

std::string YulUtilFunctions::copyToMemoryFunction(bool _fromCalldata, bool _cleanup)
{
	std::string functionName =
		"copy_"s +
		(_fromCalldata ? "calldata"s : "memory"s) +
		"_to_memory"s +
		(_cleanup ? "_with_cleanup"s : ""s);

	return m_functionCollector.createFunction(functionName, [&](std::vector<std::string>& _args, std::vector<std::string>&) {
		_args = {"src", "dst", "length"};

		if (_fromCalldata)
			return Whiskers(R"(
				calldatacopy(dst, src, length)
				<?cleanup>mstore(add(dst, length), 0)</cleanup>
			)")
			("cleanup", _cleanup)
			.render();
		else
		{
			if (m_evmVersion.hasMcopy())
				return Whiskers(R"(
					mcopy(dst, src, length)
					<?cleanup>mstore(add(dst, length), 0)</cleanup>
				)")
				("cleanup", _cleanup)
				.render();
			else
				return Whiskers(R"(
					let i := 0
					for { } lt(i, length) { i := add(i, 32) }
					{
						mstore(add(dst, i), mload(add(src, i)))
					}
					<?cleanup>mstore(add(dst, length), 0)</cleanup>
				)")
				("cleanup", _cleanup)
				.render();
		}
	});
}

std::string YulUtilFunctions::copyLiteralToMemoryFunction(std::string const& _literal)
{
	std::string functionName = "copy_literal_to_memory_" + util::toHex(util::keccak256(_literal).asBytes());

	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>() -> memPtr {
				memPtr := <arrayAllocationFunction>(<size>)
				<storeLiteralInMem>(add(memPtr, 32))
			}
			)")
			("functionName", functionName)
			("arrayAllocationFunction", allocateMemoryArrayFunction(*TypeProvider::array(DataLocation::Memory, true)))
			("size", std::to_string(_literal.size()))
			("storeLiteralInMem", storeLiteralInMemoryFunction(_literal))
			.render();
	});
}

std::string YulUtilFunctions::storeLiteralInMemoryFunction(std::string const& _literal)
{
	std::string functionName = "store_literal_in_memory_" + util::toHex(util::keccak256(_literal).asBytes());

	return m_functionCollector.createFunction(functionName, [&]() {
		size_t words = (_literal.length() + 31) / 32;
		std::vector<std::map<std::string, std::string>> wordParams(words);
		for (size_t i = 0; i < words; ++i)
		{
			wordParams[i]["offset"] = std::to_string(i * 32);
			wordParams[i]["wordValue"] = formatAsStringOrNumber(_literal.substr(32 * i, 32));
		}

		return Whiskers(R"(
			function <functionName>(memPtr) {
				<#word>
					mstore(add(memPtr, <offset>), <wordValue>)
				</word>
			}
			)")
			("functionName", functionName)
			("word", wordParams)
			.render();
	});
}

std::string YulUtilFunctions::copyLiteralToStorageFunction(std::string const& _literal)
{
	std::string functionName = "copy_literal_to_storage_" + util::toHex(util::keccak256(_literal).asBytes());

	return m_functionCollector.createFunction(functionName, [&](std::vector<std::string>& _args, std::vector<std::string>&) {
		_args = {"slot"};

		if (_literal.size() >= 32)
		{
			size_t words = (_literal.length() + 31) / 32;
			std::vector<std::map<std::string, std::string>> wordParams(words);
			for (size_t i = 0; i < words; ++i)
			{
				wordParams[i]["offset"] = std::to_string(i);
				wordParams[i]["wordValue"] = formatAsStringOrNumber(_literal.substr(32 * i, 32));
			}
			return Whiskers(R"(
				let oldLen := <byteArrayLength>(sload(slot))
				<cleanUpArrayEnd>(slot, oldLen, <length>)
				sstore(slot, <encodedLen>)
				let dstPtr := <dataArea>(slot)
				<#word>
					sstore(add(dstPtr, <offset>), <wordValue>)
				</word>
			)")
			("byteArrayLength", extractByteArrayLengthFunction())
			("cleanUpArrayEnd", cleanUpDynamicByteArrayEndSlotsFunction(*TypeProvider::bytesStorage()))
			("dataArea", arrayDataAreaFunction(*TypeProvider::bytesStorage()))
			("word", wordParams)
			("length", std::to_string(_literal.size()))
			("encodedLen", std::to_string(2 * _literal.size() + 1))
			.render();
		}
		else
			return Whiskers(R"(
				let oldLen := <byteArrayLength>(sload(slot))
				<cleanUpArrayEnd>(slot, oldLen, <length>)
				sstore(slot, add(<wordValue>, <encodedLen>))
			)")
			("byteArrayLength", extractByteArrayLengthFunction())
			("cleanUpArrayEnd", cleanUpDynamicByteArrayEndSlotsFunction(*TypeProvider::bytesStorage()))
			("wordValue", formatAsStringOrNumber(_literal))
			("length", std::to_string(_literal.size()))
			("encodedLen", std::to_string(2 * _literal.size()))
			.render();
	});
}

std::string YulUtilFunctions::revertWithError(
	std::string const& _signature,
	std::vector<Type const*> const& _parameterTypes,
	std::vector<ASTPointer<Expression const>> const& _errorArguments,
	std::string const& _posVar,
	std::string const& _endVar
)
{
	solAssert((!_posVar.empty() && !_endVar.empty()) || (_posVar.empty() && _endVar.empty()));
	bool const needsNewVariable = !_posVar.empty() && !_endVar.empty();
	bool needsAllocation = true;

	if (std::optional<size_t> size = staticEncodingSize(_parameterTypes))
		if (
			ranges::all_of(_parameterTypes, [](auto const* type) {
				solAssert(!dynamic_cast<InaccessibleDynamicType const*>(type));
				return type && type->isValueType();
			})
		)
		{
			constexpr size_t errorSelectorSize = 4;
			needsAllocation = *size + errorSelectorSize > CompilerUtils::generalPurposeMemoryStart;
		}

	Whiskers templ(R"({
		<?needsAllocation>
		let <pos> := <allocateUnbounded>()
		<!needsAllocation>
		let <pos> := 0
		</needsAllocation>
		mstore(<pos>, <hash>)
		let <end> := <encode>(add(<pos>, 4) <argumentVars>)
		revert(<pos>, sub(<end>, <pos>))
	})");
	templ("pos", needsNewVariable ? _posVar : "memPtr");
	templ("end", needsNewVariable ? _endVar : "end");
	templ("hash", formatNumber(util::selectorFromSignatureU256(_signature)));
	templ("needsAllocation", needsAllocation);
	if (needsAllocation)
		templ("allocateUnbounded", allocateUnboundedFunction());

	std::vector<std::string> errorArgumentVars;
	std::vector<Type const*> errorArgumentTypes;
	for (ASTPointer<Expression const> const& arg: _errorArguments)
	{
		errorArgumentVars += IRVariable(*arg).stackSlots();
		solAssert(arg->annotation().type);
		errorArgumentTypes.push_back(arg->annotation().type);
	}
	templ("argumentVars", joinHumanReadablePrefixed(errorArgumentVars));
	templ("encode", ABIFunctions(m_evmVersion, m_revertStrings, m_functionCollector).tupleEncoder(errorArgumentTypes, _parameterTypes));

	return templ.render();
}

std::string YulUtilFunctions::requireOrAssertFunction(bool _assert, Type const* _messageType, ASTPointer<Expression const> _stringArgumentExpression)
{
	std::string functionName =
		std::string(_assert ? "assert_helper" : "require_helper") +
		(_messageType ? ("_" + _messageType->identifier()) : "");

	solAssert(!_assert || !_messageType, "Asserts can't have messages!");

	return m_functionCollector.createFunction(functionName, [&]() {
		if (!_messageType)
			return Whiskers(R"(
				function <functionName>(condition) {
					if iszero(condition) { <error> }
				}
			)")
			("error", _assert ? panicFunction(PanicCode::Assert) + "()" : "revert(0, 0)")
			("functionName", functionName)
			.render();

		solAssert(_stringArgumentExpression, "Require with string must have a string argument");
		solAssert(_stringArgumentExpression->annotation().type);
		std::vector<std::string> functionParameterNames = IRVariable(*_stringArgumentExpression).stackSlots();

		return Whiskers(R"(
			function <functionName>(condition <functionParameterNames>) {
				if iszero(condition)
					<revertWithError>
			}
		)")
		("functionName", functionName)
		("revertWithError", revertWithError("Error(string)", {TypeProvider::stringMemory()}, {_stringArgumentExpression}))
		("functionParameterNames", joinHumanReadablePrefixed(functionParameterNames))
		.render();
	});
}

std::string YulUtilFunctions::requireWithErrorFunction(FunctionCall const& errorConstructorCall)
{
	ErrorDefinition const* errorDefinition = dynamic_cast<ErrorDefinition const*>(ASTNode::referencedDeclaration(errorConstructorCall.expression()));
	solAssert(errorDefinition);

	std::string const errorSignature = errorDefinition->functionType(true)->externalSignature();
	// Note that in most cases we'll always generate one function per error definition,
	// because types in the constructor call will match the ones in the definition. The only
	// exception are calls with types, where each instance has its own type (e.g. literals).
	std::string functionName = "require_helper_t_error_" + std::to_string(errorDefinition->id()) + "_" + errorDefinition->name();
	for (ASTPointer<Expression const> const& argument: errorConstructorCall.sortedArguments())
	{
		solAssert(argument->annotation().type);
		functionName += ("_" + argument->annotation().type->identifier());
	}

	std::vector<std::string> functionParameterNames;
	for (ASTPointer<Expression const> const& arg: errorConstructorCall.sortedArguments())
	{
		solAssert(arg->annotation().type);
		if (arg->annotation().type->sizeOnStack() > 0)
			functionParameterNames += IRVariable(*arg).stackSlots();
	}

	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(condition <functionParameterNames>) {
				if iszero(condition)
					<revertWithError>
			}
		)")
		("functionName", functionName)
		("functionParameterNames", joinHumanReadablePrefixed(functionParameterNames))
		// We're creating parameter names from the expressions passed into the constructor call,
		// which will result in odd names like `expr_29` that would normally be used for locals.
		// Note that this is the naming expected by `revertWithError()`.
		("revertWithError", revertWithError(errorSignature, errorDefinition->functionType(true)->parameterTypes(), errorConstructorCall.sortedArguments()))
		.render();
	});
}

std::string YulUtilFunctions::leftAlignFunction(Type const& _type)
{
	std::string functionName = std::string("leftAlign_") + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(value) -> aligned {
				<body>
			}
		)");
		templ("functionName", functionName);
		switch (_type.category())
		{
		case Type::Category::Address:
			templ("body", "aligned := " + leftAlignFunction(IntegerType(160)) + "(value)");
			break;
		case Type::Category::Integer:
		{
			IntegerType const& type = dynamic_cast<IntegerType const&>(_type);
			if (type.numBits() == 256)
				templ("body", "aligned := value");
			else
				templ("body", "aligned := " + shiftLeftFunction(256 - type.numBits()) + "(value)");
			break;
		}
		case Type::Category::RationalNumber:
			solAssert(false, "Left align requested for rational number.");
			break;
		case Type::Category::Bool:
			templ("body", "aligned := " + leftAlignFunction(IntegerType(8)) + "(value)");
			break;
		case Type::Category::FixedPoint:
			solUnimplemented("Fixed point types not implemented.");
			break;
		case Type::Category::Array:
		case Type::Category::Struct:
			solAssert(false, "Left align requested for non-value type.");
			break;
		case Type::Category::FixedBytes:
			templ("body", "aligned := value");
			break;
		case Type::Category::Contract:
			templ("body", "aligned := " + leftAlignFunction(*TypeProvider::address()) + "(value)");
			break;
		case Type::Category::Enum:
		{
			solAssert(dynamic_cast<EnumType const&>(_type).storageBytes() == 1, "");
			templ("body", "aligned := " + leftAlignFunction(IntegerType(8)) + "(value)");
			break;
		}
		case Type::Category::InaccessibleDynamic:
			solAssert(false, "Left align requested for inaccessible dynamic type.");
			break;
		default:
			solAssert(false, "Left align of type " + _type.identifier() + " requested.");
		}

		return templ.render();
	});
}

std::string YulUtilFunctions::shiftLeftFunction(size_t _numBits)
{
	solAssert(_numBits < 256, "");

	std::string functionName = "shift_left_" + std::to_string(_numBits);
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(value) -> newValue {
				newValue :=
				<?hasShifts>
					shl(<numBits>, value)
				<!hasShifts>
					mul(value, <multiplier>)
				</hasShifts>
			}
			)")
			("functionName", functionName)
			("numBits", std::to_string(_numBits))
			("hasShifts", m_evmVersion.hasBitwiseShifting())
			("multiplier", toCompactHexWithPrefix(u256(1) << _numBits))
			.render();
	});
}

std::string YulUtilFunctions::shiftLeftFunctionDynamic()
{
	std::string functionName = "shift_left_dynamic";
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(bits, value) -> newValue {
				newValue :=
				<?hasShifts>
					shl(bits, value)
				<!hasShifts>
					mul(value, exp(2, bits))
				</hasShifts>
			}
			)")
			("functionName", functionName)
			("hasShifts", m_evmVersion.hasBitwiseShifting())
			.render();
	});
}

std::string YulUtilFunctions::shiftRightFunction(size_t _numBits)
{
	solAssert(_numBits < 256, "");

	// Note that if this is extended with signed shifts,
	// the opcodes SAR and SDIV behave differently with regards to rounding!

	std::string functionName = "shift_right_" + std::to_string(_numBits) + "_unsigned";
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(value) -> newValue {
				newValue :=
				<?hasShifts>
					shr(<numBits>, value)
				<!hasShifts>
					div(value, <multiplier>)
				</hasShifts>
			}
			)")
			("functionName", functionName)
			("hasShifts", m_evmVersion.hasBitwiseShifting())
			("numBits", std::to_string(_numBits))
			("multiplier", toCompactHexWithPrefix(u256(1) << _numBits))
			.render();
	});
}

std::string YulUtilFunctions::shiftRightFunctionDynamic()
{
	std::string const functionName = "shift_right_unsigned_dynamic";
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(bits, value) -> newValue {
				newValue :=
				<?hasShifts>
					shr(bits, value)
				<!hasShifts>
					div(value, exp(2, bits))
				</hasShifts>
			}
			)")
			("functionName", functionName)
			("hasShifts", m_evmVersion.hasBitwiseShifting())
			.render();
	});
}

std::string YulUtilFunctions::shiftRightSignedFunctionDynamic()
{
	std::string const functionName = "shift_right_signed_dynamic";
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(bits, value) -> result {
				<?hasShifts>
					result := sar(bits, value)
				<!hasShifts>
					let divisor := exp(2, bits)
					let xor_mask := sub(0, slt(value, 0))
					result := xor(div(xor(value, xor_mask), divisor), xor_mask)
					// combined version of
					//   switch slt(value, 0)
					//   case 0 { result := div(value, divisor) }
					//   default { result := not(div(not(value), divisor)) }
				</hasShifts>
			}
			)")
			("functionName", functionName)
			("hasShifts", m_evmVersion.hasBitwiseShifting())
			.render();
	});
}


std::string YulUtilFunctions::typedShiftLeftFunction(Type const& _type, Type const& _amountType)
{
	solUnimplementedAssert(_type.category() != Type::Category::FixedPoint, "Not yet implemented - FixedPointType.");
	solAssert(_type.category() == Type::Category::FixedBytes || _type.category() == Type::Category::Integer, "");
	solAssert(_amountType.category() == Type::Category::Integer, "");
	solAssert(!dynamic_cast<IntegerType const&>(_amountType).isSigned(), "");
	std::string const functionName = "shift_left_" + _type.identifier() + "_" + _amountType.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(value, bits) -> result {
				bits := <cleanAmount>(bits)
				result := <cleanup>(<shift>(bits, <cleanup>(value)))
			}
			)")
			("functionName", functionName)
			("cleanAmount", cleanupFunction(_amountType))
			("shift", shiftLeftFunctionDynamic())
			("cleanup", cleanupFunction(_type))
			.render();
	});
}

std::string YulUtilFunctions::typedShiftRightFunction(Type const& _type, Type const& _amountType)
{
	solUnimplementedAssert(_type.category() != Type::Category::FixedPoint, "Not yet implemented - FixedPointType.");
	solAssert(_type.category() == Type::Category::FixedBytes || _type.category() == Type::Category::Integer, "");
	solAssert(_amountType.category() == Type::Category::Integer, "");
	solAssert(!dynamic_cast<IntegerType const&>(_amountType).isSigned(), "");
	IntegerType const* integerType = dynamic_cast<IntegerType const*>(&_type);
	bool valueSigned = integerType && integerType->isSigned();

	std::string const functionName = "shift_right_" + _type.identifier() + "_" + _amountType.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(value, bits) -> result {
				bits := <cleanAmount>(bits)
				result := <cleanup>(<shift>(bits, <cleanup>(value)))
			}
			)")
			("functionName", functionName)
			("cleanAmount", cleanupFunction(_amountType))
			("shift", valueSigned ? shiftRightSignedFunctionDynamic() : shiftRightFunctionDynamic())
			("cleanup", cleanupFunction(_type))
			.render();
	});
}

std::string YulUtilFunctions::updateByteSliceFunction(size_t _numBytes, size_t _shiftBytes)
{
	solAssert(_numBytes <= 32, "");
	solAssert(_shiftBytes <= 32, "");
	size_t numBits = _numBytes * 8;
	size_t shiftBits = _shiftBytes * 8;
	std::string functionName = "update_byte_slice_" + std::to_string(_numBytes) + "_shift_" + std::to_string(_shiftBytes);
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(value, toInsert) -> result {
				let mask := <mask>
				toInsert := <shl>(toInsert)
				value := and(value, not(mask))
				result := or(value, and(toInsert, mask))
			}
			)")
			("functionName", functionName)
			("mask", formatNumber(((bigint(1) << numBits) - 1) << shiftBits))
			("shl", shiftLeftFunction(shiftBits))
			.render();
	});
}

std::string YulUtilFunctions::updateByteSliceFunctionDynamic(size_t _numBytes)
{
	solAssert(_numBytes <= 32, "");
	size_t numBits = _numBytes * 8;
	std::string functionName = "update_byte_slice_dynamic" + std::to_string(_numBytes);
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(value, shiftBytes, toInsert) -> result {
				let shiftBits := mul(shiftBytes, 8)
				let mask := <shl>(shiftBits, <mask>)
				toInsert := <shl>(shiftBits, toInsert)
				value := and(value, not(mask))
				result := or(value, and(toInsert, mask))
			}
			)")
			("functionName", functionName)
			("mask", formatNumber((bigint(1) << numBits) - 1))
			("shl", shiftLeftFunctionDynamic())
			.render();
	});
}

std::string YulUtilFunctions::maskBytesFunctionDynamic()
{
	std::string functionName = "mask_bytes_dynamic";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(data, bytes) -> result {
				let mask := not(<shr>(mul(8, bytes), not(0)))
				result := and(data, mask)
			})")
			("functionName", functionName)
			("shr", shiftRightFunctionDynamic())
			.render();
	});
}

std::string YulUtilFunctions::maskLowerOrderBytesFunction(size_t _bytes)
{
	std::string functionName = "mask_lower_order_bytes_" + std::to_string(_bytes);
	solAssert(_bytes <= 32, "");
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(data) -> result {
				result := and(data, <mask>)
			})")
			("functionName", functionName)
			("mask", formatNumber((~u256(0)) >> (256 - 8 * _bytes)))
			.render();
	});
}

std::string YulUtilFunctions::maskLowerOrderBytesFunctionDynamic()
{
	std::string functionName = "mask_lower_order_bytes_dynamic";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(data, bytes) -> result {
				let mask := not(<shl>(mul(8, bytes), not(0)))
				result := and(data, mask)
			})")
			("functionName", functionName)
			("shl", shiftLeftFunctionDynamic())
			.render();
	});
}

std::string YulUtilFunctions::roundUpFunction()
{
	std::string functionName = "round_up_to_mul_of_32";
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(value) -> result {
				result := and(add(value, 31), not(31))
			}
			)")
			("functionName", functionName)
			.render();
	});
}

std::string YulUtilFunctions::divide32CeilFunction()
{
	return m_functionCollector.createFunction(
		"divide_by_32_ceil",
		[&](std::vector<std::string>& _args, std::vector<std::string>& _ret) {
			_args = {"value"};
			_ret = {"result"};
			return "result := div(add(value, 31), 32)";
		}
	);
}

std::string YulUtilFunctions::overflowCheckedIntAddFunction(IntegerType const& _type)
{
	std::string functionName = "checked_add_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(x, y) -> sum {
				x := <cleanupFunction>(x)
				y := <cleanupFunction>(y)
				sum := add(x, y)
				<?signed>
					<?256bit>
						// overflow, if x >= 0 and sum < y
						// underflow, if x < 0 and sum >= y
						if or(
							and(iszero(slt(x, 0)), slt(sum, y)),
							and(slt(x, 0), iszero(slt(sum, y)))
						) { <panic>() }
					<!256bit>
						if or(
							sgt(sum, <maxValue>),
							slt(sum, <minValue>)
						) { <panic>() }
					</256bit>
				<!signed>
					<?256bit>
						if gt(x, sum) { <panic>() }
					<!256bit>
						if gt(sum, <maxValue>) { <panic>() }
					</256bit>
				</signed>
			}
			)")
			("functionName", functionName)
			("signed", _type.isSigned())
			("maxValue", toCompactHexWithPrefix(u256(_type.maxValue())))
			("minValue", toCompactHexWithPrefix(u256(_type.minValue())))
			("cleanupFunction", cleanupFunction(_type))
			("panic", panicFunction(PanicCode::UnderOverflow))
			("256bit", _type.numBits() == 256)
			.render();
	});
}

std::string YulUtilFunctions::wrappingIntAddFunction(IntegerType const& _type)
{
	std::string functionName = "wrapping_add_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(x, y) -> sum {
				sum := <cleanupFunction>(add(x, y))
			}
			)")
			("functionName", functionName)
			("cleanupFunction", cleanupFunction(_type))
			.render();
	});
}

std::string YulUtilFunctions::overflowCheckedIntMulFunction(IntegerType const& _type)
{
	std::string functionName = "checked_mul_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			// Multiplication by zero could be treated separately and directly return zero.
			Whiskers(R"(
			function <functionName>(x, y) -> product {
				x := <cleanupFunction>(x)
				y := <cleanupFunction>(y)
				let product_raw := mul(x, y)
				product := <cleanupFunction>(product_raw)
				<?signed>
					<?gt128bit>
						<?256bit>
							// special case
							if and(slt(x, 0), eq(y, <minValue>)) { <panic>() }
						</256bit>
						// overflow, if x != 0 and y != product/x
						if iszero(
							or(
								iszero(x),
								eq(y, sdiv(product, x))
							)
						) { <panic>() }
					<!gt128bit>
						if iszero(eq(product, product_raw)) { <panic>() }
					</gt128bit>
				<!signed>
					<?gt128bit>
						// overflow, if x != 0 and y != product/x
						if iszero(
							or(
								iszero(x),
								eq(y, div(product, x))
							)
						) { <panic>() }
					<!gt128bit>
						if iszero(eq(product, product_raw)) { <panic>() }
					</gt128bit>
				</signed>
			}
			)")
			("functionName", functionName)
			("signed", _type.isSigned())
			("cleanupFunction", cleanupFunction(_type))
			("panic", panicFunction(PanicCode::UnderOverflow))
			("minValue", toCompactHexWithPrefix(u256(_type.minValue())))
			("256bit", _type.numBits() == 256)
			("gt128bit", _type.numBits() > 128)
			.render();
	});
}

std::string YulUtilFunctions::wrappingIntMulFunction(IntegerType const& _type)
{
	std::string functionName = "wrapping_mul_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(x, y) -> product {
				product := <cleanupFunction>(mul(x, y))
			}
			)")
			("functionName", functionName)
			("cleanupFunction", cleanupFunction(_type))
			.render();
	});
}

std::string YulUtilFunctions::overflowCheckedIntDivFunction(IntegerType const& _type)
{
	std::string functionName = "checked_div_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(x, y) -> r {
				x := <cleanupFunction>(x)
				y := <cleanupFunction>(y)
				if iszero(y) { <panicDivZero>() }
				<?signed>
				// overflow for minVal / -1
				if and(
					eq(x, <minVal>),
					eq(y, sub(0, 1))
				) { <panicOverflow>() }
				</signed>
				r := <?signed>s</signed>div(x, y)
			}
			)")
			("functionName", functionName)
			("signed", _type.isSigned())
			("minVal", toCompactHexWithPrefix(u256(_type.minValue())))
			("cleanupFunction", cleanupFunction(_type))
			("panicDivZero", panicFunction(PanicCode::DivisionByZero))
			("panicOverflow", panicFunction(PanicCode::UnderOverflow))
			.render();
	});
}

std::string YulUtilFunctions::wrappingIntDivFunction(IntegerType const& _type)
{
	std::string functionName = "wrapping_div_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(x, y) -> r {
				x := <cleanupFunction>(x)
				y := <cleanupFunction>(y)
				if iszero(y) { <error>() }
				r := <?signed>s</signed>div(x, y)
			}
			)")
			("functionName", functionName)
			("cleanupFunction", cleanupFunction(_type))
			("signed", _type.isSigned())
			("error", panicFunction(PanicCode::DivisionByZero))
			.render();
	});
}

std::string YulUtilFunctions::intModFunction(IntegerType const& _type)
{
	std::string functionName = "mod_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(x, y) -> r {
				x := <cleanupFunction>(x)
				y := <cleanupFunction>(y)
				if iszero(y) { <panic>() }
				r := <?signed>s</signed>mod(x, y)
			}
			)")
			("functionName", functionName)
			("signed", _type.isSigned())
			("cleanupFunction", cleanupFunction(_type))
			("panic", panicFunction(PanicCode::DivisionByZero))
			.render();
	});
}

std::string YulUtilFunctions::overflowCheckedIntSubFunction(IntegerType const& _type)
{
	std::string functionName = "checked_sub_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&] {
		return
			Whiskers(R"(
			function <functionName>(x, y) -> diff {
				x := <cleanupFunction>(x)
				y := <cleanupFunction>(y)
				diff := sub(x, y)
				<?signed>
					<?256bit>
						// underflow, if y >= 0 and diff > x
						// overflow, if y < 0 and diff < x
						if or(
							and(iszero(slt(y, 0)), sgt(diff, x)),
							and(slt(y, 0), slt(diff, x))
						) { <panic>() }
					<!256bit>
						if or(
							slt(diff, <minValue>),
							sgt(diff, <maxValue>)
						) { <panic>() }
					</256bit>
				<!signed>
					<?256bit>
						if gt(diff, x) { <panic>() }
					<!256bit>
						if gt(diff, <maxValue>) { <panic>() }
					</256bit>
				</signed>
			}
			)")
			("functionName", functionName)
			("signed", _type.isSigned())
			("maxValue", toCompactHexWithPrefix(u256(_type.maxValue())))
			("minValue", toCompactHexWithPrefix(u256(_type.minValue())))
			("cleanupFunction", cleanupFunction(_type))
			("panic", panicFunction(PanicCode::UnderOverflow))
			("256bit", _type.numBits() == 256)
			.render();
	});
}

std::string YulUtilFunctions::wrappingIntSubFunction(IntegerType const& _type)
{
	std::string functionName = "wrapping_sub_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&] {
		return
			Whiskers(R"(
			function <functionName>(x, y) -> diff {
				diff := <cleanupFunction>(sub(x, y))
			}
			)")
			("functionName", functionName)
			("cleanupFunction", cleanupFunction(_type))
			.render();
	});
}

std::string YulUtilFunctions::overflowCheckedIntExpFunction(
	IntegerType const& _type,
	IntegerType const& _exponentType
)
{
	solAssert(!_exponentType.isSigned(), "");

	std::string functionName = "checked_exp_" + _type.identifier() + "_" + _exponentType.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(base, exponent) -> power {
				base := <baseCleanupFunction>(base)
				exponent := <exponentCleanupFunction>(exponent)
				<?signed>
					power := <exp>(base, exponent, <minValue>, <maxValue>)
				<!signed>
					power := <exp>(base, exponent, <maxValue>)
				</signed>

			}
			)")
			("functionName", functionName)
			("signed", _type.isSigned())
			("exp", _type.isSigned() ? overflowCheckedSignedExpFunction() : overflowCheckedUnsignedExpFunction())
			("maxValue", toCompactHexWithPrefix(_type.max()))
			("minValue", toCompactHexWithPrefix(_type.min()))
			("baseCleanupFunction", cleanupFunction(_type))
			("exponentCleanupFunction", cleanupFunction(_exponentType))
			.render();
	});
}

std::string YulUtilFunctions::overflowCheckedIntLiteralExpFunction(
	RationalNumberType const& _baseType,
	IntegerType const& _exponentType,
	IntegerType const& _commonType
)
{
	solAssert(!_exponentType.isSigned(), "");
	solAssert(_baseType.isNegative() == _commonType.isSigned(), "");
	solAssert(_commonType.numBits() == 256, "");

	std::string functionName = "checked_exp_" + _baseType.richIdentifier() + "_" + _exponentType.identifier();

	return m_functionCollector.createFunction(functionName, [&]()
	{
		// Converts a bigint number into u256 (negative numbers represented in two's complement form.)
		// We assume that `_v` fits in 256 bits.
		auto bigint2u = [&](bigint const& _v) -> u256
		{
			if (_v < 0)
				return s2u(s256(_v));
			return u256(_v);
		};

		// Calculates the upperbound for exponentiation, that is, calculate `b`, such that
		// _base**b <= _maxValue and _base**(b + 1) > _maxValue
		auto findExponentUpperbound = [](bigint const _base, bigint const _maxValue) -> unsigned
		{
			// There is no overflow for these cases
			if (_base == 0 || _base == -1 || _base == 1)
				return 0;

			unsigned first = 0;
			unsigned last = 255;
			unsigned middle;

			while (first < last)
			{
				middle = (first + last) / 2;

				if (
					// The condition on msb is a shortcut that avoids computing large powers in
					// arbitrary precision.
					boost::multiprecision::msb(_base) * middle <= boost::multiprecision::msb(_maxValue) &&
					boost::multiprecision::pow(_base, middle) <= _maxValue
				)
				{
					if (boost::multiprecision::pow(_base, middle + 1) > _maxValue)
						return middle;
					else
						first = middle + 1;
				}
				else
					last = middle;
			}

			return last;
		};

		bigint baseValue = _baseType.isNegative() ?
			u2s(_baseType.literalValue(nullptr)) :
			_baseType.literalValue(nullptr);
		bool needsOverflowCheck = !((baseValue == 0) || (baseValue == -1) || (baseValue == 1));
		unsigned exponentUpperbound;

		if (_baseType.isNegative())
		{
			// Only checks for underflow. The only case where this can be a problem is when, for a
			// negative base, say `b`, and an even exponent, say `e`, `b**e = 2**255` (which is an
			// overflow.) But this never happens because, `255 = 3*5*17`, and therefore there is no even
			// number `e` such that `b**e = 2**255`.
			exponentUpperbound = findExponentUpperbound(abs(baseValue), abs(_commonType.minValue()));

			bigint power = boost::multiprecision::pow(baseValue, exponentUpperbound);
			bigint overflowedPower = boost::multiprecision::pow(baseValue, exponentUpperbound + 1);

			if (needsOverflowCheck)
				solAssert(
					(power <= _commonType.maxValue()) && (power >= _commonType.minValue()) &&
					!((overflowedPower <= _commonType.maxValue()) && (overflowedPower >= _commonType.minValue())),
					"Incorrect exponent upper bound calculated."
				);
		}
		else
		{
			exponentUpperbound = findExponentUpperbound(baseValue, _commonType.maxValue());

			if (needsOverflowCheck)
				solAssert(
					boost::multiprecision::pow(baseValue, exponentUpperbound) <= _commonType.maxValue() &&
					boost::multiprecision::pow(baseValue, exponentUpperbound + 1) > _commonType.maxValue(),
					"Incorrect exponent upper bound calculated."
				);
		}

		return Whiskers(R"(
			function <functionName>(exponent) -> power {
				exponent := <exponentCleanupFunction>(exponent)
				<?needsOverflowCheck>
				if gt(exponent, <exponentUpperbound>) { <panic>() }
				</needsOverflowCheck>
				power := exp(<base>, exponent)
			}
			)")
			("functionName", functionName)
			("exponentCleanupFunction", cleanupFunction(_exponentType))
			("needsOverflowCheck", needsOverflowCheck)
			("exponentUpperbound", std::to_string(exponentUpperbound))
			("panic", panicFunction(PanicCode::UnderOverflow))
			("base", bigint2u(baseValue).str())
			.render();
	});
}

std::string YulUtilFunctions::overflowCheckedUnsignedExpFunction()
{
	// Checks for the "small number specialization" below.
	using namespace boost::multiprecision;
	solAssert(pow(bigint(10), 77) < pow(bigint(2), 256), "");
	solAssert(pow(bigint(11), 77) >= pow(bigint(2), 256), "");
	solAssert(pow(bigint(10), 78) >= pow(bigint(2), 256), "");

	solAssert(pow(bigint(306), 31) < pow(bigint(2), 256), "");
	solAssert(pow(bigint(307), 31) >= pow(bigint(2), 256), "");
	solAssert(pow(bigint(306), 32) >= pow(bigint(2), 256), "");

	std::string functionName = "checked_exp_unsigned";
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(base, exponent, max) -> power {
				// This function currently cannot be inlined because of the
				// "leave" statements. We have to improve the optimizer.

				// Note that 0**0 == 1
				if iszero(exponent) { power := 1 leave }
				if iszero(base) { power := 0 leave }

				// Specializations for small bases
				switch base
				// 0 is handled above
				case 1 { power := 1 leave }
				case 2
				{
					if gt(exponent, 255) { <panic>() }
					power := exp(2, exponent)
					if gt(power, max) { <panic>() }
					leave
				}
				if or(
					and(lt(base, 11), lt(exponent, 78)),
					and(lt(base, 307), lt(exponent, 32))
				)
				{
					power := exp(base, exponent)
					if gt(power, max) { <panic>() }
					leave
				}

				power, base := <expLoop>(1, base, exponent, max)

				if gt(power, div(max, base)) { <panic>() }
				power := mul(power, base)
			}
			)")
			("functionName", functionName)
			("panic", panicFunction(PanicCode::UnderOverflow))
			("expLoop", overflowCheckedExpLoopFunction())
			.render();
	});
}

std::string YulUtilFunctions::overflowCheckedSignedExpFunction()
{
	std::string functionName = "checked_exp_signed";
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(base, exponent, min, max) -> power {
				// Currently, `leave` avoids this function being inlined.
				// We have to improve the optimizer.

				// Note that 0**0 == 1
				switch exponent
				case 0 { power := 1 leave }
				case 1 { power := base leave }
				if iszero(base) { power := 0 leave }

				power := 1

				// We pull out the first iteration because it is the only one in which
				// base can be negative.
				// Exponent is at least 2 here.

				// overflow check for base * base
				switch sgt(base, 0)
				case 1 { if gt(base, div(max, base)) { <panic>() } }
				case 0 { if slt(base, sdiv(max, base)) { <panic>() } }
				if and(exponent, 1)
				{
					power := base
				}
				base := mul(base, base)
				exponent := <shr_1>(exponent)

				// Below this point, base is always positive.

				power, base := <expLoop>(power, base, exponent, max)

				if and(sgt(power, 0), gt(power, div(max, base))) { <panic>() }
				if and(slt(power, 0), slt(power, sdiv(min, base))) { <panic>() }
				power := mul(power, base)
			}
			)")
			("functionName", functionName)
			("panic", panicFunction(PanicCode::UnderOverflow))
			("expLoop", overflowCheckedExpLoopFunction())
			("shr_1", shiftRightFunction(1))
			.render();
	});
}

std::string YulUtilFunctions::overflowCheckedExpLoopFunction()
{
	// We use this loop for both signed and unsigned exponentiation
	// because we pull out the first iteration in the signed case which
	// results in the base always being positive.

	// This function does not include the final multiplication.

	std::string functionName = "checked_exp_helper";
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(_power, _base, exponent, max) -> power, base {
				power := _power
				base  := _base
				for { } gt(exponent, 1) {}
				{
					// overflow check for base * base
					if gt(base, div(max, base)) { <panic>() }
					if and(exponent, 1)
					{
						// No checks for power := mul(power, base) needed, because the check
						// for base * base above is sufficient, since:
						// |power| <= base (proof by induction) and thus:
						// |power * base| <= base * base <= max <= |min| (for signed)
						// (this is equally true for signed and unsigned exp)
						power := mul(power, base)
					}
					base := mul(base, base)
					exponent := <shr_1>(exponent)
				}
			}
			)")
			("functionName", functionName)
			("panic", panicFunction(PanicCode::UnderOverflow))
			("shr_1", shiftRightFunction(1))
			.render();
	});
}

std::string YulUtilFunctions::wrappingIntExpFunction(
	IntegerType const& _type,
	IntegerType const& _exponentType
)
{
	solAssert(!_exponentType.isSigned(), "");

	std::string functionName = "wrapping_exp_" + _type.identifier() + "_" + _exponentType.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return
			Whiskers(R"(
			function <functionName>(base, exponent) -> power {
				base := <baseCleanupFunction>(base)
				exponent := <exponentCleanupFunction>(exponent)
				power := <baseCleanupFunction>(exp(base, exponent))
			}
			)")
			("functionName", functionName)
			("baseCleanupFunction", cleanupFunction(_type))
			("exponentCleanupFunction", cleanupFunction(_exponentType))
			.render();
	});
}

std::string YulUtilFunctions::arrayLengthFunction(ArrayType const& _type)
{
	std::string functionName = "array_length_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers w(R"(
			function <functionName>(value<?dynamic><?calldata>, len</calldata></dynamic>) -> length {
				<?dynamic>
					<?memory>
						length := mload(value)
					</memory>
					<?storage>
						length := sload(value)
						<?byteArray>
							length := <extractByteArrayLength>(length)
						</byteArray>
					</storage>
					<?calldata>
						length := len
					</calldata>
				<!dynamic>
					length := <length>
				</dynamic>
			}
		)");
		w("functionName", functionName);
		w("dynamic", _type.isDynamicallySized());
		if (!_type.isDynamicallySized())
			w("length", toCompactHexWithPrefix(_type.length()));
		w("memory", _type.location() == DataLocation::Memory);
		w("storage", _type.location() == DataLocation::Storage);
		w("calldata", _type.location() == DataLocation::CallData);
		if (_type.location() == DataLocation::Storage)
		{
			w("byteArray", _type.isByteArrayOrString());
			if (_type.isByteArrayOrString())
				w("extractByteArrayLength", extractByteArrayLengthFunction());
		}

		return w.render();
	});
}

std::string YulUtilFunctions::extractByteArrayLengthFunction()
{
	std::string functionName = "extract_byte_array_length";
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers w(R"(
			function <functionName>(data) -> length {
				length := div(data, 2)
				let outOfPlaceEncoding := and(data, 1)
				if iszero(outOfPlaceEncoding) {
					length := and(length, 0x7f)
				}

				if eq(outOfPlaceEncoding, lt(length, 32)) {
					<panic>()
				}
			}
		)");
		w("functionName", functionName);
		w("panic", panicFunction(PanicCode::StorageEncodingError));
		return w.render();
	});
}

std::string YulUtilFunctions::resizeArrayFunction(ArrayType const& _type)
{
	solAssert(_type.location() == DataLocation::Storage, "");
	solUnimplementedAssert(_type.baseType()->storageBytes() <= 32);

	if (_type.isByteArrayOrString())
		return resizeDynamicByteArrayFunction(_type);

	std::string functionName = "resize_array_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(array, newLen) {
				if gt(newLen, <maxArrayLength>) {
					<panic>()
				}

				let oldLen := <fetchLength>(array)

				<?isDynamic>
					// Store new length
					sstore(array, newLen)
				</isDynamic>

				<?needsClearing>
					<cleanUpArrayEnd>(array, oldLen, newLen)
				</needsClearing>
			})");
			templ("functionName", functionName);
			templ("maxArrayLength", (u256(1) << 64).str());
			templ("panic", panicFunction(util::PanicCode::ResourceError));
			templ("fetchLength", arrayLengthFunction(_type));
			templ("isDynamic", _type.isDynamicallySized());
			bool isMappingBase = _type.baseType()->category() == Type::Category::Mapping;
			templ("needsClearing", !isMappingBase);
			if (!isMappingBase)
				templ("cleanUpArrayEnd", cleanUpStorageArrayEndFunction(_type));
			return templ.render();
	});
}

std::string YulUtilFunctions::cleanUpStorageArrayEndFunction(ArrayType const& _type)
{
	solAssert(_type.location() == DataLocation::Storage, "");
	solAssert(_type.baseType()->category() != Type::Category::Mapping, "");
	solAssert(!_type.isByteArrayOrString(), "");
	solUnimplementedAssert(_type.baseType()->storageBytes() <= 32);

	std::string functionName = "cleanup_storage_array_end_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&](std::vector<std::string>& _args, std::vector<std::string>&) {
		_args = {"array", "len", "startIndex"};
		return Whiskers(R"(
			if lt(startIndex, len) {
				// Size was reduced, clear end of array
				let oldSlotCount := <convertToSize>(len)
				let newSlotCount := <convertToSize>(startIndex)
				let arrayDataStart := <dataPosition>(array)
				let deleteStart := add(arrayDataStart, newSlotCount)
				let deleteEnd := add(arrayDataStart, oldSlotCount)
				<?packed>
					// if we are dealing with packed array and offset is greater than zero
					// we have  to partially clear last slot that is still used, so decreasing start by one
					let offset := mul(mod(startIndex, <itemsPerSlot>), <storageBytes>)
					if gt(offset, 0) { <partialClearStorageSlot>(sub(deleteStart, 1), offset) }
				</packed>
				<clearStorageRange>(deleteStart, deleteEnd)
			}
		)")
		("convertToSize", arrayConvertLengthToSize(_type))
		("dataPosition", arrayDataAreaFunction(_type))
		("clearStorageRange", clearStorageRangeFunction(*_type.baseType()))
		("packed", _type.baseType()->storageBytes() <= 16)
		("itemsPerSlot", std::to_string(32 / _type.baseType()->storageBytes()))
		("storageBytes", std::to_string(_type.baseType()->storageBytes()))
		("partialClearStorageSlot", partialClearStorageSlotFunction())
		.render();
	});
}

std::string YulUtilFunctions::resizeDynamicByteArrayFunction(ArrayType const& _type)
{
	std::string functionName = "resize_array_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&](std::vector<std::string>& _args, std::vector<std::string>&) {
		_args = {"array", "newLen"};
		return Whiskers(R"(
			let data := sload(array)
			let oldLen := <extractLength>(data)

			if gt(newLen, oldLen) {
				<increaseSize>(array, data, oldLen, newLen)
			}

			if lt(newLen, oldLen) {
				<decreaseSize>(array, data, oldLen, newLen)
			}
		)")
		("extractLength", extractByteArrayLengthFunction())
		("decreaseSize", decreaseByteArraySizeFunction(_type))
		("increaseSize", increaseByteArraySizeFunction(_type))
		.render();
	});
}

std::string YulUtilFunctions::cleanUpDynamicByteArrayEndSlotsFunction(ArrayType const& _type)
{
	solAssert(_type.isByteArrayOrString(), "");
	solAssert(_type.isDynamicallySized(), "");

	std::string functionName = "clean_up_bytearray_end_slots_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&](std::vector<std::string>& _args, std::vector<std::string>&) {
		_args = {"array", "len", "startIndex"};
		return Whiskers(R"(
			if gt(len, 31) {
				let dataArea := <dataLocation>(array)
				let deleteStart := add(dataArea, <div32Ceil>(startIndex))
				// If we are clearing array to be short byte array, we want to clear only data starting from array data area.
				if lt(startIndex, 32) { deleteStart := dataArea }
				<clearStorageRange>(deleteStart, add(dataArea, <div32Ceil>(len)))
			}
		)")
		("dataLocation", arrayDataAreaFunction(_type))
		("div32Ceil", divide32CeilFunction())
		("clearStorageRange", clearStorageRangeFunction(*_type.baseType()))
		.render();
	});
}

std::string YulUtilFunctions::decreaseByteArraySizeFunction(ArrayType const& _type)
{
	std::string functionName = "byte_array_decrease_size_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(array, data, oldLen, newLen) {
				switch lt(newLen, 32)
				case  0 {
					let arrayDataStart := <dataPosition>(array)
					let deleteStart := add(arrayDataStart, <div32Ceil>(newLen))

					// we have to partially clear last slot that is still used
					let offset := and(newLen, 0x1f)
					if offset { <partialClearStorageSlot>(sub(deleteStart, 1), offset) }

					<clearStorageRange>(deleteStart, add(arrayDataStart, <div32Ceil>(oldLen)))

					sstore(array, or(mul(2, newLen), 1))
				}
				default {
					switch gt(oldLen, 31)
					case 1 {
						let arrayDataStart := <dataPosition>(array)
						// clear whole old array, as we are transforming to short bytes array
						<clearStorageRange>(add(arrayDataStart, 1), add(arrayDataStart, <div32Ceil>(oldLen)))
						<transitLongToShort>(array, newLen)
					}
					default {
						sstore(array, <encodeUsedSetLen>(data, newLen))
					}
				}
			})")
			("functionName", functionName)
			("dataPosition", arrayDataAreaFunction(_type))
			("partialClearStorageSlot", partialClearStorageSlotFunction())
			("clearStorageRange", clearStorageRangeFunction(*_type.baseType()))
			("transitLongToShort", byteArrayTransitLongToShortFunction(_type))
			("div32Ceil", divide32CeilFunction())
			("encodeUsedSetLen", shortByteArrayEncodeUsedAreaSetLengthFunction())
			.render();
	});
}

std::string YulUtilFunctions::increaseByteArraySizeFunction(ArrayType const& _type)
{
	std::string functionName = "byte_array_increase_size_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&](std::vector<std::string>& _args, std::vector<std::string>&) {
		_args = {"array", "data", "oldLen", "newLen"};
		return Whiskers(R"(
			if gt(newLen, <maxArrayLength>) { <panic>() }

			switch lt(oldLen, 32)
			case 0 {
				// in this case array stays unpacked, so we just set new length
				sstore(array, add(mul(2, newLen), 1))
			}
			default {
				switch lt(newLen, 32)
				case 0 {
					// we need to copy elements to data area as we changed array from packed to unpacked
					data := and(not(0xff), data)
					sstore(<dataPosition>(array), data)
					sstore(array, add(mul(2, newLen), 1))
				}
				default {
					// here array stays packed, we just need to increase length
					sstore(array, <encodeUsedSetLen>(data, newLen))
				}
			}
		)")
		("panic", panicFunction(PanicCode::ResourceError))
		("maxArrayLength", (u256(1) << 64).str())
		("dataPosition", arrayDataAreaFunction(_type))
		("encodeUsedSetLen", shortByteArrayEncodeUsedAreaSetLengthFunction())
		.render();
	});
}

std::string YulUtilFunctions::byteArrayTransitLongToShortFunction(ArrayType const& _type)
{
	std::string functionName = "transit_byte_array_long_to_short_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(array, len) {
				// we need to copy elements from old array to new
				// we want to copy only elements that are part of the array after resizing
				let dataPos := <dataPosition>(array)
				let data := <extractUsedApplyLen>(sload(dataPos), len)
				sstore(array, data)
				sstore(dataPos, 0)
			})")
			("functionName", functionName)
			("dataPosition", arrayDataAreaFunction(_type))
			("extractUsedApplyLen", shortByteArrayEncodeUsedAreaSetLengthFunction())
			.render();
	});
}

std::string YulUtilFunctions::shortByteArrayEncodeUsedAreaSetLengthFunction()
{
	std::string functionName = "extract_used_part_and_set_length_of_short_byte_array";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(data, len) -> used {
				// we want to save only elements that are part of the array after resizing
				// others should be set to zero
				data := <maskBytes>(data, len)
				used := or(data, mul(2, len))
			})")
			("functionName", functionName)
			("maskBytes", maskBytesFunctionDynamic())
			.render();
	});
}

std::string YulUtilFunctions::longByteArrayStorageIndexAccessNoCheckFunction()
{
	return m_functionCollector.createFunction(
		"long_byte_array_index_access_no_checks",
		[&](std::vector<std::string>& _args, std::vector<std::string>& _returnParams) {
			_args = {"array", "index"};
			_returnParams = {"slot", "offset"};
			return Whiskers(R"(
				offset := sub(31, mod(index, 0x20))
				let dataArea := <dataAreaFunc>(array)
				slot := add(dataArea, div(index, 0x20))
			)")
			("dataAreaFunc", arrayDataAreaFunction(*TypeProvider::bytesStorage()))
			.render();
		}
	);
}

std::string YulUtilFunctions::storageArrayPopFunction(ArrayType const& _type)
{
	solAssert(_type.location() == DataLocation::Storage, "");
	solAssert(_type.isDynamicallySized(), "");
	solUnimplementedAssert(_type.baseType()->storageBytes() <= 32, "Base type is not yet implemented.");
	if (_type.isByteArrayOrString())
		return storageByteArrayPopFunction(_type);

	std::string functionName = "array_pop_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(array) {
				let oldLen := <fetchLength>(array)
				if iszero(oldLen) { <panic>() }
				let newLen := sub(oldLen, 1)
				let slot, offset := <indexAccess>(array, newLen)
				<?+setToZero><setToZero>(slot, offset)</+setToZero>
				sstore(array, newLen)
			})")
			("functionName", functionName)
			("panic", panicFunction(PanicCode::EmptyArrayPop))
			("fetchLength", arrayLengthFunction(_type))
			("indexAccess", storageArrayIndexAccessFunction(_type))
			(
				"setToZero",
				_type.baseType()->category() != Type::Category::Mapping ? storageSetToZeroFunction(*_type.baseType(), VariableDeclaration::Location::Unspecified) : ""
			)
			.render();
	});
}

std::string YulUtilFunctions::storageByteArrayPopFunction(ArrayType const& _type)
{
	solAssert(_type.location() == DataLocation::Storage, "");
	solAssert(_type.isDynamicallySized(), "");
	solAssert(_type.isByteArrayOrString(), "");

	std::string functionName = "byte_array_pop_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(array) {
				let data := sload(array)
				let oldLen := <extractByteArrayLength>(data)
				if iszero(oldLen) { <panic>() }

				switch oldLen
				case 32 {
					// Here we have a special case where array transitions to shorter than 32
					// So we need to copy data
					<transitLongToShort>(array, 31)
				}
				default {
					let newLen := sub(oldLen, 1)
					switch lt(oldLen, 32)
					case 1 {
						sstore(array, <encodeUsedSetLen>(data, newLen))
					}
					default {
						let slot, offset := <indexAccessNoChecks>(array, newLen)
						<setToZero>(slot, offset)
						sstore(array, sub(data, 2))
					}
				}
			})")
			("functionName", functionName)
			("panic", panicFunction(PanicCode::EmptyArrayPop))
			("extractByteArrayLength", extractByteArrayLengthFunction())
			("transitLongToShort", byteArrayTransitLongToShortFunction(_type))
			("encodeUsedSetLen", shortByteArrayEncodeUsedAreaSetLengthFunction())
			("indexAccessNoChecks", longByteArrayStorageIndexAccessNoCheckFunction())
			("setToZero", storageSetToZeroFunction(*_type.baseType(), VariableDeclaration::Location::Unspecified))
			.render();
	});
}

std::string YulUtilFunctions::storageArrayPushFunction(ArrayType const& _type, Type const* _fromType)
{
	solAssert(_type.location() == DataLocation::Storage, "");
	solAssert(_type.isDynamicallySized(), "");
	if (!_fromType)
		_fromType = _type.baseType();
	else if (_fromType->isValueType())
		solUnimplementedAssert(*_fromType == *_type.baseType());

	std::string functionName =
		std::string{"array_push_from_"} +
		_fromType->identifier() +
		"_to_" +
		_type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(array <values>) {
				<?isByteArrayOrString>
					let data := sload(array)
					let oldLen := <extractByteArrayLength>(data)
					if iszero(lt(oldLen, <maxArrayLength>)) { <panic>() }

					switch gt(oldLen, 31)
					case 0 {
						let value := byte(0 <values>)
						switch oldLen
						case 31 {
							// Here we have special case when array switches from short array to long array
							// We need to copy data
							let dataArea := <dataAreaFunction>(array)
							data := and(data, not(0xff))
							sstore(dataArea, or(and(0xff, value), data))
							// New length is 32, encoded as (32 * 2 + 1)
							sstore(array, 65)
						}
						default {
							data := add(data, 2)
							let shiftBits := mul(8, sub(31, oldLen))
							let valueShifted := <shl>(shiftBits, and(0xff, value))
							let mask := <shl>(shiftBits, 0xff)
							data := or(and(data, not(mask)), valueShifted)
							sstore(array, data)
						}
					}
					default {
						sstore(array, add(data, 2))
						let slot, offset := <indexAccess>(array, oldLen)
						<storeValue>(slot, offset <values>)
					}
				<!isByteArrayOrString>
					let oldLen := sload(array)
					if iszero(lt(oldLen, <maxArrayLength>)) { <panic>() }
					sstore(array, add(oldLen, 1))
					let slot, offset := <indexAccess>(array, oldLen)
					<storeValue>(slot, offset <values>)
				</isByteArrayOrString>
			})")
			("functionName", functionName)
			("values", _fromType->sizeOnStack() == 0 ? "" : ", " + suffixedVariableNameList("value", 0, _fromType->sizeOnStack()))
			("panic", panicFunction(PanicCode::ResourceError))
			("extractByteArrayLength", _type.isByteArrayOrString() ? extractByteArrayLengthFunction() : "")
			("dataAreaFunction", arrayDataAreaFunction(_type))
			("isByteArrayOrString", _type.isByteArrayOrString())
			("indexAccess", storageArrayIndexAccessFunction(_type))
			("storeValue", updateStorageValueFunction(*_fromType, *_type.baseType(), VariableDeclaration::Location::Unspecified))
			("maxArrayLength", (u256(1) << 64).str())
			("shl", shiftLeftFunctionDynamic())
			.render();
	});
}

std::string YulUtilFunctions::storageArrayPushZeroFunction(ArrayType const& _type)
{
	solAssert(_type.location() == DataLocation::Storage, "");
	solAssert(_type.isDynamicallySized(), "");
	solUnimplementedAssert(_type.baseType()->storageBytes() <= 32, "Base type is not yet implemented.");

	std::string functionName = "array_push_zero_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(array) -> slot, offset {
				<?isBytes>
					let data := sload(array)
					let oldLen := <extractLength>(data)
					<increaseBytesSize>(array, data, oldLen, add(oldLen, 1))
				<!isBytes>
					let oldLen := <fetchLength>(array)
					if iszero(lt(oldLen, <maxArrayLength>)) { <panic>() }
					sstore(array, add(oldLen, 1))
				</isBytes>
				slot, offset := <indexAccess>(array, oldLen)
			})")
			("functionName", functionName)
			("isBytes", _type.isByteArrayOrString())
			("increaseBytesSize", _type.isByteArrayOrString() ? increaseByteArraySizeFunction(_type) : "")
			("extractLength", _type.isByteArrayOrString() ? extractByteArrayLengthFunction() : "")
			("panic", panicFunction(PanicCode::ResourceError))
			("fetchLength", arrayLengthFunction(_type))
			("indexAccess", storageArrayIndexAccessFunction(_type))
			("maxArrayLength", (u256(1) << 64).str())
			.render();
	});
}

std::string YulUtilFunctions::partialClearStorageSlotFunction()
{
	std::string functionName = "partial_clear_storage_slot";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
		function <functionName>(slot, offset) {
			let mask := <shr>(mul(8, sub(32, offset)), <ones>)
			sstore(slot, and(mask, sload(slot)))
		}
		)")
		("functionName", functionName)
		("ones", formatNumber((bigint(1) << 256) - 1))
		("shr", shiftRightFunctionDynamic())
		.render();
	});
}

std::string YulUtilFunctions::clearStorageRangeFunction(Type const& _type)
{
	if (_type.storageBytes() < 32)
		solAssert(_type.isValueType(), "");

	std::string functionName = "clear_storage_range_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(start, end) {
				for {} lt(start, end) { start := add(start, <increment>) }
				{
					<setToZero>(start, 0)
				}
			}
		)")
		("functionName", functionName)
		("setToZero", storageSetToZeroFunction(_type.storageBytes() < 32 ? *TypeProvider::uint256() : _type, VariableDeclaration::Location::Unspecified))
		("increment", _type.storageSize().str())
		.render();
	});
}

std::string YulUtilFunctions::clearStorageArrayFunction(ArrayType const& _type)
{
	solAssert(_type.location() == DataLocation::Storage, "");

	if (_type.baseType()->storageBytes() < 32)
	{
		solAssert(_type.baseType()->isValueType(), "Invalid storage size for non-value type.");
		solAssert(_type.baseType()->storageSize() <= 1, "Invalid storage size for type.");
	}

	if (_type.baseType()->isValueType())
		solAssert(_type.baseType()->storageSize() <= 1, "Invalid size for value type.");

	std::string functionName = "clear_storage_array_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(slot) {
				<?dynamic>
					<resizeArray>(slot, 0)
				<!dynamic>
					<?+clearRange><clearRange>(slot, add(slot, <lenToSize>(<len>)))</+clearRange>
				</dynamic>
			}
		)")
		("functionName", functionName)
		("dynamic", _type.isDynamicallySized())
		("resizeArray", _type.isDynamicallySized() ? resizeArrayFunction(_type) : "")
		(
			"clearRange",
			_type.baseType()->category() != Type::Category::Mapping ?
			clearStorageRangeFunction((_type.baseType()->storageBytes() < 32) ? *TypeProvider::uint256() : *_type.baseType()) :
			""
		)
		("lenToSize", arrayConvertLengthToSize(_type))
		("len", _type.length().str())
		.render();
	});
}

std::string YulUtilFunctions::clearStorageStructFunction(StructType const& _type)
{
	solAssert(_type.location() == DataLocation::Storage, "");

	std::string functionName = "clear_struct_storage_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&] {
		MemberList::MemberMap structMembers = _type.nativeMembers(nullptr);
		std::vector<std::map<std::string, std::string>> memberSetValues;

		std::set<u256> slotsCleared;
		for (auto const& member: structMembers)
		{
			if (member.type->category() == Type::Category::Mapping)
				continue;
			if (member.type->storageBytes() < 32)
			{
				auto const& slotDiff = _type.storageOffsetsOfMember(member.name).first;
				if (!slotsCleared.count(slotDiff))
				{
					memberSetValues.emplace_back().emplace("clearMember", "sstore(add(slot, " + slotDiff.str() + "), 0)");
					slotsCleared.emplace(slotDiff);
				}
			}
			else
			{
				auto const& [memberSlotDiff, memberStorageOffset] = _type.storageOffsetsOfMember(member.name);
				solAssert(memberStorageOffset == 0, "");

				memberSetValues.emplace_back().emplace("clearMember", Whiskers(R"(
						<setZero>(add(slot, <memberSlotDiff>), <memberStorageOffset>)
					)")
					("setZero", storageSetToZeroFunction(*member.type, VariableDeclaration::Location::Unspecified))
					("memberSlotDiff",  memberSlotDiff.str())
					("memberStorageOffset", std::to_string(memberStorageOffset))
					.render()
				);
			}
		}

		return Whiskers(R"(
			function <functionName>(slot) {
				<#member>
					<clearMember>
				</member>
			}
		)")
		("functionName", functionName)
		("member", memberSetValues)
		.render();
	});
}

std::string YulUtilFunctions::copyArrayToStorageFunction(ArrayType const& _fromType, ArrayType const& _toType)
{
	solAssert(
		(*_fromType.copyForLocation(_toType.location(), _toType.isPointer())).equals(dynamic_cast<ReferenceType const&>(_toType)),
		""
	);
	if (!_toType.isDynamicallySized())
		solAssert(!_fromType.isDynamicallySized() && _fromType.length() <= _toType.length(), "");

	if (_fromType.isByteArrayOrString())
		return copyByteArrayToStorageFunction(_fromType, _toType);
	if (_toType.baseType()->isValueType())
		return copyValueArrayToStorageFunction(_fromType, _toType);

	solAssert(_toType.storageStride() == 32);
	solAssert(!_fromType.baseType()->isValueType());

	std::string functionName = "copy_array_to_storage_from_" + _fromType.identifier() + "_to_" + _toType.identifier();
	return m_functionCollector.createFunction(functionName, [&](){
		Whiskers templ(R"(
			function <functionName>(slot, value<?isFromDynamicCalldata>, len</isFromDynamicCalldata>) {
				<?fromStorage> if eq(slot, value) { leave } </fromStorage>
				let length := <arrayLength>(value<?isFromDynamicCalldata>, len</isFromDynamicCalldata>)

				<resizeArray>(slot, length)

				let srcPtr := <srcDataLocation>(value)

				let elementSlot := <dstDataLocation>(slot)

				for { let i := 0 } lt(i, length) {i := add(i, 1)} {
					<?fromCalldata>
						let <stackItems> :=
						<?dynamicallyEncodedBase>
							<accessCalldataTail>(value, srcPtr)
						<!dynamicallyEncodedBase>
							srcPtr
						</dynamicallyEncodedBase>
					</fromCalldata>

					<?fromMemory>
						let <stackItems> := <readFromMemoryOrCalldata>(srcPtr)
					</fromMemory>

					<?fromStorage>
						let <stackItems> := srcPtr
					</fromStorage>

					<updateStorageValue>(elementSlot, <stackItems>)

					srcPtr := add(srcPtr, <srcStride>)

					elementSlot := add(elementSlot, <storageSize>)
				}
			}
		)");
		if (_fromType.dataStoredIn(DataLocation::Storage))
			solAssert(!_fromType.isValueType(), "");
		templ("functionName", functionName);
		bool fromCalldata = _fromType.dataStoredIn(DataLocation::CallData);
		templ("isFromDynamicCalldata", _fromType.isDynamicallySized() && fromCalldata);
		templ("fromStorage", _fromType.dataStoredIn(DataLocation::Storage));
		bool fromMemory = _fromType.dataStoredIn(DataLocation::Memory);
		templ("fromMemory", fromMemory);
		templ("fromCalldata", fromCalldata);
		templ("srcDataLocation", arrayDataAreaFunction(_fromType));
		if (fromCalldata)
		{
			templ("dynamicallyEncodedBase", _fromType.baseType()->isDynamicallyEncoded());
			if (_fromType.baseType()->isDynamicallyEncoded())
				templ("accessCalldataTail", accessCalldataTailFunction(*_fromType.baseType()));
		}
		templ("resizeArray", resizeArrayFunction(_toType));
		templ("arrayLength",arrayLengthFunction(_fromType));
		templ("dstDataLocation", arrayDataAreaFunction(_toType));
		if (fromMemory || (fromCalldata && _fromType.baseType()->isValueType()))
			templ("readFromMemoryOrCalldata", readFromMemoryOrCalldata(*_fromType.baseType(), fromCalldata));
		templ("stackItems", suffixedVariableNameList(
			"stackItem_",
			0,
			_fromType.baseType()->stackItems().size()
		));
		templ("updateStorageValue", updateStorageValueFunction(*_fromType.baseType(), *_toType.baseType(), VariableDeclaration::Location::Unspecified, 0));
		templ("srcStride",
			fromCalldata ?
			std::to_string(_fromType.calldataStride()) :
				fromMemory ?
				std::to_string(_fromType.memoryStride()) :
				formatNumber(_fromType.baseType()->storageSize())
		);
		templ("storageSize", _toType.baseType()->storageSize().str());

		return templ.render();
	});
}


std::string YulUtilFunctions::copyByteArrayToStorageFunction(ArrayType const& _fromType, ArrayType const& _toType)
{
	solAssert(
		(*_fromType.copyForLocation(_toType.location(), _toType.isPointer())).equals(dynamic_cast<ReferenceType const&>(_toType)),
		""
	);
	solAssert(_fromType.isByteArrayOrString(), "");
	solAssert(_toType.isByteArrayOrString(), "");

	std::string functionName = "copy_byte_array_to_storage_from_" + _fromType.identifier() + "_to_" + _toType.identifier();
	return m_functionCollector.createFunction(functionName, [&](){
		Whiskers templ(R"(
			function <functionName>(slot, src<?fromCalldata>, len</fromCalldata>) {
				<?fromStorage> if eq(slot, src) { leave } </fromStorage>

				let newLen := <arrayLength>(src<?fromCalldata>, len</fromCalldata>)
				// Make sure array length is sane
				if gt(newLen, 0xffffffffffffffff) { <panic>() }

				let oldLen := <byteArrayLength>(sload(slot))

				// potentially truncate data
				<cleanUpEndArray>(slot, oldLen, newLen)

				let srcOffset := 0
				<?fromMemory>
					srcOffset := 0x20
				</fromMemory>

				switch gt(newLen, 31)
				case 1 {
					let loopEnd := and(newLen, not(0x1f))
					<?fromStorage> src := <srcDataLocation>(src) </fromStorage>
					let dstPtr := <dstDataLocation>(slot)
					let i := 0
					for { } lt(i, loopEnd) { i := add(i, 0x20) } {
						sstore(dstPtr, <read>(add(src, srcOffset)))
						dstPtr := add(dstPtr, 1)
						srcOffset := add(srcOffset, <srcIncrement>)
					}
					if lt(loopEnd, newLen) {
						let lastValue := <read>(add(src, srcOffset))
						sstore(dstPtr, <maskBytes>(lastValue, and(newLen, 0x1f)))
					}
					sstore(slot, add(mul(newLen, 2), 1))
				}
				default {
					let value := 0
					if newLen {
						value := <read>(add(src, srcOffset))
					}
					sstore(slot, <byteArrayCombineShort>(value, newLen))
				}
			}
		)");
		templ("functionName", functionName);
		bool fromStorage = _fromType.dataStoredIn(DataLocation::Storage);
		templ("fromStorage", fromStorage);
		bool fromCalldata = _fromType.dataStoredIn(DataLocation::CallData);
		templ("fromMemory", _fromType.dataStoredIn(DataLocation::Memory));
		templ("fromCalldata", fromCalldata);
		templ("arrayLength", arrayLengthFunction(_fromType));
		templ("panic", panicFunction(PanicCode::ResourceError));
		templ("byteArrayLength", extractByteArrayLengthFunction());
		templ("dstDataLocation", arrayDataAreaFunction(_toType));
		if (fromStorage)
			templ("srcDataLocation", arrayDataAreaFunction(_fromType));
		templ("cleanUpEndArray", cleanUpDynamicByteArrayEndSlotsFunction(_toType));
		templ("srcIncrement", std::to_string(fromStorage ? 1 : 0x20));
		templ("read", fromStorage ? "sload" : fromCalldata ? "calldataload" : "mload");
		templ("maskBytes", maskBytesFunctionDynamic());
		templ("byteArrayCombineShort", shortByteArrayEncodeUsedAreaSetLengthFunction());

		return templ.render();
	});
}

std::string YulUtilFunctions::copyValueArrayToStorageFunction(ArrayType const& _fromType, ArrayType const& _toType)
{
	solAssert(_fromType.baseType()->isValueType(), "");
	solAssert(_toType.baseType()->isValueType(), "");
	solAssert(_fromType.baseType()->isImplicitlyConvertibleTo(*_toType.baseType()), "");

	solAssert(!_fromType.isByteArrayOrString(), "");
	solAssert(!_toType.isByteArrayOrString(), "");
	solAssert(_toType.dataStoredIn(DataLocation::Storage), "");

	solAssert(_fromType.storageStride() <= _toType.storageStride(), "");
	solAssert(_toType.storageStride() <= 32, "");

	std::string functionName = "copy_array_to_storage_from_" + _fromType.identifier() + "_to_" + _toType.identifier();
	return m_functionCollector.createFunction(functionName, [&](){
		Whiskers templ(R"(
			function <functionName>(dst, src<?isFromDynamicCalldata>, len</isFromDynamicCalldata>) {
				<?isFromStorage>
				if eq(dst, src) { leave }
				</isFromStorage>
				let length := <arrayLength>(src<?isFromDynamicCalldata>, len</isFromDynamicCalldata>)
				// Make sure array length is sane
				if gt(length, 0xffffffffffffffff) { <panic>() }
				<resizeArray>(dst, length)

				let srcPtr := <srcDataLocation>(src)
				let dstSlot := <dstDataLocation>(dst)

				let fullSlots := div(length, <itemsPerSlot>)

				<?isFromStorage>
				let srcSlotValue := sload(srcPtr)
				let srcItemIndexInSlot := 0
				</isFromStorage>

				for { let i := 0 } lt(i, fullSlots) { i := add(i, 1) } {
					let dstSlotValue := 0
					<?sameTypeFromStorage>
						dstSlotValue := <maskFull>(srcSlotValue)
						<updateSrcPtr>
					<!sameTypeFromStorage>
						<?multipleItemsPerSlotDst>for { let j := 0 } lt(j, <itemsPerSlot>) { j := add(j, 1) } </multipleItemsPerSlotDst>
						{
							<?isFromStorage>
							let <stackItems> := <convert>(
								<extractFromSlot>(srcSlotValue, mul(<srcStride>, srcItemIndexInSlot))
							)
							<!isFromStorage>
							let <stackItems> := <readFromMemoryOrCalldata>(srcPtr)
							</isFromStorage>
							let itemValue := <prepareStore>(<stackItems>)
							dstSlotValue :=
							<?multipleItemsPerSlotDst>
								<updateByteSlice>(dstSlotValue, mul(<dstStride>, j), itemValue)
							<!multipleItemsPerSlotDst>
								itemValue
							</multipleItemsPerSlotDst>

							<updateSrcPtr>
						}
					</sameTypeFromStorage>

					sstore(add(dstSlot, i), dstSlotValue)
				}

				<?multipleItemsPerSlotDst>
					let spill := sub(length, mul(fullSlots, <itemsPerSlot>))
					if gt(spill, 0) {
						let dstSlotValue := 0
						<?sameTypeFromStorage>
							dstSlotValue := <maskBytes>(srcSlotValue, mul(spill, <srcStride>))
							<updateSrcPtr>
						<!sameTypeFromStorage>
							for { let j := 0 } lt(j, spill) { j := add(j, 1) } {
								<?isFromStorage>
								let <stackItems> := <convert>(
									<extractFromSlot>(srcSlotValue, mul(<srcStride>, srcItemIndexInSlot))
								)
								<!isFromStorage>
								let <stackItems> := <readFromMemoryOrCalldata>(srcPtr)
								</isFromStorage>
								let itemValue := <prepareStore>(<stackItems>)
								dstSlotValue := <updateByteSlice>(dstSlotValue, mul(<dstStride>, j), itemValue)

								<updateSrcPtr>
							}
						</sameTypeFromStorage>
						sstore(add(dstSlot, fullSlots), dstSlotValue)
					}
				</multipleItemsPerSlotDst>
			}
		)");
		if (_fromType.dataStoredIn(DataLocation::Storage))
			solAssert(!_fromType.isValueType(), "");

		bool fromCalldata = _fromType.dataStoredIn(DataLocation::CallData);
		bool fromStorage = _fromType.dataStoredIn(DataLocation::Storage);
		templ("functionName", functionName);
		templ("resizeArray", resizeArrayFunction(_toType));
		templ("arrayLength", arrayLengthFunction(_fromType));
		templ("panic", panicFunction(PanicCode::ResourceError));
		templ("isFromDynamicCalldata", _fromType.isDynamicallySized() && fromCalldata);
		templ("isFromStorage", fromStorage);
		templ("readFromMemoryOrCalldata", readFromMemoryOrCalldata(*_fromType.baseType(), fromCalldata));
		templ("srcDataLocation", arrayDataAreaFunction(_fromType));
		templ("dstDataLocation", arrayDataAreaFunction(_toType));
		templ("srcStride", std::to_string(_fromType.storageStride()));
		templ("stackItems", suffixedVariableNameList(
			"stackItem_",
			0,
			_fromType.baseType()->stackItems().size()
		));
		unsigned itemsPerSlot = 32 / _toType.storageStride();
		templ("itemsPerSlot", std::to_string(itemsPerSlot));
		templ("multipleItemsPerSlotDst", itemsPerSlot > 1);
		bool sameTypeFromStorage = fromStorage && (*_fromType.baseType() == *_toType.baseType());
		if (auto functionType = dynamic_cast<FunctionType const*>(_fromType.baseType()))
		{
			solAssert(functionType->equalExcludingStateMutability(
				dynamic_cast<FunctionType const&>(*_toType.baseType())
			));
			sameTypeFromStorage = fromStorage;
		}
		templ("sameTypeFromStorage", sameTypeFromStorage);
		if (sameTypeFromStorage)
		{
			templ("maskFull", maskLowerOrderBytesFunction(itemsPerSlot * _toType.storageStride()));
			templ("maskBytes", maskLowerOrderBytesFunctionDynamic());
		}
		else
		{
			templ("dstStride", std::to_string(_toType.storageStride()));
			templ("extractFromSlot", extractFromStorageValueDynamic(*_fromType.baseType()));
			templ("updateByteSlice", updateByteSliceFunctionDynamic(_toType.storageStride()));
			templ("convert", conversionFunction(*_fromType.baseType(), *_toType.baseType()));
			templ("prepareStore", prepareStoreFunction(*_toType.baseType()));
		}
		if (fromStorage)
			templ("updateSrcPtr", Whiskers(R"(
				<?srcReadMultiPerSlot>
					srcItemIndexInSlot := add(srcItemIndexInSlot, 1)
					if eq(srcItemIndexInSlot, <srcItemsPerSlot>) {
						// here we are done with this slot, we need to read next one
						srcPtr := add(srcPtr, 1)
						srcSlotValue := sload(srcPtr)
						srcItemIndexInSlot := 0
					}
				<!srcReadMultiPerSlot>
					srcPtr := add(srcPtr, 1)
					srcSlotValue := sload(srcPtr)
				</srcReadMultiPerSlot>
				)")
				("srcReadMultiPerSlot", !sameTypeFromStorage && _fromType.storageStride() <= 16)
				("srcItemsPerSlot", std::to_string(32 / _fromType.storageStride()))
				.render()
			);
		else
			templ("updateSrcPtr", Whiskers(R"(
					srcPtr := add(srcPtr, <srcStride>)
				)")
				("srcStride", fromCalldata ? std::to_string(_fromType.calldataStride()) : std::to_string(_fromType.memoryStride()))
				.render()
			);

		return templ.render();
	});
}


std::string YulUtilFunctions::arrayConvertLengthToSize(ArrayType const& _type)
{
	std::string functionName = "array_convert_length_to_size_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Type const& baseType = *_type.baseType();

		switch (_type.location())
		{
			case DataLocation::Storage:
			{
				unsigned const baseStorageBytes = baseType.storageBytes();
				solAssert(baseStorageBytes > 0, "");
				solAssert(32 / baseStorageBytes > 0, "");

				return Whiskers(R"(
					function <functionName>(length) -> size {
						size := length
						<?multiSlot>
							size := <mul>(<storageSize>, length)
						<!multiSlot>
							// Number of slots rounded up
							size := div(add(length, sub(<itemsPerSlot>, 1)), <itemsPerSlot>)
						</multiSlot>
					})")
					("functionName", functionName)
					("multiSlot", baseType.storageSize() > 1)
					("itemsPerSlot", std::to_string(32 / baseStorageBytes))
					("storageSize", baseType.storageSize().str())
					("mul", overflowCheckedIntMulFunction(*TypeProvider::uint256()))
					.render();
			}
			case DataLocation::CallData: // fallthrough
			case DataLocation::Memory:
				return Whiskers(R"(
					function <functionName>(length) -> size {
						<?byteArray>
							size := length
						<!byteArray>
							size := <mul>(length, <stride>)
						</byteArray>
					})")
					("functionName", functionName)
					("stride", std::to_string(_type.location() == DataLocation::Memory ? _type.memoryStride() : _type.calldataStride()))
					("byteArray", _type.isByteArrayOrString())
					("mul", overflowCheckedIntMulFunction(*TypeProvider::uint256()))
					.render();
			default:
				solAssert(false, "");
		}

	});
}

std::string YulUtilFunctions::arrayAllocationSizeFunction(ArrayType const& _type)
{
	solAssert(_type.dataStoredIn(DataLocation::Memory), "");
	std::string functionName = "array_allocation_size_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers w(R"(
			function <functionName>(length) -> size {
				// Make sure we can allocate memory without overflow
				if gt(length, 0xffffffffffffffff) { <panic>() }
				<?byteArray>
					size := <roundUp>(length)
				<!byteArray>
					size := mul(length, 0x20)
				</byteArray>
				<?dynamic>
					// add length slot
					size := add(size, 0x20)
				</dynamic>
			}
		)");
		w("functionName", functionName);
		w("panic", panicFunction(PanicCode::ResourceError));
		w("byteArray", _type.isByteArrayOrString());
		w("roundUp", roundUpFunction());
		w("dynamic", _type.isDynamicallySized());
		return w.render();
	});
}

std::string YulUtilFunctions::arrayDataAreaFunction(ArrayType const& _type)
{
	std::string functionName = "array_dataslot_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		// No special processing for calldata arrays, because they are stored as
		// offset of the data area and length on the stack, so the offset already
		// points to the data area.
		// This might change, if calldata arrays are stored in a single
		// stack slot at some point.
		return Whiskers(R"(
			function <functionName>(ptr) -> data {
				data := ptr
				<?dynamic>
					<?memory>
						data := add(ptr, 0x20)
					</memory>
					<?storage>
						mstore(0, ptr)
						data := keccak256(0, 0x20)
					</storage>
				</dynamic>
			}
		)")
		("functionName", functionName)
		("dynamic", _type.isDynamicallySized())
		("memory", _type.location() == DataLocation::Memory)
		("storage", _type.location() == DataLocation::Storage)
		.render();
	});
}

std::string YulUtilFunctions::storageArrayIndexAccessFunction(ArrayType const& _type)
{
	std::string functionName = "storage_array_index_access_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(array, index) -> slot, offset {
				let arrayLength := <arrayLen>(array)
				if iszero(lt(index, arrayLength)) { <panic>() }

				<?multipleItemsPerSlot>
					<?isBytesArray>
						switch lt(arrayLength, 0x20)
						case 0 {
							slot, offset := <indexAccessNoChecks>(array, index)
						}
						default {
							offset := sub(31, mod(index, 0x20))
							slot := array
						}
					<!isBytesArray>
						let dataArea := <dataAreaFunc>(array)
						slot := add(dataArea, div(index, <itemsPerSlot>))
						offset := mul(mod(index, <itemsPerSlot>), <storageBytes>)
					</isBytesArray>
				<!multipleItemsPerSlot>
					let dataArea := <dataAreaFunc>(array)
					slot := add(dataArea, mul(index, <storageSize>))
					offset := 0
				</multipleItemsPerSlot>
			}
		)")
		("functionName", functionName)
		("panic", panicFunction(PanicCode::ArrayOutOfBounds))
		("arrayLen", arrayLengthFunction(_type))
		("dataAreaFunc", arrayDataAreaFunction(_type))
		("indexAccessNoChecks", longByteArrayStorageIndexAccessNoCheckFunction())
		("multipleItemsPerSlot", _type.baseType()->storageBytes() <= 16)
		("isBytesArray", _type.isByteArrayOrString())
		("storageSize", _type.baseType()->storageSize().str())
		("storageBytes", toString(_type.baseType()->storageBytes()))
		("itemsPerSlot", std::to_string(32 / _type.baseType()->storageBytes()))
		.render();
	});
}

std::string YulUtilFunctions::memoryArrayIndexAccessFunction(ArrayType const& _type)
{
	std::string functionName = "memory_array_index_access_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(baseRef, index) -> addr {
				if iszero(lt(index, <arrayLen>(baseRef))) {
					<panic>()
				}

				let offset := mul(index, <stride>)
				<?dynamicallySized>
					offset := add(offset, 32)
				</dynamicallySized>
				addr := add(baseRef, offset)
			}
		)")
		("functionName", functionName)
		("panic", panicFunction(PanicCode::ArrayOutOfBounds))
		("arrayLen", arrayLengthFunction(_type))
		("stride", std::to_string(_type.memoryStride()))
		("dynamicallySized", _type.isDynamicallySized())
		.render();
	});
}

std::string YulUtilFunctions::calldataArrayIndexAccessFunction(ArrayType const& _type)
{
	solAssert(_type.dataStoredIn(DataLocation::CallData), "");
	std::string functionName = "calldata_array_index_access_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(base_ref<?dynamicallySized>, length</dynamicallySized>, index) -> addr<?dynamicallySizedBase>, len</dynamicallySizedBase> {
				if iszero(lt(index, <?dynamicallySized>length<!dynamicallySized><arrayLen></dynamicallySized>)) { <panic>() }
				addr := add(base_ref, mul(index, <stride>))
				<?dynamicallyEncodedBase>
					addr<?dynamicallySizedBase>, len</dynamicallySizedBase> := <accessCalldataTail>(base_ref, addr)
				</dynamicallyEncodedBase>
			}
		)")
		("functionName", functionName)
		("panic", panicFunction(PanicCode::ArrayOutOfBounds))
		("stride", std::to_string(_type.calldataStride()))
		("dynamicallySized", _type.isDynamicallySized())
		("dynamicallyEncodedBase", _type.baseType()->isDynamicallyEncoded())
		("dynamicallySizedBase", _type.baseType()->isDynamicallySized())
		("arrayLen",  toCompactHexWithPrefix(_type.length()))
		("accessCalldataTail", _type.baseType()->isDynamicallyEncoded() ? accessCalldataTailFunction(*_type.baseType()): "")
		.render();
	});
}

std::string YulUtilFunctions::calldataArrayIndexRangeAccess(ArrayType const& _type)
{
	solAssert(_type.dataStoredIn(DataLocation::CallData), "");
	solAssert(_type.isDynamicallySized(), "");
	std::string functionName = "calldata_array_index_range_access_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(offset, length, startIndex, endIndex) -> offsetOut, lengthOut {
				if gt(startIndex, endIndex) { <revertSliceStartAfterEnd>() }
				if gt(endIndex, length) { <revertSliceGreaterThanLength>() }
				offsetOut := add(offset, mul(startIndex, <stride>))
				lengthOut := sub(endIndex, startIndex)
			}
		)")
		("functionName", functionName)
		("stride", std::to_string(_type.calldataStride()))
		("revertSliceStartAfterEnd", revertReasonIfDebugFunction("Slice starts after end"))
		("revertSliceGreaterThanLength", revertReasonIfDebugFunction("Slice is greater than length"))
		.render();
	});
}

std::string YulUtilFunctions::accessCalldataTailFunction(Type const& _type)
{
	solAssert(_type.isDynamicallyEncoded(), "");
	solAssert(_type.dataStoredIn(DataLocation::CallData), "");
	std::string functionName = "access_calldata_tail_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(base_ref, ptr_to_tail) -> addr<?dynamicallySized>, length</dynamicallySized> {
				let rel_offset_of_tail := calldataload(ptr_to_tail)
				if iszero(slt(rel_offset_of_tail, sub(sub(calldatasize(), base_ref), sub(<neededLength>, 1)))) { <invalidCalldataTailOffset>() }
				addr := add(base_ref, rel_offset_of_tail)
				<?dynamicallySized>
					length := calldataload(addr)
					if gt(length, 0xffffffffffffffff) { <invalidCalldataTailLength>() }
					addr := add(addr, 32)
					if sgt(addr, sub(calldatasize(), mul(length, <calldataStride>))) { <shortCalldataTail>() }
				</dynamicallySized>
			}
		)")
		("functionName", functionName)
		("dynamicallySized", _type.isDynamicallySized())
		("neededLength", toCompactHexWithPrefix(_type.calldataEncodedTailSize()))
		("calldataStride", toCompactHexWithPrefix(_type.isDynamicallySized() ? dynamic_cast<ArrayType const&>(_type).calldataStride() : 0))
		("invalidCalldataTailOffset", revertReasonIfDebugFunction("Invalid calldata tail offset"))
		("invalidCalldataTailLength", revertReasonIfDebugFunction("Invalid calldata tail length"))
		("shortCalldataTail", revertReasonIfDebugFunction("Calldata tail too short"))
		.render();
	});
}

std::string YulUtilFunctions::nextArrayElementFunction(ArrayType const& _type)
{
	solAssert(!_type.isByteArrayOrString(), "");
	if (_type.dataStoredIn(DataLocation::Storage))
		solAssert(_type.baseType()->storageBytes() > 16, "");
	std::string functionName = "array_nextElement_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(ptr) -> next {
				next := add(ptr, <advance>)
			}
		)");
		templ("functionName", functionName);
		switch (_type.location())
		{
		case DataLocation::Memory:
			templ("advance", "0x20");
			break;
		case DataLocation::Storage:
		{
			u256 size = _type.baseType()->storageSize();
			solAssert(size >= 1, "");
			templ("advance", toCompactHexWithPrefix(size));
			break;
		}
		case DataLocation::Transient:
			solUnimplemented("Transient data location is only supported for value types.");
			break;
		case DataLocation::CallData:
		{
			u256 size = _type.calldataStride();
			solAssert(size >= 32 && size % 32 == 0, "");
			templ("advance", toCompactHexWithPrefix(size));
			break;
		}
		}
		return templ.render();
	});
}

std::string YulUtilFunctions::copyArrayFromStorageToMemoryFunction(ArrayType const& _from, ArrayType const& _to)
{
	solAssert(_from.dataStoredIn(DataLocation::Storage), "");
	solAssert(_to.dataStoredIn(DataLocation::Memory), "");
	solAssert(_from.isDynamicallySized() == _to.isDynamicallySized(), "");
	if (!_from.isDynamicallySized())
		solAssert(_from.length() == _to.length(), "");

	std::string functionName = "copy_array_from_storage_to_memory_" + _from.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		if (_from.baseType()->isValueType())
		{
			solAssert(*_from.baseType() == *_to.baseType(), "");
			ABIFunctions abi(m_evmVersion, m_revertStrings, m_functionCollector);
			return Whiskers(R"(
				function <functionName>(slot) -> memPtr {
					memPtr := <allocateUnbounded>()
					let end := <encode>(slot, memPtr)
					<finalizeAllocation>(memPtr, sub(end, memPtr))
				}
			)")
			("functionName", functionName)
			("allocateUnbounded", allocateUnboundedFunction())
			(
				"encode",
				abi.abiEncodeAndReturnUpdatedPosFunction(_from, _to, ABIFunctions::EncodingOptions{})
			)
			("finalizeAllocation", finalizeAllocationFunction())
			.render();
		}
		else
		{
			solAssert(_to.memoryStride() == 32, "");
			solAssert(_to.baseType()->dataStoredIn(DataLocation::Memory), "");
			solAssert(_from.baseType()->dataStoredIn(DataLocation::Storage), "");
			solAssert(!_from.isByteArrayOrString(), "");
			solAssert(*_to.withLocation(DataLocation::Storage, _from.isPointer()) == _from, "");
			return Whiskers(R"(
				function <functionName>(slot) -> memPtr {
					let length := <lengthFunction>(slot)
					memPtr := <allocateArray>(length)
					let mpos := memPtr
					<?dynamic>mpos := add(mpos, 0x20)</dynamic>
					let spos := <arrayDataArea>(slot)
					for { let i := 0 } lt(i, length) { i := add(i, 1) } {
						mstore(mpos, <convert>(spos))
						mpos := add(mpos, 0x20)
						spos := add(spos, <baseStorageSize>)
					}
				}
			)")
			("functionName", functionName)
			("lengthFunction", arrayLengthFunction(_from))
			("allocateArray", allocateMemoryArrayFunction(_to))
			("arrayDataArea", arrayDataAreaFunction(_from))
			("dynamic", _to.isDynamicallySized())
			("convert", conversionFunction(*_from.baseType(), *_to.baseType()))
			("baseStorageSize", _from.baseType()->storageSize().str())
			.render();
		}
	});
}

std::string YulUtilFunctions::bytesOrStringConcatFunction(
	std::vector<Type const*> const& _argumentTypes,
	FunctionType::Kind _functionTypeKind
)
{
	solAssert(_functionTypeKind == FunctionType::Kind::BytesConcat || _functionTypeKind == FunctionType::Kind::StringConcat);
	std::string functionName = (_functionTypeKind == FunctionType::Kind::StringConcat) ? "string_concat" : "bytes_concat";
	size_t totalParams = 0;
	std::vector<Type const*> targetTypes;

	for (Type const* argumentType: _argumentTypes)
	{
		if (_functionTypeKind == FunctionType::Kind::StringConcat)
			solAssert(argumentType->isImplicitlyConvertibleTo(*TypeProvider::stringMemory()));
		else if (_functionTypeKind == FunctionType::Kind::BytesConcat)
			solAssert(
				argumentType->isImplicitlyConvertibleTo(*TypeProvider::bytesMemory()) ||
				argumentType->isImplicitlyConvertibleTo(*TypeProvider::fixedBytes(32))
			);

		if (argumentType->category() == Type::Category::FixedBytes)
			targetTypes.emplace_back(argumentType);
		else if (
			auto const* literalType = dynamic_cast<StringLiteralType const*>(argumentType);
			literalType && !literalType->value().empty() && literalType->value().size() <= 32
		)
			targetTypes.emplace_back(TypeProvider::fixedBytes(static_cast<unsigned>(literalType->value().size())));
		else
		{
			solAssert(!dynamic_cast<RationalNumberType const*>(argumentType));
			targetTypes.emplace_back(
				_functionTypeKind == FunctionType::Kind::StringConcat ?
				TypeProvider::stringMemory() :
				TypeProvider::bytesMemory()
			);
		}
		totalParams += argumentType->sizeOnStack();
		functionName += "_" + argumentType->identifier();
	}
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(<parameters>) -> outPtr {
				outPtr := <allocateUnbounded>()
				let dataStart := add(outPtr, 0x20)
				let dataEnd := <encodePacked>(dataStart<?+parameters>, <parameters></+parameters>)
				mstore(outPtr, sub(dataEnd, dataStart))
				<finalizeAllocation>(outPtr, sub(dataEnd, outPtr))
			}
		)");
		templ("functionName", functionName);
		templ("parameters", suffixedVariableNameList("param_", 0, totalParams));
		templ("allocateUnbounded", allocateUnboundedFunction());
		templ("finalizeAllocation", finalizeAllocationFunction());
		templ(
			"encodePacked",
			ABIFunctions{m_evmVersion, m_revertStrings, m_functionCollector}.tupleEncoderPacked(
				_argumentTypes,
				targetTypes
			)
		);
		return templ.render();
	});
}

std::string YulUtilFunctions::mappingIndexAccessFunction(MappingType const& _mappingType, Type const& _keyType)
{
	std::string functionName = "mapping_index_access_" + _mappingType.identifier() + "_of_" + _keyType.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		if (_mappingType.keyType()->isDynamicallySized())
			return Whiskers(R"(
				function <functionName>(slot <?+key>,</+key> <key>) -> dataSlot {
					dataSlot := <hash>(<key> <?+key>,</+key> slot)
				}
			)")
			("functionName", functionName)
			("key", suffixedVariableNameList("key_", 0, _keyType.sizeOnStack()))
			("hash", packedHashFunction(
				{&_keyType, TypeProvider::uint256()},
				{_mappingType.keyType(), TypeProvider::uint256()}
			))
			.render();
		else
		{
			solAssert(CompilerUtils::freeMemoryPointer >= 0x40, "");
			solAssert(!_mappingType.keyType()->isDynamicallyEncoded(), "");
			solAssert(_mappingType.keyType()->calldataEncodedSize(false) <= 0x20, "");
			Whiskers templ(R"(
				function <functionName>(slot <key>) -> dataSlot {
					mstore(0, <convertedKey>)
					mstore(0x20, slot)
					dataSlot := keccak256(0, 0x40)
				}
			)");
			templ("functionName", functionName);
			templ("key", _keyType.sizeOnStack() == 1 ? ", key" : "");
			if (_keyType.sizeOnStack() == 0)
				templ("convertedKey", conversionFunction(_keyType, *_mappingType.keyType()) + "()");
			else
				templ("convertedKey", conversionFunction(_keyType, *_mappingType.keyType()) + "(key)");
			return templ.render();
		}
	});
}

std::string YulUtilFunctions::readFromStorage(
	Type const& _type,
	size_t _offset,
	bool _splitFunctionTypes,
	VariableDeclaration::Location _location
)
{
	if (_type.isValueType())
		return readFromStorageValueType(_type, _offset, _splitFunctionTypes, _location);
	else
	{
		solAssert(_location != VariableDeclaration::Location::Transient);
		solAssert(_offset == 0, "");
		return readFromStorageReferenceType(_type);
	}
}

std::string YulUtilFunctions::readFromStorageDynamic(
	Type const& _type,
	bool _splitFunctionTypes,
	VariableDeclaration::Location _location
)
{
	if (_type.isValueType())
		return readFromStorageValueType(_type, {}, _splitFunctionTypes, _location);

	solAssert(_location != VariableDeclaration::Location::Transient);
	std::string functionName =
		"read_from_storage__dynamic_" +
		std::string(_splitFunctionTypes ? "split_" : "") +
		_type.identifier();

	return m_functionCollector.createFunction(functionName, [&] {
		return Whiskers(R"(
			function <functionName>(slot, offset) -> value {
				if gt(offset, 0) { <panic>() }
				value := <readFromStorage>(slot)
			}
		)")
		("functionName", functionName)
		("panic", panicFunction(util::PanicCode::Generic))
		("readFromStorage", readFromStorageReferenceType(_type))
		.render();
	});
}

std::string YulUtilFunctions::readFromStorageValueType(
	Type const& _type,
	std::optional<size_t> _offset,
	bool _splitFunctionTypes,
	VariableDeclaration::Location _location
)
{
	solAssert(_type.isValueType(), "");
	solAssert(
		_location == VariableDeclaration::Location::Transient ||
		_location == VariableDeclaration::Location::Unspecified,
		"Variable location can only be transient or plain storage"
	);

	std::string functionName =
		"read_from_" +
		(_location == VariableDeclaration::Location::Transient ? "transient_"s : "") +
		"storage_" +
		std::string(_splitFunctionTypes ? "split_" : "") + (
			_offset.has_value() ?
			"offset_" + std::to_string(*_offset) :
			"dynamic"
		) +
		"_" +
		_type.identifier();

	return m_functionCollector.createFunction(functionName, [&] {
		Whiskers templ(R"(
			function <functionName>(slot<?dynamic>, offset</dynamic>) -> <?split>addr, selector<!split>value</split> {
				<?split>let</split> value := <extract>(<loadOpcode>(slot)<?dynamic>, offset</dynamic>)
				<?split>
					addr, selector := <splitFunction>(value)
				</split>
			}
		)");
		templ("functionName", functionName);
		templ("dynamic", !_offset.has_value());
		templ("loadOpcode", _location == VariableDeclaration::Location::Transient ? "tload" : "sload");
		if (_offset.has_value())
			templ("extract", extractFromStorageValue(_type, *_offset));
		else
			templ("extract", extractFromStorageValueDynamic(_type));
		auto const* funType = dynamic_cast<FunctionType const*>(&_type);
		bool split = _splitFunctionTypes && funType && funType->kind() == FunctionType::Kind::External;
		templ("split", split);
		if (split)
			templ("splitFunction", splitExternalFunctionIdFunction());
		return templ.render();
	});
}

std::string YulUtilFunctions::readFromStorageReferenceType(Type const& _type)
{
	if (auto const* arrayType = dynamic_cast<ArrayType const*>(&_type))
	{
		solAssert(arrayType->dataStoredIn(DataLocation::Memory), "");
		return copyArrayFromStorageToMemoryFunction(
			dynamic_cast<ArrayType const&>(*arrayType->copyForLocation(DataLocation::Storage, false)),
			*arrayType
		);
	}
	solAssert(_type.category() == Type::Category::Struct, "");

	std::string functionName = "read_from_storage_reference_type_" + _type.identifier();

	auto const& structType = dynamic_cast<StructType const&>(_type);
	solAssert(structType.location() == DataLocation::Memory, "");
	MemberList::MemberMap structMembers = structType.nativeMembers(nullptr);
	std::vector<std::map<std::string, std::string>> memberSetValues(structMembers.size());
	for (size_t i = 0; i < structMembers.size(); ++i)
	{
		auto const& [memberSlotDiff, memberStorageOffset] = structType.storageOffsetsOfMember(structMembers[i].name);
		solAssert(structMembers[i].type->isValueType() || memberStorageOffset == 0, "");

		memberSetValues[i]["setMember"] = Whiskers(R"(
			{
				let <memberValues> := <readFromStorage>(add(slot, <memberSlotDiff>))
				<writeToMemory>(add(value, <memberMemoryOffset>), <memberValues>)
			}
		)")
		("memberValues", suffixedVariableNameList("memberValue_", 0, structMembers[i].type->stackItems().size()))
		("memberMemoryOffset", structType.memoryOffsetOfMember(structMembers[i].name).str())
		("memberSlotDiff",  memberSlotDiff.str())
		("readFromStorage", readFromStorage(*structMembers[i].type, memberStorageOffset, true, VariableDeclaration::Location::Unspecified))
		("writeToMemory", writeToMemoryFunction(*structMembers[i].type))
		.render();
	}

	return m_functionCollector.createFunction(functionName, [&] {
		return Whiskers(R"(
			function <functionName>(slot) -> value {
				value := <allocStruct>()
				<#member>
					<setMember>
				</member>
			}
		)")
		("functionName", functionName)
		("allocStruct", allocateMemoryStructFunction(structType))
		("member", memberSetValues)
		.render();
	});
}

std::string YulUtilFunctions::readFromMemory(Type const& _type)
{
	return readFromMemoryOrCalldata(_type, false);
}

std::string YulUtilFunctions::readFromCalldata(Type const& _type)
{
	return readFromMemoryOrCalldata(_type, true);
}

std::string YulUtilFunctions::updateStorageValueFunction(
	Type const& _fromType,
	Type const& _toType,
	VariableDeclaration::Location _location,
	std::optional<unsigned> const& _offset
)
{
	solAssert(
		_location == VariableDeclaration::Location::Transient ||
		_location == VariableDeclaration::Location::Unspecified,
		"Variable location can only be transient or plain storage"
	);

	std::string const functionName =
		"update_" +
		(_location == VariableDeclaration::Location::Transient ? "transient_"s : "") +
		"storage_value_" +
		(_offset.has_value() ? ("offset_" + std::to_string(*_offset)) + "_" : "") +
		_fromType.identifier() +
		"_to_" +
		_toType.identifier();

	return m_functionCollector.createFunction(functionName, [&] {
		if (_toType.isValueType())
		{
			solAssert(_fromType.isImplicitlyConvertibleTo(_toType), "");
			solAssert(_toType.storageBytes() <= 32, "Invalid storage bytes size.");
			solAssert(_toType.storageBytes() > 0, "Invalid storage bytes size.");

			return Whiskers(R"(
				function <functionName>(slot, <offset><fromValues>) {
					let <toValues> := <convert>(<fromValues>)
					<storeOpcode>(slot, <update>(<loadOpcode>(slot), <offset><prepare>(<toValues>)))
				}

			)")
			("functionName", functionName)
			("update",
				_offset.has_value() ?
					updateByteSliceFunction(_toType.storageBytes(), *_offset) :
					updateByteSliceFunctionDynamic(_toType.storageBytes())
			)
			("offset", _offset.has_value() ? "" : "offset, ")
			("convert", conversionFunction(_fromType, _toType))
			("fromValues", suffixedVariableNameList("value_", 0, _fromType.sizeOnStack()))
			("toValues", suffixedVariableNameList("convertedValue_", 0, _toType.sizeOnStack()))
			("storeOpcode", _location == VariableDeclaration::Location::Transient ? "tstore" : "sstore")
			("loadOpcode", _location == VariableDeclaration::Location::Transient ? "tload" : "sload")
			("prepare", prepareStoreFunction(_toType))
			.render();
		}

		solAssert(_location != VariableDeclaration::Location::Transient);
		auto const* toReferenceType = dynamic_cast<ReferenceType const*>(&_toType);
		auto const* fromReferenceType = dynamic_cast<ReferenceType const*>(&_fromType);
		solAssert(toReferenceType, "");

		if (!fromReferenceType)
		{
			solAssert(_fromType.category() == Type::Category::StringLiteral, "");
			solAssert(toReferenceType->category() == Type::Category::Array, "");
			auto const& toArrayType = dynamic_cast<ArrayType const&>(*toReferenceType);
			solAssert(toArrayType.isByteArrayOrString(), "");

			return Whiskers(R"(
				function <functionName>(slot<?dynamicOffset>, offset</dynamicOffset>) {
					<?dynamicOffset>if offset { <panic>() }</dynamicOffset>
					<copyToStorage>(slot)
				}
			)")
			("functionName", functionName)
			("dynamicOffset", !_offset.has_value())
			("panic", panicFunction(PanicCode::Generic))
			("copyToStorage", copyLiteralToStorageFunction(dynamic_cast<StringLiteralType const&>(_fromType).value()))
			.render();
		}

		solAssert((*toReferenceType->copyForLocation(
			fromReferenceType->location(),
			fromReferenceType->isPointer()
		).get()).equals(*fromReferenceType), "");

		if (fromReferenceType->category() == Type::Category::ArraySlice)
			solAssert(toReferenceType->category() == Type::Category::Array, "");
		else
			solAssert(toReferenceType->category() == fromReferenceType->category(), "");
		solAssert(_offset.value_or(0) == 0, "");

		Whiskers templ(R"(
			function <functionName>(slot, <?dynamicOffset>offset, </dynamicOffset><value>) {
				<?dynamicOffset>if offset { <panic>() }</dynamicOffset>
				<copyToStorage>(slot, <value>)
			}
		)");
		templ("functionName", functionName);
		templ("dynamicOffset", !_offset.has_value());
		templ("panic", panicFunction(PanicCode::Generic));
		templ("value", suffixedVariableNameList("value_", 0, _fromType.sizeOnStack()));
		if (_fromType.category() == Type::Category::Array)
			templ("copyToStorage", copyArrayToStorageFunction(
				dynamic_cast<ArrayType const&>(_fromType),
				dynamic_cast<ArrayType const&>(_toType)
			));
		else if (_fromType.category() == Type::Category::ArraySlice)
		{
			solAssert(
				_fromType.dataStoredIn(DataLocation::CallData),
				"Currently only calldata array slices are supported!"
			);
			templ("copyToStorage", copyArrayToStorageFunction(
				dynamic_cast<ArraySliceType const&>(_fromType).arrayType(),
				dynamic_cast<ArrayType const&>(_toType)
			));
		}
		else
			templ("copyToStorage", copyStructToStorageFunction(
				dynamic_cast<StructType const&>(_fromType),
				dynamic_cast<StructType const&>(_toType)
			));

		return templ.render();
	});
}

std::string YulUtilFunctions::writeToMemoryFunction(Type const& _type)
{
	std::string const functionName = "write_to_memory_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&] {
		solAssert(!dynamic_cast<StringLiteralType const*>(&_type), "");
		if (auto ref = dynamic_cast<ReferenceType const*>(&_type))
		{
			solAssert(
				ref->location() == DataLocation::Memory,
				"Can only update types with location memory."
			);

			return Whiskers(R"(
				function <functionName>(memPtr, value) {
					mstore(memPtr, value)
				}
			)")
			("functionName", functionName)
			.render();
		}
		else if (
			_type.category() == Type::Category::Function &&
			dynamic_cast<FunctionType const&>(_type).kind() == FunctionType::Kind::External
		)
		{
			return Whiskers(R"(
				function <functionName>(memPtr, addr, selector) {
					mstore(memPtr, <combine>(addr, selector))
				}
			)")
			("functionName", functionName)
			("combine", combineExternalFunctionIdFunction())
			.render();
		}
		else if (_type.isValueType())
		{
			return Whiskers(R"(
				function <functionName>(memPtr, value) {
					mstore(memPtr, <cleanup>(value))
				}
			)")
			("functionName", functionName)
			("cleanup", cleanupFunction(_type))
			.render();
		}
		else // Should never happen
		{
			solAssert(
				false,
				"Memory store of type " + _type.toString(true) + " not allowed."
			);
		}
	});
}

std::string YulUtilFunctions::extractFromStorageValueDynamic(Type const& _type)
{
	std::string functionName =
		"extract_from_storage_value_dynamic" +
		_type.identifier();
	return m_functionCollector.createFunction(functionName, [&] {
		return Whiskers(R"(
			function <functionName>(slot_value, offset) -> value {
				value := <cleanupStorage>(<shr>(mul(offset, 8), slot_value))
			}
		)")
		("functionName", functionName)
		("shr", shiftRightFunctionDynamic())
		("cleanupStorage", cleanupFromStorageFunction(_type))
		.render();
	});
}

std::string YulUtilFunctions::extractFromStorageValue(Type const& _type, size_t _offset)
{
	std::string functionName = "extract_from_storage_value_offset_" + std::to_string(_offset) + "_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&] {
		return Whiskers(R"(
			function <functionName>(slot_value) -> value {
				value := <cleanupStorage>(<shr>(slot_value))
			}
		)")
		("functionName", functionName)
		("shr", shiftRightFunction(_offset * 8))
		("cleanupStorage", cleanupFromStorageFunction(_type))
		.render();
	});
}

std::string YulUtilFunctions::cleanupFromStorageFunction(Type const& _type)
{
	solAssert(_type.isValueType(), "");

	std::string functionName = std::string("cleanup_from_storage_") + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&] {
		Whiskers templ(R"(
			function <functionName>(value) -> cleaned {
				cleaned := <cleaned>
			}
		)");
		templ("functionName", functionName);

		Type const* encodingType = &_type;
		if (_type.category() == Type::Category::UserDefinedValueType)
			encodingType = _type.encodingType();
		unsigned storageBytes = encodingType->storageBytes();
		if (IntegerType const* intType = dynamic_cast<IntegerType const*>(encodingType))
			if (intType->isSigned() && storageBytes != 32)
			{
				templ("cleaned", "signextend(" + std::to_string(storageBytes - 1) + ", value)");
				return templ.render();
			}

		if (storageBytes == 32)
			templ("cleaned", "value");
		else if (encodingType->leftAligned())
			templ("cleaned", shiftLeftFunction(256 - 8 * storageBytes) + "(value)");
		else
			templ("cleaned", "and(value, " + toCompactHexWithPrefix((u256(1) << (8 * storageBytes)) - 1) + ")");

		return templ.render();
	});
}

std::string YulUtilFunctions::prepareStoreFunction(Type const& _type)
{
	std::string functionName = "prepare_store_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		solAssert(_type.isValueType(), "");
		auto const* funType = dynamic_cast<FunctionType const*>(&_type);
		if (funType && funType->kind() == FunctionType::Kind::External)
		{
			Whiskers templ(R"(
				function <functionName>(addr, selector) -> ret {
					ret := <prepareBytes>(<combine>(addr, selector))
				}
			)");
			templ("functionName", functionName);
			templ("prepareBytes", prepareStoreFunction(*TypeProvider::fixedBytes(24)));
			templ("combine", combineExternalFunctionIdFunction());
			return templ.render();
		}
		else
		{
			solAssert(_type.sizeOnStack() == 1, "");
			Whiskers templ(R"(
				function <functionName>(value) -> ret {
					ret := <actualPrepare>
				}
			)");
			templ("functionName", functionName);
			if (_type.leftAligned())
				templ("actualPrepare", shiftRightFunction(256 - 8 * _type.storageBytes()) + "(value)");
			else
				templ("actualPrepare", "value");
			return templ.render();
		}
	});
}

std::string YulUtilFunctions::allocationFunction()
{
	std::string functionName = "allocate_memory";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(size) -> memPtr {
				memPtr := <allocateUnbounded>()
				<finalizeAllocation>(memPtr, size)
			}
		)")
		("functionName", functionName)
		("allocateUnbounded", allocateUnboundedFunction())
		("finalizeAllocation", finalizeAllocationFunction())
		.render();
	});
}

std::string YulUtilFunctions::allocateUnboundedFunction()
{
	std::string functionName = "allocate_unbounded";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>() -> memPtr {
				memPtr := mload(<freeMemoryPointer>)
			}
		)")
		("freeMemoryPointer", std::to_string(CompilerUtils::freeMemoryPointer))
		("functionName", functionName)
		.render();
	});
}

std::string YulUtilFunctions::finalizeAllocationFunction()
{
	std::string functionName = "finalize_allocation";
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(memPtr, size) {
				let newFreePtr := add(memPtr, <roundUp>(size))
				// protect against overflow
				if or(gt(newFreePtr, 0xffffffffffffffff), lt(newFreePtr, memPtr)) { <panic>() }
				mstore(<freeMemoryPointer>, newFreePtr)
			}
		)")
		("functionName", functionName)
		("freeMemoryPointer", std::to_string(CompilerUtils::freeMemoryPointer))
		("roundUp", roundUpFunction())
		("panic", panicFunction(PanicCode::ResourceError))
		.render();
	});
}

std::string YulUtilFunctions::zeroMemoryArrayFunction(ArrayType const& _type)
{
	if (_type.baseType()->hasSimpleZeroValueInMemory())
		return zeroMemoryFunction(*_type.baseType());
	return zeroComplexMemoryArrayFunction(_type);
}

std::string YulUtilFunctions::zeroMemoryFunction(Type const& _type)
{
	solAssert(_type.hasSimpleZeroValueInMemory(), "");

	std::string functionName = "zero_memory_chunk_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(dataStart, dataSizeInBytes) {
				calldatacopy(dataStart, calldatasize(), dataSizeInBytes)
			}
		)")
		("functionName", functionName)
		.render();
	});
}

std::string YulUtilFunctions::zeroComplexMemoryArrayFunction(ArrayType const& _type)
{
	solAssert(!_type.baseType()->hasSimpleZeroValueInMemory(), "");

	std::string functionName = "zero_complex_memory_array_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		solAssert(_type.memoryStride() == 32, "");
		return Whiskers(R"(
			function <functionName>(dataStart, dataSizeInBytes) {
				for {let i := 0} lt(i, dataSizeInBytes) { i := add(i, <stride>) } {
					mstore(add(dataStart, i), <zeroValue>())
				}
			}
		)")
		("functionName", functionName)
		("stride", std::to_string(_type.memoryStride()))
		("zeroValue", zeroValueFunction(*_type.baseType(), false))
		.render();
	});
}

std::string YulUtilFunctions::allocateMemoryArrayFunction(ArrayType const& _type)
{
	std::string functionName = "allocate_memory_array_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
				function <functionName>(length) -> memPtr {
					let allocSize := <allocSize>(length)
					memPtr := <alloc>(allocSize)
					<?dynamic>
					mstore(memPtr, length)
					</dynamic>
				}
			)")
			("functionName", functionName)
			("alloc", allocationFunction())
			("allocSize", arrayAllocationSizeFunction(_type))
			("dynamic", _type.isDynamicallySized())
			.render();
	});
}

std::string YulUtilFunctions::allocateAndInitializeMemoryArrayFunction(ArrayType const& _type)
{
	std::string functionName = "allocate_and_zero_memory_array_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
				function <functionName>(length) -> memPtr {
					memPtr := <allocArray>(length)
					let dataStart := memPtr
					let dataSize := <allocSize>(length)
					<?dynamic>
					dataStart := add(dataStart, 32)
					dataSize := sub(dataSize, 32)
					</dynamic>
					<zeroArrayFunction>(dataStart, dataSize)
				}
			)")
			("functionName", functionName)
			("allocArray", allocateMemoryArrayFunction(_type))
			("allocSize", arrayAllocationSizeFunction(_type))
			("zeroArrayFunction", zeroMemoryArrayFunction(_type))
			("dynamic", _type.isDynamicallySized())
			.render();
	});
}

std::string YulUtilFunctions::allocateMemoryStructFunction(StructType const& _type)
{
	std::string functionName = "allocate_memory_struct_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
		function <functionName>() -> memPtr {
			memPtr := <alloc>(<allocSize>)
		}
		)");
		templ("functionName", functionName);
		templ("alloc", allocationFunction());
		templ("allocSize", _type.memoryDataSize().str());

		return templ.render();
	});
}

std::string YulUtilFunctions::allocateAndInitializeMemoryStructFunction(StructType const& _type)
{
	std::string functionName = "allocate_and_zero_memory_struct_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
		function <functionName>() -> memPtr {
			memPtr := <allocStruct>()
			let offset := memPtr
			<#member>
				mstore(offset, <zeroValue>())
				offset := add(offset, 32)
			</member>
		}
		)");
		templ("functionName", functionName);
		templ("allocStruct", allocateMemoryStructFunction(_type));

		TypePointers const& members = _type.memoryMemberTypes();

		std::vector<std::map<std::string, std::string>> memberParams(members.size());
		for (size_t i = 0; i < members.size(); ++i)
		{
			solAssert(members[i]->memoryHeadSize() == 32, "");
			memberParams[i]["zeroValue"] = zeroValueFunction(
				*TypeProvider::withLocationIfReference(DataLocation::Memory, members[i]),
				false
			);
		}
		templ("member", memberParams);
		return templ.render();
	});
}

std::string YulUtilFunctions::conversionFunction(Type const& _from, Type const& _to)
{
	if (_from.category() == Type::Category::UserDefinedValueType)
	{
		solAssert(_from == _to || _to == dynamic_cast<UserDefinedValueType const&>(_from).underlyingType(), "");
		return conversionFunction(dynamic_cast<UserDefinedValueType const&>(_from).underlyingType(), _to);
	}
	if (_to.category() == Type::Category::UserDefinedValueType)
	{
		solAssert(_from == _to || _from.isImplicitlyConvertibleTo(dynamic_cast<UserDefinedValueType const&>(_to).underlyingType()), "");
		return conversionFunction(_from, dynamic_cast<UserDefinedValueType const&>(_to).underlyingType());
	}
	if (_from.category() == Type::Category::Function)
	{
		solAssert(_to.category() == Type::Category::Function, "");
		FunctionType const& fromType = dynamic_cast<FunctionType const&>(_from);
		FunctionType const& targetType = dynamic_cast<FunctionType const&>(_to);
		solAssert(
			fromType.isImplicitlyConvertibleTo(targetType) &&
			fromType.sizeOnStack() == targetType.sizeOnStack() &&
			(fromType.kind() == FunctionType::Kind::Internal || fromType.kind() == FunctionType::Kind::External) &&
			fromType.kind() == targetType.kind(),
			"Invalid function type conversion requested."
		);
		std::string const functionName =
			"convert_" +
			_from.identifier() +
			"_to_" +
			_to.identifier();
		return m_functionCollector.createFunction(functionName, [&]() {
			return Whiskers(R"(
				function <functionName>(<?external>addr, </external>functionId) -> <?external>outAddr, </external>outFunctionId {
					<?external>outAddr := addr</external>
					outFunctionId := functionId
				}
			)")
			("functionName", functionName)
			("external", fromType.kind() == FunctionType::Kind::External)
			.render();
		});
	}
	else if (_from.category() == Type::Category::ArraySlice)
	{
		auto const& fromType = dynamic_cast<ArraySliceType const&>(_from);
		if (_to.category() == Type::Category::FixedBytes)
		{
			solAssert(fromType.arrayType().isByteArray(), "Array types other than bytes not convertible to bytesNN.");
			return bytesToFixedBytesConversionFunction(fromType.arrayType(), dynamic_cast<FixedBytesType const &>(_to));
		}
		solAssert(_to.category() == Type::Category::Array);
		auto const& targetType = dynamic_cast<ArrayType const&>(_to);

		solAssert(
			fromType.arrayType().isImplicitlyConvertibleTo(targetType) ||
			(fromType.arrayType().isByteArrayOrString() && targetType.isByteArrayOrString())
		);
		solAssert(
			fromType.arrayType().dataStoredIn(DataLocation::CallData) &&
			fromType.arrayType().isDynamicallySized() &&
			!fromType.arrayType().baseType()->isDynamicallyEncoded()
		);

		if (!targetType.dataStoredIn(DataLocation::CallData))
			return arrayConversionFunction(fromType.arrayType(), targetType);

		std::string const functionName =
			"convert_" +
			_from.identifier() +
			"_to_" +
			_to.identifier();
		return m_functionCollector.createFunction(functionName, [&]() {
			return Whiskers(R"(
				function <functionName>(offset, length) -> outOffset, outLength {
					outOffset := offset
					outLength := length
				}
			)")
			("functionName", functionName)
			.render();
		});
	}
	else if (_from.category() == Type::Category::Array)
	{
		auto const& fromArrayType =  dynamic_cast<ArrayType const&>(_from);
		if (_to.category() == Type::Category::FixedBytes)
		{
			solAssert(fromArrayType.isByteArray(), "Array types other than bytes not convertible to bytesNN.");
			return bytesToFixedBytesConversionFunction(fromArrayType, dynamic_cast<FixedBytesType const &>(_to));
		}
		solAssert(_to.category() == Type::Category::Array, "");
		return arrayConversionFunction(fromArrayType, dynamic_cast<ArrayType const&>(_to));
	}

	if (_from.sizeOnStack() != 1 || _to.sizeOnStack() != 1)
		return conversionFunctionSpecial(_from, _to);

	std::string functionName =
		"convert_" +
		_from.identifier() +
		"_to_" +
		_to.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(value) -> converted {
				<body>
			}
		)");
		templ("functionName", functionName);
		std::string body;
		auto toCategory = _to.category();
		auto fromCategory = _from.category();
		switch (fromCategory)
		{
		case Type::Category::Address:
		case Type::Category::Contract:
			body =
				Whiskers("converted := <convert>(value)")
					("convert", conversionFunction(IntegerType(160), _to))
					.render();
			break;
		case Type::Category::Integer:
		case Type::Category::RationalNumber:
		{
			solAssert(_from.mobileType(), "");
			if (RationalNumberType const* rational = dynamic_cast<RationalNumberType const*>(&_from))
				if (rational->isFractional())
					solAssert(toCategory == Type::Category::FixedPoint, "");

			if (toCategory == Type::Category::Address || toCategory == Type::Category::Contract)
				body =
					Whiskers("converted := <convert>(value)")
					("convert", conversionFunction(_from, IntegerType(160)))
					.render();
			else
			{
				Whiskers bodyTemplate("converted := <cleanOutput>(<convert>(<cleanInput>(value)))");
				bodyTemplate("cleanInput", cleanupFunction(_from));
				bodyTemplate("cleanOutput", cleanupFunction(_to));
				std::string convert;

				solAssert(_to.category() != Type::Category::UserDefinedValueType, "");
				if (auto const* toFixedBytes = dynamic_cast<FixedBytesType const*>(&_to))
					convert = shiftLeftFunction(256 - toFixedBytes->numBytes() * 8);
				else if (dynamic_cast<FixedPointType const*>(&_to))
					solUnimplemented("");
				else if (dynamic_cast<IntegerType const*>(&_to))
				{
					solUnimplementedAssert(fromCategory != Type::Category::FixedPoint);
					convert = identityFunction();
				}
				else if (toCategory == Type::Category::Enum)
				{
					solAssert(fromCategory != Type::Category::FixedPoint, "");
					convert = identityFunction();
				}
				else
					solAssert(false, "");
				solAssert(!convert.empty(), "");
				bodyTemplate("convert", convert);
				body = bodyTemplate.render();
			}
			break;
		}
		case Type::Category::Bool:
		{
			solAssert(_from == _to, "Invalid conversion for bool.");
			body =
				Whiskers("converted := <clean>(value)")
				("clean", cleanupFunction(_from))
				.render();
			break;
		}
		case Type::Category::FixedPoint:
			solUnimplemented("Fixed point types not implemented.");
			break;
		case Type::Category::Struct:
		{
			solAssert(toCategory == Type::Category::Struct, "");
			auto const& fromStructType = dynamic_cast<StructType const &>(_from);
			auto const& toStructType = dynamic_cast<StructType const &>(_to);
			solAssert(fromStructType.structDefinition() == toStructType.structDefinition(), "");

			if (fromStructType.location() == toStructType.location() && toStructType.isPointer())
				body = "converted := value";
			else
			{
				solUnimplementedAssert(toStructType.location() == DataLocation::Memory);
				solUnimplementedAssert(fromStructType.location() != DataLocation::Memory);

				if (fromStructType.location() == DataLocation::CallData)
					body = Whiskers(R"(
						converted := <abiDecode>(value, calldatasize())
					)")
					(
						"abiDecode",
						ABIFunctions(m_evmVersion, m_revertStrings, m_functionCollector).abiDecodingFunctionStruct(
							toStructType,
							false
						)
					).render();
				else
				{
					solAssert(fromStructType.location() == DataLocation::Storage, "");

					body = Whiskers(R"(
						converted := <readFromStorage>(value)
					)")
					("readFromStorage", readFromStorage(toStructType, 0, true, VariableDeclaration::Location::Unspecified))
					.render();
				}
			}

			break;
		}
		case Type::Category::FixedBytes:
		{
			FixedBytesType const& from = dynamic_cast<FixedBytesType const&>(_from);
			if (toCategory == Type::Category::Integer)
				body =
					Whiskers("converted := <convert>(<shift>(value))")
					("shift", shiftRightFunction(256 - from.numBytes() * 8))
					("convert", conversionFunction(IntegerType(from.numBytes() * 8), _to))
					.render();
			else if (toCategory == Type::Category::Address)
				body =
					Whiskers("converted := <convert>(value)")
						("convert", conversionFunction(_from, IntegerType(160)))
						.render();
			else
			{
				solAssert(toCategory == Type::Category::FixedBytes, "Invalid type conversion requested.");
				FixedBytesType const& to = dynamic_cast<FixedBytesType const&>(_to);
				body =
					Whiskers("converted := <clean>(value)")
					("clean", cleanupFunction((to.numBytes() <= from.numBytes()) ? to : from))
					.render();
			}
			break;
		}
		case Type::Category::Function:
		{
			solAssert(false, "Conversion should not be called for function types.");
			break;
		}
		case Type::Category::Enum:
		{
			solAssert(toCategory == Type::Category::Integer || _from == _to, "");
			EnumType const& enumType = dynamic_cast<decltype(enumType)>(_from);
			body =
				Whiskers("converted := <clean>(value)")
				("clean", cleanupFunction(enumType))
				.render();
			break;
		}
		case Type::Category::Tuple:
		{
			solUnimplemented("Tuple conversion not implemented.");
			break;
		}
		case Type::Category::TypeType:
		{
			TypeType const& typeType = dynamic_cast<decltype(typeType)>(_from);
			if (
				auto const* contractType = dynamic_cast<ContractType const*>(typeType.actualType());
				contractType->contractDefinition().isLibrary() &&
				_to == *TypeProvider::address()
			)
				body = "converted := value";
			else
				solAssert(false, "Invalid conversion from " + _from.canonicalName() + " to " + _to.canonicalName());
			break;
		}
		case Type::Category::Mapping:
		{
			solAssert(_from == _to, "");
			body = "converted := value";
			break;
		}
		default:
			solAssert(false, "Invalid conversion from " + _from.canonicalName() + " to " + _to.canonicalName());
		}

		solAssert(!body.empty(), _from.canonicalName() + " to " + _to.canonicalName());
		templ("body", body);
		return templ.render();
	});
}

std::string YulUtilFunctions::bytesToFixedBytesConversionFunction(ArrayType const& _from, FixedBytesType const& _to)
{
	solAssert(_from.isByteArray(), "");
	solAssert(_from.isDynamicallySized(), "");
	std::string functionName = "convert_bytes_to_fixedbytes_from_" + _from.identifier() + "_to_" + _to.identifier();
	return m_functionCollector.createFunction(functionName, [&](auto& _args, auto& _returnParams) {
		_args = { "array" };
		bool fromCalldata = _from.dataStoredIn(DataLocation::CallData);
		if (fromCalldata)
			_args.emplace_back("len");
		_returnParams = {"value"};
		Whiskers templ(R"(
			let length := <arrayLen>(array<?fromCalldata>, len</fromCalldata>)
			let dataArea := array
			<?fromMemory>
				dataArea := <dataArea>(array)
			</fromMemory>
			<?fromStorage>
				if gt(length, 31) { dataArea := <dataArea>(array) }
			</fromStorage>

			<?fromCalldata>
				value := <cleanup>(calldataload(dataArea))
			<!fromCalldata>
				value := <extractValue>(dataArea)
			</fromCalldata>

			if lt(length, <fixedBytesLen>) {
				value := and(
					value,
					<shl>(
						mul(8, sub(<fixedBytesLen>, length)),
						<mask>
					)
				)
			}
		)");
		templ("fromCalldata", fromCalldata);
		templ("arrayLen", arrayLengthFunction(_from));
		templ("fixedBytesLen", std::to_string(_to.numBytes()));
		templ("fromMemory", _from.dataStoredIn(DataLocation::Memory));
		templ("fromStorage", _from.dataStoredIn(DataLocation::Storage));
		templ("dataArea", arrayDataAreaFunction(_from));
		if (fromCalldata)
			templ("cleanup", cleanupFunction(_to));
		else
			templ(
				"extractValue",
				_from.dataStoredIn(DataLocation::Storage) ?
				readFromStorage(_to, 32 - _to.numBytes(), false, VariableDeclaration::Location::Unspecified) :
				readFromMemory(_to)
			);
		templ("shl", shiftLeftFunctionDynamic());
		templ("mask", formatNumber(~((u256(1) << (256 - _to.numBytes() * 8)) - 1)));
		return templ.render();
	});
}

std::string YulUtilFunctions::copyStructToStorageFunction(StructType const& _from, StructType const& _to)
{
	solAssert(_to.dataStoredIn(DataLocation::Storage), "");
	solAssert(_from.structDefinition() == _to.structDefinition(), "");

	std::string functionName =
		"copy_struct_to_storage_from_" +
		_from.identifier() +
		"_to_" +
		_to.identifier();

	return m_functionCollector.createFunction(functionName, [&](auto& _arguments, auto&) {
		_arguments = {"slot", "value"};
		Whiskers templ(R"(
			<?fromStorage> if iszero(eq(slot, value)) { </fromStorage>
			<#member>
			{
				<updateMemberCall>
			}
			</member>
			<?fromStorage> } </fromStorage>
		)");
		templ("fromStorage", _from.dataStoredIn(DataLocation::Storage));

		MemberList::MemberMap structMembers = _from.nativeMembers(nullptr);
		MemberList::MemberMap toStructMembers = _to.nativeMembers(nullptr);

		std::vector<std::map<std::string, std::string>> memberParams(structMembers.size());
		for (size_t i = 0; i < structMembers.size(); ++i)
		{
			Type const& memberType = *structMembers[i].type;
			solAssert(memberType.memoryHeadSize() == 32, "");
			auto const&[slotDiff, offset] = _to.storageOffsetsOfMember(structMembers[i].name);

			Whiskers t(R"(
				let memberSlot := add(slot, <memberStorageSlotDiff>)
				let memberSrcPtr := add(value, <memberOffset>)

				<?fromCalldata>
					let <memberValues> :=
						<?dynamicallyEncodedMember>
							<accessCalldataTail>(value, memberSrcPtr)
						<!dynamicallyEncodedMember>
							memberSrcPtr
						</dynamicallyEncodedMember>

					<?isValueType>
						<memberValues> := <read>(<memberValues>)
					</isValueType>
				</fromCalldata>

				<?fromMemory>
					let <memberValues> := <read>(memberSrcPtr)
				</fromMemory>

				<?fromStorage>
					let <memberValues> :=
						<?isValueType>
							<read>(memberSrcPtr)
						<!isValueType>
							memberSrcPtr
						</isValueType>
				</fromStorage>

				<updateStorageValue>(memberSlot, <memberValues>)
			)");
			bool fromCalldata = _from.location() == DataLocation::CallData;
			t("fromCalldata", fromCalldata);
			bool fromMemory = _from.location() == DataLocation::Memory;
			t("fromMemory", fromMemory);
			bool fromStorage = _from.location() == DataLocation::Storage;
			t("fromStorage", fromStorage);
			t("isValueType", memberType.isValueType());
			t("memberValues", suffixedVariableNameList("memberValue_", 0, memberType.stackItems().size()));

			t("memberStorageSlotDiff", slotDiff.str());
			if (fromCalldata)
			{
				t("memberOffset", std::to_string(_from.calldataOffsetOfMember(structMembers[i].name)));
				t("dynamicallyEncodedMember", memberType.isDynamicallyEncoded());
				if (memberType.isDynamicallyEncoded())
					t("accessCalldataTail", accessCalldataTailFunction(memberType));
				if (memberType.isValueType())
					t("read", readFromCalldata(memberType));
			}
			else if (fromMemory)
			{
				t("memberOffset", _from.memoryOffsetOfMember(structMembers[i].name).str());
				t("read", readFromMemory(memberType));
			}
			else if (fromStorage)
			{
				auto const& [srcSlotOffset, srcOffset] = _from.storageOffsetsOfMember(structMembers[i].name);
				t("memberOffset", formatNumber(srcSlotOffset));
				if (memberType.isValueType())
					t("read", readFromStorageValueType(memberType, srcOffset, true, VariableDeclaration::Location::Unspecified));
				else
					solAssert(srcOffset == 0, "");

			}
			t("updateStorageValue", updateStorageValueFunction(
				memberType,
				*toStructMembers[i].type,
				VariableDeclaration::Location::Unspecified,
				std::optional<unsigned>{offset}
			));
			memberParams[i]["updateMemberCall"] = t.render();
		}
		templ("member", memberParams);

		return templ.render();
	});
}

std::string YulUtilFunctions::arrayConversionFunction(ArrayType const& _from, ArrayType const& _to)
{
	if (_to.dataStoredIn(DataLocation::CallData))
		solAssert(
			_from.dataStoredIn(DataLocation::CallData) && _from.isByteArrayOrString() && _to.isByteArrayOrString(),
			""
		);

	// Other cases are done explicitly in LValue::storeValue, and only possible by assignment.
	if (_to.location() == DataLocation::Storage)
		solAssert(
			(_to.isPointer() || (_from.isByteArrayOrString() && _to.isByteArrayOrString())) &&
			_from.location() == DataLocation::Storage,
			"Invalid conversion to storage type."
		);

	std::string functionName =
		"convert_array_" +
		_from.identifier() +
		"_to_" +
		_to.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(value<?fromCalldataDynamic>, length</fromCalldataDynamic>) -> converted <?toCalldataDynamic>, outLength</toCalldataDynamic> {
				<body>
				<?toCalldataDynamic>
					outLength := <length>
				</toCalldataDynamic>
			}
		)");
		templ("functionName", functionName);
		templ("fromCalldataDynamic", _from.dataStoredIn(DataLocation::CallData) && _from.isDynamicallySized());
		templ("toCalldataDynamic", _to.dataStoredIn(DataLocation::CallData) && _to.isDynamicallySized());
		templ("length", _from.isDynamicallySized() ? "length" : _from.length().str());

		if (
			_from == _to ||
			(_from.dataStoredIn(DataLocation::Memory) && _to.dataStoredIn(DataLocation::Memory)) ||
			(_from.dataStoredIn(DataLocation::CallData) && _to.dataStoredIn(DataLocation::CallData)) ||
			_to.dataStoredIn(DataLocation::Storage)
		)
			templ("body", "converted := value");
		else if (_to.dataStoredIn(DataLocation::Memory))
			templ(
				"body",
				Whiskers(R"(
					// Copy the array to a free position in memory
					converted :=
					<?fromStorage>
						<arrayStorageToMem>(value)
					</fromStorage>
					<?fromCalldata>
						<abiDecode>(value, <length>, calldatasize())
					</fromCalldata>
				)")
				("fromStorage", _from.dataStoredIn(DataLocation::Storage))
				("fromCalldata", _from.dataStoredIn(DataLocation::CallData))
				("length", _from.isDynamicallySized() ? "length" : _from.length().str())
				(
					"abiDecode",
					_from.dataStoredIn(DataLocation::CallData) ?
					ABIFunctions(
						m_evmVersion,
						m_revertStrings,
						m_functionCollector
					).abiDecodingFunctionArrayAvailableLength(_to, false) :
					""
				)
				(
					"arrayStorageToMem",
					_from.dataStoredIn(DataLocation::Storage) ? copyArrayFromStorageToMemoryFunction(_from, _to) : ""
				)
				.render()
			);
		else
			solAssert(false, "");

		return templ.render();
	});
}

std::string YulUtilFunctions::cleanupFunction(Type const& _type)
{
	if (auto userDefinedValueType = dynamic_cast<UserDefinedValueType const*>(&_type))
		return cleanupFunction(userDefinedValueType->underlyingType());

	std::string functionName = std::string("cleanup_") + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(value) -> cleaned {
				<body>
			}
		)");
		templ("functionName", functionName);
		switch (_type.category())
		{
		case Type::Category::Address:
			templ("body", "cleaned := " + cleanupFunction(IntegerType(160)) + "(value)");
			break;
		case Type::Category::Integer:
		{
			IntegerType const& type = dynamic_cast<IntegerType const&>(_type);
			if (type.numBits() == 256)
				templ("body", "cleaned := value");
			else if (type.isSigned())
				templ("body", "cleaned := signextend(" + std::to_string(type.numBits() / 8 - 1) + ", value)");
			else
				templ("body", "cleaned := and(value, " + toCompactHexWithPrefix((u256(1) << type.numBits()) - 1) + ")");
			break;
		}
		case Type::Category::RationalNumber:
			templ("body", "cleaned := value");
			break;
		case Type::Category::Bool:
			templ("body", "cleaned := iszero(iszero(value))");
			break;
		case Type::Category::FixedPoint:
			solUnimplemented("Fixed point types not implemented.");
			break;
		case Type::Category::Function:
			switch (dynamic_cast<FunctionType const&>(_type).kind())
			{
				case FunctionType::Kind::External:
					templ("body", "cleaned := " + cleanupFunction(FixedBytesType(24)) + "(value)");
					break;
				case FunctionType::Kind::Internal:
					templ("body", "cleaned := value");
					break;
				default:
					solAssert(false, "");
					break;
			}
			break;
		case Type::Category::Array:
		case Type::Category::Struct:
		case Type::Category::Mapping:
			solAssert(_type.dataStoredIn(DataLocation::Storage), "Cleanup requested for non-storage reference type.");
			templ("body", "cleaned := value");
			break;
		case Type::Category::FixedBytes:
		{
			FixedBytesType const& type = dynamic_cast<FixedBytesType const&>(_type);
			if (type.numBytes() == 32)
				templ("body", "cleaned := value");
			else if (type.numBytes() == 0)
				// This is disallowed in the type system.
				solAssert(false, "");
			else
			{
				size_t numBits = type.numBytes() * 8;
				u256 mask = ((u256(1) << numBits) - 1) << (256 - numBits);
				templ("body", "cleaned := and(value, " + toCompactHexWithPrefix(mask) + ")");
			}
			break;
		}
		case Type::Category::Contract:
		{
			AddressType addressType(dynamic_cast<ContractType const&>(_type).isPayable() ?
				StateMutability::Payable :
				StateMutability::NonPayable
			);
			templ("body", "cleaned := " + cleanupFunction(addressType) + "(value)");
			break;
		}
		case Type::Category::Enum:
		{
			// Out of range enums cannot be truncated unambiguously and therefore it should be an error.
			templ("body", "cleaned := value " + validatorFunction(_type, false) + "(value)");
			break;
		}
		case Type::Category::InaccessibleDynamic:
			templ("body", "cleaned := 0");
			break;
		default:
			solAssert(false, "Cleanup of type " + _type.identifier() + " requested.");
		}

		return templ.render();
	});
}

std::string YulUtilFunctions::validatorFunction(Type const& _type, bool _revertOnFailure)
{
	std::string functionName = std::string("validator_") + (_revertOnFailure ? "revert_" : "assert_") + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(value) {
				if iszero(<condition>) { <failure> }
			}
		)");
		templ("functionName", functionName);
		PanicCode panicCode = PanicCode::Generic;

		switch (_type.category())
		{
		case Type::Category::Address:
		case Type::Category::Integer:
		case Type::Category::RationalNumber:
		case Type::Category::Bool:
		case Type::Category::FixedPoint:
		case Type::Category::Function:
		case Type::Category::Array:
		case Type::Category::Struct:
		case Type::Category::Mapping:
		case Type::Category::FixedBytes:
		case Type::Category::Contract:
		case Type::Category::UserDefinedValueType:
		{
			templ("condition", "eq(value, " + cleanupFunction(_type) + "(value))");
			break;
		}
		case Type::Category::Enum:
		{
			size_t members = dynamic_cast<EnumType const&>(_type).numberOfMembers();
			solAssert(members > 0, "empty enum should have caused a parser error.");
			panicCode = PanicCode::EnumConversionError;
			templ("condition", "lt(value, " + std::to_string(members) + ")");
			break;
		}
		case Type::Category::InaccessibleDynamic:
			templ("condition", "1");
			break;
		default:
			solAssert(false, "Validation of type " + _type.identifier() + " requested.");
		}

		if (_revertOnFailure)
			templ("failure", "revert(0, 0)");
		else
			templ("failure", panicFunction(panicCode) + "()");

		return templ.render();
	});
}

std::string YulUtilFunctions::packedHashFunction(
	std::vector<Type const*> const& _givenTypes,
	std::vector<Type const*> const& _targetTypes
)
{
	std::string functionName = std::string("packed_hashed_");
	for (auto const& t: _givenTypes)
		functionName += t->identifier() + "_";
	functionName += "_to_";
	for (auto const& t: _targetTypes)
		functionName += t->identifier() + "_";
	size_t sizeOnStack = 0;
	for (Type const* t: _givenTypes)
		sizeOnStack += t->sizeOnStack();
	return m_functionCollector.createFunction(functionName, [&]() {
		Whiskers templ(R"(
			function <functionName>(<variables>) -> hash {
				let pos := <allocateUnbounded>()
				let end := <packedEncode>(pos <comma> <variables>)
				hash := keccak256(pos, sub(end, pos))
			}
		)");
		templ("functionName", functionName);
		templ("variables", suffixedVariableNameList("var_", 1, 1 + sizeOnStack));
		templ("comma", sizeOnStack > 0 ? "," : "");
		templ("allocateUnbounded", allocateUnboundedFunction());
		templ("packedEncode", ABIFunctions(m_evmVersion, m_revertStrings, m_functionCollector).tupleEncoderPacked(_givenTypes, _targetTypes));
		return templ.render();
	});
}

std::string YulUtilFunctions::forwardingRevertFunction()
{
	bool forward = m_evmVersion.supportsReturndata();
	std::string functionName = "revert_forward_" + std::to_string(forward);
	return m_functionCollector.createFunction(functionName, [&]() {
		if (forward)
			return Whiskers(R"(
				function <functionName>() {
					let pos := <allocateUnbounded>()
					returndatacopy(pos, 0, returndatasize())
					revert(pos, returndatasize())
				}
			)")
			("functionName", functionName)
			("allocateUnbounded", allocateUnboundedFunction())
			.render();
		else
			return Whiskers(R"(
				function <functionName>() {
					revert(0, 0)
				}
			)")
			("functionName", functionName)
			.render();
	});
}

std::string YulUtilFunctions::decrementCheckedFunction(Type const& _type)
{
	solAssert(_type.category() == Type::Category::Integer, "");
	IntegerType const& type = dynamic_cast<IntegerType const&>(_type);

	std::string const functionName = "decrement_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(value) -> ret {
				value := <cleanupFunction>(value)
				if eq(value, <minval>) { <panic>() }
				ret := sub(value, 1)
			}
		)")
		("functionName", functionName)
		("panic", panicFunction(PanicCode::UnderOverflow))
		("minval", toCompactHexWithPrefix(type.min()))
		("cleanupFunction", cleanupFunction(_type))
		.render();
	});
}

std::string YulUtilFunctions::decrementWrappingFunction(Type const& _type)
{
	solAssert(_type.category() == Type::Category::Integer, "");
	IntegerType const& type = dynamic_cast<IntegerType const&>(_type);

	std::string const functionName = "decrement_wrapping_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(value) -> ret {
				ret := <cleanupFunction>(sub(value, 1))
			}
		)")
		("functionName", functionName)
		("cleanupFunction", cleanupFunction(type))
		.render();
	});
}

std::string YulUtilFunctions::incrementCheckedFunction(Type const& _type)
{
	solAssert(_type.category() == Type::Category::Integer, "");
	IntegerType const& type = dynamic_cast<IntegerType const&>(_type);

	std::string const functionName = "increment_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(value) -> ret {
				value := <cleanupFunction>(value)
				if eq(value, <maxval>) { <panic>() }
				ret := add(value, 1)
			}
		)")
		("functionName", functionName)
		("maxval", toCompactHexWithPrefix(type.max()))
		("panic", panicFunction(PanicCode::UnderOverflow))
		("cleanupFunction", cleanupFunction(_type))
		.render();
	});
}

std::string YulUtilFunctions::incrementWrappingFunction(Type const& _type)
{
	solAssert(_type.category() == Type::Category::Integer, "");
	IntegerType const& type = dynamic_cast<IntegerType const&>(_type);

	std::string const functionName = "increment_wrapping_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(value) -> ret {
				ret := <cleanupFunction>(add(value, 1))
			}
		)")
		("functionName", functionName)
		("cleanupFunction", cleanupFunction(type))
		.render();
	});
}

std::string YulUtilFunctions::negateNumberCheckedFunction(Type const& _type)
{
	solAssert(_type.category() == Type::Category::Integer, "");
	IntegerType const& type = dynamic_cast<IntegerType const&>(_type);
	solAssert(type.isSigned(), "Expected signed type!");

	std::string const functionName = "negate_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(value) -> ret {
				value := <cleanupFunction>(value)
				if eq(value, <minval>) { <panic>() }
				ret := sub(0, value)
			}
		)")
		("functionName", functionName)
		("minval", toCompactHexWithPrefix(type.min()))
		("cleanupFunction", cleanupFunction(_type))
		("panic", panicFunction(PanicCode::UnderOverflow))
		.render();
	});
}

std::string YulUtilFunctions::negateNumberWrappingFunction(Type const& _type)
{
	solAssert(_type.category() == Type::Category::Integer, "");
	IntegerType const& type = dynamic_cast<IntegerType const&>(_type);
	solAssert(type.isSigned(), "Expected signed type!");

	std::string const functionName = "negate_wrapping_" + _type.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>(value) -> ret {
				ret := <cleanupFunction>(sub(0, value))
			}
		)")
		("functionName", functionName)
		("cleanupFunction", cleanupFunction(type))
		.render();
	});
}

std::string YulUtilFunctions::zeroValueFunction(Type const& _type, bool _splitFunctionTypes)
{
	solAssert(_type.category() != Type::Category::Mapping, "");

	std::string const functionName = "zero_value_for_" + std::string(_splitFunctionTypes ? "split_" : "") + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		FunctionType const* fType = dynamic_cast<FunctionType const*>(&_type);
		if (fType && fType->kind() == FunctionType::Kind::External && _splitFunctionTypes)
			return Whiskers(R"(
				function <functionName>() -> retAddress, retFunction {
					retAddress := 0
					retFunction := 0
				}
			)")
			("functionName", functionName)
			.render();

		if (_type.dataStoredIn(DataLocation::CallData))
		{
			solAssert(
				_type.category() == Type::Category::Struct ||
				_type.category() == Type::Category::Array,
			"");
			Whiskers templ(R"(
				function <functionName>() -> offset<?hasLength>, length</hasLength> {
					offset := calldatasize()
					<?hasLength> length := 0 </hasLength>
				}
			)");
			templ("functionName", functionName);
			templ("hasLength",
				_type.category() == Type::Category::Array &&
				dynamic_cast<ArrayType const&>(_type).isDynamicallySized()
			);

			return templ.render();
		}

		Whiskers templ(R"(
			function <functionName>() -> ret {
				ret := <zeroValue>
			}
		)");
		templ("functionName", functionName);

		if (_type.isValueType())
		{
			solAssert((
				_type.hasSimpleZeroValueInMemory() ||
				(fType && (fType->kind() == FunctionType::Kind::Internal || fType->kind() == FunctionType::Kind::External))
			), "");
			templ("zeroValue", "0");
		}
		else
		{
			solAssert(_type.dataStoredIn(DataLocation::Memory), "");
			if (auto const* arrayType = dynamic_cast<ArrayType const*>(&_type))
			{
				if (_type.isDynamicallySized())
					templ("zeroValue", std::to_string(CompilerUtils::zeroPointer));
				else
					templ("zeroValue", allocateAndInitializeMemoryArrayFunction(*arrayType) + "(" + std::to_string(unsigned(arrayType->length())) + ")");

			}
			else if (auto const* structType = dynamic_cast<StructType const*>(&_type))
				templ("zeroValue", allocateAndInitializeMemoryStructFunction(*structType) + "()");
			else
				solUnimplemented("");
		}

		return templ.render();
	});
}

std::string YulUtilFunctions::storageSetToZeroFunction(Type const& _type, VariableDeclaration::Location _location)
{
	std::string const functionName = "storage_set_to_zero_" + _type.identifier();

	return m_functionCollector.createFunction(functionName, [&]() {
		if (_type.isValueType())
			return Whiskers(R"(
				function <functionName>(slot, offset) {
					let <values> := <zeroValue>()
					<store>(slot, offset, <values>)
				}
			)")
			("functionName", functionName)
			("store", updateStorageValueFunction(_type, _type, _location))
			("values", suffixedVariableNameList("zero_", 0, _type.sizeOnStack()))
			("zeroValue", zeroValueFunction(_type))
			.render();
		else if (_type.category() == Type::Category::Array)
			return Whiskers(R"(
				function <functionName>(slot, offset) {
					if iszero(eq(offset, 0)) { <panic>() }
					<clearArray>(slot)
				}
			)")
			("functionName", functionName)
			("clearArray", clearStorageArrayFunction(dynamic_cast<ArrayType const&>(_type)))
			("panic", panicFunction(PanicCode::Generic))
			.render();
		else if (_type.category() == Type::Category::Struct)
			return Whiskers(R"(
				function <functionName>(slot, offset) {
					if iszero(eq(offset, 0)) { <panic>() }
					<clearStruct>(slot)
				}
			)")
			("functionName", functionName)
			("clearStruct", clearStorageStructFunction(dynamic_cast<StructType const&>(_type)))
			("panic", panicFunction(PanicCode::Generic))
			.render();
		else
			solUnimplemented("setToZero for type " + _type.identifier() + " not yet implemented!");
	});
}

std::string YulUtilFunctions::conversionFunctionSpecial(Type const& _from, Type const& _to)
{
	std::string functionName =
		"convert_" +
		_from.identifier() +
		"_to_" +
		_to.identifier();
	return m_functionCollector.createFunction(functionName, [&]() {
		if (
			auto fromTuple = dynamic_cast<TupleType const*>(&_from), toTuple = dynamic_cast<TupleType const*>(&_to);
			fromTuple && toTuple && fromTuple->components().size() == toTuple->components().size()
		)
		{
			size_t sourceStackSize = 0;
			size_t destStackSize = 0;
			std::string conversions;
			for (size_t i = 0; i < fromTuple->components().size(); ++i)
			{
				auto fromComponent = fromTuple->components()[i];
				auto toComponent = toTuple->components()[i];
				solAssert(fromComponent, "");
				if (toComponent)
				{
					conversions +=
						suffixedVariableNameList("converted", destStackSize, destStackSize + toComponent->sizeOnStack()) +
						(toComponent->sizeOnStack() > 0 ? " := " : "") +
						conversionFunction(*fromComponent, *toComponent) +
						"(" +
						suffixedVariableNameList("value", sourceStackSize, sourceStackSize + fromComponent->sizeOnStack()) +
						")\n";
					destStackSize += toComponent->sizeOnStack();
				}
				sourceStackSize += fromComponent->sizeOnStack();
			}
			return Whiskers(R"(
				function <functionName>(<values>) <arrow> <converted> {
					<conversions>
				}
			)")
			("functionName", functionName)
			("values", suffixedVariableNameList("value", 0, sourceStackSize))
			("arrow", destStackSize > 0 ? "->" : "")
			("converted", suffixedVariableNameList("converted", 0, destStackSize))
			("conversions", conversions)
			.render();
		}

		solUnimplementedAssert(
			_from.category() == Type::Category::StringLiteral,
			"Type conversion " + _from.toString() + " -> " + _to.toString() + " not yet implemented."
		);
		std::string const& data = dynamic_cast<StringLiteralType const&>(_from).value();
		if (_to.category() == Type::Category::FixedBytes)
		{
			unsigned const numBytes = dynamic_cast<FixedBytesType const&>(_to).numBytes();
			solAssert(data.size() <= 32, "");
			Whiskers templ(R"(
				function <functionName>() -> converted {
					converted := <data>
				}
			)");
			templ("functionName", functionName);
			templ("data", formatNumber(
				h256::Arith(h256(data, h256::AlignLeft)) &
				(~(u256(-1) >> (8 * numBytes)))
			));
			return templ.render();
		}
		else if (_to.category() == Type::Category::Array)
		{
			solAssert(dynamic_cast<ArrayType const&>(_to).isByteArrayOrString(), "");
			Whiskers templ(R"(
				function <functionName>() -> converted {
					converted := <copyLiteralToMemory>()
				}
			)");
			templ("functionName", functionName);
			templ("copyLiteralToMemory", copyLiteralToMemoryFunction(data));
			return templ.render();
		}
		else
			solAssert(
				false,
				"Invalid conversion from std::string literal to " + _to.toString() + " requested."
			);
	});
}

std::string YulUtilFunctions::readFromMemoryOrCalldata(Type const& _type, bool _fromCalldata)
{
	std::string functionName =
		std::string("read_from_") +
		(_fromCalldata ? "calldata" : "memory") +
		_type.identifier();

	// TODO use ABI functions for handling calldata
	if (_fromCalldata)
		solAssert(!_type.isDynamicallyEncoded(), "");

	return m_functionCollector.createFunction(functionName, [&] {
		if (auto refType = dynamic_cast<ReferenceType const*>(&_type))
		{
			solAssert(refType->sizeOnStack() == 1, "");
			solAssert(!_fromCalldata, "");

			return Whiskers(R"(
				function <functionName>(memPtr) -> value {
					value := mload(memPtr)
				}
			)")
			("functionName", functionName)
			.render();
		}

		solAssert(_type.isValueType(), "");
		Whiskers templ(R"(
			function <functionName>(ptr) -> <returnVariables> {
				<?fromCalldata>
					let value := calldataload(ptr)
					<validate>(value)
				<!fromCalldata>
					let value := <cleanup>(mload(ptr))
				</fromCalldata>

				<returnVariables> :=
				<?externalFunction>
					<splitFunction>(value)
				<!externalFunction>
					value
				</externalFunction>
			}
		)");
		templ("functionName", functionName);
		templ("fromCalldata", _fromCalldata);
		if (_fromCalldata)
			templ("validate", validatorFunction(_type, true));
		auto const* funType = dynamic_cast<FunctionType const*>(&_type);
		if (funType && funType->kind() == FunctionType::Kind::External)
		{
			templ("externalFunction", true);
			templ("splitFunction", splitExternalFunctionIdFunction());
			templ("returnVariables", "addr, selector");
		}
		else
		{
			templ("externalFunction", false);
			templ("returnVariables", "returnValue");
		}

		// Byte array elements generally need cleanup.
		// Other types are cleaned as well to account for dirty memory e.g. due to inline assembly.
		templ("cleanup", cleanupFunction(_type));
		return templ.render();
	});
}

std::string YulUtilFunctions::revertReasonIfDebugFunction(std::string const& _message)
{
	std::string functionName = "revert_error_" + util::toHex(util::keccak256(_message).asBytes());
	return m_functionCollector.createFunction(functionName, [&](auto&, auto&) -> std::string {
		return revertReasonIfDebugBody(m_revertStrings, allocateUnboundedFunction() + "()", _message);
	});
}

std::string YulUtilFunctions::revertReasonIfDebugBody(
	RevertStrings _revertStrings,
	std::string const& _allocation,
	std::string const& _message
)
{
	if (_revertStrings < RevertStrings::Debug || _message.empty())
		return "revert(0, 0)";

	Whiskers templ(R"(
		let start := <allocate>
		let pos := start
		mstore(pos, <sig>)
		pos := add(pos, 4)
		mstore(pos, 0x20)
		pos := add(pos, 0x20)
		mstore(pos, <length>)
		pos := add(pos, 0x20)
		<#word>
			mstore(add(pos, <offset>), <wordValue>)
		</word>
		revert(start, <overallLength>)
	)");
	templ("allocate", _allocation);
	templ("sig", util::selectorFromSignatureU256("Error(string)").str());
	templ("length", std::to_string(_message.length()));

	size_t words = (_message.length() + 31) / 32;
	std::vector<std::map<std::string, std::string>> wordParams(words);
	for (size_t i = 0; i < words; ++i)
	{
		wordParams[i]["offset"] = std::to_string(i * 32);
		wordParams[i]["wordValue"] = formatAsStringOrNumber(_message.substr(32 * i, 32));
	}
	templ("word", wordParams);
	templ("overallLength", std::to_string(4 + 0x20 + 0x20 + words * 32));

	return templ.render();
}

std::string YulUtilFunctions::panicFunction(util::PanicCode _code)
{
	std::string functionName = "panic_error_" + toCompactHexWithPrefix(uint64_t(_code));
	return m_functionCollector.createFunction(functionName, [&]() {
		return Whiskers(R"(
			function <functionName>() {
				mstore(0, <selector>)
				mstore(4, <code>)
				revert(0, 0x24)
			}
		)")
		("functionName", functionName)
		("selector", util::selectorFromSignatureU256("Panic(uint256)").str())
		("code", toCompactHexWithPrefix(static_cast<unsigned>(_code)))
		.render();
	});
}

std::string YulUtilFunctions::returnDataSelectorFunction()
{
	std::string const functionName = "return_data_selector";
	solAssert(m_evmVersion.supportsReturndata(), "");

	return m_functionCollector.createFunction(functionName, [&]() {
		return util::Whiskers(R"(
			function <functionName>() -> sig {
				if gt(returndatasize(), 3) {
					returndatacopy(0, 0, 4)
					sig := <shr224>(mload(0))
				}
			}
		)")
		("functionName", functionName)
		("shr224", shiftRightFunction(224))
		.render();
	});
}

std::string YulUtilFunctions::tryDecodeErrorMessageFunction()
{
	std::string const functionName = "try_decode_error_message";
	solAssert(m_evmVersion.supportsReturndata(), "");

	return m_functionCollector.createFunction(functionName, [&]() {
		return util::Whiskers(R"(
			function <functionName>() -> ret {
				if lt(returndatasize(), 0x44) { leave }

				let data := <allocateUnbounded>()
				returndatacopy(data, 4, sub(returndatasize(), 4))

				let offset := mload(data)
				if or(
					gt(offset, 0xffffffffffffffff),
					gt(add(offset, 0x24), returndatasize())
				) {
					leave
				}

				let msg := add(data, offset)
				let length := mload(msg)
				if gt(length, 0xffffffffffffffff) { leave }

				let end := add(add(msg, 0x20), length)
				if gt(end, add(data, sub(returndatasize(), 4))) { leave }

				<finalizeAllocation>(data, add(offset, add(0x20, length)))
				ret := msg
			}
		)")
		("functionName", functionName)
		("allocateUnbounded", allocateUnboundedFunction())
		("finalizeAllocation", finalizeAllocationFunction())
		.render();
	});
}

std::string YulUtilFunctions::tryDecodePanicDataFunction()
{
	std::string const functionName = "try_decode_panic_data";
	solAssert(m_evmVersion.supportsReturndata(), "");

	return m_functionCollector.createFunction(functionName, [&]() {
		return util::Whiskers(R"(
			function <functionName>() -> success, data {
				if gt(returndatasize(), 0x23) {
					returndatacopy(0, 4, 0x20)
					success := 1
					data := mload(0)
				}
			}
		)")
		("functionName", functionName)
		.render();
	});
}

std::string YulUtilFunctions::extractReturndataFunction()
{
	std::string const functionName = "extract_returndata";

	return m_functionCollector.createFunction(functionName, [&]() {
		return util::Whiskers(R"(
			function <functionName>() -> data {
				<?supportsReturndata>
					switch returndatasize()
					case 0 {
						data := <emptyArray>()
					}
					default {
						data := <allocateArray>(returndatasize())
						returndatacopy(add(data, 0x20), 0, returndatasize())
					}
				<!supportsReturndata>
					data := <emptyArray>()
				</supportsReturndata>
			}
		)")
		("functionName", functionName)
		("supportsReturndata", m_evmVersion.supportsReturndata())
		("allocateArray", allocateMemoryArrayFunction(*TypeProvider::bytesMemory()))
		("emptyArray", zeroValueFunction(*TypeProvider::bytesMemory()))
		.render();
	});
}

std::string YulUtilFunctions::copyConstructorArgumentsToMemoryFunction(
	ContractDefinition const& _contract,
	std::string const& _creationObjectName
)
{
	std::string functionName = "copy_arguments_for_constructor_" +
		toString(_contract.constructor()->id()) +
		"_object_" +
		_contract.name() +
		"_" +
		toString(_contract.id());

	return m_functionCollector.createFunction(functionName, [&]() {
		std::string returnParams = suffixedVariableNameList("ret_param_",0, CompilerUtils::sizeOnStack(_contract.constructor()->parameters()));
		ABIFunctions abiFunctions(m_evmVersion, m_revertStrings, m_functionCollector);

		return util::Whiskers(R"(
			function <functionName>() -> <retParams> {
				let programSize := datasize("<object>")
				let argSize := sub(codesize(), programSize)

				let memoryDataOffset := <allocate>(argSize)
				codecopy(memoryDataOffset, programSize, argSize)

				<retParams> := <abiDecode>(memoryDataOffset, add(memoryDataOffset, argSize))
			}
		)")
		("functionName", functionName)
		("retParams", returnParams)
		("object", _creationObjectName)
		("allocate", allocationFunction())
		("abiDecode", abiFunctions.tupleDecoder(FunctionType(*_contract.constructor()).parameterTypes(), true))
		.render();
	});
}

std::string YulUtilFunctions::externalCodeFunction()
{
	std::string functionName = "external_code_at";

	return m_functionCollector.createFunction(functionName, [&]() {
		return util::Whiskers(R"(
			function <functionName>(addr) -> mpos {
				let length := extcodesize(addr)
				mpos := <allocateArray>(length)
				extcodecopy(addr, add(mpos, 0x20), 0, length)
			}
		)")
		("functionName", functionName)
		("allocateArray", allocateMemoryArrayFunction(*TypeProvider::bytesMemory()))
		.render();
	});
}

std::string YulUtilFunctions::externalFunctionPointersEqualFunction()
{
	std::string const functionName = "externalFunctionPointersEqualFunction";
	return m_functionCollector.createFunction(functionName, [&]() {
		return util::Whiskers(R"(
			function <functionName>(
				leftAddress,
				leftSelector,
				rightAddress,
				rightSelector
			) -> result {
				result := and(
					eq(
						<addressCleanUpFunction>(leftAddress), <addressCleanUpFunction>(rightAddress)
					),
					eq(
						<selectorCleanUpFunction>(leftSelector), <selectorCleanUpFunction>(rightSelector)
					)
				)
			}
		)")
		("functionName", functionName)
		("addressCleanUpFunction", cleanupFunction(*TypeProvider::address()))
		("selectorCleanUpFunction", cleanupFunction(*TypeProvider::uint(32)))
		.render();
	});
}
