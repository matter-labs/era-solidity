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
 * @author Christian <c@ethdev.com>
 * @date 2016
 * Unit tests for inline assembly.
 */

#include <test/Common.h>

#include <test/libsolidity/ErrorCheck.h>

#include <libsolidity/ast/AST.h>

#include <libyul/YulStack.h>

#include <liblangutil/DebugInfoSelection.h>
#include <liblangutil/Exceptions.h>
#include <liblangutil/Scanner.h>
#include <liblangutil/SourceReferenceFormatter.h>

#include <libevmasm/Assembly.h>

#include <boost/algorithm/string/replace.hpp>
#include <boost/test/unit_test.hpp>

#include <memory>
#include <optional>
#include <string>

using namespace solidity::langutil;
using namespace solidity::yul;

namespace solidity::frontend::test
{

namespace
{

std::optional<Error> parseAndReturnFirstError(
	std::string const& _source,
	bool _assemble = false,
	bool _allowWarnings = true,
	YulStack::Language _language = YulStack::Language::Assembly,
	YulStack::Machine _machine = YulStack::Machine::EVM
)
{
	YulStack stack(
		solidity::test::CommonOptions::get().evmVersion(),
		solidity::test::CommonOptions::get().eofVersion(),
		_language,
		solidity::frontend::OptimiserSettings::none(),
		DebugInfoSelection::None()
	);
	bool success = stack.parseAndAnalyze("", _source);
	if (success && _assemble)
		stack.assemble(_machine);
	std::shared_ptr<Error const> error;
	for (auto const& e: stack.errors())
	{
		if (_allowWarnings && e->type() == Error::Type::Warning)
			continue;
		if (error)
			BOOST_FAIL(
				"Found more than one error:\n" +
				SourceReferenceFormatter::formatErrorInformation(stack.errors(), stack)
			);
		error = e;
	}
	if (!success)
		BOOST_REQUIRE(error);
	if (error)
		return *error;
	return {};
}

bool successParse(
	std::string const& _source,
	bool _assemble = false,
	bool _allowWarnings = true,
	YulStack::Language _language = YulStack::Language::Assembly,
	YulStack::Machine _machine = YulStack::Machine::EVM
)
{
	return !parseAndReturnFirstError(_source, _assemble, _allowWarnings, _language, _machine);
}

bool successAssemble(std::string const& _source, bool _allowWarnings = true, YulStack::Language _language = YulStack::Language::Assembly)
{
	return
		successParse(_source, true, _allowWarnings, _language, YulStack::Machine::EVM);
}

Error expectError(
	std::string const& _source,
	bool _assemble,
	bool _allowWarnings = false,
	YulStack::Language _language = YulStack::Language::Assembly
)
{

	auto error = parseAndReturnFirstError(_source, _assemble, _allowWarnings, _language);
	BOOST_REQUIRE(error);
	return *error;
}

void parsePrintCompare(std::string const& _source, bool _canWarn = false)
{
	YulStack stack(
		solidity::test::CommonOptions::get().evmVersion(),
		solidity::test::CommonOptions::get().eofVersion(),
		YulStack::Language::Assembly,
		OptimiserSettings::none(),
		DebugInfoSelection::None()
	);
	BOOST_REQUIRE(stack.parseAndAnalyze("", _source));
	if (_canWarn)
		BOOST_REQUIRE(!Error::containsErrors(stack.errors()));
	else
		BOOST_REQUIRE(!Error::hasErrorsWarningsOrInfos(stack.errors()));
	std::string expectation = "object \"object\" {\n    code " + boost::replace_all_copy(_source, "\n", "\n    ") + "\n}\n";
	BOOST_CHECK_EQUAL(stack.print(), expectation);
}

}

#define CHECK_ERROR_LANG(text, assemble, typ, substring, warnings, language) \
do \
{ \
	Error err = expectError((text), (assemble), warnings, (language)); \
	BOOST_CHECK(err.type() == (Error::Type::typ)); \
	BOOST_CHECK(searchErrorMessage(err, (substring))); \
} while(0)

#define CHECK_ERROR(text, assemble, typ, substring, warnings) \
CHECK_ERROR_LANG(text, assemble, typ, substring, warnings, YulStack::Language::Assembly)

#define CHECK_PARSE_ERROR(text, type, substring) \
CHECK_ERROR(text, false, type, substring, false)

#define CHECK_PARSE_WARNING(text, type, substring) \
CHECK_ERROR(text, false, type, substring, false)

#define CHECK_ASSEMBLE_ERROR(text, type, substring) \
CHECK_ERROR(text, true, type, substring, false)

#define CHECK_STRICT_ERROR(text, type, substring) \
CHECK_ERROR_LANG(text, false, type, substring, false, YulStack::Language::StrictAssembly)

#define CHECK_STRICT_WARNING(text, type, substring) \
CHECK_ERROR(text, false, type, substring, false, YulStack::Language::StrictAssembly)

#define SUCCESS_STRICT(text) \
do { successParse((text), false, false, YulStack::Language::StrictAssembly); } while (false)


BOOST_AUTO_TEST_SUITE(SolidityInlineAssembly)

BOOST_AUTO_TEST_SUITE(Printing) // {{{

BOOST_AUTO_TEST_CASE(print_smoke)
{
	parsePrintCompare("{ }");
}

BOOST_AUTO_TEST_CASE(print_instructions)
{
	parsePrintCompare("{ pop(7) }");
}

BOOST_AUTO_TEST_CASE(print_subblock)
{
	parsePrintCompare("{ { pop(7) } }");
}

BOOST_AUTO_TEST_CASE(print_functional)
{
	parsePrintCompare("{ let x := mul(sload(0x12), 7) }");
}

BOOST_AUTO_TEST_CASE(print_assignments)
{
	parsePrintCompare("{\n    let x := mul(2, 3)\n    pop(7)\n    x := add(1, 2)\n}");
}

BOOST_AUTO_TEST_CASE(print_multi_assignments)
{
	parsePrintCompare("{\n    function f() -> x, y\n    { }\n    let x, y := f()\n}");
}

BOOST_AUTO_TEST_CASE(print_string_literals)
{
	parsePrintCompare("{ let x := \"\\n'\\xab\\x95\\\"\" }");
}

BOOST_AUTO_TEST_CASE(print_string_literal_unicode)
{
	std::string source = "{ let x := \"\\u1bac\" }";
	std::string parsed = "object \"object\" {\n    code { let x := \"\\xe1\\xae\\xac\" }\n}\n";
	YulStack stack(
		solidity::test::CommonOptions::get().evmVersion(),
		solidity::test::CommonOptions::get().eofVersion(),
		YulStack::Language::Assembly,
		OptimiserSettings::none(),
		DebugInfoSelection::None()
	);
	BOOST_REQUIRE(stack.parseAndAnalyze("", source));
	BOOST_REQUIRE(!Error::hasErrorsWarningsOrInfos(stack.errors()));
	BOOST_CHECK_EQUAL(stack.print(), parsed);

	std::string parsedInner = "{ let x := \"\\xe1\\xae\\xac\" }";
	parsePrintCompare(parsedInner);
}

BOOST_AUTO_TEST_CASE(print_if)
{
	parsePrintCompare("{ if 2 { pop(mload(0)) } }");
}

BOOST_AUTO_TEST_CASE(print_switch)
{
	parsePrintCompare("{\n    switch 42\n    case 1 { }\n    case 2 { }\n    default { }\n}");
}

BOOST_AUTO_TEST_CASE(print_for)
{
	parsePrintCompare("{\n    let ret := 5\n    for { let i := 1 } lt(i, 15) { i := add(i, 1) }\n    { ret := mul(ret, i) }\n}");
}

BOOST_AUTO_TEST_CASE(function_definitions_multiple_args)
{
	parsePrintCompare("{\n    function f(a, d)\n    { mstore(a, d) }\n    function g(a, d) -> x, y\n    { }\n}");
}

BOOST_AUTO_TEST_CASE(function_calls)
{
	std::string source = R"({
	function y()
	{ }
	function f(a) -> b
	{ }
	function g(a, b, c)
	{ }
	g(1, mul(2, address()), f(mul(2, caller())))
	y()
})";
	boost::replace_all(source, "\t", "    ");
	parsePrintCompare(source);
}

BOOST_AUTO_TEST_SUITE_END()
// }}}

BOOST_AUTO_TEST_SUITE(Analysis) // {{{

BOOST_AUTO_TEST_CASE(string_literals)
{
	BOOST_CHECK(successAssemble("{ let x := \"12345678901234567890123456789012\" }"));
}

BOOST_AUTO_TEST_CASE(oversize_string_literals)
{
	CHECK_ASSEMBLE_ERROR("{ let x := \"123456789012345678901234567890123\" }", TypeError, "String literal too long");
}

BOOST_AUTO_TEST_CASE(magic_variables)
{
	CHECK_ASSEMBLE_ERROR("{ pop(this) }", DeclarationError, "Identifier \"this\" not found");
	CHECK_ASSEMBLE_ERROR("{ pop(ecrecover) }", DeclarationError, "Identifier \"ecrecover\" not found");
	BOOST_CHECK(successAssemble("{ let ecrecover := 1 pop(ecrecover) }"));
}

BOOST_AUTO_TEST_CASE(stack_variables)
{
	BOOST_CHECK(successAssemble("{ let y := 3 { let z := 2 { let x := y } } }"));
}

BOOST_AUTO_TEST_CASE(designated_invalid_instruction)
{
	BOOST_CHECK(successAssemble("{ invalid() }"));
}

BOOST_AUTO_TEST_CASE(inline_assembly_shadowed_instruction_declaration)
{
	CHECK_ASSEMBLE_ERROR("{ let gas := 1 }", ParserError, "Cannot use builtin");
}

BOOST_AUTO_TEST_CASE(revert)
{
	BOOST_CHECK(successAssemble("{ revert(0, 0) }"));
}

BOOST_AUTO_TEST_CASE(large_constant)
{
	auto source = R"({
		switch mul(1, 2)
		case 0x0000000000000000000000000000000000000000000000000000000026121ff0 {
		}
	})";
	BOOST_CHECK(successAssemble(source));
}

BOOST_AUTO_TEST_CASE(keccak256)
{
	BOOST_CHECK(successAssemble("{ pop(keccak256(0, 0)) }"));
}

BOOST_AUTO_TEST_CASE(returndatasize)
{
	if (!solidity::test::CommonOptions::get().evmVersion().supportsReturndata())
		return;
	BOOST_CHECK(successAssemble("{ let r := returndatasize() }"));
}

BOOST_AUTO_TEST_CASE(returndatacopy)
{
	if (!solidity::test::CommonOptions::get().evmVersion().supportsReturndata())
		return;
	BOOST_CHECK(successAssemble("{ returndatacopy(0, 32, 64) }"));
}

BOOST_AUTO_TEST_CASE(staticcall)
{
	if (!solidity::test::CommonOptions::get().evmVersion().hasStaticCall())
		return;
	BOOST_CHECK(successAssemble("{ pop(staticcall(10000, 0x123, 64, 0x10, 128, 0x10)) }"));
}

BOOST_AUTO_TEST_CASE(create2)
{
	if (!solidity::test::CommonOptions::get().evmVersion().hasCreate2())
		return;
	BOOST_CHECK(successAssemble("{ pop(create2(10, 0x123, 32, 64)) }"));
}

BOOST_AUTO_TEST_CASE(shift)
{
	if (!solidity::test::CommonOptions::get().evmVersion().hasBitwiseShifting())
		return;
	BOOST_CHECK(successAssemble("{ pop(shl(10, 32)) }"));
	BOOST_CHECK(successAssemble("{ pop(shr(10, 32)) }"));
	BOOST_CHECK(successAssemble("{ pop(sar(10, 32)) }"));
}

BOOST_AUTO_TEST_CASE(shift_constantinople_warning)
{
	if (solidity::test::CommonOptions::get().evmVersion().hasBitwiseShifting())
		return;
	CHECK_PARSE_WARNING("{ shl(10, 32) }", TypeError, "The \"shl\" instruction is only available for Constantinople-compatible VMs");
	CHECK_PARSE_WARNING("{ shr(10, 32) }", TypeError, "The \"shr\" instruction is only available for Constantinople-compatible VMs");
	CHECK_PARSE_WARNING("{ sar(10, 32) }", TypeError, "The \"sar\" instruction is only available for Constantinople-compatible VMs");
}

BOOST_AUTO_TEST_SUITE_END() // }}}

BOOST_AUTO_TEST_SUITE_END()

} // end namespaces
