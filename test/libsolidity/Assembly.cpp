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
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 * Unit tests for Assembly Items from evmasm/Assembly.h
 */

#include <test/Common.h>

#include <liblangutil/SourceLocation.h>
#include <libevmasm/Assembly.h>

#include <liblangutil/CharStream.h>

#include <libsolidity/parsing/Parser.h>
#include <libsolidity/analysis/DeclarationTypeChecker.h>
#include <libsolidity/analysis/NameAndTypeResolver.h>
#include <libsolidity/analysis/Scoper.h>
#include <libsolidity/codegen/Compiler.h>
#include <libsolidity/ast/AST.h>
#include <libsolidity/analysis/TypeChecker.h>
#include <libsolidity/analysis/SyntaxChecker.h>
#include <liblangutil/ErrorReporter.h>

#include <boost/test/unit_test.hpp>

#include <string>
#include <iostream>

using namespace solidity::langutil;
using namespace solidity::evmasm;

namespace solidity::frontend::test
{

namespace
{

evmasm::AssemblyItems compileContract(std::shared_ptr<CharStream> _sourceCode)
{
	ErrorList errors;
	ErrorReporter errorReporter(errors);
	Parser parser(
		errorReporter,
		solidity::test::CommonOptions::get().evmVersion(),
		solidity::test::CommonOptions::get().eofVersion()
	);
	ASTPointer<SourceUnit> sourceUnit;
	BOOST_REQUIRE_NO_THROW(sourceUnit = parser.parse(*_sourceCode));
	BOOST_CHECK(!!sourceUnit);

	Scoper::assignScopes(*sourceUnit);
	BOOST_REQUIRE(SyntaxChecker(errorReporter, false).checkSyntax(*sourceUnit));
	GlobalContext globalContext(solidity::test::CommonOptions::get().evmVersion());
	NameAndTypeResolver resolver(globalContext, solidity::test::CommonOptions::get().evmVersion(), errorReporter, false);
	DeclarationTypeChecker declarationTypeChecker(errorReporter, solidity::test::CommonOptions::get().evmVersion());
	solAssert(!Error::containsErrors(errorReporter.errors()), "");
	resolver.registerDeclarations(*sourceUnit);
	BOOST_REQUIRE_NO_THROW(resolver.resolveNamesAndTypes(*sourceUnit));
	if (Error::containsErrors(errorReporter.errors()))
		return AssemblyItems();
	for (ASTPointer<ASTNode> const& node: sourceUnit->nodes())
	{
		BOOST_REQUIRE_NO_THROW(declarationTypeChecker.check(*node));
		if (Error::containsErrors(errorReporter.errors()))
			return AssemblyItems();
	}
	TypeChecker checker(solidity::test::CommonOptions::get().evmVersion(), errorReporter);
	BOOST_REQUIRE_NO_THROW(checker.checkTypeRequirements(*sourceUnit));
	if (Error::containsErrors(errorReporter.errors()))
		return AssemblyItems();
	for (ASTPointer<ASTNode> const& node: sourceUnit->nodes())
		if (ContractDefinition* contract = dynamic_cast<ContractDefinition*>(node.get()))
		{
			Compiler compiler(
				solidity::test::CommonOptions::get().evmVersion(),
				RevertStrings::Default,
				solidity::test::CommonOptions::get().optimize ? OptimiserSettings::standard() : OptimiserSettings::minimal()
			);
			compiler.compileContract(*contract, std::map<ContractDefinition const*, std::shared_ptr<Compiler const>>{}, bytes());

			BOOST_REQUIRE(compiler.runtimeAssembly().codeSections().size() == 1);
			return compiler.runtimeAssembly().codeSections().at(0).items;
		}
	BOOST_FAIL("No contract found in source.");
	return AssemblyItems();
}

void printAssemblyLocations(AssemblyItems const& _items)
{
	auto printRepeated = [](SourceLocation const& _loc, size_t _repetitions)
	{
		std::cout <<
			"\t\tvector<SourceLocation>(" <<
			_repetitions <<
			", SourceLocation{" <<
			_loc.start <<
			", " <<
			_loc.end <<
			", make_shared<string>(\"" <<
			*_loc.sourceName <<
			"\")}) +" << std::endl;
	};

	std::vector<SourceLocation> locations;
	for (auto const& item: _items)
		locations.push_back(item.location());
	size_t repetitions = 0;
	SourceLocation const* previousLoc = nullptr;
	for (size_t i = 0; i < locations.size(); ++i)
	{
		SourceLocation& loc = locations[i];
		if (previousLoc && *previousLoc == loc)
			repetitions++;
		else
		{
			if (previousLoc)
				printRepeated(*previousLoc, repetitions);
			previousLoc = &loc;
			repetitions = 1;
		}
	}
	if (previousLoc)
		printRepeated(*previousLoc, repetitions);
}

void checkAssemblyLocations(AssemblyItems const& _items, std::vector<SourceLocation> const& _locations)
{
	BOOST_CHECK_EQUAL(_items.size(), _locations.size());
	for (size_t i = 0; i < std::min(_items.size(), _locations.size()); ++i)
	{
		if (_items[i].location().start != _locations[i].start ||
			_items[i].location().end != _locations[i].end)
		{
			BOOST_CHECK_MESSAGE(false, "Location mismatch for item " + std::to_string(i) + ". Found the following locations:");
			printAssemblyLocations(_items);
			return;
		}
	}
}


} // end anonymous namespace

BOOST_AUTO_TEST_SUITE(Assembly)

BOOST_AUTO_TEST_CASE(location_test)
{
	std::string sourceCode = R"(
	pragma abicoder v1;
	contract test {
		function f() public returns (uint256 a) {
			return 16;
		}
	}
	)";
	AssemblyItems items = compileContract(std::make_shared<CharStream>(sourceCode, ""));
	std::shared_ptr<std::string> sourceName = std::make_shared<std::string>();
	bool hasShifts = solidity::test::CommonOptions::get().evmVersion().hasBitwiseShifting();

	auto codegenCharStream = std::make_shared<CharStream>("", "--CODEGEN--");

	std::vector<SourceLocation> locations;
	if (solidity::test::CommonOptions::get().optimize)
		locations =
			std::vector<SourceLocation>(31, SourceLocation{23, 103, sourceName}) +
			std::vector<SourceLocation>(1, SourceLocation{41, 100, sourceName}) +
			std::vector<SourceLocation>(1, SourceLocation{93, 95, sourceName}) +
			std::vector<SourceLocation>(15, SourceLocation{41, 100, sourceName});
	else
		locations =
			std::vector<SourceLocation>(hasShifts ? 31 : 32, SourceLocation{23, 103, sourceName}) +
			std::vector<SourceLocation>(24, SourceLocation{41, 100, sourceName}) +
			std::vector<SourceLocation>(1, SourceLocation{70, 79, sourceName}) +
			std::vector<SourceLocation>(1, SourceLocation{93, 95, sourceName}) +
			std::vector<SourceLocation>(2, SourceLocation{86, 95, sourceName}) +
			std::vector<SourceLocation>(2, SourceLocation{41, 100, sourceName});
	checkAssemblyLocations(items, locations);
}


BOOST_AUTO_TEST_CASE(jump_type)
{
	auto sourceCode = std::make_shared<CharStream>(R"(
	pragma abicoder v1;
	contract C {
		function f(uint a) public pure returns (uint t) {
			assembly {
				function g(x) -> y { if x { leave } y := 8 }
				t := g(a)
			}
		}
	}
	)", "");
	AssemblyItems items = compileContract(sourceCode);

	std::string jumpTypes;
	for (AssemblyItem const& item: items)
		if (item.getJumpType() != AssemblyItem::JumpType::Ordinary)
			jumpTypes += item.getJumpTypeAsString() + "\n";

	if (solidity::test::CommonOptions::get().optimize)
		BOOST_CHECK_EQUAL(jumpTypes, "[in]\n[out]\n[out]\n[in]\n[out]\n");
	else
		BOOST_CHECK_EQUAL(jumpTypes, "[in]\n[out]\n[in]\n[out]\n");
}


BOOST_AUTO_TEST_SUITE_END()

} // end namespaces
