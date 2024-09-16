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

#include <fstream>

#include <test/tools/ossfuzz/yulProto.pb.h>
#include <test/tools/fuzzer_common.h>
#include <test/tools/ossfuzz/protoToYul.h>

#include <test/libyul/YulOptimizerTestCommon.h>

#include <src/libfuzzer/libfuzzer_macro.h>

#include <libyul/AST.h>
#include <libyul/YulStack.h>
#include <libyul/backends/evm/EVMDialect.h>
#include <libyul/Exceptions.h>

#include <liblangutil/DebugInfoSelection.h>
#include <liblangutil/EVMVersion.h>
#include <liblangutil/SourceReferenceFormatter.h>

#include <test/tools/ossfuzz/yulFuzzerCommon.h>

using namespace solidity;
using namespace solidity::util;
using namespace solidity::langutil;
using namespace solidity::yul;
using namespace solidity::yul::test;
using namespace solidity::yul::test::yul_fuzzer;

DEFINE_PROTO_FUZZER(Program const& _input)
{
	ProtoConverter converter;
	std::string yul_source = converter.programToString(_input);
	EVMVersion version = converter.version();

	if (const char* dump_path = getenv("PROTO_FUZZER_DUMP_PATH"))
	{
		// With libFuzzer binary run this to generate a YUL source file x.yul:
		// PROTO_FUZZER_DUMP_PATH=x.yul ./a.out proto-input
		std::ofstream of(dump_path);
		of.write(yul_source.data(), static_cast<std::streamsize>(yul_source.size()));
	}

	YulStringRepository::reset();

	// YulStack entry point
	YulStack stack(
		version,
		std::nullopt,
		YulStack::Language::StrictAssembly,
		solidity::frontend::OptimiserSettings::full(),
		DebugInfoSelection::All()
	);

	// Parse protobuf mutated YUL code
	if (
		!stack.parseAndAnalyze("source", yul_source) ||
		!stack.parserResult()->code() ||
		!stack.parserResult()->analysisInfo ||
		Error::containsErrors(stack.errors())
	)
	{
		SourceReferenceFormatter{std::cout, stack, false, false}.printErrorInformation(stack.errors());
		yulAssert(false, "Proto fuzzer generated malformed program");
	}

	std::ostringstream os1;
	std::ostringstream os2;
	// Disable memory tracing to avoid false positive reports
	// such as unused write to memory e.g.,
	// { mstore(0, 1) }
	// that would be removed by the redundant store eliminator.
	yulFuzzerUtil::TerminationReason termReason = yulFuzzerUtil::interpret(
		os1,
		stack.parserResult()->code()->root(),
		EVMDialect::strictAssemblyForEVMObjects(version),
		/*disableMemoryTracing=*/true
	);

	if (yulFuzzerUtil::resourceLimitsExceeded(termReason))
		return;

	YulOptimizerTestCommon optimizerTest(
		stack.parserResult(),
		EVMDialect::strictAssemblyForEVMObjects(version)
	);
	optimizerTest.setStep(optimizerTest.randomOptimiserStep(_input.step()));
	auto const* astRoot = optimizerTest.run();
	yulAssert(astRoot != nullptr, "Optimiser error.");
	termReason = yulFuzzerUtil::interpret(
		os2,
		*astRoot,
		EVMDialect::strictAssemblyForEVMObjects(version),
		true
	);
	if (yulFuzzerUtil::resourceLimitsExceeded(termReason))
		return;

	bool isTraceEq = (os1.str() == os2.str());
	if (!isTraceEq)
	{
		std::cout << os1.str() << std::endl;
		std::cout << os2.str() << std::endl;
		yulAssert(false, "Interpreted traces for optimized and unoptimized code differ.");
	}
	return;
}
