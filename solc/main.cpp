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
 * Solidity commandline compiler.
 */

#include <solc/CommandLineInterface.h>

#include <liblangutil/Exceptions.h>

#include <boost/exception/all.hpp>

#include <iostream>

#include <fstream>
#include <libsolc/libsolc.h>
#include <sstream>

using namespace solidity;
using namespace solidity::frontend;


int main(int argc, char** argv)
{
	const char* src = R"(
{
  "language": "Yul",
  "sources": {
    "Test.yul": {
      "urls": [
        "Test.yul"
      ]
    }
  },
  "settings": {
    "optimizer": {
      "enabled": true
    },
    "outputSelection": {
      "*": {
        "*": [
          "evm"
        ]
      }
    },
    "metadata": {
      "useLiteralContent": false
    }
  }
}
)";
	CStyleReadFileCallback callback{
		[](void* _context, char const* _kind, char const* _path, char** o_contents, char** o_error)
		{
			assert(_context == nullptr);
			assert(std::string(_kind) == ReadCallback::kindString(ReadCallback::Kind::ReadFile));

			// The client has to find the file (so, --base-path, --include-path etc. handling should be done by the
			// client).  solc cmdline uses UniversalCallback (the default import callback), but I don't see any way to
			// use that in libsolc's C interface.

			// For demonstration:
			if (std::string(_path) == "Test.yul")
			{
				// Read the file.
				std::ifstream fs("/home/abinavpp/tst/run/Test.yul");
				std::stringstream ss;
				ss << fs.rdbuf();
				std::string content = ss.str();

				// Copy its contents.
				*o_contents = solidity_alloc(content.length());
				std::memcpy(*o_contents, content.c_str(), content.length());
				*o_error = nullptr;
			}
			else
			{
				*o_error = nullptr;
				*o_contents = nullptr;
			}
		}};

	std::cout << solidity_compile(src, callback, nullptr) << "\n";
	exit(12);

	try
	{
		solidity::frontend::CommandLineInterface cli(std::cin, std::cout, std::cerr);
		return cli.run(argc, argv) ? 0 : 1;
	}
	catch (smtutil::SMTLogicError const& _exception)
	{
		std::cerr << "SMT logic error:" << std::endl;
		std::cerr << boost::diagnostic_information(_exception);
		return 2;
	}
	catch (langutil::InternalCompilerError const& _exception)
	{
		std::cerr << "Internal compiler error:" << std::endl;
		std::cerr << boost::diagnostic_information(_exception);
		return 2;
	}
	catch (...)
	{
		std::cerr << "Uncaught exception:" << std::endl;
		std::cerr << boost::current_exception_diagnostic_information() << std::endl;
		return 2;
	}
}
