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

#pragma once

#include <libsolutil/Exceptions.h>
#include <liblangutil/EVMVersion.h>
#include <liblangutil/Exceptions.h>
#include <libsolutil/Numeric.h>

#include <test/evmc/evmc.h>

#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>
#include <boost/test/unit_test.hpp>

namespace solidity::test
{

#ifdef _WIN32
static constexpr auto evmoneFilename = "evmone.dll";
static constexpr auto evmoneDownloadLink = "https://github.com/ethereum/evmone/releases/download/v0.12.0/evmone-0.12.0-windows-amd64.zip";
#elif defined(__APPLE__)
static constexpr auto evmoneFilename = "libevmone.dylib";
static constexpr auto evmoneDownloadLink = "https://github.com/ethereum/evmone/releases/download/v0.12.0/evmone-0.12.0-darwin-arm64.tar.gz";
#else
static constexpr auto evmoneFilename = "libevmone.so";
static constexpr auto evmoneDownloadLink = "https://github.com/ethereum/evmone/releases/download/v0.12.0/evmone-0.12.0-linux-x86_64.tar.gz";
#endif

struct ConfigException: public util::Exception {};

struct CommonOptions
{
	/// Noncopyable.
	CommonOptions(CommonOptions const&) = delete;
	CommonOptions& operator=(CommonOptions const&) = delete;

	std::vector<boost::filesystem::path> vmPaths;
	boost::filesystem::path testPath;
	bool optimize = false;
	bool enforceGasTest = false;
	u256 enforceGasTestMinValue = 100000;
	bool disableSemanticTests = false;
	bool disableSMT = false;
	bool useABIEncoderV1 = false;
	bool showMessages = false;
	bool showMetadata = false;
	size_t batches = 1;
	size_t selectedBatch = 0;

	langutil::EVMVersion evmVersion() const;
	std::optional<uint8_t> eofVersion() const { return m_eofVersion; }

	virtual void addOptions();
	// @returns true if the program should continue, false if it should exit immediately without
	// reporting an error.
	// Throws ConfigException or std::runtime_error if parsing fails.
	virtual bool parse(int argc, char const* const* argv);
	// Throws a ConfigException on error
	virtual void validate() const;

	/// @returns string with a key=value list of the options separated by comma
	/// Ex.: "evmVersion=london, optimize=true, useABIEncoderV1=false"
	virtual std::string toString(std::vector<std::string> const& _selectedOptions) const;
	/// Helper to print the value of settings used
	virtual void printSelectedOptions(std::ostream& _stream, std::string const& _linePrefix, std::vector<std::string> const& _selectedOptions) const;

	static CommonOptions const& get();
	static void setSingleton(std::unique_ptr<CommonOptions const>&& _instance);

	CommonOptions(std::string caption = "");
	virtual ~CommonOptions() {}

protected:
	boost::program_options::options_description options;

private:
	std::string evmVersionString;
	std::optional<uint8_t> m_eofVersion;
	static std::unique_ptr<CommonOptions const> m_singleton;
};

/// @return true if it is ok to treat the file located under the specified path as a semantic test.
/// I.e. if the test is located in the semantic test directory and is not excluded due to being a part of external sources.
/// Note: @p _testPath can be relative but must include at least the `/test/libsolidity/semanticTests/` part
bool isValidSemanticTestPath(boost::filesystem::path const& _testPath);

/// Helper that can be used to skip tests when the EVM version selected on the command line
/// is older than @p _minEVMVersion.
/// @return A predicate (function) that can be passed into @a boost::unit_test::precondition().
boost::unit_test::precondition::predicate_t minEVMVersionCheck(langutil::EVMVersion _minEVMVersion);

bool loadVMs(CommonOptions const& _options);

/**
 * Component to help with splitting up all tests into batches.
 */
class Batcher
{
public:
	Batcher(size_t _offset, size_t _batches):
		m_offset(_offset),
		m_batches(_batches)
	{
		solAssert(m_batches > 0 && m_offset < m_batches);
	}
	Batcher(Batcher const&) = delete;
	Batcher& operator=(Batcher const&) = delete;

	bool checkAndAdvance() { return (m_counter++) % m_batches == m_offset; }

private:
	size_t const m_offset;
	size_t const m_batches;
	size_t m_counter = 0;
};

}
