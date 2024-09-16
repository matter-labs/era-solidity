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

#include <libsolidity/lsp/FileRepository.h>
#include <libsolidity/lsp/Transport.h>
#include <libsolidity/lsp/Utils.h>

#include <libsolutil/StringUtils.h>
#include <libsolutil/CommonIO.h>

#include <range/v3/algorithm/none_of.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <regex>

#include <boost/algorithm/string/predicate.hpp>

#include <fmt/format.h>

using namespace solidity;
using namespace solidity::lsp;
using namespace solidity::frontend;

using solidity::util::readFileAsString;
using solidity::util::joinHumanReadable;
using solidity::util::Result;

FileRepository::FileRepository(boost::filesystem::path _basePath, std::vector<boost::filesystem::path> _includePaths):
	m_basePath(std::move(_basePath)),
	m_includePaths(std::move(_includePaths))
{
}

void FileRepository::setIncludePaths(std::vector<boost::filesystem::path> _paths)
{
	m_includePaths = std::move(_paths);
}

std::string FileRepository::sourceUnitNameToUri(std::string const& _sourceUnitName) const
{
	std::regex const windowsDriveLetterPath("^[a-zA-Z]:/");

	auto const ensurePathIsUnixLike = [&](std::string inputPath) -> std::string {
		if (!regex_search(inputPath, windowsDriveLetterPath))
			return inputPath;
		else
			return "/" + std::move(inputPath);
	};

	if (m_sourceUnitNamesToUri.count(_sourceUnitName))
	{
		solAssert(boost::starts_with(m_sourceUnitNamesToUri.at(_sourceUnitName), "file://"), "");
		return m_sourceUnitNamesToUri.at(_sourceUnitName);
	}
	else if (_sourceUnitName.find("file://") == 0)
		return _sourceUnitName;
	else if (regex_search(_sourceUnitName, windowsDriveLetterPath))
		return "file:///" + _sourceUnitName;
	else if (
		auto const resolvedPath = tryResolvePath(_sourceUnitName);
		resolvedPath.message().empty()
	)
		return "file://" + ensurePathIsUnixLike(resolvedPath.get().generic_string());
	else if (m_basePath.generic_string() != "/")
		return "file://" + m_basePath.generic_string() + "/" + _sourceUnitName;
	else
		// Avoid double-/ in case base-path itself is simply a UNIX root filesystem root.
		return "file:///" + _sourceUnitName;
}

std::string FileRepository::uriToSourceUnitName(std::string const& _path) const
{
	lspRequire(boost::algorithm::starts_with(_path, "file://"), ErrorCode::InternalError, "URI must start with file://");
	return stripFileUriSchemePrefix(_path);
}

void FileRepository::setSourceByUri(std::string const& _uri, std::string _source)
{
	// This is needed for uris outside the base path. It can lead to collisions,
	// but we need to mostly rewrite this in a future version anyway.
	auto sourceUnitName = uriToSourceUnitName(_uri);
	lspDebug(fmt::format("FileRepository.setSourceByUri({}): {}", _uri, _source));
	m_sourceUnitNamesToUri.emplace(sourceUnitName, _uri);
	m_sourceCodes[sourceUnitName] = std::move(_source);
}

Result<boost::filesystem::path> FileRepository::tryResolvePath(std::string const& _strippedSourceUnitName) const
{
	if (
		boost::filesystem::path(_strippedSourceUnitName).has_root_path() &&
		boost::filesystem::exists(_strippedSourceUnitName)
	)
		return boost::filesystem::path(_strippedSourceUnitName);

	std::vector<boost::filesystem::path> candidates;
	std::vector<std::reference_wrapper<boost::filesystem::path const>> prefixes = {m_basePath};
	prefixes += (m_includePaths | ranges::to<std::vector<std::reference_wrapper<boost::filesystem::path const>>>);
	auto const defaultInclude = m_basePath / "node_modules";
	if (m_includePaths.empty())
		prefixes.emplace_back(defaultInclude);

	auto const pathToQuotedString = [](boost::filesystem::path const& _path) { return "\"" + _path.string() + "\""; };

	for (auto const& prefix: prefixes)
	{
		boost::filesystem::path canonicalPath = boost::filesystem::path(prefix) / boost::filesystem::path(_strippedSourceUnitName);

		if (boost::filesystem::exists(canonicalPath))
			candidates.push_back(std::move(canonicalPath));
	}

	if (candidates.empty())
		return Result<boost::filesystem::path>::err(
			"File not found. Searched the following locations: " +
			joinHumanReadable(prefixes | ranges::views::transform(pathToQuotedString), ", ") +
			"."
		);

	if (candidates.size() >= 2)
		return Result<boost::filesystem::path>::err(
			"Ambiguous import. "
			"Multiple matching files found inside base path and/or include paths: " +
			joinHumanReadable(candidates | ranges::views::transform(pathToQuotedString), ", ") +
			"."
		);

	if (!boost::filesystem::is_regular_file(candidates[0]))
		return Result<boost::filesystem::path>::err("Not a valid file.");

	return candidates[0];
}

frontend::ReadCallback::Result FileRepository::readFile(std::string const& _kind, std::string const& _sourceUnitName)
{
	solAssert(
		_kind == ReadCallback::kindString(ReadCallback::Kind::ReadFile),
		"ReadFile callback used as callback kind " + _kind
	);

	try
	{
		// File was read already. Use local store.
		if (m_sourceCodes.count(_sourceUnitName))
			return ReadCallback::Result{true, m_sourceCodes.at(_sourceUnitName)};

		std::string const strippedSourceUnitName = stripFileUriSchemePrefix(_sourceUnitName);
		Result<boost::filesystem::path> const resolvedPath = tryResolvePath(strippedSourceUnitName);
		if (!resolvedPath.message().empty())
			return ReadCallback::Result{false, resolvedPath.message()};

		auto contents = readFileAsString(resolvedPath.get());
		solAssert(m_sourceCodes.count(_sourceUnitName) == 0, "");
		m_sourceCodes[_sourceUnitName] = contents;
		return ReadCallback::Result{true, std::move(contents)};
	}
	catch (...)
	{
		return ReadCallback::Result{false, "Exception in read callback: " + boost::current_exception_diagnostic_information()};
	}
}

