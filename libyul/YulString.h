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
 * String abstraction that avoids copies.
 */

#pragma once

#include <fmt/format.h>

#include <cstdint>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace solidity::yul
{

/// Repository for YulStrings.
/// Owns the string data for all YulStrings, which can be referenced by a Handle.
/// A Handle consists of an ID (that depends on the insertion order of YulStrings and is potentially
/// non-deterministic) and a deterministic string hash.
class YulStringRepository
{
public:
	struct Handle
	{
		size_t id;
		std::uint64_t hash;
	};

	static YulStringRepository& instance()
	{
		static YulStringRepository inst;
		return inst;
	}

	Handle stringToHandle(std::string const& _string)
	{
		if (_string.empty())
			return { 0, emptyHash() };
		std::uint64_t h = hash(_string);
		auto range = m_hashToID.equal_range(h);
		for (auto it = range.first; it != range.second; ++it)
			if (*m_strings[it->second] == _string)
				return Handle{it->second, h};
		m_strings.emplace_back(std::make_shared<std::string>(_string));
		size_t id = m_strings.size() - 1;
		m_hashToID.emplace_hint(range.second, std::make_pair(h, id));

		return Handle{id, h};
	}
	std::string const& idToString(size_t _id) const { return *m_strings.at(_id); }

	static std::uint64_t hash(std::string const& v)
	{
		// FNV hash - can be replaced by a better one, e.g. xxhash64
		std::uint64_t hash = emptyHash();
		for (char c: v)
		{
			hash *= 1099511628211u;
			hash ^= static_cast<uint64_t>(c);
		}

		return hash;
	}
	static constexpr std::uint64_t emptyHash() { return 14695981039346656037u; }
	/// Clear the repository.
	/// Use with care - there cannot be any dangling YulString references.
	/// If references need to be cleared manually, register the callback via
	/// resetCallback.
	static void reset()
	{
		for (auto const& cb: resetCallbacks())
			cb();
		instance() = YulStringRepository{};
	}
	/// Struct that registers a reset callback as a side-effect of its construction.
	/// Useful as static local variable to register a reset callback once.
	struct ResetCallback
	{
		ResetCallback(std::function<void()> _fun)
		{
			YulStringRepository::resetCallbacks().emplace_back(std::move(_fun));
		}
	};

private:
	YulStringRepository() = default;
	YulStringRepository(YulStringRepository const&) = delete;
	YulStringRepository(YulStringRepository&&) = default;
	YulStringRepository& operator=(YulStringRepository const& _rhs) = delete;
	YulStringRepository& operator=(YulStringRepository&& _rhs) = default;

	static std::vector<std::function<void()>>& resetCallbacks()
	{
		static std::vector<std::function<void()>> callbacks;
		return callbacks;
	}

	std::vector<std::shared_ptr<std::string>> m_strings = {std::make_shared<std::string>()};
	std::unordered_multimap<std::uint64_t, size_t> m_hashToID = {{emptyHash(), 0}};
};

/// Wrapper around handles into the YulString repository.
/// Equality of two YulStrings is determined by comparing their ID.
/// The <-operator depends on the string hash and is not consistent
/// with string comparisons (however, it is still deterministic).
class YulString
{
public:
	YulString() = default;
	explicit YulString(std::string const& _s): m_handle(YulStringRepository::instance().stringToHandle(_s)) {}
	YulString(YulString const&) = default;
	YulString(YulString&&) = default;
	YulString& operator=(YulString const&) = default;
	YulString& operator=(YulString&&) = default;

	/// This is not consistent with the string <-operator!
	/// First compares the string hashes. If they are equal
	/// it checks for identical IDs (only identical strings have
	/// identical IDs and identical strings do not compare as "less").
	/// If the hashes are identical and the strings are distinct, it
	/// falls back to string comparison.
	bool operator<(YulString const& _other) const
	{
		if (m_handle.hash < _other.m_handle.hash) return true;
		if (_other.m_handle.hash < m_handle.hash) return false;
		if (m_handle.id == _other.m_handle.id) return false;
		return str() < _other.str();
	}
	/// Equality is determined based on the string ID.
	bool operator==(YulString const& _other) const { return m_handle.id == _other.m_handle.id; }
	bool operator!=(YulString const& _other) const { return m_handle.id != _other.m_handle.id; }

	bool empty() const { return m_handle.id == 0; }
	std::string const& str() const
	{
		return YulStringRepository::instance().idToString(m_handle.id);
	}

	uint64_t hash() const { return m_handle.hash; }

private:
	/// Handle of the string. Assumes that the empty string has ID zero.
	YulStringRepository::Handle m_handle{ 0, YulStringRepository::emptyHash() };
};

inline YulString operator "" _yulstring(char const* _string, std::size_t _size)
{
	return YulString(std::string(_string, _size));
}

}

namespace fmt
{
template <>
struct formatter<solidity::yul::YulString>
{
	template <typename ParseContext>
	constexpr auto parse(ParseContext& _context)
	{
		return _context.begin();
	}

	template <typename FormatContext>
	auto format(solidity::yul::YulString _value, FormatContext& _context)
	{
		return format_to(_context.out(), "{}", _value.str());
	}
};
}

namespace std
{
template<> struct hash<solidity::yul::YulString>
{
	size_t operator()(solidity::yul::YulString const& x) const
	{
		return static_cast<size_t>(x.hash());
	}
};
}
