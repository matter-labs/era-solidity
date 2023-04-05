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

#include <libsolidity/ast/CallGraph.h>
#include <libsolutil/Algorithms.h>

using namespace std;
using namespace solidity::frontend;

bool CallGraph::CompareByID::operator()(Node const& _lhs, Node const& _rhs) const
{
	if (_lhs.index() != _rhs.index())
		return _lhs.index() < _rhs.index();

	if (holds_alternative<SpecialNode>(_lhs))
		return get<SpecialNode>(_lhs) < get<SpecialNode>(_rhs);
	return get<CallableDeclaration const*>(_lhs)->id() < get<CallableDeclaration const*>(_rhs)->id();
}

bool CallGraph::CompareByID::operator()(Node const& _lhs, int64_t _rhs) const
{
	solAssert(!holds_alternative<SpecialNode>(_lhs), "");

	return get<CallableDeclaration const*>(_lhs)->id() < _rhs;
}

bool CallGraph::CompareByID::operator()(int64_t _lhs, Node const& _rhs) const
{
	solAssert(!holds_alternative<SpecialNode>(_rhs), "");

	return _lhs < get<CallableDeclaration const*>(_rhs)->id();
}

bool CallGraph::inCycle(CallableDeclaration const* _callable) const
{
	auto callees = edges.find(_callable);
	if (callees == edges.end())
		return false;
	auto visitor
		= [&](CallableDeclaration const& _node, util::CycleDetector<CallableDeclaration>& _cycleDetector, size_t)
	{
		auto callees = edges.find(&_node);
		if (callees == edges.end())
			return;
		for (auto const& calleeVariant: callees->second)
		{
			// FIXME: Signal failure here and in other such instances
			if (!holds_alternative<CallableDeclaration const*>(calleeVariant))
				return;
			_cycleDetector.run(*get<CallableDeclaration const*>(calleeVariant));
		}
	};

	auto* start = util::CycleDetector<CallableDeclaration>(visitor).run(*_callable);

	if (!start)
		return false;
	else if (start == _callable)
		return true;
	else if (start)
	{
		auto callees = edges.find(start);
		for (auto const& callee: callees->second)
		{
			if (!holds_alternative<CallableDeclaration const*>(callee))
				continue;
			if (get<CallableDeclaration const*>(callee) == _callable)
				return true;
		}
	}
	return false;
}
