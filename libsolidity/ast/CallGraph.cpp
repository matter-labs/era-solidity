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

// TODO? Merge this with CycleDetector?
/// Populates reachable cycles from m_src into paths;
class CycleFinder
{
	CallGraph const& m_callGraph;
	CallableDeclaration const* m_src;
	set<CallableDeclaration const*> m_processing;
	set<CallableDeclaration const*> m_processed;

public:
	CycleFinder(CallGraph const& _callGraph, CallableDeclaration const* _src, vector<CallGraph::Path>& _paths)
		: m_callGraph(_callGraph), m_src(_src), paths(_paths)
	{
	}
	vector<CallGraph::Path>& paths;

	void find(CallableDeclaration const* _callable, CallGraph::Path& _path)
	{
		if (m_processed.count(_callable))
			return;

		auto callees = m_callGraph.edges.find(_callable);
		if (callees == m_callGraph.edges.end())
		{
			solAssert(m_processing.count(_callable) == 0, "");
			m_processed.insert(_callable);
			return;
		}

		m_processing.insert(_callable);
		_path.push_back(_callable);
		for (auto const& calleeVariant: callees->second)
		{
			if (!holds_alternative<CallableDeclaration const*>(calleeVariant))
				continue;
			auto* callee = get<CallableDeclaration const*>(calleeVariant);

			if (m_processing.count(callee))
			{
				paths.push_back(_path);
				continue;
			}

			find(callee, _path);
		}

		m_processing.erase(_callable);
		m_processed.insert(_callable);
		_path.pop_back();
	}

	void run()
	{
		CallGraph::Path p;
		find(m_src, p);
	}

	void dump(ostream& _out)
	{
		for (CallGraph::Path const& path: paths)
		{
			for (CallableDeclaration const* func: path)
			{
				_out << func->name() << " -> ";
			}
			_out << "\n";
		}
	}
};

void CallGraph::getCycles(CallableDeclaration const* _src, vector<Path>& _paths) const
{
	CycleFinder cf{*this, _src, _paths};
	cf.run();
}

void CallGraph::getReachableFuncs(CallableDeclaration const* _src, std::set<CallableDeclaration const*>& _funcs) const
{
	if (_funcs.count(_src))
		return;
	_funcs.insert(_src);

	auto callees = edges.find(_src);
	if (callees == edges.end())
		return;

	for (auto const& calleeVariant: callees->second)
	{
		if (!holds_alternative<CallableDeclaration const*>(calleeVariant))
			continue;
		auto* callee = get<CallableDeclaration const*>(calleeVariant);
		getReachableFuncs(callee, _funcs);
	}
}

void CallGraph::getReachableCycleFuncs(
	CallableDeclaration const* _src, std::set<CallableDeclaration const*>& _funcs) const
{
	vector<CallGraph::Path> paths;
	getCycles(_src, paths);

	for (CallGraph::Path const& path: paths)
	{
		for (CallableDeclaration const* func: path)
		{
			_funcs.insert(func);
		}
	}
}

bool CallGraph::hasReachableCycle(CallableDeclaration const* _callable) const
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
			if (!holds_alternative<CallableDeclaration const*>(calleeVariant))
				return;
			_cycleDetector.run(*get<CallableDeclaration const*>(calleeVariant));
		}
	};

	return util::CycleDetector<CallableDeclaration>(visitor).run(*_callable);
}
