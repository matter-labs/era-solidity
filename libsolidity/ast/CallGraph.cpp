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

using namespace solidity::frontend;

bool CallGraph::CompareByID::operator()(Node const& _lhs, Node const& _rhs) const
{
	if (_lhs.index() != _rhs.index())
		return _lhs.index() < _rhs.index();

	if (std::holds_alternative<SpecialNode>(_lhs))
		return std::get<SpecialNode>(_lhs) < std::get<SpecialNode>(_rhs);
	return std::get<CallableDeclaration const*>(_lhs)->id() < std::get<CallableDeclaration const*>(_rhs)->id();
}

bool CallGraph::CompareByID::operator()(Node const& _lhs, int64_t _rhs) const
{
	solAssert(!std::holds_alternative<SpecialNode>(_lhs), "");

	return std::get<CallableDeclaration const*>(_lhs)->id() < _rhs;
}

bool CallGraph::CompareByID::operator()(int64_t _lhs, Node const& _rhs) const
{
	solAssert(!std::holds_alternative<SpecialNode>(_rhs), "");

	return _lhs < std::get<CallableDeclaration const*>(_rhs)->id();
}

/// Populates reachable cycles from m_src into paths;
class CycleFinder
{
	CallGraph const& m_callGraph;
	CallableDeclaration const* m_src;
	std::set<CallableDeclaration const*> m_processing;
	std::set<CallableDeclaration const*> m_processed;
	std::vector<CallGraph::Path> m_paths;

	/// Populates `m_paths` with cycles reachable from @a _callable
	void getCyclesInternal(CallableDeclaration const* _callable, CallGraph::Path& _path)
	{
		if (m_processed.count(_callable))
			return;

		auto callees = m_callGraph.edges.find(_callable);
		// A leaf node?
		if (callees == m_callGraph.edges.end())
		{
			m_processed.insert(_callable);
			return;
		}


		std::set<CallableDeclaration const*> allPossibleCallees;
		for (auto callee: callees->second)
		{
			if (auto calleeFn = std::get_if<CallableDeclaration const*>(&callee))
			{
				allPossibleCallees.insert(*calleeFn);
				continue;
			}
			auto specialNode = std::get<CallGraph::SpecialNode>(callee);
			if (specialNode == CallGraph::SpecialNode::InternalDispatch)
			{
				auto dispatchNode = m_callGraph.edges.find(specialNode);
				if (dispatchNode == m_callGraph.edges.end())
					continue;
				for (auto indirectCallee: dispatchNode->second)
					if (auto indirectCalleeFn = std::get_if<CallableDeclaration const*>(&indirectCallee))
						allPossibleCallees.insert(*indirectCalleeFn);
			}
		}

		// A leaf node?
		if (allPossibleCallees.empty())
		{
			m_processed.insert(_callable);
			return;
		}

		m_processing.insert(_callable);
		_path.push_back(_callable);

		for (auto callee: allPossibleCallees)
		{
			if (m_processing.count(callee))
			{
				// Extract the cycle
				auto cycleStart = std::find(_path.begin(), _path.end(), callee);
				solAssert(cycleStart != _path.end(), "");
				m_paths.emplace_back(cycleStart, _path.end());
				continue;
			}

			getCyclesInternal(callee, _path);
		}

		m_processing.erase(_callable);
		m_processed.insert(_callable);
		_path.pop_back();
	}

public:
	CycleFinder(CallGraph const& _callGraph, CallableDeclaration const* _src): m_callGraph(_callGraph), m_src(_src) {}

	std::vector<CallGraph::Path> getCycles()
	{
		CallGraph::Path p;
		getCyclesInternal(m_src, p);
		return m_paths;
	}

	void dump(std::ostream& _out)
	{
		for (CallGraph::Path const& path: m_paths)
		{
			for (CallableDeclaration const* func: path)
				_out << func->name() << " -> ";
			_out << "\n";
		}
	}
};

std::set<CallableDeclaration const*> CallGraph::getFuncs() const
{
	std::set<CallableDeclaration const*> funcs;
	for (auto edge: edges)
	{
		Node src = edge.first;
		if (auto srcFn = std::get_if<CallableDeclaration const*>(&src))
			funcs.insert(*srcFn);
		for (Node dst: edge.second)
			if (auto dstFn = std::get_if<CallableDeclaration const*>(&dst))
				funcs.insert(*dstFn);
	}
	return funcs;
}

std::set<CallableDeclaration const*> CallGraph::getReachableCycleFuncs(CallableDeclaration const* _src) const
{
	std::set<CallableDeclaration const*> funcs;
	CycleFinder cf{*this, _src};
	std::vector<CallGraph::Path> paths = cf.getCycles();

	for (CallGraph::Path const& path: paths)
	{
		for (CallableDeclaration const* func: path)
		{
			funcs.insert(func);
		}
	}
	return funcs;
}
