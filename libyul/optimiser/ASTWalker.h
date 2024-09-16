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
 * Generic AST walker.
 */

#pragma once

#include <libyul/ASTForward.h>

#include <libyul/Exceptions.h>
#include <libyul/YulName.h>

#include <map>
#include <optional>
#include <set>
#include <vector>

namespace solidity::yul
{

/**
 * Generic AST walker.
 */
class ASTWalker
{
public:
	virtual ~ASTWalker() = default;
	virtual void operator()(Literal const&) {}
	virtual void operator()(Identifier const&) {}
	virtual void operator()(FunctionCall const& _funCall);
	virtual void operator()(ExpressionStatement const& _statement);
	virtual void operator()(Assignment const& _assignment);
	virtual void operator()(VariableDeclaration const& _varDecl);
	virtual void operator()(If const& _if);
	virtual void operator()(Switch const& _switch);
	virtual void operator()(FunctionDefinition const&);
	virtual void operator()(ForLoop const&);
	virtual void operator()(Break const&) {}
	virtual void operator()(Continue const&) {}
	virtual void operator()(Leave const&) {}
	virtual void operator()(Block const& _block);

	virtual void visit(Statement const& _st);
	virtual void visit(Expression const& _e);

protected:
	template <class T>
	void walkVector(T const& _statements)
	{
		for (auto const& statement: _statements)
			visit(statement);
	}
};

/**
 * Generic AST modifier (i.e. non-const version of ASTWalker).
 */
class ASTModifier
{
public:
	virtual ~ASTModifier() = default;
	virtual void operator()(Literal&) {}
	virtual void operator()(Identifier&) {}
	virtual void operator()(FunctionCall& _funCall);
	virtual void operator()(ExpressionStatement& _statement);
	virtual void operator()(Assignment& _assignment);
	virtual void operator()(VariableDeclaration& _varDecl);
	virtual void operator()(If& _if);
	virtual void operator()(Switch& _switch);
	virtual void operator()(FunctionDefinition&);
	virtual void operator()(ForLoop&);
	virtual void operator()(Break&);
	virtual void operator()(Continue&);
	virtual void operator()(Leave&);
	virtual void operator()(Block& _block);

	virtual void visit(Statement& _st);
	virtual void visit(Expression& _e);

protected:
	template <class T>
	void walkVector(T&& _statements)
	{
		for (auto& st: _statements)
			visit(st);
	}
};

namespace detail
{
template <
	typename Node,
	typename Visitor,
	typename Base = std::conditional_t<std::is_const_v<Node>, ASTWalker, ASTModifier>
>
struct ForEach: Base
{
	ForEach(Visitor& _visitor): visitor(_visitor) {}

	using Base::operator();
	void operator()(Node& _node) override
	{
		visitor(_node);
		Base::operator()(_node);
	}

	Visitor& visitor;
};
}

/// Helper function that traverses the AST and calls the visitor for each
/// node of a specific type.
template<typename Node, typename Entry, typename Visitor>
void forEach(Entry&& _entry, Visitor&& _visitor)
{
	detail::ForEach<Node, Visitor&>{_visitor}(_entry);
}

}
