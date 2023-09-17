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
/**
 * Generic AST walker.
 */

#pragma once

#include <libjulia/ASTDataForward.h>

#include <libsolidity/interface/Exceptions.h>

#include <boost/variant.hpp>
#include <boost/optional.hpp>

#include <vector>
#include <set>
#include <map>

namespace dev
{
namespace julia
{

/**
 * Generic AST walker.
 */
class ASTWalker: public boost::static_visitor<>
{
public:
	virtual void operator()(Literal const&) {}
	virtual void operator()(Instruction const&) { solAssert(false, ""); }
	virtual void operator()(Identifier const&) {}
	virtual void operator()(FunctionalInstruction const& _instr);
	virtual void operator()(FunctionCall const& _funCall);
	virtual void operator()(Label const&) { solAssert(false, ""); }
	virtual void operator()(StackAssignment const&) { solAssert(false, ""); }
	virtual void operator()(Assignment const& _assignment);
	virtual void operator()(VariableDeclaration const& _varDecl);
	virtual void operator()(If const& _if);
	virtual void operator()(Switch const& _switch);
	virtual void operator()(FunctionDefinition const&);
	virtual void operator()(ForLoop const&);
	virtual void operator()(Block const& _block);

protected:
	template <class T>
	void walkVector(T const& _statements)
	{
		for (auto const& st: _statements)
			boost::apply_visitor(*this, st);
	}
};

/**
 * Generic AST modifier (i.e. non-const version of ASTWalker).
 */
class ASTModifier: public boost::static_visitor<>
{
public:
	virtual void operator()(Literal&) {}
	virtual void operator()(Instruction&) { solAssert(false, ""); }
	virtual void operator()(Identifier&) {}
	virtual void operator()(FunctionalInstruction& _instr);
	virtual void operator()(FunctionCall& _funCall);
	virtual void operator()(Label&) { solAssert(false, ""); }
	virtual void operator()(StackAssignment&) { solAssert(false, ""); }
	virtual void operator()(Assignment& _assignment);
	virtual void operator()(VariableDeclaration& _varDecl);
	virtual void operator()(If& _if);
	virtual void operator()(Switch& _switch);
	virtual void operator()(FunctionDefinition&);
	virtual void operator()(ForLoop&);
	virtual void operator()(Block& _block);

protected:
	template <class T>
	void walkVector(T&& _statements)
	{
		for (auto& st: _statements)
			boost::apply_visitor(*this, st);
	}
};

}
}
