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

namespace solidity::frontend
{

class ASTNode;
class Declaration;
class Expression;
class SourceUnit;
class VariableDeclaration;

/// Find the topmost referenced constant variable declaration when the given variable
/// declaration value is an identifier. Works only for constant variable declarations.
/// Returns nullptr if an identifier in the chain is not referencing a constant variable declaration.
VariableDeclaration const* rootConstVariableDeclaration(VariableDeclaration const& _varDecl);

/// Returns true if the constant variable declaration is recursive.
bool isConstantVariableRecursive(VariableDeclaration const& _varDecl);

/// Returns the innermost AST node that covers the given location or nullptr if not found.
ASTNode const* locateInnermostASTNode(int _offsetInFile, SourceUnit const& _sourceUnit);

/// @returns @a _expr itself, in case it is not a unary tuple expression. Otherwise it descends recursively
/// into unary tuples and returns the contained expression.
Expression const* resolveOuterUnaryTuples(Expression const* _expr);

/// @returns the type of an expression and asserts that it is present.
Type const* type(Expression const& _expression);
/// @returns the type of the given variable and throws if the type is not present
/// (this can happen for variables with non-explicit types before their types are resolved)
Type const* type(VariableDeclaration const& _variable);

}
