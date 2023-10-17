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

#include <libsolidity/codegen/mlir/GenFromYul.h>

#include <libyul/AST.h>
#include <libyul/optimiser/ASTWalker.h>

#include <liblangutil/Exceptions.h>

#include <iostream>

using namespace solidity::yul;

namespace solidity::frontend
{

class MLIRGenFromYul: public ASTWalker
{
public:
	void operator()(Block const& _blk)
	{
		solUnimplementedAssert(false, "TODO: Lower yul blocks");
		return;
	}

private:
};

}

void solidity::frontend::runMLIRGenFromYul(yul::Block const& _blk)
{
	MLIRGenFromYul gen;
	gen(_blk);
}
