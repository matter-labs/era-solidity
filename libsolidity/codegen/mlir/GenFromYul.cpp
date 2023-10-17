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

#include "libsolidity/codegen/mlir/Solidity/SolidityOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
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
	mlir::OpBuilder b;
	mlir::ModuleOp mod;

public:
	mlir::ModuleOp getModule() { return mod; }

	explicit MLIRGenFromYul(mlir::MLIRContext& _ctx): b(&_ctx)
	{
		mod = mlir::ModuleOp::create(b.getUnknownLoc());
		b.setInsertionPointToEnd(mod.getBody());
	}

	void operator()(Block const& _blk)
	{
		// TODO: Add real source location
		auto op = b.create<mlir::solidity::YulBlockOp>(b.getUnknownLoc());
		solUnimplementedAssert(_blk.statements.empty(), "TODO: Lower non-empty yul blocks");
		return;
	}

private:
};

}

bool solidity::frontend::runMLIRGenFromYul(yul::Block const& _blk)
{
	mlir::MLIRContext ctx;
	ctx.getOrLoadDialect<mlir::solidity::SolidityDialect>();
	MLIRGenFromYul gen(ctx);
	gen(_blk);

	if (failed(mlir::verify(gen.getModule())))
	{
		gen.getModule().emitError("Module verification error");
		return false;
	}

	gen.getModule().print(llvm::outs());
	llvm::outs() << "\n";
	llvm::outs().flush();

	return true;
}
