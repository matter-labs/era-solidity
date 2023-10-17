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

using namespace solidity::langutil;
using namespace solidity::yul;

namespace solidity::frontend
{

class MLIRGenFromYul: public ASTWalker
{
	mlir::OpBuilder b;
	mlir::ModuleOp mod;
	CharStream const& m_stream;

public:
	mlir::ModuleOp getModule() { return mod; }

	explicit MLIRGenFromYul(mlir::MLIRContext& _ctx, CharStream const& _stream): b(&_ctx), m_stream(_stream)
	{
		mod = mlir::ModuleOp::create(b.getUnknownLoc());
		b.setInsertionPointToEnd(mod.getBody());
	}

	/// Returns the mlir location for the solidity source location `_loc`
	mlir::Location loc(SourceLocation _loc)
	{
		// FIXME: Track _loc.end as well
		LineColumn lineCol = m_stream.translatePositionToLineColumn(_loc.start);
		return mlir::FileLineColLoc::get(b.getStringAttr(m_stream.name()), lineCol.line, lineCol.column);
	}

	void operator()(Block const& _blk)
	{
		// TODO: Add real source location
		auto op = b.create<mlir::solidity::YulBlockOp>(loc(_blk.debugData->nativeLocation));
		solUnimplementedAssert(_blk.statements.empty(), "TODO: Lower non-empty yul blocks");
		return;
	}

private:
};

}

bool solidity::frontend::runMLIRGenFromYul(yul::Block const& _blk, CharStream const& _stream)
{
	mlir::MLIRContext ctx;
	ctx.getOrLoadDialect<mlir::solidity::SolidityDialect>();
	MLIRGenFromYul gen(ctx, _stream);
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
