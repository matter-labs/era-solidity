// This file is part of solidity.

// solidity is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// solidity is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with solidity.  If not, see <http://www.gnu.org/licenses/>.

// SPDX-License-Identifier: GPL-3.0

#include "libsolidity/codegen/mlir/GenFromYul.h"
#include "liblangutil/Exceptions.h"
#include "libsolidity/codegen/mlir/Solidity/SolidityOps.h"
#include "libyul/AST.h"
#include "libyul/optimiser/ASTWalker.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace solidity::langutil;
using namespace solidity::yul;

namespace solidity::frontend {

class MLIRGenFromYul : public ASTWalker {
  mlir::OpBuilder b;
  mlir::ModuleOp mod;
  CharStream const &stream;
  Dialect const &yulDialect;

public:
  mlir::ModuleOp getModule() { return mod; }

  explicit MLIRGenFromYul(mlir::MLIRContext &ctx, CharStream const &stream,
                          Dialect const &yulDialect)
      : b(&ctx), stream(stream), yulDialect(yulDialect) {
    mod = mlir::ModuleOp::create(b.getUnknownLoc());
    b.setInsertionPointToEnd(mod.getBody());
  }

  /// Returns the mlir location for the solidity source location `loc`
  mlir::Location loc(SourceLocation loc) {
    // FIXME: Track loc.end as well
    LineColumn lineCol = stream.translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream.name()),
                                     lineCol.line, lineCol.column);
  }

  /// Returns the mlir expression for the function call `call`
  mlir::Value genExpr(FunctionCall const &call) {
    BuiltinFunction const *builtin = yulDialect.builtin(call.functionName.name);
    if (builtin) {
      solUnimplementedAssert(builtin->name.str() == "return",
                             "TODO: Lower other builtins");
      b.create<mlir::solidity::ReturnOp>(loc(call.debugData->nativeLocation),
                                         genExpr(call.arguments[0]),
                                         genExpr(call.arguments[1]));
      return {};

    } else {
      solUnimplementedAssert(false, "TODO: Lower non builtin function call");
    }

    solAssert(false);
  }

  /// Returns the mlir expression for the identifier `id`
  mlir::Value genExpr(Identifier const &id) {
    solUnimplementedAssert(false, "TODO: Lower identifier");
  }

  /// Returns the mlir expression for the literal `lit`
  mlir::Value genExpr(Literal const &lit) {
    mlir::Location lc = loc(lit.debugData->nativeLocation);

    // Do we need to represent constants as u256? Can we do that in
    // arith::ConstantOp?
    auto i256Ty = b.getIntegerType(256);
    return b.create<mlir::arith::ConstantOp>(
        lc, b.getIntegerAttr(i256Ty, llvm::APInt(256, lit.value.str(),
                                                 /*radix=*/10)));
  }

  /// Returns the mlir expression for the expression `expr`
  mlir::Value genExpr(Expression const &expr) {
    return std::visit(
        [&](auto &&resolvedExpr) { return this->genExpr(resolvedExpr); }, expr);
  }

  /// Lowers an expression statement
  void operator()(ExpressionStatement const &expr) { genExpr(expr.expression); }

  void operator()(Block const &blk) {
    // TODO: Add real source location
    auto op = b.create<mlir::solidity::YulBlockOp>(
        loc(blk.debugData->nativeLocation));

    b.setInsertionPointToEnd(op.getBody());
    ASTWalker::operator()(blk);
    return;
  }

private:
};

} // namespace solidity::frontend

bool solidity::frontend::runMLIRGenFromYul(Block const &blk,
                                           CharStream const &stream,
                                           Dialect const &yulDialect) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::solidity::SolidityDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  MLIRGenFromYul gen(ctx, stream, yulDialect);
  gen(blk);

  if (failed(mlir::verify(gen.getModule()))) {
    gen.getModule().print(llvm::errs());
    gen.getModule().emitError("Module verification error");
    return false;
  }

  gen.getModule().print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs().flush();

  return true;
}
