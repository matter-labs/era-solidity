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
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Solidity/SolidityOps.h"
#include "libyul/AST.h"
#include "libyul/optimiser/ASTWalker.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace solidity::langutil;
using namespace solidity::yul;

namespace solidity::frontend {

class YulToMLIRPass : public ASTWalker {
  mlir::OpBuilder b;
  mlir::ModuleOp mod;
  CharStream const &stream;
  Dialect const &yulDialect;

public:
  mlir::ModuleOp getModule() { return mod; }

  explicit YulToMLIRPass(mlir::MLIRContext &ctx, CharStream const &stream,
                         Dialect const &yulDialect)
      : b(&ctx), stream(stream), yulDialect(yulDialect) {
    mod = mlir::ModuleOp::create(b.getUnknownLoc());
    b.setInsertionPointToEnd(mod.getBody());
  }

  /// Lowers a subobject
  void lowerObj(Object const &obj);

  /// Lowers a top level object
  void lowerTopLevelObj(Object const &obj);

private:
  /// Returns the mlir location for the solidity source location `loc`
  mlir::Location loc(SourceLocation loc) {
    // FIXME: Track loc.end as well
    LineColumn lineCol = stream.translatePositionToLineColumn(loc.start);
    return mlir::FileLineColLoc::get(b.getStringAttr(stream.name()),
                                     lineCol.line, lineCol.column);
  }

  /// Returns the mlir expression for the literal `lit`
  mlir::Value genExpr(Literal const &lit);

  /// Returns the mlir expression for the identifier `id`
  mlir::Value genExpr(Identifier const &id);

  /// Returns the mlir expression for the function call `call`
  mlir::Value genExpr(FunctionCall const &call);

  /// Returns the mlir expression for the expression `expr`
  mlir::Value genExpr(Expression const &expr);

  /// Lowers an expression statement
  void operator()(ExpressionStatement const &expr);

  /// Lowers a block
  void operator()(Block const &blk);
};

mlir::Value YulToMLIRPass::genExpr(Literal const &lit) {
  mlir::Location lc = this->loc(lit.debugData->nativeLocation);

  // Do we need to represent constants as u256? Can we do that in
  // arith::ConstantOp?
  auto i256Ty = b.getIntegerType(256);
  return b.create<mlir::arith::ConstantOp>(
      lc, b.getIntegerAttr(i256Ty, llvm::APInt(256, lit.value.str(),
                                               /*radix=*/10)));
}

mlir::Value YulToMLIRPass::genExpr(Identifier const &id) {
  solUnimplementedAssert(false, "TODO: Lower identifier");
}

mlir::Value YulToMLIRPass::genExpr(FunctionCall const &call) {
  BuiltinFunction const *builtin = yulDialect.builtin(call.functionName.name);
  if (builtin) {

    // TODO: The lowering of builtin function should be auto generated from
    // evmasm::InstructionInfo and the corresponding mlir ops
    if (builtin->name.str() == "return") {
      b.create<mlir::solidity::ReturnOp>(loc(call.debugData->nativeLocation),
                                         genExpr(call.arguments[0]),
                                         genExpr(call.arguments[1]));
      return {};

    } else if (builtin->name.str() == "mstore") {
      b.create<mlir::solidity::MstoreOp>(loc(call.debugData->nativeLocation),
                                         genExpr(call.arguments[0]),
                                         genExpr(call.arguments[1]));
      return {};
    } else {
      solUnimplementedAssert(false, "TODO: Lower other builtin function call");
    }
  } else {
    solUnimplementedAssert(false, "TODO: Lower non builtin function call");
  }

  solAssert(false);
}

mlir::Value YulToMLIRPass::genExpr(Expression const &expr) {
  return std::visit(
      [&](auto &&resolvedExpr) { return this->genExpr(resolvedExpr); }, expr);
}

void YulToMLIRPass::operator()(ExpressionStatement const &expr) {
  genExpr(expr.expression);
}

void YulToMLIRPass::operator()(Block const &blk) { ASTWalker::operator()(blk); }

void YulToMLIRPass::lowerObj(Object const &obj) {
  // TODO: Where is the source location info for Object? Do we need to track it?
  auto objOp =
      b.create<mlir::solidity::ObjectOp>(b.getUnknownLoc(), obj.name.str());
  b.setInsertionPointToEnd(objOp.getBody());
  // TODO? Do we need a separate op for the `code` block?
  operator()(*obj.code);
}

void YulToMLIRPass::lowerTopLevelObj(Object const &obj) {
  lowerObj(obj);

  // TODO: Does it make sense to nest subobjects in the top level ObjectOp's
  // body?
  for (auto const &subNode : obj.subObjects) {
    if (auto *subObj = dynamic_cast<Object const *>(subNode.get())) {
      lowerObj(*subObj);
    } else {
      solUnimplementedAssert(false, "TODO: Metadata translation");
    }
  }
}

} // namespace solidity::frontend

bool solidity::frontend::runYulToMLIRPass(Object const &obj,
                                          CharStream const &stream,
                                          Dialect const &yulDialect) {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::solidity::SolidityDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  YulToMLIRPass yulToMLIR(ctx, stream, yulDialect);
  yulToMLIR.lowerTopLevelObj(obj);

  if (failed(mlir::verify(yulToMLIR.getModule()))) {
    yulToMLIR.getModule().print(llvm::errs());
    yulToMLIR.getModule().emitError("Module verification error");
    return false;
  }

  // mlir::PassManager passMgr(&ctx);
  // passMgr.addPass(mlir::solidity::createSolidityDialectLoweringPassForEraVM());
  // if (mlir::failed(passMgr.run(gen.getModule())))
  //   ; // return false;

  yulToMLIR.getModule().print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs().flush();

  return true;
}
