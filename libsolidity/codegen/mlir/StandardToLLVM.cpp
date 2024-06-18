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

//
// A pass to convert standard dialects to llvm dialect.
//

#include "libsolidity/codegen/mlir/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {

struct ConvertStandardToLLVM
    : public PassWrapper<ConvertStandardToLLVM, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertStandardToLLVM)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    LLVMConversionTarget tgt(getContext());
    tgt.addLegalOp<ModuleOp>();

    LLVMTypeConverter tyConv(&getContext());

    RewritePatternSet pats(&getContext());
    populateFuncToLLVMConversionPatterns(tyConv, pats);
    populateSCFToControlFlowConversionPatterns(pats);
    cf::populateControlFlowToLLVMConversionPatterns(tyConv, pats);
    mlir::arith::populateArithToLLVMConversionPatterns(tyConv, pats);

    if (failed(applyFullConversion(getOperation(), tgt, std::move(pats))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> sol::createConvertStandardToLLVMPass() {
  return std::make_unique<ConvertStandardToLLVM>();
}
