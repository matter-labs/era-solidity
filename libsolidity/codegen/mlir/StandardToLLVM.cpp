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
// A pass to convert standard dialects to llvm dialect
//

#include "libsolidity/codegen/mlir/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"

using namespace mlir;

namespace {

struct ConvertStandardToLLVM
    : public PassWrapper<ConvertStandardToLLVM, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertStandardToLLVM)

  /// LLVM target triple.
  StringRef triple;

  /// Bitwidth of index type.
  unsigned indexBitwidth;

  /// LLVM target datalayout.
  StringRef dataLayout;

  explicit ConvertStandardToLLVM(StringRef triple, unsigned indexBitwidth,
                                 StringRef dataLayout)
      : triple(triple), indexBitwidth(indexBitwidth), dataLayout(dataLayout) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    LLVMConversionTarget tgt(getContext());
    tgt.addLegalOp<ModuleOp>();

    // Create the llvm type converter.
    LowerToLLVMOptions opts(&getContext());
    opts.overrideIndexBitwidth(indexBitwidth);
    opts.dataLayout = llvm::DataLayout(dataLayout);
    LLVMTypeConverter tyConv(&getContext(), opts);

    // Set the llvm target triple and data-layout.
    ModuleOp mod = getOperation();
    mod->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                 StringAttr::get(&getContext(), triple));
    mod->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                 StringAttr::get(&getContext(), dataLayout));

    // Populate patterns.
    RewritePatternSet pats(&getContext());
    populateFuncToLLVMConversionPatterns(tyConv, pats);
    populateSCFToControlFlowConversionPatterns(pats);
    cf::populateControlFlowToLLVMConversionPatterns(tyConv, pats);
    mlir::arith::populateArithToLLVMConversionPatterns(tyConv, pats);

    // Run the conversion.
    if (failed(applyFullConversion(mod, tgt, std::move(pats))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
sol::createConvertStandardToLLVMPass(StringRef triple, unsigned indexBitwidth,
                                     StringRef dataLayout) {
  return std::make_unique<ConvertStandardToLLVM>(triple, indexBitwidth,
                                                 dataLayout);
}
