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
// LLVM dialect generator
//

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Solidity/SolidityOps.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>

using namespace mlir;

namespace {

// FIXME: The high level dialects are translated to the llvm dialect tailored to
// the EraVM backend in llvm. How should we perform the translation when we
// support other targets?
//
// (a) If we do a condition translation in this pass, the code can quickly get
// messy
//
// (b) If we have a high level dialect for each target, the lowering will be,
// for instance, solidity.object -> eravm.object -> llvm.func with eravm
// details. Unnecessary abstractions?
//
// (c) I think a sensible design is to create different ModuleOp passes for each
// target that translate high level dialects to the llvm dialect.
//

enum EraVMAddrSpace : unsigned {
  Stack = 0,
  Heap = 1,
  HeapAuxiliary = 2,
  Generic = 3,
  Code = 4,
  Storage = 5,
};

class ObjectOpTranslation : public ConversionPattern {
public:
  explicit ObjectOpTranslation(MLIRContext *ctx)
      : ConversionPattern(solidity::ObjectOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto i256Ty = rewriter.getIntegerType(256);
    auto genericAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), EraVMAddrSpace::Generic);

    std::vector<Type> inTys{genericAddrSpacePtrTy};
    constexpr unsigned argCnt = 2 /* Entry::MANDATORY_ARGUMENTS_COUNT */ +
                                10 /* eravm::EXTRA_ABI_DATA_SIZE */;
    for (unsigned i = 0; i < argCnt - 1; ++i) {
      inTys.push_back(i256Ty);
    }
    FunctionType funcType = rewriter.getFunctionType(inTys, {i256Ty});
    func::FuncOp entryFunc =
        rewriter.create<func::FuncOp>(loc, "__entry", funcType);
    Block *entryBlk = rewriter.createBlock(&entryFunc.getRegion());
    for (auto inTy : inTys) {
      entryBlk->addArgument(inTy, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class YulBlockOpTranslation : public ConversionPattern {
public:
  explicit YulBlockOpTranslation(MLIRContext *ctx)
      : ConversionPattern(solidity::YulBlockOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class ContractOpLowering : public ConversionPattern {
public:
  explicit ContractOpLowering(MLIRContext *ctx)
      : ConversionPattern(solidity::ContractOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto contOp = cast<solidity::ContractOp>(op);
    assert(isa<ModuleOp>(contOp->getParentOp()));
    auto modOp = cast<ModuleOp>(contOp->getParentOp());
    Block *modBody = modOp.getBody();

    // Move functions to the parent ModuleOp
    std::vector<Operation *> funcs;
    for (Operation &func : contOp.getBody()->getOperations()) {
      assert(isa<func::FuncOp>(&func));
      funcs.push_back(&func);
    }
    for (Operation *func : funcs) {
      func->moveAfter(modBody, modBody->begin());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerToLLVMPass
    : public PassWrapper<LowerToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLLVMPass)

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // We only lower till the llvm dialect
    LLVMConversionTarget llConv(getContext());
    llConv.addLegalOp<ModuleOp>();
    LLVMTypeConverter llTyConv(&getContext());

    // Lower arith, memref and func dialects to the llvm dialect
    RewritePatternSet pats(&getContext());
    arith::populateArithmeticToLLVMConversionPatterns(llTyConv, pats);
    populateMemRefToLLVMConversionPatterns(llTyConv, pats);
    populateFuncToLLVMConversionPatterns(llTyConv, pats);
    pats.add<ContractOpLowering>(&getContext());
    pats.add<ObjectOpTranslation>(&getContext());
    pats.add<YulBlockOpTranslation>(&getContext());

    ModuleOp mod = getOperation();
    if (failed(applyFullConversion(mod, llConv, std::move(pats))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> solidity::createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}
