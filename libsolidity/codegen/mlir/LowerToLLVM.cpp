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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ValueRange.h"
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

namespace eravm {
enum AddrSpace : unsigned {
  Stack = 0,
  Heap = 1,
  HeapAuxiliary = 2,
  Generic = 3,
  Code = 4,
  Storage = 5,
};

enum ByteLen { Byte = 1, X32 = 4, X64 = 8, EthAddr = 20, Field = 32 };
enum : unsigned { HeapAuxOffsetCtorRetData = ByteLen::Field * 8 };
} // namespace eravm

class BuilderHelper {
  OpBuilder b;
  Location loc;

public:
  explicit BuilderHelper(OpBuilder &b, Location loc) : b(b), loc(loc) {}

  Value getConst(int64_t val, unsigned width = 256) {
    IntegerType ty = b.getIntegerType(width);
    auto op = b.create<arith::ConstantOp>(
        loc, b.getIntegerAttr(ty, llvm::APInt(width, val, /*radix=*/10)));
    return op.getResult();
  }
};

static SymbolRefAttr getOrInsertReturn(PatternRewriter &rewriter,
                                       ModuleOp mod) {
  auto *ctx = mod.getContext();
  if (mod.lookupSymbol<LLVM::LLVMFuncOp>("__return"))
    return SymbolRefAttr::get(ctx, "__return");

  auto i256Ty = IntegerType::get(ctx, 256);
  auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                            {i256Ty, i256Ty, i256Ty});

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(mod.getLoc(), "__return", fnType);
  return SymbolRefAttr::get(ctx, "__return");
}

class ReturnOpLowering : public ConversionPattern {
public:
  explicit ReturnOpLowering(MLIRContext *ctx)
      : ConversionPattern(solidity::ReturnOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    BuilderHelper b(rewriter, loc);

    auto heapAuxAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), eravm::AddrSpace::HeapAuxiliary);

    auto immutablesOffsetPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        b.getConst(eravm::HeapAuxOffsetCtorRetData));
    rewriter.create<LLVM::StoreOp>(loc, b.getConst(eravm::ByteLen::Field),
                                   immutablesOffsetPtr);

    auto immutablesSize = 0; // TODO: Implement this!
    auto immutablesNumPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        b.getConst(eravm::HeapAuxOffsetCtorRetData + eravm::ByteLen::Field));
    rewriter.create<LLVM::StoreOp>(
        loc, b.getConst(immutablesSize / eravm::ByteLen::Field),
        immutablesNumPtr);

    auto immutablesCalcSize = rewriter.create<arith::MulIOp>(
        loc, b.getConst(immutablesSize), b.getConst(2));
    auto returnDataLen =
        rewriter.create<arith::AddIOp>(loc, immutablesCalcSize.getResult(),
                                       b.getConst(eravm::ByteLen::Field * 2));
    auto returnFunc =
        getOrInsertReturn(rewriter, op->getParentOfType<ModuleOp>());
    bool isCreation = true; // TODO: Implement this!
    auto returnOpMode = b.getConst(isCreation ? eravm::AddrSpace::HeapAuxiliary
                                              : eravm::AddrSpace::Heap);
    rewriter.create<func::CallOp>(
        loc, returnFunc, TypeRange{},
        ValueRange{b.getConst(eravm::HeapAuxOffsetCtorRetData),
                   returnDataLen.getResult(), returnOpMode});

    rewriter.create<LLVM::UnreachableOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
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
        rewriter.getContext(), eravm::AddrSpace::Generic);

    std::vector<Type> inTys{genericAddrSpacePtrTy};
    constexpr unsigned argCnt = 2 /* Entry::MANDATORY_ARGUMENTS_COUNT */ +
                                10 /* eravm::EXTRA_ABI_DATA_SIZE */;
    for (unsigned i = 0; i < argCnt - 1; ++i) {
      inTys.push_back(i256Ty);
    }
    FunctionType funcType = rewriter.getFunctionType(inTys, {i256Ty});
    auto mod = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToEnd(mod.getBody());
    func::FuncOp entryFunc =
        rewriter.create<func::FuncOp>(loc, "__entry", funcType);
    assert(op->getNumRegions() == 1);

    auto &entryFuncRegion = entryFunc.getRegion();
    rewriter.inlineRegionBefore(op->getRegion(0), entryFuncRegion,
                                entryFuncRegion.begin());
    Block *entryBlk = &entryFuncRegion.getBlocks().front();
    for (auto inTy : inTys) {
      entryBlk->addArgument(inTy, loc);
    }

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
    pats.add<ReturnOpLowering>(&getContext());

    ModuleOp mod = getOperation();
    if (failed(applyFullConversion(mod, llConv, std::move(pats))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> solidity::createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}
