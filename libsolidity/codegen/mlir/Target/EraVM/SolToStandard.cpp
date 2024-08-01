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
// Sol dialect lowering pass for EraVM.
//

#include "libsolidity/codegen/CompilerUtils.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolidity/codegen/mlir/Target/EraVM/Util.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolutil/ErrorCodes.h"
#include "libsolutil/FixedHash.h"
#include "libsolutil/FunctionSelector.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IntrinsicsEraVM.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <climits>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

// FIXME: The high level dialects are lowered to the llvm dialect tailored to
// the EraVM backend in llvm. How should we perform the lowering when we support
// other targets?
//
// (a) If we do a conditional lowering in this pass, the code can quickly get
// messy
//
// (b) If we have a high level dialect for each target, the lowering will be,
// for instance, solidity.object -> eravm.object -> llvm.func with eravm
// details. Unnecessary abstractions?
//
// (c) I think a sensible design is to create different ModuleOp passes for each
// target that lower high level dialects to the llvm dialect.
//

// FIXME: getGlobalOp() should be used iff we can guarantee the presence of the
// global op. Only ContractOpLowering is following this.

// FIXME: How can we break up lengthy op conversion? MallocOpLowering's helper
// function needs to always pass around invariants like the rewriter, location
// etc. It would be nice if they could be member variables. Do we need to create
// helper classes for such lowering?

// TODO: Document differences in the memory and storage layout like:
// - 32 byte alignment for all types in storage (excluding the data of
// string/bytes).
//
// - The simpler string layout in storage.

/// Returns true if `op` is defined in a runtime context
static bool inRuntimeContext(Operation *op) {
  assert(!isa<sol::FuncOp>(op) && !isa<sol::ObjectOp>(op));

  // Check if the parent FuncOp has isRuntime attribute set
  auto parentFunc = op->getParentOfType<sol::FuncOp>();
  if (parentFunc)
    return parentFunc.getRuntime();

  // If there's no parent FuncOp, check the parent ObjectOp
  auto parentObj = op->getParentOfType<sol::ObjectOp>();
  if (parentObj) {
    return parentObj.getSymName().endswith("_deployed");
  }

  llvm_unreachable("op has no parent FuncOp or ObjectOp");
}

/// A generic conversion pattern that replaces the operands with the legalized
/// ones and legalizes the return types.
template <typename OpT>
struct GenericTypeConversion : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    SmallVector<Type> retTys;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTys)))
      return failure();

    // TODO: Use updateRootInPlace instead?
    r.replaceOpWithNewOp<OpT>(op, retTys, adaptor.getOperands(),
                              op->getAttrs());
    return success();
  }
};

struct ConvCastOpLowering : public OpConversionPattern<sol::ConvCastOp> {
  using OpConversionPattern<sol::ConvCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ConvCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

// TODO? Move simple builtin lowering to tblgen (`Pat` records)?

struct Keccak256OpLowering : public OpRewritePattern<sol::Keccak256Op> {
  using OpRewritePattern<sol::Keccak256Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::Keccak256Op op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r);
    eravm::Builder eraB(r, loc);

    // Setup arguments for the call to sha3.
    Value offset = eraB.genHeapPtr(op.getInp0());
    Value length = op.getInp1();

    // FIXME: When will this
    // (`context.get_function(EraVMFunction::ZKSYNC_NEAR_CALL_ABI_EXCEPTION_HANDLER).is_some()`)
    // be true?
    auto throwAtFail = bExt.genBool(false);

    // Generate the call to sha3.
    auto i256Ty = r.getIntegerType(256);
    FlatSymbolRefAttr sha3Func =
        eraB.getOrInsertSha3(op->getParentOfType<ModuleOp>());
    r.replaceOpWithNewOp<sol::CallOp>(op, sha3Func, TypeRange{i256Ty},
                                      ValueRange{offset, length, throwAtFail});
    return success();
  }
};

struct LogOpLowering : public OpRewritePattern<sol::LogOp> {
  using OpRewritePattern<sol::LogOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::LogOp op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    // Collect arguments for the far-call.
    std::vector<Value> farCallArgs;
    farCallArgs.push_back(eraB.genABIData(op.getAddr(), op.getSize(),
                                          eravm::AddrSpace_Heap,
                                          /*isSysCall=*/true));
    farCallArgs.push_back(bExt.genI256Const(eravm::Address_EventWriter));
    farCallArgs.push_back(bExt.genI256Const(op.getTopics().size()));
    for (Value topic : op.getTopics()) {
      farCallArgs.push_back(topic);
    }

    sol::FuncOp farCallOp =
        eraB.getOrInsertFarCall(op->getParentOfType<ModuleOp>());

    // Pad remaining args with undefs.
    size_t farCallArgCnt = farCallOp.getFunctionType().getInputs().size();
    assert(farCallArgCnt > farCallArgs.size());
    size_t undefArgCnt = farCallArgCnt - farCallArgs.size();
    for (size_t i = 0; i < undefArgCnt; ++i) {
      farCallArgs.push_back(
          r.create<LLVM::UndefOp>(loc, r.getIntegerType(256)));
    }

    // Generate the call to the far-call function.
    auto farCall = r.create<sol::CallOp>(loc, farCallOp, farCallArgs);

    // Generate the status check and revert.
    auto farCallFailCond =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                farCall.getResult(1), bExt.genBool(false));
    r.create<scf::IfOp>(loc, farCallFailCond,
                        /*thenBuilder=*/[&](OpBuilder &b, Location loc) {
                          b.create<sol::RevertOp>(loc, bExt.genI256Const(0),
                                                  bExt.genI256Const(0));
                          b.create<scf::YieldOp>(loc);
                        });

    r.eraseOp(op);
    return success();
  }
};

struct CallerOpLowering : public OpRewritePattern<sol::CallerOp> {
  using OpRewritePattern<sol::CallerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallerOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, /*resTy=*/r.getIntegerType(256),
        r.getI32IntegerAttr(llvm::Intrinsic::eravm_caller),
        r.getStringAttr("eravm.caller"));

    return success();
  }
};

struct CallValOpLowering : public OpRewritePattern<sol::CallValOp> {
  using OpRewritePattern<sol::CallValOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallValOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, /*resTy=*/rewriter.getIntegerType(256),
        rewriter.getI32IntegerAttr(llvm::Intrinsic::eravm_getu128),
        rewriter.getStringAttr("eravm.getu128"));
    return success();
  }
};

struct CallDataLoadOpLowering : public OpRewritePattern<sol::CallDataLoadOp> {
  using OpRewritePattern<sol::CallDataLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallDataLoadOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(rewriter);
    eravm::Builder eraB(rewriter, loc);

    Value offset = op.getInp();

    if (inRuntimeContext(op)) {
      // Generate the `GlobCallDataPtr` + `offset` load
      LLVM::LoadOp callDataPtr =
          eraB.genCallDataPtrLoad(op->getParentOfType<ModuleOp>());
      unsigned callDataPtrAddrSpace =
          cast<LLVM::LLVMPointerType>(callDataPtr.getType()).getAddressSpace();
      auto callDataOffset = rewriter.create<LLVM::GEPOp>(
          loc,
          /*resultType=*/
          LLVM::LLVMPointerType::get(rewriter.getContext(),
                                     callDataPtrAddrSpace),
          /*basePtrType=*/rewriter.getIntegerType(eravm::BitLen_Byte),
          callDataPtr, ValueRange{offset});
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
          op, op.getType(), callDataOffset,
          eravm::getAlignment(callDataOffset));

    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, /*value=*/0,
                                                        /*width=*/256);
    }

    return success();
  }
};

struct CallDataSizeOpLowering : public OpRewritePattern<sol::CallDataSizeOp> {
  using OpRewritePattern<sol::CallDataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallDataSizeOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    if (inRuntimeContext(op)) {
      eravm::Builder eraB(rewriter, loc);
      rewriter.replaceOp(op, eraB.genCallDataSizeLoad(mod));
    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, /*value=*/0,
                                                        /*width=*/256);
    }

    return success();
  }
};

struct CallDataCopyOpLowering : public OpRewritePattern<sol::CallDataCopyOp> {
  using OpRewritePattern<sol::CallDataCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallDataCopyOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    solidity::mlirgen::BuilderExt bExt(rewriter);
    eravm::Builder eraB(rewriter, loc);

    mlir::Value srcOffset;
    if (inRuntimeContext(op)) {
      srcOffset = op.getInp1();
    } else {
      eravm::Builder eraB(rewriter, loc);
      srcOffset = eraB.genCallDataSizeLoad(mod);
    }

    mlir::Value dst = eraB.genHeapPtr(op.getInp0());
    mlir::Value size = op.getInp2();

    // Generate the source pointer.
    LLVM::LoadOp callDataPtr =
        eraB.genCallDataPtrLoad(op->getParentOfType<ModuleOp>());
    unsigned callDataPtrAddrSpace =
        cast<LLVM::LLVMPointerType>(callDataPtr.getType()).getAddressSpace();
    auto src = rewriter.create<LLVM::GEPOp>(
        loc,
        /*resultType=*/
        LLVM::LLVMPointerType::get(mod.getContext(), callDataPtrAddrSpace),
        /*basePtrType=*/rewriter.getIntegerType(eravm::BitLen_Byte),
        callDataPtr, ValueRange{srcOffset});

    // Generate the memcpy.
    rewriter.create<LLVM::MemcpyOp>(loc, dst, src, size, /*isVolatile=*/false);

    rewriter.eraseOp(op);
    return success();
  }
};

struct SLoadOpLowering : public OpRewritePattern<sol::SLoadOp> {
  using OpRewritePattern<sol::SLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::SLoadOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();

    mlir::Value addr = op.getInp();
    auto storageAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), eravm::AddrSpace_Storage);
    mlir::Value offset =
        rewriter.create<LLVM::IntToPtrOp>(loc, storageAddrSpacePtrTy, addr);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, rewriter.getIntegerType(256), offset, eravm::getAlignment(offset));

    return success();
  }
};

struct SStoreOpLowering : public OpRewritePattern<sol::SStoreOp> {
  using OpRewritePattern<sol::SStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::SStoreOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();

    mlir::Value addr = op.getInp0();
    mlir::Value val = op.getInp1();
    auto storageAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), eravm::AddrSpace_Storage);
    mlir::Value offset =
        rewriter.create<LLVM::IntToPtrOp>(loc, storageAddrSpacePtrTy, addr);
    rewriter.create<LLVM::StoreOp>(loc, val, offset,
                                   eravm::getAlignment(offset));

    rewriter.eraseOp(op);
    return success();
  }
};

struct DataOffsetOpLowering : public OpRewritePattern<sol::DataOffsetOp> {
  using OpRewritePattern<sol::DataOffsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::DataOffsetOp op,
                                PatternRewriter &rewriter) const override {
    // FIXME:
    // - Handle references to objects outside the current module
    // - Check if the reference object is valid
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getIntegerType(256), 0));
    return success();
  }
};

struct DataSizeOpLowering : public OpRewritePattern<sol::DataSizeOp> {
  using OpRewritePattern<sol::DataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::DataSizeOp op,
                                PatternRewriter &rewriter) const override {
    // FIXME:
    // - Handle references to objects outside the current module
    // - Check if the reference object is valid
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getIntegerType(256), 0));
    return success();
  }
};

struct CodeSizeOpLowering : public OpRewritePattern<sol::CodeSizeOp> {
  using OpRewritePattern<sol::CodeSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CodeSizeOp op,
                                PatternRewriter &r) const override {
    mlir::Location loc = op->getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    if (inRuntimeContext(op)) {
      llvm_unreachable("NYI");
    } else {
      eravm::Builder eraB(r, loc);
      r.replaceOp(op, eraB.genCallDataSizeLoad(mod));
    }
    return success();
  }
};

struct CodeCopyOpLowering : public OpRewritePattern<sol::CodeCopyOp> {
  using OpRewritePattern<sol::CodeCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CodeCopyOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    solidity::mlirgen::BuilderExt bExt(rewriter);
    eravm::Builder eraB(rewriter, loc);

    assert(!inRuntimeContext(op) &&
           "codecopy is not supported in runtime context");

    Value dst = eraB.genHeapPtr(op.getInp0());
    Value srcOffset = op.getInp1();
    Value size = op.getInp2();

    // Generate the source pointer
    LLVM::LoadOp callDataPtr =
        eraB.genCallDataPtrLoad(op->getParentOfType<ModuleOp>());
    unsigned callDataPtrAddrSpace =
        cast<LLVM::LLVMPointerType>(callDataPtr.getType()).getAddressSpace();
    auto src = rewriter.create<LLVM::GEPOp>(
        loc,
        /*resultType=*/
        LLVM::LLVMPointerType::get(mod.getContext(), callDataPtrAddrSpace),
        /*basePtrType=*/rewriter.getIntegerType(eravm::BitLen_Byte),
        callDataPtr, ValueRange{srcOffset});

    // Generate the memcpy.
    rewriter.create<LLVM::MemcpyOp>(loc, dst, src, size, /*isVolatile=*/false);

    rewriter.eraseOp(op);
    return success();
  }
};

struct MLoadOpLowering : public OpRewritePattern<sol::MLoadOp> {
  using OpRewritePattern<sol::MLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MLoadOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    eravm::Builder eraB(rewriter, loc);

    mlir::Value addr = eraB.genHeapPtr(op.getInp());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, rewriter.getIntegerType(256),
                                              addr, eravm::getAlignment(addr));
    return success();
  }
};

struct MStoreOpLowering : public OpRewritePattern<sol::MStoreOp> {
  using OpRewritePattern<sol::MStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MStoreOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    eravm::Builder eraB(rewriter, loc);

    mlir::Value offset = eraB.genHeapPtr(op.getInp0());
    mlir::Value val = op.getInp1();
    rewriter.create<LLVM::StoreOp>(loc, val, offset,
                                   eravm::getAlignment(offset));

    rewriter.eraseOp(op);
    return success();
  }
};

struct MCopyOpLowering : public OpRewritePattern<sol::MCopyOp> {
  using OpRewritePattern<sol::MCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MCopyOp op,
                                PatternRewriter &rewriter) const override {
    // TODO? Check m_evmVersion.hasMcopy() and legalize here?

    mlir::Location loc = op->getLoc();
    eravm::Builder eraB(rewriter, loc);

    Value dstAddr = eraB.genHeapPtr(op.getInp0());
    Value srcAddr = eraB.genHeapPtr(op.getInp1());
    Value size = op.getInp2();

    // Generate the memmove.
    // FIXME: Add align 1 param attribute.
    rewriter.create<LLVM::MemmoveOp>(loc, dstAddr, srcAddr, size,
                                     /*isVolatile=*/false);

    rewriter.eraseOp(op);
    return success();
  }
};

struct MemGuardOpLowering : public OpRewritePattern<sol::MemGuardOp> {
  using OpRewritePattern<sol::MemGuardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MemGuardOp op,
                                PatternRewriter &rewriter) const override {
    auto inp = op->getAttrOfType<IntegerAttr>("inp");
    assert(inp);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, inp);
    return success();
  }
};

struct RevertOpLowering : public OpRewritePattern<sol::RevertOp> {
  using OpRewritePattern<sol::RevertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::RevertOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    solidity::mlirgen::BuilderExt bExt(rewriter, loc);
    eravm::Builder eraB(rewriter);

    // Create the revert call (__revert(offset, length, RetForwardPageType)) and
    // the unreachable op
    FlatSymbolRefAttr revertFunc = eraB.getOrInsertRevert(mod);
    rewriter.create<sol::CallOp>(
        loc, revertFunc, TypeRange{},
        ValueRange{
            op.getInp0(), op.getInp1(),
            bExt.genI256Const(inRuntimeContext(op)
                                  ? eravm::RetForwardPageType::UseHeap
                                  : eravm::RetForwardPageType::UseAuxHeap)});
    bExt.createCallToUnreachableWrapper(mod);

    rewriter.eraseOp(op);
    return success();
  }
};

struct BuiltinRetOpLowering : public OpRewritePattern<sol::BuiltinRetOp> {
  using OpRewritePattern<sol::BuiltinRetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::BuiltinRetOp op,
                                PatternRewriter &r) const override {
    mlir::Location loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r);
    SymbolRefAttr returnFunc =
        eraB.getOrInsertReturn(op->getParentOfType<ModuleOp>());

    //
    // Lowering in the runtime context.
    //
    if (inRuntimeContext(op)) {
      // Create the return call (__return(offset, length,
      // RetForwardPageType::UseHeap)) and the unreachable op.
      r.create<sol::CallOp>(
          loc, returnFunc, TypeRange{},
          ValueRange{op.getInp0(), op.getInp1(),
                     bExt.genI256Const(eravm::RetForwardPageType::UseHeap)});
      bExt.createCallToUnreachableWrapper(mod);

      r.eraseOp(op);
      return success();
    }

    //
    // Lowering in the creation context.
    //
    auto heapAuxAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        r.getContext(), eravm::AddrSpace_HeapAuxiliary);

    // Generate the store of ByteLen_Field to the immutables offset.
    auto immutablesOffsetPtr = r.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        bExt.genI256Const(eravm::HeapAuxOffsetCtorRetData));
    r.create<LLVM::StoreOp>(loc, bExt.genI256Const(eravm::ByteLen_Field),
                            immutablesOffsetPtr,
                            eravm::getAlignment(immutablesOffsetPtr));

    // Generate the store of the size of immutables in terms of ByteLen_Field to
    // the immutables number offset.
    auto immutablesSize = 0; // TODO: Implement this!
    auto immutablesNumPtr = r.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        bExt.genI256Const(eravm::HeapAuxOffsetCtorRetData +
                          eravm::ByteLen_Field));
    r.create<LLVM::StoreOp>(
        loc, bExt.genI256Const(immutablesSize / eravm::ByteLen_Field),
        immutablesNumPtr, eravm::getAlignment(immutablesNumPtr));

    // Generate the return data length (immutablesSize * 2 + ByteLen_Field * 2).
    auto immutablesCalcSize = r.create<arith::MulIOp>(
        loc, bExt.genI256Const(immutablesSize), bExt.genI256Const(2));
    auto returnDataLen =
        r.create<arith::AddIOp>(loc, immutablesCalcSize.getResult(),
                                bExt.genI256Const(eravm::ByteLen_Field * 2));

    // Create the return call (__return(HeapAuxOffsetCtorRetData, returnDataLen,
    // RetForwardPageType::UseAuxHeap)) and the unreachable op.
    r.create<sol::CallOp>(
        loc, returnFunc, TypeRange{},
        ValueRange{bExt.genI256Const(eravm::HeapAuxOffsetCtorRetData),
                   returnDataLen.getResult(),
                   bExt.genI256Const(eravm::RetForwardPageType::UseAuxHeap)});
    bExt.createCallToUnreachableWrapper(mod);

    r.eraseOp(op);
    return success();
  }
};

//
// TODO: Move evm specific code to solidity::mlirgen
//
using AllocSize = int64_t;

struct AllocaOpLowering : public OpConversionPattern<sol::AllocaOp> {
  using OpConversionPattern<sol::AllocaOp>::OpConversionPattern;

  /// Returns the total size (in bytes) of type (recursively).
  template <AllocSize ValSize>
  AllocSize getTotalSize(Type ty) const {
    // Array type.
    if (auto arrayTy = dyn_cast<sol::ArrayType>(ty)) {
      return arrayTy.getSize() * getTotalSize<ValSize>(arrayTy.getEltType());
    }
    // Struct type.
    if (auto structTy = dyn_cast<sol::StructType>(ty)) {
      assert(false && "NYI: Struct type");
    }

    // Value type.
    return ValSize;
  }

  LogicalResult matchAndRewrite(sol::AllocaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);

    Type convertedEltTy = getTypeConverter()->convertType(op.getAllocType());
    AllocSize size = getTotalSize<1>(op.getAllocType());
    r.replaceOpWithNewOp<LLVM::AllocaOp>(op, convertedEltTy,
                                         bExt.genI256Const(size));
    return success();
  }
};

struct MallocOpLowering : public OpRewritePattern<sol::MallocOp> {
  using OpRewritePattern<sol::MallocOp>::OpRewritePattern;

  /// Returns the size (in bytes) of static type without recursively calculating
  /// the element type size.
  AllocSize getSize(Type ty) const {
    // String type is dynamic.
    assert(!isa<sol::StringType>(ty));
    // Array type.
    if (auto arrayTy = dyn_cast<sol::ArrayType>(ty)) {
      assert(!arrayTy.isDynSized());
      return arrayTy.getSize() * 32;
    }
    // Struct type.
    if (auto structTy = dyn_cast<sol::StructType>(ty)) {
      // FIXME: Is the memoryHeadSize 32 for all the types (assuming padding is
      // enabled by default) in StructType::memoryDataSize?
      return structTy.getMemTypes().size() * 32;
    }

    // Value type.
    return 32;
  }

  /// Generates the memory allocation and zeroing code.
  Value genZeroedMemAlloc(Type ty, Value sizeVar, int64_t recDepth,
                          PatternRewriter &r, Location loc) const {
    recDepth++;

    Value memPtr;
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    // Array type.
    if (auto arrayTy = dyn_cast<sol::ArrayType>(ty)) {
      assert(arrayTy.getDataLocation() == sol::DataLocation::Memory);

      Value sizeInBytes, dataPtr;
      // FIXME: Round up size for byte arays.
      if (arrayTy.isDynSized()) {
        // Dynamic allocation is only performed for the outermost dimension.
        if (recDepth == 0) {
          assert(sizeVar);
          sizeInBytes =
              r.create<arith::MulIOp>(loc, sizeVar, bExt.genI256Const(32));
          memPtr = eraB.genMemAllocForDynArray(sizeVar, sizeInBytes);
          dataPtr = r.create<arith::AddIOp>(loc, memPtr, bExt.genI256Const(32));
        } else {
          return bExt.genI256Const(
              solidity::frontend::CompilerUtils::zeroPointer);
        }
      } else {
        sizeInBytes = bExt.genI256Const(getSize(ty));
        memPtr = eraB.genMemAlloc(sizeInBytes, loc);
        dataPtr = memPtr;
      }
      assert(sizeInBytes && dataPtr && memPtr);

      Type eltTy = arrayTy.getEltType();

      // Multi-dimensional array / array of structs.
      if (isa<sol::StructType>(eltTy) || isa<sol::ArrayType>(eltTy)) {
        //
        // Store the offsets to the "inner" allocations.
        //
        if (auto constSize =
                dyn_cast<arith::ConstantIntOp>(sizeInBytes.getDefiningOp())) {
          // FIXME: .value() yields s64!
          assert(constSize.value() >= 32);
          // Generate the "unrolled" stores of offsets since the size is static.
          r.create<sol::MStoreOp>(
              loc, dataPtr,
              genZeroedMemAlloc(eltTy, sizeVar, recDepth, r, loc));
          // FIXME: Is the stride always 32?
          assert(constSize.value() % 32 == 0);
          auto sizeInWords = constSize.value() / 32;
          for (int64_t i = 1; i < sizeInWords; ++i) {
            Value incrMemPtr = r.create<arith::AddIOp>(
                loc, dataPtr, bExt.genI256Const(32 * i));
            r.create<sol::MStoreOp>(
                loc, incrMemPtr,
                genZeroedMemAlloc(eltTy, sizeVar, recDepth, r, loc));
          }
        } else {
          // Generate the loop for the stores of offsets.

          // TODO? Generate an unrolled loop if the size variable is a
          // ConstantOp?

          // `size` should be a multiple of 32.

          // FIXME: Make sure that the index type is lowered to i256 in the
          // pipeline.
          r.create<scf::ForOp>(
              loc, /*lowerBound=*/bExt.genIdxConst(0),
              /*upperBound=*/bExt.genCastToIdx(sizeInBytes),
              /*step=*/bExt.genIdxConst(32), /*iterArgs=*/std::nullopt,
              /*builder=*/
              [&](OpBuilder &b, Location loc, Value indVar,
                  ValueRange iterArgs) {
                Value incrMemPtr = r.create<arith::AddIOp>(
                    loc, dataPtr, bExt.genCastToI256(indVar));
                r.create<sol::MStoreOp>(
                    loc, incrMemPtr,
                    genZeroedMemAlloc(eltTy, sizeVar, recDepth, r, loc));
                b.create<scf::YieldOp>(loc);
              });
        }

      } else {
        Value callDataSz = r.create<sol::CallDataSizeOp>(loc);
        r.create<sol::CallDataCopyOp>(loc, dataPtr, callDataSz, sizeInBytes);
      }

      // String type.
    } else if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
      if (sizeVar)
        memPtr = eraB.genMemAllocForDynArray(
            sizeVar, bExt.genRoundUpToMultiple<32>(sizeVar));
      else
        return bExt.genI256Const(
            solidity::frontend::CompilerUtils::zeroPointer);

      // Struct type.
    } else if (auto structTy = dyn_cast<sol::StructType>(ty)) {
      memPtr = eraB.genMemAlloc(getSize(ty), loc);
      assert(structTy.getDataLocation() == sol::DataLocation::Memory);

      for (auto memTy : structTy.getMemTypes()) {
        Value initVal;
        if (isa<sol::StructType>(memTy) || isa<sol::ArrayType>(memTy))
          initVal = genZeroedMemAlloc(memTy, sizeVar, recDepth, r, loc);
        else
          initVal = bExt.genI256Const(0);

        r.create<sol::MStoreOp>(loc, memPtr, initVal);
      }
      // TODO: Support other types.
    } else {
      llvm_unreachable("Invalid type");
    }

    assert(memPtr);
    return memPtr;
  }

  Value genZeroedMemAlloc(Type ty, Value sizeVar, PatternRewriter &r,
                          Location loc) const {
    return genZeroedMemAlloc(ty, sizeVar, /*recDepth=*/-1, r, loc);
  }

  LogicalResult matchAndRewrite(sol::MallocOp op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    Value freePtr = genZeroedMemAlloc(op.getAllocType(), op.getSize(), r, loc);
    r.replaceOp(op, freePtr);
    return success();
  }
};

template <typename OpT>
static Value genAddrCalc(OpT op, typename OpT::Adaptor adaptor,
                         ConversionPatternRewriter &r) {
  Location loc = op.getLoc();
  solidity::mlirgen::BuilderExt bExt(r, loc);
  eravm::Builder eraB(r, loc);

  Type baseAddrTy = op.getBaseAddr().getType();
  Value remappedBaseAddr = adaptor.getBaseAddr();

  switch (sol::getDataLocation(baseAddrTy)) {
  case sol::DataLocation::Stack: {
    auto stkPtrTy =
        LLVM::LLVMPointerType::get(r.getContext(), eravm::AddrSpace_Stack);
    Value addrAtIdx = remappedBaseAddr;
    if (!op.getIndices().empty())
      addrAtIdx =
          r.create<LLVM::GEPOp>(loc, /*resultType=*/stkPtrTy,
                                /*basePtrType=*/remappedBaseAddr.getType(),
                                remappedBaseAddr, op.getIndices());
    return addrAtIdx;
  }
  case sol::DataLocation::Memory: {
    assert(!op.getIndices().empty());

    Value idx = op.getIndices()[0];
    Value addrAtIdx;

    if (auto arrayTy = dyn_cast<sol::ArrayType>(baseAddrTy)) {
      if (!isa<BlockArgument>(idx)) {
        auto constIdx = dyn_cast<arith::ConstantIntOp>(idx.getDefiningOp());
        if (constIdx && !arrayTy.isDynSized()) {
          // FIXME: Should this be done by the verifier?
          assert(constIdx.value() < arrayTy.getSize());
          addrAtIdx = r.create<arith::AddIOp>(
              loc, remappedBaseAddr, bExt.genI256Const(constIdx.value() * 32));
        }
      }
      if (!addrAtIdx) {
        //
        // Generate PanicCode::ArrayOutOfBounds check.
        //
        Value size;
        if (arrayTy.isDynSized()) {
          size = r.create<sol::MLoadOp>(loc, remappedBaseAddr);
        } else {
          size = bExt.genI256Const(arrayTy.getSize());
        }

        // Generate `if iszero(lt(index, <arrayLen>(baseRef)))` (yul)
        auto panicCond =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, idx, size);
        eraB.genPanic(solidity::util::PanicCode::ArrayOutOfBounds, panicCond);

        //
        // Generate the address
        //
        Value scaledIdx =
            r.create<arith::MulIOp>(loc, idx, bExt.genI256Const(32));
        if (arrayTy.isDynSized()) {
          // Get the address after the length-slot.
          Value dataAddr = r.create<arith::AddIOp>(loc, remappedBaseAddr,
                                                   bExt.genI256Const(32));
          addrAtIdx = r.create<arith::AddIOp>(loc, dataAddr, scaledIdx);
        } else {
          addrAtIdx = r.create<arith::AddIOp>(loc, remappedBaseAddr, scaledIdx);
        }
      }
    } else if (auto structTy = dyn_cast<sol::StructType>(baseAddrTy)) {
      auto constIdx = cast<arith::ConstantIntOp>(idx.getDefiningOp());
      (void)constIdx;
      assert(constIdx.value() <
             static_cast<int64_t>(structTy.getMemTypes().size()));
      auto scaledIdx = r.create<arith::MulIOp>(loc, idx, bExt.genI256Const(32));
      addrAtIdx = r.create<arith::AddIOp>(loc, remappedBaseAddr, scaledIdx);
    }

    return addrAtIdx;
  }
  case sol::DataLocation::Storage: {
    assert(op.getIndices().empty() && "NYI");
    return remappedBaseAddr;
  }
  default:
    break;
  };

  llvm_unreachable("NYI: Calldata data-location");
}

struct LoadOpLowering : public OpConversionPattern<sol::LoadOp> {
  using OpConversionPattern<sol::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Value addrAtIdx = genAddrCalc(op, adaptor, r);

    switch (sol::getDataLocation(op.getBaseAddr().getType())) {
    case sol::DataLocation::Stack:
      r.replaceOpWithNewOp<LLVM::LoadOp>(
          op, addrAtIdx, eravm::getAlignment(adaptor.getBaseAddr()));
      return success();
    case sol::DataLocation::Memory:
      r.replaceOpWithNewOp<sol::MLoadOp>(op, addrAtIdx);
      return success();
    case sol::DataLocation::Storage:
      r.replaceOpWithNewOp<sol::SLoadOp>(op, addrAtIdx);
      return success();
    default:
      break;
    };
    llvm_unreachable("NYI: Calldata data-location");
  }
};

struct AddrOfOpLowering : public OpRewritePattern<sol::AddrOfOp> {
  using OpRewritePattern<sol::AddrOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::AddrOfOp op,
                                PatternRewriter &r) const override {
    solidity::mlirgen::BuilderExt bExt(r, op.getLoc());

    auto parentContract = op->getParentOfType<sol::ContractOp>();
    auto *stateVarSym = parentContract.lookupSymbol(op.getVar());
    assert(stateVarSym);
    auto stateVarOp = cast<sol::StateVarOp>(stateVarSym);
    assert(stateVarOp->hasAttr("slot"));
    IntegerAttr slot = cast<IntegerAttr>(stateVarOp->getAttr("slot"));
    r.replaceOp(op, bExt.genI256Const(slot.getValue()));
    return success();
  }
};

struct MapOpLowering : public OpConversionPattern<sol::MapOp> {
  using OpConversionPattern<sol::MapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::MapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);

    // Assert that the mapping is a slot (result of sol.addr_of or sol.map).
    assert(cast<IntegerType>(adaptor.getMapping().getType()).getWidth() == 256);

    // Setup arguments to keccak256.
    auto zero = bExt.genI256Const(0);
    r.create<sol::MStoreOp>(loc, zero, op.getKey());
    r.create<sol::MStoreOp>(loc, bExt.genI256Const(0x20), adaptor.getMapping());

    r.replaceOpWithNewOp<sol::Keccak256Op>(op, zero, bExt.genI256Const(0x40));
    return success();
  }
};

struct GetSlotOpLowering : public OpRewritePattern<sol::GetSlotOp> {
  using OpRewritePattern<sol::GetSlotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::GetSlotOp op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);

    // Setup arguments to keccak256
    // FIXME: Is it necessary to do the "cleanup" of the key?
    auto zero = bExt.genI256Const(0);
    r.create<sol::MStoreOp>(loc, zero, op.getKey());
    r.create<sol::MStoreOp>(loc, bExt.genI256Const(0x20), op.getInpSlot());

    r.replaceOpWithNewOp<sol::Keccak256Op>(op, zero, bExt.genI256Const(0x40));
    return success();
  }
};

struct StorageLoadOpLowering : public OpRewritePattern<sol::StorageLoadOp> {
  using OpRewritePattern<sol::StorageLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::StorageLoadOp op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);

    Type effTy = op.getEffectiveType();
    assert(isa<IntegerType>(effTy) &&
           "NYI: Storage types other than int types");

    // Genrate the slot load.
    auto slotLd = r.create<sol::SLoadOp>(loc, op.getSlot());

    // TODO: Can we align storage allocations by 32 bytes for all value types?
    // Then we can avoid the following "extraction" code (and also get rid of
    // the byte offset).

    // Generate the byte offset in bits.
    //
    // FIXME: Can we expect this instead? (The current version follows the
    // "extract_from_storage_value_offset_" yul util function).
    auto byteOffset =
        cast<arith::ConstantIntOp>(op.getByteOffset().getDefiningOp());
    assert(byteOffset.getType() == r.getI32Type());
    auto byteOffsetZext =
        r.create<arith::ExtUIOp>(loc, r.getIntegerType(256), byteOffset);
    auto byteOffsetInBits =
        r.create<arith::MulIOp>(loc, byteOffsetZext, bExt.genI256Const(8));

    // Generate the extraction of the sload'ed value.
    auto partiallyExtractedVal =
        r.create<arith::ShRUIOp>(loc, slotLd, byteOffsetInBits);
    Value extractedVal;
    if (eravm::getStorageByteCount(effTy) == 32) {
      extractedVal = partiallyExtractedVal;
    } else {
      if (auto intTy = dyn_cast<IntegerType>(effTy)) {
        if (intTy.isSigned()) {
          llvm_unreachable("NYI: signextend builtin");
        } else {
          if (sol::isLeftAligned(effTy)) {
            extractedVal = r.create<arith::ShLIOp>(
                loc, partiallyExtractedVal,
                bExt.genI256Const(256 - 8 * eravm::getStorageByteCount(effTy)));
          } else {
            // Zero the irrelevant high bits.
            llvm::APInt maskVal(256, 0);
            maskVal.setLowBits(8 * eravm::getStorageByteCount(effTy));
            extractedVal = r.create<arith::AndIOp>(loc, partiallyExtractedVal,
                                                   bExt.genI256Const(maskVal));
          }
        }
      }
    }
    assert(extractedVal);

    r.replaceOp(op, extractedVal);
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<sol::StoreOp> {
  using OpConversionPattern<sol::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Value addrAtIdx = genAddrCalc(op, adaptor, r);

    switch (sol::getDataLocation(op.getBaseAddr().getType())) {
    case sol::DataLocation::Stack:
      r.replaceOpWithNewOp<LLVM::StoreOp>(
          op, adaptor.getVal(), addrAtIdx,
          eravm::getAlignment(adaptor.getBaseAddr()));
      return success();
    case sol::DataLocation::Memory:
      r.replaceOpWithNewOp<sol::MStoreOp>(op, addrAtIdx, adaptor.getVal());
      return success();
    case sol::DataLocation::Storage:
      r.replaceOpWithNewOp<sol::SStoreOp>(op, addrAtIdx, adaptor.getVal());
      return success();
    default:
      break;
    };
    llvm_unreachable("NYI: Calldata data-location");
  }
};

struct DataLocCastOpLowering : public OpConversionPattern<sol::DataLocCastOp> {
  using OpConversionPattern<sol::DataLocCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::DataLocCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    Type srcTy = op.getInp().getType();
    Type dstTy = op.getType();

    mlir::Value resAddr;
    // From storage to memory.
    if (sol::getDataLocation(srcTy) == sol::DataLocation::Storage &&
        sol::getDataLocation(dstTy) == sol::DataLocation::Memory) {
      // String type
      if (auto srcStrTy = dyn_cast<sol::StringType>(srcTy)) {
        assert(isa<sol::StringType>(dstTy));
        Value slot = adaptor.getInp();
        assert(cast<IntegerType>(slot.getType()).getWidth() == 256);

        // Generate keccak256(slot) to get the slot having the data.
        auto zero = bExt.genI256Const(0);
        auto thirtyTwo = bExt.genI256Const(32);
        r.create<sol::MStoreOp>(loc, zero, slot);
        auto dataSlot = r.create<sol::Keccak256Op>(loc, zero, thirtyTwo);

        // Generate the memory allocation.
        auto sizeInBytes = r.create<sol::SLoadOp>(loc, slot);
        auto memPtrTy =
            sol::StringType::get(r.getContext(), sol::DataLocation::Memory);
        // TODO: Define custom builder (or remove the type-attr arg?) in
        // sol.malloc.
        Value mallocRes =
            r.create<sol::MallocOp>(loc, memPtrTy, memPtrTy, sizeInBytes);
        Type i256Ty = r.getIntegerType(256);
        assert(getTypeConverter()->convertType(mallocRes.getType()) == i256Ty);
        mlir::Value memAddr = getTypeConverter()->materializeSourceConversion(
            r, loc, /*resultType=*/i256Ty, /*inputs=*/mallocRes);
        resAddr = memAddr;

        // Generate the loop to copy the data (sol.malloc lowering will generate
        // the store of the size field).
        auto dataMemAddr = r.create<arith::AddIOp>(loc, memAddr, thirtyTwo);
        auto sizeInWords = bExt.genRoundUpToMultiple<32>(sizeInBytes);
        r.create<scf::ForOp>(
            loc, /*lowerBound=*/bExt.genIdxConst(0),
            /*upperBound=*/bExt.genCastToIdx(sizeInWords),
            /*step=*/bExt.genIdxConst(1), /*iterArgs=*/std::nullopt,
            /*builder=*/
            [&](OpBuilder &b, Location loc, Value indVar, ValueRange iterArgs) {
              Value i256IndVar = bExt.genCastToI256(indVar);

              Value slotIdx =
                  r.create<arith::AddIOp>(loc, dataSlot, i256IndVar);
              Value slotVal = r.create<sol::SLoadOp>(loc, slotIdx);

              Value memIdx =
                  r.create<arith::MulIOp>(loc, i256IndVar, thirtyTwo);
              Value memAddrAtIdx =
                  r.create<arith::AddIOp>(loc, dataMemAddr, memIdx);

              r.create<sol::MStoreOp>(loc, memAddrAtIdx, slotVal);
              b.create<scf::YieldOp>(loc);
            });
      } else {
        llvm_unreachable("NYI");
      }

    } else {
      llvm_unreachable("NYI");
    }

    assert(resAddr);

    r.replaceOp(op, resAddr);
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<sol::ReturnOp> {
  using OpConversionPattern<sol::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<sol::CallOp> {
  using OpConversionPattern<sol::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    SmallVector<Type> convertedResTys;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                convertedResTys)))
      return failure();
    r.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), convertedResTys,
                                       adaptor.getOperands());
    return success();
  }
};

struct EmitOpLowering : public OpConversionPattern<sol::EmitOp> {
  using OpConversionPattern<sol::EmitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::EmitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    // Generate the tuple encoding for the non-indexed args.
    std::vector<Value> nonIndexedArgs;
    std::vector<Type> nonIndexedArgsType;
    for (Value arg : op.getNonIndexedArgs()) {
      // FIXME: The remapped non-indexed args should be pushed here. How do we
      // get getNonIndexedArgs in the OpAdaptor?
      nonIndexedArgs.push_back(arg);
      assert(cast<IntegerType>(arg.getType()).getWidth() == 256);
      nonIndexedArgsType.push_back(arg.getType());
    }
    // TODO: Are we sure we need an unbounded allocation here?
    Value headAddr = eraB.genFreePtr();
    Value tailAddr =
        eraB.genABITupleEncoding(nonIndexedArgsType, nonIndexedArgs, headAddr);
    Value tupleSize = r.create<arith::SubIOp>(loc, tailAddr, headAddr);

    // Collect indexed args.
    std::vector<Value> indexedArgs;
    if (op.getSignature()) {
      auto signatureHash =
          solidity::util::h256::Arith(*op.getSignature()).str();
      indexedArgs.push_back(bExt.genI256Const(signatureHash));
    }
    for (Value arg : op.getIndexedArgs())
      indexedArgs.push_back(arg);

    // Generate sol.log and replace sol.emit with it.
    r.replaceOpWithNewOp<sol::LogOp>(op, headAddr, tupleSize, indexedArgs);

    return success();
  }
};

struct RequireOpLowering : public OpRewritePattern<sol::RequireOp> {
  using OpRewritePattern<sol::RequireOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::RequireOp op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();

    // Generate the revert condition.
    mlir::Value falseVal =
        r.create<arith::ConstantIntOp>(loc, 0, r.getI1Type());
    mlir::Value negCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  op.getCond(), falseVal);
    // Generate the revert.
    eravm::Builder eraB(r, loc);
    if (!op.getMsg().empty())
      eraB.genRevertWithMsg(negCond, op.getMsg().str());
    else
      eraB.genRevert(negCond);

    r.eraseOp(op);
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<sol::FuncOp> {
  using OpConversionPattern<sol::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    mlir::Location loc = op.getLoc();
    eravm::Builder eraB(r, loc);

    // Collect non-core attributes.
    std::vector<NamedAttribute> attrs;
    bool hasLinkageAttr = false;
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef attrName = attr.getName();
      if (attrName == "function_type" || attrName == "sym_name" ||
          attrName.startswith("sol."))
        continue;
      if (attrName == "llvm.linkage")
        hasLinkageAttr = true;
      attrs.push_back(attr);
    }

    // Set llvm.linkage attribute to private if not explicitly specified.
    if (!hasLinkageAttr)
      attrs.push_back(r.getNamedAttr(
          "llvm.linkage",
          LLVM::LinkageAttr::get(r.getContext(), LLVM::Linkage::Private)));

    // Set the personality attribute of llvm.
    attrs.push_back(r.getNamedAttr("personality", eraB.getPersonality()));

    // Add the nofree and null_pointer_is_valid attributes of llvm via the
    // passthrough attribute.
    std::vector<Attribute> passthroughAttrs;
    passthroughAttrs.push_back(r.getStringAttr("nofree"));
    passthroughAttrs.push_back(r.getStringAttr("null_pointer_is_valid"));
    attrs.push_back(r.getNamedAttr(
        "passthrough", ArrayAttr::get(r.getContext(), passthroughAttrs)));

    // TODO: Add additional attribute for -O0 and -Oz

    auto convertedFuncTy = cast<FunctionType>(
        getTypeConverter()->convertType(op.getFunctionType()));
    auto newOp =
        r.create<func::FuncOp>(loc, op.getName(), convertedFuncTy, attrs);
    r.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    r.eraseOp(op);
    return success();
  }
};

struct ObjectOpLowering : public OpRewritePattern<sol::ObjectOp> {
  using OpRewritePattern<sol::ObjectOp>::OpRewritePattern;

  /// Move FuncOps in the ObjectOp to the ModuleOp by disambiguating the symbol
  /// names.
  void moveFuncsToModule(sol::ObjectOp objOp, ModuleOp mod) const {
    // Track the names of the existing function in the module.
    std::set<std::string> fnNamesInMod;
    // TODO: Is there a way to run .walk for 1 level depth?
    for (mlir::Operation &op : *mod.getBody()) {
      if (auto fn = dyn_cast<sol::FuncOp>(&op)) {
        std::string name(fn.getName());
        assert(fnNamesInMod.count(name) == 0 &&
               "Duplicate functions in module");
        fnNamesInMod.insert(name);
      }
    }

    // TODO:
    // - Is there a better way to do the symbol disambiguation?
    // - replaceAllSymbolUses doesn't affect symbolic reference by attributes.
    //   Is that a problem here?
    objOp.walk([&](sol::FuncOp fn) {
      std::string name(fn.getName());

      // Does the module have a function with the same name?
      if (fnNamesInMod.count(name)) {
        // Disambiguate the symbol.
        unsigned counter = 0;
        auto newName = name + "." + std::to_string(counter);
        while (fnNamesInMod.count(newName))
          newName = name + "." + std::to_string(++counter);

        // The visitor might be at a inner object, so we need to explicitly
        // fetch the parent ObjectOp.
        auto parentObjectOp = fn->getParentOfType<sol::ObjectOp>();
        assert(parentObjectOp);
        if (failed(fn.replaceAllSymbolUses(
                StringAttr::get(objOp.getContext(), newName), parentObjectOp)))
          llvm_unreachable("replaceAllSymbolUses failed");
        fn.setName(newName);
        fnNamesInMod.insert(newName);

      } else {
        fnNamesInMod.insert(name);
      }

      fn->moveBefore(mod.getBody(), mod.getBody()->begin());
    });
  }

  LogicalResult matchAndRewrite(sol::ObjectOp objOp,
                                PatternRewriter &r) const override {
    mlir::Location loc = objOp.getLoc();

    auto mod = objOp->getParentOfType<ModuleOp>();
    auto i256Ty = r.getIntegerType(256);
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    moveFuncsToModule(objOp, mod);

    // Is this a runtime object?
    // FIXME: Is there a better way to check this?
    if (objOp.getSymName().endswith("_deployed")) {
      // Move the runtime object region under the __runtime function
      sol::FuncOp runtimeFunc = eraB.getOrInsertRuntimeFuncOp(
          "__runtime", r.getFunctionType({}, {}), mod);
      Region &runtimeFuncRegion = runtimeFunc.getRegion();
      assert(runtimeFuncRegion.empty());
      r.inlineRegionBefore(objOp.getRegion(), runtimeFuncRegion,
                           runtimeFuncRegion.begin());
      r.eraseOp(objOp);
      return success();
    }

    // Define the __entry function
    auto genericAddrSpacePtrTy =
        LLVM::LLVMPointerType::get(r.getContext(), eravm::AddrSpace_Generic);
    std::vector<Type> inTys{genericAddrSpacePtrTy};
    constexpr unsigned argCnt =
        eravm::MandatoryArgCnt + eravm::ExtraABIDataSize;
    for (unsigned i = 0; i < argCnt - 1; ++i) {
      inTys.push_back(i256Ty);
    }
    FunctionType funcType = r.getFunctionType(inTys, {i256Ty});
    r.setInsertionPointToEnd(mod.getBody());
    // FIXME: Assert __entry is created here.
    sol::FuncOp entryFunc = bExt.getOrInsertFuncOp(
        "__entry", funcType, LLVM::Linkage::External, mod);
    assert(objOp->getNumRegions() == 1);

    // Setup the entry block and set insertion point to it
    auto &entryFuncRegion = entryFunc.getRegion();
    Block *entryBlk = r.createBlock(&entryFuncRegion);
    for (auto inTy : inTys) {
      entryBlk->addArgument(inTy, loc);
    }
    r.setInsertionPointToStart(entryBlk);

    // Initialize globals.
    eraB.genGlobalVarsInit(mod);

    // Store the calldata ABI arg to the global calldata ptr.
    LLVM::AddressOfOp callDataPtrAddr = eraB.genCallDataPtrAddr(mod);
    r.create<LLVM::StoreOp>(
        loc, entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallDataABI),
        callDataPtrAddr, eravm::getAlignment(callDataPtrAddr));

    // Store the calldata ABI size to the global calldata size.
    Value abiLen = eraB.genABILen(callDataPtrAddr);
    LLVM::AddressOfOp callDataSizeAddr = eraB.genCallDataSizeAddr(mod);
    r.create<LLVM::StoreOp>(loc, abiLen, callDataSizeAddr,
                            eravm::getAlignment(callDataSizeAddr));

    // Store calldatasize[calldata abi arg] to the global ret data ptr and
    // active ptr
    auto callDataSz = r.create<LLVM::LoadOp>(
        loc, callDataSizeAddr, eravm::getAlignment(callDataSizeAddr));
    auto retDataABIInitializer = r.create<LLVM::GEPOp>(
        loc,
        /*resultType=*/
        LLVM::LLVMPointerType::get(
            mod.getContext(),
            bExt.getGlobalOp(callDataPtrAddr.getGlobalName(), mod)
                .getAddrSpace()),
        /*basePtrType=*/r.getIntegerType(eravm::BitLen_Byte),
        /*basePtr=*/
        entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallDataABI),
        /*indices=*/callDataSz.getResult());
    auto storeRetDataABIInitializer = [&](const char *name) {
      LLVM::GlobalOp globDef = bExt.getOrInsertPtrGlobalOp(
          name, eravm::AddrSpace_Generic, LLVM::Linkage::Private, mod);
      Value globAddr = r.create<LLVM::AddressOfOp>(loc, globDef);
      if (name == eravm::GlobActivePtr) {
        auto arrTy = cast<LLVM::LLVMArrayType>(globDef.getType());
        for (unsigned i = 0; i < arrTy.getNumElements(); ++i) {
          auto gep = r.create<LLVM::GEPOp>(
              loc, /*resultType=*/
              LLVM::LLVMPointerType::get(mod.getContext(),
                                         eravm::AddrSpace_Generic),
              /*basePtrType=*/arrTy, /*basePtr=*/globAddr,
              /*indices=*/
              ValueRange{bExt.genI256Const(0), bExt.genI256Const(i)});
          gep.setElemTypeAttr(TypeAttr::get(
              LLVM::LLVMArrayType::get(i256Ty, arrTy.getNumElements())));
          r.create<LLVM::StoreOp>(loc, retDataABIInitializer, gep,
                                  eravm::getAlignment(globAddr));
        }
      } else {
        r.create<LLVM::StoreOp>(loc, retDataABIInitializer, globAddr,
                                eravm::getAlignment(globAddr));
      }
    };
    storeRetDataABIInitializer(eravm::GlobRetDataPtr);
    storeRetDataABIInitializer(eravm::GlobDecommitPtr);
    storeRetDataABIInitializer(eravm::GlobActivePtr);

    // Store call flags arg to the global call flags
    auto globCallFlagsDef = bExt.getGlobalOp(eravm::GlobCallFlags, mod);
    Value globCallFlags = r.create<LLVM::AddressOfOp>(loc, globCallFlagsDef);
    r.create<LLVM::StoreOp>(
        loc, entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallFlags),
        globCallFlags, eravm::getAlignment(globCallFlags));

    // Store the remaining args to the global extra ABI data
    auto globExtraABIDataDef = bExt.getGlobalOp(eravm::GlobExtraABIData, mod);
    Value globExtraABIData =
        r.create<LLVM::AddressOfOp>(loc, globExtraABIDataDef);
    for (unsigned i = 2; i < entryBlk->getNumArguments(); ++i) {
      auto gep = r.create<LLVM::GEPOp>(
          loc,
          /*resultType=*/
          LLVM::LLVMPointerType::get(mod.getContext(),
                                     globExtraABIDataDef.getAddrSpace()),
          /*basePtrType=*/globExtraABIDataDef.getType(), globExtraABIData,
          ValueRange{bExt.genI256Const(0), bExt.genI256Const(i - 2)});
      // FIXME: How does the opaque ptr geps with scalar element types lower
      // without explictly setting the elem_type attr?
      gep.setElemTypeAttr(TypeAttr::get(globExtraABIDataDef.getType()));
      r.create<LLVM::StoreOp>(loc, entryBlk->getArgument(i), gep,
                              eravm::getAlignment(gep));
    }

    // Check Deploy call flag
    auto deployCallFlag = r.create<arith::AndIOp>(
        loc, entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallFlags),
        bExt.genI256Const(1));
    auto isDeployCallFlag = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, deployCallFlag.getResult(),
        bExt.genI256Const(1));

    // Create the __runtime function
    sol::FuncOp runtimeFunc = eraB.getOrInsertRuntimeFuncOp(
        "__runtime", r.getFunctionType({}, {}), mod);
    Region &runtimeFuncRegion = runtimeFunc.getRegion();
    // Move the runtime object getter under the ObjectOp public API
    for (auto const &op : *objOp.getBody()) {
      if (auto runtimeObj = dyn_cast<sol::ObjectOp>(&op)) {
        assert(runtimeObj.getSymName().endswith("_deployed"));
        assert(runtimeFuncRegion.empty());
        r.inlineRegionBefore(runtimeObj.getRegion(), runtimeFuncRegion,
                             runtimeFuncRegion.begin());
        OpBuilder::InsertionGuard insertGuard(r);
        r.setInsertionPointToEnd(&runtimeFuncRegion.front());
        r.create<LLVM::UnreachableOp>(runtimeObj.getLoc());
        r.eraseOp(runtimeObj);
      }
    }

    // Create the __deploy function
    sol::FuncOp deployFunc = eraB.getOrInsertCreationFuncOp(
        "__deploy", r.getFunctionType({}, {}), mod);
    Region &deployFuncRegion = deployFunc.getRegion();
    assert(deployFuncRegion.empty());
    r.inlineRegionBefore(objOp.getRegion(), deployFuncRegion,
                         deployFuncRegion.begin());
    {
      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToEnd(&deployFuncRegion.front());
      r.create<LLVM::UnreachableOp>(loc);
    }

    // If the deploy call flag is set, call __deploy()
    auto ifOp = r.create<scf::IfOp>(loc, isDeployCallFlag.getResult(),
                                    /*withElseRegion=*/true);
    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
    thenBuilder.create<sol::CallOp>(loc, deployFunc, ValueRange{});
    // FIXME: Why the following fails with a "does not reference a valid
    // function" error but generating the sol::CallOp to __return is fine
    // thenBuilder.create<sol::CallOp>(
    //     loc, SymbolRefAttr::get(mod.getContext(), "__deploy"), TypeRange{},
    //     ValueRange{});

    // Else call __runtime()
    OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
    elseBuilder.create<sol::CallOp>(loc, runtimeFunc, ValueRange{});
    r.setInsertionPointAfter(ifOp);
    r.create<LLVM::UnreachableOp>(loc);

    r.eraseOp(objOp);
    return success();
  }
};

struct ContractOpLowering : public OpRewritePattern<sol::ContractOp> {
  using OpRewritePattern<sol::ContractOp>::OpRewritePattern;

  /// Generate the call value check.
  void genCallValChk(PatternRewriter &r, Location loc) const {
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    auto callVal = r.create<sol::CallValOp>(loc);
    auto callValChk = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                              callVal, bExt.genI256Const(0));
    eraB.genRevert(callValChk);
  };

  /// Generate the free pointer initialization.
  void genFreePtrInit(PatternRewriter &r, Location loc) const {
    solidity::mlirgen::BuilderExt bExt(r, loc);
    mlir::Value freeMem;
    if (/* TODO: op.memoryUnsafeInlineAssemblySeen */ false) {
      freeMem = bExt.genI256Const(
          solidity::frontend::CompilerUtils::generalPurposeMemoryStart +
          /* TODO: op.getReservedMem() */ 0);
    } else {
      freeMem = r.create<sol::MemGuardOp>(
          loc,
          r.getIntegerAttr(
              r.getIntegerType(256),
              solidity::frontend::CompilerUtils::generalPurposeMemoryStart +
                  /* TODO: op.getReservedMem() */ 0));
    }
    r.create<sol::MStoreOp>(loc, bExt.genI256Const(64), freeMem);
  };

  /// Generates the dispatch to interface function of the contract op inside the
  /// object op.
  void genDispatch(sol::ContractOp contrOp, sol::ObjectOp objOp,
                   PatternRewriter &r) const {
    Location loc = contrOp.getLoc();

    ArrayAttr ifcFnsAttr = contrOp.getInterfaceFnsAttr();
    // Do nothing if there are no interface functions.
    if (ifcFnsAttr.empty())
      return;

    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    // Generate `if iszero(lt(calldatasize(), 4))` and set the insertion point
    // to its then block.
    auto callDataSz = r.create<sol::CallDataSizeOp>(loc);
    auto callDataSzCmp = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, callDataSz, bExt.genI256Const(4));
    auto ifOp =
        r.create<scf::IfOp>(loc, callDataSzCmp, /*withElseRegion=*/false);
    OpBuilder::InsertionGuard insertGuard(r);
    r.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Load the selector from the calldata.
    auto callDataLd = r.create<sol::CallDataLoadOp>(loc, bExt.genI256Const(0));
    Value callDataSelector =
        r.create<arith::ShRUIOp>(loc, callDataLd, bExt.genI256Const(224));
    callDataSelector =
        r.create<arith::TruncIOp>(loc, r.getIntegerType(32), callDataSelector);

    // Create an attribute to track all the selectors.
    std::vector<uint32_t> selectors;
    for (Attribute attr : ifcFnsAttr) {
      DictionaryAttr ifcFnAttr = cast<DictionaryAttr>(attr);
      selectors.push_back(
          cast<IntegerAttr>(ifcFnAttr.get("selector")).getInt());
    }
    auto selectorsAttr = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get(static_cast<int64_t>(selectors.size()),
                                    r.getIntegerType(32)),
        selectors);

    // Generate the switch op.
    auto switchOp = r.create<mlir::scf::IntSwitchOp>(
        loc, /*resultTypes=*/std::nullopt, callDataSelector, selectorsAttr,
        selectors.size());

    // Generate the default block.
    {
      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToStart(r.createBlock(&switchOp.getDefaultRegion()));
      r.create<scf::YieldOp>(loc);
    }

    for (auto [caseRegion, attr] :
         llvm::zip(switchOp.getCaseRegions(), ifcFnsAttr)) {
      DictionaryAttr ifcFnAttr = cast<DictionaryAttr>(attr);
      auto ifcFnSym = cast<SymbolRefAttr>(ifcFnAttr.get("sym"));
      sol::FuncOp ifcFnOp = objOp.lookupSymbol<sol::FuncOp>(ifcFnSym);
      assert(ifcFnOp);
      auto origIfcFnTy =
          cast<FunctionType>(cast<TypeAttr>(ifcFnAttr.get("type")).getValue());

      if (contrOp.getKind() == sol::ContractKind::Library) {
        auto optStateMutability = ifcFnOp.getStateMutability();
        assert(optStateMutability);
        sol::StateMutability stateMutability = *optStateMutability;

        assert(!(stateMutability == sol::StateMutability::Payable));

        if (stateMutability > sol::StateMutability::View) {
          assert(false && "NYI: Delegate call check");
        }
      }

      OpBuilder::InsertionGuard insertGuard(r);
      mlir::Block *caseBlk = r.createBlock(&caseRegion);
      r.setInsertionPointToStart(caseBlk);

      if (!(contrOp.getKind() ==
            sol::ContractKind::Library /* || func.isPayable() */)) {
        genCallValChk(r, loc);
      }

      // Decode the input parameters (if required).
      std::vector<Value> params;
      if (!origIfcFnTy.getInputs().empty()) {
        Value headStart = bExt.genI256Const(4);
        eraB.genABITupleSizeAssert(
            origIfcFnTy.getInputs(),
            r.create<arith::SubIOp>(loc, callDataSz, headStart));
        eraB.genABITupleDecoding(origIfcFnTy.getInputs(), headStart, params,
                                 /*fromMem=*/false);
      }

      // Generate the actual call.
      auto callOp = r.create<sol::CallOp>(loc, ifcFnOp, params);

      // Encode the result using the ABI's tuple encoder.
      auto headStart = eraB.genFreePtr();
      mlir::Value tupleSize;
      if (!callOp.getResultTypes().empty()) {
        auto tail = eraB.genABITupleEncoding(origIfcFnTy.getResults(),
                                             callOp.getResults(), headStart);
        tupleSize = r.create<arith::SubIOp>(loc, tail, headStart);
      } else {
        tupleSize = bExt.genI256Const(0);
      }

      // Generate the return.
      assert(tupleSize);
      r.create<sol::BuiltinRetOp>(loc, headStart, tupleSize);

      r.create<mlir::scf::YieldOp>(loc);
    }
  }

  LogicalResult matchAndRewrite(sol::ContractOp op,
                                PatternRewriter &r) const override {
    mlir::Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    // Generate the creation and runtime ObjectOp.
    auto creationObj = r.create<sol::ObjectOp>(loc, op.getSymName());
    r.setInsertionPointToStart(creationObj.getBody());
    auto runtimeObj = r.create<sol::ObjectOp>(
        loc, std::string(op.getSymName()) + "_deployed");

    // Copy contained function to creation and runtime ObjectOp.
    std::vector<sol::FuncOp> funcs;
    sol::FuncOp ctor;
    for (Operation &i : op.getBody()->getOperations()) {
      if (auto func = dyn_cast<sol::FuncOp>(i))
        funcs.push_back(func);
      else
        llvm_unreachable("NYI: Non function entities in contract");
    }
    for (sol::FuncOp func : funcs) {
      if (func.getCtor()) {
        assert(!ctor);
        ctor = func;
        ctor->moveBefore(creationObj.getBody(), creationObj.getBody()->begin());
        continue;
      }
      // Duplicate non-ctor func in both the creation and runtime objects.
      r.clone(*func);
      func->moveBefore(runtimeObj.getBody(), runtimeObj.getBody()->begin());
      func.setRuntimeAttr(r.getUnitAttr());
    }

    //
    // Creation context
    //

    r.setInsertionPointToStart(creationObj.getBody());

    genFreePtrInit(r, loc);

    if (!ctor /* TODO: || !op.ctor.isPayable() */) {
      genCallValChk(r, loc);
    }

    // Generate the call to constructor (if required).
    if (ctor && op.getKind() != sol::ContractKind::Library) {
      auto progSize = r.create<sol::DataSizeOp>(loc, creationObj.getName());
      auto codeSize = r.create<sol::CodeSizeOp>(loc);
      auto argSize = r.create<arith::SubIOp>(loc, codeSize, progSize);
      Value memPtr = eraB.genMemAlloc(argSize);
      r.create<sol::CodeCopyOp>(loc, memPtr, progSize, argSize);
      std::vector<Value> decodedArgs;
      if (!ctor.getFunctionType().getInputs().empty()) {
        eraB.genABITupleSizeAssert(ctor.getFunctionType().getInputs(), argSize);
        eraB.genABITupleDecoding(ctor.getFunctionType().getInputs(), memPtr,
                                 decodedArgs, /*fromMem=*/true);
      }
      r.create<sol::CallOp>(loc, ctor, decodedArgs);
    }

    // Generate the codecopy of the runtime object to the free ptr.
    auto freePtr = r.create<sol::MLoadOp>(loc, bExt.genI256Const(64));
    auto runtimeObjSym = FlatSymbolRefAttr::get(runtimeObj);
    auto runtimeObjOffset = r.create<sol::DataOffsetOp>(loc, runtimeObjSym);
    auto runtimeObjSize = r.create<sol::DataSizeOp>(loc, runtimeObjSym);
    r.create<sol::CodeCopyOp>(loc, freePtr, runtimeObjOffset, runtimeObjSize);

    // TODO: Generate the setimmutable's.

    // Generate the return for the creation context.
    r.create<sol::BuiltinRetOp>(loc, freePtr, runtimeObjOffset);

    //
    // Runtime context
    //

    r.setInsertionPointToStart(runtimeObj.getBody());

    // Generate the memory init.
    // TODO: Confirm if this should be the same as in the creation context.
    genFreePtrInit(r, loc);

    if (op.getKind() == sol::ContractKind::Library) {
      // TODO: called_via_delegatecall
    }

    // Generate the dispatch to interface functions.
    genDispatch(op, runtimeObj, r);

    if (op.getReceiveFnAttr()) {
      assert(false && "NYI: Receive functions");
      // TODO: Handle ether recieve function.
    }

    if (op.getFallbackFnAttr()) {
      assert(false && "NYI: Fallback functions");
      // TODO: Handle fallback function.

    } else {
      // TODO: Generate error message.
      r.create<sol::RevertOp>(loc, bExt.genI256Const(0), bExt.genI256Const(0));
    }

    assert(op.getBody()->empty());
    r.eraseOp(op);
    // TODO: Subobjects
    return success();
  }
};

/// The type converter for the ConvertSolToStandard pass.
class SolTypeConverter : public TypeConverter {
public:
  SolTypeConverter() {
    addConversion([&](IntegerType ty) -> Type { return ty; });
    addConversion([&](LLVM::LLVMPointerType ty) -> Type { return ty; });

    addConversion([&](FunctionType ty) -> Type {
      SmallVector<Type> convertedInpTys, convertedResTys;
      if (failed(convertTypes(ty.getInputs(), convertedInpTys)))
        llvm_unreachable("Invalid type");
      if (failed(convertTypes(ty.getResults(), convertedResTys)))
        llvm_unreachable("Invalid type");

      return FunctionType::get(ty.getContext(), convertedInpTys,
                               convertedResTys);
    });

    addConversion([&](sol::ArrayType ty) -> Type {
      switch (ty.getDataLocation()) {
      case sol::DataLocation::Stack: {
        Type eltTy = convertType(ty.getEltType());
        return LLVM::LLVMArrayType::get(eltTy, ty.getSize());
      }

      // Map to the 256 bit address in memory.
      case sol::DataLocation::Memory:
        return IntegerType::get(ty.getContext(), 256,
                                IntegerType::SignednessSemantics::Signless);

      default:
        break;
      }

      llvm_unreachable("Unimplemented type conversion");
    });

    addConversion([&](sol::StringType ty) -> Type {
      switch (ty.getDataLocation()) {
      // Map to the 256 bit address in memory.
      case sol::DataLocation::Memory:
      // Map to the 256 bit slot offset.
      case sol::DataLocation::Storage:
        return IntegerType::get(ty.getContext(), 256,
                                IntegerType::SignednessSemantics::Signless);

      default:
        break;
      }

      llvm_unreachable("Unimplemented type conversion");
    });

    addConversion([&](sol::MappingType ty) -> Type {
      // Map to the 256 bit slot offset.
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);
    });

    addConversion([&](sol::StructType ty) -> Type {
      switch (ty.getDataLocation()) {
      case sol::DataLocation::Memory:
        return IntegerType::get(ty.getContext(), 256,
                                IntegerType::SignednessSemantics::Signless);
      default:
        break;
      }

      llvm_unreachable("Unimplemented type conversion");
    });

    addConversion([&](sol::PointerType ty) -> Type {
      switch (ty.getDataLocation()) {
      case sol::DataLocation::Stack: {
        Type eltTy = convertType(ty.getPointeeType());
        return LLVM::LLVMPointerType::get(eltTy);
      }

      // Map to the 256 bit slot offset.
      //
      // TODO: Can we get all storage types to be 32 byte aligned? If so, we can
      // avoid the byte offset. Otherwise we should consider the
      // OneToNTypeConversion to map the pointer to the slot + byte offset pair.
      case sol::DataLocation::Storage:
        return IntegerType::get(ty.getContext(), 256,
                                IntegerType::SignednessSemantics::Signless);

      default:
        break;
      }

      llvm_unreachable("Unimplemented type conversion");
    });

    addSourceMaterialization([](OpBuilder &b, Type resTy, ValueRange ins,
                                Location loc) -> Value {
      if (ins.size() != 1)
        return b.create<UnrealizedConversionCastOp>(loc, resTy, ins)
            .getResult(0);

      Type i256Ty = b.getIntegerType(256);

      Type inpTy = ins[0].getType();

      if ((sol::isRefType(inpTy) && resTy == i256Ty) ||
          (inpTy == i256Ty && sol::isRefType(resTy)))
        return b.create<sol::ConvCastOp>(loc, resTy, ins);

      return b.create<UnrealizedConversionCastOp>(loc, resTy, ins).getResult(0);
    });
  }
};

/// Pass for lowering the sol dialect to the standard dialects.
/// TODO:
/// - Generate this using mlir-tblgen.
/// - Move this and createConvertSolToStandardPass out of EraVM.
struct ConvertSolToStandard
    : public PassWrapper<ConvertSolToStandard, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSolToStandard)

  ConvertSolToStandard() = default;

  ConvertSolToStandard(solidity::mlirgen::Target tgt) : tgt(tgt) {}

  ConvertSolToStandard(ConvertSolToStandard const &other)
      : PassWrapper(other) {}

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<func::FuncDialect, scf::SCFDialect, arith::ArithDialect,
               LLVM::LLVMDialect>();
  }

  // TODO: Generalize this comment.
  //
  // FIXME: Some of the conversion patterns depends on the ancestor/descendant
  // sol.func ops (e.g.: checking the runtime/creation context). Also APIs
  // like getOrInsert*FuncOp that are used in the conversion pass works with
  // sol.func. sol.func conversion in the same conversion pass will require
  // other conversion to be able to work with sol.func and func.func. To keep
  // things simple for now, the sol.func and related ops lowering is scheduled
  // in a separate conversion pass after the main one.
  //
  // How can we do the conversion cleanly with one pass? Generally speaking,
  // how should we work with conversion patterns that depends on other
  // operations?

  // FIXME: Separate yul specific sol ops to a "yul" dialect? This could
  // simplify the implementation of this multi-staged lowering.

  /// Converts sol dialect ops except sol.contract and sol.func + related ops.
  /// This pass also legalizes all the sol dialect types.
  void runStage1Conversion(ModuleOp mod, SolTypeConverter &tyConv) {
    OpBuilder b(mod.getContext());
    eravm::Builder eraB(b);

    // FIXME: DialectConversion complains "pattern was already applied" if we do
    // this in the sol.func lowering (might work if we generate a llvm/func.func
    // op instead? Should we switch all such external functions to
    // llvm/func.func?)
    eraB.getOrInsertPersonality(mod);

    ConversionTarget tgt(getContext());
    tgt.addLegalOp<mlir::ModuleOp>();
    tgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                        arith::ArithDialect, LLVM::LLVMDialect>();
    tgt.addIllegalOp<sol::AllocaOp, sol::MallocOp, sol::AddrOfOp, sol::MapOp,
                     sol::DataLocCastOp, sol::LoadOp, sol::StoreOp, sol::EmitOp,
                     sol::RequireOp, sol::ConvCastOp>();
    tgt.addDynamicallyLegalOp<sol::FuncOp>([&](sol::FuncOp op) {
      return tyConv.isSignatureLegal(op.getFunctionType());
    });
    tgt.addDynamicallyLegalOp<sol::CallOp, sol::ReturnOp>(
        [&](Operation *op) { return tyConv.isLegal(op); });

    RewritePatternSet pats(&getContext());
    pats.add<AllocaOpLowering, MapOpLowering, DataLocCastOpLowering,
             LoadOpLowering, StoreOpLowering, ConvCastOpLowering,
             EmitOpLowering>(tyConv, &getContext());
    populateAnyFunctionOpInterfaceTypeConversionPattern(pats, tyConv);
    pats.add<GenericTypeConversion<sol::CallOp>,
             GenericTypeConversion<sol::ReturnOp>>(tyConv, &getContext());
    pats.add<MallocOpLowering, AddrOfOpLowering, RequireOpLowering>(
        &getContext());

    // Assign slots to state variables.
    mod.walk([&](sol::ContractOp contr) {
      APInt slot(256, 0);
      contr.walk([&](sol::StateVarOp stateVar) {
        assert(eravm::getStorageByteCount(stateVar.getType()) == 32);
        stateVar->setAttr(
            "slot",
            IntegerAttr::get(IntegerType::get(&getContext(), 256), slot++));
      });
    });

    if (failed(applyPartialConversion(mod, tgt, std::move(pats))))
      signalPassFailure();

    // Remove all state variables.
    mod.walk([](sol::StateVarOp op) { op.erase(); });
  }

  /// Converts sol.contract and all yul dialect ops.
  void runStage2Conversion(ModuleOp mod) {
    ConversionTarget tgt(getContext());
    tgt.addLegalOp<mlir::ModuleOp>();
    tgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                        arith::ArithDialect, LLVM::LLVMDialect>();
    tgt.addIllegalDialect<sol::SolDialect>();
    tgt.addLegalOp<sol::FuncOp, sol::CallOp, sol::ReturnOp, sol::ConvCastOp>();

    RewritePatternSet pats(&getContext());
    pats.add<ContractOpLowering, ObjectOpLowering, BuiltinRetOpLowering,
             RevertOpLowering, MLoadOpLowering, MStoreOpLowering,
             MCopyOpLowering, DataOffsetOpLowering, DataSizeOpLowering,
             CodeSizeOpLowering, CodeCopyOpLowering, MemGuardOpLowering,
             CallValOpLowering, CallDataLoadOpLowering, CallDataSizeOpLowering,
             CallDataCopyOpLowering, SLoadOpLowering, SStoreOpLowering,
             Keccak256OpLowering, LogOpLowering, CallerOpLowering>(
        &getContext());

    if (failed(applyPartialConversion(mod, tgt, std::move(pats))))
      signalPassFailure();
  }

  /// Converts sol.func and related ops.
  void runStage3Conversion(ModuleOp mod, SolTypeConverter &tyConv) {
    ConversionTarget tgt(getContext());
    tgt.addLegalOp<mlir::ModuleOp>();
    tgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                        arith::ArithDialect, LLVM::LLVMDialect>();
    tgt.addIllegalDialect<sol::SolDialect>();

    RewritePatternSet pats(&getContext());
    pats.add<FuncOpLowering, CallOpLowering, ReturnOpLowering>(tyConv,
                                                               &getContext());

    if (failed(applyPartialConversion(mod, tgt, std::move(pats))))
      signalPassFailure();
  }

  void runOnOperation() override {
    // We can't check this in the ctor since cl::ParseCommandLineOptions won't
    // be called then.
    if (clTgt.getNumOccurrences() > 0) {
      assert(tgt == solidity::mlirgen::Target::Undefined);
      tgt = clTgt;
    }

    ModuleOp mod = getOperation();
    SolTypeConverter tyConv;
    runStage1Conversion(mod, tyConv);
    runStage2Conversion(mod);
    runStage3Conversion(mod, tyConv);
  }

  StringRef getArgument() const override { return "convert-sol-to-std"; }

protected:
  solidity::mlirgen::Target tgt = solidity::mlirgen::Target::Undefined;
  Pass::Option<solidity::mlirgen::Target> clTgt{
      *this, "target", llvm::cl::desc("Target for the sol lowering"),
      llvm::cl::init(solidity::mlirgen::Target::Undefined),
      llvm::cl::values(clEnumValN(solidity::mlirgen::Target::EraVM, "eravm",
                                  "EraVM target"))};
};

} // namespace

std::unique_ptr<Pass> sol::createConvertSolToStandardPass() {
  return std::make_unique<ConvertSolToStandard>();
}

std::unique_ptr<Pass>
sol::createConvertSolToStandardPass(solidity::mlirgen::Target tgt) {
  return std::make_unique<ConvertSolToStandard>(tgt);
}
