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

#include "libsolidity/codegen/mlir/Target/EraVM/SolToStandard.h"
#include "libsolidity/codegen/CompilerUtils.h"
#include "libsolidity/codegen/mlir/Interface.h"
#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolidity/codegen/mlir/Target/EVM/SolToStandard.h"
#include "libsolidity/codegen/mlir/Target/EraVM/Util.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolutil/ErrorCodes.h"
#include "libsolutil/FixedHash.h"
#include "libsolutil/FunctionSelector.h"
#include "libsolutil/Keccak256.h"
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
#include "llvm/ADT/STLExtras.h"
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
    r.replaceOp(op, adaptor.getInp());
    return success();
  }
};

// TODO? Move simple builtin lowering to tblgen (`Pat` records)?

struct Keccak256OpLowering : public OpRewritePattern<sol::Keccak256Op> {
  using OpRewritePattern<sol::Keccak256Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::Keccak256Op op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
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
    solidity::mlirgen::BuilderExt bExt(rewriter, loc);
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
    solidity::mlirgen::BuilderExt bExt(rewriter, loc);
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
    solidity::mlirgen::BuilderExt bExt(rewriter, loc);
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
    eravm::Builder eraB(rewriter, loc);

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
    eravm::Builder eraB(r, loc);
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
  void runStage1Conversion(ModuleOp mod, evm::SolTypeConverter &tyConv) {
    OpBuilder b(mod.getContext());
    eravm::Builder eraB(b);

    // FIXME: DialectConversion complains "pattern was already applied" if we do
    // this in the sol.func lowering (might work if we generate a llvm/func.func
    // op instead? Should we switch all such external functions to
    // llvm/func.func?)
    eraB.getOrInsertPersonality(mod);

    ConversionTarget convTgt(getContext());
    convTgt.addLegalOp<mlir::ModuleOp>();
    convTgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                            arith::ArithDialect, LLVM::LLVMDialect>();
    convTgt.addIllegalOp<sol::ConstantOp, sol::ExtOp, sol::TruncOp, sol::AddOp,
                         sol::SubOp, sol::MulOp, sol::CmpOp, sol::CAddOp,
                         sol::CSubOp, sol::AllocaOp, sol::MallocOp,
                         sol::AddrOfOp, sol::GepOp, sol::MapOp, sol::CopyOp,
                         sol::DataLocCastOp, sol::LoadOp, sol::StoreOp,
                         sol::EmitOp, sol::RequireOp, sol::ConvCastOp>();
    convTgt.addDynamicallyLegalOp<sol::FuncOp>([&](sol::FuncOp op) {
      return tyConv.isSignatureLegal(op.getFunctionType());
    });
    convTgt.addDynamicallyLegalOp<sol::CallOp, sol::ReturnOp>(
        [&](Operation *op) { return tyConv.isLegal(op); });

    RewritePatternSet pats(&getContext());
    pats.add<ConvCastOpLowering>(tyConv, &getContext());
    populateAnyFunctionOpInterfaceTypeConversionPattern(pats, tyConv);
    pats.add<GenericTypeConversion<sol::CallOp>,
             GenericTypeConversion<sol::ReturnOp>>(tyConv, &getContext());

    switch (tgt) {
    case solidity::mlirgen::Target::EVM:
      evm::populateStage1Pats(pats, tyConv);
      break;
    case solidity::mlirgen::Target::EraVM:
      eravm::populateStage1Pats(pats, tyConv);
      break;
    default:
      llvm_unreachable("Invalid target");
    };

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

    if (failed(applyPartialConversion(mod, convTgt, std::move(pats))))
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
    evm::populateContrPat(pats);
    pats.add<ObjectOpLowering, BuiltinRetOpLowering, RevertOpLowering,
             MLoadOpLowering, MStoreOpLowering, MCopyOpLowering,
             DataOffsetOpLowering, DataSizeOpLowering, CodeSizeOpLowering,
             CodeCopyOpLowering, MemGuardOpLowering, CallValOpLowering,
             CallDataLoadOpLowering, CallDataSizeOpLowering,
             CallDataCopyOpLowering, SLoadOpLowering, SStoreOpLowering,
             Keccak256OpLowering, LogOpLowering, CallerOpLowering>(
        &getContext());

    if (failed(applyPartialConversion(mod, tgt, std::move(pats))))
      signalPassFailure();
  }

  /// Converts sol.func and related ops.
  void runStage3Conversion(ModuleOp mod, evm::SolTypeConverter &tyConv) {
    ConversionTarget tgt(getContext());
    tgt.addLegalOp<mlir::ModuleOp>();
    tgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                        arith::ArithDialect, LLVM::LLVMDialect>();
    tgt.addIllegalDialect<sol::SolDialect>();

    RewritePatternSet pats(&getContext());
    evm::populateFuncPats(pats, tyConv);

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
    evm::SolTypeConverter tyConv;
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
      llvm::cl::values(
          clEnumValN(solidity::mlirgen::Target::EVM, "evm", "EVM target"),
          clEnumValN(solidity::mlirgen::Target::EraVM, "eravm",
                     "EraVM target"))};
};

} // namespace

// TODO: Move this out of EraVM/
std::unique_ptr<Pass> sol::createConvertSolToStandardPass() {
  return std::make_unique<ConvertSolToStandard>();
}

std::unique_ptr<Pass>
sol::createConvertSolToStandardPass(solidity::mlirgen::Target tgt) {
  return std::make_unique<ConvertSolToStandard>(tgt);
}

void eravm::populateStage1Pats(RewritePatternSet &pats, TypeConverter &tyConv) {
  evm::populateArithPats(pats, tyConv);
  // TODO: Generate the overflow version of arith ops instead.
  evm::populateCheckedArithPats(pats, tyConv);
  evm::populateMemPats(pats, tyConv);
  evm::populateFuncPats(pats, tyConv);
  evm::populateEmitPat(pats, tyConv);
  evm::populateRequirePat(pats);
  evm::populateContrPat(pats);
}
