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

#include "libsolidity/codegen/mlir/Target/EVM/SolToYul.h"
#include "libsolidity/codegen/CompilerUtils.h"
#include "libsolidity/codegen/mlir/Sol/Sol.h"
#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolidity/codegen/mlir/Target/EVM/YulToStandard.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolidity/codegen/mlir/Yul/Yul.h"
#include "libsolutil/FunctionSelector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {

struct ConstantOpLowering : public OpConversionPattern<sol::ConstantOp> {
  using OpConversionPattern<sol::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto signlessTy =
        r.getIntegerType(cast<IntegerType>(op.getType()).getWidth());
    auto attr = cast<IntegerAttr>(op.getValue());
    r.replaceOpWithNewOp<arith::ConstantOp>(
        op, signlessTy, r.getIntegerAttr(signlessTy, attr.getValue()));
    return success();
  }
};

struct CastOpLowering : public OpConversionPattern<sol::CastOp> {
  using OpConversionPattern<sol::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto inpTy = cast<IntegerType>(op.getInp().getType());
    auto inpTyWidth = inpTy.getWidth();
    auto outTyWidth = cast<IntegerType>(op.getType()).getWidth();

    if (inpTyWidth == outTyWidth) {
      r.replaceOp(op, adaptor.getInp());
      return success();
    }

    IntegerType signlessOutTy = r.getIntegerType(outTyWidth);

    if (inpTyWidth > outTyWidth) {
      r.replaceOpWithNewOp<arith::TruncIOp>(op, signlessOutTy,
                                            adaptor.getInp());
      return success();
    }
    if (inpTy.isSigned())
      r.replaceOpWithNewOp<arith::ExtSIOp>(op, signlessOutTy, adaptor.getInp());
    else
      r.replaceOpWithNewOp<arith::ExtUIOp>(op, signlessOutTy, adaptor.getInp());
    return success();
  }
};

struct BytesCastOpLowering : public OpConversionPattern<sol::BytesCastOp> {
  using OpConversionPattern<sol::BytesCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::BytesCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);

    // Bytes to int
    if (auto inpBytesTy = dyn_cast<sol::BytesType>(op.getInp().getType())) {
      auto outIntTy = cast<IntegerType>(op.getType());
      auto shiftAmt = bExt.genI256Const(256 - (8 * inpBytesTy.getSize()));
      auto shr = r.create<arith::ShRUIOp>(loc, adaptor.getInp(), shiftAmt);
      auto repl = bExt.genIntCast(outIntTy.getWidth(), /*isSigned=*/false, shr);
      r.replaceOp(op, repl);
      return success();
    }

    // Int to bytes
    assert(isa<IntegerType>(adaptor.getInp().getType()));
    auto outBytesTy = cast<sol::BytesType>(op.getType());
    Value inpAsI256 =
        bExt.genIntCast(/*width=*/256, /*isSigned=*/false, adaptor.getInp());
    auto shiftAmt = bExt.genI256Const(256 - (8 * outBytesTy.getSize()));
    r.replaceOpWithNewOp<arith::ShLIOp>(op, inpAsI256, shiftAmt);

    return success();
  }
};

/// A templatized version of a conversion pattern for lowering add, sub, mul
/// and exp ops.
template <typename SrcOpT, typename DstOpT>
struct ArithBinOpLowering : public OpConversionPattern<SrcOpT> {
  using OpConversionPattern<SrcOpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(SrcOpT op, typename SrcOpT::Adaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<DstOpT>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

/// A templatized version of a conversion pattern for lowering div and mod ops.
template <typename SolOp, typename ArithSignedOp, typename ArithUnsignedOp>
struct DivOrModOpLowering : public OpConversionPattern<SolOp> {
  using OpConversionPattern<SolOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SolOp op, typename SolOp::Adaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto zero = bExt.genConst(0, ty.getWidth());
    auto rhsEqZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, zero);
    evmB.genPanic(solidity::util::PanicCode::DivisionByZero, rhsEqZero);

    if (ty.isSigned())
      r.replaceOpWithNewOp<ArithSignedOp>(op, lhs, rhs);
    else
      r.replaceOpWithNewOp<ArithUnsignedOp>(op, lhs, rhs);
    return success();
  }
};

struct CAddOpLowering : public OpConversionPattern<sol::CAddOp> {
  using OpConversionPattern<sol::CAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Unlike via-ir, small int (< i256) arithmetic is "legalized" by the llvm
    // backend, so we don't need a different codegen for its overflow/underflow
    // check since the legalized arithmetic works as if the small int is native
    // to evm.

    Value sum = r.create<arith::AddIOp>(loc, lhs, rhs);

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // overflow, if y >= 0 and sum < x
      // underflow, if y < 0 and sum >= x
      //
      // We compare rhs with zero since the canonicalizer could make the rhs a
      // constant which would enable the arith dialect to optimize away the
      // comparison.

      auto zero = bExt.genConst(0, ty.getWidth());

      // Generate the overflow condition.
      auto rhsGtEqZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, rhs, zero);
      auto sumLtLhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, sum, lhs);
      auto overflowCond = r.create<arith::AndIOp>(loc, rhsGtEqZero, sumLtLhs);

      // Generate the underflow condition.
      auto rhsLtZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rhs, zero);
      auto sumGtEqLhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, sum, lhs);
      auto underflowCond = r.create<arith::AndIOp>(loc, rhsLtZero, sumGtEqLhs);

      evmB.genPanic(solidity::util::PanicCode::UnderOverflow,
                    r.create<arith::OrIOp>(loc, overflowCond, underflowCond));

      // Unsigned case
    } else {
      evmB.genPanic(
          solidity::util::PanicCode::UnderOverflow,
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, lhs, sum));
    }

    r.replaceOp(op, sum);
    return success();
  }
};

struct CSubOpLowering : public OpConversionPattern<sol::CSubOp> {
  using OpConversionPattern<sol::CSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    // See comments in sol.cadd lowering on why we don't have a different
    // codegen for small ints.

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value diff = r.create<arith::SubIOp>(loc, lhs, rhs);

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if y >= 0 and diff > x
      // overflow, if y < 0 and diff < x

      auto zero = bExt.genConst(0, ty.getWidth());

      // Generate the overflow condition.
      auto rhsGtEqZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, rhs, zero);
      auto diffGtLhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, diff, lhs);
      auto overflowCond = r.create<arith::AndIOp>(loc, rhsGtEqZero, diffGtLhs);

      // Generate the underflow condition.
      auto rhsLtZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rhs, zero);
      auto diffLtRhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, diff, lhs);
      auto underflowCond = r.create<arith::AndIOp>(loc, rhsLtZero, diffLtRhs);

      evmB.genPanic(solidity::util::PanicCode::UnderOverflow,
                    r.create<arith::OrIOp>(loc, overflowCond, underflowCond));

      // Unsigned case
    } else {
      evmB.genPanic(
          solidity::util::PanicCode::UnderOverflow,
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, diff, lhs));
    }

    r.replaceOp(op, diff);
    return success();
  }
};

struct CMulOpLowering : public OpConversionPattern<sol::CMulOp> {
  using OpConversionPattern<sol::CMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // See comments in sol.cadd lowering on why we don't have a different
    // codegen for small ints.

    Value product = r.create<arith::MulIOp>(loc, lhs, rhs);
    auto zero = bExt.genConst(0, ty.getWidth());
    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if x < 0 and y == int.min
      auto lhsLtZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, zero);
      auto minVal = bExt.genConst(APInt::getSignedMinValue(ty.getWidth()));
      auto rhsEqMin =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, minVal);
      evmB.genPanic(solidity::util::PanicCode::UnderOverflow,
                    r.create<arith::AndIOp>(loc, lhsLtZero, rhsEqMin));

      // over/underflow, if x != 0 and product/x != y
      auto lhsNeqZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, zero);
      auto quotient = r.create<arith::DivSIOp>(loc, product, lhs);
      auto quotientNeqRhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, quotient, rhs);
      evmB.genPanic(solidity::util::PanicCode::UnderOverflow,
                    r.create<arith::AndIOp>(loc, lhsNeqZero, quotientNeqRhs));

      // Unsigned case
    } else {
      // over/underflow, if x != 0 and product/x != y
      auto lhsNeqZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, zero);
      auto quotient = r.create<arith::DivUIOp>(loc, product, lhs);
      auto quotientNeqRhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, quotient, rhs);
      evmB.genPanic(solidity::util::PanicCode::UnderOverflow,
                    r.create<arith::AndIOp>(loc, lhsNeqZero, quotientNeqRhs));
    }

    r.replaceOp(op, product);
    return success();
  }
};

struct CDivOpLowering : public OpConversionPattern<sol::CDivOp> {
  using OpConversionPattern<sol::CDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // See comments in sol.cadd lowering on why we don't have a different
    // codegen for small ints.

    auto zero = bExt.genConst(0, ty.getWidth());
    auto rhsEqZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, zero);
    evmB.genPanic(solidity::util::PanicCode::DivisionByZero, rhsEqZero);

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if x == int.min and y == -1
      auto minVal = bExt.genConst(APInt::getSignedMinValue(ty.getWidth()));
      auto lhsEqMin =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, minVal);
      auto minusOne = bExt.genConst(-1, ty.getWidth());
      auto rhsEqMinusOne =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, minusOne);
      evmB.genPanic(solidity::util::PanicCode::UnderOverflow,
                    r.create<arith::AndIOp>(loc, lhsEqMin, rhsEqMinusOne));
      r.replaceOpWithNewOp<arith::DivSIOp>(op, lhs, rhs);

    } else {
      r.replaceOpWithNewOp<arith::DivUIOp>(op, lhs, rhs);
    }

    return success();
  }
};

struct CmpOpLowering : public OpConversionPattern<sol::CmpOp> {
  using OpConversionPattern<sol::CmpOp>::OpConversionPattern;

  arith::CmpIPredicate getSignlessPred(sol::CmpPredicate pred,
                                       bool isSigned) const {
    // Sign insensitive predicates.
    switch (pred) {
    case sol::CmpPredicate::eq:
      return arith::CmpIPredicate::eq;
    case sol::CmpPredicate::ne:
      return arith::CmpIPredicate::ne;
    default:
      break;
    }

    // Sign sensitive predicates.
    if (isSigned) {
      switch (pred) {
      case sol::CmpPredicate::lt:
        return arith::CmpIPredicate::slt;
      case sol::CmpPredicate::le:
        return arith::CmpIPredicate::sle;
      case sol::CmpPredicate::gt:
        return arith::CmpIPredicate::sgt;
      case sol::CmpPredicate::ge:
        return arith::CmpIPredicate::sge;
      default:
        break;
      }
    } else {
      switch (pred) {
      case sol::CmpPredicate::lt:
        return arith::CmpIPredicate::ult;
      case sol::CmpPredicate::le:
        return arith::CmpIPredicate::ule;
      case sol::CmpPredicate::gt:
        return arith::CmpIPredicate::ugt;
      case sol::CmpPredicate::ge:
        return arith::CmpIPredicate::uge;
      default:
        break;
      }
    }
    llvm_unreachable("Invalid predicate");
  }

  LogicalResult matchAndRewrite(sol::CmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    arith::CmpIPredicate signlessPred = getSignlessPred(
        op.getPredicate(), cast<IntegerType>(op.getLhs().getType()).isSigned());
    r.replaceOpWithNewOp<arith::CmpIOp>(op, signlessPred, adaptor.getLhs(),
                                        adaptor.getRhs());
    return success();
  }
};

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
    r.replaceOpWithNewOp<LLVM::AllocaOp>(
        op, /*resTy=*/LLVM::LLVMPointerType::get(r.getContext()),
        /*eltTy=*/convertedEltTy, bExt.genI256Const(size));
    return success();
  }
};

struct MallocOpLowering : public OpConversionPattern<sol::MallocOp> {
  using OpConversionPattern<sol::MallocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::MallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());
    r.replaceOp(op, evmB.genMemAlloc(op.getType(), op.getZeroInit(), {},
                                     adaptor.getSize()));
    return success();
  }
};

struct ArrayLitOpLowering : public OpConversionPattern<sol::ArrayLitOp> {
  using OpConversionPattern<sol::ArrayLitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ArrayLitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    r.replaceOp(op,
                evmB.genMemAlloc(op.getType(), false, adaptor.getIns(), {}));
    return success();
  }
};

struct GetCallDataOpLowering : public OpConversionPattern<sol::GetCallDataOp> {
  using OpConversionPattern<sol::GetCallDataOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::GetCallDataOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    solidity::mlirgen::BuilderExt bExt(r, op.getLoc());
    r.replaceOp(op, bExt.genI256Const(0));
    return success();
  }
};

struct PushOpLowering : public OpConversionPattern<sol::PushOp> {
  using OpConversionPattern<sol::PushOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::PushOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    solidity::mlirgen::BuilderExt bExt(r, loc);

    Value slot = adaptor.getInp();
    Value size = r.create<yul::SLoadOp>(loc, slot);
    Value newSize = r.create<arith::AddIOp>(loc, size, bExt.genI256Const(1));
    r.create<yul::SStoreOp>(loc, slot, newSize);
    Value dataSlot = evmB.genDataAddrPtr(slot, sol::DataLocation::Storage);

    r.replaceOp(op, r.create<arith::AddIOp>(loc, dataSlot, size));
    return success();
  }
};

struct PopOpLowering : public OpConversionPattern<sol::PopOp> {
  using OpConversionPattern<sol::PopOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::PopOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    solidity::mlirgen::BuilderExt bExt(r, loc);

    Value slot = adaptor.getInp();
    Value oldSize = r.create<yul::SLoadOp>(loc, slot);

    // Generate the empty array panic check.
    Value panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              oldSize, bExt.genI256Const(0));
    evmB.genPanic(solidity::util::PanicCode::EmptyArrayPop, panicCond);

    Value newSize = r.create<arith::SubIOp>(loc, oldSize, bExt.genI256Const(1));
    r.create<yul::SStoreOp>(loc, slot, newSize);
    Value dataSlot = evmB.genDataAddrPtr(slot, sol::DataLocation::Storage);

    // Zero the deleted slot.
    Value tailAddr = r.create<arith::AddIOp>(loc, dataSlot, newSize);
    r.create<yul::SStoreOp>(loc, tailAddr, bExt.genI256Const(0));

    r.eraseOp(op);
    return success();
  }
};

struct AddrOfOpLowering : public OpRewritePattern<sol::AddrOfOp> {
  using OpRewritePattern<sol::AddrOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::AddrOfOp op,
                                PatternRewriter &r) const override {
    solidity::mlirgen::BuilderExt bExt(r, op.getLoc());

    auto parentContract = op->getParentOfType<sol::ContractOp>();
    auto *sym = parentContract.lookupSymbol(op.getVar());
    assert(sym);
    if (auto stateVarOp = dyn_cast<sol::StateVarOp>(sym)) {
      assert(stateVarOp->hasAttr("slot"));
      IntegerAttr slot = cast<IntegerAttr>(stateVarOp->getAttr("slot"));
      r.replaceOp(op, bExt.genI256Const(slot.getValue()));
      return success();
    }
    auto immOp = cast<sol::ImmutableOp>(sym);
    assert(immOp->hasAttr("addr"));
    IntegerAttr addr = cast<IntegerAttr>(immOp->getAttr("addr"));
    r.replaceOp(op, bExt.genI256Const(addr.getValue()));
    return success();
  }
};

struct GepOpLowering : public OpConversionPattern<sol::GepOp> {
  using OpConversionPattern<sol::GepOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::GepOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type baseAddrTy = op.getBaseAddr().getType();
    Value remappedBaseAddr = adaptor.getBaseAddr();
    Value idx = adaptor.getIdx();
    sol::DataLocation dataLoc = sol::getDataLocation(baseAddrTy);
    Value res;

    switch (dataLoc) {
    case sol::DataLocation::Stack: {
      auto stkPtrTy =
          LLVM::LLVMPointerType::get(r.getContext(), evm::AddrSpace_Stack);
      res = r.create<LLVM::GEPOp>(loc, /*resultType=*/stkPtrTy,
                                  /*basePtrType=*/remappedBaseAddr.getType(),
                                  remappedBaseAddr, idx);
      break;
    }

    case sol::DataLocation::CallData:
    case sol::DataLocation::Memory:
    case sol::DataLocation::Storage: {
      // Memory/calldata/storage array
      if (auto arrTy = dyn_cast<sol::ArrayType>(baseAddrTy)) {
        Value addrAtIdx;

        Type eltTy = arrTy.getEltType();
        (void)eltTy;
        assert((isa<IntegerType>(eltTy) || sol::isNonPtrRefType(eltTy)) &&
               "NYI");

        // Don't generate out-of-bounds check for constant indexing of static
        // arrays.
        if (!isa<BlockArgument>(idx) &&
            isa<arith::ConstantIntOp>(idx.getDefiningOp())) {
          auto constIdx = cast<arith::ConstantIntOp>(idx.getDefiningOp());
          if (!arrTy.isDynSized()) {
            // FIXME: Should this be done by the verifier?
            assert(constIdx.value() < arrTy.getSize());
            unsigned stride = dataLoc == sol::DataLocation::Storage
                                  ? evm::getStorageSlotCount(eltTy)
                                  : 32;
            addrAtIdx = r.create<arith::AddIOp>(
                loc, remappedBaseAddr,
                bExt.genI256Const(constIdx.value() * stride));
          }
        }

        if (!addrAtIdx) {
          //
          // Generate PanicCode::ArrayOutOfBounds check.
          //
          Value size;
          if (arrTy.isDynSized())
            size = evmB.genLoad(remappedBaseAddr, dataLoc);
          else
            size = bExt.genI256Const(arrTy.getSize());

          // Generate `if iszero(lt(index, <arrayLen>(baseRef)))` (yul).
          auto idxTy = cast<IntegerType>(idx.getType());
          Value castedIdx = bExt.genIntCast(256, idxTy.isSigned(), idx);
          auto panicCond = r.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::uge, castedIdx, size);
          evmB.genPanic(solidity::util::PanicCode::ArrayOutOfBounds, panicCond);

          //
          // Generate the address.
          //
          Value stride =
              dataLoc == sol::DataLocation::Storage
                  ? bExt.genI256Const(evm::getStorageSlotCount(eltTy))
                  : bExt.genI256Const(32);
          Value scaledIdx = r.create<arith::MulIOp>(loc, castedIdx, stride);
          if (arrTy.isDynSized()) {
            Value dataAddr = evmB.genDataAddrPtr(remappedBaseAddr, dataLoc);
            addrAtIdx = r.create<arith::AddIOp>(loc, dataAddr, scaledIdx);
          } else {
            addrAtIdx =
                r.create<arith::AddIOp>(loc, remappedBaseAddr, scaledIdx);
          }
        }
        assert(addrAtIdx);

        res = addrAtIdx;

        // Memory struct
      } else if (auto structTy = dyn_cast<sol::StructType>(baseAddrTy)) {
#ifndef NDEBUG
        for (Type ty : structTy.getMemberTypes())
          assert(isa<IntegerType>(ty) || sol::isNonPtrRefType(ty) && "NYI");
#endif

        auto idxConstOp = cast<arith::ConstantIntOp>(idx.getDefiningOp());
        Value memberIdx =
            bExt.genIntCast(/*width=*/256, /*isSigned=*/false, idxConstOp);
        auto scaledIdx =
            r.create<arith::MulIOp>(loc, memberIdx, bExt.genI256Const(32));
        res = r.create<arith::AddIOp>(loc, remappedBaseAddr, scaledIdx);

        // Bytes (!sol.string)
      } else if (auto strTy = dyn_cast<sol::StringType>(baseAddrTy)) {
        Value size = evmB.genLoad(remappedBaseAddr, dataLoc);
        Value castedIdx =
            bExt.genIntCast(/*width=*/256, /*isSigned=*/false, idx);
        auto panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                 castedIdx, size);
        evmB.genPanic(solidity::util::PanicCode::ArrayOutOfBounds, panicCond);

        // Generate the address after the length-slot.
        Value dataAddr = r.create<arith::AddIOp>(loc, remappedBaseAddr,
                                                 bExt.genI256Const(32));
        res = r.create<arith::AddIOp>(loc, dataAddr, castedIdx);
      }

      assert(res);
      break;
    }

    default:
      llvm_unreachable("NYI");
      break;
    }

    r.replaceOp(op, res);
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
    assert(isa<IntegerType>(op.getKey().getType()) && "NYI");
    auto key = bExt.genIntCast(
        /*width=*/256, cast<IntegerType>(op.getKey().getType()).isSigned(),
        adaptor.getKey());
    r.create<yul::MStoreOp>(loc, zero, key);
    r.create<yul::MStoreOp>(loc, bExt.genI256Const(0x20), adaptor.getMapping());

    r.replaceOpWithNewOp<yul::Keccak256Op>(op, zero, bExt.genI256Const(0x40));
    return success();
  }
};

struct LoadOpLowering : public OpConversionPattern<sol::LoadOp> {
  using OpConversionPattern<sol::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value addr = adaptor.getAddr();
    sol::DataLocation dataLoc = sol::getDataLocation(op.getAddr().getType());

    switch (dataLoc) {
    case sol::DataLocation::Stack: {
      Type eltTy =
          cast<sol::PointerType>(op.getAddr().getType()).getPointeeType();
      Type loadTy = getTypeConverter()->convertType(eltTy);
      assert(loadTy);
      r.replaceOpWithNewOp<LLVM::LoadOp>(op, loadTy, addr,
                                         evm::getAlignment(addr));
      return success();
    }
    default: {
      auto addrTy = cast<sol::PointerType>(op.getAddr().getType());
      auto bytesEleTy = dyn_cast<sol::BytesType>(addrTy.getPointeeType());
      // If loading from `bytes`, generate the low bits mask-off of the loaded
      // value.
      if (bytesEleTy && dataLoc == sol::DataLocation::Memory) {
        unsigned numBits = bytesEleTy.getSize() * 8;
        APInt mask(/*numBits=*/256, 0);
        assert(numBits <= 256);
        mask.setHighBits(numBits);
        auto load = evmB.genLoad(addr, dataLoc);
        r.replaceOpWithNewOp<arith::AndIOp>(op, load, bExt.genI256Const(mask));
        return success();
      }

      auto ld = evmB.genLoad(addr, dataLoc);
      if (auto intTy = dyn_cast<IntegerType>(op.getType())) {
        Value castedRes =
            bExt.genIntCast(intTy.getWidth(), intTy.isSigned(), ld);
        r.replaceOp(op, castedRes);
        return success();
      }
      r.replaceOp(op, ld);
      return success();
    }
    };
  }
};

struct LoadImmutableOpLowering
    : public OpConversionPattern<sol::LoadImmutableOp> {
  using OpConversionPattern<sol::LoadImmutableOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LoadImmutableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    // Track the "addr" attribute from the referenced sol.immutable op.
    SmallVector<NamedAttribute> attrs = llvm::to_vector(op->getAttrs());
    auto parentContract = op->getParentOfType<sol::ContractOp>();
    Operation *sym = parentContract.lookupSymbol(op.getName());
    auto immOp = cast<sol::ImmutableOp>(sym);
    attrs.push_back(
        r.getNamedAttr("addr", immOp->getAttrOfType<IntegerAttr>("addr")));

    // Legalize result types and add the "addr" attribute.
    SmallVector<Type> newResTys;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      newResTys)))
      return failure();
    r.replaceOpWithNewOp<sol::LoadImmutableOp>(op, newResTys,
                                               adaptor.getOperands(), attrs);
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<sol::StoreOp> {
  using OpConversionPattern<sol::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value remappedVal = adaptor.getVal();
    Value remappedAddr = adaptor.getAddr();

    switch (sol::getDataLocation(op.getAddr().getType())) {
    case sol::DataLocation::Stack:
      r.replaceOpWithNewOp<LLVM::StoreOp>(op, remappedVal, remappedAddr,
                                          evm::getAlignment(remappedAddr));
      return success();
    case sol::DataLocation::Immutable:
    case sol::DataLocation::Memory: {
      Type addrTy = op.getAddr().getType();
      sol::DataLocation dataLoc = sol::getDataLocation(addrTy);

      // Generate mstore8 for storing to `bytes`.
      auto bytesEleTy = dyn_cast<sol::BytesType>(sol::getEltType(addrTy));
      if (bytesEleTy && dataLoc == sol::DataLocation::Memory) {
        assert(bytesEleTy.getSize() == 1 && "NYI");
        auto byteVal =
            r.create<yul::ByteOp>(loc, bExt.genI256Const(0), remappedVal);
        r.replaceOpWithNewOp<yul::MStore8Op>(op, remappedAddr, byteVal);
        return success();
      }

      if (auto intTy = dyn_cast<IntegerType>(op.getVal().getType())) {
        Value castedVal =
            bExt.genIntCast(/*width=*/256, intTy.isSigned(), remappedVal);
        evmB.genStore(castedVal, remappedAddr, dataLoc);
        r.eraseOp(op);
        return success();
      }
      evmB.genStore(remappedVal, remappedAddr, dataLoc);
      r.eraseOp(op);
      return success();
    }
    case sol::DataLocation::Storage:
      r.replaceOpWithNewOp<yul::SStoreOp>(op, remappedAddr, remappedVal);
      return success();
    default:
      break;
    };

    llvm_unreachable("NYI: Calldata data-location");
  }
};

struct DataLocCastOpLowering : public OpConversionPattern<sol::DataLocCastOp> {
  using OpConversionPattern<sol::DataLocCastOp>::OpConversionPattern;

  Value genCopy(Value srcAddr, Type ty, PatternRewriter &r,
                Location loc) const {
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    sol::DataLocation srcDataLoc = sol::DataLocation::Storage;
    sol::DataLocation dstDataLoc = sol::DataLocation::Memory;
    assert((srcDataLoc == sol::DataLocation::Storage &&
            dstDataLoc == sol::DataLocation::Memory) &&
           "FIXME");

    if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
      Value size, dstAddr, dstDataAddr, srcDataAddr;
      if (arrTy.isDynSized()) {
        size = evmB.genLoad(srcAddr, srcDataLoc);
        dstAddr = evmB.genMemAllocForDynArray(
            size, r.create<arith::MulIOp>(loc, size, bExt.genI256Const(32)));
        dstDataAddr = evmB.genDataAddrPtr(dstAddr, dstDataLoc);
        srcDataAddr = evmB.genDataAddrPtr(srcAddr, srcDataLoc);
      } else {
        size = bExt.genI256Const(arrTy.getSize());
        dstAddr = evmB.genMemAlloc(arrTy.getSize() * 32);
        dstDataAddr = dstAddr;
        srcDataAddr = srcAddr;
      }
      r.create<scf::ForOp>(
          loc, /*lowerBound=*/bExt.genIdxConst(0),
          /*upperBound=*/bExt.genCastToIdx(size),
          /*step=*/bExt.genIdxConst(1),
          /*iterArgs=*/std::nullopt,
          /*builder=*/
          [&](OpBuilder &b, Location loc, Value indVar, ValueRange iterArgs) {
            Value i256IndVar = bExt.genCastToI256(indVar);
            Value srcAddrI = evmB.genAddrAtIdx(srcDataAddr, i256IndVar, arrTy,
                                               srcDataLoc, loc);
            Value dstAddrI = evmB.genAddrAtIdx(dstDataAddr, i256IndVar, arrTy,
                                               dstDataLoc, loc);
            evmB.genStore(genCopy(srcAddrI, arrTy.getEltType(), r, loc),
                          dstAddrI, dstDataLoc);
            r.create<scf::YieldOp>(loc);
          });
      return dstAddr;
    }

    assert(isa<IntegerType>(ty) && "NYI");
    return evmB.genLoad(srcAddr, srcDataLoc);
  }

  LogicalResult matchAndRewrite(sol::DataLocCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type srcTy = op.getInp().getType();
    Type dstTy = op.getType();
    sol::DataLocation srcDataLoc = sol::getDataLocation(srcTy);
    sol::DataLocation dstDataLoc = sol::getDataLocation(dstTy);

    mlir::Value resAddr;
    // From storage to memory.
    if (srcDataLoc == sol::DataLocation::Storage &&
        dstDataLoc == sol::DataLocation::Memory) {
      // String type
      if (isa<sol::StringType>(srcTy)) {
        auto dataSlot = evmB.genDataAddrPtr(adaptor.getInp(), srcDataLoc);

        // Generate the memory allocation.
        auto sizeInBytes = evmB.genLoad(adaptor.getInp(), srcDataLoc);
        Value memAddr = evmB.genMemAllocForDynArray(
            sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes));
        resAddr = memAddr;

        // Generate the loop to copy the data.
        auto dataMemAddr =
            r.create<arith::AddIOp>(loc, memAddr, bExt.genI256Const(32));
        auto sizeInWords = bExt.genRoundUpToMultiple<32>(sizeInBytes);
        evmB.genCopyLoop(dataSlot, dataMemAddr, sizeInWords, srcTy, dstTy,
                         srcDataLoc, dstDataLoc);

        r.replaceOp(op, memAddr);
        return success();
      }

      if (isa<sol::ArrayType>(srcTy)) {
        r.replaceOp(op, genCopy(adaptor.getInp(), srcTy, r, loc));
        return success();
      }
    }

    llvm_unreachable("NYI");
    return failure();
  }
};

struct LengthOpLowering : public OpConversionPattern<sol::LengthOp> {
  using OpConversionPattern<sol::LengthOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LengthOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type ty = op.getInp().getType();
    sol::DataLocation dataLoc = sol::getDataLocation(ty);

    if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
      r.replaceOp(op, evmB.genLoad(adaptor.getInp(), dataLoc));
      return success();
    }
    if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
      if (arrTy.isDynSized()) {
        r.replaceOp(op, evmB.genLoad(adaptor.getInp(), dataLoc));
        return success();
      }
      r.replaceOp(op, bExt.genI256Const(arrTy.getSize()));
      return success();
    }
    llvm_unreachable("NYI");
  }
};

struct CopyOpLowering : public OpConversionPattern<sol::CopyOp> {
  using OpConversionPattern<sol::CopyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();
    sol::DataLocation srcDataLoc = sol::getDataLocation(srcTy);
    sol::DataLocation dstDataLoc = sol::getDataLocation(dstTy);
    assert(srcDataLoc == sol::DataLocation::Memory &&
           dstDataLoc == sol::DataLocation::Storage && "NYI");

    if (isa<sol::StringType>(srcTy)) {
      assert(isa<sol::StringType>(dstTy));

      // Generate the size update.
      Value srcSize = evmB.genLoad(adaptor.getSrc(), srcDataLoc);
      evmB.genStore(srcSize, adaptor.getDst(), dstDataLoc);

      // Generate the copy loop.
      Value srcDataAddr = evmB.genDataAddrPtr(adaptor.getSrc(), srcDataLoc);
      Value dstDataAddr = evmB.genDataAddrPtr(adaptor.getDst(), dstDataLoc);
      Value sizeInWords = bExt.genRoundUpToMultiple<32>(srcSize);
      evmB.genCopyLoop(srcDataAddr, dstDataAddr, sizeInWords, srcTy, dstTy,
                       srcDataLoc, dstDataLoc);
    } else {
      llvm_unreachable("NYI");
    }

    r.eraseOp(op);
    return success();
  }
};

struct ThisOpLowering : public OpRewritePattern<sol::ThisOp> {
  using OpRewritePattern<sol::ThisOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::ThisOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::AddressOp>(op);
    return success();
  }
};

struct LibAddrOpLowering : public OpConversionPattern<sol::LibAddrOp> {
  using OpConversionPattern<sol::LibAddrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LibAddrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::LinkerSymbolOp>(op, op.getName());
    return success();
  }
};

struct EncodeOpLowering : public OpConversionPattern<sol::EncodeOp> {
  using OpConversionPattern<sol::EncodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::EncodeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value freePtr = evmB.genFreePtr();
    Value tupleStart =
        r.create<arith::AddIOp>(loc, freePtr, bExt.genI256Const(32));
    Value tupleEnd = evmB.genABITupleEncoding(
        op.getOperandTypes(), adaptor.getOperands(), tupleStart);
    Value tupleSize = r.create<arith::SubIOp>(loc, tupleEnd, tupleStart);
    r.create<yul::MStoreOp>(loc, freePtr, tupleSize);
    Value allocationSize = r.create<arith::SubIOp>(loc, tupleEnd, freePtr);
    evmB.genFreePtrUpd(freePtr, allocationSize);
    r.replaceOp(op, freePtr);
    return success();
  }
};

struct DecodeOpLowering : public OpConversionPattern<sol::DecodeOp> {
  using OpConversionPattern<sol::DecodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::DecodeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    std::vector<Value> results;
    Value tupleSize = r.create<yul::MLoadOp>(loc, adaptor.getAddr());
    Value tupleStart =
        r.create<arith::AddIOp>(loc, adaptor.getAddr(), bExt.genI256Const(32));
    Value tupleEnd = r.create<arith::AddIOp>(loc, tupleStart, tupleSize);
    bool fromMem = sol::getDataLocation(op.getAddr().getType()) ==
                   sol::DataLocation::Memory;
    assert(fromMem && "NYI");
    evmB.genABITupleDecoding(op.getResultTypes(), tupleStart, tupleEnd, results,
                             fromMem);
    r.replaceOp(op, results);
    return success();
  }
};

struct ExtCallOpLowering : public OpConversionPattern<sol::ExtCallOp> {
  using OpConversionPattern<sol::ExtCallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ExtCallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);
    auto mod = op->getParentOfType<ModuleOp>();

    assert(sol::evmCanOverchargeGasForCall(mod) && "NYI");

    // TODO:
    // - The return arg analysis is done for evm's the supports return data (See
    // solidity::frontend::ReturnInfo).
    // - The generated code for the return has returndatacopy and returnsize.
    assert(sol::evmSupportsReturnData(mod) && "NYI");

    // Check if we need to generate the extcodesize check.
    unsigned totHeadSize = 0;
    for (auto resTy : op.getCalleeType().getResults()) {
      totHeadSize += evm::getCallDataHeadSize(resTy);
    }
    // TODO: Do we really need to check revertStrings() >= RevertStrings::Debug
    // here?
    bool extCodeSizeCheck =
        totHeadSize == 0 || !sol::evmSupportsReturnData(mod);

    if (extCodeSizeCheck) {
      // Generate the revert code.
      auto extCodeSize = r.create<yul::ExtCodeSizeOp>(loc, adaptor.getAddr());
      auto isExtCodeSizeZero = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, extCodeSize, bExt.genI256Const(0));
      if (sol::isRevertStringsEnabled(mod))
        evmB.genRevertWithMsg(isExtCodeSizeZero,
                              "Target contract does not contain code");
      else
        evmB.genRevert(isExtCodeSizeZero);
    }

    // Generate the store of the selector.
    Value selectorAddr = evmB.genFreePtr();
    auto shiftedSelector = APInt(/*numBits=*/256, op.getSelector()) << 224;
    r.create<yul::MStoreOp>(loc, selectorAddr,
                            bExt.genI256Const(shiftedSelector));

    // Generate the abi encoding code.
    Value tupleStart =
        r.create<arith::AddIOp>(loc, selectorAddr, bExt.genI256Const(4));
    Value tupleEnd = evmB.genABITupleEncoding(op.getIns().getType(),
                                              adaptor.getIns(), tupleStart);

    // Calculate the return size statically and/or check if it's dynamic. This
    // is copied from solidity::frontend::ReturnInfo.
    unsigned staticRetSizeVal = 0;
    bool isRetSizeDynamic = false;
    for (Type ty : op.getCalleeType().getResults()) {
      if (sol::isDynamicallySized(ty)) {
        isRetSizeDynamic = true;
        staticRetSizeVal = 0;
        break;
      }
      staticRetSizeVal += evm::getCallDataHeadSize(ty);
    }

    // Generate the call.
    Value inpSize = r.create<arith::SubIOp>(loc, tupleEnd, selectorAddr);
    Value staticRetSize = bExt.genI256Const(staticRetSizeVal);
    mlir::Value status;

    // Order is important here, staticcall might overlap with delegatecall.
    if (op.getDelegateCall())
      status = r.create<yul::DelegateCallOp>(
          loc, adaptor.getGas(), adaptor.getAddr(),
          /*inpOffset=*/selectorAddr, inpSize,
          /*outOffset=*/selectorAddr, /*outSize=*/staticRetSize);
    else if (op.getStaticCall())
      status = r.create<yul::StaticCallOp>(
          loc, adaptor.getGas(), adaptor.getAddr(),
          /*inpOffset=*/selectorAddr, inpSize,
          /*outOffset=*/selectorAddr, /*outSize=*/staticRetSize);
    else
      status = r.create<yul::CallOp>(
          loc, adaptor.getGas(), adaptor.getAddr(), adaptor.getVal(),
          /*inpOffset=*/selectorAddr, inpSize,
          /*outOffset=*/selectorAddr, /*outSize=*/staticRetSize);

    // Generate forwarding revert if not try-call.
    if (!op.getTryCall()) {
      auto statusIsZero = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  status, bExt.genI256Const(0));
      evmB.genForwardingRevert(statusIsZero);
    }

    // Get the types of the results from the decoding, which should be the same
    // as the corsp legal types.
    SmallVector<Type> decodedResultTys;
    if (failed(getTypeConverter()->convertTypes(op.getCalleeType().getResults(),
                                                decodedResultTys)))
      return failure();

    // Generate the if-else op of the status that yields the decoded results.
    auto statusIsNotZero = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, status, bExt.genI256Const(0));
    auto statusIfOp = r.create<scf::IfOp>(loc, /*resultTys=*/decodedResultTys,
                                          /*cond=*/statusIsNotZero);

    // Generate the else block (failure) which yields undefs. The undefs will
    // not be used during execution as the either control flow will skip the
    // sol.try's success block or hit the previous forwarding revert.
    r.setInsertionPointToStart(&statusIfOp.getElseRegion().emplaceBlock());
    SmallVector<Value, 2> undefYields;
    for (Type ty : decodedResultTys) {
      undefYields.push_back(r.create<LLVM::UndefOp>(loc, ty));
    }
    r.create<scf::YieldOp>(loc, undefYields);

    // Generte the then block (success).
    r.setInsertionPointToStart(&statusIfOp.getThenRegion().emplaceBlock());

    // The allocation `selectorAddr` will be reused for the return data.

    // Generate the decoding of the results.
    Value retDataSize = r.create<yul::ReturnDataSizeOp>(loc);
    std::vector<Value> decodedResults;
    if (isRetSizeDynamic) {
      r.create<yul::ReturnDataCopyOp>(loc, selectorAddr,
                                      /*src=*/bExt.genI256Const(0),
                                      retDataSize);
      evmB.genFreePtrUpd(selectorAddr, retDataSize);
      Value tupleEnd = r.create<arith::AddIOp>(loc, selectorAddr, retDataSize);
      evmB.genABITupleDecoding(op.getCalleeType().getResults(), selectorAddr,
                               tupleEnd, decodedResults, /*fromMem=*/true);
    } else {
      // See https://github.com/ethereum/solidity/pull/12684
      Value staticRetSizeGreater = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, staticRetSize, retDataSize);

      auto ifOp = r.create<scf::IfOp>(loc, /*resultTy=*/r.getIntegerType(256),
                                      /*cond=*/staticRetSizeGreater,
                                      /*withElse=*/true);
      // Then block:
      r.setInsertionPointToStart(&ifOp.getThenRegion().front());
      evmB.genFreePtrUpd(selectorAddr, retDataSize);
      Value tupleEnd = r.create<arith::AddIOp>(loc, selectorAddr, retDataSize);
      r.create<scf::YieldOp>(loc, tupleEnd);
      // Else block:
      r.setInsertionPointToStart(&ifOp.getElseRegion().front());
      evmB.genFreePtrUpd(selectorAddr, staticRetSize);
      tupleEnd = r.create<arith::AddIOp>(loc, selectorAddr, staticRetSize);
      r.create<scf::YieldOp>(loc, tupleEnd);

      r.setInsertionPointAfter(ifOp);
      evmB.genABITupleDecoding(op.getCalleeType().getResults(), selectorAddr,
                               ifOp.getResult(0), decodedResults,
                               /*fromMem=*/true);
    }
    r.create<scf::YieldOp>(loc, decodedResults);

    // Replace the sol.ext_call op with the status check + the decoded results.
    assert(decodedResults.size() <= 1 && "NYI");
    SmallVector<Value, 2> newResults{statusIsNotZero};
    newResults.append(statusIfOp.getResults().begin(),
                      statusIfOp.getResults().end());
    r.replaceOp(op, newResults);
    return success();
  }
};

struct NewOpLowering : public OpConversionPattern<sol::NewOp> {
  using OpConversionPattern<sol::NewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::NewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value bytecodeAddr = evmB.genFreePtr();
    Value dataOffset = r.create<yul::DataOffsetOp>(loc, op.getObjName());
    Value dataSize = r.create<yul::DataSizeOp>(loc, op.getObjName());
    r.create<yul::CodeCopyOp>(loc, bytecodeAddr, dataOffset, dataSize);

    Value tupleStart = r.create<arith::AddIOp>(loc, bytecodeAddr, dataSize);
    Value tupleEnd = evmB.genABITupleEncoding(
        op.getCtorArgs().getType(), adaptor.getCtorArgs(), tupleStart);
    Value allocSize = r.create<arith::SubIOp>(loc, tupleEnd, bytecodeAddr);

    if (op.getSalt())
      r.replaceOpWithNewOp<yul::Create2Op>(op, adaptor.getVal(), bytecodeAddr,
                                           allocSize, adaptor.getSalt());
    else
      r.replaceOpWithNewOp<yul::CreateOp>(op, adaptor.getVal(), bytecodeAddr,
                                          allocSize);
    return success();
  }
};

struct CodeOpLowering : public OpConversionPattern<sol::CodeOp> {
  using OpConversionPattern<sol::CodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CodeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {

    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto extCodeSize = r.create<yul::ExtCodeSizeOp>(loc, adaptor.getContAddr());
    Value alloc = evmB.genMemAlloc(op.getType(), /*zeroInit=*/false,
                                   /*initVals=*/{}, extCodeSize);
    auto codeAddr = evmB.genDataAddrPtr(alloc, sol::DataLocation::Memory);
    r.create<yul::ExtCodeCopyOp>(
        loc, adaptor.getContAddr(), /*dstOffset=*/codeAddr,
        /*srcOffset=*/bExt.genI256Const(0), extCodeSize);
    r.replaceOp(op, alloc);
    return success();
  }
};

struct TryOpLowering : public OpConversionPattern<sol::TryOp> {
  using OpConversionPattern<sol::TryOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::TryOp tryOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = tryOp.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);

    auto ifStatus = r.create<sol::IfOp>(loc, tryOp.getStatus());

    //
    // Success region
    //

    if (tryOp.getSuccessRegion().empty()) {
      r.setInsertionPointToStart(&tryOp.getSuccessRegion().emplaceBlock());
      r.create<sol::YieldOp>(loc);
    } else {
      r.inlineRegionBefore(tryOp.getSuccessRegion(), ifStatus.getThenRegion(),
                           ifStatus.getThenRegion().begin());
    }

    //
    // Failure region
    //

    if (tryOp.getFallbackRegion().empty()) {
      evm::Builder evmB(r, loc);
      r.setInsertionPointToStart(&ifStatus.getElseRegion().emplaceBlock());
      evmB.genForwardingRevert();
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));
    } else {
      r.inlineRegionBefore(tryOp.getFallbackRegion(), ifStatus.getElseRegion(),
                           ifStatus.getElseRegion().begin());
    }

    if (tryOp.getPanicRegion().empty() && tryOp.getErrorRegion().empty()) {
      r.eraseOp(tryOp);
      return success();
    }

    r.setInsertionPointToStart(&ifStatus.getElseRegion().front());

    // Generate a flag to check if we need to run the fallback. The flag will be
    // set to false by any of the other clause.
    auto boolAllocaTy = sol::PointerType::get(r.getContext(), r.getI1Type(),
                                              sol::DataLocation::Stack);
    auto runFallbackFlag = r.create<sol::AllocaOp>(loc, boolAllocaTy);
    r.create<sol::StoreOp>(loc, bExt.genBool(true), runFallbackFlag);

    // Generate an if op that checks if the returndata is large enough to hold
    // the selector.
    //
    // The ops after this if op belong to the fallback. Track it for the
    // fallback lowering.
    auto returnDataSize = r.create<yul::ReturnDataSizeOp>(loc);
    auto selectorRetCond = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, returnDataSize, bExt.genI256Const(3));
    auto ifSelectorRet = r.create<sol::IfOp>(loc, selectorRetCond);
    auto fallbackPoint = r.saveInsertionPoint().getPoint();

    //
    // Selector region
    //
    r.setInsertionPointToStart(&ifSelectorRet.getThenRegion().emplaceBlock());
    r.setInsertionPoint(r.create<sol::YieldOp>(loc));

    // Generate the selector extraction code.
    auto zero = bExt.genI256Const(0);
    r.create<yul::ReturnDataCopyOp>(loc, /*dst=*/zero, /*src=*/zero,
                                    /*size=*/bExt.genI256Const(4));
    auto selectorWord = r.create<yul::MLoadOp>(loc, zero);
    auto selector =
        r.create<arith::ShRUIOp>(loc, selectorWord, bExt.genI256Const(224));

    // Generate the switch for the panic and error catch blocks.
    SmallVector<APInt, 2> switchSelectors;
    APInt panicSelector(
        /*numBits=*/256,
        solidity::util::selectorFromSignatureU32("Panic(uint256)"));
    APInt errorSelector(
        /*numBits=*/256,
        solidity::util::selectorFromSignatureU32("Error(string)"));
    if (!tryOp.getPanicRegion().empty())
      switchSelectors.push_back(panicSelector);
    if (!tryOp.getErrorRegion().empty())
      switchSelectors.push_back(errorSelector);
    assert(!switchSelectors.empty());
    auto switchSelectorsAttr = mlir::DenseIntElementsAttr::get(
        RankedTensorType::get(static_cast<int64_t>(switchSelectors.size()),
                              r.getIntegerType(256)),
        switchSelectors);
    auto switchOp = r.create<sol::SwitchOp>(loc, selector, switchSelectorsAttr,
                                            switchSelectors.size());
    r.setInsertionPointToStart(&switchOp.getDefaultRegion().emplaceBlock());
    r.setInsertionPoint(r.create<sol::YieldOp>(loc));

    //
    // Panic case region
    //
    if (!tryOp.getPanicRegion().empty()) {
      // FIXME: We should query for the case region using the attribute!
      r.setInsertionPointToStart(&switchOp.getCaseRegions()[0].emplaceBlock());
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));

      // Genereate an if op that checks if the returndata is large enough to
      // hold the panic code.
      auto panicRetCond =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                  returnDataSize, bExt.genI256Const(0x23));
      auto ifPanicRet = r.create<sol::IfOp>(loc, panicRetCond);

      // Inline the panic region.
      r.inlineRegionBefore(tryOp.getPanicRegion(), ifPanicRet.getThenRegion(),
                           ifPanicRet.getThenRegion().begin());
      Block &thenEntry = ifPanicRet.getThenRegion().front();
      r.setInsertionPointToStart(&thenEntry);
      r.create<sol::StoreOp>(loc, bExt.genBool(false), runFallbackFlag);
      // Replace the panic code block arg with the panic code at offset 4.
      BlockArgument blkArg = thenEntry.getArgument(0);
      r.create<yul::ReturnDataCopyOp>(blkArg.getLoc(), /*dst=*/zero,
                                      /*src=*/bExt.genI256Const(4),
                                      /*size=*/bExt.genI256Const(0x20));
      Value panicCode = r.create<yul::MLoadOp>(blkArg.getLoc(), zero);
      auto blkArgRepl = getTypeConverter()->materializeSourceConversion(
          r, loc, blkArg.getType(), panicCode);
      // FIXME: Why does the following cause a "no matched legalization pattern"
      // of the op from the materialization? Is this related to early
      // legalization of in-place modifications?
      // (https://discourse.llvm.org/t/dialect-conversion-fails-if-some-requires-recursive-application/79371/6)
      // r.replaceAllUsesWith(blkArg, blkArgRepl);
      blkArg.replaceAllUsesWith(blkArgRepl);
      thenEntry.eraseArgument(0);
    }

    //
    // Error case region
    //
    if (!tryOp.getErrorRegion().empty()) {
      // FIXME: We should query for the case region using the attribute!
      unsigned errorCaseIdx = switchSelectors.size() == 2 ? 1 : 0;
      r.setInsertionPointToStart(
          &switchOp.getCaseRegions()[errorCaseIdx].emplaceBlock());
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));

      // Genereate an if op that checks if the returndata is large enough to
      // hold the error message.
      auto errMsgRetCond =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                  returnDataSize, bExt.genI256Const(0x44));
      auto ifErrMsg = r.create<sol::IfOp>(loc, errMsgRetCond);

      // Inline the error region.
      r.inlineRegionBefore(tryOp.getErrorRegion(), ifErrMsg.getThenRegion(),
                           ifErrMsg.getThenRegion().begin());
      Block &thenEntry = ifErrMsg.getThenRegion().front();
      r.setInsertionPointToStart(&thenEntry);
      r.create<sol::StoreOp>(loc, bExt.genBool(false), runFallbackFlag);

      // Generate the error message extraction code from the return data and
      // replace the block argument with it.
      //
      // TODO: Is it necessary to generate all the checks in
      // YulUtilFunctions::tryDecodeErrorMessageFunction()?
      BlockArgument blkArg = thenEntry.getArgument(0);
      Location loc = blkArg.getLoc();
      evm::Builder evmB(r, loc);
      Value abiTupleSize =
          r.create<arith::SubIOp>(loc, returnDataSize, bExt.genI256Const(4));
      Value abiTuple = evmB.genMemAlloc(abiTupleSize);
      r.create<yul::ReturnDataCopyOp>(loc, /*dst=*/abiTuple,
                                      /*src=*/bExt.genI256Const(4),
                                      abiTupleSize);
      Value errMsgOffset = r.create<yul::MLoadOp>(loc, abiTuple);
      Value errMsg = r.create<arith::AddIOp>(loc, abiTuple, errMsgOffset);
      auto blkArgRepl = getTypeConverter()->materializeSourceConversion(
          r, loc, blkArg.getType(), errMsg);
      // FIXME: See panic clause lowering.
      blkArg.replaceAllUsesWith(blkArgRepl);
      thenEntry.eraseArgument(0);
    }

    //
    // Fallback region
    //
    Block *fallbackBlk = r.splitBlock(fallbackPoint->getBlock(), fallbackPoint);
    r.setInsertionPointAfter(ifSelectorRet);
    auto runFallbackFlagLd = r.create<sol::LoadOp>(loc, runFallbackFlag);
    auto ifRunFallback = r.create<sol::IfOp>(loc, runFallbackFlagLd);
    ifRunFallback.getThenRegion().emplaceBlock();
    r.inlineBlockBefore(fallbackBlk, &ifRunFallback.getThenRegion().front(),
                        ifRunFallback.getThenRegion().front().begin());
    r.create<sol::YieldOp>(loc);

    r.eraseOp(tryOp);
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
    evm::Builder evmB(r, loc);
    if (!op.getMsg().empty())
      evmB.genRevertWithMsg(negCond, op.getMsg().str());
    else
      evmB.genRevert(negCond);

    r.eraseOp(op);
    return success();
  }
};

struct EmitOpLowering : public OpConversionPattern<sol::EmitOp> {
  using OpConversionPattern<sol::EmitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::EmitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    // Collect the remapped indexed and non-indexed args.
    //
    // FIXME: How do we get `extraClassDeclaration` functions to be part of the
    // OpAdaptor?
    auto remappedOperands = adaptor.getOperands();
    std::vector<Value> indexedArgs, nonIndexedArgs;
    std::vector<Type> nonIndexedArgsType;
    if (op.getSignature()) {
      auto signatureHash =
          solidity::util::h256::Arith(
              solidity::util::keccak256(op.getSignature()->str()))
              .str();
      indexedArgs.push_back(bExt.genI256Const(signatureHash));
    }
    unsigned argIdx = 0;
    while (argIdx < op.getIndexedArgsCount())
      indexedArgs.push_back(remappedOperands[argIdx++]);
    for (Value arg : op.getNonIndexedArgs()) {
      nonIndexedArgsType.push_back(arg.getType());
      nonIndexedArgs.push_back(remappedOperands[argIdx++]);
    }

    // Generate the tuple encoding for the non-indexed args.
    // TODO: Are we sure we need an unbounded allocation here?
    Value tupleStart = evmB.genFreePtr();
    Value tupleEnd = evmB.genABITupleEncoding(nonIndexedArgsType,
                                              nonIndexedArgs, tupleStart);
    Value tupleSize = r.create<arith::SubIOp>(loc, tupleEnd, tupleStart);

    // Generate sol.log and replace sol.emit with it.
    r.replaceOpWithNewOp<yul::LogOp>(op, tupleStart, tupleSize, indexedArgs);

    return success();
  }
};

// (Copied and modified from clangir).
struct IfOpLowering : public OpRewritePattern<sol::IfOp> {
  using OpRewritePattern<sol::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::IfOp ifOp,
                                PatternRewriter &r) const override {
    Location loc = ifOp.getLoc();

    bool emptyElse = ifOp.getElseRegion().empty();
    Block *currentBlock = r.getInsertionBlock();
    Block *remainingOpsBlock =
        r.splitBlock(currentBlock, r.getInsertionPoint());
    Block *continueBlock;
    if (ifOp->getResults().empty())
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline then region.
    Block *thenBeforeBody = &ifOp.getThenRegion().front();
    Block *thenAfterBody = &ifOp.getThenRegion().back();
    r.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    r.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<sol::YieldOp>(thenAfterBody->getTerminator())) {
      r.replaceOpWithNewOp<cf::BranchOp>(thenYieldOp, thenYieldOp.getIns(),
                                         continueBlock);
    }

    r.setInsertionPointToEnd(continueBlock);

    // Has else region: inline it.
    Block *elseBeforeBody = nullptr;
    Block *elseAfterBody = nullptr;
    if (!emptyElse) {
      elseBeforeBody = &ifOp.getElseRegion().front();
      elseAfterBody = &ifOp.getElseRegion().back();
      r.inlineRegionBefore(ifOp.getElseRegion(), thenAfterBody);
    } else {
      elseBeforeBody = elseAfterBody = continueBlock;
    }

    r.setInsertionPointToEnd(currentBlock);
    r.create<cf::CondBranchOp>(loc, ifOp.getCond(), thenBeforeBody,
                               elseBeforeBody);

    if (!emptyElse) {
      r.setInsertionPointToEnd(elseAfterBody);
      if (auto elseYieldOp =
              dyn_cast<sol::YieldOp>(elseAfterBody->getTerminator())) {
        r.replaceOpWithNewOp<cf::BranchOp>(elseYieldOp, elseYieldOp.getIns(),
                                           continueBlock);
      }
    }

    r.replaceOp(ifOp, continueBlock->getArguments());
    return success();
  }
};

struct SwitchOpLowering : public OpRewritePattern<sol::SwitchOp> {
  using OpRewritePattern<sol::SwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::SwitchOp switchOp,
                                PatternRewriter &r) const override {
    // Split the block at the op.
    Block *condBlk = r.getInsertionBlock();
    Block *continueBlk = r.splitBlock(condBlk, Block::iterator(switchOp));

    auto convertRegion = [&](Region &region) -> Block * {
      for (Block &blk : region) {
        auto yield = dyn_cast<sol::YieldOp>(blk.getTerminator());
        if (!yield)
          continue;
        // Convert the yield terminator to a branch to the continue block.
        r.setInsertionPoint(yield);
        r.replaceOpWithNewOp<cf::BranchOp>(yield, continueBlk,
                                           yield.getOperands());
      }

      // Inline the region.
      Block *entryBlk = &region.front();
      r.inlineRegionBefore(region, continueBlk);
      return entryBlk;
    };

    // Convert the case regions.
    SmallVector<Block *> caseSuccessors;
    SmallVector<llvm::APInt> caseVals;
    caseSuccessors.reserve(switchOp.getCases().size());
    caseVals.reserve(switchOp.getCases().size());
    for (auto [region, val] :
         llvm::zip(switchOp.getCaseRegions(), switchOp.getCases())) {
      caseSuccessors.push_back(convertRegion(region));
      caseVals.push_back(val);
    }

    // Convert the default region.
    Block *defaultBlk = convertRegion(switchOp.getDefaultRegion());

    // Create the switch.
    r.setInsertionPointToEnd(condBlk);
    SmallVector<ValueRange> caseOperands(caseSuccessors.size(), {});

    // Create the attribute for the case values.
    auto caseValsAttr = DenseIntElementsAttr::get(
        VectorType::get(static_cast<int64_t>(caseVals.size()),
                        switchOp.getArg().getType()),
        caseVals);

    r.create<cf::SwitchOp>(switchOp.getLoc(), switchOp.getArg(), defaultBlk,
                           ValueRange(), caseValsAttr, caseSuccessors,
                           caseOperands);
    r.replaceOp(switchOp, continueBlk->getArguments());
    return success();
  }
};

// (Copied and modified from clangir).
struct LoopOpInterfaceLowering
    : public OpInterfaceRewritePattern<sol::LoopOpInterface> {
  using OpInterfaceRewritePattern<
      sol::LoopOpInterface>::OpInterfaceRewritePattern;

  /// Walks a region while skipping operations of type `Ops`. This ensures the
  /// callback is not applied to said operations and its children.
  template <typename... Ops>
  void
  walkRegionSkipping(Region &region,
                     function_ref<WalkResult(Operation *)> callback) const {
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<Ops...>(op))
        return WalkResult::skip();
      return callback(op);
    });
  }

  /// Lowers operations with the terminator trait that have a single successor.
  void lowerTerminator(Operation *op, Block *dest, PatternRewriter &r) const {
    assert(op->hasTrait<OpTrait::IsTerminator>() && "not a terminator");
    OpBuilder::InsertionGuard guard(r);
    r.setInsertionPoint(op);
    r.replaceOpWithNewOp<cf::BranchOp>(op, dest);
  }

  void lowerConditionOp(sol::ConditionOp op, Block *body, Block *exit,
                        PatternRewriter &r) const {
    OpBuilder::InsertionGuard guard(r);
    r.setInsertionPoint(op);
    r.replaceOpWithNewOp<cf::CondBranchOp>(op, op.getCondition(), body, exit);
  }

  LogicalResult matchAndRewrite(sol::LoopOpInterface op,
                                PatternRewriter &r) const override {
    // Setup CFG blocks.
    Block *entry = r.getInsertionBlock();
    Block *exit = r.splitBlock(entry, r.getInsertionPoint());
    Block *cond = &op.getCond().front();
    Block *body = &op.getBody().front();
    Block *step = (op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr);

    // Setup loop entry branch.
    r.setInsertionPointToEnd(entry);
    r.create<cf::BranchOp>(op.getLoc(), &op.getEntry().front());

    // Branch from condition region to body or exit.
    auto conditionOp = cast<sol::ConditionOp>(cond->getTerminator());
    lowerConditionOp(conditionOp, body, exit, r);

    // TODO: Remove the walks below. It visits operations unnecessarily,
    // however, to solve this we would likely need a custom DialectConversion
    // driver to customize the order that operations are visited.

    // Lower continue statements.
    Block *dest = (step ? step : cond);
    op.walkBodySkippingNestedLoops([&](Operation *op) {
      if (!isa<sol::ContinueOp>(op))
        return WalkResult::advance();

      lowerTerminator(op, dest, r);
      return WalkResult::skip();
    });

    // Lower break statements.
    // FIXME: Skip sol.switch once we implement it.
    walkRegionSkipping<sol::LoopOpInterface /* TODO:, sol::SwitchOp */>(
        op.getBody(), [&](Operation *op) {
          if (!isa<sol::BreakOp>(op))
            return WalkResult::advance();

          lowerTerminator(op, exit, r);
          return mlir::WalkResult::skip();
        });

    // Lower optional body region yield.
    for (Block &blk : op.getBody().getBlocks()) {
      auto bodyYield = dyn_cast<sol::YieldOp>(blk.getTerminator());
      if (bodyYield)
        lowerTerminator(bodyYield, (step ? step : cond), r);
    }

    // Lower mandatory step region yield.
    if (step)
      lowerTerminator(cast<sol::YieldOp>(step->getTerminator()), cond, r);

    // Move region contents out of the loop op.
    r.inlineRegionBefore(op.getCond(), exit);
    r.inlineRegionBefore(op.getBody(), exit);
    if (step)
      r.inlineRegionBefore(*op.maybeGetStep(), exit);

    r.eraseOp(op);
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

struct ReturnOpLowering : public OpConversionPattern<sol::ReturnOp> {
  using OpConversionPattern<sol::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<sol::FuncOp> {
  using OpConversionPattern<sol::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    // Set llvm.linkage attribute to private if not explicitly specified.
    std::vector<NamedAttribute> attrs;
    if (auto linkageAttr = op->getAttr("llvm.linkage"))
      attrs.push_back(r.getNamedAttr("llvm.linkage", linkageAttr));
    else
      attrs.push_back(r.getNamedAttr(
          "llvm.linkage",
          LLVM::LinkageAttr::get(r.getContext(), LLVM::Linkage::Private)));

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
    // FIXME: The location of the block arguments are lost here!
    auto newOp = r.create<func::FuncOp>(op.getLoc(), op.getName(),
                                        convertedFuncTy, attrs);
    r.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    r.eraseOp(op);
    return success();
  }
};

struct ContractOpLowering : public OpRewritePattern<sol::ContractOp> {
  const char *libAddrName = "library_deploy_address";

  using OpRewritePattern<sol::ContractOp>::OpRewritePattern;

  /// Generate the call value check.
  void genCallValChk(PatternRewriter &r, Location loc) const {
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto callVal = r.create<yul::CallValOp>(loc);
    auto callValChk = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                              callVal, bExt.genI256Const(0));
    evmB.genRevert(callValChk);
  };

  /// Generate the free pointer initialization.
  void genFreePtrInit(PatternRewriter &r, Location loc,
                      size_t reservedMem = 0) const {
    solidity::mlirgen::BuilderExt bExt(r, loc);
    mlir::Value freeMem = bExt.genI256Const(
        solidity::frontend::CompilerUtils::generalPurposeMemoryStart +
        reservedMem);
    r.create<yul::MStoreOp>(loc, bExt.genI256Const(64), freeMem);
  };

  /// Generates the dispatch to interface functions.
  void genDispatch(sol::ContractOp contrOp,
                   SmallVector<uint32_t, 4> const &selectors,
                   SmallVector<sol::FuncOp, 4> &ifcFns,
                   PatternRewriter &r) const {
    Location loc = contrOp.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value notDelegateCallCond;
    if (contrOp.getKind() == sol::ContractKind::Library) {
      auto libAddr = r.create<yul::LoadImmutableOp>(loc, r.getIntegerType(256),
                                                    libAddrName);
      auto currAddr = r.create<yul::AddressOp>(loc);
      notDelegateCallCond = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, libAddr, currAddr);
    }

    // Do nothing if there are no interface functions.
    if (ifcFns.empty())
      return;

    // Generate `if iszero(lt(calldatasize(), 4))` and set the insertion point
    // to its then block.
    auto callDataSz = r.create<yul::CallDataSizeOp>(loc);
    auto callDataSzCmp = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, callDataSz, bExt.genI256Const(4));
    auto ifOp =
        r.create<scf::IfOp>(loc, callDataSzCmp, /*withElseRegion=*/false);
    OpBuilder::InsertionGuard insertGuard(r);
    r.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Load the selector from the calldata.
    auto callDataLd = r.create<yul::CallDataLoadOp>(loc, bExt.genI256Const(0));
    Value callDataSelector =
        r.create<arith::ShRUIOp>(loc, callDataLd, bExt.genI256Const(224));
    callDataSelector =
        r.create<arith::TruncIOp>(loc, r.getIntegerType(32), callDataSelector);

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

    for (auto [caseRegion, ifcFnOp] :
         llvm::zip(switchOp.getCaseRegions(), ifcFns)) {
      OpBuilder::InsertionGuard insertGuard(r);
      mlir::Block *caseBlk = r.createBlock(&caseRegion);
      r.setInsertionPointToStart(caseBlk);

      assert(ifcFnOp.getStateMutability());
      sol::StateMutability stateMutability = *ifcFnOp.getStateMutability();
      if (contrOp.getKind() == sol::ContractKind::Library) {
        assert(stateMutability != sol::StateMutability::Payable);
        if (stateMutability > sol::StateMutability::View) {
          assert(notDelegateCallCond);
          evmB.genRevert(notDelegateCallCond);
        }
      }

      if (contrOp.getKind() != sol::ContractKind::Library &&
          stateMutability != sol::StateMutability::Payable) {
        genCallValChk(r, loc);
      }

      // Decode the input parameters (if required).
      FunctionType origIfcFnTy = *ifcFnOp.getOrigFnType();
      std::vector<Value> decodedArgs;
      if (!origIfcFnTy.getInputs().empty()) {
        evmB.genABITupleDecoding(origIfcFnTy.getInputs(),
                                 /*tupleStart=*/bExt.genI256Const(4),
                                 /*tupleEnd=*/callDataSz, decodedArgs,
                                 /*fromMem=*/false);
      }

      // Generate the actual call.
      auto callOp = r.create<sol::CallOp>(loc, ifcFnOp, decodedArgs);

      // Encode the result using the ABI's tuple encoder.
      auto tupleStart = evmB.genFreePtr();
      mlir::Value tupleSize;
      if (!callOp.getResultTypes().empty()) {
        auto tupleEnd = evmB.genABITupleEncoding(
            origIfcFnTy.getResults(), callOp.getResults(), tupleStart);
        tupleSize = r.create<arith::SubIOp>(loc, tupleEnd, tupleStart);
      } else {
        tupleSize = bExt.genI256Const(0);
      }

      // Generate the return.
      assert(tupleSize);
      r.create<yul::ReturnOp>(loc, tupleStart, tupleSize);

      r.create<mlir::scf::YieldOp>(loc);
    }
  }

  /// Collects reachable function from `fn` in `reachableFns`.
  void getReachableFuncs(sol::FuncOp fn,
                         llvm::SetVector<sol::FuncOp> &reachableFns) const {
    reachableFns.insert(fn);
    // FIXME! Handle indirect function references once we support it.
    fn.walk([&](CallOpInterface callOp) {
      auto calleeSym = cast<SymbolRefAttr>(callOp.getCallableForCallee());
      auto callee = cast<sol::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(fn, calleeSym));
      if (reachableFns.contains(callee))
        return;
      getReachableFuncs(callee, reachableFns);
    });
  }

  LogicalResult matchAndRewrite(sol::ContractOp op,
                                PatternRewriter &r) const override {
    mlir::Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    // Generate the creation and runtime ObjectOp.
    auto creationObj = r.create<yul::ObjectOp>(loc, op.getName());
    r.setInsertionPointToStart(&creationObj.getBody().front());
    auto runtimeObj =
        r.create<yul::ObjectOp>(loc, std::string(op.getName()) + "_deployed");

    SmallVector<sol::FuncOp, 4> ifcFns;
    sol::FuncOp ctor, receiveFn, fallbackFn;
    SmallVector<uint32_t, 4> selectors;
    size_t reservedMemSize = 0;
    SmallVector<sol::ImmutableOp, 2> immOps;

    // Track relevant types of functions (with selector, ctor, fallback,
    // receive) immutables (and reserved memory) etc; Remove state variables.
    for (Operation &i :
         llvm::make_early_inc_range(op.getBody()->getOperations())) {
      // Functions
      if (auto fn = dyn_cast<sol::FuncOp>(i)) {
        if (auto selector = fn.getSelector()) {
          selectors.push_back(*selector);
          ifcFns.push_back(fn);
        }
        auto fnKind = fn.getKind();
        if (fnKind) {
          switch (*fnKind) {
          case sol::FunctionKind::Constructor:
            assert(!ctor);
            ctor = fn;
            break;
          case sol::FunctionKind::Fallback:
            assert(!fallbackFn);
            fallbackFn = fn;
            break;
          case sol::FunctionKind::Receive:
            assert(!receiveFn);
            receiveFn = fn;
            break;
          default:
            break;
          }
        }

        // Immutables
      } else if (auto immOp = dyn_cast<sol::ImmutableOp>(i)) {
        reservedMemSize += evm::getCallDataHeadSize(immOp.getType());
        immOps.push_back(immOp);

        // State variables
      } else if (isa<sol::StateVarOp>(i)) {
        r.eraseOp(&i);

      } else {
        llvm_unreachable("NYI");
      }
    }

    if (ctor) {
      // Clone functions reachable from the ctor to creation and runtime
      // objects.
      llvm::SetVector<sol::FuncOp> reachableFns;
      getReachableFuncs(ctor, reachableFns);
      ctor->moveBefore(creationObj.getEntryBlock(),
                       creationObj.getEntryBlock()->begin());
      for (auto reachableFn : reachableFns) {
        if (reachableFn == ctor)
          continue;
        // Clone in the creation object and move the original to runtime object.
        r.clone(*reachableFn);
        reachableFn->moveBefore(runtimeObj.getEntryBlock(),
                                runtimeObj.getEntryBlock()->begin());
        reachableFn.setRuntimeAttr(r.getUnitAttr());
      }
    }

    for (auto fn :
         llvm::make_early_inc_range(op.getBody()->getOps<sol::FuncOp>())) {
      fn.setRuntimeAttr(r.getUnitAttr());
      fn->moveBefore(runtimeObj.getEntryBlock(),
                     runtimeObj.getEntryBlock()->begin());
    }

    //
    // Creation context
    //

    r.setInsertionPointToStart(creationObj.getEntryBlock());

    genFreePtrInit(r, loc, reservedMemSize);

    if (!ctor) {
      genCallValChk(r, loc);
    } else {
      assert(ctor.getStateMutability());
      if (*ctor.getStateMutability() != sol::StateMutability::Payable)
        genCallValChk(r, loc);
    }

    // Generate the call to constructor (if required).
    if (ctor && op.getKind() != sol::ContractKind::Library) {
      auto progSize = r.create<yul::DataSizeOp>(loc, creationObj.getName());
      auto codeSize = r.create<yul::CodeSizeOp>(loc);
      auto argSize = r.create<arith::SubIOp>(loc, codeSize, progSize);
      Value tupleStart = evmB.genMemAlloc(argSize);
      r.create<yul::CodeCopyOp>(loc, tupleStart, progSize, argSize);
      std::vector<Value> decodedArgs;
      FunctionType ctorFnTy = *ctor.getOrigFnType();
      if (!ctorFnTy.getInputs().empty()) {
        evmB.genABITupleDecoding(
            ctorFnTy.getInputs(), tupleStart,
            /*tupleEnd=*/r.create<arith::AddIOp>(loc, tupleStart, argSize),
            decodedArgs,
            /*fromMem=*/true);
      }
      r.create<sol::CallOp>(loc, ctor, decodedArgs);
    }

    // Generate the codecopy of the runtime object for the return from the
    // creation object.
    auto freePtr = r.create<yul::MLoadOp>(loc, bExt.genI256Const(64));
    auto runtimeObjSym = FlatSymbolRefAttr::get(runtimeObj);
    auto runtimeObjOffset = r.create<yul::DataOffsetOp>(loc, runtimeObjSym);
    auto runtimeObjSize = r.create<yul::DataSizeOp>(loc, runtimeObjSym);
    r.create<yul::CodeCopyOp>(loc, freePtr, runtimeObjOffset, runtimeObjSize);

    // Generate setimmutable of the library address (the runtime object uses
    // this for the delegate call check).
    if (op.getKind() == sol::ContractKind::Library)
      r.create<yul::SetImmutableOp>(loc, freePtr, libAddrName,
                                    r.create<yul::AddressOp>(loc));

    // Generate setimmutable's and remove all sol.immutable ops.
    for (sol::ImmutableOp immOp : llvm::make_early_inc_range(immOps)) {
      assert(immOp->getAttr("addr"));
      auto addr = bExt.genI256Const(
          cast<IntegerAttr>(immOp->getAttr("addr")).getValue());
      auto val = r.create<yul::MLoadOp>(loc, addr);
      r.create<yul::SetImmutableOp>(loc, freePtr, immOp.getName(), val);
      r.eraseOp(immOp);
    }

    // Generate the return for the creation context.
    r.create<yul::ReturnOp>(loc, freePtr, runtimeObjSize);

    //
    // Runtime context
    //

    r.setInsertionPointToStart(runtimeObj.getEntryBlock());

    // Generate the memory init.
    // TODO: Confirm if this should be the same as in the creation context.
    genFreePtrInit(r, loc);

    // Generate the dispatch to interface functions.
    genDispatch(op, selectors, ifcFns, r);

    // Generate receive function.
    if (receiveFn) {
      auto callDataSz = r.create<yul::CallDataSizeOp>(loc);
      auto callDataSzIsZero = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, callDataSz, bExt.genI256Const(0));
      auto ifOp =
          r.create<scf::IfOp>(loc, callDataSzIsZero, /*withElseRegion=*/false);
      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToStart(&ifOp.getThenRegion().front());
      r.create<sol::CallOp>(loc, receiveFn, /*operands=*/ValueRange{});
      r.create<yul::StopOp>(loc);
    }

    // Generate fallback function.
    if (fallbackFn) {
      FunctionType fallbackFnTy = fallbackFn.getFunctionType();
      assert(fallbackFnTy.getNumInputs() == fallbackFnTy.getNumResults() &&
             "NYI");
      (void)fallbackFnTy;

      if (fallbackFn.getStateMutability() != sol::StateMutability::Payable) {
        genCallValChk(r, loc);
      }
      r.create<sol::CallOp>(loc, fallbackFn, /*operands=*/ValueRange{});
      r.create<yul::StopOp>(loc);

    } else {
      // TODO: Generate error message.
      r.create<yul::RevertOp>(loc, bExt.genI256Const(0), bExt.genI256Const(0));
    }

    // TODO? Make sure op.getBody() is either empty or has only ops marked for
    // deletion.
    r.eraseOp(op);
    // TODO: Subobjects
    return success();
  }
};

} // namespace

// TODO: Assert that the type converter is compatible with the conversions. (We
// could only accept SolTypeConverter instead, but do we need need to be that
// strict?) (Also, can we do the assert at build time? If not, then only in
// debug builds?)

void evm::populateArithPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<ConstantOpLowering, CastOpLowering, BytesCastOpLowering,
           ArithBinOpLowering<sol::AddOp, arith::AddIOp>,
           ArithBinOpLowering<sol::SubOp, arith::SubIOp>,
           ArithBinOpLowering<sol::MulOp, arith::MulIOp>,
           ArithBinOpLowering<sol::ExpOp, yul::ExpOp>,
           DivOrModOpLowering<sol::DivOp, arith::DivSIOp, arith::DivUIOp>,
           DivOrModOpLowering<sol::ModOp, arith::RemSIOp, arith::RemUIOp>,
           CmpOpLowering>(tyConv, pats.getContext());
}

void evm::populateCheckedArithPats(RewritePatternSet &pats,
                                   TypeConverter &tyConv) {
  pats.add<CAddOpLowering, CSubOpLowering, CMulOpLowering, CDivOpLowering>(
      tyConv, pats.getContext());
}

void evm::populateMemPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<AllocaOpLowering, MallocOpLowering, ArrayLitOpLowering,
           GetCallDataOpLowering, PushOpLowering, PopOpLowering, GepOpLowering,
           MapOpLowering, LoadOpLowering, LoadImmutableOpLowering,
           StoreOpLowering, DataLocCastOpLowering, LengthOpLowering,
           CopyOpLowering>(tyConv, pats.getContext());
  pats.add<AddrOfOpLowering>(pats.getContext());
}

void evm::populateControlFlowPats(RewritePatternSet &pats) {
  pats.add<IfOpLowering, SwitchOpLowering, LoopOpInterfaceLowering>(
      pats.getContext());
}

void evm::populateFuncPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<CallOpLowering, ReturnOpLowering, FuncOpLowering>(tyConv,
                                                             pats.getContext());
}

void evm::populateAddrPat(RewritePatternSet &pats) {
  pats.add<ThisOpLowering, LibAddrOpLowering>(pats.getContext());
}

void evm::populateAbiPats(mlir::RewritePatternSet &pats,
                          mlir::TypeConverter &tyConv) {
  pats.add<EncodeOpLowering, DecodeOpLowering>(tyConv, pats.getContext());
}

void evm::populateExtCallPat(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<ExtCallOpLowering, TryOpLowering, NewOpLowering, CodeOpLowering>(
      tyConv, pats.getContext());
}

void evm::populateEmitPat(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<EmitOpLowering>(tyConv, pats.getContext());
}

void evm::populateRequirePat(RewritePatternSet &pats) {
  pats.add<RequireOpLowering>(pats.getContext());
}

void evm::populateContractPat(RewritePatternSet &pats) {
  pats.add<ContractOpLowering>(pats.getContext());
}

void evm::populateStage1Pats(RewritePatternSet &pats, TypeConverter &tyConv) {
  populateArithPats(pats, tyConv);
  populateCheckedArithPats(pats, tyConv);
  populateMemPats(pats, tyConv);
  populateAddrPat(pats);
  populateAbiPats(pats, tyConv);
  populateExtCallPat(pats, tyConv);
  populateEmitPat(pats, tyConv);
  populateRequirePat(pats);
  populateControlFlowPats(pats);
}

void evm::populateStage2Pats(RewritePatternSet &pats) {
  populateContractPat(pats);
  populateYulPats(pats);
}
