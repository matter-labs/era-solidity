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

#include "libsolidity/codegen/mlir/Target/EVM/SolToStandard.h"
#include "libsolidity/codegen/CompilerUtils.h"
#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolidity/codegen/mlir/Target/EraVM/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

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

struct ExtOpLowering : public OpConversionPattern<sol::ExtOp> {
  using OpConversionPattern<sol::ExtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ExtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto origOutTy = cast<IntegerType>(op.getType());
    IntegerType signlessOutTy = r.getIntegerType(origOutTy.getWidth());

    if (origOutTy.isSigned())
      r.replaceOpWithNewOp<arith::ExtSIOp>(op, signlessOutTy, adaptor.getIn());
    else
      r.replaceOpWithNewOp<arith::ExtUIOp>(op, signlessOutTy, adaptor.getIn());

    return success();
  }
};

struct TruncOpLowering : public OpConversionPattern<sol::TruncOp> {
  using OpConversionPattern<sol::TruncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::TruncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    IntegerType signlessOutTy =
        r.getIntegerType(cast<IntegerType>(op.getType()).getWidth());
    r.replaceOpWithNewOp<arith::TruncIOp>(op, signlessOutTy, adaptor.getIn());

    return success();
  }
};

/// A templatized version of a conversion pattern for lowering arithmetic binary
/// ops.
template <typename SrcOpT, typename DstOpT>
struct ArithBinOpConvPat : public OpConversionPattern<SrcOpT> {
  using OpConversionPattern<SrcOpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(SrcOpT op, typename SrcOpT::Adaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<DstOpT>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct CAddOpLowering : public OpConversionPattern<sol::CAddOp> {
  using OpConversionPattern<sol::CAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value sum = r.create<arith::AddIOp>(loc, lhs, rhs);

    if (ty.getWidth() == 256) {
      // Signed 256-bit int type.
      if (ty.isSigned()) {
        // (Copied from the yul codegen)
        // overflow, if x >= 0 and sum < y
        // underflow, if x < 0 and sum >= y

        auto zero = bExt.genI256Const(0);

        // Generate the overflow condition.
        auto lhsGtEqZero =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs, zero);
        auto sumLtRhs =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, sum, rhs);
        auto overflowCond = r.create<arith::AndIOp>(loc, lhsGtEqZero, sumLtRhs);

        // Generate the underflow condition.
        auto lhsLtZero =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, zero);
        auto sumGtEqRhs =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, sum, rhs);
        auto underflowCond =
            r.create<arith::AndIOp>(loc, lhsLtZero, sumGtEqRhs);

        eraB.genPanic(solidity::util::PanicCode::UnderOverflow,
                      r.create<arith::OrIOp>(loc, overflowCond, underflowCond));
        // Unsigned 256-bit int type.
      } else {
        eraB.genPanic(
            solidity::util::PanicCode::UnderOverflow,
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, lhs, sum));
      }
    } else {
      llvm_unreachable("NYI");
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
    eravm::Builder eraB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value diff = r.create<arith::SubIOp>(loc, lhs, rhs);

    if (ty.getWidth() == 256) {
      // Signed 256-bit int type.
      if (ty.isSigned()) {
        // (Copied from the yul codegen)
        // underflow, if y >= 0 and diff > x
        // overflow, if y < 0 and diff < x

        auto zero = bExt.genI256Const(0);

        // Generate the overflow condition.
        auto rhsGtEqZero =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, rhs, zero);
        auto diffGtLhs =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, diff, lhs);
        auto overflowCond =
            r.create<arith::AndIOp>(loc, rhsGtEqZero, diffGtLhs);

        // Generate the underflow condition.
        auto rhsLtZero =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rhs, zero);
        auto diffLtRhs =
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, diff, lhs);
        auto underflowCond = r.create<arith::AndIOp>(loc, rhsLtZero, diffLtRhs);

        eraB.genPanic(solidity::util::PanicCode::UnderOverflow,
                      r.create<arith::OrIOp>(loc, overflowCond, underflowCond));
        // Unsigned 256-bit int type.
      } else {
        eraB.genPanic(
            solidity::util::PanicCode::UnderOverflow,
            r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, diff, lhs));
      }
    } else {
      llvm_unreachable("NYI");
    }

    r.replaceOp(op, diff);
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
    r.replaceOpWithNewOp<LLVM::AllocaOp>(op, convertedEltTy,
                                         bExt.genI256Const(size));
    return success();
  }
};

struct MallocOpLowering : public OpConversionPattern<sol::MallocOp> {
  using OpConversionPattern<sol::MallocOp>::OpConversionPattern;

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
      return structTy.getMemberTypes().size() * 32;
    }

    // Value type.
    return 32;
  }

  /// Generates the memory allocation and optionally the zero initializer code.
  Value genMemAlloc(Type ty, bool zeroInit, Value sizeVar, int64_t recDepth,
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
        if (sizeVar) {
          assert(recDepth == 0);
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
              genMemAlloc(eltTy, zeroInit, sizeVar, recDepth, r, loc));
          // FIXME: Is the stride always 32?
          assert(constSize.value() % 32 == 0);
          auto sizeInWords = constSize.value() / 32;
          for (int64_t i = 1; i < sizeInWords; ++i) {
            Value incrMemPtr = r.create<arith::AddIOp>(
                loc, dataPtr, bExt.genI256Const(32 * i));
            r.create<sol::MStoreOp>(
                loc, incrMemPtr,
                genMemAlloc(eltTy, zeroInit, sizeVar, recDepth, r, loc));
          }
        } else {
          // Generate the loop for the stores of offsets.

          // TODO? Generate an unrolled loop if the size variable is a
          // ConstantOp?

          // `size` should be a multiple of 32.

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
                    genMemAlloc(eltTy, zeroInit, sizeVar, recDepth, r, loc));
                b.create<scf::YieldOp>(loc);
              });
        }

      } else if (zeroInit) {
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

      for (auto memTy : structTy.getMemberTypes()) {
        Value initVal;
        if (isa<sol::StructType>(memTy) || isa<sol::ArrayType>(memTy)) {
          initVal = genMemAlloc(memTy, zeroInit, sizeVar, recDepth, r, loc);
          r.create<sol::MStoreOp>(loc, memPtr, initVal);
        } else if (zeroInit) {
          r.create<sol::MStoreOp>(loc, memPtr, bExt.genI256Const(0));
        }
      }
      // TODO: Support other types.
    } else {
      llvm_unreachable("Invalid type");
    }

    assert(memPtr);
    return memPtr;
  }

  Value genMemAlloc(sol::MallocOp op, PatternRewriter &r) const {
    return genMemAlloc(op.getType(), op.getZeroInit(), op.getSize(),
                       /*recDepth=*/-1, r, op.getLoc());
  }

  LogicalResult matchAndRewrite(sol::MallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Value freePtr = genMemAlloc(op, r);
    r.replaceOp(op, freePtr);
    return success();
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

struct GepOpLowering : public OpConversionPattern<sol::GepOp> {
  using OpConversionPattern<sol::GepOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::GepOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    Type baseAddrTy = op.getBaseAddr().getType();
    Value remappedBaseAddr = adaptor.getBaseAddr();
    Value idx = adaptor.getIdx();
    Value res;

    switch (sol::getDataLocation(baseAddrTy)) {
    case sol::DataLocation::Stack: {
      auto stkPtrTy =
          LLVM::LLVMPointerType::get(r.getContext(), eravm::AddrSpace_Stack);
      res = r.create<LLVM::GEPOp>(loc, /*resultType=*/stkPtrTy,
                                  /*basePtrType=*/remappedBaseAddr.getType(),
                                  remappedBaseAddr, idx);
      break;
    }

    case sol::DataLocation::Memory: {
      // Memory array
      if (auto arrTy = dyn_cast<sol::ArrayType>(baseAddrTy)) {
        Value addrAtIdx;

        Type eltTy = arrTy.getEltType();
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
            addrAtIdx = r.create<arith::AddIOp>(
                loc, remappedBaseAddr,
                bExt.genI256Const(constIdx.value() * 32));
          }
        }

        if (!addrAtIdx) {
          //
          // Generate PanicCode::ArrayOutOfBounds check.
          //
          Value size;
          if (arrTy.isDynSized()) {
            size = r.create<sol::MLoadOp>(loc, remappedBaseAddr);
          } else {
            size = bExt.genI256Const(arrTy.getSize());
          }

          // Generate `if iszero(lt(index, <arrayLen>(baseRef)))` (yul).
          auto panicCond = r.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::uge, idx, size);
          eraB.genPanic(solidity::util::PanicCode::ArrayOutOfBounds, panicCond);

          //
          // Generate the address.
          //
          Value castedIdx =
              bExt.genIntCast(/*width=*/256, /*isSigned=*/false, idx);
          Value scaledIdx =
              r.create<arith::MulIOp>(loc, castedIdx, bExt.genI256Const(32));
          if (arrTy.isDynSized()) {
            // Generate the address after the length-slot.
            Value dataAddr = r.create<arith::AddIOp>(loc, remappedBaseAddr,
                                                     bExt.genI256Const(32));
            addrAtIdx = r.create<arith::AddIOp>(loc, dataAddr, scaledIdx);
          } else {
            addrAtIdx =
                r.create<arith::AddIOp>(loc, remappedBaseAddr, scaledIdx);
          }
        }
        assert(addrAtIdx);

        // We need to generate load for member/element having a reference type
        // since reference types track address.
        if (sol::isNonPtrRefType(eltTy)) {
          assert(sol::getDataLocation(eltTy) == sol::DataLocation::Memory);
          res = r.create<sol::MLoadOp>(loc, addrAtIdx);
        } else {
          res = addrAtIdx;
        }

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
        Value addrAtIdx =
            r.create<arith::AddIOp>(loc, remappedBaseAddr, scaledIdx);

        auto memberTy = structTy.getMemberTypes()[idxConstOp.value()];
        if (sol::isNonPtrRefType(memberTy)) {
          assert(sol::getDataLocation(memberTy) == sol::DataLocation::Memory);
          res = r.create<sol::MLoadOp>(loc, addrAtIdx);
        } else {
          res = addrAtIdx;
        }
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
    r.create<sol::MStoreOp>(loc, zero, key);
    r.create<sol::MStoreOp>(loc, bExt.genI256Const(0x20), adaptor.getMapping());

    r.replaceOpWithNewOp<sol::Keccak256Op>(op, zero, bExt.genI256Const(0x40));
    return success();
  }
};

struct LoadOpLowering : public OpConversionPattern<sol::LoadOp> {
  using OpConversionPattern<sol::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Value addr = adaptor.getAddr();

    switch (sol::getDataLocation(op.getAddr().getType())) {
    case sol::DataLocation::Stack:
      r.replaceOpWithNewOp<LLVM::LoadOp>(op, addr, eravm::getAlignment(addr));
      return success();
    case sol::DataLocation::Memory:
      r.replaceOpWithNewOp<sol::MLoadOp>(op, addr);
      return success();
    case sol::DataLocation::Storage:
      r.replaceOpWithNewOp<sol::SLoadOp>(op, addr);
      return success();
    default:
      break;
    };

    llvm_unreachable("NYI: Calldata data-location");
  }
};

struct StoreOpLowering : public OpConversionPattern<sol::StoreOp> {
  using OpConversionPattern<sol::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Value val = adaptor.getVal();
    Value addr = adaptor.getAddr();

    switch (sol::getDataLocation(op.getAddr().getType())) {
    case sol::DataLocation::Stack:
      r.replaceOpWithNewOp<LLVM::StoreOp>(op, val, addr,
                                          eravm::getAlignment(addr));
      return success();
    case sol::DataLocation::Memory:
      r.replaceOpWithNewOp<sol::MStoreOp>(op, addr, val);
      return success();
    case sol::DataLocation::Storage:
      r.replaceOpWithNewOp<sol::SStoreOp>(op, addr, val);
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
    eravm::Builder eraB(r, loc);

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
        assert(isa<sol::StringType>(dstTy));
        auto dataSlot = eraB.genDataAddrPtr(adaptor.getInp(), srcDataLoc);

        // Generate the memory allocation.
        auto sizeInBytes = eraB.genLoad(adaptor.getInp(), srcDataLoc);
        auto memPtrTy =
            sol::StringType::get(r.getContext(), sol::DataLocation::Memory);
        Value mallocRes =
            r.create<sol::MallocOp>(loc, memPtrTy,
                                    /*zeroInit=*/false, sizeInBytes);
        Type i256Ty = r.getIntegerType(256);
        assert(getTypeConverter()->convertType(mallocRes.getType()) == i256Ty);
        mlir::Value memAddr = getTypeConverter()->materializeSourceConversion(
            r, loc, /*resultType=*/i256Ty, /*inputs=*/mallocRes);
        resAddr = memAddr;

        // Generate the loop to copy the data (sol.malloc lowering will generate
        // the store of the size field).
        auto dataMemAddr =
            r.create<arith::AddIOp>(loc, memAddr, bExt.genI256Const(32));
        auto sizeInWords = bExt.genRoundUpToMultiple<32>(sizeInBytes);
        eraB.genCopyLoop(dataSlot, dataMemAddr, sizeInWords, srcDataLoc,
                         dstDataLoc);
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

struct CopyOpLowering : public OpConversionPattern<sol::CopyOp> {
  using OpConversionPattern<sol::CopyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    eravm::Builder eraB(r, loc);

    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();
    sol::DataLocation srcDataLoc = sol::getDataLocation(srcTy);
    sol::DataLocation dstDataLoc = sol::getDataLocation(dstTy);
    assert(srcDataLoc == sol::DataLocation::Memory &&
           dstDataLoc == sol::DataLocation::Storage && "NYI");

    if (isa<sol::StringType>(srcTy)) {
      assert(isa<sol::StringType>(dstTy));

      // Generate the size update.
      Value srcSize = eraB.genLoad(adaptor.getSrc(), srcDataLoc);
      eraB.genStore(srcSize, adaptor.getDst(), dstDataLoc);

      // Generate the copy loop.
      Value srcDataAddr = eraB.genDataAddrPtr(adaptor.getSrc(), srcDataLoc);
      Value dstDataAddr = eraB.genDataAddrPtr(adaptor.getDst(), dstDataLoc);
      Value sizeInWords = bExt.genRoundUpToMultiple<32>(srcSize);
      eraB.genCopyLoop(srcDataAddr, dstDataAddr, sizeInWords, srcDataLoc,
                       dstDataLoc);
    } else {
      llvm_unreachable("NYI");
    }

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
    // FIXME: The location of the block arguments are lost here!
    auto newOp =
        r.create<func::FuncOp>(loc, op.getName(), convertedFuncTy, attrs);
    r.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    r.eraseOp(op);
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
      std::vector<Value> decodedArgs;
      if (!origIfcFnTy.getInputs().empty()) {
        eraB.genABITupleDecoding(origIfcFnTy.getInputs(),
                                 /*tupleStart=*/bExt.genI256Const(4),
                                 /*tupleEnd=*/callDataSz, decodedArgs,
                                 /*fromMem=*/false);
      }

      // Generate the actual call.
      auto callOp = r.create<sol::CallOp>(loc, ifcFnOp, decodedArgs);

      // Encode the result using the ABI's tuple encoder.
      auto tupleStart = eraB.genFreePtr();
      mlir::Value tupleSize;
      if (!callOp.getResultTypes().empty()) {
        auto tupleEnd = eraB.genABITupleEncoding(
            origIfcFnTy.getResults(), callOp.getResults(), tupleStart);
        tupleSize = r.create<arith::SubIOp>(loc, tupleEnd, tupleStart);
      } else {
        tupleSize = bExt.genI256Const(0);
      }

      // Generate the return.
      assert(tupleSize);
      r.create<sol::BuiltinRetOp>(loc, tupleStart, tupleSize);

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
      Value tupleStart = eraB.genMemAlloc(argSize);
      r.create<sol::CodeCopyOp>(loc, tupleStart, progSize, argSize);
      std::vector<Value> decodedArgs;
      assert(op.getCtorFnType());
      auto ctorFnTy = *op.getCtorFnType();
      if (!ctorFnTy.getInputs().empty()) {
        eraB.genABITupleDecoding(
            ctorFnTy.getInputs(), tupleStart,
            /*tupleEnd=*/r.create<arith::AddIOp>(loc, tupleStart, argSize),
            decodedArgs,
            /*fromMem=*/true);
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

} // namespace

void evm::populateArithPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<ConstantOpLowering, ExtOpLowering, TruncOpLowering,
           ArithBinOpConvPat<sol::AddOp, arith::AddIOp>,
           ArithBinOpConvPat<sol::SubOp, arith::SubIOp>,
           ArithBinOpConvPat<sol::MulOp, arith::MulIOp>, CmpOpLowering>(
      tyConv, pats.getContext());
}

void evm::populateCheckedArithPats(RewritePatternSet &pats,
                                   TypeConverter &tyConv) {
  pats.add<CAddOpLowering, CSubOpLowering>(tyConv, pats.getContext());
}

void evm::populateMemPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<AllocaOpLowering, MallocOpLowering, GepOpLowering, MapOpLowering,
           LoadOpLowering, StoreOpLowering, DataLocCastOpLowering,
           CopyOpLowering>(tyConv, pats.getContext());
  pats.add<AddrOfOpLowering>(pats.getContext());
}

void evm::populateFuncPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<CallOpLowering, ReturnOpLowering, FuncOpLowering>(tyConv,
                                                             pats.getContext());
}

void evm::populateContrPat(RewritePatternSet &pats) {
  pats.add<ContractOpLowering>(pats.getContext());
}
