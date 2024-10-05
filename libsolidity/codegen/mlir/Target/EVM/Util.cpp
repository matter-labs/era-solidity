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

#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolutil/FunctionSelector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

unsigned evm::getAlignment(evm::AddrSpace addrSpace) {
  // FIXME: Confirm this!
  return addrSpace == evm::AddrSpace_Stack ? evm::ByteLen_Field
                                           : evm::ByteLen_Byte;
}

unsigned evm::getAlignment(Value ptr) {
  auto ty = cast<LLVM::LLVMPointerType>(ptr.getType());
  return getAlignment(static_cast<evm::AddrSpace>(ty.getAddressSpace()));
}

unsigned evm::getCallDataHeadSize(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return 32;

  if (sol::hasDynamicallySizedElt(ty))
    return 32;

  llvm_unreachable("NYI: Other types");
}

unsigned evm::getStorageByteCount(Type ty) {
  if (isa<IntegerType>(ty) || isa<sol::MappingType>(ty) ||
      isa<sol::StringType>(ty))
    return 32;
  llvm_unreachable("NYI: Other types");
}

Value evm::Builder::genFreePtr(std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);
  return b.create<sol::MLoadOp>(loc, bExt.genI256Const(64));
}

Value evm::Builder::genMemAlloc(Value size, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  Value freePtr = genFreePtr(locArg);

  // FIXME: Shouldn't we check for overflow in the freePtr + size operation
  // and generate PanicCode::ResourceError?
  Value newFreePtr = b.create<arith::AddIOp>(loc, freePtr, size);

  // Generate the PanicCode::ResourceError check.
  //
  // FIXME: Do we need to check this in EraVM? I assume this is from the
  // following ir-breaking-changes:
  // - The new code generator imposes a hard limit of ``type(uint64).max``
  //   (``0xffffffffffffffff``) for the free memory pointer. Allocations that
  //   would increase its value beyond this limit revert. The old code generator
  //   does not have this limit.
  auto newPtrGtMax =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, newFreePtr,
                              bExt.genI256Const("0xffffffffffffffff"));
  auto newPtrLtOrig = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                              newFreePtr, freePtr);
  auto panicCond = b.create<arith::OrIOp>(loc, newPtrGtMax, newPtrLtOrig);
  genPanic(solidity::util::PanicCode::ResourceError, panicCond);

  b.create<sol::MStoreOp>(loc, bExt.genI256Const(64), newFreePtr);

  return freePtr;
}

Value evm::Builder::genMemAlloc(AllocSize size,
                                std::optional<Location> locArg) {
  assert(size % 32 == 0);
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);
  return genMemAlloc(bExt.genI256Const(size), loc);
}

Value evm::Builder::genMemAllocForDynArray(Value sizeVar, Value sizeInBytes,
                                           std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  solidity::mlirgen::BuilderExt bExt(b, loc);

  // dynSize is size + length-slot where length-slot's size is 32 bytes.
  auto dynSizeInBytes =
      b.create<arith::AddIOp>(loc, sizeInBytes, bExt.genI256Const(32));
  auto memPtr = genMemAlloc(dynSizeInBytes, loc);
  b.create<sol::MStoreOp>(loc, memPtr, sizeVar);
  return memPtr;
}

Value evm::Builder::genDataAddrPtr(Value addr, sol::DataLocation dataLoc,
                                   std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  if (dataLoc == sol::DataLocation::Memory) {
    // Return the address after the first word.
    return b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(32));
  }

  if (dataLoc == sol::DataLocation::Storage) {
    // Return the keccak256 of addr.
    auto zero = bExt.genI256Const(0);
    b.create<sol::MStoreOp>(loc, zero, addr);
    return b.create<sol::Keccak256Op>(loc, zero, bExt.genI256Const(32));
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genLoad(Value addr, sol::DataLocation dataLoc,
                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  if (dataLoc == sol::DataLocation::Memory)
    return b.create<sol::MLoadOp>(loc, addr);

  if (dataLoc == sol::DataLocation::Storage)
    return b.create<sol::SLoadOp>(loc, addr);

  llvm_unreachable("NYI");
}

void evm::Builder::genStore(Value val, Value addr, sol::DataLocation dataLoc,
                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  if (dataLoc == sol::DataLocation::Memory) {
    b.create<sol::MStoreOp>(loc, addr, val);
  } else if (dataLoc == sol::DataLocation::Storage) {
    b.create<sol::SStoreOp>(loc, addr, val);
  } else {
    llvm_unreachable("NYI");
  }
}

void evm::Builder::genStringStore(std::string const &str, Value addr,
                                  std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  // Generate the size store.
  b.create<sol::MStoreOp>(loc, addr, bExt.genI256Const(str.length()));

  // Store the strings in 32 byte chunks of their numerical representation.
  for (size_t i = 0; i < str.length(); i += 32) {
    // Copied from solidity::yul::valueOfStringLiteral.
    std::string subStrAsI256 =
        solidity::u256(solidity::util::h256(str.substr(i, 32),
                                            solidity::util::h256::FromBinary,
                                            solidity::util::h256::AlignLeft))
            .str();
    addr = b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(32));
    b.create<sol::MStoreOp>(loc, addr, bExt.genI256Const(subStrAsI256));
  }
}

void evm::Builder::genCopyLoop(Value srcAddr, Value dstAddr, Value sizeInWords,
                               sol::DataLocation srcDataLoc,
                               sol::DataLocation dstDataLoc,
                               std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  solidity::mlirgen::BuilderExt bExt(b, loc);

  auto genAddrAtIdx = [&](Value baseAddr, Value idx, sol::DataLocation dataLoc,
                          Location loc) {
    if (dataLoc == sol::DataLocation::Memory) {
      Value memIdx = b.create<arith::MulIOp>(loc, idx, bExt.genI256Const(32));
      return b.create<arith::AddIOp>(loc, baseAddr, memIdx);
    }

    if (dataLoc == sol::DataLocation::Storage) {
      return b.create<arith::AddIOp>(loc, baseAddr, idx);
    }

    llvm_unreachable("NYI");
  };

  b.create<scf::ForOp>(
      loc, /*lowerBound=*/bExt.genIdxConst(0),
      /*upperBound=*/bExt.genCastToIdx(sizeInWords),
      /*step=*/bExt.genIdxConst(1),
      /*iterArgs=*/std::nullopt,
      /*builder=*/
      [&](OpBuilder &b, Location loc, Value indVar, ValueRange iterArgs) {
        Value i256IndVar = bExt.genCastToI256(indVar);

        Value srcAddrAtIdx = genAddrAtIdx(srcAddr, i256IndVar, srcDataLoc, loc);
        Value val = genLoad(srcAddrAtIdx, srcDataLoc, loc);
        Value dstAddrAtIdx = genAddrAtIdx(dstAddr, i256IndVar, dstDataLoc, loc);
        genStore(val, dstAddrAtIdx, dstDataLoc, loc);

        b.create<scf::YieldOp>(loc);
      });
}

void evm::Builder::genABITupleSizeAssert(TypeRange tys, Value tupleSize,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  unsigned totCallDataHeadSz = 0;
  for (Type ty : tys)
    totCallDataHeadSz += getCallDataHeadSize(ty);

  auto shortTupleCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tupleSize,
                              bExt.genI256Const(totCallDataHeadSz));
  if (/*TODO: m_revertStrings < RevertStrings::Debug*/ false)
    genRevertWithMsg(shortTupleCond, "ABI decoding: tuple data too short");
  else
    genRevert(shortTupleCond);
}

Value evm::Builder::genABITupleEncoding(TypeRange tys, ValueRange vals,
                                        Value tupleStart,
                                        std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  unsigned totCallDataHeadSz = 0;
  for (Type ty : tys)
    totCallDataHeadSz += getCallDataHeadSize(ty);

  Value headAddr = tupleStart;
  Value tailAddr = b.create<arith::AddIOp>(
      loc, tupleStart, bExt.genI256Const(totCallDataHeadSz));
  for (auto it : llvm::zip(tys, vals)) {
    Type ty = std::get<0>(it);
    Value val = std::get<1>(it);

    // String type
    if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
      b.create<sol::MStoreOp>(
          loc, headAddr, b.create<arith::SubIOp>(loc, tailAddr, tupleStart));

      // Copy the length field.
      auto size = b.create<sol::MLoadOp>(loc, val);
      b.create<sol::MStoreOp>(loc, tailAddr, size);

      // Copy the data.
      auto dataAddr = b.create<arith::AddIOp>(loc, val, bExt.genI256Const(32));
      auto tailDataAddr =
          b.create<arith::AddIOp>(loc, tailAddr, bExt.genI256Const(32));
      b.create<sol::MCopyOp>(loc, tailDataAddr, dataAddr, size);

      // Write 0 at the end.
      b.create<sol::MStoreOp>(loc,
                              b.create<arith::AddIOp>(loc, tailDataAddr, size),
                              bExt.genI256Const(0));

      tailAddr = b.create<arith::AddIOp>(loc, tailDataAddr,
                                         bExt.genRoundUpToMultiple<32>(size));

      // Integer type
    } else if (auto intTy = dyn_cast<IntegerType>(ty)) {
      val = bExt.genIntCast(/*width=*/256, intTy.isSigned(), val);
      b.create<sol::MStoreOp>(loc, headAddr, val);

    } else {
      llvm_unreachable("NYI");
    }
    headAddr = b.create<arith::AddIOp>(
        loc, headAddr, bExt.genI256Const(getCallDataHeadSize(ty)));
  }

  return tailAddr;
}

Value evm::Builder::genABITupleEncoding(std::string const &str, Value headStart,
                                        std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  // Generate the offset store at the head address.
  mlir::Value thirtyTwo = bExt.genI256Const(32);
  b.create<sol::MStoreOp>(loc, headStart, thirtyTwo);

  // Generate the string creation at the tail address.
  auto tailAddr = b.create<arith::AddIOp>(loc, headStart, thirtyTwo);
  genStringStore(str, tailAddr, locArg);

  return tailAddr;
}

void evm::Builder::genABITupleDecoding(TypeRange tys, Value tupleStart,
                                       Value tupleEnd,
                                       std::vector<Value> &results,
                                       bool fromMem,
                                       std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  // TODO? {en|de}codingType() for sol dialect types.

  genABITupleSizeAssert(tys,
                        b.create<arith::SubIOp>(loc, tupleEnd, tupleStart));
  Value headAddr = tupleStart;
  auto genLoad = [&](Value addr) -> Value {
    if (fromMem)
      return b.create<sol::MLoadOp>(loc, headAddr);
    return b.create<sol::CallDataLoadOp>(loc, headAddr);
  };

  for (auto ty : tys) {
    if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
      Value tailAddr =
          b.create<arith::AddIOp>(loc, tupleStart, genLoad(headAddr));

      // FIXME: Shouldn't we check tailAddr < tupleEnd instead? The following
      // code is copied from the yul codegen. I assume this is from the
      // following ir-breaking-changes:
      //
      // - The new code generator imposes a hard limit of ``type(uint64).max``
      //   (``0xffffffffffffffff``) for the free memory pointer. Allocations
      //   that would increase its value beyond this limit revert. The old code
      //   generator does not have this limit.
      auto invalidTailAddrChk =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, tailAddr,
                                  bExt.genI256Const("0xffffffffffffffff"));
      genRevertWithMsg(invalidTailAddrChk, "ABI decoding: invalid tuple offset",
                       locArg);

      // TODO: "ABI decoding: invalid calldata array offset" revert check. We
      // need to track the "headEnd" for this.

      Value sizeInBytes = genLoad(tailAddr);

      // Copy the decoded string to a new memory allocation.
      Value dstAddr = genMemAllocForDynArray(
          sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes));
      Value thirtyTwo = bExt.genI256Const(32);
      Value dstDataAddr = b.create<arith::AddIOp>(loc, dstAddr, thirtyTwo);
      Value srcDataAddr = b.create<arith::AddIOp>(loc, tailAddr, thirtyTwo);
      // TODO: "ABI decoding: invalid byte array length" revert check.

      // FIXME: ABIFunctions::abiDecodingFunctionByteArrayAvailableLength only
      // allocates length + 32 (where length is rounded up to a multiple of 32)
      // bytes. The "+ 32" is for the size field. But it calls
      // YulUtilFunctions::copyToMemoryFunction with the _cleanup param enabled
      // which makes the writing of the zero at the end an out-of-bounds write.
      // Even if the allocation was done correctly, why should we write zero at
      // the end?
      if (fromMem)
        // TODO? Check m_evmVersion.hasMcopy() and legalize here or in sol.mcopy
        // lowering?
        b.create<sol::MCopyOp>(loc, dstDataAddr, srcDataAddr, sizeInBytes);
      else
        b.create<sol::CallDataCopyOp>(loc, dstDataAddr, srcDataAddr,
                                      sizeInBytes);

      results.push_back(dstAddr);

    } else if (auto intTy = dyn_cast<IntegerType>(ty)) {
      Value castedArg = bExt.genIntCast(intTy.getWidth(), intTy.isSigned(),
                                        genLoad(headAddr));
      // TODO: Generate the validator for non 256 bit types. The validator
      // should just check if the loaded value is what we would get if we
      // extended the original `intTy` value to 256 bits.
      results.push_back(castedArg);
    }

    headAddr = b.create<arith::AddIOp>(
        loc, headAddr, bExt.genI256Const(getCallDataHeadSize(ty)));
  }
}

void evm::Builder::genPanic(solidity::util::PanicCode code, Value cond,
                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  std::string selector =
      solidity::util::selectorFromSignatureU256("Panic(uint256)").str();

  b.create<scf::IfOp>(
      loc, cond, /*thenBuilder=*/[&](OpBuilder &b, Location loc) {
        b.create<sol::MStoreOp>(loc, bExt.genI256Const(0),
                                bExt.genI256Const(selector));
        b.create<sol::MStoreOp>(loc, bExt.genI256Const(4),
                                bExt.genI256Const(static_cast<int64_t>(code)));
        b.create<sol::RevertOp>(loc, bExt.genI256Const(0),
                                bExt.genI256Const(24));
        b.create<scf::YieldOp>(loc);
      });
}

void evm::Builder::genRevert(Value cond, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());

  solidity::mlirgen::BuilderExt bExt(b, loc);
  mlir::Value zero = bExt.genI256Const(0);
  b.create<sol::RevertOp>(loc, zero, zero);
}

void evm::Builder::genRevertWithMsg(std::string const &msg,
                                    std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  solidity::mlirgen::BuilderExt bExt(b, loc);

  // Generate the "Error(string)" selector store at free ptr.
  std::string selector =
      solidity::util::selectorFromSignatureU256("Error(string)").str();
  Value freePtr = genFreePtr();
  b.create<sol::MStoreOp>(loc, freePtr, bExt.genI256Const(selector));

  // Generate the tuple encoding of the message after the selector.
  auto freePtrPostSelector =
      b.create<arith::AddIOp>(loc, freePtr, bExt.genI256Const(4));
  Value tailAddr = genABITupleEncoding(msg, /*headStart=*/freePtrPostSelector);

  // Generate the revert.
  auto retDataSize = b.create<arith::SubIOp>(loc, tailAddr, freePtr);
  b.create<sol::RevertOp>(loc, freePtr, retDataSize);
}

void evm::Builder::genRevertWithMsg(Value cond, std::string const &msg,
                                    std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genRevertWithMsg(msg);
}

FlatSymbolRefAttr evm::Builder::getPersonality() {
  return FlatSymbolRefAttr::get(b.getContext(), "__personality");
}
