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

#include "libsolidity/codegen/mlir/Target/EraVM/Util.h"
#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolutil/ErrorCodes.h"
#include "libsolutil/FixedHash.h"
#include "libsolutil/FunctionSelector.h"
#include "libsolutil/Numeric.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IntrinsicsEraVM.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

unsigned eravm::getAlignment(eravm::AddrSpace addrSpace) {
  return addrSpace == eravm::AddrSpace_Stack ? eravm::ByteLen_Field
                                             : eravm::ByteLen_Byte;
}

unsigned eravm::getAlignment(Value ptr) {
  auto ty = cast<LLVM::LLVMPointerType>(ptr.getType());
  return getAlignment(static_cast<eravm::AddrSpace>(ty.getAddressSpace()));
}

unsigned eravm::getCallDataHeadSize(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return 32;

  if (sol::hasDynamicallySizedElt(ty))
    return 32;

  llvm_unreachable("NYI: Other types");
}

unsigned eravm::getStorageByteCount(Type ty) {
  if (isa<IntegerType>(ty) || isa<sol::MappingType>(ty) ||
      isa<sol::StringType>(ty))
    return 32;
  llvm_unreachable("NYI: Other types");
}

Value eravm::Builder::genHeapPtr(Value addr, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto heapAddrSpacePtrTy =
      LLVM::LLVMPointerType::get(b.getContext(), eravm::AddrSpace_Heap);
  return b.create<LLVM::IntToPtrOp>(loc, heapAddrSpacePtrTy, addr);
}

void eravm::Builder::genGlobalVarsInit(ModuleOp mod,
                                       std::optional<Location> locArg) {

  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  auto initInt = [&](const char *name) {
    LLVM::GlobalOp globOp = bExt.getOrInsertI256GlobalOp(
        name, AddrSpace_Stack,
        /*alignment=*/0, LLVM::Linkage::Private, mod);
    Value globAddr = b.create<LLVM::AddressOfOp>(loc, globOp);
    b.create<LLVM::StoreOp>(loc, bExt.genI256Const(0, locArg), globAddr,
                            getAlignment(globAddr));
  };

  auto i256Ty = b.getIntegerType(256);

  // Initialize the following global ints.
  initInt(eravm::GlobHeapMemPtr);
  initInt(eravm::GlobCallDataSize);
  initInt(eravm::GlobRetDataSize);
  initInt(eravm::GlobCallFlags);

  // Initialize the GlobExtraABIData int array.
  auto extraABIData = bExt.getOrInsertGlobalOp(
      eravm::GlobExtraABIData, LLVM::LLVMArrayType::get(i256Ty, 10),
      eravm::AddrSpace_Stack,
      /*alignment=*/0, LLVM::Linkage::Private,
      b.getZeroAttr(RankedTensorType::get({10}, i256Ty)), mod);
  Value extraABIDataAddr = b.create<LLVM::AddressOfOp>(loc, extraABIData);
  b.create<LLVM::StoreOp>(
      loc,
      bExt.genI256ConstSplat(std::vector<llvm::APInt>(10, llvm::APInt(256, 0)),
                             locArg),
      extraABIDataAddr);

  // Initialize the GlobActivePtr array.
  auto genericAddrSpacePtrTy =
      LLVM::LLVMPointerType::get(b.getContext(), eravm::AddrSpace_Generic);
  bExt.getOrInsertGlobalOp(
      eravm::GlobActivePtr, LLVM::LLVMArrayType::get(genericAddrSpacePtrTy, 16),
      eravm::AddrSpace_Stack,
      /*alignment=*/0, LLVM::Linkage::Private, /*attr=*/{}, mod);
}

void eravm::Builder::genStringStore(std::string const &str, Value addr,
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

Value eravm::Builder::genABILen(Value ptr, std::optional<Location> locArg) {
  auto i256Ty = b.getIntegerType(256);
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  Value ptrToInt = b.create<LLVM::PtrToIntOp>(loc, i256Ty, ptr).getResult();
  Value lShr = b.create<LLVM::LShrOp>(
      loc, ptrToInt, bExt.genI256Const(eravm::BitLen_X32 * 3, locArg));
  return b.create<LLVM::AndOp>(loc, lShr, bExt.genI256Const(UINT_MAX, locArg));
}

void eravm::Builder::genABITupleSizeAssert(TypeRange tys, Value tupleSize,
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

Value eravm::Builder::genABITupleEncoding(
    TypeRange tys, ValueRange vals, Value tupleStart,
    std::optional<mlir::Location> locArg) {
  // TODO: Move this to the evm namespace.

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
      if (intTy.getWidth() == 1)
        val = b.create<arith::ExtUIOp>(loc, b.getIntegerType(256), val);
      else
        // FIXME: We might need to track the sign in integral types for
        // generating the correct extension.
        assert(intTy.getWidth() == 256 && "NYI");
      b.create<sol::MStoreOp>(loc, headAddr, val);

    } else {
      llvm_unreachable("NYI");
    }
    headAddr = b.create<arith::AddIOp>(
        loc, headAddr, bExt.genI256Const(getCallDataHeadSize(ty)));
  }

  return tailAddr;
}

Value eravm::Builder::genABITupleEncoding(
    std::string const &str, Value headStart,
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

void eravm::Builder::genABITupleDecoding(TypeRange tys, Value tupleStart,
                                         Value tupleEnd,
                                         std::vector<Value> &results,
                                         bool fromMem,
                                         std::optional<mlir::Location> locArg) {
  // TODO: Move this to the evm namespace.
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
      assert(intTy.getWidth() == 256 && "NYI");
      // TODO: Generate the validator for non 256 bit types. The validator
      // should just check if the loaded value is what we would get if we
      // extended the original `intTy` value to 256 bits.
      results.push_back(genLoad(headAddr));
    }

    headAddr = b.create<arith::AddIOp>(
        loc, headAddr, bExt.genI256Const(getCallDataHeadSize(ty)));
  }
}

void eravm::Builder::genRevert(Value cond, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());

  solidity::mlirgen::BuilderExt bExt(b, loc);
  mlir::Value zero = bExt.genI256Const(0);
  b.create<sol::RevertOp>(loc, zero, zero);
}

void eravm::Builder::genRevertWithMsg(std::string const &msg,
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

void eravm::Builder::genRevertWithMsg(Value cond, std::string const &msg,
                                      std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genRevertWithMsg(msg);
}

Value eravm::Builder::genABIData(Value addr, Value size,
                                 eravm::AddrSpace addrSpace, bool isSysCall,
                                 std::optional<Location> locArg) {
  assert(cast<IntegerType>(addr.getType()).getWidth() == 256);
  assert(cast<IntegerType>(size.getType()).getWidth() == 256);

  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  // TODO? Move this to eravm::Builder?
  auto genClamp = [&](mlir::Value val, uint64_t maxValLiteral) {
    IntegerType valTy = cast<IntegerType>(val.getType());
    assert(valTy.getWidth() > 64);

    // Generate the clamped value allocation and set it to max.
    auto maxVal = bExt.genConst(maxValLiteral, valTy.getWidth());
    auto clampedValAddr = b.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(valTy), bExt.genConst(1),
        eravm::getAlignment(AddrSpace_Stack));
    b.create<LLVM::StoreOp>(loc, maxVal, clampedValAddr,
                            eravm::getAlignment(AddrSpace_Stack));

    // Generate the check for value <= max and the store of value.
    auto valLEMax =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, val, maxVal);
    b.create<scf::IfOp>(loc, valLEMax,
                        /*thenBuilder=*/[&](OpBuilder &b, Location loc) {
                          b.create<LLVM::StoreOp>(loc, val, clampedValAddr);
                          b.create<scf::YieldOp>(loc);
                        });

    return b.create<LLVM::LoadOp>(loc, clampedValAddr,
                                  eravm::getAlignment(AddrSpace_Stack));
  };

  // Generate the clamp of the address, length and gas.
  mlir::Value clampedAddr = genClamp(addr, UINT32_MAX);
  mlir::Value clampedSize = genClamp(size, UINT32_MAX);
  auto gasLeft = b.create<LLVM::IntrCallOp>(
      loc, /*resTy=*/b.getIntegerType(256),
      b.getI32IntegerAttr(llvm::Intrinsic::eravm_gasleft),
      b.getStringAttr("eravm.gasleft"));
  mlir::Value clampedGas = genClamp(gasLeft, UINT32_MAX);

  // Generate the abi data from the address, length and gas.
  auto shiftedAddr = b.create<arith::ShLIOp>(
      loc, clampedAddr, bExt.genI256Const(eravm::BitLen_X32 * 2));
  auto shiftedLen = b.create<arith::ShLIOp>(
      loc, clampedSize, bExt.genI256Const(eravm::BitLen_X32 * 3));
  auto shiftedGas = b.create<arith::ShLIOp>(
      loc, clampedGas, bExt.genI256Const(eravm::BitLen_X32 * 6));
  auto abiData = b.create<arith::AddIOp>(loc, shiftedAddr, shiftedLen);
  abiData = b.create<arith::AddIOp>(loc, abiData, shiftedGas);

  if (addrSpace == eravm::AddrSpace_HeapAuxiliary) {
    APInt shiftedAuxHeapMarker(/*numBits=*/256,
                               eravm::RetForwardPageType::UseAuxHeap);
    shiftedAuxHeapMarker <<= eravm::BitLen_X32 * 7;
    abiData = b.create<arith::AddIOp>(loc, abiData,
                                      bExt.genI256Const(shiftedAuxHeapMarker));
  }

  if (isSysCall) {
    APInt shiftedAuxHeapMarker(/*numBits=*/256,
                               eravm::RetForwardPageType::UseAuxHeap);
    shiftedAuxHeapMarker <<= eravm::BitLen_X32 * 7 + eravm::BitLen_Byte * 3;
    abiData = b.create<arith::AddIOp>(loc, abiData,
                                      bExt.genI256Const(shiftedAuxHeapMarker));
  }

  return abiData;
}

sol::FuncOp eravm::Builder::getOrInsertCreationFuncOp(llvm::StringRef name,
                                                      FunctionType fnTy,
                                                      ModuleOp mod) {
  solidity::mlirgen::BuilderExt bExt(b);
  return bExt.getOrInsertFuncOp(name, fnTy, LLVM::Linkage::Private, mod);
}

sol::FuncOp eravm::Builder::getOrInsertRuntimeFuncOp(llvm::StringRef name,
                                                     FunctionType fnTy,
                                                     ModuleOp mod) {
  solidity::mlirgen::BuilderExt bExt(b);
  sol::FuncOp fn =
      bExt.getOrInsertFuncOp(name, fnTy, LLVM::Linkage::Private, mod);
  fn.setRuntimeAttr(b.getUnitAttr());
  return fn;
}

FlatSymbolRefAttr eravm::Builder::getPersonality() {
  return FlatSymbolRefAttr::get(b.getContext(), "__personality");
}

FlatSymbolRefAttr eravm::Builder::getOrInsertPersonality(ModuleOp mod) {
  solidity::mlirgen::BuilderExt bExt(b);

  auto fnTy = FunctionType::get(b.getContext(), {},
                                {IntegerType::get(b.getContext(), 32)});
  auto fn = bExt.getOrInsertFuncOp("__personality", fnTy,
                                   LLVM::Linkage::External, mod);
  fn.setPrivate();
  return getPersonality();
}

FlatSymbolRefAttr eravm::Builder::getOrInsertReturn(ModuleOp mod) {
  auto *ctx = mod.getContext();
  solidity::mlirgen::BuilderExt bExt(b);
  auto i256Ty = IntegerType::get(ctx, 256);

  auto fnTy = FunctionType::get(ctx, {i256Ty, i256Ty, i256Ty}, {});
  auto fn =
      bExt.getOrInsertFuncOp("__return", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(mod.getContext(), "__return");
}

FlatSymbolRefAttr eravm::Builder::getOrInsertRevert(ModuleOp mod) {
  auto *ctx = mod.getContext();
  solidity::mlirgen::BuilderExt bExt(b);
  auto i256Ty = IntegerType::get(ctx, 256);

  auto fnTy = FunctionType::get(ctx, {i256Ty, i256Ty, i256Ty}, {});
  auto fn =
      bExt.getOrInsertFuncOp("__revert", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(mod.getContext(), "__revert");
}

FlatSymbolRefAttr eravm::Builder::getOrInsertSha3(ModuleOp mod) {
  auto *ctx = mod.getContext();
  solidity::mlirgen::BuilderExt bExt(b);

  auto i1Ty = IntegerType::get(ctx, 1);
  auto i256Ty = IntegerType::get(ctx, 256);
  auto heapPtrTy = LLVM::LLVMPointerType::get(ctx, AddrSpace_Heap);
  auto fnTy = FunctionType::get(ctx, {heapPtrTy, i256Ty, i1Ty}, {i256Ty});
  auto fn =
      bExt.getOrInsertFuncOp("__sha3", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(mod.getContext(), "__sha3");
}

sol::FuncOp eravm::Builder::getOrInsertFarCall(ModuleOp mod) {
  auto *ctx = mod.getContext();
  solidity::mlirgen::BuilderExt bExt(b);

  auto i1Ty = IntegerType::get(ctx, 1);
  auto i256Ty = IntegerType::get(ctx, 256);
  auto genericPtrTy = LLVM::LLVMPointerType::get(ctx, AddrSpace_Generic);

  std::vector<Type> inpTys(eravm::MandatoryArgCnt + eravm::ExtraABIDataSize,
                           i256Ty);
  auto fnTy = FunctionType::get(ctx, inpTys, {genericPtrTy, i1Ty});

  auto fn =
      bExt.getOrInsertFuncOp("__farcall", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return fn;
}

LLVM::AddressOfOp
eravm::Builder::genCallDataSizeAddr(ModuleOp mod,
                                    std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  LLVM::GlobalOp globCallDataSzDef = bExt.getOrInsertI256GlobalOp(
      eravm::GlobCallDataSize, eravm::AddrSpace_Stack, /*alignment=*/0,
      LLVM::Linkage::Private, mod);
  return b.create<LLVM::AddressOfOp>(loc, globCallDataSzDef);
}

LLVM::LoadOp
eravm::Builder::genCallDataSizeLoad(ModuleOp mod,
                                    std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  LLVM::AddressOfOp addr = genCallDataSizeAddr(mod, loc);
  return b.create<LLVM::LoadOp>(loc, addr, eravm::getAlignment(addr));
}

LLVM::AddressOfOp
eravm::Builder::genCallDataPtrAddr(ModuleOp mod,
                                   std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  LLVM::GlobalOp callDataPtrDef = bExt.getOrInsertPtrGlobalOp(
      eravm::GlobCallDataPtr, eravm::AddrSpace_Generic, LLVM::Linkage::Private,
      mod);
  return b.create<LLVM::AddressOfOp>(loc, callDataPtrDef);
}

LLVM::LoadOp
eravm::Builder::genCallDataPtrLoad(ModuleOp mod,
                                   std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  LLVM::AddressOfOp callDataPtrAddr = genCallDataPtrAddr(mod, loc);
  return b.create<LLVM::LoadOp>(loc, callDataPtrAddr,
                                eravm::getAlignment(callDataPtrAddr));
}

Value eravm::Builder::genFreePtr(std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);
  return b.create<sol::MLoadOp>(loc, bExt.genI256Const(64));
}

Value eravm::Builder::genMemAlloc(Value size, std::optional<Location> locArg) {
  // TODO: Move this to the evm namespace.

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

Value eravm::Builder::genMemAlloc(AllocSize size,
                                  std::optional<Location> locArg) {
  assert(size % 32 == 0);
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);
  return genMemAlloc(bExt.genI256Const(size), loc);
}

Value eravm::Builder::genMemAllocForDynArray(Value sizeVar, Value sizeInBytes,
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

Value eravm::Builder::genLoad(Value addr, sol::DataLocation dataLoc,
                              std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  if (dataLoc == sol::DataLocation::Memory)
    return b.create<sol::MLoadOp>(loc, addr);

  if (dataLoc == sol::DataLocation::Storage)
    return b.create<sol::SLoadOp>(loc, addr);

  llvm_unreachable("NYI");
}

void eravm::Builder::genStore(Value val, Value addr, sol::DataLocation dataLoc,
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

void eravm::Builder::genCopyLoop(Value srcAddr, Value dstAddr,
                                 Value sizeInWords,
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

Value eravm::Builder::genDataAddrPtr(Value addr, sol::DataLocation dataLoc,
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

void eravm::Builder::genPanic(solidity::util::PanicCode code, Value cond,
                              std::optional<Location> locArg) {
  // TODO: Move this to the evm namespace.

  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  std::string selector =
      solidity::util::selectorFromSignatureU256("Panic(uint256)").str();

  b.create<scf::IfOp>(
      loc, cond, /*thenBuilder=*/[&](OpBuilder &b, Location loc) {
        b.create<sol::MStoreOp>(loc, bExt.genI256Const(0, locArg),
                                bExt.genI256Const(selector, locArg));
        b.create<sol::MStoreOp>(
            loc, bExt.genI256Const(4, locArg),
            bExt.genI256Const(static_cast<int64_t>(code), locArg));
        b.create<sol::RevertOp>(loc, bExt.genI256Const(0, locArg),
                                bExt.genI256Const(24, locArg));
        b.create<scf::YieldOp>(loc);
      });
}
