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
#include "libsolutil/FunctionSelector.h"
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

using namespace eravm;
using namespace mlir;

unsigned eravm::getAlignment(AddrSpace addrSpace) {
  return addrSpace == AddrSpace_Stack ? ByteLen_Field : ByteLen_Byte;
}

unsigned eravm::getAlignment(Value ptr) {
  auto ty = cast<LLVM::LLVMPointerType>(ptr.getType());
  return getAlignment(static_cast<AddrSpace>(ty.getAddressSpace()));
}

unsigned eravm::getCallDataHeadSize(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    return 32;
  }
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
      LLVM::LLVMPointerType::get(b.getContext(), AddrSpace_Heap);
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

  // Initialize the following global ints
  initInt(GlobHeapMemPtr);
  initInt(GlobCallDataSize);
  initInt(GlobRetDataSize);
  initInt(GlobCallFlags);

  // Initialize the GlobExtraABIData int array
  auto extraABIData = bExt.getOrInsertGlobalOp(
      GlobExtraABIData, LLVM::LLVMArrayType::get(i256Ty, 10), AddrSpace_Stack,
      /*alignment=*/0, LLVM::Linkage::Private,
      b.getZeroAttr(RankedTensorType::get({10}, i256Ty)), mod);
  Value extraABIDataAddr = b.create<LLVM::AddressOfOp>(loc, extraABIData);
  b.create<LLVM::StoreOp>(
      loc,
      bExt.genI256ConstSplat(std::vector<llvm::APInt>(10, llvm::APInt(256, 0)),
                             locArg),
      extraABIDataAddr);
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

Value eravm::Builder::genABITupleEncoding(
    TypeRange tys, ValueRange vals, Value headStart,
    std::optional<mlir::Location> locArg) {
  // TODO: Move this to the evm namespace.

  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  unsigned totCallDataHeadSz = 0;
  for (Type ty : tys)
    totCallDataHeadSz += getCallDataHeadSize(ty);

  size_t headOffset = 0;
  Value currTailAddr;
  for (auto it : llvm::zip(tys, vals)) {
    Type ty = std::get<0>(it);
    Value val = std::get<1>(it);
    auto currHeadAddr =
        b.create<arith::AddIOp>(loc, headStart, bExt.genI256Const(headOffset));
    currTailAddr = b.create<arith::AddIOp>(
        loc, headStart, bExt.genI256Const(totCallDataHeadSz));

    // String type.
    if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
      b.create<sol::MStoreOp>(
          loc, currHeadAddr,
          b.create<arith::SubIOp>(loc, currTailAddr, headStart));

      // Copy the length field.
      auto size = b.create<sol::MLoadOp>(loc, val);
      b.create<sol::MStoreOp>(loc, currTailAddr, size);

      // Copy the data.
      auto dataAddr = b.create<arith::AddIOp>(loc, val, bExt.genI256Const(32));
      auto tailDataAddr =
          b.create<arith::AddIOp>(loc, currTailAddr, bExt.genI256Const(32));
      b.create<sol::MCopyOp>(loc, tailDataAddr, dataAddr, size);

      // Write 0 at the end.
      b.create<sol::MStoreOp>(loc,
                              b.create<arith::AddIOp>(loc, tailDataAddr, size),
                              bExt.genI256Const(0));

      currTailAddr = b.create<arith::AddIOp>(
          loc, tailDataAddr, bExt.genRoundUpToMultiple<32>(size));
      // Integer type.
    } else if (isa<IntegerType>(ty)) {
      b.create<sol::MStoreOp>(loc, currHeadAddr, val);

    } else {
      llvm_unreachable("NYI");
    }
    headOffset += getCallDataHeadSize(ty);
  }

  assert(currTailAddr);
  return currTailAddr;
}

void eravm::Builder::genABITupleDecoding(TypeRange tys, Value headStart,
                                         std::vector<Value> &results,
                                         bool fromMem,
                                         std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderExt bExt(b, loc);

  // TODO? {en|de}codingType() for sol dialect types.

  // TODO: Generate "ABI decoding: tuple data too short" revert check.

  size_t headPos = 0;
  for (auto ty : tys) {
    if (!isa<IntegerType>(ty))
      llvm_unreachable("NYI");

    Value offset = bExt.genI256Const(headPos);

    auto headStartPlusOffset = b.create<arith::AddIOp>(loc, headStart, offset);
    if (fromMem)
      results.push_back(b.create<sol::MLoadOp>(loc, headStartPlusOffset));
    else
      results.push_back(
          b.create<sol::CallDataLoadOp>(loc, headStartPlusOffset));

    // TODO: Generate "Validator".

    headPos += getCallDataHeadSize(ty);
  }
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
    APInt shiftedAuxHeapMarker(/*numBits=*/256, RetForwardPageType::UseAuxHeap);
    shiftedAuxHeapMarker <<= eravm::BitLen_X32 * 7;
    abiData = b.create<arith::AddIOp>(loc, abiData,
                                      bExt.genI256Const(shiftedAuxHeapMarker));
  }

  if (isSysCall) {
    APInt shiftedAuxHeapMarker(/*numBits=*/256, RetForwardPageType::UseAuxHeap);
    shiftedAuxHeapMarker <<= eravm::BitLen_X32 * 7 + BitLen_Byte * 3;
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
