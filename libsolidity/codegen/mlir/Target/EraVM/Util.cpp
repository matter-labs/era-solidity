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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace eravm;
using namespace mlir;

unsigned eravm::getAlignment(AddrSpace addrSpace) {
  return addrSpace == AddrSpace_Stack ? ByteLen_Field : ByteLen_Byte;
}

unsigned eravm::getAlignment(Value ptr) {
  auto ty = ptr.getType().cast<LLVM::LLVMPointerType>();
  return getAlignment(static_cast<AddrSpace>(ty.getAddressSpace()));
}

unsigned eravm::getCallDataHeadSize(Type ty) {
  if (auto intTy = ty.dyn_cast<IntegerType>()) {
    return 32;
  }
  if (sol::hasDynamicallySizedElt(ty))
    return 32;

  llvm_unreachable("NYI: Other types");
}

void eravm::BuilderHelper::initGlobs(ModuleOp mod,
                                     std::optional<Location> locArg) {

  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderHelper h(b, loc);

  auto initInt = [&](const char *name) {
    LLVM::GlobalOp globOp =
        h.getOrInsertIntGlobalOp(name, mod, AddrSpace_Stack);
    Value globAddr = b.create<LLVM::AddressOfOp>(loc, globOp);
    b.create<LLVM::StoreOp>(loc, h.genI256Const(0, locArg), globAddr,
                            getAlignment(globAddr));
  };

  auto i256Ty = b.getIntegerType(256);

  // Initialize the following global ints
  initInt(GlobHeapMemPtr);
  initInt(GlobCallDataSize);
  initInt(GlobRetDataSz);
  initInt(GlobCallFlags);

  // Initialize the GlobExtraABIData int array
  auto extraABIData = h.getOrInsertGlobalOp(
      GlobExtraABIData, mod, LLVM::LLVMArrayType::get(i256Ty, 10),
      getAlignment(AddrSpace_Stack), AddrSpace_Stack, LLVM::Linkage::Private,
      b.getZeroAttr(RankedTensorType::get({10}, i256Ty)));
  Value extraABIDataAddr = b.create<LLVM::AddressOfOp>(loc, extraABIData);
  b.create<LLVM::StoreOp>(
      loc,
      h.genI256ConstSplat(std::vector<llvm::APInt>(10, llvm::APInt(256, 0)),
                          locArg),
      extraABIDataAddr);
}

Value eravm::BuilderHelper::getABILen(Value ptr,
                                      std::optional<Location> locArg) {
  auto i256Ty = b.getIntegerType(256);
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderHelper h(b, loc);

  Value ptrToInt = b.create<LLVM::PtrToIntOp>(loc, i256Ty, ptr).getResult();
  Value lShr = b.create<LLVM::LShrOp>(
      loc, ptrToInt, h.genI256Const(eravm::BitLen_X32 * 3, locArg));
  return b.create<LLVM::AndOp>(loc, lShr, h.genI256Const(UINT_MAX, locArg));
}

Value eravm::BuilderHelper::genABITupleEncoding(
    ArrayRef<Type> tys, ArrayRef<Value> vals, Value headStart,
    std::optional<mlir::Location> locArg) {
  // TODO: Move this to the evm namespace.

  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderHelper h(b, loc);

  unsigned totCallDataHeadSz = 0;
  for (Type ty : tys)
    totCallDataHeadSz += getCallDataHeadSize(ty);

  auto tail = b.create<arith::AddIOp>(loc, headStart,
                                      h.genI256Const(totCallDataHeadSz));
  size_t headPos = 0;

  for (auto it : llvm::zip(tys, vals)) {
    Type ty = std::get<0>(it);
    Value val = std::get<1>(it);
    if (sol::hasDynamicallySizedElt(ty)) {
      auto headPosOffset =
          b.create<arith::AddIOp>(loc, headStart, h.genI256Const(headPos));
      b.create<sol::MStoreOp>(loc, headPosOffset,
                              b.create<arith::SubIOp>(loc, tail, headStart));

      if (auto stringTy = ty.dyn_cast<sol::StringType>()) {
        // Copy the length field.
        auto size = b.create<sol::MLoadOp>(loc, val);
        b.create<sol::MStoreOp>(loc, tail, size);

        // Copy the data.
        auto dataAddr = b.create<arith::AddIOp>(loc, val, h.genI256Const(32));
        auto tailDataAddr =
            b.create<arith::AddIOp>(loc, tail, h.genI256Const(32));
        b.create<sol::MCopyOp>(loc, tailDataAddr, dataAddr, size);

        // Write 0 at the end.
        b.create<sol::MStoreOp>(
            loc, b.create<arith::AddIOp>(loc, tailDataAddr, size),
            h.genI256Const(0));

        // FIXME: The round up should be consistent with the lowering of
        // memory-string
        tail = b.create<arith::AddIOp>(loc, tailDataAddr,
                                       h.genRoundUpToMultiple<32>(size));
      } else {
        llvm_unreachable("NYI");
      }
    } else {
      llvm_unreachable("NYI");
    }
    headPos += getCallDataHeadSize(ty);
  }

  return tail;
}

sol::FuncOp eravm::BuilderHelper::getOrInsertCreationFuncOp(
    llvm::StringRef name, FunctionType fnTy, ModuleOp mod) {
  solidity::mlirgen::BuilderHelper h(b);
  return h.getOrInsertFuncOp(name, fnTy, LLVM::Linkage::Private, mod);
}

sol::FuncOp eravm::BuilderHelper::getOrInsertRuntimeFuncOp(llvm::StringRef name,
                                                           FunctionType fnTy,
                                                           ModuleOp mod) {
  solidity::mlirgen::BuilderHelper h(b);
  sol::FuncOp fn = h.getOrInsertFuncOp(name, fnTy, LLVM::Linkage::Private, mod);
  fn.setRuntimeAttr(b.getUnitAttr());
  return fn;
}

FlatSymbolRefAttr eravm::BuilderHelper::getOrInsertReturn(ModuleOp mod) {
  auto *ctx = mod.getContext();
  solidity::mlirgen::BuilderHelper h(b);
  auto i256Ty = IntegerType::get(ctx, 256);

  auto fnTy = FunctionType::get(ctx, {i256Ty, i256Ty, i256Ty}, {});
  auto fn = h.getOrInsertFuncOp("__return", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(mod.getContext(), "__return");
}

FlatSymbolRefAttr eravm::BuilderHelper::getOrInsertRevert(ModuleOp mod) {
  auto *ctx = mod.getContext();
  solidity::mlirgen::BuilderHelper h(b);
  auto i256Ty = IntegerType::get(ctx, 256);

  auto fnTy = FunctionType::get(ctx, {i256Ty, i256Ty, i256Ty}, {});
  auto fn = h.getOrInsertFuncOp("__revert", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(mod.getContext(), "__revert");
}

FlatSymbolRefAttr eravm::BuilderHelper::getOrInsertSha3(ModuleOp mod) {
  auto *ctx = mod.getContext();
  solidity::mlirgen::BuilderHelper h(b);

  auto i1Ty = IntegerType::get(ctx, 1);
  auto i256Ty = IntegerType::get(ctx, 256);
  auto heapPtrTy = LLVM::LLVMPointerType::get(ctx, AddrSpace_Heap);
  auto fnTy = FunctionType::get(ctx, {heapPtrTy, i256Ty, i1Ty}, {i256Ty});
  auto fn = h.getOrInsertFuncOp("__sha3", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(mod.getContext(), "__sha3");
}

LLVM::AddressOfOp
eravm::BuilderHelper::getCallDataSizeAddr(ModuleOp mod,
                                          std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderHelper h(b, loc);

  LLVM::GlobalOp globCallDataSzDef = h.getOrInsertIntGlobalOp(
      eravm::GlobCallDataSize, mod, eravm::AddrSpace_Stack);
  return b.create<LLVM::AddressOfOp>(loc, globCallDataSzDef);
}

LLVM::AddressOfOp
eravm::BuilderHelper::getCallDataPtrAddr(ModuleOp mod,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderHelper h(b, loc);

  LLVM::GlobalOp callDataPtrDef = h.getOrInsertPtrGlobalOp(
      eravm::GlobCallDataPtr, mod, eravm::AddrSpace_Generic);
  return b.create<LLVM::AddressOfOp>(loc, callDataPtrDef);
}

LLVM::LoadOp
eravm::BuilderHelper::loadCallDataPtr(ModuleOp mod,
                                      std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderHelper h(b, loc);

  LLVM::AddressOfOp callDataPtrAddr = getCallDataPtrAddr(mod, loc);
  return b.create<LLVM::LoadOp>(loc, callDataPtrAddr,
                                eravm::getAlignment(callDataPtrAddr));
}

void eravm::BuilderHelper::genPanic(solidity::util::PanicCode code, Value cond,
                                    std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  solidity::mlirgen::BuilderHelper h(b, loc);

  std::string selector =
      solidity::util::selectorFromSignatureU256("Panic(uint256)").str();

  b.create<scf::IfOp>(
      loc, cond, /*thenBuilder=*/[&](OpBuilder &b, Location loc) {
        b.create<sol::MStoreOp>(loc, h.genI256Const(0, locArg),
                                h.genI256Const(selector, locArg));
        b.create<sol::MStoreOp>(
            loc, h.genI256Const(4, locArg),
            h.genI256Const(static_cast<int64_t>(code), locArg));
        b.create<sol::RevertOp>(loc, h.genI256Const(0, locArg),
                                h.genI256Const(24, locArg));
        b.create<scf::YieldOp>(loc);
      });
}
