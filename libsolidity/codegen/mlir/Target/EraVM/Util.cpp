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
#include "libsolidity/codegen/mlir/Util.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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

void eravm::BuilderHelper::initGlobs(ModuleOp mod,
                                     std::optional<Location> locArg) {

  Location loc = locArg ? *locArg : defLoc;

  auto initInt = [&](const char *name) {
    LLVM::GlobalOp globOp =
        h.getOrInsertIntGlobalOp(name, mod, AddrSpace_Stack);
    Value globAddr = b.create<LLVM::AddressOfOp>(loc, globOp);
    b.create<LLVM::StoreOp>(loc, h.getConst(0, 256, locArg), globAddr,
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
      h.getConstSplat(std::vector<llvm::APInt>(10, llvm::APInt(256, 0)), 256,
                      locArg),
      extraABIDataAddr);
}

Value eravm::BuilderHelper::getABILen(Value ptr,
                                      std::optional<Location> locArg) {
  auto i256Ty = b.getIntegerType(256);
  Location loc = locArg ? *locArg : defLoc;

  Value ptrToInt = b.create<LLVM::PtrToIntOp>(loc, i256Ty, ptr).getResult();
  // TODO: Pass argLoc to h.getConst once it has support for default location.
  Value lShr = b.create<LLVM::LShrOp>(
      loc, ptrToInt, h.getConst(eravm::BitLen_X32 * 3, 256, locArg));
  return b.create<LLVM::AndOp>(loc, lShr, h.getConst(UINT_MAX, 256, locArg));
}

sol::FuncOp eravm::BuilderHelper::getOrInsertCreationFuncOp(
    llvm::StringRef name, FunctionType fnTy, ModuleOp mod) {
  return h.getOrInsertFuncOp(name, fnTy, LLVM::Linkage::Private, mod);
}

sol::FuncOp eravm::BuilderHelper::getOrInsertRuntimeFuncOp(llvm::StringRef name,
                                                           FunctionType fnTy,
                                                           ModuleOp mod) {
  sol::FuncOp fn = h.getOrInsertFuncOp(name, fnTy, LLVM::Linkage::Private, mod);
  fn.setRuntimeAttr(b.getUnitAttr());
  return fn;
}

FlatSymbolRefAttr eravm::BuilderHelper::getOrInsertReturn(ModuleOp mod) {
  auto *ctx = mod.getContext();

  auto i256Ty = IntegerType::get(ctx, 256);
  auto fnTy = FunctionType::get(ctx, {i256Ty, i256Ty, i256Ty}, {});
  auto fn = h.getOrInsertFuncOp("__return", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(mod.getContext(), "__return");
}

FlatSymbolRefAttr eravm::BuilderHelper::getOrInsertRevert(ModuleOp mod) {
  auto *ctx = mod.getContext();

  auto i256Ty = IntegerType::get(ctx, 256);
  auto fnTy = FunctionType::get(ctx, {i256Ty, i256Ty, i256Ty}, {});
  auto fn = h.getOrInsertFuncOp("__revert", fnTy, LLVM::Linkage::External, mod);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(mod.getContext(), "__revert");
}

LLVM::AddressOfOp
eravm::BuilderHelper::getCallDataSizeAddr(ModuleOp mod,
                                          std::optional<Location> loc) {
  LLVM::GlobalOp globCallDataSzDef = h.getOrInsertIntGlobalOp(
      eravm::GlobCallDataSize, mod, eravm::AddrSpace_Stack);
  return b.create<LLVM::AddressOfOp>(loc ? *loc : defLoc, globCallDataSzDef);
}

LLVM::AddressOfOp
eravm::BuilderHelper::getCallDataPtrAddr(ModuleOp mod,
                                         std::optional<Location> loc) {
  LLVM::GlobalOp callDataPtrDef = h.getOrInsertPtrGlobalOp(
      eravm::GlobCallDataPtr, mod, eravm::AddrSpace_Generic);
  return b.create<LLVM::AddressOfOp>(loc ? *loc : defLoc, callDataPtrDef);
}

LLVM::LoadOp
eravm::BuilderHelper::loadCallDataPtr(ModuleOp mod,
                                      std::optional<Location> loc) {
  LLVM::AddressOfOp callDataPtrAddr = getCallDataPtrAddr(mod, loc);
  return b.create<LLVM::LoadOp>(loc ? *loc : defLoc, callDataPtrAddr,
                                eravm::getAlignment(callDataPtrAddr));
}
