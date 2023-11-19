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

void eravm::BuilderHelper::initGlobs(ModuleOp mod, Location loc) {

  auto initInt = [&](const char *name) {
    LLVM::GlobalOp globOp =
        h.getOrInsertIntGlobalOp(name, mod, AddrSpace_Stack);
    Value globAddr = b.create<LLVM::AddressOfOp>(loc, globOp);
    b.create<LLVM::StoreOp>(loc, h.getConst(loc, 0), globAddr,
                            /*alignment=*/32);
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
      /*alignment=*/32, AddrSpace_Stack, LLVM::Linkage::Private,
      b.getZeroAttr(RankedTensorType::get({10}, i256Ty)));
  Value extraABIDataAddr = b.create<LLVM::AddressOfOp>(loc, extraABIData);
  b.create<LLVM::StoreOp>(
      loc,
      h.getConstSplat(loc, std::vector<llvm::APInt>(10, llvm::APInt(256, 0))),
      extraABIDataAddr);
}

Value eravm::BuilderHelper::getABILen(Value ptr, Location loc) {
  auto i256Ty = b.getIntegerType(256);

  Value ptrToInt = b.create<LLVM::PtrToIntOp>(loc, i256Ty, ptr).getResult();
  Value lShr = b.create<LLVM::LShrOp>(loc, ptrToInt,
                                      h.getConst(loc, eravm::BitLen_X32 * 3));
  return b.create<LLVM::AndOp>(loc, lShr, h.getConst(loc, UINT_MAX));
}

LLVM::LoadOp eravm::BuilderHelper::genLoad(Location loc, Value addr) {
  auto addrOp = llvm::cast<LLVM::AddressOfOp>(addr.getDefiningOp());
  LLVM::GlobalOp globOp = addrOp.getGlobal();
  assert(globOp);
  AddrSpace addrSpace = static_cast<AddrSpace>(globOp.getAddrSpace());
  unsigned alignment =
      addrSpace == AddrSpace_Stack ? ByteLen_Field : ByteLen_Byte;
  return b.create<LLVM::LoadOp>(loc, addrOp, alignment);
}

LLVM::LLVMFuncOp eravm::BuilderHelper::getOrInsertCreationFuncOp(
    llvm::StringRef name, Type resTy, llvm::ArrayRef<Type> argTys,
    ModuleOp mod) {

  return h.getOrInsertLLVMFuncOp(
      name, resTy, argTys, mod, LLVM::Linkage::Private,
      {NamedAttribute{b.getStringAttr("isRuntime"), b.getBoolAttr(false)}});
}

LLVM::LLVMFuncOp
eravm::BuilderHelper::getOrInsertRuntimeFuncOp(llvm::StringRef name, Type resTy,
                                               llvm::ArrayRef<Type> argTys,
                                               ModuleOp mod) {

  return h.getOrInsertLLVMFuncOp(
      name, resTy, argTys, mod, LLVM::Linkage::Private,
      {NamedAttribute{b.getStringAttr("isRuntime"), b.getBoolAttr(true)}});
}

SymbolRefAttr eravm::BuilderHelper::getOrInsertReturn(ModuleOp mod) {
  auto *ctx = mod.getContext();
  auto i256Ty = IntegerType::get(ctx, 256);
  h.getOrInsertLLVMFuncOp("__return", LLVM::LLVMVoidType::get(ctx),
                          {i256Ty, i256Ty, i256Ty}, mod);
  return SymbolRefAttr::get(mod.getContext(), "__return");
}
