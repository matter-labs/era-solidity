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

#include "libsolidity/codegen/mlir/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include <vector>

using namespace mlir;
using namespace solidity::mlirgen;

LLVM::GlobalOp BuilderHelper::getGlobalOp(llvm::StringRef name, ModuleOp mod) {
  LLVM::GlobalOp found = mod.lookupSymbol<LLVM::GlobalOp>(name);
  assert(found);
  return found;
}

LLVM::GlobalOp
BuilderHelper::getOrInsertGlobalOp(llvm::StringRef name, ModuleOp mod, Type ty,
                                   unsigned alignment, unsigned addrSpace,
                                   LLVM::Linkage linkage, Attribute attr) {
  if (LLVM::GlobalOp found = mod.lookupSymbol<LLVM::GlobalOp>(name))
    return found;

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(mod.getBody());
  return b.create<LLVM::GlobalOp>(b.getUnknownLoc(), ty, /*isConstant=*/false,
                                  linkage, name, attr, alignment, addrSpace);
}

LLVM::GlobalOp BuilderHelper::getOrInsertIntGlobalOp(llvm::StringRef name,
                                                     ModuleOp mod,
                                                     unsigned addrSpace,
                                                     unsigned width,
                                                     LLVM::Linkage linkage) {
  auto ty = b.getIntegerType(width);
  assert(width > 2 && width % 8 == 0);
  unsigned alignment = width / 8;

  return getOrInsertGlobalOp(name, mod, ty, alignment, addrSpace, linkage,
                             b.getIntegerAttr(ty, 0));
}

LLVM::GlobalOp BuilderHelper::getOrInsertPtrGlobalOp(llvm::StringRef name,
                                                     ModuleOp mod,
                                                     unsigned addrSpace,
                                                     LLVM::Linkage linkage) {
  auto ty = LLVM::LLVMPointerType::get(mod.getContext(), addrSpace);
  // FIXME: What attribute corresponds to llvm's null?
  return getOrInsertGlobalOp(name, mod, ty, 0, 0, linkage, {});
}

ArrayAttr BuilderHelper::getZeroInitialzedAttr(IntegerType ty, unsigned sz) {
  std::vector<Attribute> attrs(sz, b.getIntegerAttr(ty, 0));
  return b.getArrayAttr(attrs);
}

LLVM::LLVMFuncOp BuilderHelper::getOrInsertLLVMFuncOp(
    llvm::StringRef name, Type resTy, llvm::ArrayRef<Type> argTys, ModuleOp mod,
    LLVM::Linkage linkage, llvm::ArrayRef<NamedAttribute> attrs) {
  if (LLVM::LLVMFuncOp found = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return found;

  auto fnType = LLVM::LLVMFunctionType::get(resTy, argTys);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(mod.getBody());
  return b.create<LLVM::LLVMFuncOp>(mod.getLoc(), name, fnType, linkage,
                                    /*dsoLocal=*/false, LLVM::CConv::C, attrs);
}

void BuilderHelper::createCallToUnreachableWrapper(Location loc, ModuleOp mod) {
  LLVM::LLVMFuncOp func = getOrInsertLLVMFuncOp(
      ".unreachable", LLVM::LLVMVoidType::get(mod.getContext()), {}, mod,
      LLVM::Linkage::Private);
  b.create<func::CallOp>(loc,
                         SymbolRefAttr::get(mod.getContext(), ".unreachable"),
                         TypeRange{}, ValueRange{});

  // Define the wrapper if haven't already
  if (func.getBody().empty()) {
    Block *blk = b.createBlock(&func.getBody());
    OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(blk);
    b.create<LLVM::UnreachableOp>(loc);
  }
}
