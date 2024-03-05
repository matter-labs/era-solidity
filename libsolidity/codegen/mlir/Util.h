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

//
// MLIR utilities
//

#pragma once

#include "Sol/SolOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace solidity {
namespace mlirgen {

class BuilderHelper {
  mlir::OpBuilder &b;
  mlir::Location defLoc;

public:
  explicit BuilderHelper(mlir::OpBuilder &b)
      : b(b), defLoc(b.getUnknownLoc()) {}

  explicit BuilderHelper(mlir::OpBuilder &b, mlir::Location loc)
      : b(b), defLoc(loc) {}

  mlir::Value getConst(int64_t val, unsigned width = 256,
                       std::optional<mlir::Location> locArg = std::nullopt) {
    mlir::IntegerType ty = b.getIntegerType(width);
    auto op = b.create<mlir::arith::ConstantOp>(
        locArg ? *locArg : defLoc,
        b.getIntegerAttr(ty, llvm::APInt(width, val, /*isSigned=*/true)));
    return op.getResult();
  }

  mlir::Value getConst(std::string val, unsigned width = 256,
                       std::optional<mlir::Location> locArg = std::nullopt) {
    mlir::IntegerType ty = b.getIntegerType(width);

    uint8_t radix = 10;
    llvm::StringRef intStr = val;
    if (intStr.consume_front("0x")) {
      radix = 16;
    }

    auto op = b.create<mlir::arith::ConstantOp>(
        locArg ? *locArg : defLoc,
        b.getIntegerAttr(ty, llvm::APInt(width, intStr, radix)));
    return op.getResult();
  }

  // FIXME: How do we create a constant int array? What's wrong with using
  // LLVM::LLVMArrayType instead of VectorType here?  Is
  // https://github.com/llvm/llvm-project/pull/65508 the only way?
  mlir::Value
  getConstSplat(llvm::ArrayRef<llvm::APInt> vals, unsigned width = 256,
                std::optional<mlir::Location> locArg = std::nullopt) {
    auto ty = mlir::VectorType::get(vals.size(), b.getIntegerType(width));
    auto attr = mlir::DenseIntElementsAttr::get(ty, vals);
    auto op =
        b.create<mlir::LLVM::ConstantOp>(locArg ? *locArg : defLoc, ty, attr);
    return op.getResult();
  }

  /// Returns an existing LLVM::GlobalOp with the name `name`; Assert fails if
  /// not found
  mlir::LLVM::GlobalOp getGlobalOp(llvm::StringRef name, mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) LLVM::GlobalOp
  mlir::LLVM::GlobalOp
  getOrInsertGlobalOp(llvm::StringRef name, mlir::ModuleOp mod, mlir::Type,
                      unsigned alignment, unsigned addrSpace,
                      mlir::LLVM::Linkage, mlir::Attribute);

  /// Returns an existing or a new (if not found) integral LLVM::GlobalOp
  mlir::LLVM::GlobalOp getOrInsertIntGlobalOp(
      llvm::StringRef name, mlir::ModuleOp mod, unsigned addrSpace = 0,
      unsigned width = 256,
      mlir::LLVM::Linkage linkage = mlir::LLVM::Linkage::Private);

  mlir::LLVM::GlobalOp getOrInsertPtrGlobalOp(
      llvm::StringRef name, mlir::ModuleOp mod, unsigned ptrTyAddrSpace = 0,
      mlir::LLVM::Linkage linkage = mlir::LLVM::Linkage::Private);

  mlir::ArrayAttr getZeroInitialzedAttr(mlir::IntegerType ty, unsigned sz);

  /// Returns an existing or a new (if not found) FuncOp in the ModuleOp `mod`.
  mlir::sol::FuncOp
  getOrInsertFuncOp(mlir::StringRef name, mlir::FunctionType fnTy,
                    mlir::LLVM::Linkage linkage, mlir::ModuleOp mod,
                    std::vector<mlir::NamedAttribute> attrs = {});

  /// Creates a call to a wrapper function of the LLVM::UnreachableOp. This is a
  /// hack to create a non-terminator unreachable op
  void createCallToUnreachableWrapper(
      mlir::ModuleOp mod, std::optional<mlir::Location> locArg = std::nullopt);
};

} // namespace mlirgen
} // namespace solidity
