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
#include "mlir/Dialect/Arith/IR/Arith.h"
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

/// Extension of mlir::OpBuilder with APIs helpful for codegen in solidity.
class BuilderExt {
  mlir::OpBuilder &b;
  mlir::Location defLoc;

public:
  explicit BuilderExt(mlir::OpBuilder &b) : b(b), defLoc(b.getUnknownLoc()) {}

  explicit BuilderExt(mlir::OpBuilder &b, mlir::Location loc)
      : b(b), defLoc(loc) {}

  mlir::Value genBool(bool val,
                      std::optional<mlir::Location> locArg = std::nullopt) {
    mlir::IntegerType ty = b.getIntegerType(1);
    auto op = b.create<mlir::arith::ConstantOp>(locArg ? *locArg : defLoc,
                                                b.getIntegerAttr(ty, val));
    return op.getResult();
  }

  mlir::Value genConst(llvm::APInt const &val, unsigned width,
                       std::optional<mlir::Location> locArg = std::nullopt) {
    mlir::IntegerType ty = b.getIntegerType(width);
    auto op = b.create<mlir::arith::ConstantOp>(locArg ? *locArg : defLoc,
                                                b.getIntegerAttr(ty, val));
    return op.getResult();
  }

  mlir::Value genConst(llvm::APInt const &val,
                       std::optional<mlir::Location> locArg = std::nullopt) {
    mlir::IntegerType ty = b.getIntegerType(val.getBitWidth());
    auto op = b.create<mlir::arith::ConstantOp>(locArg ? *locArg : defLoc,
                                                b.getIntegerAttr(ty, val));
    return op.getResult();
  }

  mlir::Value genConst(int64_t val, unsigned width = 64,
                       std::optional<mlir::Location> locArg = std::nullopt) {
    return genConst(llvm::APInt(width, val, /*isSigned=*/true), width, locArg);
  }

  mlir::Value genConst(std::string const &val, unsigned width,
                       std::optional<mlir::Location> locArg = std::nullopt) {
    uint8_t radix = 10;
    llvm::StringRef intStr = val;
    if (intStr.consume_front("0x")) {
      radix = 16;
    }

    return genConst(llvm::APInt(width, intStr, radix), width, locArg);
  }

  mlir::Value
  genI256Const(llvm::APInt const &val,
               std::optional<mlir::Location> locArg = std::nullopt) {

    return genConst(val, 256, locArg);
  }

  mlir::Value
  genI256Const(std::string const &val,
               std::optional<mlir::Location> locArg = std::nullopt) {
    return genConst(val, 256, locArg);
  }

  mlir::Value
  genI256Const(int64_t val,
               std::optional<mlir::Location> locArg = std::nullopt) {
    return genConst(val, 256, locArg);
  }

  /// Generates an arith dialect cast op (if required) (as per the desired width
  /// and signedness) of the signless typed value.
  mlir::Value genIntCast(unsigned width, bool isSigned, mlir::Value val,
                         std::optional<mlir::Location> locArg = std::nullopt);

  mlir::Value
  genCastToIdx(mlir::Value val,
               std::optional<mlir::Location> locArg = std::nullopt) {
    assert(val.getType() == b.getIntegerType(256));
    return b.create<mlir::arith::IndexCastUIOp>(locArg ? *locArg : defLoc,
                                                b.getIndexType(), val);
  }

  mlir::Value
  genCastToI256(mlir::Value val,
                std::optional<mlir::Location> locArg = std::nullopt) {
    // TODO: Support other source types.
    assert(val.getType() == b.getIndexType());
    return b.create<mlir::arith::IndexCastUIOp>(locArg ? *locArg : defLoc,
                                                b.getIntegerType(256), val);
  }

  mlir::Value genIdxConst(int64_t val,
                          std::optional<mlir::Location> locArg = std::nullopt) {
    return b.create<mlir::arith::ConstantOp>(locArg ? *locArg : defLoc,
                                             b.getIndexAttr(val));
  }

  // FIXME: How do we create a constant int array? What's wrong with using
  // LLVM::LLVMArrayType instead of VectorType here?  Is
  // https://github.com/llvm/llvm-project/pull/65508 the only way?
  mlir::Value
  genI256ConstSplat(llvm::ArrayRef<llvm::APInt> vals,
                    std::optional<mlir::Location> locArg = std::nullopt) {
    auto ty = mlir::VectorType::get(vals.size(), b.getIntegerType(256));
    auto attr = mlir::DenseIntElementsAttr::get(ty, vals);
    auto op =
        b.create<mlir::LLVM::ConstantOp>(locArg ? *locArg : defLoc, ty, attr);
    return op.getResult();
  }

  /// Generates the round-up to multiple.
  template <unsigned multiple>
  mlir::Value
  genRoundUpToMultiple(mlir::Value val,
                       std::optional<mlir::Location> locArg = std::nullopt) {
    mlir::Location loc = locArg ? *locArg : defLoc;
    auto add =
        b.create<mlir::arith::AddIOp>(loc, val, genI256Const(multiple - 1));
    return b.create<mlir::arith::AndIOp>(
        loc, add, genI256Const(~(llvm::APInt(/*numBits=*/256, multiple - 1))));
  }

  /// Returns an existing LLVM::GlobalOp; Assert fails if not found.
  mlir::LLVM::GlobalOp getGlobalOp(llvm::StringRef name, mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) LLVM::GlobalOp.
  mlir::LLVM::GlobalOp
  getOrInsertGlobalOp(llvm::StringRef name, mlir::Type, unsigned addrSpace,
                      unsigned alignment, mlir::LLVM::Linkage linkage,
                      mlir::Attribute attr, mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) i256 LLVM::GlobalOp.
  mlir::LLVM::GlobalOp getOrInsertI256GlobalOp(llvm::StringRef name,
                                               unsigned addrSpace,
                                               unsigned alignment,
                                               mlir::LLVM::Linkage linkage,
                                               mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) ptr LLVM::GlobalOp.
  mlir::LLVM::GlobalOp getOrInsertPtrGlobalOp(llvm::StringRef name,
                                              unsigned ptrTyAddrSpace,
                                              mlir::LLVM::Linkage linkage,
                                              mlir::ModuleOp mod);

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
