#pragma once

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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

namespace solidity {
namespace mlirgen {

class BuilderHelper {
  mlir::OpBuilder &b;

public:
  explicit BuilderHelper(mlir::OpBuilder &b) : b(b) {}

  mlir::Value getConst(int64_t val, mlir::Location loc, unsigned width = 256) {
    mlir::IntegerType ty = b.getIntegerType(width);
    auto op = b.create<mlir::arith::ConstantOp>(
        loc, b.getIntegerAttr(ty, llvm::APInt(width, val, /*isSigned=*/true)));
    return op.getResult();
  }

  // FIXME: How do we create a constant int array? What's wrong with using
  // LLVM::LLVMArrayType instead of VectorType here?  Is
  // https://github.com/llvm/llvm-project/pull/65508 the only way?
  mlir::Value getConstSplat(llvm::ArrayRef<llvm::APInt> vals,
                            mlir::Location loc, unsigned width = 256) {
    auto ty = mlir::VectorType::get(vals.size(), b.getIntegerType(width));
    auto attr = mlir::DenseIntElementsAttr::get(ty, vals);
    auto op = b.create<mlir::LLVM::ConstantOp>(loc, ty, attr);
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
      llvm::StringRef name, mlir::ModuleOp mod, unsigned addrSpace = 0,
      mlir::LLVM::Linkage linkage = mlir::LLVM::Linkage::Private);

  mlir::ArrayAttr getZeroInitialzedAttr(mlir::IntegerType ty, unsigned sz);

  /// Returns an existing or a new (if not found) LLVM::FuncOp
  mlir::LLVM::LLVMFuncOp getOrInsertLLVMFuncOp(
      llvm::StringRef name, mlir::Type resTy, llvm::ArrayRef<mlir::Type> argTys,
      mlir::ModuleOp mod,
      mlir::LLVM::Linkage linkage = mlir::LLVM::Linkage::External,
      llvm::ArrayRef<mlir::NamedAttribute> attrs = {});
};

} // namespace mlirgen
} // namespace solidity
