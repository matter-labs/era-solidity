#include "libsolidity/codegen/mlir/Util.h"
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
