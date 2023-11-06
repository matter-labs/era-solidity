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
// Solidity dialect lowering pass
//

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Solidity/SolidityOps.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>

using namespace mlir;

namespace {

// FIXME: The high level dialects are lowered to the llvm dialect tailored to
// the EraVM backend in llvm. How should we perform the lowering when we support
// other targets?
//
// (a) If we do a condition lowering in this pass, the code can quickly get
// messy
//
// (b) If we have a high level dialect for each target, the lowering will be,
// for instance, solidity.object -> eravm.object -> llvm.func with eravm
// details. Unnecessary abstractions?
//
// (c) I think a sensible design is to create different ModuleOp passes for each
// target that lower high level dialects to the llvm dialect.
//

namespace eravm {

enum AddrSpace : unsigned {
  Stack = 0,
  Heap = 1,
  HeapAuxiliary = 2,
  Generic = 3,
  Code = 4,
  Storage = 5,
};

enum ByteLen { Byte = 1, X32 = 4, X64 = 8, EthAddr = 20, Field = 32 };
enum : unsigned { HeapAuxOffsetCtorRetData = ByteLen::Field * 8 };

enum EntryInfo {
  ArgIndexCallDataABI = 0,
  ArgIndexCallFlags = 1,
  MandatoryArgCnt = 2,
};

} // namespace eravm

class BuilderHelper {
  OpBuilder &b;
  Location loc;

public:
  explicit BuilderHelper(OpBuilder &b, Location loc) : b(b), loc(loc) {}

  Value getConst(int64_t val, unsigned width = 256) {
    IntegerType ty = b.getIntegerType(width);
    auto op = b.create<arith::ConstantOp>(
        loc, b.getIntegerAttr(ty, llvm::APInt(width, val, /*radix=*/10)));
    return op.getResult();
  }
};

static LLVM::LLVMFuncOp
getOrInsertLLVMFuncOp(llvm::StringRef name, Type resTy,
                      llvm::ArrayRef<Type> argTys, OpBuilder &b, ModuleOp mod,
                      LLVM::Linkage linkage = LLVM::Linkage::External,
                      llvm::ArrayRef<NamedAttribute> attrs = {}) {
  if (LLVM::LLVMFuncOp found = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return found;

  auto fnType = LLVM::LLVMFunctionType::get(resTy, argTys);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(mod.getBody());
  return b.create<LLVM::LLVMFuncOp>(mod.getLoc(), name, fnType, linkage,
                                    /*dsoLocal=*/false, LLVM::CConv::C, attrs);
}

static LLVM::LLVMFuncOp getOrInsertCreationFuncOp(llvm::StringRef name,
                                                  Type resTy,
                                                  llvm::ArrayRef<Type> argTys,
                                                  OpBuilder &b, ModuleOp mod) {

  return getOrInsertLLVMFuncOp(
      name, resTy, argTys, b, mod, LLVM::Linkage::Private,
      {NamedAttribute{b.getStringAttr("isRuntime"), b.getBoolAttr(false)}});
}

static LLVM::LLVMFuncOp getOrInsertRuntimeFuncOp(llvm::StringRef name,
                                                 Type resTy,
                                                 llvm::ArrayRef<Type> argTys,
                                                 OpBuilder &b, ModuleOp mod) {

  return getOrInsertLLVMFuncOp(
      name, resTy, argTys, b, mod, LLVM::Linkage::Private,
      {NamedAttribute{b.getStringAttr("isRuntime"), b.getBoolAttr(true)}});
}

static SymbolRefAttr getOrInsertLLVMFuncSym(llvm::StringRef name, Type resTy,
                                            llvm::ArrayRef<Type> argTys,
                                            OpBuilder &b, ModuleOp mod) {
  getOrInsertLLVMFuncOp(name, resTy, argTys, b, mod);
  return SymbolRefAttr::get(mod.getContext(), name);
}

static SymbolRefAttr getOrInsertReturn(PatternRewriter &rewriter,
                                       ModuleOp mod) {
  auto *ctx = mod.getContext();
  auto i256Ty = IntegerType::get(ctx, 256);
  return getOrInsertLLVMFuncSym("__return", LLVM::LLVMVoidType::get(ctx),
                                {i256Ty, i256Ty, i256Ty}, rewriter, mod);
}

class ReturnOpLowering : public ConversionPattern {
public:
  explicit ReturnOpLowering(MLIRContext *ctx)
      : ConversionPattern(solidity::ReturnOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    BuilderHelper b(rewriter, loc);

    auto heapAuxAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), eravm::AddrSpace::HeapAuxiliary);

    // Store ByteLen::Field to the immutables offset
    auto immutablesOffsetPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        b.getConst(eravm::HeapAuxOffsetCtorRetData));
    rewriter.create<LLVM::StoreOp>(loc, b.getConst(eravm::ByteLen::Field),
                                   immutablesOffsetPtr);

    // Store size of immutables in terms of ByteLen::Field to the immutables
    // number offset
    auto immutablesSize = 0; // TODO: Implement this!
    auto immutablesNumPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        b.getConst(eravm::HeapAuxOffsetCtorRetData + eravm::ByteLen::Field));
    rewriter.create<LLVM::StoreOp>(
        loc, b.getConst(immutablesSize / eravm::ByteLen::Field),
        immutablesNumPtr);

    // Calculate the return data length (i.e. immutablesSize * 2 +
    // ByteLen::Field * 2
    auto immutablesCalcSize = rewriter.create<arith::MulIOp>(
        loc, b.getConst(immutablesSize), b.getConst(2));
    auto returnDataLen =
        rewriter.create<arith::AddIOp>(loc, immutablesCalcSize.getResult(),
                                       b.getConst(eravm::ByteLen::Field * 2));
    auto returnFunc =
        getOrInsertReturn(rewriter, op->getParentOfType<ModuleOp>());

    // Create the call: __return(HeapAuxOffsetCtorRetData, returnDataLen,
    // returnOpMode)
    bool isCreation = true; // TODO: Implement this!
    auto returnOpMode = b.getConst(isCreation ? eravm::AddrSpace::HeapAuxiliary
                                              : eravm::AddrSpace::Heap);
    rewriter.create<func::CallOp>(
        loc, returnFunc, TypeRange{},
        ValueRange{b.getConst(eravm::HeapAuxOffsetCtorRetData),
                   returnDataLen.getResult(), returnOpMode});

    // Create unreachable
    rewriter.create<LLVM::UnreachableOp>(loc);

    rewriter.eraseOp(op);
    return success();
  } // namespace
};

class ObjectOpLowering : public ConversionPattern {
public:
  explicit ObjectOpLowering(MLIRContext *ctx)
      : ConversionPattern(solidity::ObjectOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto objOp = cast<mlir::solidity::ObjectOp>(op);
    auto loc = op->getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(op->getContext());
    auto i256Ty = rewriter.getIntegerType(256);

    // Is this a runtime object?
    // FIXME: Is there a better way to check this?
    if (objOp.getSymName().endswith("_deployed")) {
      // Move the runtime object region under the __runtime function
      auto runtimeFunc =
          getOrInsertRuntimeFuncOp("__runtime", voidTy, {}, rewriter, mod);
      Region &runtimeFuncRegion = runtimeFunc.getRegion();
      rewriter.inlineRegionBefore(objOp.getRegion(), runtimeFuncRegion,
                                  runtimeFuncRegion.begin());
      rewriter.eraseOp(op);
      return success();
    }

    auto genericAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), eravm::AddrSpace::Generic);

    std::vector<Type> inTys{genericAddrSpacePtrTy};
    constexpr unsigned argCnt = 2 /* Entry::MANDATORY_ARGUMENTS_COUNT */ +
                                10 /* eravm::EXTRA_ABI_DATA_SIZE */;
    for (unsigned i = 0; i < argCnt - 1; ++i) {
      inTys.push_back(i256Ty);
    }
    FunctionType funcType = rewriter.getFunctionType(inTys, {i256Ty});
    rewriter.setInsertionPointToEnd(mod.getBody());
    func::FuncOp entryFunc =
        rewriter.create<func::FuncOp>(loc, "__entry", funcType);
    assert(op->getNumRegions() == 1);

    auto &entryFuncRegion = entryFunc.getRegion();
    Block *entryBlk = rewriter.createBlock(&entryFuncRegion);
    for (auto inTy : inTys) {
      entryBlk->addArgument(inTy, loc);
    }

    // Check Deploy call flag
    rewriter.setInsertionPointToStart(entryBlk);
    BuilderHelper b(rewriter, loc);
    auto deployCallFlag = rewriter.create<arith::AndIOp>(
        loc, entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallFlags),
        b.getConst(1));
    auto isDeployCallFlag = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, deployCallFlag.getResult(),
        b.getConst(1));

    // Create the __runtime function
    auto runtimeFunc =
        getOrInsertRuntimeFuncOp("__runtime", voidTy, {}, rewriter, mod);
    Region &runtimeFuncRegion = runtimeFunc.getRegion();
    // Move the runtime object getter under the ObjectOp public API
    for (auto const &op : *objOp.getBody()) {
      if (auto runtimeObj = llvm::dyn_cast<solidity::ObjectOp>(&op)) {
        assert(runtimeObj.getSymName().endswith("_deployed"));
        rewriter.inlineRegionBefore(runtimeObj.getRegion(), runtimeFuncRegion,
                                    runtimeFuncRegion.begin());
        rewriter.eraseOp(runtimeObj);
      }
    }

    // Create the __deploy function
    auto deployFunc =
        getOrInsertCreationFuncOp("__deploy", voidTy, {}, rewriter, mod);
    Region &deployFuncRegion = deployFunc.getRegion();
    rewriter.inlineRegionBefore(objOp.getRegion(), deployFuncRegion,
                                deployFuncRegion.begin());

    // If the deploy call flag is set, call __deploy()
    auto ifOp = rewriter.create<scf::IfOp>(loc, isDeployCallFlag.getResult(),
                                           /*withElseRegion=*/true);
    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
    thenBuilder.create<LLVM::CallOp>(loc, deployFunc, ValueRange{});
    // FIXME: Why the following fails with a "does not reference a valid
    // function" error but generating the func::CallOp to __return is fine
    // thenBuilder.create<func::CallOp>(
    //     loc, SymbolRefAttr::get(mod.getContext(), "__deploy"), TypeRange{},
    //     ValueRange{});

    // Else call __runtime()
    OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
    elseBuilder.create<LLVM::CallOp>(loc, runtimeFunc, ValueRange{});
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<LLVM::UnreachableOp>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

class ContractOpLowering : public ConversionPattern {
public:
  explicit ContractOpLowering(MLIRContext *ctx)
      : ConversionPattern(solidity::ContractOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto contOp = cast<solidity::ContractOp>(op);
    assert(isa<ModuleOp>(contOp->getParentOp()));
    auto modOp = cast<ModuleOp>(contOp->getParentOp());
    Block *modBody = modOp.getBody();

    // Move functions to the parent ModuleOp
    std::vector<Operation *> funcs;
    for (Operation &func : contOp.getBody()->getOperations()) {
      assert(isa<func::FuncOp>(&func));
      funcs.push_back(&func);
    }
    for (Operation *func : funcs) {
      func->moveAfter(modBody, modBody->begin());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct SolidityDialectLowering
    : public PassWrapper<SolidityDialectLowering, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SolidityDialectLowering)

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<LLVM::LLVMDialect, func::FuncDialect, arith::ArithmeticDialect,
               scf::SCFDialect>();
  }

  void runOnOperation() override {
    // We only lower till the llvm dialect
    LLVMConversionTarget llConv(getContext());
    llConv.addLegalOp<ModuleOp>();
    llConv.addLegalOp<scf::YieldOp>();
    LLVMTypeConverter llTyConv(&getContext());

    // Lower arith, memref and func dialects to the llvm dialect
    RewritePatternSet pats(&getContext());
    arith::populateArithmeticToLLVMConversionPatterns(llTyConv, pats);
    populateSCFToControlFlowConversionPatterns(pats);
    cf::populateControlFlowToLLVMConversionPatterns(llTyConv, pats);
    populateMemRefToLLVMConversionPatterns(llTyConv, pats);
    populateFuncToLLVMConversionPatterns(llTyConv, pats);
    pats.add<ContractOpLowering>(&getContext());
    pats.add<ObjectOpLowering>(&getContext());
    pats.add<ReturnOpLowering>(&getContext());

    ModuleOp mod = getOperation();
    if (failed(applyFullConversion(mod, llConv, std::move(pats))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> solidity::createSolidityDialectLoweringPassForEraVM() {
  return std::make_unique<SolidityDialectLowering>();
}
