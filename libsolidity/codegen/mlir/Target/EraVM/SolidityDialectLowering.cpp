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
#include "libsolidity/codegen/mlir/Target/EraVM/Util.h"
#include "libsolidity/codegen/mlir/Util.h"
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IntrinsicsEraVM.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <climits>
#include <vector>

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

/// Returns true if `op` is defined in a runtime context
static bool inRuntimeContext(Operation *op) {
  assert(!isa<LLVM::LLVMFuncOp>(op) && !isa<sol::ObjectOp>(op));

  // Check if the parent FuncOp has isRuntime attribute set
  auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
  if (parentFunc) {
    auto isRuntimeAttr = parentFunc->getAttr("isRuntime");
    assert(isRuntimeAttr);
    return isRuntimeAttr.cast<BoolAttr>().getValue();
    // TODO: The following doesn't work. Find the rationale (or fix?) for the
    // inconsistent behaviour of llvm::cast and .cast with MLIR data structures
    // return llvm::cast<BoolAttr>(isRuntimeAttr).getValue();
  }

  // If there's no parent FuncOp, check the parent ObjectOp
  auto parentObj = op->getParentOfType<sol::ObjectOp>();
  if (parentObj) {
    return parentObj.getSymName().endswith("_deployed");
  }

  llvm_unreachable("op has no parent FuncOp or ObjectOp");
}

// TODO? Move simple builtin lowering to tblgen (`Pat` records)?

struct CallValOpLowering : public OpRewritePattern<sol::CallValOp> {
  using OpRewritePattern<sol::CallValOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallValOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, /*resTy=*/rewriter.getIntegerType(256),
        rewriter.getI32IntegerAttr(llvm::Intrinsic::eravm_getu128),
        rewriter.getStringAttr("eravm.getu128"));
    return success();
  }
};

struct DataOffsetOpLowering : public OpRewritePattern<sol::DataOffsetOp> {
  using OpRewritePattern<sol::DataOffsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::DataOffsetOp op,
                                PatternRewriter &rewriter) const override {
    // FIXME:
    // - Handle references to objects outside the current module
    // - Check if the reference object is valid
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getIntegerType(256), 0));
    return success();
  }
};

struct DataSizeOpLowering : public OpRewritePattern<sol::DataSizeOp> {
  using OpRewritePattern<sol::DataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::DataSizeOp op,
                                PatternRewriter &rewriter) const override {
    // FIXME:
    // - Handle references to objects outside the current module
    // - Check if the reference object is valid
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getIntegerType(256), 0));
    return success();
  }
};

struct CodeCopyOpLowering : public OpRewritePattern<sol::CodeCopyOp> {
  using OpRewritePattern<sol::CodeCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CodeCopyOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    solidity::mlirgen::BuilderHelper h(rewriter);
    eravm::BuilderHelper eravmHelper(rewriter);

    assert(!inRuntimeContext(op) &&
           "codecopy is not supported in runtime context");

    Value dstOffset = op.getInp0();
    Value srcOffset = op.getInp1();
    Value size = op.getInp2();

    // Generate the destination pointer
    auto heapAddrSpacePtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                                         eravm::AddrSpace_Heap);
    auto dst =
        rewriter.create<LLVM::IntToPtrOp>(loc, heapAddrSpacePtrTy, dstOffset);

    // Generate the source pointer
    LLVM::GlobalOp callDataPtrDef = h.getGlobalOp(eravm::GlobCallDataPtr, mod);
    auto callDataPtrAddr =
        rewriter.create<LLVM::AddressOfOp>(loc, callDataPtrDef);
    unsigned callDataPtrAddrSpace = callDataPtrDef.getType()
                                        .cast<LLVM::LLVMPointerType>()
                                        .getAddressSpace();
    LLVM::LoadOp callDataPtr = eravmHelper.genLoad(loc, callDataPtrAddr);
    auto src = rewriter.create<LLVM::GEPOp>(
        loc,
        /*resultType=*/
        LLVM::LLVMPointerType::get(mod.getContext(), callDataPtrAddrSpace),
        /*basePtrType=*/rewriter.getIntegerType(eravm::BitLen_Byte),
        callDataPtr, ValueRange{srcOffset});

    // Generate the memcpy
    Value isVolatile = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
    rewriter.create<LLVM::MemcpyOp>(loc, dst, src, size, isVolatile);

    rewriter.eraseOp(op);
    return success();
  }
};

struct MLoadOpLowering : public OpRewritePattern<sol::MLoadOp> {
  using OpRewritePattern<sol::MLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MLoadOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();

    mlir::Value offsetInt = op.getInp0();
    auto heapAddrSpacePtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                                         eravm::AddrSpace_Heap);
    mlir::Value offset =
        rewriter.create<LLVM::IntToPtrOp>(loc, heapAddrSpacePtrTy, offsetInt);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, rewriter.getIntegerType(256),
                                              offset,
                                              /*alignment=*/1);

    return success();
  }
};

struct MStoreOpLowering : public OpRewritePattern<sol::MStoreOp> {
  using OpRewritePattern<sol::MStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MStoreOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();

    mlir::Value offsetInt = op.getInp0();
    mlir::Value val = op.getInp1();
    auto heapAddrSpacePtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                                         eravm::AddrSpace_Heap);
    mlir::Value offset =
        rewriter.create<LLVM::IntToPtrOp>(loc, heapAddrSpacePtrTy, offsetInt);
    rewriter.create<LLVM::StoreOp>(loc, val, offset, /*alignment=*/1);

    rewriter.eraseOp(op);
    return success();
  }
};

struct MemGuardOpLowering : public OpRewritePattern<sol::MemGuardOp> {
  using OpRewritePattern<sol::MemGuardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MemGuardOp op,
                                PatternRewriter &rewriter) const override {
    auto inp = op->getAttrOfType<IntegerAttr>("inp");
    assert(inp);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, inp);
    return success();
  }
};

struct RevertOpLowering : public OpRewritePattern<sol::RevertOp> {
  using OpRewritePattern<sol::RevertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::RevertOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    solidity::mlirgen::BuilderHelper h(rewriter);
    eravm::BuilderHelper eravmHelper(rewriter);

    // Create the revert call (__revert(offset, length, RetForwardPageType)) and
    // the unreachable op
    SymbolRefAttr revertFunc = eravmHelper.getOrInsertRevert(mod);
    rewriter.create<func::CallOp>(
        loc, revertFunc, TypeRange{},
        ValueRange{
            op.getInp0(), op.getInp1(),
            h.getConst(loc, inRuntimeContext(op)
                                ? eravm::RetForwardPageType::UseHeap
                                : eravm::RetForwardPageType::UseAuxHeap)});
    h.createCallToUnreachableWrapper(loc, mod);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<sol::ReturnOp> {
  using OpRewritePattern<sol::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();

    solidity::mlirgen::BuilderHelper h(rewriter);
    eravm::BuilderHelper eravmHelper(rewriter);
    SymbolRefAttr returnFunc =
        eravmHelper.getOrInsertReturn(op->getParentOfType<ModuleOp>());

    //
    // Lowering in the runtime context
    //
    if (inRuntimeContext(op)) {
      // Create the return call (__return(offset, length,
      // RetForwardPageType::UseHeap)) and the unreachable op
      rewriter.create<func::CallOp>(
          loc, returnFunc, TypeRange{},
          ValueRange{op.getLhs(), op.getRhs(),
                     h.getConst(loc, eravm::RetForwardPageType::UseHeap)});
      h.createCallToUnreachableWrapper(loc, mod);

      rewriter.eraseOp(op);
      return success();
    }

    //
    // Lowering in the creation context
    //
    auto heapAuxAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), eravm::AddrSpace_HeapAuxiliary);

    // Store ByteLen_Field to the immutables offset
    auto immutablesOffsetPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        h.getConst(loc, eravm::HeapAuxOffsetCtorRetData));
    rewriter.create<LLVM::StoreOp>(loc, h.getConst(loc, eravm::ByteLen_Field),
                                   immutablesOffsetPtr, /*alignment=*/32);

    // Store size of immutables in terms of ByteLen_Field to the immutables
    // number offset
    auto immutablesSize = 0; // TODO: Implement this!
    auto immutablesNumPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        h.getConst(loc,
                   eravm::HeapAuxOffsetCtorRetData + eravm::ByteLen_Field));
    rewriter.create<LLVM::StoreOp>(
        loc, h.getConst(loc, immutablesSize / eravm::ByteLen_Field),
        immutablesNumPtr, /*alignment=*/32);

    // Calculate the return data length (i.e. immutablesSize * 2 +
    // ByteLen_Field * 2
    auto immutablesCalcSize = rewriter.create<arith::MulIOp>(
        loc, h.getConst(loc, immutablesSize), h.getConst(loc, 2));
    auto returnDataLen = rewriter.create<arith::AddIOp>(
        loc, immutablesCalcSize.getResult(),
        h.getConst(loc, eravm::ByteLen_Field * 2));

    // Create the return call (__return(HeapAuxOffsetCtorRetData, returnDataLen,
    // RetForwardPageType::UseAuxHeap)) and the unreachable op
    rewriter.create<func::CallOp>(
        loc, returnFunc, TypeRange{},
        ValueRange{h.getConst(loc, eravm::HeapAuxOffsetCtorRetData),
                   returnDataLen.getResult(),
                   h.getConst(loc, eravm::RetForwardPageType::UseAuxHeap)});
    h.createCallToUnreachableWrapper(loc, mod);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ObjectOpLowering : public OpRewritePattern<sol::ObjectOp> {
  using OpRewritePattern<sol::ObjectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::ObjectOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    auto mod = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(op->getContext());
    auto i256Ty = rewriter.getIntegerType(256);
    solidity::mlirgen::BuilderHelper h(rewriter);
    eravm::BuilderHelper eravmHelper(rewriter);

    // Move FuncOps under the ModuleOp
    op.walk([&](func::FuncOp fn) {
      Block *modBlk = mod.getBody();
      fn->moveBefore(modBlk, modBlk->begin());
    });

    // Is this a runtime object?
    // FIXME: Is there a better way to check this?
    if (op.getSymName().endswith("_deployed")) {
      // Move the runtime object region under the __runtime function
      LLVM::LLVMFuncOp runtimeFunc =
          eravmHelper.getOrInsertRuntimeFuncOp("__runtime", voidTy, {}, mod);
      Region &runtimeFuncRegion = runtimeFunc.getRegion();
      assert(runtimeFuncRegion.empty());
      rewriter.inlineRegionBefore(op.getRegion(), runtimeFuncRegion,
                                  runtimeFuncRegion.begin());
      rewriter.eraseOp(op);
      return success();
    }

    // Define the __entry function
    auto genericAddrSpacePtrTy = LLVM::LLVMPointerType::get(
        rewriter.getContext(), eravm::AddrSpace_Generic);
    std::vector<Type> inTys{genericAddrSpacePtrTy};
    constexpr unsigned argCnt =
        eravm::MandatoryArgCnt + eravm::ExtraABIDataSize;
    for (unsigned i = 0; i < argCnt - 1; ++i) {
      inTys.push_back(i256Ty);
    }
    FunctionType funcType = rewriter.getFunctionType(inTys, {i256Ty});
    rewriter.setInsertionPointToEnd(mod.getBody());
    func::FuncOp entryFunc =
        rewriter.create<func::FuncOp>(loc, "__entry", funcType);
    assert(op->getNumRegions() == 1);

    // Setup the entry block and set insertion point to it
    auto &entryFuncRegion = entryFunc.getRegion();
    Block *entryBlk = rewriter.createBlock(&entryFuncRegion);
    for (auto inTy : inTys) {
      entryBlk->addArgument(inTy, loc);
    }
    rewriter.setInsertionPointToStart(entryBlk);

    // Initialize globals
    eravmHelper.initGlobs(loc, mod);

    // Store the calldata ABI arg to the global calldata ptr
    LLVM::GlobalOp globCallDataPtrDef = h.getOrInsertPtrGlobalOp(
        eravm::GlobCallDataPtr, mod, eravm::AddrSpace_Generic);
    Value globCallDataPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, globCallDataPtrDef);
    rewriter.create<LLVM::StoreOp>(
        loc, entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallDataABI),
        globCallDataPtr, /*alignment=*/32);

    // Store the calldata ABI size to the global calldata size
    Value abiLen = eravmHelper.getABILen(loc, globCallDataPtr);
    LLVM::GlobalOp globCallDataSzDef =
        h.getGlobalOp(eravm::GlobCallDataSize, mod);
    Value globCallDataSz =
        rewriter.create<LLVM::AddressOfOp>(loc, globCallDataSzDef);
    rewriter.create<LLVM::StoreOp>(loc, abiLen, globCallDataSz,
                                   /*alignment=*/32);

    // Store calldatasize[calldata abi arg] to the global ret data ptr and
    // active ptr
    LLVM::LoadOp callDataSz = eravmHelper.genLoad(loc, globCallDataSz);
    auto retDataABIInitializer = rewriter.create<LLVM::GEPOp>(
        loc,
        /*resultType=*/
        LLVM::LLVMPointerType::get(mod.getContext(),
                                   globCallDataPtrDef.getAddrSpace()),
        /*basePtrType=*/rewriter.getIntegerType(eravm::BitLen_Byte),
        entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallDataABI),
        callDataSz.getResult());
    auto storeRetDataABIInitializer = [&](const char *name) {
      LLVM::GlobalOp globDef =
          h.getOrInsertPtrGlobalOp(name, mod, eravm::AddrSpace_Generic);
      Value globAddr = rewriter.create<LLVM::AddressOfOp>(loc, globDef);
      rewriter.create<LLVM::StoreOp>(loc, retDataABIInitializer, globAddr,
                                     /*alignment=*/32);
    };
    storeRetDataABIInitializer(eravm::GlobRetDataPtr);
    storeRetDataABIInitializer(eravm::GlobActivePtr);

    // Store call flags arg to the global call flags
    auto globCallFlagsDef = h.getGlobalOp(eravm::GlobCallFlags, mod);
    Value globCallFlags =
        rewriter.create<LLVM::AddressOfOp>(loc, globCallFlagsDef);
    rewriter.create<LLVM::StoreOp>(
        loc, entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallFlags),
        globCallFlags, /*alignment=*/32);

    // Store the remaining args to the global extra ABI data
    auto globExtraABIDataDef = h.getGlobalOp(eravm::GlobExtraABIData, mod);
    Value globExtraABIData =
        rewriter.create<LLVM::AddressOfOp>(loc, globExtraABIDataDef);
    for (unsigned i = 2; i < entryBlk->getNumArguments(); ++i) {
      auto gep = rewriter.create<LLVM::GEPOp>(
          loc,
          /*resultType=*/
          LLVM::LLVMPointerType::get(mod.getContext(),
                                     globExtraABIDataDef.getAddrSpace()),
          /*basePtrType=*/globExtraABIDataDef.getType(), globExtraABIData,
          ValueRange{h.getConst(loc, 0), h.getConst(loc, i - 2)});
      // FIXME: How does the opaque ptr geps with scalar element types lower
      // without explictly setting the elem_type attr?
      gep.setElemTypeAttr(TypeAttr::get(globExtraABIDataDef.getType()));
      rewriter.create<LLVM::StoreOp>(loc, entryBlk->getArgument(i), gep,
                                     /*alignment=*/32);
    }

    // Check Deploy call flag
    auto deployCallFlag = rewriter.create<arith::AndIOp>(
        loc, entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallFlags),
        h.getConst(loc, 1));
    auto isDeployCallFlag = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, deployCallFlag.getResult(),
        h.getConst(loc, 1));

    // Create the __runtime function
    LLVM::LLVMFuncOp runtimeFunc =
        eravmHelper.getOrInsertRuntimeFuncOp("__runtime", voidTy, {}, mod);
    Region &runtimeFuncRegion = runtimeFunc.getRegion();
    // Move the runtime object getter under the ObjectOp public API
    for (auto const &op : *op.getBody()) {
      if (auto runtimeObj = llvm::dyn_cast<sol::ObjectOp>(&op)) {
        assert(runtimeObj.getSymName().endswith("_deployed"));
        assert(runtimeFuncRegion.empty());
        rewriter.inlineRegionBefore(runtimeObj.getRegion(), runtimeFuncRegion,
                                    runtimeFuncRegion.begin());
        OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToEnd(&runtimeFuncRegion.front());
        rewriter.create<LLVM::UnreachableOp>(runtimeObj.getLoc());
        rewriter.eraseOp(runtimeObj);
      }
    }

    // Create the __deploy function
    LLVM::LLVMFuncOp deployFunc =
        eravmHelper.getOrInsertCreationFuncOp("__deploy", voidTy, {}, mod);
    Region &deployFuncRegion = deployFunc.getRegion();
    assert(deployFuncRegion.empty());
    rewriter.inlineRegionBefore(op.getRegion(), deployFuncRegion,
                                deployFuncRegion.begin());
    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToEnd(&deployFuncRegion.front());
      rewriter.create<LLVM::UnreachableOp>(loc);
    }

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
      : ConversionPattern(sol::ContractOp::getOperationName(),
                          /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto contOp = cast<sol::ContractOp>(op);
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
    LLVMConversionTarget llConv(getContext());
    llConv.addLegalOp<ModuleOp>();
    llConv.addLegalOp<scf::YieldOp>();
    LLVMTypeConverter llTyConv(&getContext());

    RewritePatternSet pats(&getContext());
    arith::populateArithmeticToLLVMConversionPatterns(llTyConv, pats);
    populateMemRefToLLVMConversionPatterns(llTyConv, pats);
    populateSCFToControlFlowConversionPatterns(pats);
    cf::populateControlFlowToLLVMConversionPatterns(llTyConv, pats);
    populateFuncToLLVMConversionPatterns(llTyConv, pats);
    pats.add<ObjectOpLowering, ReturnOpLowering, RevertOpLowering,
             MLoadOpLowering, MStoreOpLowering, DataOffsetOpLowering,
             DataSizeOpLowering, CodeCopyOpLowering, MemGuardOpLowering,
             CallValOpLowering>(&getContext());

    ModuleOp mod = getOperation();
    if (failed(applyFullConversion(mod, llConv, std::move(pats))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> sol::createSolidityDialectLoweringPassForEraVM() {
  return std::make_unique<SolidityDialectLowering>();
}
