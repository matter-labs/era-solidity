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

#include "libsolidity/codegen/CompilerUtils.h"
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
#include <utility>
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

struct CallDataLoadOpLowering : public OpRewritePattern<sol::CallDataLoadOp> {
  using OpRewritePattern<sol::CallDataLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallDataLoadOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    solidity::mlirgen::BuilderHelper h(rewriter);
    eravm::BuilderHelper eravmHelper(rewriter);

    Value offset = op.getInp0();

    if (inRuntimeContext(op)) {
      // Generate the `GlobCallDataPtr` + `offset` load
      LLVM::GlobalOp callDataPtrDef = h.getGlobalOp(
          eravm::GlobCallDataPtr, op->getParentOfType<ModuleOp>());
      auto callDataPtrAddr =
          rewriter.create<LLVM::AddressOfOp>(loc, callDataPtrDef);
      unsigned callDataPtrAddrSpace = callDataPtrDef.getType()
                                          .cast<LLVM::LLVMPointerType>()
                                          .getAddressSpace();
      auto callDataPtr = rewriter.create<LLVM::LoadOp>(
          loc, callDataPtrAddr, eravm::getAlignment(callDataPtrAddr));
      auto callDataOffset = rewriter.create<LLVM::GEPOp>(
          loc,
          /*resultType=*/
          LLVM::LLVMPointerType::get(rewriter.getContext(),
                                     callDataPtrAddrSpace),
          /*basePtrType=*/rewriter.getIntegerType(eravm::BitLen_Byte),
          callDataPtr, ValueRange{offset});
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
          op, op.getType(), callDataOffset,
          eravm::getAlignment(callDataOffset));

    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, /*value=*/0,
                                                        /*width=*/256);
    }

    return success();
  }
};

struct CallDataSizeOpLowering : public OpRewritePattern<sol::CallDataSizeOp> {
  using OpRewritePattern<sol::CallDataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallDataSizeOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (inRuntimeContext(op)) {
      // Generate GlobCallDataSize load
      solidity::mlirgen::BuilderHelper h(rewriter);
      LLVM::GlobalOp globCallDataSzDef = h.getGlobalOp(
          eravm::GlobCallDataSize, op->getParentOfType<ModuleOp>());
      auto globCallDataSz =
          rewriter.create<LLVM::AddressOfOp>(loc, globCallDataSzDef);
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
          op, globCallDataSz, eravm::getAlignment(globCallDataSz));
    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, /*value=*/0,
                                                        /*width=*/256);
    }

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
    auto callDataPtr = rewriter.create<LLVM::LoadOp>(
        loc, callDataPtrAddr, eravm::getAlignment(callDataPtrAddr));
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
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, rewriter.getIntegerType(256), offset, eravm::getAlignment(offset));

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
    rewriter.create<LLVM::StoreOp>(loc, val, offset,
                                   eravm::getAlignment(offset));

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
                                   immutablesOffsetPtr,
                                   eravm::getAlignment(immutablesOffsetPtr));

    // Store size of immutables in terms of ByteLen_Field to the immutables
    // number offset
    auto immutablesSize = 0; // TODO: Implement this!
    auto immutablesNumPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, heapAuxAddrSpacePtrTy,
        h.getConst(loc,
                   eravm::HeapAuxOffsetCtorRetData + eravm::ByteLen_Field));
    rewriter.create<LLVM::StoreOp>(
        loc, h.getConst(loc, immutablesSize / eravm::ByteLen_Field),
        immutablesNumPtr, eravm::getAlignment(immutablesNumPtr));

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

  LogicalResult matchAndRewrite(sol::ObjectOp objOp,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = objOp.getLoc();

    auto mod = objOp->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(objOp->getContext());
    auto i256Ty = rewriter.getIntegerType(256);
    solidity::mlirgen::BuilderHelper h(rewriter);
    eravm::BuilderHelper eravmHelper(rewriter);

    // Track the names of the existing function in the module.
    std::set<std::string> fnNamesInMod;
    for (mlir::Operation &op : *mod.getBody()) {
      if (auto fn = dyn_cast<func::FuncOp>(&op)) {
        std::string name(fn.getName());
        assert(fnNamesInMod.count(name) == 0 &&
               "Duplicate functions in module");
        fnNamesInMod.insert(name);
      }
    }

    // Move FuncOps under the ModuleOp by disambiguating functions with same
    // names.
    // TODO:
    // - Is there a better way to do the symbol disambiguation?
    // - replaceAllSymbolUses doesn't affect symbolic reference by attributes.
    //   Is that a problem here?
    objOp.walk([&](func::FuncOp fn) {
      std::string name(fn.getName());

      // Does the module have a function with the same name?
      if (fnNamesInMod.count(name)) {
        // Disambiguate the symbol.
        unsigned counter = 0;
        auto newName = name + "." + std::to_string(counter);
        while (fnNamesInMod.count(newName))
          newName = name + "." + std::to_string(++counter);

        // The visitor might be at a inner object, so we need to explicitly
        // fetch the parent ObjectOp.
        auto parentObjectOp = fn->getParentOfType<sol::ObjectOp>();
        assert(parentObjectOp);
        if (failed(fn.replaceAllSymbolUses(rewriter.getStringAttr(newName),
                                           parentObjectOp)))
          llvm_unreachable("replaceAllSymbolUses failed");
        fn.setName(newName);
        fnNamesInMod.insert(newName);

      } else {
        fnNamesInMod.insert(name);
      }

      Block *modBlk = mod.getBody();
      fn->moveBefore(modBlk, modBlk->begin());
    });

    // Is this a runtime object?
    // FIXME: Is there a better way to check this?
    if (objOp.getSymName().endswith("_deployed")) {
      // Move the runtime object region under the __runtime function
      LLVM::LLVMFuncOp runtimeFunc =
          eravmHelper.getOrInsertRuntimeFuncOp("__runtime", voidTy, {}, mod);
      Region &runtimeFuncRegion = runtimeFunc.getRegion();
      assert(runtimeFuncRegion.empty());
      rewriter.inlineRegionBefore(objOp.getRegion(), runtimeFuncRegion,
                                  runtimeFuncRegion.begin());
      rewriter.eraseOp(objOp);
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
    assert(objOp->getNumRegions() == 1);

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
        globCallDataPtr, eravm::getAlignment(globCallDataPtr));

    // Store the calldata ABI size to the global calldata size
    Value abiLen = eravmHelper.getABILen(loc, globCallDataPtr);
    LLVM::GlobalOp globCallDataSzDef =
        h.getGlobalOp(eravm::GlobCallDataSize, mod);
    Value globCallDataSz =
        rewriter.create<LLVM::AddressOfOp>(loc, globCallDataSzDef);
    rewriter.create<LLVM::StoreOp>(loc, abiLen, globCallDataSz,
                                   eravm::getAlignment(globCallDataSz));

    // Store calldatasize[calldata abi arg] to the global ret data ptr and
    // active ptr
    auto callDataSz = rewriter.create<LLVM::LoadOp>(
        loc, globCallDataSz, eravm::getAlignment(globCallDataSz));
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
                                     eravm::getAlignment(globAddr));
    };
    storeRetDataABIInitializer(eravm::GlobRetDataPtr);
    storeRetDataABIInitializer(eravm::GlobActivePtr);

    // Store call flags arg to the global call flags
    auto globCallFlagsDef = h.getGlobalOp(eravm::GlobCallFlags, mod);
    Value globCallFlags =
        rewriter.create<LLVM::AddressOfOp>(loc, globCallFlagsDef);
    rewriter.create<LLVM::StoreOp>(
        loc, entryBlk->getArgument(eravm::EntryInfo::ArgIndexCallFlags),
        globCallFlags, eravm::getAlignment(globCallFlags));

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
                                     eravm::getAlignment(gep));
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
    for (auto const &op : *objOp.getBody()) {
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
    rewriter.inlineRegionBefore(objOp.getRegion(), deployFuncRegion,
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

    rewriter.eraseOp(objOp);
    return success();
  }
};

// TODO: Move the lambda function to separate helper class.
// TODO: This lowering is copied from libyul's IRGenerator. Move it to the
// generic solidity dialect lowering.
struct ContractOpLowering : public OpRewritePattern<sol::ContractOp> {
  using OpRewritePattern<sol::ContractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::ContractOp op,
                                PatternRewriter &r) const override {
    mlir::Location loc = op.getLoc();
    solidity::mlirgen::BuilderHelper h(r);

    // Generate the creation and runtime ObjectOp.
    auto creationObj = r.replaceOpWithNewOp<sol::ObjectOp>(op, op.getSymName());
    r.setInsertionPointToStart(creationObj.getBody());
    auto runtimeObj = r.create<sol::ObjectOp>(
        loc, std::string(op.getSymName()) + "_deployed");

    // Copy contained function to creation and runtime ObjectOp.
    std::vector<func::FuncOp> funcs;
    for (Operation &i : op.getBody()->getOperations()) {
      if (auto func = dyn_cast<func::FuncOp>(i))
        funcs.push_back(func);
      else
        llvm_unreachable("NYI: Non function entities in contract");
    }
    for (func::FuncOp func : funcs) {
      // Duplicate the func in both the creation and runtime objects.
      r.clone(*func);
      func->moveBefore(runtimeObj.getBody(), runtimeObj.getBody()->begin());
    }

    //
    // Creation context
    //

    r.setInsertionPointToStart(creationObj.getBody());

    // Generate the memory init.
    auto genMemInit = [&]() {
      mlir::Value freeMemStart{};
      if (/* TODO: op.memoryUnsafeInlineAssemblySeen */ false) {
        freeMemStart = h.getConst(
            loc, solidity::frontend::CompilerUtils::generalPurposeMemoryStart +
                     /* TODO: op.getReservedMem() */ 0);
      } else {
        freeMemStart = r.create<sol::MemGuardOp>(
            loc,
            r.getIntegerAttr(
                r.getIntegerType(256),
                solidity::frontend::CompilerUtils::generalPurposeMemoryStart +
                    /* TODO: op.getReservedMem() */ 0));
      }
      r.create<sol::MStoreOp>(loc, h.getConst(loc, 64), freeMemStart);
    };
    genMemInit();

    // Generate the call value check.
    auto genCallValChk = [&]() {
      auto callVal = r.create<sol::CallValOp>(loc);
      auto callValChk = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                callVal, h.getConst(loc, 0));
      auto ifOp =
          r.create<scf::IfOp>(loc, callValChk, /*withElseRegion=*/false);
      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToStart(&ifOp.getThenRegion().front());
      r.create<sol::RevertOp>(loc, h.getConst(loc, 0), h.getConst(loc, 0));
    };

    assert(!op.getCtorAttr() && "NYI: Ctors");

    if (!op.getCtorAttr() /* TODO: || !op.ctor.isPayable() */) {
      genCallValChk();
    }

    // Generate the constructor.
    if (op.getKind() == sol::ContractKind::Library) {
      // TODO: Ctor
    }

    // Generate the codecopy of the runtime object to the free ptr.
    auto freePtr = r.create<sol::MLoadOp>(loc, h.getConst(loc, 64));
    auto runtimeObjSym = FlatSymbolRefAttr::get(runtimeObj);
    auto runtimeObjOffset = r.create<sol::DataOffsetOp>(loc, runtimeObjSym);
    auto runtimeObjSize = r.create<sol::DataSizeOp>(loc, runtimeObjSym);
    r.create<sol::CodeCopyOp>(loc, freePtr, runtimeObjOffset, runtimeObjSize);

    // TODO: Generate the setimmutable's.

    // Generate the return for the creation context.
    r.create<sol::ReturnOp>(loc, freePtr, runtimeObjOffset);

    // TODO: Ctor

    //
    // Runtime context
    //

    r.setInsertionPointToStart(runtimeObj.getBody());

    // Generate the memory init.
    // TODO: Confirm if this should be the same as in the creation context.
    genMemInit();

    if (op.getKind() == sol::ContractKind::Library) {
      // TODO: called_via_delegatecall
    }

    //
    // Dispatch generation in the runtime context
    //

    // Collect interface function infos.
    std::vector<std::pair<func::FuncOp, DictionaryAttr>> interfaceFnInfos;
    for (func::FuncOp fn : funcs) {
      DictionaryAttr interfaceFnAttr = op.getInterfaceFnAttr(fn);
      if (interfaceFnAttr)
        interfaceFnInfos.emplace_back(fn, interfaceFnAttr);
    }

    if (!interfaceFnInfos.empty()) {
      auto callDataSz = r.create<sol::CallDataSizeOp>(loc);
      // Generate `iszero(lt(calldatasize(), 4))`.
      auto callDataSzCmp = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::uge, callDataSz, h.getConst(loc, 4));
      auto ifOp =
          r.create<scf::IfOp>(loc, callDataSzCmp, /*withElseRegion=*/false);

      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto callDataLd = r.create<sol::CallDataLoadOp>(loc, h.getConst(loc, 0));

      // Collect the ABI selectors and create the switch case attribute.
      std::vector<llvm::APInt> selectors;
      selectors.reserve(interfaceFnInfos.size());
      for (auto const &interfaceFn : interfaceFnInfos) {
        DictionaryAttr attr = interfaceFn.second;
        llvm::StringRef selectorStr =
            attr.get("selector").cast<StringAttr>().getValue();
        selectors.emplace_back(256, selectorStr, 16);
      }
      auto selectorsAttr = mlir::DenseIntElementsAttr::get(
          mlir::RankedTensorType::get(static_cast<int64_t>(selectors.size()),
                                      r.getIntegerType(256)),
          selectors);

      // Generate the switch op.
      auto switchOp = r.create<mlir::scf::IntSwitchOp>(
          loc, /*resultTypes=*/llvm::None, callDataLd, selectorsAttr,
          selectors.size());

      // Generate the default block.
      {
        OpBuilder::InsertionGuard insertGuard(r);
        r.setInsertionPointToStart(r.createBlock(&switchOp.getDefaultRegion()));
        r.create<scf::YieldOp>(loc);
      }

      // Generate the case blocks.
      for (auto [caseRegion, func] :
           llvm::zip(switchOp.getCaseRegions(), funcs)) {
        if (op.getKind() == sol::ContractKind::Library) {
          // TODO: assert(!func.isPayable());
          if (/* func.stateMutability() > StateMutability::View */ false) {
            // TODO: Generate delegate call check.
          }
        }

        OpBuilder::InsertionGuard insertGuard(r);
        mlir::Block *caseBlk = r.createBlock(&caseRegion);
        r.setInsertionPointToStart(caseBlk);

        // Generate the call value check.
        if (!(op.getKind() ==
              sol::ContractKind::Library /* || func.isPayable() */)) {
          genCallValChk();
        }

        // Generate the actual call.
        assert(func.getNumArguments() == 0 &&
               "NYI: Interface function arguments");
        // TODO: libyul's ABIFunctions::tupleDecoder().
        assert(func.getNumResults() == 1 &&
               "NYI: Multi-valued and empty return");
        // TODO: libyul's ABIFunctions::tupleEncoder().
        auto out = r.create<func::CallOp>(loc, func, ValueRange{});
        assert(out.getType(0) == r.getIntegerType(256) &&
               "NYI: Non-ui256 return type");
        auto memPos = r.create<sol::MLoadOp>(loc, h.getConst(loc, 64));
        auto memEnd = r.create<arith::AddIOp>(loc, memPos, h.getConst(loc, 32));
        auto retMemPos = r.create<arith::SubIOp>(loc, memEnd, memPos);
        r.create<sol::MStoreOp>(loc, retMemPos, out.getResult(0));
        r.create<mlir::scf::YieldOp>(loc);
      }
    }

    if (op.getReceiveFnAttr()) {
      assert(false && "NYI: Receive functions");
      // TODO: Handle ether recieve function.
    }

    if (op.getFallbackFnAttr()) {
      assert(false && "NYI: Fallback functions");
      // TODO: Handle fallback function.

    } else {
      // TODO: Generate error message.
      r.create<sol::RevertOp>(loc, h.getConst(loc, 0), h.getConst(loc, 0));
    }

    // TODO: Subobjects
    return success();
  }
};

/// Pass for converting the sol dialect to other high level dialects that can be
/// lowered to the llvm dialect.
struct SolConvert : public PassWrapper<SolConvert, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SolConvert)

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<func::FuncDialect, scf::SCFDialect, arith::ArithmeticDialect,
               LLVM::LLVMDialect>();
  }

  StringRef getArgument() const override { return "sol-convert"; }

  void runOnOperation() override {
    ConversionTarget convTgt(getContext());
    convTgt.addLegalOp<mlir::ModuleOp>();
    convTgt.addLegalDialect<func::FuncDialect, scf::SCFDialect,
                            arith::ArithmeticDialect, LLVM::LLVMDialect>();
    convTgt.addIllegalDialect<sol::SolidityDialect>();

    RewritePatternSet pats(&getContext());
    sol::populateSolLoweringPatterns(pats);

    ModuleOp mod = getOperation();
    if (failed(applyPartialConversion(mod, convTgt, std::move(pats))))
      signalPassFailure();
  }
};

struct SolidityDialectLowering
    : public PassWrapper<SolidityDialectLowering, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SolidityDialectLowering)

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<LLVM::LLVMDialect, func::FuncDialect, arith::ArithmeticDialect,
               scf::SCFDialect>();
  }

  StringRef getArgument() const override { return "sol-to-llvm"; }

  void runOnOperation() override {
    LLVMConversionTarget llConv(getContext());
    llConv.addLegalOp<ModuleOp>();
    llConv.addLegalOp<scf::YieldOp>();
    LLVMTypeConverter llTyConv(&getContext());

    RewritePatternSet pats(&getContext());
    sol::populateSolLoweringPatterns(pats);
    arith::populateArithmeticToLLVMConversionPatterns(llTyConv, pats);
    populateMemRefToLLVMConversionPatterns(llTyConv, pats);
    populateSCFToControlFlowConversionPatterns(pats);
    cf::populateControlFlowToLLVMConversionPatterns(llTyConv, pats);
    populateFuncToLLVMConversionPatterns(llTyConv, pats);

    ModuleOp mod = getOperation();
    if (failed(applyFullConversion(mod, llConv, std::move(pats))))
      signalPassFailure();
  }
};

} // namespace

void sol::populateSolLoweringPatterns(RewritePatternSet &pats) {
  pats.add<ContractOpLowering, ObjectOpLowering, ReturnOpLowering,
           RevertOpLowering, MLoadOpLowering, MStoreOpLowering,
           DataOffsetOpLowering, DataSizeOpLowering, CodeCopyOpLowering,
           MemGuardOpLowering, CallValOpLowering, CallDataLoadOpLowering,
           CallDataSizeOpLowering>(pats.getContext());
}

std::unique_ptr<Pass> sol::createSolConvertPassForEraVM() {
  return std::make_unique<SolConvert>();
}

std::unique_ptr<Pass> sol::createSolidityDialectLoweringPassForEraVM() {
  return std::make_unique<SolidityDialectLowering>();
}
