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

#include "libsolidity/codegen/mlir/Target/EVM/YulToStandard.h"
#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IntrinsicsEVM.h"

using namespace mlir;

namespace {

struct Keccak256OpLowering : public OpRewritePattern<sol::Keccak256Op> {
  using OpRewritePattern<sol::Keccak256Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::Keccak256Op op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_sha3,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{evmB.genHeapPtr(op.getAddr()), op.getSize()},
        "evm.sha3");

    return success();
  }
};

struct LogOpLowering : public OpRewritePattern<sol::LogOp> {
  using OpRewritePattern<sol::LogOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::LogOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    std::vector<Value> ins{evmB.genHeapPtr(op.getAddr()), op.getSize()};
    for (Value topic : op.getTopics())
      ins.push_back(topic);

    switch (op.getTopics().size()) {
    case 0:
      r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_log0,
                                             /*resTy=*/Type{}, ins, "evm.log0");
      break;
    case 1:
      r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_log1,
                                             /*resTy=*/Type{}, ins, "evm.log1");
      break;
    case 2:
      r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_log2,
                                             /*resTy=*/Type{}, ins, "evm.log2");
      break;
    case 3:
      r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_log3,
                                             /*resTy=*/Type{}, ins, "evm.log3");
      break;
    case 4:
      r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_log4,
                                             /*resTy=*/Type{}, ins, "evm.log4");
      break;
    default:
      llvm_unreachable("Invalid log op");
    }

    return success();
  }
};

struct CallerOpLowering : public OpRewritePattern<sol::CallerOp> {
  using OpRewritePattern<sol::CallerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallerOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_caller,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{}, "evm.caller");

    return success();
  }
};

struct CallValOpLowering : public OpRewritePattern<sol::CallValOp> {
  using OpRewritePattern<sol::CallValOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallValOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_callvalue,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.callvalue");
    return success();
  }
};

struct CallDataLoadOpLowering : public OpRewritePattern<sol::CallDataLoadOp> {
  using OpRewritePattern<sol::CallDataLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallDataLoadOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    Value ptr = evmB.genCallDataPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), ptr,
                                       evm::getAlignment(ptr));
    return success();
  }
};

struct CallDataSizeOpLowering : public OpRewritePattern<sol::CallDataSizeOp> {
  using OpRewritePattern<sol::CallDataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallDataSizeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_calldatasize,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{}, "evm.calldatasize");
    return success();
  }
};

struct CallDataCopyOpLowering : public OpRewritePattern<sol::CallDataCopyOp> {
  using OpRewritePattern<sol::CallDataCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallDataCopyOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::MemcpyOp>(
        op, /*dst=*/evmB.genHeapPtr(op.getDst()),
        /*src=*/evmB.genCallDataPtr(op.getSrc()), op.getSize(),
        /*isVolatile=*/false);
    return success();
  }
};

struct SLoadOpLowering : public OpRewritePattern<sol::SLoadOp> {
  using OpRewritePattern<sol::SLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::SLoadOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value ptr = evmB.genStoragePtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, r.getIntegerType(256), ptr,
                                       evm::getAlignment(ptr));
    return success();
  }
};

struct SStoreOpLowering : public OpRewritePattern<sol::SStoreOp> {
  using OpRewritePattern<sol::SStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::SStoreOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value ptr = evmB.genStoragePtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::StoreOp>(op, op.getVal(), ptr,
                                        evm::getAlignment(ptr));
    return success();
  }
};

struct DataOffsetOpLowering : public OpRewritePattern<sol::DataOffsetOp> {
  using OpRewritePattern<sol::DataOffsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::DataOffsetOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_dataoffset,
        /*resTy=*/r.getIntegerType(256),
        /*metadata=*/r.getStrArrayAttr(op.getObj()), "evm.dataoffset");
    return success();
  }
};

struct DataSizeOpLowering : public OpRewritePattern<sol::DataSizeOp> {
  using OpRewritePattern<sol::DataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::DataSizeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_datasize,
        /*resTy=*/r.getIntegerType(256),
        /*metadata=*/r.getStrArrayAttr(op.getObj()), "evm.datasize");
    return success();
  }
};

struct CodeSizeOpLowering : public OpRewritePattern<sol::CodeSizeOp> {
  using OpRewritePattern<sol::CodeSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CodeSizeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_codesize,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.codesize");
    return success();
  }
};

struct CodeCopyOpLowering : public OpRewritePattern<sol::CodeCopyOp> {
  using OpRewritePattern<sol::CodeCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CodeCopyOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::MemcpyOp>(
        op, /*dst=*/evmB.genHeapPtr(op.getDst()),
        /*src=*/evmB.genCodePtr(op.getSrc()), op.getSize(),
        /*isVolatile=*/false);
    return success();
  }
};

struct MLoadOpLowering : public OpRewritePattern<sol::MLoadOp> {
  using OpRewritePattern<sol::MLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MLoadOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value addr = evmB.genHeapPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, r.getIntegerType(256), addr,
                                       evm::getAlignment(addr));
    return success();
  }
};

struct MStoreOpLowering : public OpRewritePattern<sol::MStoreOp> {
  using OpRewritePattern<sol::MStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MStoreOp op,
                                PatternRewriter &r) const override {
    evm::Builder eraB(r, op->getLoc());

    Value addr = eraB.genHeapPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::StoreOp>(op, op.getVal(), addr,
                                        evm::getAlignment(addr));
    return success();
  }
};

struct MemGuardOpLowering : public OpRewritePattern<sol::MemGuardOp> {
  using OpRewritePattern<sol::MemGuardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::MemGuardOp op,
                                PatternRewriter &r) const override {
    auto size = op->getAttrOfType<IntegerAttr>("size");
    r.replaceOpWithNewOp<arith::ConstantOp>(op, size);
    return success();
  }
};

struct RevertOpLowering : public OpRewritePattern<sol::RevertOp> {
  using OpRewritePattern<sol::RevertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::RevertOp op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    solidity::mlirgen::BuilderExt bExt(r, loc);

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_revert,
        /*resTy=*/Type{},
        /*ins=*/ValueRange{evmB.genHeapPtr(op.getAddr()), op.getSize()},
        "evm.revert");
    bExt.createCallToUnreachableWrapper(op->getParentOfType<ModuleOp>());
    return success();
  }
};

struct BuiltinRetOpLowering : public OpRewritePattern<sol::BuiltinRetOp> {
  using OpRewritePattern<sol::BuiltinRetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::BuiltinRetOp op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    solidity::mlirgen::BuilderExt bExt(r, loc);

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_return,
        /*resTy=*/Type{},
        /*ins=*/ValueRange{evmB.genHeapPtr(op.getAddr()), op.getSize()},
        "evm.return");
    bExt.createCallToUnreachableWrapper(op->getParentOfType<ModuleOp>());
    return success();
  }
};

struct ObjectOpLowering : public OpRewritePattern<sol::ObjectOp> {
  using OpRewritePattern<sol::ObjectOp>::OpRewritePattern;

  // "Moves" the sol.object to the module.
  void moveObjToMod(sol::ObjectOp obj, ModuleOp mod, PatternRewriter &r) const {
    Location loc = obj.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    OpBuilder::InsertionGuard insertGuard(r);

    // Generate the entry function.
    sol::FuncOp entryFn = bExt.getOrInsertFuncOp(
        "__entry", r.getFunctionType({}, {}), LLVM::Linkage::External, mod);
    Block *entryFnBlk = r.createBlock(&entryFn.getBody());

    Block *modBlk = mod.getBody();

    // The entry code is all ops in the object that are neither a function nor
    // an object.
    //
    // Move the entry code to the entry function and everything else to the
    // module.
    for (auto &op : llvm::make_early_inc_range(*obj.getBody())) {
      if (isa<sol::FuncOp>(op) || isa<sol::ObjectOp>(op))
        op.moveBefore(modBlk, modBlk->end());
      else
        op.moveBefore(entryFnBlk, entryFnBlk->end());
    }

    // Generate an unreachable op as a terminator in the entry function block.
    r.setInsertionPointToEnd(entryFnBlk);
    r.create<LLVM::UnreachableOp>(loc);
  }

  LogicalResult matchAndRewrite(sol::ObjectOp obj,
                                PatternRewriter &r) const override {
    Location loc = obj.getLoc();

    StringRef objName = obj.getSymName();

    // Is this a runtime object?
    // FIXME: Is there a better way to check this?
    if (objName.endswith("_deployed")) {
      auto runtimeMod = r.create<ModuleOp>(loc, objName);
      moveObjToMod(obj, runtimeMod, r);

    } else {
      auto creationMod = obj->getParentOfType<ModuleOp>();
      assert(creationMod);
      moveObjToMod(obj, creationMod, r);
    }

    r.eraseOp(obj);
    return success();
  }
};

} // namespace

void evm::populateYulPats(RewritePatternSet &pats) {
  pats.add<Keccak256OpLowering, LogOpLowering, CallerOpLowering,
           CallValOpLowering, CallDataLoadOpLowering, CallDataSizeOpLowering,
           CallDataCopyOpLowering, SLoadOpLowering, SStoreOpLowering,
           DataOffsetOpLowering, DataSizeOpLowering, CodeSizeOpLowering,
           CodeCopyOpLowering, MLoadOpLowering, MStoreOpLowering,
           MemGuardOpLowering, RevertOpLowering, BuiltinRetOpLowering,
           ObjectOpLowering>(pats.getContext());
}