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
#include "llvm/IR/IntrinsicsEVM.h"

using namespace mlir;

namespace {

struct Keccak256OpLowering : public OpRewritePattern<sol::Keccak256Op> {
  using OpRewritePattern<sol::Keccak256Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::Keccak256Op op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

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
      r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_log3,
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
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value ptr = evmB.genCallDataPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), ptr,
                                       evm::getAlignment(ptr));
    return success();
  }
};

} // namespace

void evm::populateYulPats(RewritePatternSet &pats) {
  pats.add<Keccak256OpLowering, LogOpLowering, CallerOpLowering,
           CallValOpLowering, CallDataLoadOpLowering>(pats.getContext());
}
