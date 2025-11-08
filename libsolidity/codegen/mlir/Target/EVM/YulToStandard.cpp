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
#include "libsolidity/codegen/mlir/Sol/Sol.h"
#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolidity/codegen/mlir/Util.h"
#include "libsolidity/codegen/mlir/Yul/Yul.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IntrinsicsEVM.h"

using namespace mlir;

namespace {

struct UpdFreePtrOpLowering : public OpRewritePattern<yul::UpdFreePtrOp> {
  using OpRewritePattern<yul::UpdFreePtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::UpdFreePtrOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());
    Value freePtr = evmB.genFreePtr();
    evmB.genFreePtrUpd(freePtr, op.getSize());
    r.replaceOp(op, freePtr);
    return success();
  }
};

struct Keccak256OpLowering : public OpRewritePattern<yul::Keccak256Op> {
  using OpRewritePattern<yul::Keccak256Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::Keccak256Op op,
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

struct DivOpLowering : public OpRewritePattern<yul::DivOp> {
  using OpRewritePattern<yul::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::DivOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_div,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getDividend(), op.getDivisor()}, "evm.div");

    return success();
  }
};

struct SDivOpLowering : public OpRewritePattern<yul::SDivOp> {
  using OpRewritePattern<yul::SDivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::SDivOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_sdiv,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getDividend(), op.getDivisor()}, "evm.sdiv");

    return success();
  }
};

struct ModOpLowering : public OpRewritePattern<yul::ModOp> {
  using OpRewritePattern<yul::ModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ModOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_mod,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getValue(), op.getMod()}, "evm.mod");

    return success();
  }
};

struct SModOpLowering : public OpRewritePattern<yul::SModOp> {
  using OpRewritePattern<yul::SModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::SModOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_smod,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getValue(), op.getMod()}, "evm.smod");

    return success();
  }
};

struct ShlOpLowering : public OpRewritePattern<yul::ShlOp> {
  using OpRewritePattern<yul::ShlOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ShlOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_shl,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getShift(), op.getVal()}, "evm.shl");

    return success();
  }
};

struct ShrOpLowering : public OpRewritePattern<yul::ShrOp> {
  using OpRewritePattern<yul::ShrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ShrOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_shr,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getShift(), op.getVal()}, "evm.shr");

    return success();
  }
};

struct SarOpLowering : public OpRewritePattern<yul::SarOp> {
  using OpRewritePattern<yul::SarOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::SarOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_sar,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getShift(), op.getVal()}, "evm.sar");

    return success();
  }
};

struct ExpOpLowering : public OpRewritePattern<yul::ExpOp> {
  using OpRewritePattern<yul::ExpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ExpOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_exp,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getBase(), op.getExp()}, "evm.exp");

    return success();
  }
};

struct AddModOpLowering : public OpRewritePattern<yul::AddModOp> {
  using OpRewritePattern<yul::AddModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::AddModOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_addmod,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getX(), op.getY(), op.getMod()}, "evm.addmod");

    return success();
  }
};

struct MulModOpLowering : public OpRewritePattern<yul::MulModOp> {
  using OpRewritePattern<yul::MulModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::MulModOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_mulmod,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getX(), op.getY(), op.getMod()}, "evm.mulmod");

    return success();
  }
};

struct SignExtendOpLowering : public OpRewritePattern<yul::SignExtendOp> {
  using OpRewritePattern<yul::SignExtendOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::SignExtendOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_signextend,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getVal(), op.getOff()}, "evm.signextend");

    return success();
  }
};

struct LogOpLowering : public OpRewritePattern<yul::LogOp> {
  using OpRewritePattern<yul::LogOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::LogOp op,
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

struct AddressOpLowering : public OpRewritePattern<yul::AddressOp> {
  using OpRewritePattern<yul::AddressOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::AddressOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_address,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{}, "evm.address");
    return success();
  }
};

struct BalanceOpLowering : public OpRewritePattern<yul::BalanceOp> {
  using OpRewritePattern<yul::BalanceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::BalanceOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_balance,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{op.getAddr()},
                                           "evm.balance");
    return success();
  }
};

struct SelfBalanceOpLowering : public OpRewritePattern<yul::SelfBalanceOp> {
  using OpRewritePattern<yul::SelfBalanceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::SelfBalanceOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_selfbalance,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.selfbalance");
    return success();
  }
};

struct CallerOpLowering : public OpRewritePattern<yul::CallerOp> {
  using OpRewritePattern<yul::CallerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CallerOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_caller,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{}, "evm.caller");

    return success();
  }
};

struct GasOpLowering : public OpRewritePattern<yul::GasOp> {
  using OpRewritePattern<yul::GasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::GasOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_gas,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{}, "evm.gas");
    return success();
  }
};

struct ChainIdOpLowering : public OpRewritePattern<yul::ChainIdOp> {
  using OpRewritePattern<yul::ChainIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ChainIdOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_chainid,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{}, "evm.chainid");
    return success();
  }
};

struct BaseFeeOpLowering : public OpRewritePattern<yul::BaseFeeOp> {
  using OpRewritePattern<yul::BaseFeeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::BaseFeeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_basefee,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{}, "evm.basefee");
    return success();
  }
};

struct BlobBaseFeeOpLowering : public OpRewritePattern<yul::BlobBaseFeeOp> {
  using OpRewritePattern<yul::BlobBaseFeeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::BlobBaseFeeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_blobbasefee,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.blobbasefee");
    return success();
  }
};

struct OriginOpLowering : public OpRewritePattern<yul::OriginOp> {
  using OpRewritePattern<yul::OriginOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::OriginOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_origin,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{}, "evm.origin");
    return success();
  }
};

struct GasPriceOpLowering : public OpRewritePattern<yul::GasPriceOp> {
  using OpRewritePattern<yul::GasPriceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::GasPriceOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_gasprice,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.gasprice");
    return success();
  }
};

struct BlockHashOpLowering : public OpRewritePattern<yul::BlockHashOp> {
  using OpRewritePattern<yul::BlockHashOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::BlockHashOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_blockhash,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{op.getBlock()},
                                           "evm.blockhash");
    return success();
  }
};

struct BlobHashOpLowering : public OpRewritePattern<yul::BlobHashOp> {
  using OpRewritePattern<yul::BlobHashOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::BlobHashOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_blobhash,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{op.getIdx()},
                                           "evm.blobhash");
    return success();
  }
};

struct CoinBaseOpLowering : public OpRewritePattern<yul::CoinBaseOp> {
  using OpRewritePattern<yul::CoinBaseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CoinBaseOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_coinbase,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.coinbase");
    return success();
  }
};

struct TimeStampOpLowering : public OpRewritePattern<yul::TimeStampOp> {
  using OpRewritePattern<yul::TimeStampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::TimeStampOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_timestamp,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.timestamp");
    return success();
  }
};

struct NumberOpLowering : public OpRewritePattern<yul::NumberOp> {
  using OpRewritePattern<yul::NumberOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::NumberOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_number,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{}, "evm.number");
    return success();
  }
};

struct PrevrandaoOpLowering : public OpRewritePattern<yul::PrevrandaoOp> {
  using OpRewritePattern<yul::PrevrandaoOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::PrevrandaoOp op,
                                PatternRewriter &r) const override {
    // TODO: fix the intrinsic name in LLVM.
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_difficulty,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.difficulty");
    return success();
  }
};

struct GasLimitOpLowering : public OpRewritePattern<yul::GasLimitOp> {
  using OpRewritePattern<yul::GasLimitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::GasLimitOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_gaslimit,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.gaslimit");
    return success();
  }
};

struct CallValOpLowering : public OpRewritePattern<yul::CallValOp> {
  using OpRewritePattern<yul::CallValOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CallValOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_callvalue,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.callvalue");
    return success();
  }
};

struct CallDataLoadOpLowering : public OpRewritePattern<yul::CallDataLoadOp> {
  using OpRewritePattern<yul::CallDataLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CallDataLoadOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    Value ptr = evmB.genCallDataPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), ptr,
                                       evm::getAlignment(ptr));
    return success();
  }
};

struct CallDataSizeOpLowering : public OpRewritePattern<yul::CallDataSizeOp> {
  using OpRewritePattern<yul::CallDataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CallDataSizeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_calldatasize,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{}, "evm.calldatasize");
    return success();
  }
};

struct CallDataCopyOpLowering : public OpRewritePattern<yul::CallDataCopyOp> {
  using OpRewritePattern<yul::CallDataCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CallDataCopyOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::MemcpyOp>(
        op, /*dst=*/evmB.genHeapPtr(op.getDst()),
        /*src=*/evmB.genCallDataPtr(op.getSrc()), op.getSize(),
        /*isVolatile=*/false);
    return success();
  }
};

struct ReturnDataSizeOpLowering
    : public OpRewritePattern<yul::ReturnDataSizeOp> {
  using OpRewritePattern<yul::ReturnDataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ReturnDataSizeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_returndatasize,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{}, "evm.returndatasize");
    return success();
  }
};

struct ReturnDataCopyOpLowering
    : public OpRewritePattern<yul::ReturnDataCopyOp> {
  using OpRewritePattern<yul::ReturnDataCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ReturnDataCopyOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::MemcpyOp>(
        op, /*dst=*/evmB.genHeapPtr(op.getDst()),
        /*src=*/evmB.genReturnDataPtr(op.getSrc()), op.getSize(),
        /*isVolatile=*/false);
    return success();
  }
};

struct SLoadOpLowering : public OpRewritePattern<yul::SLoadOp> {
  using OpRewritePattern<yul::SLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::SLoadOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value ptr = evmB.genStoragePtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, r.getIntegerType(256), ptr,
                                       evm::getAlignment(ptr));
    return success();
  }
};

struct SStoreOpLowering : public OpRewritePattern<yul::SStoreOp> {
  using OpRewritePattern<yul::SStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::SStoreOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value ptr = evmB.genStoragePtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::StoreOp>(op, op.getVal(), ptr,
                                        evm::getAlignment(ptr));
    return success();
  }
};

struct TLoadOpLowering : public OpRewritePattern<yul::TLoadOp> {
  using OpRewritePattern<yul::TLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::TLoadOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value ptr = evmB.genTStoragePtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, r.getIntegerType(256), ptr,
                                       evm::getAlignment(ptr));
    return success();
  }
};

struct TStoreOpLowering : public OpRewritePattern<yul::TStoreOp> {
  using OpRewritePattern<yul::TStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::TStoreOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value ptr = evmB.genTStoragePtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::StoreOp>(op, op.getVal(), ptr,
                                        evm::getAlignment(ptr));
    return success();
  }
};

struct DataOffsetOpLowering : public OpRewritePattern<yul::DataOffsetOp> {
  using OpRewritePattern<yul::DataOffsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::DataOffsetOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_dataoffset,
        /*resTy=*/r.getIntegerType(256),
        /*metadata=*/r.getStrArrayAttr(op.getObj()), "evm.dataoffset");
    return success();
  }
};

struct DataSizeOpLowering : public OpRewritePattern<yul::DataSizeOp> {
  using OpRewritePattern<yul::DataSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::DataSizeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_datasize,
        /*resTy=*/r.getIntegerType(256),
        /*metadata=*/r.getStrArrayAttr(op.getObj()), "evm.datasize");
    return success();
  }
};

struct CodeSizeOpLowering : public OpRewritePattern<yul::CodeSizeOp> {
  using OpRewritePattern<yul::CodeSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CodeSizeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_codesize,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/ValueRange{},
                                           "evm.codesize");
    return success();
  }
};

struct CodeCopyOpLowering : public OpRewritePattern<yul::CodeCopyOp> {
  using OpRewritePattern<yul::CodeCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CodeCopyOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::MemcpyOp>(
        op, /*dst=*/evmB.genHeapPtr(op.getDst()),
        /*src=*/evmB.genCodePtr(op.getSrc()), op.getSize(),
        /*isVolatile=*/false);
    return success();
  }
};

struct ExtCodeSizeOpLowering : public OpRewritePattern<yul::ExtCodeSizeOp> {
  using OpRewritePattern<yul::ExtCodeSizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ExtCodeSizeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_extcodesize,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/op.getAddr(),
                                           "evm.extcodesize");
    return success();
  }
};

struct ExtCodeCopyOpLowering : public OpRewritePattern<yul::ExtCodeCopyOp> {
  using OpRewritePattern<yul::ExtCodeCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ExtCodeCopyOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_extcodecopy, /*resTy=*/Type{}, /*ins=*/
        ValueRange{op.getAddr(),
                   /*dst=*/evmB.genHeapPtr(op.getDst()),
                   /*src=*/evmB.genCodePtr(op.getSrc()), op.getSize()},
        "evm.extcodecopy");
    return success();
  }
};

struct ExtCodeHashOpLowering : public OpRewritePattern<yul::ExtCodeHashOp> {
  using OpRewritePattern<yul::ExtCodeHashOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ExtCodeHashOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_extcodehash,
                                           /*resTy=*/r.getIntegerType(256),
                                           /*ins=*/op.getAddr(),
                                           "evm.extcodehash");
    return success();
  }
};

struct CreateOpLowering : public OpRewritePattern<yul::CreateOp> {
  using OpRewritePattern<yul::CreateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CreateOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());
    Value addr = evmB.genHeapPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_create,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getVal(), addr, op.getSize()}, "evm.create");
    return success();
  }
};

struct Create2OpLowering : public OpRewritePattern<yul::Create2Op> {
  using OpRewritePattern<yul::Create2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::Create2Op op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());
    Value addr = evmB.genHeapPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_create2,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getVal(), addr, op.getSize(), op.getSalt()},
        "evm.create2");
    return success();
  }
};

struct MLoadOpLowering : public OpRewritePattern<yul::MLoadOp> {
  using OpRewritePattern<yul::MLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::MLoadOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value addr = evmB.genHeapPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, r.getIntegerType(256), addr,
                                       evm::getAlignment(addr));
    return success();
  }
};

struct LoadImmutable2OpLowering
    : public OpRewritePattern<yul::LoadImmutableOp> {
  using OpRewritePattern<yul::LoadImmutableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::LoadImmutableOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_loadimmutable,
        /*resTy=*/r.getIntegerType(256),
        /*name=*/r.getStrArrayAttr(op.getName()), "evm.loadimmutable");
    return success();
  }
};

struct LoadImmutableOpLowering
    : public OpConversionPattern<sol::LoadImmutableOp> {
  using OpConversionPattern<sol::LoadImmutableOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LoadImmutableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);

    bool inRuntime = op->getParentOfType<sol::FuncOp>().getRuntime();
    Value repl;
    if (inRuntime) {
      repl = r.create<yul::LoadImmutableOp>(loc, op.getName());
    } else {
      assert(op->hasAttr("addr"));
      IntegerAttr addr = cast<IntegerAttr>(op->getAttr("addr"));
      repl = r.create<yul::MLoadOp>(loc, bExt.genI256Const(addr.getValue()));
    }

    if (auto intTy = dyn_cast<IntegerType>(op.getType())) {
      Value castedRepl =
          bExt.genIntCast(intTy.getWidth(), intTy.isSigned(), repl);
      r.replaceOp(op, castedRepl);
      return success();
    }

    r.replaceOp(op, repl);
    return success();
  }
};

struct LinkerSymbolOpLowering : public OpRewritePattern<yul::LinkerSymbolOp> {
  using OpRewritePattern<yul::LinkerSymbolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::LinkerSymbolOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_linkersymbol,
        /*resTy=*/r.getIntegerType(256),
        /*name=*/r.getStrArrayAttr(op.getName()), "evm.linkersymbol");
    return success();
  }
};

struct MStoreOpLowering : public OpRewritePattern<yul::MStoreOp> {
  using OpRewritePattern<yul::MStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::MStoreOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    Value addr = evmB.genHeapPtr(op.getAddr());
    r.replaceOpWithNewOp<LLVM::StoreOp>(op, op.getVal(), addr,
                                        evm::getAlignment(addr));
    return success();
  }
};

struct MStore8OpLowering : public OpRewritePattern<yul::MStore8Op> {
  using OpRewritePattern<yul::MStore8Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::MStore8Op op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_mstore8,
        /*resTy=*/Type{},
        /*ins=*/ValueRange{evmB.genHeapPtr(op.getAddr()), op.getVal()},
        "evm.mstore8");
    return success();
  }
};

struct SetImmutableOpLowering : public OpRewritePattern<yul::SetImmutableOp> {
  using OpRewritePattern<yul::SetImmutableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::SetImmutableOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<LLVM::SetImmutableOp>(op, op.getAddr(), op.getName(),
                                               op.getVal());
    return success();
  }
};

struct ByteOpLowering : public OpRewritePattern<yul::ByteOp> {
  using OpRewritePattern<yul::ByteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ByteOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op->getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_byte,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/ValueRange{op.getIdx(), op.getVal()}, "evm.byte");
    return success();
  }
};

struct MCopyOpLowering : public OpRewritePattern<yul::MCopyOp> {
  using OpRewritePattern<yul::MCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::MCopyOp op,
                                PatternRewriter &r) const override {
    // TODO? Check m_evmVersion.hasMcopy() and legalize here?

    evm::Builder evmB(r, op->getLoc());

    // Generate the memmove.
    // FIXME: Add align 1 param attribute.
    r.replaceOpWithNewOp<LLVM::MemmoveOp>(op, evmB.genHeapPtr(op.getDst()),
                                          evmB.genHeapPtr(op.getSrc()),
                                          op.getSize(),
                                          /*isVolatile=*/false);
    return success();
  }
};

struct MemGuardOpLowering : public OpRewritePattern<yul::MemGuardOp> {
  using OpRewritePattern<yul::MemGuardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::MemGuardOp op,
                                PatternRewriter &r) const override {
    auto size = op->getAttrOfType<IntegerAttr>("size");
    r.replaceOpWithNewOp<arith::ConstantOp>(op, size);
    return success();
  }
};

struct RevertOpLowering : public OpRewritePattern<yul::RevertOp> {
  using OpRewritePattern<yul::RevertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::RevertOp op,
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

struct BuiltinCallOpLowering : public OpRewritePattern<yul::CallOp> {
  using OpRewritePattern<yul::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CallOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_call,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/
        ValueRange{op.getGas(), op.getAddress(), op.getValue(),
                   evmB.genHeapPtr(op.getInpOffset()), op.getInpSize(),
                   evmB.genHeapPtr(op.getOutOffset()), op.getOutSize()},
        "evm.call");
    return success();
  }
};

struct CallCodeOpLowering : public OpRewritePattern<yul::CallCodeOp> {
  using OpRewritePattern<yul::CallCodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::CallCodeOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_callcode,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/
        ValueRange{op.getGas(), op.getAddress(), op.getValue(),
                   evmB.genHeapPtr(op.getInpOffset()), op.getInpSize(),
                   evmB.genHeapPtr(op.getOutOffset()), op.getOutSize()},
        "evm.callcode");
    return success();
  }
};

struct StaticCallOpLowering : public OpRewritePattern<yul::StaticCallOp> {
  using OpRewritePattern<yul::StaticCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::StaticCallOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_staticcall,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/
        ValueRange{op.getGas(), op.getAddress(),
                   evmB.genHeapPtr(op.getInpOffset()), op.getInpSize(),
                   evmB.genHeapPtr(op.getOutOffset()), op.getOutSize()},
        "evm.staticcall");
    return success();
  }
};

struct DelegateCallOpLowering : public OpRewritePattern<yul::DelegateCallOp> {
  using OpRewritePattern<yul::DelegateCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::DelegateCallOp op,
                                PatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(
        op, llvm::Intrinsic::evm_delegatecall,
        /*resTy=*/r.getIntegerType(256),
        /*ins=*/
        ValueRange{op.getGas(), op.getAddress(),
                   evmB.genHeapPtr(op.getInpOffset()), op.getInpSize(),
                   evmB.genHeapPtr(op.getOutOffset()), op.getOutSize()},
        "evm.delegatecall");
    return success();
  }
};

struct BuiltinRetOpLowering : public OpRewritePattern<yul::ReturnOp> {
  using OpRewritePattern<yul::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::ReturnOp op,
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

struct StopOpLowering : public OpRewritePattern<yul::StopOp> {
  using OpRewritePattern<yul::StopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(yul::StopOp op,
                                PatternRewriter &r) const override {
    solidity::mlirgen::BuilderExt bExt(r, op.getLoc());

    r.replaceOpWithNewOp<LLVM::IntrCallOp>(op, llvm::Intrinsic::evm_stop,
                                           /*resTy=*/Type{},
                                           /*ins=*/ValueRange{}, "evm.stop");
    bExt.createCallToUnreachableWrapper(op->getParentOfType<ModuleOp>());
    return success();
  }
};

struct ObjectOpLowering : public OpRewritePattern<yul::ObjectOp> {
  using OpRewritePattern<yul::ObjectOp>::OpRewritePattern;

  // "Moves" the sol.object to the module.
  void moveObjToMod(yul::ObjectOp obj, ModuleOp mod, PatternRewriter &r) const {
    Location loc = obj.getLoc();
    solidity::mlirgen::BuilderExt bExt(r, loc);
    OpBuilder::InsertionGuard insertGuard(r);

    // Generate the entry function.
    sol::FuncOp entryFn = bExt.getOrInsertFuncOp(
        "__entry", r.getFunctionType({}, {}), LLVM::Linkage::External, mod);
    Block *modBlk = mod.getBody();

    // The entry code is all ops in the object that are neither a function nor
    // an object.
    //
    // Move the entry code to the entry function and everything else to the
    // module.
    for (auto &op : llvm::make_early_inc_range(*obj.getEntryBlock())) {
      if (isa<sol::FuncOp>(op) || isa<yul::ObjectOp>(op))
        op.moveBefore(modBlk, modBlk->end());
    }
    // Terminate all blocks without terminators using the unreachable op.
    obj.walk([&](Block *blk) {
      if (blk->empty() || !blk->back().hasTrait<OpTrait::IsTerminator>()) {
        r.setInsertionPointToEnd(blk);
        r.create<LLVM::UnreachableOp>(loc);
      };
    });
    r.inlineRegionBefore(obj.getBody(), entryFn.getBody(),
                         entryFn.getBody().begin());
  }

  LogicalResult matchAndRewrite(yul::ObjectOp obj,
                                PatternRewriter &r) const override {
    Location loc = obj.getLoc();

    StringRef objName = obj.getName();

    // Is this a runtime object?
    // FIXME: Is there a better way to check this?
    if (objName.ends_with("_deployed")) {
      auto runtimeMod = r.create<ModuleOp>(loc, objName);
      moveObjToMod(obj, runtimeMod, r);

    } else {
      auto creationMod = obj->getParentOfType<ModuleOp>();
      assert(creationMod);
      creationMod.setName(objName);
      moveObjToMod(obj, creationMod, r);
    }

    r.eraseOp(obj);
    return success();
  }
};

} // namespace

void evm::populateYulPats(RewritePatternSet &pats) {
  pats.add<
      // clang-format off
      AddModOpLowering,
      MulModOpLowering,
      UpdFreePtrOpLowering,
      Keccak256OpLowering,
      DivOpLowering,
      SDivOpLowering,
      ModOpLowering,
      SModOpLowering,
      ShlOpLowering,
      ShrOpLowering,
      SarOpLowering,
      ExpOpLowering,
      SignExtendOpLowering,
      LogOpLowering,
      AddressOpLowering,
      BalanceOpLowering,
      SelfBalanceOpLowering,
      CallerOpLowering,
      GasOpLowering,
      ChainIdOpLowering,
      BaseFeeOpLowering,
      BlobBaseFeeOpLowering,
      OriginOpLowering,
      GasPriceOpLowering,
      BlockHashOpLowering,
      BlobHashOpLowering,
      CoinBaseOpLowering,
      TimeStampOpLowering,
      NumberOpLowering,
      PrevrandaoOpLowering,
      GasLimitOpLowering,
      RevertOpLowering,
      StopOpLowering,
      CallValOpLowering,
      CallDataLoadOpLowering,
      CallDataSizeOpLowering,
      CallDataCopyOpLowering,
      ReturnDataSizeOpLowering,
      ReturnDataCopyOpLowering,
      SLoadOpLowering,
      SStoreOpLowering,
      TLoadOpLowering,
      TStoreOpLowering,
      DataOffsetOpLowering,
      DataSizeOpLowering,
      CodeSizeOpLowering,
      CodeCopyOpLowering,
      ExtCodeSizeOpLowering,
      ExtCodeCopyOpLowering,
      ExtCodeHashOpLowering,
      CreateOpLowering,
      Create2OpLowering,
      MLoadOpLowering,
      LoadImmutableOpLowering,
      LoadImmutable2OpLowering,
      LinkerSymbolOpLowering,
      MStoreOpLowering,
      MStore8OpLowering,
      SetImmutableOpLowering,
      ByteOpLowering,
      MCopyOpLowering,
      MemGuardOpLowering,
      BuiltinCallOpLowering,
      CallCodeOpLowering,
      StaticCallOpLowering,
      DelegateCallOpLowering,
      BuiltinRetOpLowering,
      ObjectOpLowering
      // clang-format on
      >(pats.getContext());
}
