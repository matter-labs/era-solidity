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

} // namespace

void evm::populateYulPats(RewritePatternSet &pats) {
  pats.add<Keccak256OpLowering>(pats.getContext());
}
