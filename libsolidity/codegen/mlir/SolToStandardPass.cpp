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
// The sol dialect lowering pass
//

#include "libsolidity/codegen/mlir/Passes.h"
#include "libsolidity/codegen/mlir/Sol/Sol.h"
#include "libsolidity/codegen/mlir/Target/EVM/SolToYul.h"
#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolidity/codegen/mlir/Target/EraVM/Util.h"
#include "libsolidity/codegen/mlir/Target/EraVM/YulToStandard.h"
#include "libsolidity/codegen/mlir/Yul/Yul.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

/// A generic conversion pattern that replaces the operands with the legalized
/// ones and legalizes the return types.
template <typename OpT>
struct GenericTypeConversion : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    SmallVector<Type> retTys;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTys)))
      return failure();

    r.replaceOpWithNewOp<OpT>(op, retTys, adaptor.getOperands(),
                              op->getAttrs());
    return success();
  }
};

struct ConvCastOpLowering : public OpConversionPattern<sol::ConvCastOp> {
  using OpConversionPattern<sol::ConvCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ConvCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOp(op, adaptor.getInp());
    return success();
  }
};

/// Pass for lowering the sol dialect to the standard dialects.
/// TODO:
/// - Generate this using mlir-tblgen.
struct ConvertSolToStandard
    : public PassWrapper<ConvertSolToStandard, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSolToStandard)

  ConvertSolToStandard() = default;

  ConvertSolToStandard(solidity::mlirgen::Target tgt) : tgt(tgt) {}

  ConvertSolToStandard(ConvertSolToStandard const &other)
      : PassWrapper(other) {}

  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<yul::YulDialect, func::FuncDialect, scf::SCFDialect,
               cf::ControlFlowDialect, arith::ArithDialect,
               LLVM::LLVMDialect>();
  }

  // TODO: Generalize this comment.
  //
  // FIXME: Some of the conversion patterns depends on the ancestor/descendant
  // sol.func ops (e.g.: checking the runtime/creation context). Also APIs
  // like getOrInsert*FuncOp that are used in the conversion pass works with
  // sol.func. sol.func conversion in the same conversion pass will require
  // other conversion to be able to work with sol.func and func.func. To keep
  // things simple for now, the sol.func and related ops lowering is scheduled
  // in a separate conversion pass after the main one.
  //
  // How can we do the conversion cleanly with one pass? Generally speaking,
  // how should we work with conversion patterns that depends on other
  // operations?

  // FIXME: Separate yul specific sol ops to a "yul" dialect? This could
  // simplify the implementation of this multi-staged lowering.

  /// Converts sol dialect ops except sol.contract and sol.func + related ops.
  /// This pass also legalizes all the sol dialect types.
  LogicalResult runStage1Conversion(ModuleOp mod,
                                    evm::SolTypeConverter &tyConv) {
    OpBuilder b(mod.getContext());

    // FIXME: DialectConversion complains "pattern was already applied" if we do
    // this in the sol.func lowering (might work if we generate a llvm/func.func
    // op instead? Should we switch all such external functions to
    // llvm/func.func? (dealing with sol.func and func.func has been a major
    // pain point!)).
    if (tgt == solidity::mlirgen::Target::EraVM) {
      eravm::Builder eraB(b);
      eraB.getOrInsertPersonality(mod);
    }

    ConversionTarget convTgt(getContext());
    convTgt.addLegalOp<ModuleOp>();
    convTgt.addLegalDialect<sol::SolDialect, yul::YulDialect, func::FuncDialect,
                            scf::SCFDialect, cf::ControlFlowDialect,
                            arith::ArithDialect, LLVM::LLVMDialect>();
    convTgt.addIllegalOp<
        // clang-format off
        sol::ConstantOp,
        sol::CastOp,
        sol::BytesCastOp,
        sol::AddOp,
        sol::SubOp,
        sol::MulOp,
        sol::DivOp,
        sol::ModOp,
        sol::CmpOp,
        sol::CAddOp,
        sol::CSubOp,
        sol::CMulOp,
        sol::CDivOp,
        sol::ExpOp,
        sol::AllocaOp,
        sol::MallocOp,
        sol::ArrayLitOp,
        sol::GetCallDataOp,
        sol::PushOp,
        sol::PopOp,
        sol::AddrOfOp,
        sol::GepOp,
        sol::MapOp,
        sol::CopyOp,
        sol::DataLocCastOp,
        sol::LoadOp,
        sol::StoreOp,
        sol::LengthOp,
        sol::ThisOp,
        sol::LibAddrOp,
        sol::EncodeOp,
        sol::DecodeOp,
        sol::ExtCallOp,
        sol::NewOp,
        sol::CodeOp,
        sol::EmitOp,
        sol::RequireOp,
        sol::ConvCastOp,
        sol::IfOp,
        sol::SwitchOp,
        sol::WhileOp,
        sol::DoWhileOp,
        sol::ForOp,
        sol::TryOp
        // clang-format on
        >();
    convTgt.addDynamicallyLegalOp<sol::FuncOp>([&](sol::FuncOp op) {
      return tyConv.isSignatureLegal(op.getFunctionType());
    });
    convTgt.addDynamicallyLegalOp<sol::CallOp, sol::ReturnOp,
                                  sol::LoadImmutableOp>(
        [&](Operation *op) { return tyConv.isLegal(op); });

    RewritePatternSet pats(&getContext());
    pats.add<ConvCastOpLowering>(tyConv, &getContext());
    populateAnyFunctionOpInterfaceTypeConversionPattern(pats, tyConv);
    pats.add<GenericTypeConversion<sol::CallOp>,
             GenericTypeConversion<sol::ReturnOp>>(tyConv, &getContext());

    switch (tgt) {
    case solidity::mlirgen::Target::EVM:
      evm::populateStage1Pats(pats, tyConv);
      break;
    case solidity::mlirgen::Target::EraVM:
      eravm::populateStage1Pats(pats, tyConv);
      break;
    default:
      llvm_unreachable("Invalid target");
    };

    // Assign slots to state variables.
    mod.walk([&](sol::ContractOp contr) {
      // Slots start from 0 and immutables from address 128
      APInt slot(256, 0), immAddr(256, 128);
      contr.walk([&](sol::StateVarOp stateVar) {
        stateVar->setAttr(
            "slot",
            IntegerAttr::get(IntegerType::get(&getContext(), 256), slot));
        slot += evm::getStorageSlotCount(stateVar.getType());
      });
      contr.walk([&](sol::ImmutableOp immOp) {
        immOp->setAttr(
            "addr",
            IntegerAttr::get(IntegerType::get(&getContext(), 256), immAddr));
        immAddr += 32;
      });
    });

    if (failed(applyPartialConversion(mod, convTgt, std::move(pats))))
      return failure();
    return success();
  }

  /// Converts sol.contract and all yul dialect ops.
  LogicalResult runStage2Conversion(ModuleOp mod) {
    ConversionTarget convTgt(getContext());
    convTgt.addLegalOp<ModuleOp>();
    convTgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                            cf::ControlFlowDialect, arith::ArithDialect,
                            LLVM::LLVMDialect>();
    convTgt.addIllegalDialect<sol::SolDialect, yul::YulDialect>();
    convTgt
        .addLegalOp<sol::FuncOp, sol::CallOp, sol::ReturnOp, sol::ConvCastOp>();

    RewritePatternSet pats(&getContext());

    switch (tgt) {
    case solidity::mlirgen::Target::EVM:
      evm::populateStage2Pats(pats);
      break;
    case solidity::mlirgen::Target::EraVM:
      eravm::populateStage2Pats(pats);
      break;
    default:
      llvm_unreachable("Invalid target");
    };

    if (failed(applyPartialConversion(mod, convTgt, std::move(pats))))
      return failure();
    return success();
  }

  /// Converts sol.func and related ops.
  LogicalResult runStage3Conversion(ModuleOp mod,
                                    evm::SolTypeConverter &tyConv) {
    ConversionTarget convTgt(getContext());
    convTgt.addLegalOp<ModuleOp>();
    convTgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                            cf::ControlFlowDialect, arith::ArithDialect,
                            LLVM::LLVMDialect>();
    convTgt.addIllegalDialect<sol::SolDialect>();

    RewritePatternSet pats(&getContext());
    switch (tgt) {
    case solidity::mlirgen::Target::EVM:
      evm::populateFuncPats(pats, tyConv);
      break;
    case solidity::mlirgen::Target::EraVM:
      eravm::populateFuncPats(pats, tyConv);
      break;
    default:
      llvm_unreachable("Invalid target");
    };

    if (failed(applyPartialConversion(mod, convTgt, std::move(pats))))
      return failure();
    return success();
  }

  void runOnOperation() override {
    // We can't check this in the ctor since cl::ParseCommandLineOptions won't
    // be called then.
    if (clTgt.getNumOccurrences() > 0) {
      assert(tgt == solidity::mlirgen::Target::Undefined);
      tgt = clTgt;
    }

    ModuleOp mod = getOperation();
    evm::SolTypeConverter tyConv;
    if (failed(runStage1Conversion(mod, tyConv))) {
      signalPassFailure();
      return;
    }
    if (failed(runStage2Conversion(mod))) {
      signalPassFailure();
      return;
    }
    if (failed(runStage3Conversion(mod, tyConv))) {
      signalPassFailure();
      return;
    }
  }

  StringRef getArgument() const override { return "convert-sol-to-std"; }

protected:
  solidity::mlirgen::Target tgt = solidity::mlirgen::Target::Undefined;
  Pass::Option<solidity::mlirgen::Target> clTgt{
      *this, "target", llvm::cl::desc("Target for the sol lowering"),
      llvm::cl::init(solidity::mlirgen::Target::Undefined),
      llvm::cl::values(
          clEnumValN(solidity::mlirgen::Target::EVM, "evm", "EVM target"),
          clEnumValN(solidity::mlirgen::Target::EraVM, "eravm",
                     "EraVM target"))};
};

} // namespace

std::unique_ptr<Pass> sol::createConvertSolToStandardPass() {
  return std::make_unique<ConvertSolToStandard>();
}

std::unique_ptr<Pass>
sol::createConvertSolToStandardPass(solidity::mlirgen::Target tgt) {
  return std::make_unique<ConvertSolToStandard>(tgt);
}
