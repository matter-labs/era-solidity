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
#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolidity/codegen/mlir/Target/EVM/SolToStandard.h"
#include "libsolidity/codegen/mlir/Target/EVM/Util.h"
#include "libsolidity/codegen/mlir/Target/EraVM/SolToStandard.h"
#include "libsolidity/codegen/mlir/Target/EraVM/Util.h"
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

    // TODO: Use updateRootInPlace instead?
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
    reg.insert<func::FuncDialect, scf::SCFDialect, arith::ArithDialect,
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
  void runStage1Conversion(ModuleOp mod, evm::SolTypeConverter &tyConv) {
    OpBuilder b(mod.getContext());
    eravm::Builder eraB(b);

    // FIXME: DialectConversion complains "pattern was already applied" if we do
    // this in the sol.func lowering (might work if we generate a llvm/func.func
    // op instead? Should we switch all such external functions to
    // llvm/func.func?)
    eraB.getOrInsertPersonality(mod);

    ConversionTarget convTgt(getContext());
    convTgt.addLegalOp<ModuleOp>();
    convTgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                            arith::ArithDialect, LLVM::LLVMDialect>();
    convTgt.addIllegalOp<sol::ConstantOp, sol::ExtOp, sol::TruncOp, sol::AddOp,
                         sol::SubOp, sol::MulOp, sol::CmpOp, sol::CAddOp,
                         sol::CSubOp, sol::AllocaOp, sol::MallocOp,
                         sol::AddrOfOp, sol::GepOp, sol::MapOp, sol::CopyOp,
                         sol::DataLocCastOp, sol::LoadOp, sol::StoreOp,
                         sol::EmitOp, sol::RequireOp, sol::ConvCastOp>();
    convTgt.addDynamicallyLegalOp<sol::FuncOp>([&](sol::FuncOp op) {
      return tyConv.isSignatureLegal(op.getFunctionType());
    });
    convTgt.addDynamicallyLegalOp<sol::CallOp, sol::ReturnOp>(
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
      APInt slot(256, 0);
      contr.walk([&](sol::StateVarOp stateVar) {
        assert(evm::getStorageByteCount(stateVar.getType()) == 32);
        stateVar->setAttr(
            "slot",
            IntegerAttr::get(IntegerType::get(&getContext(), 256), slot++));
      });
    });

    if (failed(applyPartialConversion(mod, convTgt, std::move(pats))))
      signalPassFailure();

    // Remove all state variables.
    mod.walk([](sol::StateVarOp op) { op.erase(); });
  }

  /// Converts sol.contract and all yul dialect ops.
  void runStage2Conversion(ModuleOp mod) {
    ConversionTarget convTgt(getContext());
    convTgt.addLegalOp<ModuleOp>();
    convTgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                            arith::ArithDialect, LLVM::LLVMDialect>();
    convTgt.addIllegalDialect<sol::SolDialect>();
    convTgt
        .addLegalOp<sol::FuncOp, sol::CallOp, sol::ReturnOp, sol::ConvCastOp>();

    RewritePatternSet pats(&getContext());

    switch (tgt) {
    case solidity::mlirgen::Target::EVM:
      llvm_unreachable("NYI");
    case solidity::mlirgen::Target::EraVM:
      eravm::populateStage2Pats(pats);
      break;
    default:
      llvm_unreachable("Invalid target");
    };

    if (failed(applyPartialConversion(mod, convTgt, std::move(pats))))
      signalPassFailure();
  }

  /// Converts sol.func and related ops.
  void runStage3Conversion(ModuleOp mod, evm::SolTypeConverter &tyConv) {
    ConversionTarget tgt(getContext());
    tgt.addLegalOp<ModuleOp>();
    tgt.addLegalDialect<sol::SolDialect, func::FuncDialect, scf::SCFDialect,
                        arith::ArithDialect, LLVM::LLVMDialect>();
    tgt.addIllegalDialect<sol::SolDialect>();

    RewritePatternSet pats(&getContext());
    evm::populateFuncPats(pats, tyConv);

    if (failed(applyPartialConversion(mod, tgt, std::move(pats))))
      signalPassFailure();
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
    runStage1Conversion(mod, tyConv);
    runStage2Conversion(mod);
    runStage3Conversion(mod, tyConv);
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
