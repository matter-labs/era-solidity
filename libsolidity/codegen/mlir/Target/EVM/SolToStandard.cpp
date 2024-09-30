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

#include "libsolidity/codegen/mlir/Target/EVM/SolToStandard.h"
#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolidity/codegen/mlir/Target/EraVM/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

evm::SolTypeConverter::SolTypeConverter() {
  // Default case
  addConversion([](Type ty) { return ty; });

  // Integer type
  addConversion([&](IntegerType ty) -> Type {
    // Map to signless variant.

    if (ty.isSignless())
      return ty;
    return IntegerType::get(ty.getContext(), ty.getWidth(),
                            IntegerType::Signless);
  });

  // Function type
  addConversion([&](FunctionType ty) -> Type {
    SmallVector<Type> convertedInpTys, convertedResTys;
    if (failed(convertTypes(ty.getInputs(), convertedInpTys)))
      llvm_unreachable("Invalid type");
    if (failed(convertTypes(ty.getResults(), convertedResTys)))
      llvm_unreachable("Invalid type");

    return FunctionType::get(ty.getContext(), convertedInpTys, convertedResTys);
  });

  // Array type
  addConversion([&](sol::ArrayType ty) -> Type {
    switch (ty.getDataLocation()) {
    case sol::DataLocation::Stack: {
      Type eltTy = convertType(ty.getEltType());
      return LLVM::LLVMArrayType::get(eltTy, ty.getSize());
    }

    // Map to the 256 bit address in memory.
    case sol::DataLocation::Memory:
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);

    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  // String type
  addConversion([&](sol::StringType ty) -> Type {
    switch (ty.getDataLocation()) {
    // Map to the 256 bit address in memory.
    case sol::DataLocation::Memory:
    // Map to the 256 bit slot offset.
    case sol::DataLocation::Storage:
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);

    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  // Mapping type
  addConversion([&](sol::MappingType ty) -> Type {
    // Map to the 256 bit slot offset.
    return IntegerType::get(ty.getContext(), 256,
                            IntegerType::SignednessSemantics::Signless);
  });

  // Struct type
  addConversion([&](sol::StructType ty) -> Type {
    switch (ty.getDataLocation()) {
    case sol::DataLocation::Memory:
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);
    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  // Pointer type
  addConversion([&](sol::PointerType ty) -> Type {
    switch (ty.getDataLocation()) {
    case sol::DataLocation::Stack: {
      Type eltTy = convertType(ty.getPointeeType());
      return LLVM::LLVMPointerType::get(eltTy);
    }

    // Map to the 256 bit address in memory.
    case sol::DataLocation::Memory:
    // Map to the 256 bit slot offset.
    //
    // TODO: Can we get all storage types to be 32 byte aligned? If so, we can
    // avoid the byte offset. Otherwise we should consider the
    // OneToNTypeConversion to map the pointer to the slot + byte offset pair.
    case sol::DataLocation::Storage:
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);

    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  addSourceMaterialization([](OpBuilder &b, Type resTy, ValueRange ins,
                              Location loc) -> Value {
    if (ins.size() != 1)
      return b.create<UnrealizedConversionCastOp>(loc, resTy, ins).getResult(0);

    Type i256Ty = b.getIntegerType(256);

    Type inpTy = ins[0].getType();

    if ((sol::isRefType(inpTy) && resTy == i256Ty) ||
        (inpTy == i256Ty && sol::isRefType(resTy)))
      return b.create<sol::ConvCastOp>(loc, resTy, ins);

    return b.create<UnrealizedConversionCastOp>(loc, resTy, ins).getResult(0);
  });
}

namespace {

/// A templatized version of a conversion pattern for lowering arithmetic binary
/// ops.
template <typename SrcOpT, typename DstOpT>
struct ArithBinOpConvPat : public OpConversionPattern<SrcOpT> {
  using OpConversionPattern<SrcOpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(SrcOpT op, typename SrcOpT::Adaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<DstOpT>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<sol::CallOp> {
  using OpConversionPattern<sol::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    SmallVector<Type> convertedResTys;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                convertedResTys)))
      return failure();
    r.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), convertedResTys,
                                       adaptor.getOperands());
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<sol::ReturnOp> {
  using OpConversionPattern<sol::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<sol::FuncOp> {
  using OpConversionPattern<sol::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    mlir::Location loc = op.getLoc();
    eravm::Builder eraB(r, loc);

    // Collect non-core attributes.
    std::vector<NamedAttribute> attrs;
    bool hasLinkageAttr = false;
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef attrName = attr.getName();
      if (attrName == "function_type" || attrName == "sym_name" ||
          attrName.startswith("sol."))
        continue;
      if (attrName == "llvm.linkage")
        hasLinkageAttr = true;
      attrs.push_back(attr);
    }

    // Set llvm.linkage attribute to private if not explicitly specified.
    if (!hasLinkageAttr)
      attrs.push_back(r.getNamedAttr(
          "llvm.linkage",
          LLVM::LinkageAttr::get(r.getContext(), LLVM::Linkage::Private)));

    // Set the personality attribute of llvm.
    attrs.push_back(r.getNamedAttr("personality", eraB.getPersonality()));

    // Add the nofree and null_pointer_is_valid attributes of llvm via the
    // passthrough attribute.
    std::vector<Attribute> passthroughAttrs;
    passthroughAttrs.push_back(r.getStringAttr("nofree"));
    passthroughAttrs.push_back(r.getStringAttr("null_pointer_is_valid"));
    attrs.push_back(r.getNamedAttr(
        "passthrough", ArrayAttr::get(r.getContext(), passthroughAttrs)));

    // TODO: Add additional attribute for -O0 and -Oz

    auto convertedFuncTy = cast<FunctionType>(
        getTypeConverter()->convertType(op.getFunctionType()));
    // FIXME: The location of the block arguments are lost here!
    auto newOp =
        r.create<func::FuncOp>(loc, op.getName(), convertedFuncTy, attrs);
    r.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    r.eraseOp(op);
    return success();
  }
};

} // namespace

void evm::populateArithPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<ArithBinOpConvPat<sol::AddOp, arith::AddIOp>,
           ArithBinOpConvPat<sol::SubOp, arith::SubIOp>,
           ArithBinOpConvPat<sol::MulOp, arith::MulIOp>>(tyConv,
                                                         pats.getContext());
}

void evm::populateFuncPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<FuncOpLowering, CallOpLowering, ReturnOpLowering>(tyConv,
                                                             pats.getContext());
}
