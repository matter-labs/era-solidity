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
// EVM specific type converter of the sol dialect
//

#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolidity/codegen/mlir/Target/EVM/SolToStandard.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

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
