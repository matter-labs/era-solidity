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
// Sol dialect
//

#ifndef MLIR_SOL_SOLBASE_TD
#define MLIR_SOL_SOLBASE_TD

include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def I256 : I<256>;
def I256Attr : SignlessIntegerAttrBase<I256,
                                        "256-bit signless integer attribute">;

// TODO: Replace 'X' with something better ("Signfull" or something?).
def AnyXInt : AnyTypeOf<[AnySignedInteger, AnyUnsignedInteger]>;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Sol_Dialect : Dialect {
  let name = "sol";
  let summary = "Solidity dialect";
  let description = [{
    This dialect has high level ops for representing solidity.
  }];
  let cppNamespace = "::mlir::sol";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class Sol_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Sol_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Sol_DataLocation : I32EnumAttr<"DataLocation",
                                   "Data-location in reference-types",
                                   [I32EnumAttrCase<"Storage", 0>,
                                    I32EnumAttrCase<"CallData", 1>,
                                    I32EnumAttrCase<"Memory", 2>,
                                    I32EnumAttrCase<"Stack", 3>]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::sol";
}

def Sol_StringType : Sol_Type<"String", "string"> {
  let summary = "Sol string type";
  let description = [{
    `sol.string` represents solidity string and bytes type.

    Example:

    ```mlir
    !sol.string<Memory>
    !sol.string<Storage>
    ```
  }];

  let parameters = (ins Sol_DataLocation:$dataLocation);

  let hasCustomAssemblyFormat = 1;
}

def Sol_ArrayType : Sol_Type<"Array", "array"> {
  let summary = "Sol array type";
  let description = [{
    `sol.array` represents solidity arrays.

    Example:

    ```mlir
    !sol.array<2 x i256, CallData>
    !sol.array<? x i256, Memory>
    ```
  }];

  // TODO? Tracking the whole shape here (instead of the recursive
  // representation) can make the dump easier to read (like that of memref). Is
  // that better?  Solidity ast-type for arrays use a recursive representation.
  let parameters = (ins "int64_t":$size, "Type":$eltType,
                    Sol_DataLocation:$dataLocation);

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Returns true if the size is dynamic.
    bool isDynSized() { return getSize() == -1; }
  }];
}

def Sol_StructType : Sol_Type<"Struct", "struct"> {
  let summary = "Sol struct type";
  let description = [{
    `sol.struct` represents solidity struct.

    Example:

    ```mlir
    !sol.struct<(i256), CallData>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$memTypes,
                        Sol_DataLocation:$dataLocation);

  let hasCustomAssemblyFormat = 1;
}

def Sol_MappingType : Sol_Type<"Mapping", "mapping"> {
  let summary = "Sol mapping type";
  let description = [{
    `sol.mapping` represents the solidity mapping type.

    Example:

    ```mlir
    !sol.mapping<i160, i256>
    ```
  }];

  let parameters = (ins "Type":$keyType, "Type":$valType);

  let assemblyFormat = [{ `<` $keyType `,` $valType `>` }];
}

def Sol_PtrType : Sol_Type<"Pointer", "ptr"> {
  let summary = "Sol pointer type";
  let description = [{
    `sol.ptr` represents a pointer with an element type that's not a reference
    type.

    Example:

    ```mlir
    !sol.ptr<i256, Storage>
    ```
  }];
  let parameters = (ins "Type":$pointeeType, Sol_DataLocation:$dataLocation);
  let hasCustomAssemblyFormat = 1;
}

def Sol_AggType : AnyTypeOf<[Sol_StringType, Sol_ArrayType, Sol_StructType],
                             "Solidity aggregate types">;
def Sol_RefType : AnyTypeOf<[Sol_PtrType, Sol_AggType],
                             "Solidity aggregate types and pointers">;


//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def Sol_ContractKind : I32EnumAttr<"ContractKind",
                                   "Represents the solidity contract kind",
                                   [I32EnumAttrCase<"Interface", 0>,
                                    I32EnumAttrCase<"Contract", 1>,
                                    I32EnumAttrCase<"Library", 2>]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::sol";
}
def Sol_ContractKindAttr : EnumAttr<Sol_Dialect, Sol_ContractKind,
                                    "ContractKind">;

def Sol_StateMutability :
    I32EnumAttr<"StateMutability", "Represents the state mutability of functions",
                 [I32EnumAttrCase<"Pure", 0>,
                  I32EnumAttrCase<"View", 1>,
                  I32EnumAttrCase<"NonPayable", 2>,
                  I32EnumAttrCase<"Payable", 3>]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::sol";
}
def Sol_StateMutabilityAttr : EnumAttr<Sol_Dialect, Sol_StateMutability,
                                       "StateMutability">;

//===----------------------------------------------------------------------===//
// Op
//===----------------------------------------------------------------------===//

class Sol_Op<string mnemonic, list<Trait> traits = []> : Op<Sol_Dialect,
                                                            mnemonic, traits>;
// The following 2 classes are copied from ArithOps.td.

class Sol_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Sol_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class Sol_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    Sol_BinaryOp<mnemonic, traits #
      [DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Arguments<(ins AnyXInt:$lhs, AnyXInt:$rhs)>,
    Results<(outs AnyXInt:$result)>;
    // TODO: Add verifier that check the signedness semantics.

#endif // MLIR_SOL_SOLBASE_TD
