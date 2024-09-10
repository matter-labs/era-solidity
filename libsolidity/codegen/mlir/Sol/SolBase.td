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

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Sol_Dialect : Dialect {
  let name = "sol";
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

def I256 : I<256>;
def SolInt : AnyTypeOf<[I1, AnySignedInteger, AnyUnsignedInteger]>;

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
  let parameters = (ins Sol_DataLocation:$dataLocation);

  let hasCustomAssemblyFormat = 1;
}

def Sol_ArrayType : Sol_Type<"Array", "array"> {
  let parameters = (ins "int64_t":$size, "Type":$eltType,
                    Sol_DataLocation:$dataLocation);

  let extraClassDeclaration = [{
    /// Returns true if the size is dynamic.
    bool isDynSized() { return getSize() == -1; }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Sol_StructType : Sol_Type<"Struct", "struct"> {
  let parameters = (ins ArrayRefParameter<"Type">:$memTypes,
                        Sol_DataLocation:$dataLocation);

  let hasCustomAssemblyFormat = 1;
}

def Sol_MappingType : Sol_Type<"Mapping", "mapping"> {
  let parameters = (ins "Type":$keyType, "Type":$valType);

  let assemblyFormat = [{ `<` $keyType `,` $valType `>` }];
}

def Sol_PtrType : Sol_Type<"Pointer", "ptr"> {
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

def I256Attr : SignlessIntegerAttrBase<I256,
                                       "256-bit signless integer attribute">;

def Sol_CmpPredicateAttr :
    I64EnumAttr<"CmpPredicate", "",
                [I64EnumAttrCase<"eq", 0>,
                 I64EnumAttrCase<"ne", 1>,
                 I64EnumAttrCase<"lt", 2>,
                 I64EnumAttrCase<"le", 3>,
                 I64EnumAttrCase<"gt", 4>,
                 I64EnumAttrCase<"ge", 5>]> {
  let cppNamespace = "::mlir::sol";
}

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

//===----------------------------------------------------------------------===//
// Op classes
//===----------------------------------------------------------------------===//

class Sol_Op<string mnemonic, list<Trait> traits = []> : Op<Sol_Dialect,
                                                            mnemonic, traits>;

class Sol_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    Sol_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SolInt:$lhs, SolInt:$rhs);
  let results = (outs SolInt:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class Sol_IntCastOp<string mnemonic, list<Trait> traits = []> :
    Sol_Op<mnemonic, traits #
                     [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let arguments = (ins SolInt:$in);
  let results = (outs SolInt:$out);

  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

#endif // MLIR_SOL_SOLBASE_TD
