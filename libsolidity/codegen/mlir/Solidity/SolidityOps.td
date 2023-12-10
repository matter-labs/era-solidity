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
// Solidity operations
//

#ifndef MLIR_SOLIDITY_SOLIDITYOPS_TD
#define MLIR_SOLIDITY_SOLIDITYOPS_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def I256Attr : SignlessIntegerAttrBase<I<256>,
                                        "256-bit signless integer attribute">;

def Solidity_Dialect : Dialect {
  let name = "sol";
  let summary = "Solidity dialect";
  let description = [{
    This dialect has high level ops for representing solidity.
  }];
  let cppNamespace = "::mlir::sol";

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class Sol_Op<string mnemonic, list<Trait> traits = []> : Op<Solidity_Dialect,
                                                            mnemonic, traits>;

def Sol_ContractOp : Sol_Op<"contract", [Symbol, SingleBlock, NoTerminator]> {
  let summary = "An operation that represents a contract";
  let description = [{
    Example:
    sol.contract @C {
      ...
    }
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "$sym_name $bodyRegion attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins CArg<"StringRef", "{}">:$name)>];
}

// TODO? Move this to a "yul" dialect? Do we need a lot of yul specific ops for
// a new dialect?
def Sol_ObjectOp : Sol_Op<"object", [Symbol, SymbolTable, SingleBlock,
                          NoTerminator]> {
  let summary = "An operation that represents a yul object";
  let description = [{
    Example:
    sol.object @C {
      ...
    }
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "$sym_name $bodyRegion attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins CArg<"StringRef", "{}">:$name)>];
}

// TODO: The op definitions of yul builtins should be auto-generated from
// evmasm::InstructionInfo

def Sol_ReturnOp : Sol_Op<"return"> {
  let summary = "Represents the `return` call in yul";
  let description = [{
    Example:
    sol.return %a %b
  }];
  let arguments = (ins I<256>:$lhs, I<256>:$rhs);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sol_MstoreOp : Sol_Op<"mstore"> {
  let summary = "Represents the `mstore` call in yul";
  let description = [{
    Example:
    sol.mstore %a %b
  }];
  let arguments = (ins I<256>:$lhs, I<256>:$rhs);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sol_MSize : Sol_Op<"msize", [NoSideEffect]> {
  let summary = "Represents the `msize` call in yul";
  let arguments = (ins);
  let results = (outs I<256>:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
}

def Sol_MemGuardOp : Sol_Op<"memoryguard", [ConstantLike, NoSideEffect,
                                            AllTypesMatch<["inp", "out"]>]> {
  let summary = "Represents the `memoryguard` call in yul";
  let arguments = (ins I256Attr:$inp);
  let results = (outs I<256>:$out);
  let assemblyFormat = "$inp attr-dict `:` type($out)";
}

#endif // MLIR_SOLIDITY_SOLIDITYOPS_TD
