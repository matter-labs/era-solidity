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

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def Solidity_Dialect : Dialect {
  let name = "solidity";
  let summary = "Solidity dialect";
  let description = [{
    This dialect has high level ops for representing solidity.
  }];
  let cppNamespace = "::mlir::solidity";

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class Solidity_Op<string mnemonic,
                  list<Trait> traits = []> : Op<Solidity_Dialect, mnemonic,
                                                traits>;

def Solidity_ContractOp : Solidity_Op<"contract", [Symbol, SingleBlock,
                                      NoTerminator]> {
  let summary = "An operation that represents a contract";
  let description = [{
    Example:
    solidity.contract @C {
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
def Solidity_ObjectOp : Solidity_Op<"object", [Symbol, SingleBlock,
                                    NoTerminator]> {
  let summary = "An operation that represents a yul object";
  let description = [{
    Example:
    solidity.object @C {
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

def Solidity_ReturnOp : Solidity_Op<"return"> {
  let summary = "Represents the `return` call in yul";
  let description = [{
    Example:
    solidity.return %a %b
  }];
  let arguments = (ins I<256>:$lhs, I<256>:$rhs);
}

def Solidity_MstoreOp : Solidity_Op<"mstore"> {
  let summary = "Represents the `mstore` call in yul";
  let description = [{
    Example:
    solidity.mstore %a %b
  }];
  let arguments = (ins I<256>:$lhs, I<256>:$rhs);
}

#endif // MLIR_SOLIDITY_SOLIDITYOPS_TD
