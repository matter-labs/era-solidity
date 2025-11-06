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
// Yul dialect operations
//

#ifndef MLIR_YUL_YULOPS_TD
#define MLIR_YUL_YULOPS_TD

include "YulBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def Yul_UpdFreePtrOp : Yul_Op<"upd_free_ptr"> {
  let arguments = (ins I256:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$size attr-dict";
}

def Yul_ObjectOp : Yul_Op<"object", [Symbol, SymbolTable, NoTerminator,
                                     IsolatedFromAbove]> {
  let summary = "Represents an `object` in yul";

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins CArg<"StringRef", "{}">:$name),[{
      $_state.addRegion()->emplaceBlock();
      $_state.addAttribute(getSymNameAttrName($_state.name),
                           $_builder.getStringAttr(name));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    Block *getEntryBlock() { return &getBody().front(); }
  }];

  let assemblyFormat = "$sym_name $body attr-dict";
}

//
// We could auto-generate the following evm builtin ops from
// evmasm::InstructionInfo. But I don't think it's worth the effort.
//

def Yul_CallerOp : Yul_Op<"caller", [Pure]> {
  let summary = "Represents the `caller` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_GasOp : Yul_Op<"gas", [Pure]> {
  let summary = "Represents the `gas` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_AddressOp : Yul_Op<"address", [Pure]> {
  let summary = "Represents the `address` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CallOp : Yul_Op<"call"> {
  let summary = "Represents the `call` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$value, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$status);
  let assemblyFormat = [{
    $gas `,` $address `,` $value `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Yul_StaticCallOp : Yul_Op<"static_call"> {
  let summary = "Represents the `staticcall` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$out);
  let assemblyFormat = [{
    $gas `,` $address `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Yul_DelegateCallOp : Yul_Op<"delegate_call"> {
  let summary = "Represents the `delegatecall` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$out);
  let assemblyFormat = [{
    $gas `,` $address `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Yul_ReturnOp : Yul_Op<"return"> {
  let summary = "Represents the `return` call in yul";
  let arguments = (ins I256:$addr, I256:$size);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Yul_RevertOp : Yul_Op<"revert"> {
  let summary = "Represents the `revert` call in yul";
  // TODO: Support revert strings.
  let arguments = (ins I256:$addr, I256:$size);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Yul_StopOp : Yul_Op<"stop"> {
  let summary = "Represents the `stop` call in yul";
  let assemblyFormat = "attr-dict";
}

def Yul_MLoadOp : Yul_Op<"mload"> {
  let summary = "Represents the `mload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_MStoreOp : Yul_Op<"mstore"> {
  let summary = "Represents the `mstore` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

def Yul_MStore8Op : Yul_Op<"mstore8"> {
  let summary = "Represents the `mstore8` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

def Yul_ByteOp : Yul_Op<"byte"> {
  let summary = "Represents the `byte` call in yul";
  let arguments = (ins I256:$idx, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$idx `,` $val attr-dict";
}

def Yul_MCopyOp : Yul_Op<"mcopy"> {
  let summary = "Represents the `mcopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Yul_MSizeOp : Yul_Op<"msize", [Pure]> {
  let summary = "Represents the `msize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CallValOp : Yul_Op<"callvalue", [Pure]> {
  let summary = "Represents the `callvalue` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CallDataLoadOp : Yul_Op<"calldataload"> {
  let summary = "Represents the `calldataload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_CallDataSizeOp : Yul_Op<"calldatasize", [Pure]> {
  let summary = "Represents the `calldatasize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CallDataCopyOp : Yul_Op<"calldatacopy"> {
  let summary = "Represents the `calldatacopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Yul_ReturnDataSizeOp : Yul_Op<"returndatasize", [Pure]> {
  let summary = "Represents the `returndatasize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_ReturnDataCopyOp : Yul_Op<"returndatacopy"> {
  let summary = "Represents the `returndatacopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Yul_SLoadOp : Yul_Op<"sload"> {
  let summary = "Represents the `sload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_SStoreOp : Yul_Op<"sstore"> {
  let summary = "Represents the `sstore` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

def Yul_TLoadOp : Yul_Op<"tload"> {
  let summary = "Represents the `tload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_TStoreOp : Yul_Op<"tstore"> {
  let summary = "Represents the `tstore` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

// TODO: Support symbolic references to objects outside the current symbol table
// (including ones outside the translation unit) using SymbolRefAttr instead of
// FlatSymbolRefAttr

def Yul_DataOffsetOp : Yul_Op<"dataoffset", [Pure]> {
  let summary = "Represents the `dataoffset` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$obj);
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_DataSizeOp : Yul_Op<"datasize", [Pure]> {
  let summary = "Represents the `datasize` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$obj);
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CodeSizeOp : Yul_Op<"codesize", [Pure]> {
  let summary = "Represents the `codesize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CodeCopyOp : Yul_Op<"codecopy"> {
  let summary = "Represents the `codecopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Yul_ExtCodeSizeOp : Yul_Op<"extcodesize", [Pure]> {
  let summary = "Represents the `extcodesize` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_ExtCodeCopyOp : Yul_Op<"extcodecopy"> {
  let summary = "Represents the `extcodecopy` call in yul";
  let arguments = (ins I256:$addr, I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$addr `,` $dst `,` $src `,` $size attr-dict";
}

def Yul_CreateOp : Yul_Op<"create"> {
  let summary = "Represents the `create` call in yul";
  let arguments = (ins I256:$val, I256:$addr, I256:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$val `,` $addr `,` $size attr-dict";
}

def Yul_Create2Op : Yul_Op<"create2"> {
  let summary = "Represents the `create2` call in yul";
  let arguments = (ins I256:$val, I256:$addr, I256:$size, I256:$salt);
  let results = (outs I256:$out);
  let assemblyFormat = "$val `,` $addr `,` $size `,` $salt attr-dict";
}

// TODO: Is this `ConstantLike`? Adding it causes "expected ConstantLike op to
// be foldable" assert fail (probably due to the missing let hasFolder = 1).
def Yul_MemGuardOp : Yul_Op<"memoryguard", [Pure]> {
  let summary = "Represents the `memoryguard` call in yul";
  let arguments = (ins I256Attr:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$size attr-dict";
}

def Yul_Keccak256Op : Yul_Op<"keccak256"> {
  // Note that the keccak256 yul builtin has side effects.
  let summary = "Represents the `keccak256` call in yul";
  let arguments = (ins I256:$addr, I256:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Yul_DivOp : Yul_Op<"div", [Pure]> {
  let summary = "Represents the `div` call in yul";
  let arguments = (ins I256:$dividend, I256:$divisor);
  let results = (outs I256:$out);
  let assemblyFormat = "$dividend `,` $divisor attr-dict";
}

def Yul_SDivOp : Yul_Op<"sdiv", [Pure]> {
  let summary = "Represents the `sdiv` call in yul";
  let arguments = (ins I256:$dividend, I256:$divisor);
  let results = (outs I256:$out);
  let assemblyFormat = "$dividend `,` $divisor attr-dict";
}

def Yul_ModOp : Yul_Op<"mod", [Pure]> {
  let summary = "Represents the `mod` call in yul";
  let arguments = (ins I256:$value, I256:$mod);
  let results = (outs I256:$out);
  let assemblyFormat = "$value `,` $mod attr-dict";
}

def Yul_SModOp : Yul_Op<"smod", [Pure]> {
  let summary = "Represents the `smod` call in yul";
  let arguments = (ins I256:$value, I256:$mod);
  let results = (outs I256:$out);
  let assemblyFormat = "$value `,` $mod attr-dict";
}

def Yul_ShlOp : Yul_Op<"shl", [Pure]> {
  let summary = "Represents the `shl` call in yul";
  let arguments = (ins I256:$shift, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$shift `,` $val attr-dict";
}

def Yul_ShrOp : Yul_Op<"shr", [Pure]> {
  let summary = "Represents the `shr` call in yul";
  let arguments = (ins I256:$shift, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$shift `,` $val attr-dict";
}

def Yul_SarOp : Yul_Op<"sar", [Pure]> {
  let summary = "Represents the `sar` call in yul";
  let arguments = (ins I256:$shift, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$shift `,` $val attr-dict";
}

def Yul_ExpOp : Yul_Op<"exp", [Pure]> {
  let summary = "Represents the `exp` call in yul";
  let arguments = (ins I256:$base, I256:$exp);
  let results = (outs I256:$out);
  let assemblyFormat = "$base `,` $exp attr-dict";
}

def Yul_LogOp : Yul_Op<"log"> {
  let summary = "Represents the `log*` calls in yul";
  let arguments = (ins I256:$addr, I256:$size, Variadic<I256>:$topics);
  // FIXME: Trailing spaces with empty $topics.
  let assemblyFormat = "$addr `,` $size oilist(`topics` `(` $topics `)` ) attr-dict";
}

def Yul_SetImmutableOp : Yul_Op<"setimmutable"> {
  let summary = "Represents the `setimmutable` call in yul";
  let arguments = (ins I256:$addr, StrAttr:$name, I256:$val);
  let assemblyFormat = "$addr `,` $name `,` $val attr-dict";
}

def Yul_LoadImmutableOp : Yul_Op<"loadimmutable"> {
  let summary = "Represents the `loadimmutable` call in yul";
  let arguments = (ins StrAttr:$name);
  let results = (outs I256:$out);
  let assemblyFormat = "$name attr-dict";
}

def Yul_LinkerSymbolOp : Yul_Op<"linkersymbol"> {
  let summary = "Represents the `linkersymbol` call in yul";
  let arguments = (ins StrAttr:$name);
  let results = (outs I256:$out);
  let assemblyFormat = "$name attr-dict";
}

#endif // MLIR_YUL_YULOPS_TD
