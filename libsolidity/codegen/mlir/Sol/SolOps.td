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
// Sol dialect operations
//

#ifndef MLIR_SOL_SOLOPS_TD
#define MLIR_SOL_SOLOPS_TD

include "SolBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def Sol_ContractOp : Sol_Op<"contract", [Symbol, SymbolTable, SingleBlock,
                                         NoTerminator]> {
  let summary = "An operation that represents a contract";
  let description = [{
    Example:
    sol.contract @C {
      ...
    }
  }];

  // TODO: Represent the following for the lowering:
  // - IRGenerationContext::memoryUnsafeInlineAssemblySeen
  // - IRGenerationContext::reservedMemory
  let arguments = (ins SymbolNameAttr:$sym_name, Sol_ContractKindAttr:$kind,
                   ArrayAttr:$interface_fns,
                   OptionalAttr<FlatSymbolRefAttr>:$ctor,
                   OptionalAttr<FlatSymbolRefAttr>:$fallbackFn,
                   OptionalAttr<FlatSymbolRefAttr>:$receiveFn);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "$sym_name $bodyRegion attr-dict";

  let extraClassDeclaration = [{
    DictionaryAttr getInterfaceFnAttr(sol::FuncOp fn);
  }];
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
  // TODO: Implement a verifier that checks for things like:
  // - Maximum of 2 nesting of sol.object
  // - All func.funcs to be only under sol.object
}

def Sol_FuncOp : Sol_Op<"func", [AutomaticAllocationScope, CallableOpInterface,
                                 FunctionOpInterface, IsolatedFromAbove,
                                 Symbol, SingleBlock]> {
  let summary = "FuncOp with solidity specific semantics";
  let description = [{
    This is similar to a `func.func` but it tracks solidity specific attributes.

    Example:

    sol.func @f() -> () {
      ...
    }
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       UnitAttr:$runtime,
                       OptionalAttr<Sol_StateMutabilityAttr>:$state_mutability,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>,
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      "StateMutability":$stateMutability,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    // Copied from func.func.
    FuncOp clone(BlockAndValueMapping &mapper);
    FuncOp clone();
    void cloneInto(FuncOp dest, BlockAndValueMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def CallOp : Sol_Op<"call", [CallOpInterface,
                             DeclareOpInterfaceMethods<SymbolUserOpInterface>]
                   >{
  let summary = "call operation";
  let description = [{
    This is similar to a `func.call` but the callee is expected to be a
    `sol.func`.

    Example:

    ```mlir
    %2 = func.call @my_add(%0, %1) : (i32, i32) -> i32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType() {
      return FunctionType::get(getContext(), getOperandTypes(),
                               getResultTypes());
    }

    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Sol_ReturnOp : Sol_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                     ReturnLike, Terminator]> {
  let summary = "ReturnOp for sol.func";
  let description = [{
    This is similar to a `func.return` but is expected have sol.func as the
    parent op.

    Example:

    ```mlir
    sol.func @foo() : (i32, f8) {
      ...
      sol.return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, llvm::None);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Sol_AllocaOp : Sol_Op<"alloca"> {
  let summary = "Stack allocation op";
  let description = [{
    This operation performs allocations on the stack data-location.

    Example:

    ```mlir
    %addr = sol.alloca : !sol.ptr<i256>
    ```
  }];

  let arguments = (ins TypeAttr:$alloc_type);
  let results = (outs Res<Sol_PtrType, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Type":$allocType), [{
    $_state.addAttribute("alloc_type", mlir::TypeAttr::get(allocType));
    $_state.addTypes(allocType);
  }]>];

  let hasCustomAssemblyFormat = 1;
}

def Sol_MallocOp : Sol_Op<"malloc"> {
  let summary = "Memory allocation op";
  let description = [{
    This operation performs allocations for aggregate types on the memory
    data-location. The optional size argument is used in the dynamic allocation
    of the outermost dimension. Inner dynamic allocations are initialized to the
    zero pointer. The outermost dimension is initialized to the zero pointer if
    the size argument is absent.

    Example:

    ```mlir
    %addr = sol.malloc : !sol.array<2 x i256, Memory>
    %addr = sol.malloc %size : !sol.array<? x i256, Memory>
    ```
  }];

  let arguments = (ins TypeAttr:$alloc_type, Optional<I<256>>:$size);
  let results = (outs Sol_AggType:$addr);

  let hasCustomAssemblyFormat = 1;
}

// TODO: Can we let the target decided the layout in the storage? This can, for
// instance, enable targets to align storage allocations.
def Sol_StorageLoadOp : Sol_Op<"storage_load"> {
  let summary = "Storage load op";
  let description = [{
    This operation loads an i256 value from storage at the given offsets. The
    value is truncated or extended to the effective type (but always yields an
    i256).

    Example:
    sol.storage_load i32 @var %byte_offset {slot_offset = 0 : i256}
  }];

  let arguments = (ins FlatSymbolRefAttr:$sym_name, TypeAttr:$effective_type,
                       I256Attr:$slot_offset, I32:$byte_offset);
  let results = (outs I<256>:$out);
  // TODO: Can we get the asm format to assume that the slot_offset is an i256?
  let assemblyFormat = "$effective_type $sym_name $byte_offset attr-dict";
}

// TODO: Simplify the asm format by inferring element type of Sol_RefType.
// TODO? Use a single type (index?) for array/struct size and indexing that
// conforms to that of solidity.

def Sol_LoadOp : Sol_Op<"load"> {
  let arguments = (ins Arg<Sol_RefType, "", [MemRead]>:$baseAddr,
                       Variadic<I<256>>:$indices);
  let results = (outs AnyType:$out);
  let assemblyFormat = [{
    $baseAddr (`[` $indices^ `]`)? attr-dict `:` type($baseAddr) `,` type($out)
  }];

  let builders = [
    OpBuilder<(ins "Value":$baseAddr, CArg<"ValueRange", "{}">:$indices), [{
      build($_builder, $_state, getEltType(baseAddr.getType()),
        baseAddr, indices);
    }]>
  ];
  // TODO: Implement a verifier to check the validity of constant index of
  // struct types.
}

def Sol_StoreOp : Sol_Op<"store"> {
  let arguments = (ins AnyType:$val, Arg<Sol_RefType, "", [MemRead]>:$baseAddr,
                       Variadic<I<256>>:$indices);
  let assemblyFormat = [{
    $val `:` type($val) `,`
      $baseAddr (`[` $indices^ `]`)? attr-dict `:` type($baseAddr)
  }];

  let builders = [
    OpBuilder<(ins "Value":$val, "Value":$baseAddr), [{
      build($_builder, $_state, val, baseAddr, /*indices=*/{});
    }]>
  ];
}

// TODO: The op definitions of yul builtins should be auto-generated from
// evmasm::InstructionInfo

def Sol_BuiltinRetOp : Sol_Op<"builtin_ret"> {
  let summary = "Represents the `return` call in yul";
  let description = [{
    Example:
    sol.return %a %b
  }];
  let arguments = (ins I<256>:$lhs, I<256>:$rhs);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Sol_RevertOp : Sol_Op<"revert"> {
  let summary = "Represents the `revert` call in yul";
  // TODO: Support revert strings
  let arguments = (ins I<256>:$inp0, I<256>:$inp1);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

def Sol_MLoadOp : Sol_Op<"mload"> {
  let summary = "Represents the `mload` call in yul";
  let arguments = (ins I<256>:$inp0);
  let results = (outs I<256>:$out);
  let assemblyFormat = "$inp0 attr-dict";
}

def Sol_MStoreOp : Sol_Op<"mstore"> {
  let summary = "Represents the `mstore` call in yul";
  let description = [{
    Example:
    sol.mstore %a %b
  }];
  let arguments = (ins I<256>:$inp0, I<256>:$inp1);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

def Sol_MSizeOp : Sol_Op<"msize", [NoSideEffect]> {
  let summary = "Represents the `msize` call in yul";
  let arguments = (ins);
  let results = (outs I<256>:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
}

def Sol_CallValOp : Sol_Op<"callvalue", [NoSideEffect]> {
  let summary = "Represents the `callvalue` call in yul";
  let arguments = (ins);
  let results = (outs I<256>:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
}

def Sol_CallDataLoadOp : Sol_Op<"calldataload"> {
  let summary = "Represents the `calldataload` call in yul";
  let arguments = (ins I<256>:$inp0);
  let results = (outs I<256>:$out);
  let assemblyFormat = "$inp0 attr-dict";
}

def Sol_CallDataSizeOp : Sol_Op<"calldatasize", [NoSideEffect]> {
  let summary = "Represents the `calldatasize` call in yul";
  let arguments = (ins);
  let results = (outs I<256>:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
}

def Sol_CallDataCopyOp : Sol_Op<"calldatacopy", [NoSideEffect]> {
  let summary = "Represents the `calldatacopy` call in yul";
  let arguments = (ins I<256>:$inp0, I<256>:$inp1, I<256>:$inp2);
  let assemblyFormat = "$inp0 `,` $inp1 `,` $inp2 attr-dict";
}

def Sol_SLoadOp : Sol_Op<"sload"> {
  let summary = "Represents the `sload` call in yul";
  let arguments = (ins I<256>:$inp0);
  let results = (outs I<256>:$out);
  let assemblyFormat = "$inp0 attr-dict";
}

def Sol_SStoreOp : Sol_Op<"sstore"> {
  let summary = "Represents the `sstore` call in yul";
  let arguments = (ins I<256>:$inp0, I<256>:$inp1);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

// TODO: Support symbolic references to objects outside the current symbol table
// (including ones outside the translation unit) using SymbolRefAttr instead of
// FlatSymbolRefAttr

def Sol_DataOffsetOp : Sol_Op<"dataoffset", [NoSideEffect]> {
  let summary = "Represents the `dataoffset` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$sym);
  let results = (outs I<256>:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
}

def Sol_DataSizeOp : Sol_Op<"datasize", [NoSideEffect]> {
  let summary = "Represents the `datasize` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$sym);
  let results = (outs I<256>:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
}

def Sol_CodeCopyOp : Sol_Op<"codecopy"> {
  let summary = "Represents the `codecopy` call in yul";
  let arguments = (ins I<256>:$inp0, I<256>:$inp1, I<256>:$inp2);
  let assemblyFormat = "$inp0 `,` $inp1 `,` $inp2 attr-dict";
}

// TODO: Is this `ConstantLike`? Adding it causes "expected ConstantLike op to
// be foldable" assert fail (probably due to the missing let hasFolder = 1)
def Sol_MemGuardOp : Sol_Op<"memoryguard", [NoSideEffect,
                                            AllTypesMatch<["inp", "out"]>]> {
  let summary = "Represents the `memoryguard` call in yul";
  let arguments = (ins I256Attr:$inp);
  let results = (outs I<256>:$out);
  let assemblyFormat = "$inp attr-dict `:` type($out)";
}

def Sol_Keccak256Op : Sol_Op<"keccak256"> {
  // Note that the keccak256 yul builtin has side effects.
  let summary = "Represents the `keccak256` call in yul";
  let arguments = (ins I<256>:$inp0, I<256>:$inp1);
  let results = (outs I<256>:$out);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

#endif // MLIR_SOL_SOLOPS_TD
