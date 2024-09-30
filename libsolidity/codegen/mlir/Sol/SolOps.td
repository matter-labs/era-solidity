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
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// TODO: Add summary and description to all the op definitions.

// TODO: Implement verifiers for all the ops.
//
// Most of the high level sol dialect ops (excluding sol.func + related ops, yul
// specific ops) should verify that their types (ssa argument types and
// TypeAttr) are high level ((un)signed int types, sol dialect types etc.).

// FIXME: Can we have a custom lowering of unrealized_conversion_cast instead?
// ReconcileUnrealizedCasts conversion won't work here.
def Sol_ConvCastOp : Sol_Op<"conv_cast"> {
  let summary = "Like unrealized_conversion_cast; Represents the remapped value";

  let arguments = (ins AnyType:$inp);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `->` type($out)";
}

def Sol_ConstantOp : Op<Sol_Dialect, "constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def Sol_ExtOp : Sol_IntCastOp<"ext"> {
  // TODO: folder
}

def Sol_TruncOp : Sol_IntCastOp<"trunc"> {
  // TODO: folder
}

def Sol_CAddOp : Sol_IntBinaryOp<"cadd", [Commutative]>;

def Sol_CSubOp : Sol_IntBinaryOp<"csub">;

def Sol_AddOp : Sol_IntBinaryOp<"add", [Pure, Commutative]> {
  // TODO: folder, canonicalizer
}

def Sol_SubOp : Sol_IntBinaryOp<"sub", [Pure]> {
  // TODO: folder, canonicalizer
}

def Sol_MulOp : Sol_IntBinaryOp<"mul", [Pure, Commutative]> {
  // TODO: folder, canonicalizer
}

def Sol_CmpOp : Sol_Op<"cmp", [Pure, SameTypeOperands]> {
  let arguments = (ins Sol_CmpPredicateAttr:$predicate, SolInt:$lhs,
                       SolInt:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
  // TODO: folder, canonicalizer
}

def Sol_AllocaOp : Sol_Op<"alloca"> {
  let arguments = (ins TypeAttr:$alloc_type);
  let results = (outs Res<Sol_PtrType, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let builders = [OpBuilder<(ins "Type":$allocType), [{
    $_state.addAttribute("alloc_type", mlir::TypeAttr::get(allocType));
    $_state.addTypes(allocType);
  }]>];
  let skipDefaultBuilders = 1;

  let hasCustomAssemblyFormat = 1;
}

def Sol_MallocOp : Sol_Op<"malloc"> {
  // TODO: Allow any ints. Note that sol.malloc with i256 size could be in the
  // dialect conversion.
  let arguments = (ins UnitAttr:$zero_init, Optional<I256>:$size);
  let results = (outs Sol_AggType:$addr);

  // FIXME: Generated builders should include a version that defaults $size to
  // {}

  let assemblyFormat = [{
    ($size^ `:` type($size))? (`zero_init` $zero_init^)? attr-dict `:` type($addr)
  }];
}

def Sol_StateVarOp : Sol_Op<"state_var", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);

  let assemblyFormat = "$sym_name attr-dict `:` $type";
}

def Sol_AddrOfOp : Sol_Op<"addr_of"> {
  let arguments = (ins FlatSymbolRefAttr:$var);
  // FIXME: AnyType -> Sol_PtrTy causes the asm printer to miss the prefix!
  let results = (outs AnyType:$addr);

  let assemblyFormat = "$var attr-dict `:` type($addr)";
}

def Sol_GepOp : Sol_Op<"gep"> {
  let arguments = (ins AnyType:$baseAddr, SolInt:$idx);
  let results = (outs Sol_RefType:$addr);

  let builders = [
    OpBuilder<(ins "Value":$baseAddr, "Value":$idx)>
  ];

  // TODO: Verifier:
  // - Base address type is an array or a struct type.
  // - Element type should not be a pointer type.
  // - Index should be within bounds for static arrays and structs.
  // - Struct index should be an i64.

  let assemblyFormat = [{
    $baseAddr `,` $idx attr-dict `:` type($baseAddr) `,` type($idx)
      `,` type($addr)
  }];
}

// TODO: Merge with sol.gep.
def Sol_MapOp : Sol_Op<"map"> {
  let arguments = (ins AnyType:$mapping, AnyType:$key);
  let results = (outs AnyType:$addr);

  let assemblyFormat = [{
    $mapping `,` $key attr-dict `:` type($mapping) `,` type($key) `,`
                                    type($addr)
  }];
}

def Sol_CopyOp : Sol_Op<"copy"> {
  let arguments = (ins AnyType:$src, AnyType:$dst);

  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

// TODO: Simplify the asm format by inferring element type of Sol_RefType.

def Sol_LoadOp : Sol_Op<"load"> {
  let arguments = (ins Arg<Sol_RefType, "", [MemRead]>:$addr);
  let results = (outs AnyType:$out);

  let builders = [
    OpBuilder<(ins "Value":$addr), [{
      build($_builder, $_state, getEltType(addr.getType()), addr);
    }]>
  ];

  let assemblyFormat = "$addr attr-dict `:` type($addr) `,` type($out)";
}

def Sol_StoreOp : Sol_Op<"store"> {
  let arguments = (ins AnyType:$val, Arg<Sol_RefType, "", [MemRead]>:$addr);

  let assemblyFormat = [{
    $val `,` $addr attr-dict `:` type($val) `,` type($addr)
  }];
}

def Sol_DataLocCastOp : Sol_Op<"data_loc_cast"> {
  let arguments = (ins AnyType:$inp);
  let results = (outs AnyType:$out);

  let assemblyFormat = [{ $inp attr-dict `:` type($inp) `,` type($out) }];
}

def Sol_EmitOp : Sol_Op<"emit"> {
  let arguments = (ins Variadic<AnyType>:$args, I8Attr:$indexedArgsCount,
                       OptionalAttr<StrAttr>:$signature);

  let builders = [
    OpBuilder<(ins "std::vector<Value>":$indexedArgs,
                    CArg<"std::vector<Value>", "{}">:$nonIndexedArgs,
                    CArg<"std::optional<std::string>",
                         "std::nullopt">:$signature), [{
      $_state.addOperands(indexedArgs);
      $_state.addOperands(nonIndexedArgs);

      IntegerAttr indexedArgsCountAttr =
          $_builder.getIntegerAttr($_builder.getIntegerType(8),
                                   indexedArgs.size());
      $_state.addAttribute(getIndexedArgsCountAttrName($_state.name),
                           indexedArgsCountAttr);

      if (signature) {
        $_state.addAttribute(getSignatureAttrName($_state.name),
                             $_builder.getStringAttr(*signature));
      }
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    OperandRange getIndexedArgs() {
      return OperandRange(getOperands().begin(), getOperands().begin() +
                          getIndexedArgsCount());
    }

    OperandRange getNonIndexedArgs() {
      return OperandRange(getOperands().begin() + getIndexedArgsCount(),
                          getOperands().end());
    }
  }];

  // TODO: Verifier: ($indexedArgsCount <= 3)

  let hasCustomAssemblyFormat = 1;
}

def Sol_RequireOp : Sol_Op<"require"> {
  // FIXME: $msg should be an OptionalAttr, but it doesn't reflect in the
  // generated build API.
  let arguments = (ins I1:$cond, DefaultValuedStrAttr<StrAttr, "">:$msg);

  let assemblyFormat = "$cond `,` $msg attr-dict";
}

def Sol_ContractOp : Sol_Op<"contract", [Symbol, SymbolTable, SingleBlock,
                                         NoTerminator]> {
  // TODO: Represent the following for the lowering:
  // - IRGenerationContext::memoryUnsafeInlineAssemblySeen
  // - IRGenerationContext::reservedMemory
  let arguments = (ins SymbolNameAttr:$sym_name, Sol_ContractKindAttr:$kind,
                   ArrayAttr:$interface_fns,
                   // FIXME: This is a hack until sol.contract lowering happens
                   // during the legalization by SolTypeConverter. Also, if this
                   // is fixed, maybe we could track interface function
                   // attributes in sol.func intead?
                   OptionalAttr<TypeAttrOf<FunctionType>>:$ctorFnType,
                   OptionalAttr<FlatSymbolRefAttr>:$fallbackFn,
                   OptionalAttr<FlatSymbolRefAttr>:$receiveFn);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "$sym_name $bodyRegion attr-dict";
}

//
// FIXME: What's wrong with using func.func and related ops? We could schedule a
// custom conversion to do solidity specific things before the func dialect
// conversion.
//

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
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       UnitAttr:$runtime,
                       UnitAttr:$ctor,
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
    FuncOp clone(IRMapping &mapper);
    FuncOp clone();
    void cloneInto(FuncOp dest, IRMapping &mapper);

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

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

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

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Sol_ReturnOp : Sol_Op<"return", [Pure, HasParent<"FuncOp">,
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
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//
// TODO? Move the rest of the ops to the yul dialect?
//

def Sol_ObjectOp : Sol_Op<"object", [Symbol, SymbolTable, SingleBlock,
                          NoTerminator]> {
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [OpBuilder<(ins CArg<"StringRef", "{}">:$name)>];
  let skipDefaultBuilders = 1;

  let assemblyFormat = "$sym_name $bodyRegion attr-dict";
}

//
// TODO? Should we auto-generate op definitions of yul builtins from
// evmasm::InstructionInfo?
//

def Sol_CallerOp : Sol_Op<"caller", [Pure]> {
  let summary = "Represents the `caller` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_BuiltinRetOp : Sol_Op<"builtin_ret"> {
  let summary = "Represents the `return` call in yul";
  let arguments = (ins I256:$inp0, I256:$inp1);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

def Sol_RevertOp : Sol_Op<"revert"> {
  let summary = "Represents the `revert` call in yul";
  // TODO: Support revert strings
  let arguments = (ins I256:$inp0, I256:$inp1);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

def Sol_MLoadOp : Sol_Op<"mload"> {
  let summary = "Represents the `mload` call in yul";
  let arguments = (ins I256:$inp);
  let results = (outs I256:$out);
  let assemblyFormat = "$inp attr-dict";
}

def Sol_MStoreOp : Sol_Op<"mstore"> {
  let summary = "Represents the `mstore` call in yul";
  let arguments = (ins I256:$inp0, I256:$inp1);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

def Sol_MCopyOp : Sol_Op<"mcopy"> {
  let summary = "Represents the `mcopy` call in yul";
  let arguments = (ins I256:$inp0, I256:$inp1, I256:$inp2);
  let assemblyFormat = "$inp0 `,` $inp1 `,` $inp2 attr-dict";
}

def Sol_MSizeOp : Sol_Op<"msize", [Pure]> {
  let summary = "Represents the `msize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CallValOp : Sol_Op<"callvalue", [Pure]> {
  let summary = "Represents the `callvalue` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CallDataLoadOp : Sol_Op<"calldataload"> {
  let summary = "Represents the `calldataload` call in yul";
  let arguments = (ins I256:$inp);
  let results = (outs I256:$out);
  let assemblyFormat = "$inp attr-dict";
}

def Sol_CallDataSizeOp : Sol_Op<"calldatasize", [Pure]> {
  let summary = "Represents the `calldatasize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CallDataCopyOp : Sol_Op<"calldatacopy", [Pure]> {
  let summary = "Represents the `calldatacopy` call in yul";
  let arguments = (ins I256:$inp0, I256:$inp1, I256:$inp2);
  let assemblyFormat = "$inp0 `,` $inp1 `,` $inp2 attr-dict";
}

def Sol_SLoadOp : Sol_Op<"sload"> {
  let summary = "Represents the `sload` call in yul";
  let arguments = (ins I256:$inp);
  let results = (outs I256:$out);
  let assemblyFormat = "$inp attr-dict";
}

def Sol_SStoreOp : Sol_Op<"sstore"> {
  let summary = "Represents the `sstore` call in yul";
  let arguments = (ins I256:$inp0, I256:$inp1);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

// TODO: Support symbolic references to objects outside the current symbol table
// (including ones outside the translation unit) using SymbolRefAttr instead of
// FlatSymbolRefAttr

def Sol_DataOffsetOp : Sol_Op<"dataoffset", [Pure]> {
  let summary = "Represents the `dataoffset` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$sym);
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_DataSizeOp : Sol_Op<"datasize", [Pure]> {
  let summary = "Represents the `datasize` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$sym);
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CodeSizeOp : Sol_Op<"codesize", [Pure]> {
  let summary = "Represents the `codesize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Sol_CodeCopyOp : Sol_Op<"codecopy"> {
  let summary = "Represents the `codecopy` call in yul";
  let arguments = (ins I256:$inp0, I256:$inp1, I256:$inp2);
  let assemblyFormat = "$inp0 `,` $inp1 `,` $inp2 attr-dict";
}

// TODO: Is this `ConstantLike`? Adding it causes "expected ConstantLike op to
// be foldable" assert fail (probably due to the missing let hasFolder = 1)
def Sol_MemGuardOp : Sol_Op<"memoryguard", [Pure]> {
  let summary = "Represents the `memoryguard` call in yul";
  let arguments = (ins I256Attr:$inp);
  let results = (outs I256:$out);
  let assemblyFormat = "$inp attr-dict";
}

def Sol_Keccak256Op : Sol_Op<"keccak256"> {
  // Note that the keccak256 yul builtin has side effects.
  let summary = "Represents the `keccak256` call in yul";
  let arguments = (ins I256:$inp0, I256:$inp1);
  let results = (outs I256:$out);
  let assemblyFormat = "$inp0 `,` $inp1 attr-dict";
}

def Sol_LogOp : Sol_Op<"log"> {
  let summary = "Represents the `log*` calls in yul";
  let arguments = (ins I256:$addr, I256:$size, Variadic<I256>:$topics);
  // FIXME: Trailing spaces with empty $topics.
  let assemblyFormat = "$addr `,` $size oilist(`topics` `(` $topics `)` ) attr-dict";
}

#endif // MLIR_SOL_SOLOPS_TD
