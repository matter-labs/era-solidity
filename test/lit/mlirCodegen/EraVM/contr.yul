// RUN: solc --strict-assembly --mlir-action=print-std-mlir --mlir-target=eravm --mmlir --mlir-print-debuginfo %s | FileCheck %s

object "Test" {
  code {
    mstore(64, memoryguard(0x80))
    if callvalue() {
      revert(0, 0)
    }
    let freePtr := mload(64)
    codecopy(freePtr, dataoffset("Test_deployed"), datasize("Test_deployed"))
    return(freePtr, datasize("Test_deployed"))
  }
  object "Test_deployed" {
    code {
      mstore(64, memoryguard(0x80))
      if iszero(lt(calldatasize(), 4)) {
        let selector := shr(224, calldataload(0))
        switch selector
        case 0x26121ff0 {
          if callvalue() {
            revert(0, 0)
          }
          if slt(sub(calldatasize(), 4), 0)
          {
            revert(0, 0)
          }
          let ret := f()
          let memPos := mload(64)
          let memEnd := add(memPos, 32)
          mstore(memPos, ret)
          return(memPos, sub(memEnd, memPos))
        }
        default {}
      }
      revert(0, 0)

      function f() -> r {
        r := 42
      }
    }
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc = loc(unknown)
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func private @__return(i256, i256, i256) attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} loc(#loc)
// CHECK-NEXT:   func.func private @".unreachable"() attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     llvm.unreachable loc(#loc1)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   func.func private @__revert(i256, i256, i256) attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} loc(#loc)
// CHECK-NEXT:   func.func private @__deploy() attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256 loc(#loc2)
// CHECK-NEXT:     %c128_i256 = arith.constant 128 : i256 loc(#loc3)
// CHECK-NEXT:     %0 = llvm.inttoptr %c64_i256 : i256 to !llvm.ptr<1> loc(#loc4)
// CHECK-NEXT:     llvm.store %c128_i256, %0 {alignment = 1 : i64} : i256, !llvm.ptr<1> loc(#loc4)
// CHECK-NEXT:     %1 = "llvm.intrcall"() <{id = 3177 : i32, name = "eravm.getu128"}> : () -> i256 loc(#loc5)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc5)
// CHECK-NEXT:     %2 = arith.cmpi ne, %1, %c0_i256 : i256 loc(#loc5)
// CHECK-NEXT:     scf.if %2 {
// CHECK-NEXT:       %c0_i256_9 = arith.constant 0 : i256 loc(#loc7)
// CHECK-NEXT:       %c0_i256_10 = arith.constant 0 : i256 loc(#loc8)
// CHECK-NEXT:       %c2_i256_11 = arith.constant 2 : i256 loc(#loc1)
// CHECK-NEXT:       func.call @__revert(%c0_i256_9, %c0_i256_10, %c2_i256_11) : (i256, i256, i256) -> () loc(#loc1)
// CHECK-NEXT:       func.call @".unreachable"() : () -> () loc(#loc1)
// CHECK-NEXT:     } loc(#loc6)
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc9)
// CHECK-NEXT:     %3 = llvm.alloca %c1_i256 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc10)
// CHECK-NEXT:     %c64_i256_0 = arith.constant 64 : i256 loc(#loc11)
// CHECK-NEXT:     %4 = llvm.inttoptr %c64_i256_0 : i256 to !llvm.ptr<1> loc(#loc12)
// CHECK-NEXT:     %5 = llvm.load %4 {alignment = 1 : i64} : !llvm.ptr<1> -> i256 loc(#loc12)
// CHECK-NEXT:     llvm.store %5, %3 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc9)
// CHECK-NEXT:     %6 = llvm.load %3 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc13)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc14)
// CHECK-NEXT:     %c0_i256_2 = arith.constant 0 : i256 loc(#loc15)
// CHECK-NEXT:     %7 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>> loc(#loc16)
// CHECK-NEXT:     %8 = llvm.load %7 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc16)
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%c0_i256_1] : (!llvm.ptr<3>, i256) -> !llvm.ptr<3>, i8 loc(#loc16)
// CHECK-NEXT:     %10 = llvm.inttoptr %6 : i256 to !llvm.ptr<1> loc(#loc16)
// CHECK-NEXT:     "llvm.intr.memcpy"(%10, %9, %c0_i256_2) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<3>, i256) -> () loc(#loc16)
// CHECK-NEXT:     %11 = llvm.load %3 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc17)
// CHECK-NEXT:     %c0_i256_3 = arith.constant 0 : i256 loc(#loc18)
// CHECK-NEXT:     %c256_i256 = arith.constant 256 : i256 loc(#loc19)
// CHECK-NEXT:     %12 = llvm.inttoptr %c256_i256 : i256 to !llvm.ptr<2> loc(#loc19)
// CHECK-NEXT:     %c32_i256 = arith.constant 32 : i256 loc(#loc19)
// CHECK-NEXT:     llvm.store %c32_i256, %12 {alignment = 1 : i64} : i256, !llvm.ptr<2> loc(#loc19)
// CHECK-NEXT:     %c288_i256 = arith.constant 288 : i256 loc(#loc19)
// CHECK-NEXT:     %13 = llvm.inttoptr %c288_i256 : i256 to !llvm.ptr<2> loc(#loc19)
// CHECK-NEXT:     %c0_i256_4 = arith.constant 0 : i256 loc(#loc19)
// CHECK-NEXT:     llvm.store %c0_i256_4, %13 {alignment = 1 : i64} : i256, !llvm.ptr<2> loc(#loc19)
// CHECK-NEXT:     %c0_i256_5 = arith.constant 0 : i256 loc(#loc19)
// CHECK-NEXT:     %c2_i256 = arith.constant 2 : i256 loc(#loc19)
// CHECK-NEXT:     %14 = arith.muli %c0_i256_5, %c2_i256 : i256 loc(#loc19)
// CHECK-NEXT:     %c64_i256_6 = arith.constant 64 : i256 loc(#loc19)
// CHECK-NEXT:     %15 = arith.addi %14, %c64_i256_6 : i256 loc(#loc19)
// CHECK-NEXT:     %c256_i256_7 = arith.constant 256 : i256 loc(#loc19)
// CHECK-NEXT:     %c2_i256_8 = arith.constant 2 : i256 loc(#loc19)
// CHECK-NEXT:     call @__return(%c256_i256_7, %15, %c2_i256_8) : (i256, i256, i256) -> () loc(#loc19)
// CHECK-NEXT:     call @".unreachable"() : () -> () loc(#loc19)
// CHECK-NEXT:     llvm.unreachable loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   func.func private @__runtime() attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality, runtime} {
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256 loc(#loc20)
// CHECK-NEXT:     %c128_i256 = arith.constant 128 : i256 loc(#loc21)
// CHECK-NEXT:     %0 = llvm.inttoptr %c64_i256 : i256 to !llvm.ptr<1> loc(#loc22)
// CHECK-NEXT:     llvm.store %c128_i256, %0 {alignment = 1 : i64} : i256, !llvm.ptr<1> loc(#loc22)
// CHECK-NEXT:     %1 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256> loc(#loc23)
// CHECK-NEXT:     %2 = llvm.load %1 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc23)
// CHECK-NEXT:     %c4_i256 = arith.constant 4 : i256 loc(#loc24)
// CHECK-NEXT:     %3 = arith.cmpi ult, %2, %c4_i256 : i256 loc(#loc25)
// CHECK-NEXT:     %4 = arith.extui %3 : i1 to i256 loc(#loc25)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc26)
// CHECK-NEXT:     %5 = arith.cmpi eq, %4, %c0_i256 : i256 loc(#loc26)
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       %c1_i256 = arith.constant 1 : i256 loc(#loc28)
// CHECK-NEXT:       %6 = llvm.alloca %c1_i256 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc29)
// CHECK-NEXT:       %c0_i256_3 = arith.constant 0 : i256 loc(#loc30)
// CHECK-NEXT:       %7 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>> loc(#loc31)
// CHECK-NEXT:       %8 = llvm.load %7 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc31)
// CHECK-NEXT:       %9 = llvm.getelementptr %8[%c0_i256_3] : (!llvm.ptr<3>, i256) -> !llvm.ptr<3>, i8 loc(#loc31)
// CHECK-NEXT:       %10 = llvm.load %9 {alignment = 1 : i64} : !llvm.ptr<3> -> i256 loc(#loc31)
// CHECK-NEXT:       %c224_i256 = arith.constant 224 : i256 loc(#loc32)
// CHECK-NEXT:       %11 = arith.shrui %10, %c224_i256 : i256 loc(#loc33)
// CHECK-NEXT:       llvm.store %11, %6 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc28)
// CHECK-NEXT:       %12 = llvm.load %6 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc34)
// CHECK-NEXT:       scf.int_switch %12 : i256
// CHECK-NEXT:       case 638722032 {
// CHECK-NEXT:         %13 = "llvm.intrcall"() <{id = 3177 : i32, name = "eravm.getu128"}> : () -> i256 loc(#loc36)
// CHECK-NEXT:         %c0_i256_4 = arith.constant 0 : i256 loc(#loc36)
// CHECK-NEXT:         %14 = arith.cmpi ne, %13, %c0_i256_4 : i256 loc(#loc36)
// CHECK-NEXT:         scf.if %14 {
// CHECK-NEXT:           %c0_i256_12 = arith.constant 0 : i256 loc(#loc38)
// CHECK-NEXT:           %c0_i256_13 = arith.constant 0 : i256 loc(#loc39)
// CHECK-NEXT:           %c0_i256_14 = arith.constant 0 : i256 loc(#loc40)
// CHECK-NEXT:           func.call @__revert(%c0_i256_12, %c0_i256_13, %c0_i256_14) : (i256, i256, i256) -> () loc(#loc40)
// CHECK-NEXT:           func.call @".unreachable"() : () -> () loc(#loc40)
// CHECK-NEXT:         } loc(#loc37)
// CHECK-NEXT:         %15 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256> loc(#loc41)
// CHECK-NEXT:         %16 = llvm.load %15 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc41)
// CHECK-NEXT:         %c4_i256_5 = arith.constant 4 : i256 loc(#loc42)
// CHECK-NEXT:         %17 = arith.subi %16, %c4_i256_5 : i256 loc(#loc43)
// CHECK-NEXT:         %c0_i256_6 = arith.constant 0 : i256 loc(#loc44)
// CHECK-NEXT:         %18 = arith.cmpi slt, %17, %c0_i256_6 : i256 loc(#loc45)
// CHECK-NEXT:         scf.if %18 {
// CHECK-NEXT:           %c0_i256_12 = arith.constant 0 : i256 loc(#loc47)
// CHECK-NEXT:           %c0_i256_13 = arith.constant 0 : i256 loc(#loc48)
// CHECK-NEXT:           %c0_i256_14 = arith.constant 0 : i256 loc(#loc49)
// CHECK-NEXT:           func.call @__revert(%c0_i256_12, %c0_i256_13, %c0_i256_14) : (i256, i256, i256) -> () loc(#loc49)
// CHECK-NEXT:           func.call @".unreachable"() : () -> () loc(#loc49)
// CHECK-NEXT:         } loc(#loc46)
// CHECK-NEXT:         %c1_i256_7 = arith.constant 1 : i256 loc(#loc50)
// CHECK-NEXT:         %19 = llvm.alloca %c1_i256_7 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc51)
// CHECK-NEXT:         %20 = func.call @f() : () -> i256 loc(#loc52)
// CHECK-NEXT:         llvm.store %20, %19 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc50)
// CHECK-NEXT:         %c1_i256_8 = arith.constant 1 : i256 loc(#loc53)
// CHECK-NEXT:         %21 = llvm.alloca %c1_i256_8 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc54)
// CHECK-NEXT:         %c64_i256_9 = arith.constant 64 : i256 loc(#loc55)
// CHECK-NEXT:         %22 = llvm.inttoptr %c64_i256_9 : i256 to !llvm.ptr<1> loc(#loc56)
// CHECK-NEXT:         %23 = llvm.load %22 {alignment = 1 : i64} : !llvm.ptr<1> -> i256 loc(#loc56)
// CHECK-NEXT:         llvm.store %23, %21 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc53)
// CHECK-NEXT:         %c1_i256_10 = arith.constant 1 : i256 loc(#loc57)
// CHECK-NEXT:         %24 = llvm.alloca %c1_i256_10 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc58)
// CHECK-NEXT:         %25 = llvm.load %21 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc59)
// CHECK-NEXT:         %c32_i256 = arith.constant 32 : i256 loc(#loc60)
// CHECK-NEXT:         %26 = arith.addi %25, %c32_i256 : i256 loc(#loc61)
// CHECK-NEXT:         llvm.store %26, %24 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc57)
// CHECK-NEXT:         %27 = llvm.load %21 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc62)
// CHECK-NEXT:         %28 = llvm.load %19 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc63)
// CHECK-NEXT:         %29 = llvm.inttoptr %27 : i256 to !llvm.ptr<1> loc(#loc64)
// CHECK-NEXT:         llvm.store %28, %29 {alignment = 1 : i64} : i256, !llvm.ptr<1> loc(#loc64)
// CHECK-NEXT:         %30 = llvm.load %21 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc65)
// CHECK-NEXT:         %31 = llvm.load %24 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc66)
// CHECK-NEXT:         %32 = llvm.load %21 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc67)
// CHECK-NEXT:         %33 = arith.subi %31, %32 : i256 loc(#loc68)
// CHECK-NEXT:         %c0_i256_11 = arith.constant 0 : i256 loc(#loc69)
// CHECK-NEXT:         func.call @__return(%30, %33, %c0_i256_11) : (i256, i256, i256) -> () loc(#loc69)
// CHECK-NEXT:         func.call @".unreachable"() : () -> () loc(#loc69)
// CHECK-NEXT:         scf.yield loc(#loc35)
// CHECK-NEXT:       }
// CHECK-NEXT:       default {
// CHECK-NEXT:         scf.yield loc(#loc35)
// CHECK-NEXT:       } loc(#loc35)
// CHECK-NEXT:     } loc(#loc27)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc70)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc71)
// CHECK-NEXT:     %c0_i256_2 = arith.constant 0 : i256 loc(#loc72)
// CHECK-NEXT:     call @__revert(%c0_i256_0, %c0_i256_1, %c0_i256_2) : (i256, i256, i256) -> () loc(#loc72)
// CHECK-NEXT:     call @".unreachable"() : () -> () loc(#loc72)
// CHECK-NEXT:     llvm.unreachable loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @ptr_decommit() {addr_space = 0 : i32} : !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @ptr_return_data() {addr_space = 0 : i32} : !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @ptr_calldata() {addr_space = 0 : i32} : !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @ptr_active() {addr_space = 0 : i32} : !llvm.array<16 x ptr<3>> loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @extra_abi_data(dense<0> : tensor<10xi256>) {addr_space = 0 : i32} : !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @call_flags(0 : i256) {addr_space = 0 : i32} : i256 loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @returndatasize(0 : i256) {addr_space = 0 : i32} : i256 loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @calldatasize(0 : i256) {addr_space = 0 : i32} : i256 loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @memory_pointer(0 : i256) {addr_space = 0 : i32} : i256 loc(#loc)
// CHECK-NEXT:   func.func private @__entry(%arg0: !llvm.ptr<3> loc(unknown), %arg1: i256 loc(unknown), %arg2: i256 loc(unknown), %arg3: i256 loc(unknown), %arg4: i256 loc(unknown), %arg5: i256 loc(unknown), %arg6: i256 loc(unknown), %arg7: i256 loc(unknown), %arg8: i256 loc(unknown), %arg9: i256 loc(unknown), %arg10: i256 loc(unknown), %arg11: i256 loc(unknown)) -> i256 attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = llvm.mlir.addressof @memory_pointer : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     llvm.store %c0_i256, %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %1 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     llvm.store %c0_i256_0, %1 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %2 = llvm.mlir.addressof @returndatasize : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     llvm.store %c0_i256_1, %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %3 = llvm.mlir.addressof @call_flags : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %c0_i256_2 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     llvm.store %c0_i256_2, %3 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %4 = llvm.mlir.addressof @extra_abi_data : !llvm.ptr<array<10 x i256>> loc(#loc)
// CHECK-NEXT:     %5 = llvm.mlir.constant(dense<0> : vector<10xi256>) : vector<10xi256> loc(#loc)
// CHECK-NEXT:     llvm.store %5, %4 : !llvm.ptr<array<10 x i256>> loc(#loc)
// CHECK-NEXT:     %6 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>> loc(#loc)
// CHECK-NEXT:     llvm.store %arg0, %6 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc)
// CHECK-NEXT:     %7 = llvm.ptrtoint %6 : !llvm.ptr<ptr<3>> to i256 loc(#loc)
// CHECK-NEXT:     %c96_i256 = arith.constant 96 : i256 loc(#loc)
// CHECK-NEXT:     %8 = llvm.lshr %7, %c96_i256  : i256 loc(#loc)
// CHECK-NEXT:     %c4294967295_i256 = arith.constant 4294967295 : i256 loc(#loc)
// CHECK-NEXT:     %9 = llvm.and %8, %c4294967295_i256  : i256 loc(#loc)
// CHECK-NEXT:     %10 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     llvm.store %9, %10 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %11 = llvm.load %10 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %12 = llvm.getelementptr %arg0[%11] : (!llvm.ptr<3>, i256) -> !llvm.ptr, i8 loc(#loc)
// CHECK-NEXT:     %13 = llvm.mlir.addressof @ptr_return_data : !llvm.ptr<ptr<3>> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %13 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc)
// CHECK-NEXT:     %14 = llvm.mlir.addressof @ptr_decommit : !llvm.ptr<ptr<3>> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %14 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc)
// CHECK-NEXT:     %15 = llvm.mlir.addressof @ptr_active : !llvm.ptr<array<16 x ptr<3>>> loc(#loc)
// CHECK-NEXT:     %c0_i256_3 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c0_i256_4 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %16 = llvm.getelementptr %15[%c0_i256_3, %c0_i256_4] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %16 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_5 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc)
// CHECK-NEXT:     %17 = llvm.getelementptr %15[%c0_i256_5, %c1_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %17 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_6 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c2_i256 = arith.constant 2 : i256 loc(#loc)
// CHECK-NEXT:     %18 = llvm.getelementptr %15[%c0_i256_6, %c2_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %18 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_7 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c3_i256 = arith.constant 3 : i256 loc(#loc)
// CHECK-NEXT:     %19 = llvm.getelementptr %15[%c0_i256_7, %c3_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %19 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_8 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c4_i256 = arith.constant 4 : i256 loc(#loc)
// CHECK-NEXT:     %20 = llvm.getelementptr %15[%c0_i256_8, %c4_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %20 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_9 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c5_i256 = arith.constant 5 : i256 loc(#loc)
// CHECK-NEXT:     %21 = llvm.getelementptr %15[%c0_i256_9, %c5_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %21 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_10 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c6_i256 = arith.constant 6 : i256 loc(#loc)
// CHECK-NEXT:     %22 = llvm.getelementptr %15[%c0_i256_10, %c6_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %22 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_11 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c7_i256 = arith.constant 7 : i256 loc(#loc)
// CHECK-NEXT:     %23 = llvm.getelementptr %15[%c0_i256_11, %c7_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %23 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_12 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c8_i256 = arith.constant 8 : i256 loc(#loc)
// CHECK-NEXT:     %24 = llvm.getelementptr %15[%c0_i256_12, %c8_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %24 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_13 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c9_i256 = arith.constant 9 : i256 loc(#loc)
// CHECK-NEXT:     %25 = llvm.getelementptr %15[%c0_i256_13, %c9_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %25 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_14 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c10_i256 = arith.constant 10 : i256 loc(#loc)
// CHECK-NEXT:     %26 = llvm.getelementptr %15[%c0_i256_14, %c10_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %26 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_15 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c11_i256 = arith.constant 11 : i256 loc(#loc)
// CHECK-NEXT:     %27 = llvm.getelementptr %15[%c0_i256_15, %c11_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %27 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_16 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c12_i256 = arith.constant 12 : i256 loc(#loc)
// CHECK-NEXT:     %28 = llvm.getelementptr %15[%c0_i256_16, %c12_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %28 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_17 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c13_i256 = arith.constant 13 : i256 loc(#loc)
// CHECK-NEXT:     %29 = llvm.getelementptr %15[%c0_i256_17, %c13_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %29 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_18 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c14_i256 = arith.constant 14 : i256 loc(#loc)
// CHECK-NEXT:     %30 = llvm.getelementptr %15[%c0_i256_18, %c14_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %30 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %c0_i256_19 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c15_i256 = arith.constant 15 : i256 loc(#loc)
// CHECK-NEXT:     %31 = llvm.getelementptr %15[%c0_i256_19, %c15_i256] : (!llvm.ptr<array<16 x ptr<3>>>, i256, i256) -> !llvm.ptr<3>, !llvm.array<16 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %31 {alignment = 32 : i64} : !llvm.ptr, !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:     %32 = llvm.mlir.addressof @call_flags : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg1, %32 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %33 = llvm.mlir.addressof @extra_abi_data : !llvm.ptr<array<10 x i256>> loc(#loc)
// CHECK-NEXT:     %c0_i256_20 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c0_i256_21 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %34 = llvm.getelementptr %33[%c0_i256_20, %c0_i256_21] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg2, %34 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_22 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c1_i256_23 = arith.constant 1 : i256 loc(#loc)
// CHECK-NEXT:     %35 = llvm.getelementptr %33[%c0_i256_22, %c1_i256_23] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg3, %35 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_24 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c2_i256_25 = arith.constant 2 : i256 loc(#loc)
// CHECK-NEXT:     %36 = llvm.getelementptr %33[%c0_i256_24, %c2_i256_25] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg4, %36 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_26 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c3_i256_27 = arith.constant 3 : i256 loc(#loc)
// CHECK-NEXT:     %37 = llvm.getelementptr %33[%c0_i256_26, %c3_i256_27] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg5, %37 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_28 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c4_i256_29 = arith.constant 4 : i256 loc(#loc)
// CHECK-NEXT:     %38 = llvm.getelementptr %33[%c0_i256_28, %c4_i256_29] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg6, %38 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_30 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c5_i256_31 = arith.constant 5 : i256 loc(#loc)
// CHECK-NEXT:     %39 = llvm.getelementptr %33[%c0_i256_30, %c5_i256_31] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg7, %39 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_32 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c6_i256_33 = arith.constant 6 : i256 loc(#loc)
// CHECK-NEXT:     %40 = llvm.getelementptr %33[%c0_i256_32, %c6_i256_33] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg8, %40 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_34 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c7_i256_35 = arith.constant 7 : i256 loc(#loc)
// CHECK-NEXT:     %41 = llvm.getelementptr %33[%c0_i256_34, %c7_i256_35] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg9, %41 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_36 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c8_i256_37 = arith.constant 8 : i256 loc(#loc)
// CHECK-NEXT:     %42 = llvm.getelementptr %33[%c0_i256_36, %c8_i256_37] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg10, %42 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_38 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c9_i256_39 = arith.constant 9 : i256 loc(#loc)
// CHECK-NEXT:     %43 = llvm.getelementptr %33[%c0_i256_38, %c9_i256_39] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg11, %43 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c1_i256_40 = arith.constant 1 : i256 loc(#loc)
// CHECK-NEXT:     %44 = arith.andi %arg1, %c1_i256_40 : i256 loc(#loc)
// CHECK-NEXT:     %c1_i256_41 = arith.constant 1 : i256 loc(#loc)
// CHECK-NEXT:     %45 = arith.cmpi eq, %44, %c1_i256_41 : i256 loc(#loc)
// CHECK-NEXT:     scf.if %45 {
// CHECK-NEXT:       func.call @__deploy() : () -> () loc(#loc)
// CHECK-NEXT:     } else {
// CHECK-NEXT:       func.call @__runtime() : () -> () loc(#loc)
// CHECK-NEXT:     } loc(#loc)
// CHECK-NEXT:     llvm.unreachable loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   func.func @f() -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc73)
// CHECK-NEXT:     %0 = llvm.alloca %c1_i256 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc74)
// CHECK-NEXT:     %c42_i256 = arith.constant 42 : i256 loc(#loc75)
// CHECK-NEXT:     llvm.store %c42_i256, %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc76)
// CHECK-NEXT:     %1 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc74)
// CHECK-NEXT:     return %1 : i256 loc(#loc73)
// CHECK-NEXT:   } loc(#loc73)
// CHECK-NEXT:   func.func private @__personality() -> i32 attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} loc(#loc)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc1 = loc({{.*}}:6:6)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:11)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:15)
// CHECK-NEXT: #loc4 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc5 = loc({{.*}}:5:7)
// CHECK-NEXT: #loc6 = loc({{.*}}:5:4)
// CHECK-NEXT: #loc7 = loc({{.*}}:6:13)
// CHECK-NEXT: #loc8 = loc({{.*}}:6:16)
// CHECK-NEXT: #loc9 = loc({{.*}}:8:4)
// CHECK-NEXT: #loc10 = loc({{.*}}:8:8)
// CHECK-NEXT: #loc11 = loc({{.*}}:8:25)
// CHECK-NEXT: #loc12 = loc({{.*}}:8:19)
// CHECK-NEXT: #loc13 = loc({{.*}}:9:13)
// CHECK-NEXT: #loc14 = loc({{.*}}:9:22)
// CHECK-NEXT: #loc15 = loc({{.*}}:9:51)
// CHECK-NEXT: #loc16 = loc({{.*}}:9:4)
// CHECK-NEXT: #loc17 = loc({{.*}}:10:11)
// CHECK-NEXT: #loc18 = loc({{.*}}:10:20)
// CHECK-NEXT: #loc19 = loc({{.*}}:10:4)
// CHECK-NEXT: #loc20 = loc({{.*}}:14:13)
// CHECK-NEXT: #loc21 = loc({{.*}}:14:17)
// CHECK-NEXT: #loc22 = loc({{.*}}:14:6)
// CHECK-NEXT: #loc23 = loc({{.*}}:15:19)
// CHECK-NEXT: #loc24 = loc({{.*}}:15:35)
// CHECK-NEXT: #loc25 = loc({{.*}}:15:16)
// CHECK-NEXT: #loc26 = loc({{.*}}:15:9)
// CHECK-NEXT: #loc27 = loc({{.*}}:15:6)
// CHECK-NEXT: #loc28 = loc({{.*}}:16:8)
// CHECK-NEXT: #loc29 = loc({{.*}}:16:12)
// CHECK-NEXT: #loc30 = loc({{.*}}:16:46)
// CHECK-NEXT: #loc31 = loc({{.*}}:16:33)
// CHECK-NEXT: #loc32 = loc({{.*}}:16:28)
// CHECK-NEXT: #loc33 = loc({{.*}}:16:24)
// CHECK-NEXT: #loc34 = loc({{.*}}:17:15)
// CHECK-NEXT: #loc35 = loc({{.*}}:17:8)
// CHECK-NEXT: #loc36 = loc({{.*}}:19:13)
// CHECK-NEXT: #loc37 = loc({{.*}}:19:10)
// CHECK-NEXT: #loc38 = loc({{.*}}:20:19)
// CHECK-NEXT: #loc39 = loc({{.*}}:20:22)
// CHECK-NEXT: #loc40 = loc({{.*}}:20:12)
// CHECK-NEXT: #loc41 = loc({{.*}}:22:21)
// CHECK-NEXT: #loc42 = loc({{.*}}:22:37)
// CHECK-NEXT: #loc43 = loc({{.*}}:22:17)
// CHECK-NEXT: #loc44 = loc({{.*}}:22:41)
// CHECK-NEXT: #loc45 = loc({{.*}}:22:13)
// CHECK-NEXT: #loc46 = loc({{.*}}:22:10)
// CHECK-NEXT: #loc47 = loc({{.*}}:24:19)
// CHECK-NEXT: #loc48 = loc({{.*}}:24:22)
// CHECK-NEXT: #loc49 = loc({{.*}}:24:12)
// CHECK-NEXT: #loc50 = loc({{.*}}:26:10)
// CHECK-NEXT: #loc51 = loc({{.*}}:26:14)
// CHECK-NEXT: #loc52 = loc({{.*}}:26:21)
// CHECK-NEXT: #loc53 = loc({{.*}}:27:10)
// CHECK-NEXT: #loc54 = loc({{.*}}:27:14)
// CHECK-NEXT: #loc55 = loc({{.*}}:27:30)
// CHECK-NEXT: #loc56 = loc({{.*}}:27:24)
// CHECK-NEXT: #loc57 = loc({{.*}}:28:10)
// CHECK-NEXT: #loc58 = loc({{.*}}:28:14)
// CHECK-NEXT: #loc59 = loc({{.*}}:28:28)
// CHECK-NEXT: #loc60 = loc({{.*}}:28:36)
// CHECK-NEXT: #loc61 = loc({{.*}}:28:24)
// CHECK-NEXT: #loc62 = loc({{.*}}:29:17)
// CHECK-NEXT: #loc63 = loc({{.*}}:29:25)
// CHECK-NEXT: #loc64 = loc({{.*}}:29:10)
// CHECK-NEXT: #loc65 = loc({{.*}}:30:17)
// CHECK-NEXT: #loc66 = loc({{.*}}:30:29)
// CHECK-NEXT: #loc67 = loc({{.*}}:30:37)
// CHECK-NEXT: #loc68 = loc({{.*}}:30:25)
// CHECK-NEXT: #loc69 = loc({{.*}}:30:10)
// CHECK-NEXT: #loc70 = loc({{.*}}:34:13)
// CHECK-NEXT: #loc71 = loc({{.*}}:34:16)
// CHECK-NEXT: #loc72 = loc({{.*}}:34:6)
// CHECK-NEXT: #loc73 = loc({{.*}}:36:6)
// CHECK-NEXT: #loc74 = loc({{.*}}:36:22)
// CHECK-NEXT: #loc75 = loc({{.*}}:37:13)
// CHECK-NEXT: #loc76 = loc({{.*}}:37:8)
// CHECK-EMPTY:
