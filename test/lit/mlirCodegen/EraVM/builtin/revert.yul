// RUN: solc --yul --mlir-action=print-std-mlir --mlir-target=eravm --mmlir --mlir-print-debuginfo %s | FileCheck %s

object "Test" {
  code {
    revert(0, 0)
  }
  object "Test_deployed" {
    code {
      revert(0, 0)
    }
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc = loc(unknown)
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func private @".unreachable"() attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     llvm.unreachable loc(#loc1)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   func.func private @__revert(i256, i256, i256) attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} loc(#loc)
// CHECK-NEXT:   func.func private @__deploy() attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc2)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc3)
// CHECK-NEXT:     %c2_i256 = arith.constant 2 : i256 loc(#loc1)
// CHECK-NEXT:     call @__revert(%c0_i256, %c0_i256_0, %c2_i256) : (i256, i256, i256) -> () loc(#loc1)
// CHECK-NEXT:     call @".unreachable"() : () -> () loc(#loc1)
// CHECK-NEXT:     llvm.unreachable loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   func.func private @__runtime() attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality, runtime} {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc4)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc5)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc6)
// CHECK-NEXT:     call @__revert(%c0_i256, %c0_i256_0, %c0_i256_1) : (i256, i256, i256) -> () loc(#loc6)
// CHECK-NEXT:     call @".unreachable"() : () -> () loc(#loc6)
// CHECK-NEXT:     llvm.unreachable loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @ptr_active() {addr_space = 0 : i32} : !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @ptr_return_data() {addr_space = 0 : i32} : !llvm.ptr<3> loc(#loc)
// CHECK-NEXT:   llvm.mlir.global private @ptr_calldata() {addr_space = 0 : i32} : !llvm.ptr<3> loc(#loc)
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
// CHECK-NEXT:     %14 = llvm.mlir.addressof @ptr_active : !llvm.ptr<ptr<3>> loc(#loc)
// CHECK-NEXT:     llvm.store %12, %14 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc)
// CHECK-NEXT:     %15 = llvm.mlir.addressof @call_flags : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg1, %15 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc)
// CHECK-NEXT:     %16 = llvm.mlir.addressof @extra_abi_data : !llvm.ptr<array<10 x i256>> loc(#loc)
// CHECK-NEXT:     %c0_i256_3 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c0_i256_4 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %17 = llvm.getelementptr %16[%c0_i256_3, %c0_i256_4] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg2, %17 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_5 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc)
// CHECK-NEXT:     %18 = llvm.getelementptr %16[%c0_i256_5, %c1_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg3, %18 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_6 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c2_i256 = arith.constant 2 : i256 loc(#loc)
// CHECK-NEXT:     %19 = llvm.getelementptr %16[%c0_i256_6, %c2_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg4, %19 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_7 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c3_i256 = arith.constant 3 : i256 loc(#loc)
// CHECK-NEXT:     %20 = llvm.getelementptr %16[%c0_i256_7, %c3_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg5, %20 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_8 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c4_i256 = arith.constant 4 : i256 loc(#loc)
// CHECK-NEXT:     %21 = llvm.getelementptr %16[%c0_i256_8, %c4_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg6, %21 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_9 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c5_i256 = arith.constant 5 : i256 loc(#loc)
// CHECK-NEXT:     %22 = llvm.getelementptr %16[%c0_i256_9, %c5_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg7, %22 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_10 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c6_i256 = arith.constant 6 : i256 loc(#loc)
// CHECK-NEXT:     %23 = llvm.getelementptr %16[%c0_i256_10, %c6_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg8, %23 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_11 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c7_i256 = arith.constant 7 : i256 loc(#loc)
// CHECK-NEXT:     %24 = llvm.getelementptr %16[%c0_i256_11, %c7_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg9, %24 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_12 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c8_i256 = arith.constant 8 : i256 loc(#loc)
// CHECK-NEXT:     %25 = llvm.getelementptr %16[%c0_i256_12, %c8_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg10, %25 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c0_i256_13 = arith.constant 0 : i256 loc(#loc)
// CHECK-NEXT:     %c9_i256 = arith.constant 9 : i256 loc(#loc)
// CHECK-NEXT:     %26 = llvm.getelementptr %16[%c0_i256_13, %c9_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc)
// CHECK-NEXT:     llvm.store %arg11, %26 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc)
// CHECK-NEXT:     %c1_i256_14 = arith.constant 1 : i256 loc(#loc)
// CHECK-NEXT:     %27 = arith.andi %arg1, %c1_i256_14 : i256 loc(#loc)
// CHECK-NEXT:     %c1_i256_15 = arith.constant 1 : i256 loc(#loc)
// CHECK-NEXT:     %28 = arith.cmpi eq, %27, %c1_i256_15 : i256 loc(#loc)
// CHECK-NEXT:     scf.if %28 {
// CHECK-NEXT:       func.call @__deploy() : () -> () loc(#loc)
// CHECK-NEXT:     } else {
// CHECK-NEXT:       func.call @__runtime() : () -> () loc(#loc)
// CHECK-NEXT:     } loc(#loc)
// CHECK-NEXT:     llvm.unreachable loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   func.func private @__personality() -> i32 attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} loc(#loc)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:11)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:14)
// CHECK-NEXT: #loc4 = loc({{.*}}:8:13)
// CHECK-NEXT: #loc5 = loc({{.*}}:8:16)
// CHECK-NEXT: #loc6 = loc({{.*}}:8:6)
// CHECK-EMPTY:
