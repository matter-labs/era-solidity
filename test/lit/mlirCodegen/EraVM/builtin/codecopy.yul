// RUN: solc --yul --mlir-action=print-std-mlir --mlir-target=eravm --mmlir --mlir-print-debuginfo %s | FileCheck %s

object "Test" {
  code {
    let freePtr := mload(64)
    codecopy(freePtr, dataoffset("Test_deployed"), datasize("Test_deployed"))
    return(0, 0)
  }
  object "Test_deployed" {
    code {
      return(0, 0)
    }
  }
}
// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc0 = loc(unknown)
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func private @".unreachable"() attributes {llvm.linkage = #llvm.linkage<private>} {
// CHECK-NEXT:     llvm.unreachable loc(#loc1)
// CHECK-NEXT:   } loc(#loc0)
// CHECK-NEXT:   func.func private @__return(i256, i256, i256) attributes {llvm.linkage = #llvm.linkage<external>} loc(#loc0)
// CHECK-NEXT:   func.func private @__deploy() attributes {llvm.linkage = #llvm.linkage<private>} {
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc2)
// CHECK-NEXT:     %0 = llvm.alloca %c1_i256 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc3)
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256 loc(#loc4)
// CHECK-NEXT:     %1 = llvm.inttoptr %c64_i256 : i256 to !llvm.ptr<1> loc(#loc5)
// CHECK-NEXT:     %2 = llvm.load %1 {alignment = 1 : i64} : !llvm.ptr<1> -> i256 loc(#loc5)
// CHECK-NEXT:     llvm.store %2, %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc2)
// CHECK-NEXT:     %3 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc6)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc7)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc8)
// CHECK-NEXT:     %4 = llvm.inttoptr %3 : i256 to !llvm.ptr<1> loc(#loc9)
// CHECK-NEXT:     %5 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>> loc(#loc9)
// CHECK-NEXT:     %6 = llvm.load %5 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc9)
// CHECK-NEXT:     %7 = llvm.getelementptr %6[%c0_i256] : (!llvm.ptr<3>, i256) -> !llvm.ptr<3>, i8 loc(#loc9)
// CHECK-NEXT:     %8 = llvm.mlir.constant(false) : i1 loc(#loc9)
// CHECK-NEXT:     "llvm.intr.memcpy"(%4, %7, %c0_i256_0, %8) : (!llvm.ptr<1>, !llvm.ptr<3>, i256, i1) -> () loc(#loc9)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc10)
// CHECK-NEXT:     %c0_i256_2 = arith.constant 0 : i256 loc(#loc11)
// CHECK-NEXT:     %c256_i256 = arith.constant 256 : i256 loc(#loc1)
// CHECK-NEXT:     %9 = llvm.inttoptr %c256_i256 : i256 to !llvm.ptr<2> loc(#loc1)
// CHECK-NEXT:     %c32_i256 = arith.constant 32 : i256 loc(#loc1)
// CHECK-NEXT:     llvm.store %c32_i256, %9 {alignment = 1 : i64} : i256, !llvm.ptr<2> loc(#loc1)
// CHECK-NEXT:     %c288_i256 = arith.constant 288 : i256 loc(#loc1)
// CHECK-NEXT:     %10 = llvm.inttoptr %c288_i256 : i256 to !llvm.ptr<2> loc(#loc1)
// CHECK-NEXT:     %c0_i256_3 = arith.constant 0 : i256 loc(#loc1)
// CHECK-NEXT:     llvm.store %c0_i256_3, %10 {alignment = 1 : i64} : i256, !llvm.ptr<2> loc(#loc1)
// CHECK-NEXT:     %c0_i256_4 = arith.constant 0 : i256 loc(#loc1)
// CHECK-NEXT:     %c2_i256 = arith.constant 2 : i256 loc(#loc1)
// CHECK-NEXT:     %11 = arith.muli %c0_i256_4, %c2_i256 : i256 loc(#loc1)
// CHECK-NEXT:     %c64_i256_5 = arith.constant 64 : i256 loc(#loc1)
// CHECK-NEXT:     %12 = arith.addi %11, %c64_i256_5 : i256 loc(#loc1)
// CHECK-NEXT:     %c256_i256_6 = arith.constant 256 : i256 loc(#loc1)
// CHECK-NEXT:     %c2_i256_7 = arith.constant 2 : i256 loc(#loc1)
// CHECK-NEXT:     call @__return(%c256_i256_6, %12, %c2_i256_7) : (i256, i256, i256) -> () loc(#loc1)
// CHECK-NEXT:     call @".unreachable"() : () -> () loc(#loc1)
// CHECK-NEXT:     llvm.unreachable loc(#loc0)
// CHECK-NEXT:   } loc(#loc0)
// CHECK-NEXT:   func.func private @__runtime() attributes {llvm.linkage = #llvm.linkage<private>, runtime} {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc12)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc13)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc14)
// CHECK-NEXT:     call @__return(%c0_i256, %c0_i256_0, %c0_i256_1) : (i256, i256, i256) -> () loc(#loc14)
// CHECK-NEXT:     call @".unreachable"() : () -> () loc(#loc14)
// CHECK-NEXT:     llvm.unreachable loc(#loc0)
// CHECK-NEXT:   } loc(#loc0)
// CHECK-NEXT:   llvm.mlir.global private @ptr_active() : !llvm.ptr<3> loc(#loc0)
// CHECK-NEXT:   llvm.mlir.global private @ptr_return_data() : !llvm.ptr<3> loc(#loc0)
// CHECK-NEXT:   llvm.mlir.global private @ptr_calldata() : !llvm.ptr<3> loc(#loc0)
// CHECK-NEXT:   llvm.mlir.global private @extra_abi_data(dense<0> : tensor<10xi256>) {alignment = 32 : i64} : !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:   llvm.mlir.global private @call_flags(0 : i256) {alignment = 32 : i64} : i256 loc(#loc0)
// CHECK-NEXT:   llvm.mlir.global private @returndatasize(0 : i256) {alignment = 32 : i64} : i256 loc(#loc0)
// CHECK-NEXT:   llvm.mlir.global private @calldatasize(0 : i256) {alignment = 32 : i64} : i256 loc(#loc0)
// CHECK-NEXT:   llvm.mlir.global private @memory_pointer(0 : i256) {alignment = 32 : i64} : i256 loc(#loc0)
// CHECK-NEXT:   func.func private @__entry(%arg0: !llvm.ptr<3> loc(unknown), %arg1: i256 loc(unknown), %arg2: i256 loc(unknown), %arg3: i256 loc(unknown), %arg4: i256 loc(unknown), %arg5: i256 loc(unknown), %arg6: i256 loc(unknown), %arg7: i256 loc(unknown), %arg8: i256 loc(unknown), %arg9: i256 loc(unknown), %arg10: i256 loc(unknown), %arg11: i256 loc(unknown)) -> i256 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = llvm.mlir.addressof @memory_pointer : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     llvm.store %c0_i256, %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %1 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     llvm.store %c0_i256_0, %1 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %2 = llvm.mlir.addressof @returndatasize : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     llvm.store %c0_i256_1, %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %3 = llvm.mlir.addressof @call_flags : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %c0_i256_2 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     llvm.store %c0_i256_2, %3 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %4 = llvm.mlir.addressof @extra_abi_data : !llvm.ptr<array<10 x i256>> loc(#loc0)
// CHECK-NEXT:     %5 = llvm.mlir.constant(dense<0> : vector<10xi256>) : vector<10xi256> loc(#loc0)
// CHECK-NEXT:     llvm.store %5, %4 : !llvm.ptr<array<10 x i256>> loc(#loc0)
// CHECK-NEXT:     %6 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg0, %6 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc0)
// CHECK-NEXT:     %7 = llvm.ptrtoint %6 : !llvm.ptr<ptr<3>> to i256 loc(#loc0)
// CHECK-NEXT:     %c96_i256 = arith.constant 96 : i256 loc(#loc0)
// CHECK-NEXT:     %8 = llvm.lshr %7, %c96_i256  : i256 loc(#loc0)
// CHECK-NEXT:     %c4294967295_i256 = arith.constant 4294967295 : i256 loc(#loc0)
// CHECK-NEXT:     %9 = llvm.and %8, %c4294967295_i256  : i256 loc(#loc0)
// CHECK-NEXT:     %10 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %9, %10 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %11 = llvm.load %10 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %12 = llvm.getelementptr %arg0[%11] : (!llvm.ptr<3>, i256) -> !llvm.ptr, i8 loc(#loc0)
// CHECK-NEXT:     %13 = llvm.mlir.addressof @ptr_return_data : !llvm.ptr<ptr<3>> loc(#loc0)
// CHECK-NEXT:     llvm.store %12, %13 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc0)
// CHECK-NEXT:     %14 = llvm.mlir.addressof @ptr_active : !llvm.ptr<ptr<3>> loc(#loc0)
// CHECK-NEXT:     llvm.store %12, %14 {alignment = 32 : i64} : !llvm.ptr<ptr<3>> loc(#loc0)
// CHECK-NEXT:     %15 = llvm.mlir.addressof @call_flags : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg1, %15 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc0)
// CHECK-NEXT:     %16 = llvm.mlir.addressof @extra_abi_data : !llvm.ptr<array<10 x i256>> loc(#loc0)
// CHECK-NEXT:     %c0_i256_3 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c0_i256_4 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %17 = llvm.getelementptr %16[%c0_i256_3, %c0_i256_4] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg2, %17 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_5 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc0)
// CHECK-NEXT:     %18 = llvm.getelementptr %16[%c0_i256_5, %c1_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg3, %18 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_6 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c2_i256 = arith.constant 2 : i256 loc(#loc0)
// CHECK-NEXT:     %19 = llvm.getelementptr %16[%c0_i256_6, %c2_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg4, %19 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_7 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c3_i256 = arith.constant 3 : i256 loc(#loc0)
// CHECK-NEXT:     %20 = llvm.getelementptr %16[%c0_i256_7, %c3_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg5, %20 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_8 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c4_i256 = arith.constant 4 : i256 loc(#loc0)
// CHECK-NEXT:     %21 = llvm.getelementptr %16[%c0_i256_8, %c4_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg6, %21 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_9 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c5_i256 = arith.constant 5 : i256 loc(#loc0)
// CHECK-NEXT:     %22 = llvm.getelementptr %16[%c0_i256_9, %c5_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg7, %22 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_10 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c6_i256 = arith.constant 6 : i256 loc(#loc0)
// CHECK-NEXT:     %23 = llvm.getelementptr %16[%c0_i256_10, %c6_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg8, %23 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_11 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c7_i256 = arith.constant 7 : i256 loc(#loc0)
// CHECK-NEXT:     %24 = llvm.getelementptr %16[%c0_i256_11, %c7_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg9, %24 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_12 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c8_i256 = arith.constant 8 : i256 loc(#loc0)
// CHECK-NEXT:     %25 = llvm.getelementptr %16[%c0_i256_12, %c8_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg10, %25 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c0_i256_13 = arith.constant 0 : i256 loc(#loc0)
// CHECK-NEXT:     %c9_i256 = arith.constant 9 : i256 loc(#loc0)
// CHECK-NEXT:     %26 = llvm.getelementptr %16[%c0_i256_13, %c9_i256] : (!llvm.ptr<array<10 x i256>>, i256, i256) -> !llvm.ptr, !llvm.array<10 x i256> loc(#loc0)
// CHECK-NEXT:     llvm.store %arg11, %26 {alignment = 32 : i64} : i256, !llvm.ptr loc(#loc0)
// CHECK-NEXT:     %c1_i256_14 = arith.constant 1 : i256 loc(#loc0)
// CHECK-NEXT:     %27 = arith.andi %arg1, %c1_i256_14 : i256 loc(#loc0)
// CHECK-NEXT:     %c1_i256_15 = arith.constant 1 : i256 loc(#loc0)
// CHECK-NEXT:     %28 = arith.cmpi eq, %27, %c1_i256_15 : i256 loc(#loc0)
// CHECK-NEXT:     scf.if %28 {
// CHECK-NEXT:       func.call @__deploy() : () -> () loc(#loc0)
// CHECK-NEXT:     } else {
// CHECK-NEXT:       func.call @__runtime() : () -> () loc(#loc0)
// CHECK-NEXT:     } loc(#loc0)
// CHECK-NEXT:     llvm.unreachable loc(#loc0)
// CHECK-NEXT:   } loc(#loc0)
// CHECK-NEXT: } loc(#loc0)
// CHECK-NEXT: #loc1 = loc({{.*}}:6:4)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:8)
// CHECK-NEXT: #loc4 = loc({{.*}}:4:25)
// CHECK-NEXT: #loc5 = loc({{.*}}:4:19)
// CHECK-NEXT: #loc6 = loc({{.*}}:5:13)
// CHECK-NEXT: #loc7 = loc({{.*}}:5:22)
// CHECK-NEXT: #loc8 = loc({{.*}}:5:51)
// CHECK-NEXT: #loc9 = loc({{.*}}:5:4)
// CHECK-NEXT: #loc10 = loc({{.*}}:6:11)
// CHECK-NEXT: #loc11 = loc({{.*}}:6:14)
// CHECK-NEXT: #loc12 = loc({{.*}}:10:13)
// CHECK-NEXT: #loc13 = loc({{.*}}:10:16)
// CHECK-NEXT: #loc14 = loc({{.*}}:10:6)
// CHECK-EMPTY: