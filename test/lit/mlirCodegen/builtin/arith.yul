// RUN: solc --yul --yul-dialect=evm --mlir-action=print-init --mmlir --mlir-print-debuginfo --mlir-target=eravm %s | FileCheck %s

object "Test" {
  code {
    let a := mload(32)
    let b := mload(64)
    mstore(0, add(a, b))
    mstore(0, sub(a, b))
    mstore(0, shr(a, b))
    return(0, 0)
  }
  object "Test_deployed" {
    code {
      let a := mload(32)
      let b := mload(64)
      mstore(0, add(a, b))
      mstore(0, sub(a, b))
      mstore(0, shr(a, b))
      return(0, 0)
    }
  }
}
// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.object @Test {
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc1)
// CHECK-NEXT:     %0 = llvm.alloca %c1_i256 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc2)
// CHECK-NEXT:     %c32_i256 = arith.constant 32 : i256 loc(#loc3)
// CHECK-NEXT:     %1 = sol.mload %c32_i256 loc(#loc4)
// CHECK-NEXT:     llvm.store %1, %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc1)
// CHECK-NEXT:     %c1_i256_0 = arith.constant 1 : i256 loc(#loc5)
// CHECK-NEXT:     %2 = llvm.alloca %c1_i256_0 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc6)
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256 loc(#loc7)
// CHECK-NEXT:     %3 = sol.mload %c64_i256 loc(#loc8)
// CHECK-NEXT:     llvm.store %3, %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc5)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc9)
// CHECK-NEXT:     %4 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc10)
// CHECK-NEXT:     %5 = llvm.load %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc11)
// CHECK-NEXT:     %6 = arith.addi %4, %5 : i256 loc(#loc12)
// CHECK-NEXT:     sol.mstore %c0_i256, %6 loc(#loc13)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc14)
// CHECK-NEXT:     %7 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc15)
// CHECK-NEXT:     %8 = llvm.load %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc16)
// CHECK-NEXT:     %9 = arith.subi %7, %8 : i256 loc(#loc17)
// CHECK-NEXT:     sol.mstore %c0_i256_1, %9 loc(#loc18)
// CHECK-NEXT:     %c0_i256_2 = arith.constant 0 : i256 loc(#loc19)
// CHECK-NEXT:     %10 = llvm.load %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc20)
// CHECK-NEXT:     %11 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc21)
// CHECK-NEXT:     %12 = arith.shrui %10, %11 : i256 loc(#loc22)
// CHECK-NEXT:     sol.mstore %c0_i256_2, %12 loc(#loc23)
// CHECK-NEXT:     %c0_i256_3 = arith.constant 0 : i256 loc(#loc24)
// CHECK-NEXT:     %c0_i256_4 = arith.constant 0 : i256 loc(#loc25)
// CHECK-NEXT:     sol.return %c0_i256_3, %c0_i256_4 loc(#loc26)
// CHECK-NEXT:     sol.object @Test_deployed {
// CHECK-NEXT:       %c1_i256_5 = arith.constant 1 : i256 loc(#loc27)
// CHECK-NEXT:       %13 = llvm.alloca %c1_i256_5 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc28)
// CHECK-NEXT:       %c32_i256_6 = arith.constant 32 : i256 loc(#loc29)
// CHECK-NEXT:       %14 = sol.mload %c32_i256_6 loc(#loc30)
// CHECK-NEXT:       llvm.store %14, %13 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc27)
// CHECK-NEXT:       %c1_i256_7 = arith.constant 1 : i256 loc(#loc31)
// CHECK-NEXT:       %15 = llvm.alloca %c1_i256_7 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc32)
// CHECK-NEXT:       %c64_i256_8 = arith.constant 64 : i256 loc(#loc33)
// CHECK-NEXT:       %16 = sol.mload %c64_i256_8 loc(#loc34)
// CHECK-NEXT:       llvm.store %16, %15 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc31)
// CHECK-NEXT:       %c0_i256_9 = arith.constant 0 : i256 loc(#loc35)
// CHECK-NEXT:       %17 = llvm.load %13 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc36)
// CHECK-NEXT:       %18 = llvm.load %15 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc37)
// CHECK-NEXT:       %19 = arith.addi %17, %18 : i256 loc(#loc38)
// CHECK-NEXT:       sol.mstore %c0_i256_9, %19 loc(#loc39)
// CHECK-NEXT:       %c0_i256_10 = arith.constant 0 : i256 loc(#loc40)
// CHECK-NEXT:       %20 = llvm.load %13 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc41)
// CHECK-NEXT:       %21 = llvm.load %15 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc42)
// CHECK-NEXT:       %22 = arith.subi %20, %21 : i256 loc(#loc43)
// CHECK-NEXT:       sol.mstore %c0_i256_10, %22 loc(#loc44)
// CHECK-NEXT:       %c0_i256_11 = arith.constant 0 : i256 loc(#loc45)
// CHECK-NEXT:       %23 = llvm.load %15 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc46)
// CHECK-NEXT:       %24 = llvm.load %13 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc47)
// CHECK-NEXT:       %25 = arith.shrui %23, %24 : i256 loc(#loc48)
// CHECK-NEXT:       sol.mstore %c0_i256_11, %25 loc(#loc49)
// CHECK-NEXT:       %c0_i256_12 = arith.constant 0 : i256 loc(#loc50)
// CHECK-NEXT:       %c0_i256_13 = arith.constant 0 : i256 loc(#loc51)
// CHECK-NEXT:       sol.return %c0_i256_12, %c0_i256_13 loc(#loc52)
// CHECK-NEXT:     } loc(#loc0)
// CHECK-NEXT:   } loc(#loc0)
// CHECK-NEXT: } loc(#loc0)
// CHECK-NEXT: #loc0 = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:8)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:19)
// CHECK-NEXT: #loc4 = loc({{.*}}:4:13)
// CHECK-NEXT: #loc5 = loc({{.*}}:5:4)
// CHECK-NEXT: #loc6 = loc({{.*}}:5:8)
// CHECK-NEXT: #loc7 = loc({{.*}}:5:19)
// CHECK-NEXT: #loc8 = loc({{.*}}:5:13)
// CHECK-NEXT: #loc9 = loc({{.*}}:6:11)
// CHECK-NEXT: #loc10 = loc({{.*}}:6:18)
// CHECK-NEXT: #loc11 = loc({{.*}}:6:21)
// CHECK-NEXT: #loc12 = loc({{.*}}:6:14)
// CHECK-NEXT: #loc13 = loc({{.*}}:6:4)
// CHECK-NEXT: #loc14 = loc({{.*}}:7:11)
// CHECK-NEXT: #loc15 = loc({{.*}}:7:18)
// CHECK-NEXT: #loc16 = loc({{.*}}:7:21)
// CHECK-NEXT: #loc17 = loc({{.*}}:7:14)
// CHECK-NEXT: #loc18 = loc({{.*}}:7:4)
// CHECK-NEXT: #loc19 = loc({{.*}}:8:11)
// CHECK-NEXT: #loc20 = loc({{.*}}:8:21)
// CHECK-NEXT: #loc21 = loc({{.*}}:8:18)
// CHECK-NEXT: #loc22 = loc({{.*}}:8:14)
// CHECK-NEXT: #loc23 = loc({{.*}}:8:4)
// CHECK-NEXT: #loc24 = loc({{.*}}:9:11)
// CHECK-NEXT: #loc25 = loc({{.*}}:9:14)
// CHECK-NEXT: #loc26 = loc({{.*}}:9:4)
// CHECK-NEXT: #loc27 = loc({{.*}}:13:6)
// CHECK-NEXT: #loc28 = loc({{.*}}:13:10)
// CHECK-NEXT: #loc29 = loc({{.*}}:13:21)
// CHECK-NEXT: #loc30 = loc({{.*}}:13:15)
// CHECK-NEXT: #loc31 = loc({{.*}}:14:6)
// CHECK-NEXT: #loc32 = loc({{.*}}:14:10)
// CHECK-NEXT: #loc33 = loc({{.*}}:14:21)
// CHECK-NEXT: #loc34 = loc({{.*}}:14:15)
// CHECK-NEXT: #loc35 = loc({{.*}}:15:13)
// CHECK-NEXT: #loc36 = loc({{.*}}:15:20)
// CHECK-NEXT: #loc37 = loc({{.*}}:15:23)
// CHECK-NEXT: #loc38 = loc({{.*}}:15:16)
// CHECK-NEXT: #loc39 = loc({{.*}}:15:6)
// CHECK-NEXT: #loc40 = loc({{.*}}:16:13)
// CHECK-NEXT: #loc41 = loc({{.*}}:16:20)
// CHECK-NEXT: #loc42 = loc({{.*}}:16:23)
// CHECK-NEXT: #loc43 = loc({{.*}}:16:16)
// CHECK-NEXT: #loc44 = loc({{.*}}:16:6)
// CHECK-NEXT: #loc45 = loc({{.*}}:17:13)
// CHECK-NEXT: #loc46 = loc({{.*}}:17:23)
// CHECK-NEXT: #loc47 = loc({{.*}}:17:20)
// CHECK-NEXT: #loc48 = loc({{.*}}:17:16)
// CHECK-NEXT: #loc49 = loc({{.*}}:17:6)
// CHECK-NEXT: #loc50 = loc({{.*}}:18:13)
// CHECK-NEXT: #loc51 = loc({{.*}}:18:16)
// CHECK-NEXT: #loc52 = loc({{.*}}:18:6)
// CHECK-EMPTY: