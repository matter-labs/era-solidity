// RUN: solc --yul --yul-dialect=evm --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

object "Test" {
  code {
    if callvalue() {
      mstore(0, 42)
    }
    return(0, 0)
  }
  object "Test_deployed" {
    code {
      if callvalue() {
        mstore(1, 42)
      }
      return(0, 0)
    }
  }
}
// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.object @Test {
// CHECK-NEXT:     %0 = sol.callvalue : i256 loc(#loc1)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc1)
// CHECK-NEXT:     %1 = arith.cmpi ne, %0, %c0_i256 : i256 loc(#loc1)
// CHECK-NEXT:     scf.if %1 {
// CHECK-NEXT:       %c0_i256_2 = arith.constant 0 : i256 loc(#loc3)
// CHECK-NEXT:       %c42_i256 = arith.constant 42 : i256 loc(#loc4)
// CHECK-NEXT:       sol.mstore %c0_i256_2, %c42_i256 loc(#loc5)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc6)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc7)
// CHECK-NEXT:     sol.builtin_ret %c0_i256_0, %c0_i256_1 loc(#loc8)
// CHECK-NEXT:     sol.object @Test_deployed {
// CHECK-NEXT:       %2 = sol.callvalue : i256 loc(#loc9)
// CHECK-NEXT:       %c0_i256_2 = arith.constant 0 : i256 loc(#loc9)
// CHECK-NEXT:       %3 = arith.cmpi ne, %2, %c0_i256_2 : i256 loc(#loc9)
// CHECK-NEXT:       scf.if %3 {
// CHECK-NEXT:         %c1_i256 = arith.constant 1 : i256 loc(#loc11)
// CHECK-NEXT:         %c42_i256 = arith.constant 42 : i256 loc(#loc12)
// CHECK-NEXT:         sol.mstore %c1_i256, %c42_i256 loc(#loc13)
// CHECK-NEXT:       } loc(#loc10)
// CHECK-NEXT:       %c0_i256_3 = arith.constant 0 : i256 loc(#loc14)
// CHECK-NEXT:       %c0_i256_4 = arith.constant 0 : i256 loc(#loc15)
// CHECK-NEXT:       sol.builtin_ret %c0_i256_3, %c0_i256_4 loc(#loc16)
// CHECK-NEXT:     } loc(#loc0)
// CHECK-NEXT:   } loc(#loc0)
// CHECK-NEXT: } loc(#loc0)
// CHECK-NEXT: #loc0 = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:7)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc3 = loc({{.*}}:5:13)
// CHECK-NEXT: #loc4 = loc({{.*}}:5:16)
// CHECK-NEXT: #loc5 = loc({{.*}}:5:6)
// CHECK-NEXT: #loc6 = loc({{.*}}:7:11)
// CHECK-NEXT: #loc7 = loc({{.*}}:7:14)
// CHECK-NEXT: #loc8 = loc({{.*}}:7:4)
// CHECK-NEXT: #loc9 = loc({{.*}}:11:9)
// CHECK-NEXT: #loc10 = loc({{.*}}:11:6)
// CHECK-NEXT: #loc11 = loc({{.*}}:12:15)
// CHECK-NEXT: #loc12 = loc({{.*}}:12:18)
// CHECK-NEXT: #loc13 = loc({{.*}}:12:8)
// CHECK-NEXT: #loc14 = loc({{.*}}:14:13)
// CHECK-NEXT: #loc15 = loc({{.*}}:14:16)
// CHECK-NEXT: #loc16 = loc({{.*}}:14:6)
// CHECK-EMPTY: