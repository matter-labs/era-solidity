// RUN: solc --yul --mlir-action=print-init --mmlir --mlir-print-debuginfo --mlir-target=eravm %s | FileCheck %s

object "Test" {
  code {
    return(0, mload(64))
  }
  object "Test_deployed" {
    code {
      return(0, mload(64))
    }
  }
}
// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.object @Test {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc1)
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256 loc(#loc2)
// CHECK-NEXT:     %0 = sol.mload %c64_i256 loc(#loc3)
// CHECK-NEXT:     sol.builtin_ret %c0_i256, %0 loc(#loc4)
// CHECK-NEXT:     sol.object @Test_deployed {
// CHECK-NEXT:       %c0_i256_0 = arith.constant 0 : i256 loc(#loc5)
// CHECK-NEXT:       %c64_i256_1 = arith.constant 64 : i256 loc(#loc6)
// CHECK-NEXT:       %1 = sol.mload %c64_i256_1 loc(#loc7)
// CHECK-NEXT:       sol.builtin_ret %c0_i256_0, %1 loc(#loc8)
// CHECK-NEXT:     } loc(#loc0)
// CHECK-NEXT:   } loc(#loc0)
// CHECK-NEXT: } loc(#loc0)
// CHECK-NEXT: #loc0 = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:11)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:20)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:14)
// CHECK-NEXT: #loc4 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc5 = loc({{.*}}:8:13)
// CHECK-NEXT: #loc6 = loc({{.*}}:8:22)
// CHECK-NEXT: #loc7 = loc({{.*}}:8:16)
// CHECK-NEXT: #loc8 = loc({{.*}}:8:6)
// CHECK-EMPTY: