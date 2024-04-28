// RUN: solc --yul --mlir-action=print-init --mmlir --mlir-print-debuginfo --mlir-target=eravm %s | FileCheck %s

object "Test" {
  code {
    mstore(64, dataoffset("Test_deployed"))
    return(0, 0)
  }
  object "Test_deployed" {
    code {
      return(0, 0)
    }
  }
}
// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.object @Test {
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256 loc(#loc1)
// CHECK-NEXT:     %0 = sol.dataoffset {sym = @Test_deployed} loc(#loc2)
// CHECK-NEXT:     sol.mstore %c64_i256, %0 loc(#loc3)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc4)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc5)
// CHECK-NEXT:     sol.builtin_ret %c0_i256, %c0_i256_0 loc(#loc6)
// CHECK-NEXT:     sol.object @Test_deployed {
// CHECK-NEXT:       %c0_i256_1 = arith.constant 0 : i256 loc(#loc7)
// CHECK-NEXT:       %c0_i256_2 = arith.constant 0 : i256 loc(#loc8)
// CHECK-NEXT:       sol.builtin_ret %c0_i256_1, %c0_i256_2 loc(#loc9)
// CHECK-NEXT:     } loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:11)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:15)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc4 = loc({{.*}}:5:11)
// CHECK-NEXT: #loc5 = loc({{.*}}:5:14)
// CHECK-NEXT: #loc6 = loc({{.*}}:5:4)
// CHECK-NEXT: #loc7 = loc({{.*}}:9:13)
// CHECK-NEXT: #loc8 = loc({{.*}}:9:16)
// CHECK-NEXT: #loc9 = loc({{.*}}:9:6)
// CHECK-EMPTY:
