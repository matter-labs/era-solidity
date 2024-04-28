// RUN: solc --yul --yul-dialect=evm --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

object "Test" {
  code {
    return(0, 0)
  }
  object "Test_deployed" {
    code {
      switch mload(0)
      case 0x0 {
        mstore(0, 0)
      }
      case 0x1 {
        mstore(0, 1)
      }
      default {}
      return(0, 0)
    }
  }
}
// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.object @Test {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc1)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc2)
// CHECK-NEXT:     sol.builtin_ret %c0_i256, %c0_i256_0 loc(#loc3)
// CHECK-NEXT:     sol.object @Test_deployed {
// CHECK-NEXT:       %c0_i256_1 = arith.constant 0 : i256 loc(#loc4)
// CHECK-NEXT:       %0 = sol.mload %c0_i256_1 loc(#loc5)
// CHECK-NEXT:       "scf.int_switch"(%0) <{cases = dense<[0, 1]> : tensor<2xi256>}> ({
// CHECK-NEXT:         scf.yield loc(#loc6)
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %c0_i256_4 = arith.constant 0 : i256 loc(#loc7)
// CHECK-NEXT:         %c0_i256_5 = arith.constant 0 : i256 loc(#loc8)
// CHECK-NEXT:         sol.mstore %c0_i256_4, %c0_i256_5 loc(#loc9)
// CHECK-NEXT:         scf.yield loc(#loc6)
// CHECK-NEXT:       }, {
// CHECK-NEXT:         %c0_i256_4 = arith.constant 0 : i256 loc(#loc10)
// CHECK-NEXT:         %c1_i256 = arith.constant 1 : i256 loc(#loc11)
// CHECK-NEXT:         sol.mstore %c0_i256_4, %c1_i256 loc(#loc12)
// CHECK-NEXT:         scf.yield loc(#loc6)
// CHECK-NEXT:       }) : (i256) -> () loc(#loc6)
// CHECK-NEXT:       %c0_i256_2 = arith.constant 0 : i256 loc(#loc13)
// CHECK-NEXT:       %c0_i256_3 = arith.constant 0 : i256 loc(#loc14)
// CHECK-NEXT:       sol.builtin_ret %c0_i256_2, %c0_i256_3 loc(#loc15)
// CHECK-NEXT:     } loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:11)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:14)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc4 = loc({{.*}}:8:19)
// CHECK-NEXT: #loc5 = loc({{.*}}:8:13)
// CHECK-NEXT: #loc6 = loc({{.*}}:8:6)
// CHECK-NEXT: #loc7 = loc({{.*}}:10:15)
// CHECK-NEXT: #loc8 = loc({{.*}}:10:18)
// CHECK-NEXT: #loc9 = loc({{.*}}:10:8)
// CHECK-NEXT: #loc10 = loc({{.*}}:13:15)
// CHECK-NEXT: #loc11 = loc({{.*}}:13:18)
// CHECK-NEXT: #loc12 = loc({{.*}}:13:8)
// CHECK-NEXT: #loc13 = loc({{.*}}:16:13)
// CHECK-NEXT: #loc14 = loc({{.*}}:16:16)
// CHECK-NEXT: #loc15 = loc({{.*}}:16:6)
// CHECK-EMPTY:
