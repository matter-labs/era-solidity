// RUN: solc --strict-assembly --mlir-action=print-init --mmlir --mlir-print-debuginfo --mlir-target=eravm %s | FileCheck %s

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
// CHECK: module {
// CHECK-NEXT:   sol.object @Test {
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc1)
// CHECK-NEXT:     %0 = llvm.alloca %c1_i256 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc2)
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256 loc(#loc3)
// CHECK-NEXT:     %1 = sol.mload %c64_i256 loc(#loc4)
// CHECK-NEXT:     llvm.store %1, %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc1)
// CHECK-NEXT:     %2 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc5)
// CHECK-NEXT:     %3 = sol.dataoffset {obj = @Test_deployed} loc(#loc6)
// CHECK-NEXT:     %4 = sol.datasize {obj = @Test_deployed} loc(#loc7)
// CHECK-NEXT:     sol.codecopy %2, %3, %4 loc(#loc8)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc9)
// CHECK-NEXT:     %c0_i256_0 = arith.constant 0 : i256 loc(#loc10)
// CHECK-NEXT:     sol.builtin_ret %c0_i256, %c0_i256_0 loc(#loc11)
// CHECK-NEXT:     sol.object @Test_deployed {
// CHECK-NEXT:       %c0_i256_1 = arith.constant 0 : i256 loc(#loc12)
// CHECK-NEXT:       %c0_i256_2 = arith.constant 0 : i256 loc(#loc13)
// CHECK-NEXT:       sol.builtin_ret %c0_i256_1, %c0_i256_2 loc(#loc14)
// CHECK-NEXT:     } loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:0:49)
// CHECK-NEXT: #loc2 = loc({{.*}}:0:53)
// CHECK-NEXT: #loc3 = loc({{.*}}:0:70)
// CHECK-NEXT: #loc4 = loc({{.*}}:0:64)
// CHECK-NEXT: #loc5 = loc({{.*}}:0:95)
// CHECK-NEXT: #loc6 = loc({{.*}}:0:104)
// CHECK-NEXT: #loc7 = loc({{.*}}:2:7)
// CHECK-NEXT: #loc8 = loc({{.*}}:0:86)
// CHECK-NEXT: #loc9 = loc({{.*}}:4:28)
// CHECK-NEXT: #loc10 = loc({{.*}}:5:2)
// CHECK-NEXT: #loc11 = loc({{.*}}:4:21)
// CHECK-NEXT: #loc12 = loc({{.*}}:5:74)
// CHECK-NEXT: #loc13 = loc({{.*}}:5:77)
// CHECK-NEXT: #loc14 = loc({{.*}}:5:67)
// CHECK-EMPTY:
