// RUN: solc --mlir-action=print-init %s | FileCheck %s
// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck --check-prefix=DBG %s

contract C {
  function f0() public pure returns (uint) { return 300; }
  function f1() public pure returns (int) { return 300; }
  function f2(uint a) public returns (uint d) { return 7; }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.contract @C {
// CHECK-NEXT:     func.func @f0() -> i256 {
// CHECK-NEXT:       %c300_i16 = arith.constant 300 : i16
// CHECK-NEXT:       %0 = arith.extui %c300_i16 : i16 to i256
// CHECK-NEXT:       return %0 : i256
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func @f1() -> i256 {
// CHECK-NEXT:       %c300_i16 = arith.constant 300 : i16
// CHECK-NEXT:       %0 = arith.extsi %c300_i16 : i16 to i256
// CHECK-NEXT:       return %0 : i256
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func @f2(%arg0: i256) -> i256 {
// CHECK-NEXT:       %0 = memref.alloca() : memref<i256>
// CHECK-NEXT:       memref.store %arg0, %0[] : memref<i256>
// CHECK-NEXT:       %c7_i8 = arith.constant 7 : i8
// CHECK-NEXT:       %1 = arith.extui %c7_i8 : i8 to i256
// CHECK-NEXT:       return %1 : i256
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:
// DBG: #loc9 = loc({{.*}}:6:14)
// DBG-NEXT: module {
// DBG-NEXT:   sol.contract @C {
// DBG-NEXT:     func.func @f0() -> i256 {
// DBG-NEXT:       %c300_i16 = arith.constant 300 : i16 loc(#loc3)
// DBG-NEXT:       %0 = arith.extui %c300_i16 : i16 to i256 loc(#loc3)
// DBG-NEXT:       return %0 : i256 loc(#loc4)
// DBG-NEXT:     } loc(#loc2)
// DBG-NEXT:     func.func @f1() -> i256 {
// DBG-NEXT:       %c300_i16 = arith.constant 300 : i16 loc(#loc6)
// DBG-NEXT:       %0 = arith.extsi %c300_i16 : i16 to i256 loc(#loc6)
// DBG-NEXT:       return %0 : i256 loc(#loc7)
// DBG-NEXT:     } loc(#loc5)
// DBG-NEXT:     func.func @f2(%arg0: i256 loc({{.*}}:6:14)) -> i256 {
// DBG-NEXT:       %0 = memref.alloca() : memref<i256> loc(#loc9)
// DBG-NEXT:       memref.store %arg0, %0[] : memref<i256> loc(#loc9)
// DBG-NEXT:       %c7_i8 = arith.constant 7 : i8 loc(#loc10)
// DBG-NEXT:       %1 = arith.extui %c7_i8 : i8 to i256 loc(#loc10)
// DBG-NEXT:       return %1 : i256 loc(#loc11)
// DBG-NEXT:     } loc(#loc8)
// DBG-NEXT:   } loc(#loc1)
// DBG-NEXT: } loc(#loc0)
// DBG-NEXT: #loc0 = loc(unknown)
// DBG-NEXT: #loc1 = loc({{.*}}:3:0)
// DBG-NEXT: #loc2 = loc({{.*}}:4:2)
// DBG-NEXT: #loc3 = loc({{.*}}:4:52)
// DBG-NEXT: #loc4 = loc({{.*}}:4:45)
// DBG-NEXT: #loc5 = loc({{.*}}:5:2)
// DBG-NEXT: #loc6 = loc({{.*}}:5:51)
// DBG-NEXT: #loc7 = loc({{.*}}:5:44)
// DBG-NEXT: #loc8 = loc({{.*}}:6:2)
// DBG-NEXT: #loc10 = loc({{.*}}:6:55)
// DBG-NEXT: #loc11 = loc({{.*}}:6:48)
// DBG-EMPTY: