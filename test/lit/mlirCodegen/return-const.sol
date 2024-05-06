// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  function f0() public pure returns (uint) { return 300; }
  function f1() public pure returns (int) { return 300; }
  function f2(uint a) public returns (uint d) { return 7; }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc9 = loc({{.*}}:5:14)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_27 {
// CHECK-NEXT:     sol.func @f0_8() -> i256 attributes {state_mutability = #sol<StateMutability Pure>} {
// CHECK-NEXT:       %c300_i16 = arith.constant 300 : i16 loc(#loc3)
// CHECK-NEXT:       %0 = arith.extui %c300_i16 : i16 to i256 loc(#loc3)
// CHECK-NEXT:       sol.return %0 : i256 loc(#loc4)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:     sol.func @f1_16() -> i256 attributes {state_mutability = #sol<StateMutability Pure>} {
// CHECK-NEXT:       %c300_i16 = arith.constant 300 : i16 loc(#loc6)
// CHECK-NEXT:       %0 = arith.extsi %c300_i16 : i16 to i256 loc(#loc6)
// CHECK-NEXT:       sol.return %0 : i256 loc(#loc7)
// CHECK-NEXT:     } loc(#loc5)
// CHECK-NEXT:     sol.func @f2_26(%arg0: i256 loc({{.*}}:5:14)) -> i256 attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<i256, Stack> loc(#loc9)
// CHECK-NEXT:       sol.store %arg0 : i256, %0 : !sol.ptr<i256, Stack> loc(#loc9)
// CHECK-NEXT:       %c7_i8 = arith.constant 7 : i8 loc(#loc10)
// CHECK-NEXT:       %1 = arith.extui %c7_i8 : i8 to i256 loc(#loc10)
// CHECK-NEXT:       sol.return %1 : i256 loc(#loc11)
// CHECK-NEXT:     } loc(#loc8)
// CHECK-NEXT:   } {interface_fns = [{selector = "a5850475", sym = @f0_8, type = () -> i256}, {selector = "bf3724af", sym = @f2_26, type = (i256) -> i256}, {selector = "c27fc305", sym = @f1_16, type = () -> i256}], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc3 = loc({{.*}}:3:52)
// CHECK-NEXT: #loc4 = loc({{.*}}:3:45)
// CHECK-NEXT: #loc5 = loc({{.*}}:4:2)
// CHECK-NEXT: #loc6 = loc({{.*}}:4:51)
// CHECK-NEXT: #loc7 = loc({{.*}}:4:44)
// CHECK-NEXT: #loc8 = loc({{.*}}:5:2)
// CHECK-NEXT: #loc10 = loc({{.*}}:5:55)
// CHECK-NEXT: #loc11 = loc({{.*}}:5:48)
// CHECK-EMPTY:
