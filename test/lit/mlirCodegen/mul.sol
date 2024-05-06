// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
	function f(uint a) public returns (uint d) { return a * 7; }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc3 = loc({{.*}}:3:12)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_13 {
// CHECK-NEXT:     sol.func @f_12(%arg0: i256 loc({{.*}}:3:12)) -> i256 attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<i256, Stack> loc(#loc3)
// CHECK-NEXT:       sol.store %arg0 : i256, %0 : !sol.ptr<i256, Stack> loc(#loc3)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<i256, Stack>, i256 loc(#loc4)
// CHECK-NEXT:       %c7_i8 = arith.constant 7 : i8 loc(#loc5)
// CHECK-NEXT:       %2 = arith.extui %c7_i8 : i8 to i256 loc(#loc5)
// CHECK-NEXT:       %3 = arith.muli %1, %2 : i256 loc(#loc4)
// CHECK-NEXT:       sol.return %3 : i256 loc(#loc6)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:   } {interface_fns = [{selector = "b3de648b", sym = @f_12, type = (i256) -> i256}], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:1)
// CHECK-NEXT: #loc4 = loc({{.*}}:3:53)
// CHECK-NEXT: #loc5 = loc({{.*}}:3:57)
// CHECK-NEXT: #loc6 = loc({{.*}}:3:46)
// CHECK-EMPTY:
