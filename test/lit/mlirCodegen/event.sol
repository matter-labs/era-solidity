// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  event E(address indexed a, address indexed b, uint256 c);
  function f(address a, address b, uint256 c) public {
    emit E(a, b, c);
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc3 = loc({{.*}}:4:13)
// CHECK-NEXT: #loc4 = loc({{.*}}:4:24)
// CHECK-NEXT: #loc5 = loc({{.*}}:4:35)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_25 {
// CHECK-NEXT:     sol.func @f_24(%arg0: i256 loc({{.*}}:4:13), %arg1: i256 loc({{.*}}:4:24), %arg2: i256 loc({{.*}}:4:35)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<i256, Stack> loc(#loc3)
// CHECK-NEXT:       sol.store %arg0, %0 : i256, !sol.ptr<i256, Stack> loc(#loc3)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<i256, Stack> loc(#loc4)
// CHECK-NEXT:       sol.store %arg1, %1 : i256, !sol.ptr<i256, Stack> loc(#loc4)
// CHECK-NEXT:       %2 = sol.alloca : !sol.ptr<i256, Stack> loc(#loc5)
// CHECK-NEXT:       sol.store %arg2, %2 : i256, !sol.ptr<i256, Stack> loc(#loc5)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<i256, Stack>, i256 loc(#loc6)
// CHECK-NEXT:       %4 = sol.load %1 : !sol.ptr<i256, Stack>, i256 loc(#loc7)
// CHECK-NEXT:       %5 = sol.load %2 : !sol.ptr<i256, Stack>, i256 loc(#loc8)
// CHECK-NEXT:       "sol.emit"(%3, %4, %5) {indexedArgsCount = 2 : i8, signature = "E(address,address,uint256)"} : (i256, i256, i256) -> () loc(#loc9)
// CHECK-NEXT:       sol.return loc(#loc2)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:   } {interface_fns = [{selector = "9548dc8c", sym = @f_24, type = (i256, i256, i256) -> ()}], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:2)
// CHECK-NEXT: #loc6 = loc({{.*}}:5:11)
// CHECK-NEXT: #loc7 = loc({{.*}}:5:14)
// CHECK-NEXT: #loc8 = loc({{.*}}:5:17)
// CHECK-NEXT: #loc9 = loc({{.*}}:5:9)
// CHECK-EMPTY:
