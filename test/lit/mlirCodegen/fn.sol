// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  function i1(bool a) public returns (bool) { return a; }
  function ui64(uint64 a) public returns (uint64) { return a; }
  function ui256(uint256 a) public returns (uint256) { return ui256_internal(a); }
  function ui256_internal(uint256 a) internal returns (uint256) { return a; }

  function str(string memory a) public returns (string memory) {
    return str_internal(a);
  }
  function str_internal(string memory a) internal returns (string memory) {
    return a;
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #Contract = #sol<ContractKind Contract>
// CHECK-NEXT: #NonPayable = #sol<StateMutability NonPayable>
// CHECK-NEXT: #loc3 = loc({{.*}}:3:14)
// CHECK-NEXT: #loc7 = loc({{.*}}:4:16)
// CHECK-NEXT: #loc11 = loc({{.*}}:5:17)
// CHECK-NEXT: #loc16 = loc({{.*}}:6:26)
// CHECK-NEXT: #loc20 = loc({{.*}}:8:15)
// CHECK-NEXT: #loc25 = loc({{.*}}:11:24)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_65 {
// CHECK-NEXT:     sol.func @i1_10(%arg0: i1 loc({{.*}}:3:14)) -> i1 attributes {state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<i1, Stack> loc(#loc3)
// CHECK-NEXT:       sol.store %arg0, %0 : i1, !sol.ptr<i1, Stack> loc(#loc3)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<i1, Stack>, i1 loc(#loc4)
// CHECK-NEXT:       sol.return %1 : i1 loc(#loc5)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:     sol.func @ui64_20(%arg0: ui64 loc({{.*}}:4:16)) -> ui64 attributes {state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui64, Stack> loc(#loc7)
// CHECK-NEXT:       sol.store %arg0, %0 : ui64, !sol.ptr<ui64, Stack> loc(#loc7)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<ui64, Stack>, ui64 loc(#loc8)
// CHECK-NEXT:       sol.return %1 : ui64 loc(#loc9)
// CHECK-NEXT:     } loc(#loc6)
// CHECK-NEXT:     sol.func @ui256_32(%arg0: ui256 loc({{.*}}:5:17)) -> ui256 attributes {state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc11)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc11)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc12)
// CHECK-NEXT:       %2 = sol.call @ui256_internal_42(%1) : (ui256) -> ui256 loc(#loc13)
// CHECK-NEXT:       sol.return %2 : ui256 loc(#loc14)
// CHECK-NEXT:     } loc(#loc10)
// CHECK-NEXT:     sol.func @ui256_internal_42(%arg0: ui256 loc({{.*}}:6:26)) -> ui256 attributes {state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc16)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc16)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc17)
// CHECK-NEXT:       sol.return %1 : ui256 loc(#loc18)
// CHECK-NEXT:     } loc(#loc15)
// CHECK-NEXT:     sol.func @str_54(%arg0: !sol.string<Memory> loc({{.*}}:8:15)) -> !sol.string<Memory> attributes {state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc20)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc20)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc21)
// CHECK-NEXT:       %2 = sol.call @str_internal_64(%1) : (!sol.string<Memory>) -> !sol.string<Memory> loc(#loc22)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc23)
// CHECK-NEXT:     } loc(#loc19)
// CHECK-NEXT:     sol.func @str_internal_64(%arg0: !sol.string<Memory> loc({{.*}}:11:24)) -> !sol.string<Memory> attributes {state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc25)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc25)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc26)
// CHECK-NEXT:       sol.return %1 : !sol.string<Memory> loc(#loc27)
// CHECK-NEXT:     } loc(#loc24)
// CHECK-NEXT:   } {interface_fns = [{selector = 167553359 : i32, sym = @str_54, type = (!sol.string<Memory>) -> !sol.string<Memory>}, {selector = 1090974155 : i32, sym = @ui64_20, type = (ui64) -> ui64}, {selector = -1867629089 : i32, sym = @ui256_32, type = (ui256) -> ui256}, {selector = -948490518 : i32, sym = @i1_10, type = (i1) -> i1}], kind = #Contract} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:3:53)
// CHECK-NEXT: #loc5 = loc({{.*}}:3:46)
// CHECK-NEXT: #loc6 = loc({{.*}}:4:2)
// CHECK-NEXT: #loc8 = loc({{.*}}:4:59)
// CHECK-NEXT: #loc9 = loc({{.*}}:4:52)
// CHECK-NEXT: #loc10 = loc({{.*}}:5:2)
// CHECK-NEXT: #loc12 = loc({{.*}}:5:77)
// CHECK-NEXT: #loc13 = loc({{.*}}:5:62)
// CHECK-NEXT: #loc14 = loc({{.*}}:5:55)
// CHECK-NEXT: #loc15 = loc({{.*}}:6:2)
// CHECK-NEXT: #loc17 = loc({{.*}}:6:73)
// CHECK-NEXT: #loc18 = loc({{.*}}:6:66)
// CHECK-NEXT: #loc19 = loc({{.*}}:8:2)
// CHECK-NEXT: #loc21 = loc({{.*}}:9:24)
// CHECK-NEXT: #loc22 = loc({{.*}}:9:11)
// CHECK-NEXT: #loc23 = loc({{.*}}:9:4)
// CHECK-NEXT: #loc24 = loc({{.*}}:11:2)
// CHECK-NEXT: #loc26 = loc({{.*}}:12:11)
// CHECK-NEXT: #loc27 = loc({{.*}}:12:4)
// CHECK-EMPTY:
