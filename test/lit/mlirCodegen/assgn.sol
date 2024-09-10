// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  uint ui;
  mapping(address => mapping(address => uint256)) m1;

  function f_ui(uint a) private { ui = a; }
  function f_m1(address a, address b, uint256 c) private { m1[a][b] = c; }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #Contract = #sol<ContractKind Contract>
// CHECK-NEXT: #NonPayable = #sol<StateMutability NonPayable>
// CHECK-NEXT: #loc5 = loc({{.*}}:6:16)
// CHECK-NEXT: #loc9 = loc({{.*}}:7:16)
// CHECK-NEXT: #loc10 = loc({{.*}}:7:27)
// CHECK-NEXT: #loc11 = loc({{.*}}:7:38)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_37 {
// CHECK-NEXT:     sol.state_var @ui : ui256 loc(#loc2)
// CHECK-NEXT:     sol.state_var @m1 : !sol.mapping<ui256, !sol.mapping<ui256, ui256>> loc(#loc3)
// CHECK-NEXT:     sol.func @f_ui_18(%arg0: ui256 loc({{.*}}:6:16)) attributes {state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc5)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc5)
// CHECK-NEXT:       %1 = sol.addr_of @ui : !sol.ptr<ui256, Storage> loc(#loc2)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc6)
// CHECK-NEXT:       sol.store %2, %1 : ui256, !sol.ptr<ui256, Storage> loc(#loc7)
// CHECK-NEXT:       sol.return loc(#loc4)
// CHECK-NEXT:     } loc(#loc4)
// CHECK-NEXT:     sol.func @f_m1_36(%arg0: ui256 loc({{.*}}:7:16), %arg1: ui256 loc({{.*}}:7:27), %arg2: ui256 loc({{.*}}:7:38)) attributes {state_mutability = #NonPayable} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc9)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc9)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc10)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc10)
// CHECK-NEXT:       %2 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc11)
// CHECK-NEXT:       sol.store %arg2, %2 : ui256, !sol.ptr<ui256, Stack> loc(#loc11)
// CHECK-NEXT:       %3 = sol.addr_of @m1 : !sol.mapping<ui256, !sol.mapping<ui256, ui256>> loc(#loc3)
// CHECK-NEXT:       %4 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc12)
// CHECK-NEXT:       %5 = sol.map %3, %4 : !sol.mapping<ui256, !sol.mapping<ui256, ui256>>, ui256, !sol.mapping<ui256, ui256> loc(#loc13)
// CHECK-NEXT:       %6 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc14)
// CHECK-NEXT:       %7 = sol.map %5, %6 : !sol.mapping<ui256, ui256>, ui256, !sol.ptr<ui256, Storage> loc(#loc13)
// CHECK-NEXT:       %8 = sol.load %2 : !sol.ptr<ui256, Stack>, ui256 loc(#loc15)
// CHECK-NEXT:       sol.store %8, %7 : ui256, !sol.ptr<ui256, Storage> loc(#loc13)
// CHECK-NEXT:       sol.return loc(#loc8)
// CHECK-NEXT:     } loc(#loc8)
// CHECK-NEXT:   } {interface_fns = [], kind = #Contract} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:6:2)
// CHECK-NEXT: #loc6 = loc({{.*}}:6:39)
// CHECK-NEXT: #loc7 = loc({{.*}}:6:34)
// CHECK-NEXT: #loc8 = loc({{.*}}:7:2)
// CHECK-NEXT: #loc12 = loc({{.*}}:7:62)
// CHECK-NEXT: #loc13 = loc({{.*}}:7:59)
// CHECK-NEXT: #loc14 = loc({{.*}}:7:65)
// CHECK-NEXT: #loc15 = loc({{.*}}:7:70)
// CHECK-EMPTY:
