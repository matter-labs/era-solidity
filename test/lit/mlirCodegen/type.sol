// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  uint256 ui256;
  address addr;
  mapping(address => uint256) simpleMapping;
  mapping(address => mapping(address => uint256)) nestedMapping;
  string str;
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.contract @C_17 {
// CHECK-NEXT:     sol.state_var @ui256 : ui256 loc(#loc2)
// CHECK-NEXT:     sol.state_var @addr : ui256 loc(#loc3)
// CHECK-NEXT:     sol.state_var @simpleMapping : !sol.mapping<ui256, ui256> loc(#loc4)
// CHECK-NEXT:     sol.state_var @nestedMapping : !sol.mapping<ui256, !sol.mapping<ui256, ui256>> loc(#loc5)
// CHECK-NEXT:     sol.state_var @str : !sol.string<Storage> loc(#loc6)
// CHECK-NEXT:   } {interface_fns = [], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:5:2)
// CHECK-NEXT: #loc5 = loc({{.*}}:6:2)
// CHECK-NEXT: #loc6 = loc({{.*}}:7:2)
// CHECK-EMPTY:
