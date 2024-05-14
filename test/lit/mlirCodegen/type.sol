// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.0;

contract C {
  uint256 ui256;
  address addr;
  mapping(address => uint256) simpleMapping;
  mapping(address => mapping(address => uint256)) nestedMapping;
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.contract @C_16 {
// CHECK-NEXT:     sol.state_var @ui256 : i256 loc(#loc2)
// CHECK-NEXT:     sol.state_var @addr : i256 loc(#loc3)
// CHECK-NEXT:     sol.state_var @simpleMapping : !sol.mapping<i256, i256> loc(#loc4)
// CHECK-NEXT:     sol.state_var @nestedMapping : !sol.mapping<i256, !sol.mapping<i256, i256>> loc(#loc5)
// CHECK-NEXT:   } {interface_fns = [], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:5:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:6:2)
// CHECK-NEXT: #loc3 = loc({{.*}}:7:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:8:2)
// CHECK-NEXT: #loc5 = loc({{.*}}:9:2)
// CHECK-EMPTY: