// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
}
contract D {
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.contract @C_1 {
// CHECK-NEXT:   } {interface_fns = [], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT:   sol.contract @D_2 {
// CHECK-NEXT:   } {interface_fns = [], kind = #sol<ContractKind Contract>} loc(#loc2)
// CHECK-NEXT: } loc(#loc0)
// CHECK-NEXT: #loc0 = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:0)
// CHECK-EMPTY: