// RUN: solc --mlir %s | FileCheck %s
// RUN: solc --mlir --mmlir --mlir-print-debuginfo %s | FileCheck --check-prefix=DBG %s

contract C {
  function f() public pure {}
}

// Note: Assertions generated by test/updFileCheckTest.py:
// CHECK: module {
// CHECK-NEXT:   solidity.contract @C {
// CHECK-NEXT:     func.func @f() {
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// DBG: module {
// DBG-NEXT:   solidity.contract @C {
// DBG-NEXT:     func.func @f() {
// DBG-NEXT:       return loc(#loc2)
// DBG-NEXT:     } loc(#loc2)
// DBG-NEXT:   } loc(#loc1)
// DBG-NEXT: } loc(#loc0)
// DBG-NEXT: #loc0 = loc(unknown)
// DBG-NEXT: #loc1 = loc({{.*}}:3:0)
// DBG-NEXT: #loc2 = loc({{.*}}:4:2)
