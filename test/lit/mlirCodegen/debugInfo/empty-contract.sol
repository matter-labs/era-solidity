// RUN: solc --mlir --mmlir --mlir-print-debuginfo %s | FileCheck %s
// CHECK: module {
// CHECK-NEXT: solidity.contract @C {
// CHECK-NEXT: } loc(#loc1)
contract C {
}
// CHECK: solidity.contract @D {
// CHECK-NEXT: } loc(#loc2)
contract D {
}

// CHECK: #loc1 = loc("{{.*}}empty-contract.sol":4:0)
// CHECK: #loc2 = loc("{{.*}}empty-contract.sol":8:0)
