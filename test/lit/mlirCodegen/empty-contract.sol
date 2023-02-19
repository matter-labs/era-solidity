// RUN: solc --mlir %s |& FileCheck %s
// CHECK: module {
// CHECK-NEXT: solidity.contract @C {
contract C {
}
// CHECK: solidity.contract @D {
contract D {
}
