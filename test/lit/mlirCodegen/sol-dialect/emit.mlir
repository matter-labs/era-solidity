// RUN: sol-opt %s | sol-opt | FileCheck %s

module {
  sol.func @f(%a: i256, %b: i256, %c: i256) -> () {
    sol.emit "E(uint256,uint256,uint256)" indexed = [%a, %b] non_indexed = [%c] : i256, i256, i256
    sol.return
  }
}

// CHECK: module {
// CHECK-NEXT:   sol.func @f(%arg0: i256, %arg1: i256, %arg2: i256) {
// CHECK-NEXT:     sol.emit "E(uint256,uint256,uint256)" indexed = [%arg0, %arg1] non_indexed = [%arg2] : i256, i256, i256
// CHECK-NEXT:     sol.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
