// RUN: sol-opt %s | sol-opt | FileCheck %s

module {
  sol.func @array(%a: !sol.array<? x i256, Memory>, %b: !sol.array<3 x i256, CallData>) -> () {
    sol.return
  }
}

// CHECK: module {
// CHECK-NEXT:   sol.func @array(%arg0: !sol.array<? x i256, Memory>, %arg1: !sol.array<3 x i256, CallData>) {
// CHECK-NEXT:     sol.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
