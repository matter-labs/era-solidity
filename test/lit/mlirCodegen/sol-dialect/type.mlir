// RUN: sol-opt %s | sol-opt | FileCheck %s

module {
  sol.func @array(%a: !sol.array<? x i256, Memory>, %b: !sol.array<3 x i256, CallData>) -> () {
    sol.return
  }

  sol.func @ptr(%a: !sol.ptr<Memory>) -> () {
    sol.return
  }
}

// CHECK: module {
// CHECK-NEXT:   sol.func @array(%arg0: !sol.array<? x i256, Memory>, %arg1: !sol.array<3 x i256, CallData>) {
// CHECK-NEXT:     sol.return
// CHECK-NEXT:   }
// CHECK-NEXT:   sol.func @ptr(%arg0: !sol.ptr<Memory>) {
// CHECK-NEXT:     sol.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
