// RUN: sol-opt %s | sol-opt | FileCheck %s

module {
  sol.func @ld_scalar(%a: !sol.ptr<i256>) -> i256 {
    %ld = sol.load %a : !sol.ptr<i256>, i256
    sol.return %ld : i256
  }

  sol.func @ld_array(%a: !sol.array<2 x i256, Memory>) -> i256 {
    %zero = arith.constant 0 : i256
    %ld = sol.load %a[%zero : i256] : !sol.array<2 x i256, Memory>, i256
    sol.return %ld : i256
  }

  sol.func @st_array(%a: !sol.array<2 x i256, Memory>, %b: i256) {
    %zero = arith.constant 0 : i256
    sol.store %b : i256, %a[%zero : i256] : !sol.array<2 x i256, Memory>
    sol.return
  }
}

// CHECK: module {
// CHECK-NEXT:   sol.func @ld_scalar(%arg0: !sol.ptr<i256>) -> i256 {
// CHECK-NEXT:     %0 = sol.load %arg0 : !sol.ptr<i256>, i256
// CHECK-NEXT:     sol.return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   sol.func @ld_array(%arg0: !sol.array<2 x i256, Memory>) -> i256 {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256
// CHECK-NEXT:     %0 = sol.load %arg0[%c0_i256 : i256] : !sol.array<2 x i256, Memory>, i256
// CHECK-NEXT:     sol.return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   sol.func @st_array(%arg0: !sol.array<2 x i256, Memory>, %arg1: i256) {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256
// CHECK-NEXT:     sol.store %arg1 : i256, %arg0[%c0_i256 : i256] : !sol.array<2 x i256, Memory>
// CHECK-NEXT:     sol.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
