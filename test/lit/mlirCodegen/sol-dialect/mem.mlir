// RUN: sol-opt %s | sol-opt | FileCheck %s

module {
  sol.func @ld_scalar(%a: !sol.ptr<i256, Stack>) -> i256 {
    %ld = sol.load %a : !sol.ptr<i256, Stack>, i256
    sol.return %ld : i256
  }

  sol.func @ld_array(%a: !sol.array<2 x i256, Memory>) -> i256 {
    %zero = arith.constant 0 : i256
    %ld = sol.load %a[%zero] : !sol.array<2 x i256, Memory>, i256
    sol.return %ld : i256
  }

  sol.func @st_array(%a: !sol.array<2 x i256, Memory>, %b: i256) {
    %zero = arith.constant 0 : i256
    sol.store %b : i256, %a[%zero] : !sol.array<2 x i256, Memory>
    sol.return
  }

  sol.func @alloca() -> !sol.ptr<i256, Stack> {
    %stk = sol.alloca : !sol.ptr<i256, Stack>
    sol.return %stk : !sol.ptr<i256, Stack>
  }

  sol.func @malloc() -> !sol.array<2 x i256, Memory> {
    %mem = sol.malloc : !sol.array<2 x i256, Memory>
    sol.return %mem : !sol.array<2 x i256, Memory>
  }

  sol.func @dyn_malloc() -> !sol.array<? x i256, Memory> {
    %ten = arith.constant 10 : i256
    %mem = sol.malloc %ten : !sol.array<? x i256, Memory>
    sol.return %mem : !sol.array<? x i256, Memory>
  }
}

// CHECK: module {
// CHECK-NEXT:   sol.func @ld_scalar(%arg0: !sol.ptr<i256, Stack>) -> i256 {
// CHECK-NEXT:     %0 = sol.load %arg0 : !sol.ptr<i256, Stack>, i256
// CHECK-NEXT:     sol.return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   sol.func @ld_array(%arg0: !sol.array<2 x i256, Memory>) -> i256 {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256
// CHECK-NEXT:     %0 = sol.load %arg0[%c0_i256] : !sol.array<2 x i256, Memory>, i256
// CHECK-NEXT:     sol.return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   sol.func @st_array(%arg0: !sol.array<2 x i256, Memory>, %arg1: i256) {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256
// CHECK-NEXT:     sol.store %arg1 : i256, %arg0[%c0_i256] : !sol.array<2 x i256, Memory>
// CHECK-NEXT:     sol.return
// CHECK-NEXT:   }
// CHECK-NEXT:   sol.func @alloca() -> !sol.ptr<i256, Stack> {
// CHECK-NEXT:     %0 = sol.alloca : !sol.ptr<i256, Stack>
// CHECK-NEXT:     sol.return %0 : !sol.ptr<i256, Stack>
// CHECK-NEXT:   }
// CHECK-NEXT:   sol.func @malloc() -> !sol.array<2 x i256, Memory> {
// CHECK-NEXT:     %0 = sol.malloc : !sol.array<2 x i256, Memory>
// CHECK-NEXT:     sol.return %0 : !sol.array<2 x i256, Memory>
// CHECK-NEXT:   }
// CHECK-NEXT:   sol.func @dyn_malloc() -> !sol.array<? x i256, Memory> {
// CHECK-NEXT:     %c10_i256 = arith.constant 10 : i256
// CHECK-NEXT:     %0 = sol.malloc %c10_i256 : !sol.array<? x i256, Memory>
// CHECK-NEXT:     sol.return %0 : !sol.array<? x i256, Memory>
// CHECK-NEXT:   }
// CHECK-NEXT: }
