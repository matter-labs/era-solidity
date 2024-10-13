// RUN: sol-opt -convert-sol-to-std=target=evm %s | FileCheck %s

// FIXME: Replace this test with yul tests (once we support sol.object lowering)

module {
  sol.func @keccak256(%addr: i256, %size: i256) -> i256 {
    %ret = sol.keccak256 %addr, %size
    sol.return %ret : i256
  }

  sol.func @log(%addr: i256, %size: i256, %t0: i256, %t1: i256, %t2: i256, %t3: i256) {
    sol.log %addr, %size
    sol.log %addr, %size topics(%t0)
    sol.log %addr, %size topics(%t0, %t1)
    sol.log %addr, %size topics(%t0, %t1, %t2)
    sol.log %addr, %size topics(%t0, %t1, %t2, %t3)
    sol.return
  }

  sol.func @caller() -> i256 {
    %ret = sol.caller
    sol.return %ret : i256
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   func.func private @__personality() -> i32 attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality}
// CHECK-NEXT:   func.func @keccak256(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %1 = "llvm.intrcall"(%0, %arg1) <{id = 3258 : i32, name = "evm.sha3"}> : (!llvm.ptr<1>, i256) -> i256
// CHECK-NEXT:     return %1 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @log(%arg0: i256, %arg1: i256, %arg2: i256, %arg3: i256, %arg4: i256, %arg5: i256) attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     "llvm.intrcall"(%0, %arg1) <{id = 3234 : i32, name = "evm.log0"}> : (!llvm.ptr<1>, i256) -> ()
// CHECK-NEXT:     %1 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     "llvm.intrcall"(%1, %arg1, %arg2) <{id = 3235 : i32, name = "evm.log1"}> : (!llvm.ptr<1>, i256, i256) -> ()
// CHECK-NEXT:     %2 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     "llvm.intrcall"(%2, %arg1, %arg2, %arg3) <{id = 3236 : i32, name = "evm.log2"}> : (!llvm.ptr<1>, i256, i256, i256) -> ()
// CHECK-NEXT:     %3 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     "llvm.intrcall"(%3, %arg1, %arg2, %arg3, %arg4) <{id = 3237 : i32, name = "evm.log3"}> : (!llvm.ptr<1>, i256, i256, i256, i256) -> ()
// CHECK-NEXT:     %4 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     "llvm.intrcall"(%4, %arg1, %arg2, %arg3, %arg4, %arg5) <{id = 3237 : i32, name = "evm.log4"}> : (!llvm.ptr<1>, i256, i256, i256, i256, i256) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @caller() -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = "llvm.intrcall"() <{id = 3214 : i32, name = "evm.caller"}> : () -> i256
// CHECK-NEXT:     return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-EMPTY:
