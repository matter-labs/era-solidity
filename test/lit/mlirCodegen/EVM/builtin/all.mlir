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

  sol.func @callvalue() -> i256 {
    %ret = sol.callvalue
    sol.return %ret : i256
  }

  sol.func @calldata(%dst: i256, %src: i256) -> i256 {
    %size = sol.calldatasize
    sol.calldatacopy %dst, %src, %size
    %ret = sol.calldataload %src
    sol.return %ret : i256
  }

  sol.func @storage(%addr: i256, %val: i256) -> i256 {
    sol.sstore %addr, %val
    %ret = sol.sload %addr
    sol.return %ret : i256
  }

  sol.func @dataoffset() -> i256 {
    %ret = sol.dataoffset {obj = @Test_deployed}
    sol.return %ret : i256
  }

  sol.func @datasize() -> i256 {
    %ret = sol.datasize {obj = @Test_deployed}
    sol.return %ret : i256
  }

  sol.func @codesize() -> i256 {
    %ret = sol.codesize
    sol.return %ret : i256
  }

  sol.func @codecopy(%dst: i256, %src: i256, %size: i256) {
    sol.codecopy %dst, %src, %size
    sol.return
  }

  sol.func @memory(%addr: i256, %val: i256) -> i256 {
    sol.mstore %addr, %val
    %ret = sol.mload %addr
    sol.return %ret : i256
  }

  sol.func @memoryguard() -> i256 {
    %ret = sol.memoryguard 42
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
// CHECK-NEXT:     "llvm.intrcall"(%4, %arg1, %arg2, %arg3, %arg4, %arg5) <{id = 3238 : i32, name = "evm.log4"}> : (!llvm.ptr<1>, i256, i256, i256, i256, i256) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @caller() -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = "llvm.intrcall"() <{id = 3214 : i32, name = "evm.caller"}> : () -> i256
// CHECK-NEXT:     return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @callvalue() -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = "llvm.intrcall"() <{id = 3215 : i32, name = "evm.callvalue"}> : () -> i256
// CHECK-NEXT:     return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @calldata(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = "llvm.intrcall"() <{id = 3213 : i32, name = "evm.calldatasize"}> : () -> i256
// CHECK-NEXT:     %1 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %2 = llvm.inttoptr %arg1 : i256 to !llvm.ptr<2>
// CHECK-NEXT:     "llvm.intr.memcpy"(%1, %2, %0) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<2>, i256) -> ()
// CHECK-NEXT:     %3 = llvm.inttoptr %arg1 : i256 to !llvm.ptr<2>
// CHECK-NEXT:     %4 = llvm.load %3 {alignment = 1 : i64} : !llvm.ptr<2> -> i256
// CHECK-NEXT:     return %4 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @storage(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<5>
// CHECK-NEXT:     llvm.store %arg1, %0 {alignment = 1 : i64} : i256, !llvm.ptr<5>
// CHECK-NEXT:     %1 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<5>
// CHECK-NEXT:     %2 = llvm.load %1 {alignment = 1 : i64} : !llvm.ptr<5> -> i256
// CHECK-NEXT:     return %2 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @dataoffset() -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = "llvm.intrcall"() <{id = 3221 : i32, metadata = ["Test_deployed"], name = "evm.dataoffset"}> : () -> i256
// CHECK-NEXT:     return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @datasize() -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = "llvm.intrcall"() <{id = 3222 : i32, metadata = ["Test_deployed"], name = "evm.datasize"}> : () -> i256
// CHECK-NEXT:     return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @codesize() -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = "llvm.intrcall"() <{id = 3217 : i32, name = "evm.codesize"}> : () -> i256
// CHECK-NEXT:     return %0 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @codecopy(%arg0: i256, %arg1: i256, %arg2: i256) attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %1 = llvm.inttoptr %arg1 : i256 to !llvm.ptr<4>
// CHECK-NEXT:     "llvm.intr.memcpy"(%0, %1, %arg2) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<4>, i256) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @memory(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %0 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %arg1, %0 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %1 = llvm.inttoptr %arg0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %2 = llvm.load %1 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     return %2 : i256
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @memoryguard() -> i256 attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"], personality = @__personality} {
// CHECK-NEXT:     %c42_i256 = arith.constant 42 : i256
// CHECK-NEXT:     return %c42_i256 : i256
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-EMPTY:
