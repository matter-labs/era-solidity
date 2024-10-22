// RUN: solc --strict-assembly --mlir-action=print-std-mlir --mlir-target=evm --mmlir --mlir-print-debuginfo %s | FileCheck %s

object "Test" {
  code {
    revert(0, 1)
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   func.func private @".unreachable"() attributes {llvm.linkage = #llvm.linkage<private>, passthrough = ["nofree", "null_pointer_is_valid"]} {
// CHECK-NEXT:     llvm.unreachable loc(#loc1)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   func.func private @__entry() attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"]} {
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc2)
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc3)
// CHECK-NEXT:     %0 = llvm.inttoptr %c0_i256 : i256 to !llvm.ptr<1> loc(#loc1)
// CHECK-NEXT:     "llvm.intrcall"(%0, %c1_i256) <{id = 3253 : i32, name = "evm.revert"}> : (!llvm.ptr<1>, i256) -> () loc(#loc1)
// CHECK-NEXT:     call @".unreachable"() : () -> () loc(#loc1)
// CHECK-NEXT:     llvm.unreachable loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:11)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:14)
// CHECK-EMPTY: