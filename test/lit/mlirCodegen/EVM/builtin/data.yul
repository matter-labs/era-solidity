// RUN: solc --strict-assembly --mlir-action=print-std-mlir --mlir-target=evm --mmlir --mlir-print-debuginfo %s | FileCheck %s

object "Test" {
  code {
    codecopy(codesize(), dataoffset("Test_deployed"), datasize("Test_deployed"))
  }
  object "Test_deployed" {
    code {}
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   func.func private @__entry() attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"]} {
// CHECK-NEXT:     %0 = "llvm.intrcall"() <{id = 3217 : i32, name = "evm.codesize"}> : () -> i256 loc(#loc1)
// CHECK-NEXT:     %1 = "llvm.intrcall"() <{id = 3221 : i32, metadata = ["Test_deployed"], name = "evm.dataoffset"}> : () -> i256 loc(#loc2)
// CHECK-NEXT:     %2 = "llvm.intrcall"() <{id = 3222 : i32, metadata = ["Test_deployed"], name = "evm.datasize"}> : () -> i256 loc(#loc3)
// CHECK-NEXT:     %3 = llvm.inttoptr %0 : i256 to !llvm.ptr<1> loc(#loc4)
// CHECK-NEXT:     %4 = llvm.inttoptr %1 : i256 to !llvm.ptr<4> loc(#loc4)
// CHECK-NEXT:     "llvm.intr.memcpy"(%3, %4, %2) <{isVolatile = false}> : (!llvm.ptr<1>, !llvm.ptr<4>, i256) -> () loc(#loc4)
// CHECK-NEXT:     llvm.unreachable loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT:   module @Test_deployed {
// CHECK-NEXT:     func.func private @__entry() attributes {llvm.linkage = #llvm.linkage<external>, passthrough = ["nofree", "null_pointer_is_valid"]} {
// CHECK-NEXT:       llvm.unreachable loc(#loc)
// CHECK-NEXT:     } loc(#loc)
// CHECK-NEXT:   } loc(#loc)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:13)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:25)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:54)
// CHECK-NEXT: #loc4 = loc({{.*}}:4:4)
// CHECK-EMPTY:
