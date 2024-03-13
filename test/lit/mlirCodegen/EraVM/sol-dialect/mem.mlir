// RUN: sol-opt -convert-sol-to-std=target=eravm %s | FileCheck %s

module {
  sol.func @stk() {
    %stk = sol.alloca : !sol.ptr<i256>
    %ld = sol.load %stk : !sol.ptr<i256>, i256
    sol.return
  }

  sol.func @mem() {
    %mem = sol.malloc : !sol.array<3 x i256, Memory>
    %zero = arith.constant 0 : i256
    %ld = sol.load %mem[%zero] : !sol.array<3 x i256, Memory>, i256
    sol.return
  }

  sol.func @mem_2d() {
    %mem = sol.malloc : !sol.array<2 x !sol.array<3 x i256, Memory>, Memory>
    sol.return
  }

  sol.func @mem_struct() {
    %mem = sol.malloc : !sol.struct<(i256, !sol.array<3 x i256, Memory>), Memory>
    %zero = arith.constant 0 : i256
    %ld.0 = sol.load %mem[%zero] : !sol.struct<(i256, !sol.array<3 x i256, Memory>), Memory>, i256
    %one = arith.constant 1 : i256
    %ld.1 = sol.load %mem[%one] : !sol.struct<(i256, !sol.array<3 x i256, Memory>), Memory>, !sol.array<3 x i256, Memory>
    %ld.1.0 = sol.load %ld.1[%zero] : !sol.array<3 x i256, Memory>, i256
    sol.return
  }

  sol.func @dyn_arr() {
    %ten = arith.constant 10 : i256
    %mem = sol.malloc %ten : !sol.array<? x i256, Memory>
    %zero = arith.constant 0 : i256
    %ld = sol.load %mem[%zero] : !sol.array<? x i256, Memory>, i256
    sol.return
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   llvm.mlir.global private @ptr_calldata() : !llvm.ptr<3>
// CHECK-NEXT:   llvm.mlir.global private @calldatasize(0 : i256) {alignment = 32 : i64} : i256
// CHECK-NEXT:   func.func private @".unreachable"() attributes {llvm.linkage = #llvm.linkage<private>} {
// CHECK-NEXT:     llvm.unreachable
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @__revert(i256, i256, i256) attributes {llvm.linkage = #llvm.linkage<external>}
// CHECK-NEXT:   func.func @stk() attributes {llvm.linkage = #llvm.linkage<private>} {
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256
// CHECK-NEXT:     %0 = llvm.alloca %c1_i256 x i256 : (i256) -> !llvm.ptr<i256>
// CHECK-NEXT:     %1 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @mem() attributes {llvm.linkage = #llvm.linkage<private>} {
// CHECK-NEXT:     %c96_i256 = arith.constant 96 : i256
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256
// CHECK-NEXT:     %0 = llvm.inttoptr %c64_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %1 = llvm.load %0 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %2 = arith.addi %1, %c96_i256 : i256
// CHECK-NEXT:     %c18446744073709551615_i256 = arith.constant 18446744073709551615 : i256
// CHECK-NEXT:     %3 = arith.cmpi ugt, %2, %c18446744073709551615_i256 : i256
// CHECK-NEXT:     %4 = arith.cmpi ult, %2, %1 : i256
// CHECK-NEXT:     %5 = arith.ori %3, %4 : i1
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       %c0_i256_2 = arith.constant 0 : i256
// CHECK-NEXT:       %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256 = arith.constant 35408467139433450592217433187231851964531694900788300625387963629091585785856 : i256
// CHECK-NEXT:       %18 = llvm.inttoptr %c0_i256_2 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256, %18 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c4_i256 = arith.constant 4 : i256
// CHECK-NEXT:       %c65_i256 = arith.constant 65 : i256
// CHECK-NEXT:       %19 = llvm.inttoptr %c4_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c65_i256, %19 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c0_i256_3 = arith.constant 0 : i256
// CHECK-NEXT:       %c24_i256 = arith.constant 24 : i256
// CHECK-NEXT:       %c2_i256 = arith.constant 2 : i256
// CHECK-NEXT:       func.call @__revert(%c0_i256_3, %c24_i256, %c2_i256) : (i256, i256, i256) -> ()
// CHECK-NEXT:       func.call @".unreachable"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %c64_i256_0 = arith.constant 64 : i256
// CHECK-NEXT:     %6 = llvm.inttoptr %c64_i256_0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %2, %6 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256
// CHECK-NEXT:     %7 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256>
// CHECK-NEXT:     %8 = llvm.load %7 {alignment = 32 : i64} : !llvm.ptr<i256>
// CHECK-NEXT:     %9 = llvm.inttoptr %1 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %10 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %11 = llvm.load %10 {alignment = 32 : i64} : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %12 = llvm.getelementptr %11[%8] : (!llvm.ptr<3>, i256) -> !llvm.ptr<3>, i8
// CHECK-NEXT:     %13 = llvm.mlir.constant(false) : i1
// CHECK-NEXT:     "llvm.intr.memcpy"(%9, %12, %c96_i256, %13) : (!llvm.ptr<1>, !llvm.ptr<3>, i256, i1) -> ()
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256
// CHECK-NEXT:     %c32_i256 = arith.constant 32 : i256
// CHECK-NEXT:     %14 = arith.muli %c0_i256_1, %c32_i256 : i256
// CHECK-NEXT:     %15 = arith.addi %1, %14 : i256
// CHECK-NEXT:     %16 = llvm.inttoptr %15 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %17 = llvm.load %16 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @mem_2d() attributes {llvm.linkage = #llvm.linkage<private>} {
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256
// CHECK-NEXT:     %c64_i256_0 = arith.constant 64 : i256
// CHECK-NEXT:     %0 = llvm.inttoptr %c64_i256_0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %1 = llvm.load %0 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %2 = arith.addi %1, %c64_i256 : i256
// CHECK-NEXT:     %c18446744073709551615_i256 = arith.constant 18446744073709551615 : i256
// CHECK-NEXT:     %3 = arith.cmpi ugt, %2, %c18446744073709551615_i256 : i256
// CHECK-NEXT:     %4 = arith.cmpi ult, %2, %1 : i256
// CHECK-NEXT:     %5 = arith.ori %3, %4 : i1
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       %c0_i256_10 = arith.constant 0 : i256
// CHECK-NEXT:       %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256 = arith.constant 35408467139433450592217433187231851964531694900788300625387963629091585785856 : i256
// CHECK-NEXT:       %38 = llvm.inttoptr %c0_i256_10 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256, %38 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c4_i256 = arith.constant 4 : i256
// CHECK-NEXT:       %c65_i256 = arith.constant 65 : i256
// CHECK-NEXT:       %39 = llvm.inttoptr %c4_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c65_i256, %39 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c0_i256_11 = arith.constant 0 : i256
// CHECK-NEXT:       %c24_i256 = arith.constant 24 : i256
// CHECK-NEXT:       %c2_i256 = arith.constant 2 : i256
// CHECK-NEXT:       func.call @__revert(%c0_i256_11, %c24_i256, %c2_i256) : (i256, i256, i256) -> ()
// CHECK-NEXT:       func.call @".unreachable"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %c64_i256_1 = arith.constant 64 : i256
// CHECK-NEXT:     %6 = llvm.inttoptr %c64_i256_1 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %2, %6 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c96_i256 = arith.constant 96 : i256
// CHECK-NEXT:     %c64_i256_2 = arith.constant 64 : i256
// CHECK-NEXT:     %7 = llvm.inttoptr %c64_i256_2 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %8 = llvm.load %7 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %9 = arith.addi %8, %c96_i256 : i256
// CHECK-NEXT:     %c18446744073709551615_i256_3 = arith.constant 18446744073709551615 : i256
// CHECK-NEXT:     %10 = arith.cmpi ugt, %9, %c18446744073709551615_i256_3 : i256
// CHECK-NEXT:     %11 = arith.cmpi ult, %9, %8 : i256
// CHECK-NEXT:     %12 = arith.ori %10, %11 : i1
// CHECK-NEXT:     scf.if %12 {
// CHECK-NEXT:       %c0_i256_10 = arith.constant 0 : i256
// CHECK-NEXT:       %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256 = arith.constant 35408467139433450592217433187231851964531694900788300625387963629091585785856 : i256
// CHECK-NEXT:       %38 = llvm.inttoptr %c0_i256_10 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256, %38 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c4_i256 = arith.constant 4 : i256
// CHECK-NEXT:       %c65_i256 = arith.constant 65 : i256
// CHECK-NEXT:       %39 = llvm.inttoptr %c4_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c65_i256, %39 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c0_i256_11 = arith.constant 0 : i256
// CHECK-NEXT:       %c24_i256 = arith.constant 24 : i256
// CHECK-NEXT:       %c2_i256 = arith.constant 2 : i256
// CHECK-NEXT:       func.call @__revert(%c0_i256_11, %c24_i256, %c2_i256) : (i256, i256, i256) -> ()
// CHECK-NEXT:       func.call @".unreachable"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %c64_i256_4 = arith.constant 64 : i256
// CHECK-NEXT:     %13 = llvm.inttoptr %c64_i256_4 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %9, %13 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256
// CHECK-NEXT:     %14 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256>
// CHECK-NEXT:     %15 = llvm.load %14 {alignment = 32 : i64} : !llvm.ptr<i256>
// CHECK-NEXT:     %16 = llvm.inttoptr %8 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %17 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %18 = llvm.load %17 {alignment = 32 : i64} : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %19 = llvm.getelementptr %18[%15] : (!llvm.ptr<3>, i256) -> !llvm.ptr<3>, i8
// CHECK-NEXT:     %20 = llvm.mlir.constant(false) : i1
// CHECK-NEXT:     "llvm.intr.memcpy"(%16, %19, %c96_i256, %20) : (!llvm.ptr<1>, !llvm.ptr<3>, i256, i1) -> ()
// CHECK-NEXT:     %21 = llvm.inttoptr %1 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %8, %21 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c32_i256 = arith.constant 32 : i256
// CHECK-NEXT:     %22 = arith.addi %1, %c32_i256 : i256
// CHECK-NEXT:     %c96_i256_5 = arith.constant 96 : i256
// CHECK-NEXT:     %c64_i256_6 = arith.constant 64 : i256
// CHECK-NEXT:     %23 = llvm.inttoptr %c64_i256_6 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %24 = llvm.load %23 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %25 = arith.addi %24, %c96_i256_5 : i256
// CHECK-NEXT:     %c18446744073709551615_i256_7 = arith.constant 18446744073709551615 : i256
// CHECK-NEXT:     %26 = arith.cmpi ugt, %25, %c18446744073709551615_i256_7 : i256
// CHECK-NEXT:     %27 = arith.cmpi ult, %25, %24 : i256
// CHECK-NEXT:     %28 = arith.ori %26, %27 : i1
// CHECK-NEXT:     scf.if %28 {
// CHECK-NEXT:       %c0_i256_10 = arith.constant 0 : i256
// CHECK-NEXT:       %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256 = arith.constant 35408467139433450592217433187231851964531694900788300625387963629091585785856 : i256
// CHECK-NEXT:       %38 = llvm.inttoptr %c0_i256_10 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256, %38 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c4_i256 = arith.constant 4 : i256
// CHECK-NEXT:       %c65_i256 = arith.constant 65 : i256
// CHECK-NEXT:       %39 = llvm.inttoptr %c4_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c65_i256, %39 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c0_i256_11 = arith.constant 0 : i256
// CHECK-NEXT:       %c24_i256 = arith.constant 24 : i256
// CHECK-NEXT:       %c2_i256 = arith.constant 2 : i256
// CHECK-NEXT:       func.call @__revert(%c0_i256_11, %c24_i256, %c2_i256) : (i256, i256, i256) -> ()
// CHECK-NEXT:       func.call @".unreachable"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %c64_i256_8 = arith.constant 64 : i256
// CHECK-NEXT:     %29 = llvm.inttoptr %c64_i256_8 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %25, %29 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c0_i256_9 = arith.constant 0 : i256
// CHECK-NEXT:     %30 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256>
// CHECK-NEXT:     %31 = llvm.load %30 {alignment = 32 : i64} : !llvm.ptr<i256>
// CHECK-NEXT:     %32 = llvm.inttoptr %24 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %33 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %34 = llvm.load %33 {alignment = 32 : i64} : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %35 = llvm.getelementptr %34[%31] : (!llvm.ptr<3>, i256) -> !llvm.ptr<3>, i8
// CHECK-NEXT:     %36 = llvm.mlir.constant(false) : i1
// CHECK-NEXT:     "llvm.intr.memcpy"(%32, %35, %c96_i256_5, %36) : (!llvm.ptr<1>, !llvm.ptr<3>, i256, i1) -> ()
// CHECK-NEXT:     %37 = llvm.inttoptr %22 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %24, %37 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @mem_struct() attributes {llvm.linkage = #llvm.linkage<private>} {
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256
// CHECK-NEXT:     %c64_i256_0 = arith.constant 64 : i256
// CHECK-NEXT:     %0 = llvm.inttoptr %c64_i256_0 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %1 = llvm.load %0 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %2 = arith.addi %1, %c64_i256 : i256
// CHECK-NEXT:     %c18446744073709551615_i256 = arith.constant 18446744073709551615 : i256
// CHECK-NEXT:     %3 = arith.cmpi ugt, %2, %c18446744073709551615_i256 : i256
// CHECK-NEXT:     %4 = arith.cmpi ult, %2, %1 : i256
// CHECK-NEXT:     %5 = arith.ori %3, %4 : i1
// CHECK-NEXT:     scf.if %5 {
// CHECK-NEXT:       %c0_i256_9 = arith.constant 0 : i256
// CHECK-NEXT:       %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256 = arith.constant 35408467139433450592217433187231851964531694900788300625387963629091585785856 : i256
// CHECK-NEXT:       %35 = llvm.inttoptr %c0_i256_9 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256, %35 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c4_i256 = arith.constant 4 : i256
// CHECK-NEXT:       %c65_i256 = arith.constant 65 : i256
// CHECK-NEXT:       %36 = llvm.inttoptr %c4_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c65_i256, %36 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c0_i256_10 = arith.constant 0 : i256
// CHECK-NEXT:       %c24_i256 = arith.constant 24 : i256
// CHECK-NEXT:       %c2_i256 = arith.constant 2 : i256
// CHECK-NEXT:       func.call @__revert(%c0_i256_10, %c24_i256, %c2_i256) : (i256, i256, i256) -> ()
// CHECK-NEXT:       func.call @".unreachable"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %c64_i256_1 = arith.constant 64 : i256
// CHECK-NEXT:     %6 = llvm.inttoptr %c64_i256_1 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %2, %6 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256
// CHECK-NEXT:     %7 = llvm.inttoptr %1 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %c0_i256, %7 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c96_i256 = arith.constant 96 : i256
// CHECK-NEXT:     %c64_i256_2 = arith.constant 64 : i256
// CHECK-NEXT:     %8 = llvm.inttoptr %c64_i256_2 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %9 = llvm.load %8 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %10 = arith.addi %9, %c96_i256 : i256
// CHECK-NEXT:     %c18446744073709551615_i256_3 = arith.constant 18446744073709551615 : i256
// CHECK-NEXT:     %11 = arith.cmpi ugt, %10, %c18446744073709551615_i256_3 : i256
// CHECK-NEXT:     %12 = arith.cmpi ult, %10, %9 : i256
// CHECK-NEXT:     %13 = arith.ori %11, %12 : i1
// CHECK-NEXT:     scf.if %13 {
// CHECK-NEXT:       %c0_i256_9 = arith.constant 0 : i256
// CHECK-NEXT:       %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256 = arith.constant 35408467139433450592217433187231851964531694900788300625387963629091585785856 : i256
// CHECK-NEXT:       %35 = llvm.inttoptr %c0_i256_9 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256, %35 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c4_i256 = arith.constant 4 : i256
// CHECK-NEXT:       %c65_i256 = arith.constant 65 : i256
// CHECK-NEXT:       %36 = llvm.inttoptr %c4_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c65_i256, %36 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c0_i256_10 = arith.constant 0 : i256
// CHECK-NEXT:       %c24_i256 = arith.constant 24 : i256
// CHECK-NEXT:       %c2_i256 = arith.constant 2 : i256
// CHECK-NEXT:       func.call @__revert(%c0_i256_10, %c24_i256, %c2_i256) : (i256, i256, i256) -> ()
// CHECK-NEXT:       func.call @".unreachable"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %c64_i256_4 = arith.constant 64 : i256
// CHECK-NEXT:     %14 = llvm.inttoptr %c64_i256_4 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %10, %14 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c0_i256_5 = arith.constant 0 : i256
// CHECK-NEXT:     %15 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256>
// CHECK-NEXT:     %16 = llvm.load %15 {alignment = 32 : i64} : !llvm.ptr<i256>
// CHECK-NEXT:     %17 = llvm.inttoptr %9 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %18 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %19 = llvm.load %18 {alignment = 32 : i64} : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %20 = llvm.getelementptr %19[%16] : (!llvm.ptr<3>, i256) -> !llvm.ptr<3>, i8
// CHECK-NEXT:     %21 = llvm.mlir.constant(false) : i1
// CHECK-NEXT:     "llvm.intr.memcpy"(%17, %20, %c96_i256, %21) : (!llvm.ptr<1>, !llvm.ptr<3>, i256, i1) -> ()
// CHECK-NEXT:     %22 = llvm.inttoptr %1 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %9, %22 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c0_i256_6 = arith.constant 0 : i256
// CHECK-NEXT:     %c32_i256 = arith.constant 32 : i256
// CHECK-NEXT:     %23 = arith.muli %c0_i256_6, %c32_i256 : i256
// CHECK-NEXT:     %24 = arith.addi %1, %23 : i256
// CHECK-NEXT:     %25 = llvm.inttoptr %24 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %26 = llvm.load %25 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256
// CHECK-NEXT:     %c32_i256_7 = arith.constant 32 : i256
// CHECK-NEXT:     %27 = arith.muli %c1_i256, %c32_i256_7 : i256
// CHECK-NEXT:     %28 = arith.addi %1, %27 : i256
// CHECK-NEXT:     %29 = llvm.inttoptr %28 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %30 = llvm.load %29 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %c32_i256_8 = arith.constant 32 : i256
// CHECK-NEXT:     %31 = arith.muli %c0_i256_6, %c32_i256_8 : i256
// CHECK-NEXT:     %32 = arith.addi %30, %31 : i256
// CHECK-NEXT:     %33 = llvm.inttoptr %32 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %34 = llvm.load %33 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @dyn_arr() attributes {llvm.linkage = #llvm.linkage<private>} {
// CHECK-NEXT:     %c10_i256 = arith.constant 10 : i256
// CHECK-NEXT:     %c32_i256 = arith.constant 32 : i256
// CHECK-NEXT:     %0 = arith.muli %c10_i256, %c32_i256 : i256
// CHECK-NEXT:     %c32_i256_0 = arith.constant 32 : i256
// CHECK-NEXT:     %1 = arith.addi %0, %c32_i256_0 : i256
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256
// CHECK-NEXT:     %2 = llvm.inttoptr %c64_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %3 = llvm.load %2 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     %4 = arith.addi %3, %1 : i256
// CHECK-NEXT:     %c18446744073709551615_i256 = arith.constant 18446744073709551615 : i256
// CHECK-NEXT:     %5 = arith.cmpi ugt, %4, %c18446744073709551615_i256 : i256
// CHECK-NEXT:     %6 = arith.cmpi ult, %4, %3 : i256
// CHECK-NEXT:     %7 = arith.ori %5, %6 : i1
// CHECK-NEXT:     scf.if %7 {
// CHECK-NEXT:       %c0_i256_5 = arith.constant 0 : i256
// CHECK-NEXT:       %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256 = arith.constant 35408467139433450592217433187231851964531694900788300625387963629091585785856 : i256
// CHECK-NEXT:       %23 = llvm.inttoptr %c0_i256_5 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256, %23 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c4_i256 = arith.constant 4 : i256
// CHECK-NEXT:       %c65_i256 = arith.constant 65 : i256
// CHECK-NEXT:       %24 = llvm.inttoptr %c4_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c65_i256, %24 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c0_i256_6 = arith.constant 0 : i256
// CHECK-NEXT:       %c24_i256 = arith.constant 24 : i256
// CHECK-NEXT:       %c2_i256 = arith.constant 2 : i256
// CHECK-NEXT:       func.call @__revert(%c0_i256_6, %c24_i256, %c2_i256) : (i256, i256, i256) -> ()
// CHECK-NEXT:       func.call @".unreachable"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %c64_i256_1 = arith.constant 64 : i256
// CHECK-NEXT:     %8 = llvm.inttoptr %c64_i256_1 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %4, %8 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %9 = llvm.inttoptr %3 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     llvm.store %0, %9 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:     %c32_i256_2 = arith.constant 32 : i256
// CHECK-NEXT:     %10 = arith.addi %3, %c32_i256_2 : i256
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256
// CHECK-NEXT:     %11 = llvm.mlir.addressof @calldatasize : !llvm.ptr<i256>
// CHECK-NEXT:     %12 = llvm.load %11 {alignment = 32 : i64} : !llvm.ptr<i256>
// CHECK-NEXT:     %13 = llvm.inttoptr %10 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %14 = llvm.mlir.addressof @ptr_calldata : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %15 = llvm.load %14 {alignment = 32 : i64} : !llvm.ptr<ptr<3>>
// CHECK-NEXT:     %16 = llvm.getelementptr %15[%12] : (!llvm.ptr<3>, i256) -> !llvm.ptr<3>, i8
// CHECK-NEXT:     %17 = llvm.mlir.constant(false) : i1
// CHECK-NEXT:     "llvm.intr.memcpy"(%13, %16, %0, %17) : (!llvm.ptr<1>, !llvm.ptr<3>, i256, i1) -> ()
// CHECK-NEXT:     %c0_i256_3 = arith.constant 0 : i256
// CHECK-NEXT:     %c-1_i256 = arith.constant -1 : i256
// CHECK-NEXT:     %18 = arith.cmpi uge, %c0_i256_3, %c-1_i256 : i256
// CHECK-NEXT:     scf.if %18 {
// CHECK-NEXT:       %c0_i256_5 = arith.constant 0 : i256
// CHECK-NEXT:       %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256 = arith.constant 35408467139433450592217433187231851964531694900788300625387963629091585785856 : i256
// CHECK-NEXT:       %23 = llvm.inttoptr %c0_i256_5 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c35408467139433450592217433187231851964531694900788300625387963629091585785856_i256, %23 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c4_i256 = arith.constant 4 : i256
// CHECK-NEXT:       %c50_i256 = arith.constant 50 : i256
// CHECK-NEXT:       %24 = llvm.inttoptr %c4_i256 : i256 to !llvm.ptr<1>
// CHECK-NEXT:       llvm.store %c50_i256, %24 {alignment = 1 : i64} : i256, !llvm.ptr<1>
// CHECK-NEXT:       %c0_i256_6 = arith.constant 0 : i256
// CHECK-NEXT:       %c24_i256 = arith.constant 24 : i256
// CHECK-NEXT:       %c2_i256 = arith.constant 2 : i256
// CHECK-NEXT:       func.call @__revert(%c0_i256_6, %c24_i256, %c2_i256) : (i256, i256, i256) -> ()
// CHECK-NEXT:       func.call @".unreachable"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %c32_i256_4 = arith.constant 32 : i256
// CHECK-NEXT:     %19 = arith.muli %c0_i256_3, %c32_i256_4 : i256
// CHECK-NEXT:     %20 = arith.addi %10, %19 : i256
// CHECK-NEXT:     %21 = llvm.inttoptr %20 : i256 to !llvm.ptr<1>
// CHECK-NEXT:     %22 = llvm.load %21 {alignment = 1 : i64} : !llvm.ptr<1> -> i256
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-EMPTY:
