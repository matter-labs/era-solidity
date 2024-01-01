// RUN: solc --yul --yul-dialect=evm --mlir-action=print-init --mmlir --mlir-print-debuginfo --mlir-target=eravm %s | FileCheck %s

object "Test" {
  code {
    let a := mload(32)
    let b := mload(64)
    if lt(a, b) {
      mstore(0, 0)
    }
    mstore(0, lt(a, b))
    if slt(a, b) {
      mstore(0, 0)
    }
    mstore(0, slt(a, b))
    if iszero(a) {
      mstore(1, 0)
    }
    mstore(1, iszero(a))
    return(0, 0)
  }
  object "Test_deployed" {
    code {
      let a := mload(32)
      let b := mload(64)
      if lt(a, b) {
        mstore(0, 0)
      }
      mstore(0, lt(a, b))
      if slt(a, b) {
        mstore(0, 0)
      }
      mstore(0, slt(a, b))
      if iszero(a) {
        mstore(1, 0)
      }
      mstore(1, iszero(a))
      return(0, 0)
    }
  }
}
// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: module {
// CHECK-NEXT:   sol.object @Test {
// CHECK-NEXT:     %c1_i256 = arith.constant 1 : i256 loc(#loc1)
// CHECK-NEXT:     %0 = llvm.alloca %c1_i256 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc2)
// CHECK-NEXT:     %c32_i256 = arith.constant 32 : i256 loc(#loc3)
// CHECK-NEXT:     %1 = sol.mload %c32_i256 loc(#loc4)
// CHECK-NEXT:     llvm.store %1, %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc1)
// CHECK-NEXT:     %c1_i256_0 = arith.constant 1 : i256 loc(#loc5)
// CHECK-NEXT:     %2 = llvm.alloca %c1_i256_0 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc6)
// CHECK-NEXT:     %c64_i256 = arith.constant 64 : i256 loc(#loc7)
// CHECK-NEXT:     %3 = sol.mload %c64_i256 loc(#loc8)
// CHECK-NEXT:     llvm.store %3, %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc5)
// CHECK-NEXT:     %4 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc9)
// CHECK-NEXT:     %5 = llvm.load %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc10)
// CHECK-NEXT:     %6 = arith.cmpi ult, %4, %5 : i256 loc(#loc11)
// CHECK-NEXT:     scf.if %6 {
// CHECK-NEXT:       %c0_i256_7 = arith.constant 0 : i256 loc(#loc13)
// CHECK-NEXT:       %c0_i256_8 = arith.constant 0 : i256 loc(#loc14)
// CHECK-NEXT:       sol.mstore %c0_i256_7, %c0_i256_8 loc(#loc15)
// CHECK-NEXT:     } loc(#loc12)
// CHECK-NEXT:     %c0_i256 = arith.constant 0 : i256 loc(#loc16)
// CHECK-NEXT:     %7 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc17)
// CHECK-NEXT:     %8 = llvm.load %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc18)
// CHECK-NEXT:     %9 = arith.cmpi ult, %7, %8 : i256 loc(#loc19)
// CHECK-NEXT:     %10 = arith.extui %9 : i1 to i256 loc(#loc19)
// CHECK-NEXT:     sol.mstore %c0_i256, %10 loc(#loc20)
// CHECK-NEXT:     %11 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc21)
// CHECK-NEXT:     %12 = llvm.load %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc22)
// CHECK-NEXT:     %13 = arith.cmpi slt, %11, %12 : i256 loc(#loc23)
// CHECK-NEXT:     scf.if %13 {
// CHECK-NEXT:       %c0_i256_7 = arith.constant 0 : i256 loc(#loc25)
// CHECK-NEXT:       %c0_i256_8 = arith.constant 0 : i256 loc(#loc26)
// CHECK-NEXT:       sol.mstore %c0_i256_7, %c0_i256_8 loc(#loc27)
// CHECK-NEXT:     } loc(#loc24)
// CHECK-NEXT:     %c0_i256_1 = arith.constant 0 : i256 loc(#loc28)
// CHECK-NEXT:     %14 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc29)
// CHECK-NEXT:     %15 = llvm.load %2 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc30)
// CHECK-NEXT:     %16 = arith.cmpi slt, %14, %15 : i256 loc(#loc31)
// CHECK-NEXT:     %17 = arith.extui %16 : i1 to i256 loc(#loc31)
// CHECK-NEXT:     sol.mstore %c0_i256_1, %17 loc(#loc32)
// CHECK-NEXT:     %18 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc33)
// CHECK-NEXT:     %c0_i256_2 = arith.constant 0 : i256 loc(#loc34)
// CHECK-NEXT:     %19 = arith.cmpi eq, %18, %c0_i256_2 : i256 loc(#loc34)
// CHECK-NEXT:     scf.if %19 {
// CHECK-NEXT:       %c1_i256_7 = arith.constant 1 : i256 loc(#loc36)
// CHECK-NEXT:       %c0_i256_8 = arith.constant 0 : i256 loc(#loc37)
// CHECK-NEXT:       sol.mstore %c1_i256_7, %c0_i256_8 loc(#loc38)
// CHECK-NEXT:     } loc(#loc35)
// CHECK-NEXT:     %c1_i256_3 = arith.constant 1 : i256 loc(#loc39)
// CHECK-NEXT:     %20 = llvm.load %0 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc40)
// CHECK-NEXT:     %c0_i256_4 = arith.constant 0 : i256 loc(#loc41)
// CHECK-NEXT:     %21 = arith.cmpi eq, %20, %c0_i256_4 : i256 loc(#loc41)
// CHECK-NEXT:     %22 = arith.extui %21 : i1 to i256 loc(#loc41)
// CHECK-NEXT:     sol.mstore %c1_i256_3, %22 loc(#loc42)
// CHECK-NEXT:     %c0_i256_5 = arith.constant 0 : i256 loc(#loc43)
// CHECK-NEXT:     %c0_i256_6 = arith.constant 0 : i256 loc(#loc44)
// CHECK-NEXT:     sol.return %c0_i256_5, %c0_i256_6 loc(#loc45)
// CHECK-NEXT:     sol.object @Test_deployed {
// CHECK-NEXT:       %c1_i256_7 = arith.constant 1 : i256 loc(#loc46)
// CHECK-NEXT:       %23 = llvm.alloca %c1_i256_7 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc47)
// CHECK-NEXT:       %c32_i256_8 = arith.constant 32 : i256 loc(#loc48)
// CHECK-NEXT:       %24 = sol.mload %c32_i256_8 loc(#loc49)
// CHECK-NEXT:       llvm.store %24, %23 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc46)
// CHECK-NEXT:       %c1_i256_9 = arith.constant 1 : i256 loc(#loc50)
// CHECK-NEXT:       %25 = llvm.alloca %c1_i256_9 x i256 {alignment = 32 : i64} : (i256) -> !llvm.ptr<i256> loc(#loc51)
// CHECK-NEXT:       %c64_i256_10 = arith.constant 64 : i256 loc(#loc52)
// CHECK-NEXT:       %26 = sol.mload %c64_i256_10 loc(#loc53)
// CHECK-NEXT:       llvm.store %26, %25 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc50)
// CHECK-NEXT:       %27 = llvm.load %23 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc54)
// CHECK-NEXT:       %28 = llvm.load %25 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc55)
// CHECK-NEXT:       %29 = arith.cmpi ult, %27, %28 : i256 loc(#loc56)
// CHECK-NEXT:       scf.if %29 {
// CHECK-NEXT:         %c0_i256_18 = arith.constant 0 : i256 loc(#loc58)
// CHECK-NEXT:         %c0_i256_19 = arith.constant 0 : i256 loc(#loc59)
// CHECK-NEXT:         sol.mstore %c0_i256_18, %c0_i256_19 loc(#loc60)
// CHECK-NEXT:       } loc(#loc57)
// CHECK-NEXT:       %c0_i256_11 = arith.constant 0 : i256 loc(#loc61)
// CHECK-NEXT:       %30 = llvm.load %23 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc62)
// CHECK-NEXT:       %31 = llvm.load %25 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc63)
// CHECK-NEXT:       %32 = arith.cmpi ult, %30, %31 : i256 loc(#loc64)
// CHECK-NEXT:       %33 = arith.extui %32 : i1 to i256 loc(#loc64)
// CHECK-NEXT:       sol.mstore %c0_i256_11, %33 loc(#loc65)
// CHECK-NEXT:       %34 = llvm.load %23 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc66)
// CHECK-NEXT:       %35 = llvm.load %25 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc67)
// CHECK-NEXT:       %36 = arith.cmpi slt, %34, %35 : i256 loc(#loc68)
// CHECK-NEXT:       scf.if %36 {
// CHECK-NEXT:         %c0_i256_18 = arith.constant 0 : i256 loc(#loc70)
// CHECK-NEXT:         %c0_i256_19 = arith.constant 0 : i256 loc(#loc71)
// CHECK-NEXT:         sol.mstore %c0_i256_18, %c0_i256_19 loc(#loc72)
// CHECK-NEXT:       } loc(#loc69)
// CHECK-NEXT:       %c0_i256_12 = arith.constant 0 : i256 loc(#loc73)
// CHECK-NEXT:       %37 = llvm.load %23 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc74)
// CHECK-NEXT:       %38 = llvm.load %25 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc75)
// CHECK-NEXT:       %39 = arith.cmpi slt, %37, %38 : i256 loc(#loc76)
// CHECK-NEXT:       %40 = arith.extui %39 : i1 to i256 loc(#loc76)
// CHECK-NEXT:       sol.mstore %c0_i256_12, %40 loc(#loc77)
// CHECK-NEXT:       %41 = llvm.load %23 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc78)
// CHECK-NEXT:       %c0_i256_13 = arith.constant 0 : i256 loc(#loc79)
// CHECK-NEXT:       %42 = arith.cmpi eq, %41, %c0_i256_13 : i256 loc(#loc79)
// CHECK-NEXT:       scf.if %42 {
// CHECK-NEXT:         %c1_i256_18 = arith.constant 1 : i256 loc(#loc81)
// CHECK-NEXT:         %c0_i256_19 = arith.constant 0 : i256 loc(#loc82)
// CHECK-NEXT:         sol.mstore %c1_i256_18, %c0_i256_19 loc(#loc83)
// CHECK-NEXT:       } loc(#loc80)
// CHECK-NEXT:       %c1_i256_14 = arith.constant 1 : i256 loc(#loc84)
// CHECK-NEXT:       %43 = llvm.load %23 {alignment = 32 : i64} : !llvm.ptr<i256> loc(#loc85)
// CHECK-NEXT:       %c0_i256_15 = arith.constant 0 : i256 loc(#loc86)
// CHECK-NEXT:       %44 = arith.cmpi eq, %43, %c0_i256_15 : i256 loc(#loc86)
// CHECK-NEXT:       %45 = arith.extui %44 : i1 to i256 loc(#loc86)
// CHECK-NEXT:       sol.mstore %c1_i256_14, %45 loc(#loc87)
// CHECK-NEXT:       %c0_i256_16 = arith.constant 0 : i256 loc(#loc88)
// CHECK-NEXT:       %c0_i256_17 = arith.constant 0 : i256 loc(#loc89)
// CHECK-NEXT:       sol.return %c0_i256_16, %c0_i256_17 loc(#loc90)
// CHECK-NEXT:     } loc(#loc0)
// CHECK-NEXT:   } loc(#loc0)
// CHECK-NEXT: } loc(#loc0)
// CHECK-NEXT: #loc0 = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:4:4)
// CHECK-NEXT: #loc2 = loc({{.*}}:4:8)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:19)
// CHECK-NEXT: #loc4 = loc({{.*}}:4:13)
// CHECK-NEXT: #loc5 = loc({{.*}}:5:4)
// CHECK-NEXT: #loc6 = loc({{.*}}:5:8)
// CHECK-NEXT: #loc7 = loc({{.*}}:5:19)
// CHECK-NEXT: #loc8 = loc({{.*}}:5:13)
// CHECK-NEXT: #loc9 = loc({{.*}}:6:10)
// CHECK-NEXT: #loc10 = loc({{.*}}:6:13)
// CHECK-NEXT: #loc11 = loc({{.*}}:6:7)
// CHECK-NEXT: #loc12 = loc({{.*}}:6:4)
// CHECK-NEXT: #loc13 = loc({{.*}}:7:13)
// CHECK-NEXT: #loc14 = loc({{.*}}:7:16)
// CHECK-NEXT: #loc15 = loc({{.*}}:7:6)
// CHECK-NEXT: #loc16 = loc({{.*}}:9:11)
// CHECK-NEXT: #loc17 = loc({{.*}}:9:17)
// CHECK-NEXT: #loc18 = loc({{.*}}:9:20)
// CHECK-NEXT: #loc19 = loc({{.*}}:9:14)
// CHECK-NEXT: #loc20 = loc({{.*}}:9:4)
// CHECK-NEXT: #loc21 = loc({{.*}}:10:11)
// CHECK-NEXT: #loc22 = loc({{.*}}:10:14)
// CHECK-NEXT: #loc23 = loc({{.*}}:10:7)
// CHECK-NEXT: #loc24 = loc({{.*}}:10:4)
// CHECK-NEXT: #loc25 = loc({{.*}}:11:13)
// CHECK-NEXT: #loc26 = loc({{.*}}:11:16)
// CHECK-NEXT: #loc27 = loc({{.*}}:11:6)
// CHECK-NEXT: #loc28 = loc({{.*}}:13:11)
// CHECK-NEXT: #loc29 = loc({{.*}}:13:18)
// CHECK-NEXT: #loc30 = loc({{.*}}:13:21)
// CHECK-NEXT: #loc31 = loc({{.*}}:13:14)
// CHECK-NEXT: #loc32 = loc({{.*}}:13:4)
// CHECK-NEXT: #loc33 = loc({{.*}}:14:14)
// CHECK-NEXT: #loc34 = loc({{.*}}:14:7)
// CHECK-NEXT: #loc35 = loc({{.*}}:14:4)
// CHECK-NEXT: #loc36 = loc({{.*}}:15:13)
// CHECK-NEXT: #loc37 = loc({{.*}}:15:16)
// CHECK-NEXT: #loc38 = loc({{.*}}:15:6)
// CHECK-NEXT: #loc39 = loc({{.*}}:17:11)
// CHECK-NEXT: #loc40 = loc({{.*}}:17:21)
// CHECK-NEXT: #loc41 = loc({{.*}}:17:14)
// CHECK-NEXT: #loc42 = loc({{.*}}:17:4)
// CHECK-NEXT: #loc43 = loc({{.*}}:18:11)
// CHECK-NEXT: #loc44 = loc({{.*}}:18:14)
// CHECK-NEXT: #loc45 = loc({{.*}}:18:4)
// CHECK-NEXT: #loc46 = loc({{.*}}:22:6)
// CHECK-NEXT: #loc47 = loc({{.*}}:22:10)
// CHECK-NEXT: #loc48 = loc({{.*}}:22:21)
// CHECK-NEXT: #loc49 = loc({{.*}}:22:15)
// CHECK-NEXT: #loc50 = loc({{.*}}:23:6)
// CHECK-NEXT: #loc51 = loc({{.*}}:23:10)
// CHECK-NEXT: #loc52 = loc({{.*}}:23:21)
// CHECK-NEXT: #loc53 = loc({{.*}}:23:15)
// CHECK-NEXT: #loc54 = loc({{.*}}:24:12)
// CHECK-NEXT: #loc55 = loc({{.*}}:24:15)
// CHECK-NEXT: #loc56 = loc({{.*}}:24:9)
// CHECK-NEXT: #loc57 = loc({{.*}}:24:6)
// CHECK-NEXT: #loc58 = loc({{.*}}:25:15)
// CHECK-NEXT: #loc59 = loc({{.*}}:25:18)
// CHECK-NEXT: #loc60 = loc({{.*}}:25:8)
// CHECK-NEXT: #loc61 = loc({{.*}}:27:13)
// CHECK-NEXT: #loc62 = loc({{.*}}:27:19)
// CHECK-NEXT: #loc63 = loc({{.*}}:27:22)
// CHECK-NEXT: #loc64 = loc({{.*}}:27:16)
// CHECK-NEXT: #loc65 = loc({{.*}}:27:6)
// CHECK-NEXT: #loc66 = loc({{.*}}:28:13)
// CHECK-NEXT: #loc67 = loc({{.*}}:28:16)
// CHECK-NEXT: #loc68 = loc({{.*}}:28:9)
// CHECK-NEXT: #loc69 = loc({{.*}}:28:6)
// CHECK-NEXT: #loc70 = loc({{.*}}:29:15)
// CHECK-NEXT: #loc71 = loc({{.*}}:29:18)
// CHECK-NEXT: #loc72 = loc({{.*}}:29:8)
// CHECK-NEXT: #loc73 = loc({{.*}}:31:13)
// CHECK-NEXT: #loc74 = loc({{.*}}:31:20)
// CHECK-NEXT: #loc75 = loc({{.*}}:31:23)
// CHECK-NEXT: #loc76 = loc({{.*}}:31:16)
// CHECK-NEXT: #loc77 = loc({{.*}}:31:6)
// CHECK-NEXT: #loc78 = loc({{.*}}:32:16)
// CHECK-NEXT: #loc79 = loc({{.*}}:32:9)
// CHECK-NEXT: #loc80 = loc({{.*}}:32:6)
// CHECK-NEXT: #loc81 = loc({{.*}}:33:15)
// CHECK-NEXT: #loc82 = loc({{.*}}:33:18)
// CHECK-NEXT: #loc83 = loc({{.*}}:33:8)
// CHECK-NEXT: #loc84 = loc({{.*}}:35:13)
// CHECK-NEXT: #loc85 = loc({{.*}}:35:23)
// CHECK-NEXT: #loc86 = loc({{.*}}:35:16)
// CHECK-NEXT: #loc87 = loc({{.*}}:35:6)
// CHECK-NEXT: #loc88 = loc({{.*}}:36:13)
// CHECK-NEXT: #loc89 = loc({{.*}}:36:16)
// CHECK-NEXT: #loc90 = loc({{.*}}:36:6)
// CHECK-EMPTY: