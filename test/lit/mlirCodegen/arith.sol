// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  function f_ui(uint256 a) private {}
  function f_si(int256 a) private {}

  function unchk_ui(uint256 a, uint256 b) private {
    unchecked {
      a += b;
      a -= b;
      a *= b;

      f_ui(a + b);
      f_ui(a - b);
      f_ui(a * b);
    }
  }

  function unchk_si(int256 a, int256 b) private {
    unchecked {
      a += b;
      a -= b;
      a *= b;

      f_si(a + b);
      f_si(a - b);
      f_si(a * b);
    }
  }

  function chk_ui(uint256 a, uint256 b) private {
    a += b;
    a -= b;

    f_ui(a + b);
    f_ui(a - b);
  }

  function chk_si(int256 a, int256 b) private {
    a += b;
    a -= b;

    f_si(a + b);
    f_si(a - b);
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc3 = loc({{.*}}:3:16)
// CHECK-NEXT: #loc5 = loc({{.*}}:4:16)
// CHECK-NEXT: #loc7 = loc({{.*}}:6:20)
// CHECK-NEXT: #loc8 = loc({{.*}}:6:31)
// CHECK-NEXT: #loc25 = loc({{.*}}:18:20)
// CHECK-NEXT: #loc26 = loc({{.*}}:18:30)
// CHECK-NEXT: #loc43 = loc({{.*}}:30:18)
// CHECK-NEXT: #loc44 = loc({{.*}}:30:29)
// CHECK-NEXT: #loc56 = loc({{.*}}:38:18)
// CHECK-NEXT: #loc57 = loc({{.*}}:38:28)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_147 {
// CHECK-NEXT:     sol.func @f_ui_6(%arg0: ui256 loc({{.*}}:3:16)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc3)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc3)
// CHECK-NEXT:       sol.return loc(#loc2)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:     sol.func @f_si_12(%arg0: si256 loc({{.*}}:4:16)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<si256, Stack> loc(#loc5)
// CHECK-NEXT:       sol.store %arg0, %0 : si256, !sol.ptr<si256, Stack> loc(#loc5)
// CHECK-NEXT:       sol.return loc(#loc4)
// CHECK-NEXT:     } loc(#loc4)
// CHECK-NEXT:     sol.func @unchk_ui_51(%arg0: ui256 loc({{.*}}:6:20), %arg1: ui256 loc({{.*}}:6:31)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc7)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc7)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc8)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc8)
// CHECK-NEXT:       %2 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc9)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc10)
// CHECK-NEXT:       %4 = sol.add %3, %2 : ui256 loc(#loc10)
// CHECK-NEXT:       sol.store %4, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc10)
// CHECK-NEXT:       %5 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc11)
// CHECK-NEXT:       %6 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc12)
// CHECK-NEXT:       %7 = sol.sub %6, %5 : ui256 loc(#loc12)
// CHECK-NEXT:       sol.store %7, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc12)
// CHECK-NEXT:       %8 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc13)
// CHECK-NEXT:       %9 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc14)
// CHECK-NEXT:       %10 = sol.mul %9, %8 : ui256 loc(#loc14)
// CHECK-NEXT:       sol.store %10, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc14)
// CHECK-NEXT:       %11 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc15)
// CHECK-NEXT:       %12 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc16)
// CHECK-NEXT:       %13 = sol.add %11, %12 : ui256 loc(#loc15)
// CHECK-NEXT:       sol.call @f_ui_6(%13) : (ui256) -> () loc(#loc17)
// CHECK-NEXT:       %14 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc18)
// CHECK-NEXT:       %15 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc19)
// CHECK-NEXT:       %16 = sol.sub %14, %15 : ui256 loc(#loc18)
// CHECK-NEXT:       sol.call @f_ui_6(%16) : (ui256) -> () loc(#loc20)
// CHECK-NEXT:       %17 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc21)
// CHECK-NEXT:       %18 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc22)
// CHECK-NEXT:       %19 = sol.mul %17, %18 : ui256 loc(#loc21)
// CHECK-NEXT:       sol.call @f_ui_6(%19) : (ui256) -> () loc(#loc23)
// CHECK-NEXT:       sol.return loc(#loc6)
// CHECK-NEXT:     } loc(#loc6)
// CHECK-NEXT:     sol.func @unchk_si_90(%arg0: si256 loc({{.*}}:18:20), %arg1: si256 loc({{.*}}:18:30)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<si256, Stack> loc(#loc25)
// CHECK-NEXT:       sol.store %arg0, %0 : si256, !sol.ptr<si256, Stack> loc(#loc25)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<si256, Stack> loc(#loc26)
// CHECK-NEXT:       sol.store %arg1, %1 : si256, !sol.ptr<si256, Stack> loc(#loc26)
// CHECK-NEXT:       %2 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc27)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc28)
// CHECK-NEXT:       %4 = sol.add %3, %2 : si256 loc(#loc28)
// CHECK-NEXT:       sol.store %4, %0 : si256, !sol.ptr<si256, Stack> loc(#loc28)
// CHECK-NEXT:       %5 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc29)
// CHECK-NEXT:       %6 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc30)
// CHECK-NEXT:       %7 = sol.sub %6, %5 : si256 loc(#loc30)
// CHECK-NEXT:       sol.store %7, %0 : si256, !sol.ptr<si256, Stack> loc(#loc30)
// CHECK-NEXT:       %8 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc31)
// CHECK-NEXT:       %9 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc32)
// CHECK-NEXT:       %10 = sol.mul %9, %8 : si256 loc(#loc32)
// CHECK-NEXT:       sol.store %10, %0 : si256, !sol.ptr<si256, Stack> loc(#loc32)
// CHECK-NEXT:       %11 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc33)
// CHECK-NEXT:       %12 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc34)
// CHECK-NEXT:       %13 = sol.add %11, %12 : si256 loc(#loc33)
// CHECK-NEXT:       sol.call @f_si_12(%13) : (si256) -> () loc(#loc35)
// CHECK-NEXT:       %14 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc36)
// CHECK-NEXT:       %15 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc37)
// CHECK-NEXT:       %16 = sol.sub %14, %15 : si256 loc(#loc36)
// CHECK-NEXT:       sol.call @f_si_12(%16) : (si256) -> () loc(#loc38)
// CHECK-NEXT:       %17 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc39)
// CHECK-NEXT:       %18 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc40)
// CHECK-NEXT:       %19 = sol.mul %17, %18 : si256 loc(#loc39)
// CHECK-NEXT:       sol.call @f_si_12(%19) : (si256) -> () loc(#loc41)
// CHECK-NEXT:       sol.return loc(#loc24)
// CHECK-NEXT:     } loc(#loc24)
// CHECK-NEXT:     sol.func @chk_ui_118(%arg0: ui256 loc({{.*}}:30:18), %arg1: ui256 loc({{.*}}:30:29)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc43)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc43)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc44)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc44)
// CHECK-NEXT:       %2 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc45)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc46)
// CHECK-NEXT:       %4 = sol.cadd %3, %2 : ui256 loc(#loc46)
// CHECK-NEXT:       sol.store %4, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc46)
// CHECK-NEXT:       %5 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc47)
// CHECK-NEXT:       %6 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc48)
// CHECK-NEXT:       %7 = sol.csub %6, %5 : ui256 loc(#loc48)
// CHECK-NEXT:       sol.store %7, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc48)
// CHECK-NEXT:       %8 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc49)
// CHECK-NEXT:       %9 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc50)
// CHECK-NEXT:       %10 = sol.cadd %8, %9 : ui256 loc(#loc49)
// CHECK-NEXT:       sol.call @f_ui_6(%10) : (ui256) -> () loc(#loc51)
// CHECK-NEXT:       %11 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc52)
// CHECK-NEXT:       %12 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc53)
// CHECK-NEXT:       %13 = sol.csub %11, %12 : ui256 loc(#loc52)
// CHECK-NEXT:       sol.call @f_ui_6(%13) : (ui256) -> () loc(#loc54)
// CHECK-NEXT:       sol.return loc(#loc42)
// CHECK-NEXT:     } loc(#loc42)
// CHECK-NEXT:     sol.func @chk_si_146(%arg0: si256 loc({{.*}}:38:18), %arg1: si256 loc({{.*}}:38:28)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<si256, Stack> loc(#loc56)
// CHECK-NEXT:       sol.store %arg0, %0 : si256, !sol.ptr<si256, Stack> loc(#loc56)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<si256, Stack> loc(#loc57)
// CHECK-NEXT:       sol.store %arg1, %1 : si256, !sol.ptr<si256, Stack> loc(#loc57)
// CHECK-NEXT:       %2 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc58)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc59)
// CHECK-NEXT:       %4 = sol.cadd %3, %2 : si256 loc(#loc59)
// CHECK-NEXT:       sol.store %4, %0 : si256, !sol.ptr<si256, Stack> loc(#loc59)
// CHECK-NEXT:       %5 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc60)
// CHECK-NEXT:       %6 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc61)
// CHECK-NEXT:       %7 = sol.csub %6, %5 : si256 loc(#loc61)
// CHECK-NEXT:       sol.store %7, %0 : si256, !sol.ptr<si256, Stack> loc(#loc61)
// CHECK-NEXT:       %8 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc62)
// CHECK-NEXT:       %9 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc63)
// CHECK-NEXT:       %10 = sol.cadd %8, %9 : si256 loc(#loc62)
// CHECK-NEXT:       sol.call @f_si_12(%10) : (si256) -> () loc(#loc64)
// CHECK-NEXT:       %11 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc65)
// CHECK-NEXT:       %12 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc66)
// CHECK-NEXT:       %13 = sol.csub %11, %12 : si256 loc(#loc65)
// CHECK-NEXT:       sol.call @f_si_12(%13) : (si256) -> () loc(#loc67)
// CHECK-NEXT:       sol.return loc(#loc55)
// CHECK-NEXT:     } loc(#loc55)
// CHECK-NEXT:   } {interface_fns = [], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:4:2)
// CHECK-NEXT: #loc6 = loc({{.*}}:6:2)
// CHECK-NEXT: #loc9 = loc({{.*}}:8:11)
// CHECK-NEXT: #loc10 = loc({{.*}}:8:6)
// CHECK-NEXT: #loc11 = loc({{.*}}:9:11)
// CHECK-NEXT: #loc12 = loc({{.*}}:9:6)
// CHECK-NEXT: #loc13 = loc({{.*}}:10:11)
// CHECK-NEXT: #loc14 = loc({{.*}}:10:6)
// CHECK-NEXT: #loc15 = loc({{.*}}:12:11)
// CHECK-NEXT: #loc16 = loc({{.*}}:12:15)
// CHECK-NEXT: #loc17 = loc({{.*}}:12:6)
// CHECK-NEXT: #loc18 = loc({{.*}}:13:11)
// CHECK-NEXT: #loc19 = loc({{.*}}:13:15)
// CHECK-NEXT: #loc20 = loc({{.*}}:13:6)
// CHECK-NEXT: #loc21 = loc({{.*}}:14:11)
// CHECK-NEXT: #loc22 = loc({{.*}}:14:15)
// CHECK-NEXT: #loc23 = loc({{.*}}:14:6)
// CHECK-NEXT: #loc24 = loc({{.*}}:18:2)
// CHECK-NEXT: #loc27 = loc({{.*}}:20:11)
// CHECK-NEXT: #loc28 = loc({{.*}}:20:6)
// CHECK-NEXT: #loc29 = loc({{.*}}:21:11)
// CHECK-NEXT: #loc30 = loc({{.*}}:21:6)
// CHECK-NEXT: #loc31 = loc({{.*}}:22:11)
// CHECK-NEXT: #loc32 = loc({{.*}}:22:6)
// CHECK-NEXT: #loc33 = loc({{.*}}:24:11)
// CHECK-NEXT: #loc34 = loc({{.*}}:24:15)
// CHECK-NEXT: #loc35 = loc({{.*}}:24:6)
// CHECK-NEXT: #loc36 = loc({{.*}}:25:11)
// CHECK-NEXT: #loc37 = loc({{.*}}:25:15)
// CHECK-NEXT: #loc38 = loc({{.*}}:25:6)
// CHECK-NEXT: #loc39 = loc({{.*}}:26:11)
// CHECK-NEXT: #loc40 = loc({{.*}}:26:15)
// CHECK-NEXT: #loc41 = loc({{.*}}:26:6)
// CHECK-NEXT: #loc42 = loc({{.*}}:30:2)
// CHECK-NEXT: #loc45 = loc({{.*}}:31:9)
// CHECK-NEXT: #loc46 = loc({{.*}}:31:4)
// CHECK-NEXT: #loc47 = loc({{.*}}:32:9)
// CHECK-NEXT: #loc48 = loc({{.*}}:32:4)
// CHECK-NEXT: #loc49 = loc({{.*}}:34:9)
// CHECK-NEXT: #loc50 = loc({{.*}}:34:13)
// CHECK-NEXT: #loc51 = loc({{.*}}:34:4)
// CHECK-NEXT: #loc52 = loc({{.*}}:35:9)
// CHECK-NEXT: #loc53 = loc({{.*}}:35:13)
// CHECK-NEXT: #loc54 = loc({{.*}}:35:4)
// CHECK-NEXT: #loc55 = loc({{.*}}:38:2)
// CHECK-NEXT: #loc58 = loc({{.*}}:39:9)
// CHECK-NEXT: #loc59 = loc({{.*}}:39:4)
// CHECK-NEXT: #loc60 = loc({{.*}}:40:9)
// CHECK-NEXT: #loc61 = loc({{.*}}:40:4)
// CHECK-NEXT: #loc62 = loc({{.*}}:42:9)
// CHECK-NEXT: #loc63 = loc({{.*}}:42:13)
// CHECK-NEXT: #loc64 = loc({{.*}}:42:4)
// CHECK-NEXT: #loc65 = loc({{.*}}:43:9)
// CHECK-NEXT: #loc66 = loc({{.*}}:43:13)
// CHECK-NEXT: #loc67 = loc({{.*}}:43:4)
// CHECK-EMPTY:
