// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  function f(bool a) private {}

  function ui(uint256 a, uint256 b) private {
    f(a == b);
    f(a != b);
    f(a < b);
    f(a <= b);
    f(a > b);
    f(a >= b);
  }

  function si(int256 a, int256 b) private {
    f(a == b);
    f(a != b);
    f(a < b);
    f(a <= b);
    f(a > b);
    f(a >= b);
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc3 = loc({{.*}}:3:13)
// CHECK-NEXT: #loc5 = loc({{.*}}:5:14)
// CHECK-NEXT: #loc6 = loc({{.*}}:5:25)
// CHECK-NEXT: #loc26 = loc({{.*}}:14:14)
// CHECK-NEXT: #loc27 = loc({{.*}}:14:24)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_95 {
// CHECK-NEXT:     sol.func @f_6(%arg0: i1 loc({{.*}}:3:13)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<i1, Stack> loc(#loc3)
// CHECK-NEXT:       sol.store %arg0, %0 : i1, !sol.ptr<i1, Stack> loc(#loc3)
// CHECK-NEXT:       sol.return loc(#loc2)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:     sol.func @ui_50(%arg0: ui256 loc({{.*}}:5:14), %arg1: ui256 loc({{.*}}:5:25)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc5)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc5)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc6)
// CHECK-NEXT:       sol.store %arg1, %1 : ui256, !sol.ptr<ui256, Stack> loc(#loc6)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc7)
// CHECK-NEXT:       %3 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc8)
// CHECK-NEXT:       %4 = sol.cmp eq, %2, %3 : ui256 loc(#loc7)
// CHECK-NEXT:       sol.call @f_6(%4) : (i1) -> () loc(#loc9)
// CHECK-NEXT:       %5 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc10)
// CHECK-NEXT:       %6 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc11)
// CHECK-NEXT:       %7 = sol.cmp ne, %5, %6 : ui256 loc(#loc10)
// CHECK-NEXT:       sol.call @f_6(%7) : (i1) -> () loc(#loc12)
// CHECK-NEXT:       %8 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc13)
// CHECK-NEXT:       %9 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc14)
// CHECK-NEXT:       %10 = sol.cmp lt, %8, %9 : ui256 loc(#loc13)
// CHECK-NEXT:       sol.call @f_6(%10) : (i1) -> () loc(#loc15)
// CHECK-NEXT:       %11 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc16)
// CHECK-NEXT:       %12 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc17)
// CHECK-NEXT:       %13 = sol.cmp le, %11, %12 : ui256 loc(#loc16)
// CHECK-NEXT:       sol.call @f_6(%13) : (i1) -> () loc(#loc18)
// CHECK-NEXT:       %14 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc19)
// CHECK-NEXT:       %15 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc20)
// CHECK-NEXT:       %16 = sol.cmp gt, %14, %15 : ui256 loc(#loc19)
// CHECK-NEXT:       sol.call @f_6(%16) : (i1) -> () loc(#loc21)
// CHECK-NEXT:       %17 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc22)
// CHECK-NEXT:       %18 = sol.load %1 : !sol.ptr<ui256, Stack>, ui256 loc(#loc23)
// CHECK-NEXT:       %19 = sol.cmp ge, %17, %18 : ui256 loc(#loc22)
// CHECK-NEXT:       sol.call @f_6(%19) : (i1) -> () loc(#loc24)
// CHECK-NEXT:       sol.return loc(#loc4)
// CHECK-NEXT:     } loc(#loc4)
// CHECK-NEXT:     sol.func @si_94(%arg0: si256 loc({{.*}}:14:14), %arg1: si256 loc({{.*}}:14:24)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<si256, Stack> loc(#loc26)
// CHECK-NEXT:       sol.store %arg0, %0 : si256, !sol.ptr<si256, Stack> loc(#loc26)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<si256, Stack> loc(#loc27)
// CHECK-NEXT:       sol.store %arg1, %1 : si256, !sol.ptr<si256, Stack> loc(#loc27)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc28)
// CHECK-NEXT:       %3 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc29)
// CHECK-NEXT:       %4 = sol.cmp eq, %2, %3 : si256 loc(#loc28)
// CHECK-NEXT:       sol.call @f_6(%4) : (i1) -> () loc(#loc30)
// CHECK-NEXT:       %5 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc31)
// CHECK-NEXT:       %6 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc32)
// CHECK-NEXT:       %7 = sol.cmp ne, %5, %6 : si256 loc(#loc31)
// CHECK-NEXT:       sol.call @f_6(%7) : (i1) -> () loc(#loc33)
// CHECK-NEXT:       %8 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc34)
// CHECK-NEXT:       %9 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc35)
// CHECK-NEXT:       %10 = sol.cmp lt, %8, %9 : si256 loc(#loc34)
// CHECK-NEXT:       sol.call @f_6(%10) : (i1) -> () loc(#loc36)
// CHECK-NEXT:       %11 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc37)
// CHECK-NEXT:       %12 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc38)
// CHECK-NEXT:       %13 = sol.cmp le, %11, %12 : si256 loc(#loc37)
// CHECK-NEXT:       sol.call @f_6(%13) : (i1) -> () loc(#loc39)
// CHECK-NEXT:       %14 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc40)
// CHECK-NEXT:       %15 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc41)
// CHECK-NEXT:       %16 = sol.cmp gt, %14, %15 : si256 loc(#loc40)
// CHECK-NEXT:       sol.call @f_6(%16) : (i1) -> () loc(#loc42)
// CHECK-NEXT:       %17 = sol.load %0 : !sol.ptr<si256, Stack>, si256 loc(#loc43)
// CHECK-NEXT:       %18 = sol.load %1 : !sol.ptr<si256, Stack>, si256 loc(#loc44)
// CHECK-NEXT:       %19 = sol.cmp ge, %17, %18 : si256 loc(#loc43)
// CHECK-NEXT:       sol.call @f_6(%19) : (i1) -> () loc(#loc45)
// CHECK-NEXT:       sol.return loc(#loc25)
// CHECK-NEXT:     } loc(#loc25)
// CHECK-NEXT:   } {interface_fns = [], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:5:2)
// CHECK-NEXT: #loc7 = loc({{.*}}:6:6)
// CHECK-NEXT: #loc8 = loc({{.*}}:6:11)
// CHECK-NEXT: #loc9 = loc({{.*}}:6:4)
// CHECK-NEXT: #loc10 = loc({{.*}}:7:6)
// CHECK-NEXT: #loc11 = loc({{.*}}:7:11)
// CHECK-NEXT: #loc12 = loc({{.*}}:7:4)
// CHECK-NEXT: #loc13 = loc({{.*}}:8:6)
// CHECK-NEXT: #loc14 = loc({{.*}}:8:10)
// CHECK-NEXT: #loc15 = loc({{.*}}:8:4)
// CHECK-NEXT: #loc16 = loc({{.*}}:9:6)
// CHECK-NEXT: #loc17 = loc({{.*}}:9:11)
// CHECK-NEXT: #loc18 = loc({{.*}}:9:4)
// CHECK-NEXT: #loc19 = loc({{.*}}:10:6)
// CHECK-NEXT: #loc20 = loc({{.*}}:10:10)
// CHECK-NEXT: #loc21 = loc({{.*}}:10:4)
// CHECK-NEXT: #loc22 = loc({{.*}}:11:6)
// CHECK-NEXT: #loc23 = loc({{.*}}:11:11)
// CHECK-NEXT: #loc24 = loc({{.*}}:11:4)
// CHECK-NEXT: #loc25 = loc({{.*}}:14:2)
// CHECK-NEXT: #loc28 = loc({{.*}}:15:6)
// CHECK-NEXT: #loc29 = loc({{.*}}:15:11)
// CHECK-NEXT: #loc30 = loc({{.*}}:15:4)
// CHECK-NEXT: #loc31 = loc({{.*}}:16:6)
// CHECK-NEXT: #loc32 = loc({{.*}}:16:11)
// CHECK-NEXT: #loc33 = loc({{.*}}:16:4)
// CHECK-NEXT: #loc34 = loc({{.*}}:17:6)
// CHECK-NEXT: #loc35 = loc({{.*}}:17:10)
// CHECK-NEXT: #loc36 = loc({{.*}}:17:4)
// CHECK-NEXT: #loc37 = loc({{.*}}:18:6)
// CHECK-NEXT: #loc38 = loc({{.*}}:18:11)
// CHECK-NEXT: #loc39 = loc({{.*}}:18:4)
// CHECK-NEXT: #loc40 = loc({{.*}}:19:6)
// CHECK-NEXT: #loc41 = loc({{.*}}:19:10)
// CHECK-NEXT: #loc42 = loc({{.*}}:19:4)
// CHECK-NEXT: #loc43 = loc({{.*}}:20:6)
// CHECK-NEXT: #loc44 = loc({{.*}}:20:11)
// CHECK-NEXT: #loc45 = loc({{.*}}:20:4)
// CHECK-EMPTY:
