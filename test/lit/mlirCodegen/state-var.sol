// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  uint ui;
  mapping(address => uint) m0;
  mapping(address => mapping(address => uint256)) m1;
  string s;

  function f_ui() public view returns (uint) { return ui; }
  function f_m0(address a) public view returns (uint) { return m0[a]; }
  function f_m1(address a, address b) public view returns (uint) { return m1[a][b]; }
  function f_s() public view returns (string memory) { return s; }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc10 = loc({{.*}}:9:16)
// CHECK-NEXT: #loc15 = loc({{.*}}:10:16)
// CHECK-NEXT: #loc16 = loc({{.*}}:10:27)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_59 {
// CHECK-NEXT:     sol.state_var @ui : i256 loc(#loc2)
// CHECK-NEXT:     sol.state_var @m0 : !sol.mapping<i256, i256> loc(#loc3)
// CHECK-NEXT:     sol.state_var @m1 : !sol.mapping<i256, !sol.mapping<i256, i256>> loc(#loc4)
// CHECK-NEXT:     sol.state_var @s : !sol.string<Storage> loc(#loc5)
// CHECK-NEXT:     sol.func @f_ui_22() -> i256 attributes {state_mutability = #sol<StateMutability View>} {
// CHECK-NEXT:       %0 = sol.addr_of @ui : !sol.ptr<i256, Storage> loc(#loc2)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<i256, Storage>, i256 loc(#loc7)
// CHECK-NEXT:       sol.return %1 : i256 loc(#loc8)
// CHECK-NEXT:     } loc(#loc6)
// CHECK-NEXT:     sol.func @f_m0_34(%arg0: i256 loc({{.*}}:9:16)) -> i256 attributes {state_mutability = #sol<StateMutability View>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<i256, Stack> loc(#loc10)
// CHECK-NEXT:       sol.store %arg0, %0 : i256, !sol.ptr<i256, Stack> loc(#loc10)
// CHECK-NEXT:       %1 = sol.addr_of @m0 : !sol.mapping<i256, i256> loc(#loc3)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<i256, Stack>, i256 loc(#loc11)
// CHECK-NEXT:       %3 = sol.map %1, %2 : !sol.mapping<i256, i256>, !sol.ptr<i256, Storage> loc(#loc12)
// CHECK-NEXT:       %4 = sol.load %3 : !sol.ptr<i256, Storage>, i256 loc(#loc12)
// CHECK-NEXT:       sol.return %4 : i256 loc(#loc13)
// CHECK-NEXT:     } loc(#loc9)
// CHECK-NEXT:     sol.func @f_m1_50(%arg0: i256 loc({{.*}}:10:16), %arg1: i256 loc({{.*}}:10:27)) -> i256 attributes {state_mutability = #sol<StateMutability View>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<i256, Stack> loc(#loc15)
// CHECK-NEXT:       sol.store %arg0, %0 : i256, !sol.ptr<i256, Stack> loc(#loc15)
// CHECK-NEXT:       %1 = sol.alloca : !sol.ptr<i256, Stack> loc(#loc16)
// CHECK-NEXT:       sol.store %arg1, %1 : i256, !sol.ptr<i256, Stack> loc(#loc16)
// CHECK-NEXT:       %2 = sol.addr_of @m1 : !sol.mapping<i256, !sol.mapping<i256, i256>> loc(#loc4)
// CHECK-NEXT:       %3 = sol.load %0 : !sol.ptr<i256, Stack>, i256 loc(#loc17)
// CHECK-NEXT:       %4 = sol.map %2, %3 : !sol.mapping<i256, !sol.mapping<i256, i256>>, !sol.mapping<i256, i256> loc(#loc18)
// CHECK-NEXT:       %5 = sol.load %1 : !sol.ptr<i256, Stack>, i256 loc(#loc19)
// CHECK-NEXT:       %6 = sol.map %4, %5 : !sol.mapping<i256, i256>, !sol.ptr<i256, Storage> loc(#loc18)
// CHECK-NEXT:       %7 = sol.load %6 : !sol.ptr<i256, Storage>, i256 loc(#loc18)
// CHECK-NEXT:       sol.return %7 : i256 loc(#loc20)
// CHECK-NEXT:     } loc(#loc14)
// CHECK-NEXT:     sol.func @f_s_58() -> !sol.string<Memory> attributes {state_mutability = #sol<StateMutability View>} {
// CHECK-NEXT:       %0 = sol.addr_of @s : !sol.string<Storage> loc(#loc5)
// CHECK-NEXT:       %1 = sol.data_loc_cast %0 : !sol.string<Storage>, !sol.string<Memory> loc(#loc5)
// CHECK-NEXT:       sol.return %1 : !sol.string<Memory> loc(#loc22)
// CHECK-NEXT:     } loc(#loc21)
// CHECK-NEXT:   } {interface_fns = [{selector = "0fd708de", sym = @f_s_58, type = () -> !sol.string<Memory>}, {selector = "c2bf6003", sym = @f_m0_34, type = (i256) -> i256}, {selector = "ea3b4c07", sym = @f_m1_50, type = (i256, i256) -> i256}, {selector = "eb80bd39", sym = @f_ui_22, type = () -> i256}], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc3 = loc({{.*}}:4:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:5:2)
// CHECK-NEXT: #loc5 = loc({{.*}}:6:2)
// CHECK-NEXT: #loc6 = loc({{.*}}:8:2)
// CHECK-NEXT: #loc7 = loc({{.*}}:8:54)
// CHECK-NEXT: #loc8 = loc({{.*}}:8:47)
// CHECK-NEXT: #loc9 = loc({{.*}}:9:2)
// CHECK-NEXT: #loc11 = loc({{.*}}:9:66)
// CHECK-NEXT: #loc12 = loc({{.*}}:9:63)
// CHECK-NEXT: #loc13 = loc({{.*}}:9:56)
// CHECK-NEXT: #loc14 = loc({{.*}}:10:2)
// CHECK-NEXT: #loc17 = loc({{.*}}:10:77)
// CHECK-NEXT: #loc18 = loc({{.*}}:10:74)
// CHECK-NEXT: #loc19 = loc({{.*}}:10:80)
// CHECK-NEXT: #loc20 = loc({{.*}}:10:67)
// CHECK-NEXT: #loc21 = loc({{.*}}:11:2)
// CHECK-NEXT: #loc22 = loc({{.*}}:11:55)
// CHECK-EMPTY:
