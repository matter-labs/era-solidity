// RUN: solc --mlir-action=print-init --mmlir --mlir-print-debuginfo %s | FileCheck %s

contract C {
  function inp_ui(uint a) public returns (uint) { return internal_inp_ui(a); }
  function internal_inp_ui(uint a) internal returns (uint) { return a; }
  function out_ui() public returns (uint) { return 42; }
  function out_bool() public returns (bool) { return true; }

  function inp_string(string memory a) public { internal_inp_string(a); }
  function internal_inp_string(string memory a) internal {}
  function out_string() public returns (string memory) {
    string memory a;
    return a;
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: #loc3 = loc({{.*}}:3:18)
// CHECK-NEXT: #loc8 = loc({{.*}}:4:27)
// CHECK-NEXT: #loc18 = loc({{.*}}:8:22)
// CHECK-NEXT: #loc22 = loc({{.*}}:9:31)
// CHECK-NEXT: module {
// CHECK-NEXT:   sol.contract @C_66 {
// CHECK-NEXT:     sol.func @inp_ui_12(%arg0: ui256 loc({{.*}}:3:18)) -> ui256 attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc3)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc3)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc4)
// CHECK-NEXT:       %2 = sol.call @internal_inp_ui_22(%1) : (ui256) -> ui256 loc(#loc5)
// CHECK-NEXT:       sol.return %2 : ui256 loc(#loc6)
// CHECK-NEXT:     } loc(#loc2)
// CHECK-NEXT:     sol.func @internal_inp_ui_22(%arg0: ui256 loc({{.*}}:4:27)) -> ui256 attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<ui256, Stack> loc(#loc8)
// CHECK-NEXT:       sol.store %arg0, %0 : ui256, !sol.ptr<ui256, Stack> loc(#loc8)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<ui256, Stack>, ui256 loc(#loc9)
// CHECK-NEXT:       sol.return %1 : ui256 loc(#loc10)
// CHECK-NEXT:     } loc(#loc7)
// CHECK-NEXT:     sol.func @out_ui_30() -> ui256 attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %c42_ui8 = sol.constant 42 : ui8 loc(#loc12)
// CHECK-NEXT:       %0 = sol.ext %c42_ui8 : ui8 to ui256 loc(#loc12)
// CHECK-NEXT:       sol.return %0 : ui256 loc(#loc13)
// CHECK-NEXT:     } loc(#loc11)
// CHECK-NEXT:     sol.func @out_bool_38() -> i1 attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %true = sol.constant true loc(#loc15)
// CHECK-NEXT:       sol.return %true : i1 loc(#loc16)
// CHECK-NEXT:     } loc(#loc14)
// CHECK-NEXT:     sol.func @inp_string_48(%arg0: !sol.string<Memory> loc({{.*}}:8:22)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc18)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc18)
// CHECK-NEXT:       %1 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc19)
// CHECK-NEXT:       sol.call @internal_inp_string_54(%1) : (!sol.string<Memory>) -> () loc(#loc20)
// CHECK-NEXT:       sol.return loc(#loc17)
// CHECK-NEXT:     } loc(#loc17)
// CHECK-NEXT:     sol.func @internal_inp_string_54(%arg0: !sol.string<Memory> loc({{.*}}:9:31)) attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc22)
// CHECK-NEXT:       sol.store %arg0, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc22)
// CHECK-NEXT:       sol.return loc(#loc21)
// CHECK-NEXT:     } loc(#loc21)
// CHECK-NEXT:     sol.func @out_string_65() -> !sol.string<Memory> attributes {state_mutability = #sol<StateMutability NonPayable>} {
// CHECK-NEXT:       %0 = sol.alloca : !sol.ptr<!sol.string<Memory>, Stack> loc(#loc24)
// CHECK-NEXT:       %1 = sol.malloc : !sol.string<Memory> loc(#loc24)
// CHECK-NEXT:       sol.store %1, %0 : !sol.string<Memory>, !sol.ptr<!sol.string<Memory>, Stack> loc(#loc24)
// CHECK-NEXT:       %2 = sol.load %0 : !sol.ptr<!sol.string<Memory>, Stack>, !sol.string<Memory> loc(#loc25)
// CHECK-NEXT:       sol.return %2 : !sol.string<Memory> loc(#loc26)
// CHECK-NEXT:     } loc(#loc23)
// CHECK-NEXT:   } {interface_fns = [{selector = 1891873197 : i32, sym = @out_bool_38, type = () -> i1}, {selector = 1913244289 : i32, sym = @inp_string_48, type = (!sol.string<Memory>) -> ()}, {selector = 2072703138 : i32, sym = @out_string_65, type = () -> !sol.string<Memory>}, {selector = -1856136386 : i32, sym = @inp_ui_12, type = (ui256) -> ui256}, {selector = -597758925 : i32, sym = @out_ui_30, type = () -> ui256}], kind = #sol<ContractKind Contract>} loc(#loc1)
// CHECK-NEXT: } loc(#loc)
// CHECK-NEXT: #loc = loc(unknown)
// CHECK-NEXT: #loc1 = loc({{.*}}:2:0)
// CHECK-NEXT: #loc2 = loc({{.*}}:3:2)
// CHECK-NEXT: #loc4 = loc({{.*}}:3:73)
// CHECK-NEXT: #loc5 = loc({{.*}}:3:57)
// CHECK-NEXT: #loc6 = loc({{.*}}:3:50)
// CHECK-NEXT: #loc7 = loc({{.*}}:4:2)
// CHECK-NEXT: #loc9 = loc({{.*}}:4:68)
// CHECK-NEXT: #loc10 = loc({{.*}}:4:61)
// CHECK-NEXT: #loc11 = loc({{.*}}:5:2)
// CHECK-NEXT: #loc12 = loc({{.*}}:5:51)
// CHECK-NEXT: #loc13 = loc({{.*}}:5:44)
// CHECK-NEXT: #loc14 = loc({{.*}}:6:2)
// CHECK-NEXT: #loc15 = loc({{.*}}:6:53)
// CHECK-NEXT: #loc16 = loc({{.*}}:6:46)
// CHECK-NEXT: #loc17 = loc({{.*}}:8:2)
// CHECK-NEXT: #loc19 = loc({{.*}}:8:68)
// CHECK-NEXT: #loc20 = loc({{.*}}:8:48)
// CHECK-NEXT: #loc21 = loc({{.*}}:9:2)
// CHECK-NEXT: #loc23 = loc({{.*}}:10:2)
// CHECK-NEXT: #loc24 = loc({{.*}}:11:4)
// CHECK-NEXT: #loc25 = loc({{.*}}:12:11)
// CHECK-NEXT: #loc26 = loc({{.*}}:12:4)
// CHECK-EMPTY:
