// RUN: sol-opt %s | sol-opt | FileCheck %s

module {
  sol.contract @C {
    sol.state_var @m : i256
    sol.func @f() {
      %ptr = sol.addr_of @m : !sol.ptr<i256, Storage>
      sol.return
    }
  } {interface_fns = [], kind = #sol<ContractKind Contract>}
}

// CHECK: module {
// CHECK-NEXT:   sol.contract @C {
// CHECK-NEXT:     sol.state_var @m : i256
// CHECK-NEXT:     sol.func @f() {
// CHECK-NEXT:       %0 = sol.addr_of @m : <i256, Storage>
// CHECK-NEXT:       sol.return
// CHECK-NEXT:     }
// CHECK-NEXT:   } {interface_fns = [], kind = #sol<ContractKind Contract>}
// CHECK-NEXT: }
