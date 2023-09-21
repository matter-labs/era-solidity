contract C {
    uint public i;
    constructor(uint newI) {
        i = newI;
    }
}
contract D {
    C c;
    constructor(uint v) {
        c = new C(v);
    }
    function f() public returns (uint r) {
        return c.i();
    }
}
// ----
// constructor(): 2 ->
// gas irOptimized: 200034
// gas legacy: 260765
// gas legacyOptimized: 222027
// f() -> 2
