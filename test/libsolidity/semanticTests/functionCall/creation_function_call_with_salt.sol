contract C {
    uint public i;
    constructor(uint newI) {
        i = newI;
    }
}
contract D {
    C c;
    constructor(uint v) {
        c = new C{salt: "abc"}(v);
    }
    function f() public returns (uint r) {
        return c.i();
    }
}
// ====
// EVMVersion: >=constantinople
// ----
// constructor(): 2 ->
// gas irOptimized: 200159
// gas legacy: 261161
// gas legacyOptimized: 222387
// f() -> 2
