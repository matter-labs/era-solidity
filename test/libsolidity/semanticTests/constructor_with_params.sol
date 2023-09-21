contract C {
    uint public i;
    uint public k;

    constructor(uint newI, uint newK) {
        i = newI;
        k = newK;
    }
}
// ----
// constructor(): 2, 0 ->
// gas irOptimized: 106435
// gas legacy: 121689
// gas legacyOptimized: 112682
// i() -> 2
// k() -> 0
