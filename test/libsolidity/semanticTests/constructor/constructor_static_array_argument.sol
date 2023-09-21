contract C {
    uint256 public a;
    uint256[3] public b;

    constructor(uint256 _a, uint256[3] memory _b) {
        a = _a;
        b = _b;
    }
}

// ----
// constructor(): 1, 2, 3, 4 ->
// gas irOptimized: 177536
// gas legacy: 233435
// gas legacyOptimized: 196404
// a() -> 1
// b(uint256): 0 -> 2
// b(uint256): 1 -> 3
// b(uint256): 2 -> 4
