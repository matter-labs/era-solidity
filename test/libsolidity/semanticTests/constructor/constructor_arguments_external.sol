contract Main {
    bytes3 name;
    bool flag;

    constructor(bytes3 x, bool f) {
        name = x;
        flag = f;
    }

    function getName() public returns (bytes3 ret) {
        return name;
    }

    function getFlag() public returns (bool ret) {
        return flag;
    }
}
// ----
// constructor(): "abc", true
// gas irOptimized: 106546
// gas legacy: 148283
// gas legacyOptimized: 104017
// getFlag() -> true
// getName() -> "abc"
