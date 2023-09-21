contract A {
    uint x = 42;
    function f() public returns(uint256) {
        return x;
    }
}
contract B is A {
    uint public y = f();
}
// ====
// compileToEwasm: also
// ----
// constructor() ->
// gas irOptimized: 124398
// gas legacy: 140247
// gas legacyOptimized: 133131
// y() -> 42
