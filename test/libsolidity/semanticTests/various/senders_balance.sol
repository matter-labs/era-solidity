contract C {
    function f() public view returns (uint256) {
        return msg.sender.balance;
    }
}


contract D {
    C c = new C();

    constructor() payable {}

    function f() public view returns (uint256) {
        return c.f();
    }
}

// ----
// constructor(), 27 wei ->
// gas irOptimized: 174854
// gas legacy: 237044
// gas legacyOptimized: 202163
// f() -> 27
