// Sending zero ether to a contract should still invoke the receive ether function
// (it previously did not because the gas stipend was not provided by the EVM)
contract Receiver {
    receive() external payable {}
}


contract Main {
    constructor() payable {}

    function s() public returns (bool) {
        Receiver r = new Receiver();
        return payable(r).send(0);
    }
}

// ====
// compileToEwasm: also
// ----
// constructor(), 20 wei ->
// gas irOptimized: 100492
// gas legacy: 124947
// gas legacyOptimized: 116985
// s() -> true
