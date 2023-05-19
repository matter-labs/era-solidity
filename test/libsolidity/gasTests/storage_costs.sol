contract C {
    uint x;
    function setX(uint y) public {
        x = y;
    }
    function resetX() public {
        x = 0;
    }
    function readX() public view returns(uint) {
        return x;
    }
}
// ====
// optimize: true
// optimize-yul: true
// ----
// creation:
//   codeDepositCost: 36200
//   executionCost: 88
//   totalCost: 36288
// external:
//   readX(): 2362
//   resetX(): 5129
//   setX(uint256): 22339
