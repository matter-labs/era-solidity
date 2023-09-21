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
//   codeDepositCost: 45600
//   executionCost: 111
//   totalCost: 45711
// external:
//   readX(): 2394
//   resetX(): 5148
//   setX(uint256): 22379
