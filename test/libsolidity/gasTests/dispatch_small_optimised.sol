contract Small {
    uint public a;
    uint[] public b;
    function f1(uint x) public returns (uint) { a = x; b[uint8(msg.data[0])] = x; }
    fallback () external payable {}
}
// ====
// optimize: true
// optimize-runs: 2
// ----
// creation:
//   codeDepositCost: 92200
//   executionCost: 153
//   totalCost: 92353
// external:
//   fallback: 131
//   a(): 2330
//   b(uint256): 4678
//   f1(uint256): 46874
