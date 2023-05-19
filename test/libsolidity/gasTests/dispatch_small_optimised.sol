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
//   codeDepositCost: 78600
//   executionCost: 130
//   totalCost: 78730
// external:
//   fallback: 131
//   a(): 2317
//   b(uint256): 4635
//   f1(uint256): 46771
