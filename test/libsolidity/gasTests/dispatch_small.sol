contract Small {
    uint public a;
    uint[] public b;
    function f1(uint x) public returns (uint) { a = x; b[uint8(msg.data[0])] = x; }
    fallback () external payable {}
}
// ----
// creation:
//   codeDepositCost: 123000
//   executionCost: 183
//   totalCost: 123183
// external:
//   fallback: 131
//   a(): 2415
//   b(uint256): infinite
//   f1(uint256): infinite
