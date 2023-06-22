contract Small {
    uint public a;
    uint[] public b;
    function f1(uint x) public returns (uint) { a = x; b[uint8(msg.data[0])] = x; }
    fallback () external payable {}
}
// ----
// creation:
//   codeDepositCost: 116200
//   executionCost: 166
//   totalCost: 116366
// external:
//   fallback: 131
//   a(): 2410
//   b(uint256): infinite
//   f1(uint256): infinite
