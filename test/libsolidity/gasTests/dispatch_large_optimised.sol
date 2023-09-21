contract Large {
    uint public a;
    uint[] public b;
    function f1(uint x) public returns (uint) { a = x; b[uint8(msg.data[0])] = x; }
    function f2(uint x) public returns (uint) { b[uint8(msg.data[1])] = x; }
    function f3(uint x) public returns (uint) { b[uint8(msg.data[2])] = x; }
    function f4(uint x) public returns (uint) { b[uint8(msg.data[3])] = x; }
    function f5(uint x) public returns (uint) { b[uint8(msg.data[4])] = x; }
    function f6(uint x) public returns (uint) { b[uint8(msg.data[5])] = x; }
    function f7(uint x) public returns (uint) { b[uint8(msg.data[6])] = x; }
    function f8(uint x) public returns (uint) { b[uint8(msg.data[7])] = x; }
    function f9(uint x) public returns (uint) { b[uint8(msg.data[8])] = x; }
    function f0(uint x) public pure returns (uint) { require(x > 10); }
    function g1(uint x) public payable returns (uint) { a = x; b[uint8(msg.data[0])] = x; }
    function g2(uint x) public payable returns (uint) { b[uint8(msg.data[1])] = x; }
    function g3(uint x) public payable returns (uint) { b[uint8(msg.data[2])] = x; }
    function g4(uint x) public payable returns (uint) { b[uint8(msg.data[3])] = x; }
    function g5(uint x) public payable returns (uint) { b[uint8(msg.data[4])] = x; }
    function g6(uint x) public payable returns (uint) { b[uint8(msg.data[5])] = x; }
    function g7(uint x) public payable returns (uint) { b[uint8(msg.data[6])] = x; }
    function g8(uint x) public payable returns (uint) { b[uint8(msg.data[7])] = x; }
    function g9(uint x) public payable returns (uint) { b[uint8(msg.data[8])] = x; }
    function g0(uint x) public payable returns (uint) { require(x > 10); }
}
// ====
// optimize: true
// optimize-runs: 2
// ----
// creation:
//   codeDepositCost: 670000
//   executionCost: 714
//   totalCost: 670714
// external:
//   a(): 2352
//   b(uint256): 5030
//   f0(uint256): 455
//   f1(uint256): 47160
//   f2(uint256): 25101
//   f3(uint256): 25189
//   f4(uint256): 25167
//   f5(uint256): 25145
//   f6(uint256): 25057
//   f7(uint256): 24837
//   f8(uint256): 24969
//   f9(uint256): 24991
//   g0(uint256): 695
//   g1(uint256): 46872
//   g2(uint256): 24835
//   g3(uint256): 24923
//   g4(uint256): 24901
//   g5(uint256): 24989
//   g6(uint256): 24769
//   g7(uint256): 24879
//   g8(uint256): 24857
//   g9(uint256): 24703
