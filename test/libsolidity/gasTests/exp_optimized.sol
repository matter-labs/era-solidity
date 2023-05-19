pragma abicoder               v2;

contract C {
	function exp_neg_one(uint exponent) public returns(int) {
		unchecked { return (-1)**exponent; }
	}
	function exp_two(uint exponent) public returns(uint) {
		unchecked { return 2**exponent; }
	}
	function exp_zero(uint exponent) public returns(uint) {
		unchecked { return 0**exponent; }
	}
	function exp_one(uint exponent) public returns(uint) {
		unchecked { return 1**exponent; }
	}
}
// ====
// optimize: true
// optimize-yul: true
// ----
// creation:
//   codeDepositCost: 69400
//   executionCost: 118
//   totalCost: 69518
// external:
//   exp_neg_one(uint256): 1991
//   exp_one(uint256): 1944
//   exp_two(uint256): 1922
//   exp_zero(uint256): 1966
