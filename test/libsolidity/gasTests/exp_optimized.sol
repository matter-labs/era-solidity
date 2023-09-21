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
//   codeDepositCost: 90600
//   executionCost: 153
//   totalCost: 90753
// external:
//   exp_neg_one(uint256): 2036
//   exp_one(uint256): 1992
//   exp_two(uint256): 1970
//   exp_zero(uint256): 2014
