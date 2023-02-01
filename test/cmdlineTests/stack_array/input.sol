pragma solidity >=0.0;
// SPDX-License-Identifier: GPL-3.0

contract C {
	function h(uint i) public pure returns (uint) {
		return i + 1;
	}

	function g(uint[10] stack a, uint b, uint c) public pure {
		for (uint i = b; i < 10; ++i) {
			if (a[i] == 0)
				a[i] = c;
		}
	}

	function f(uint i) public pure returns (uint) {
		uint[10] stack a;
		a[0] = 0xa0;
		a[1] = 0xa1;
		a[2] = 0;
		g(a, i, 0xa2);
		return h(a[2]);
	}
}
