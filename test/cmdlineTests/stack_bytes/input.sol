pragma solidity >=0.0;
// SPDX-License-Identifier: GPL-3.0

contract C {
	function h(bytes1 i) public pure returns (bytes1) {
		return i & '0';
	}

	function g(bytes1[10] stack a, uint b, bytes1 c) public pure {
		for (uint i = b; i < 10; ++i) {
			if (a[i] == '0')
				a[i] = c;
		}
	}

	function f(uint i) public pure returns (bytes1) {
		bytes1[10] stack a;
		a[0] = 'a';
		a[1] = 'b';
		a[2] = '0';
		g(a, i, 'c');
		return h(a[2]);
	}
}
