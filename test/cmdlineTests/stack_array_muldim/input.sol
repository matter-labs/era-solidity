pragma solidity >=0.0;
// SPDX-License-Identifier: GPL-3.0

contract C {
  function g(uint i) public pure returns (uint) {
    return i + 1;
  }

  function f() public pure returns (uint) {
    uint32[5][10] stack a;
    a[0][0] = 0xabc;
    a[1][0] = 0xdef;
    return g(a[1][0]);
  }
}
