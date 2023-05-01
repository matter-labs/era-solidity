// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.0;

contract D {
  function d0() internal pure returns (uint) { return 0xd0; }

  function d1() internal pure returns (uint) { return 0xd1; }
  function getD1() internal pure returns (function() internal pure returns (uint)) {
    return d1;
  }

  function d2() internal pure returns (int) { return 0xd2; }
}

contract C is D {
  uint x;

  function() internal pure returns (uint) fp;
  function c0() internal pure returns (uint) { return 0xc0; }
  function c1() internal pure returns (uint) { return 0xc1; }
  function c2() internal pure returns (int) { return 0xc2; }

  constructor(int i) {
    if (i == 0) { fp = c0; }
    if (i == 1) { fp = c1; }
    if (i == 2) { fp = d0; }
    if (i == 3) { fp = getD1(); }
    x = fp();
  }

  function g() public view returns (uint) {
    return fp();
  }
}
