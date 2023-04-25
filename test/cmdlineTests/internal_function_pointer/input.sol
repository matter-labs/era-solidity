// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.0;

contract D {
  function f3() internal pure returns (uint) { return 0xa3; }
}

contract C is D {
  uint x;

  function() internal pure returns (uint) fp;
  function f0() internal pure returns (uint) { return 0xa0; }
  function f1() internal pure returns (uint) { return 0xa1; }
  function f2() internal pure returns (int) { return 0xb2; }

  constructor(int i) {
    if (i == 0) { fp = f0; }
    if (i == 1) { fp = f1; }
    if (i == 3) { fp = f3; }
    x = fp();
  }

  function g() public view returns (uint) {
    return fp();
  }
}
