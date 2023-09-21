// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.6.0;

contract Test {
  function complexOperation() public view returns (uint result) {
    for(uint i = 0; i < 10; i++) {
      try this.externalCall() returns (uint externalResult) {
        result += externalResult;
      } catch {}
    }
  }

  function externalCall() public pure returns (uint result) {
    result = 1;
  }
}
