function f() view returns (uint256) {
    return block.prevrandao;
}
// ====
// EVMVersion: <paris
// ----
// Warning 9432: (49-65): "prevrandao" is not supported by the VM version and will be treated like "difficulty".
