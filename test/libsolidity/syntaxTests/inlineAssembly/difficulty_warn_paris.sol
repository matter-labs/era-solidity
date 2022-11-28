function f() view returns (uint256) {
    return block.difficulty;
}
// ====
// EVMVersion: >=paris
// ----
//Warning 8417: (49-65): "difficulty" was replaced by "prevrandao" in the VM version paris and does not behave as before. It now always returns 0.
