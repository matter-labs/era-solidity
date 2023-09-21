contract c {
    function set(uint key) public returns (bool) { data[key] = msg.data; return true; }
    function copy(uint from, uint to) public returns (bool) { data[to] = data[from]; return true; }
    mapping(uint => bytes) data;
}
// ----
// set(uint256): 1, 2 -> true
// gas irOptimized: 110598
// gas legacy: 111392
// gas legacyOptimized: 110960
// set(uint256): 2, 2, 3, 4, 5 -> true
// gas irOptimized: 177549
// gas legacy: 178397
// gas legacyOptimized: 177965
// storageEmpty -> 0
// copy(uint256,uint256): 1, 2 -> true
// storageEmpty -> 0
// copy(uint256,uint256): 99, 1 -> true
// storageEmpty -> 0
// copy(uint256,uint256): 99, 2 -> true
// storageEmpty -> 1
