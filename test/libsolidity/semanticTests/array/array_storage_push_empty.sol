contract C {
    uint256[] storageArray;
    function pushEmpty(uint256 len) public {
        while(storageArray.length < len)
            storageArray.push();

        for (uint i = 0; i < len; i++)
            require(storageArray[i] == 0);
    }
}
// ====
// EVMVersion: >=petersburg
// ----
// pushEmpty(uint256): 128
// gas irOptimized: 408721
// gas legacy: 436383
// gas legacyOptimized: 429948
// pushEmpty(uint256): 256
// gas irOptimized: 697285
// gas legacy: 740691
// gas legacyOptimized: 727984
// pushEmpty(uint256): 38869 -> FAILURE # out-of-gas #
// gas irOptimized: 100000000
// gas legacy: 100000000
// gas legacyOptimized: 100000000
