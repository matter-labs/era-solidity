contract C {
    uint[] storageArray;
    function test_indices(uint256 len) public
    {
        while (storageArray.length < len)
            storageArray.push();
        while (storageArray.length > len)
            storageArray.pop();
        for (uint i = 0; i < len; i++)
            storageArray[i] = i + 1;

        for (uint i = 0; i < len; i++)
            require(storageArray[i] == i + 1);
    }
}
// ----
// test_indices(uint256): 1 ->
// test_indices(uint256): 129 ->
// gas irOptimized: 3023549
// gas legacy: 3083310
// gas legacyOptimized: 3043673
// test_indices(uint256): 5 ->
// gas irOptimized: 578683
// gas legacy: 578111
// gas legacyOptimized: 576756
// test_indices(uint256): 10 ->
// gas irOptimized: 158556
// gas legacy: 163494
// gas legacyOptimized: 160271
// test_indices(uint256): 15 ->
// gas irOptimized: 173681
// gas legacy: 180969
// gas legacyOptimized: 176216
// test_indices(uint256): 0xFF ->
// gas irOptimized: 5685691
// gas legacy: 5803664
// gas legacyOptimized: 5725471
// test_indices(uint256): 1000 ->
// gas irOptimized: 18224158
// gas legacy: 18687736
// gas legacyOptimized: 18381573
// test_indices(uint256): 129 ->
// gas irOptimized: 4164368
// gas legacy: 4194284
// gas legacyOptimized: 4162575
// test_indices(uint256): 128 ->
// gas irOptimized: 414374
// gas legacy: 474790
// gas legacyOptimized: 435459
// test_indices(uint256): 1 ->
// gas irOptimized: 582198
// gas legacy: 580057
// gas legacyOptimized: 579682
