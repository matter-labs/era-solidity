contract c {
    uint[3][90][] large;
    uint[3][3][] small;
    function test() public returns (uint r) {
        for (uint i = 0; i < 7; i++) {
            large.push();
            small.push();
        }
        large[3][2][0] = 2;
        large[1] = large[3];
        small[3][2][0] = 2;
        small[1] = small[2];
        r = ((
            small[3][2][0] * 0x100 |
            small[1][2][0]) * 0x100 |
            large[3][2][0]) * 0x100 |
            large[1][2][0];
        delete small;
        delete large;

    }
    function clear() public returns (uint, uint) {
        for (uint i = 0; i < 7; i++) {
            large.push();
            small.push();
        }
        small[3][2][0] = 0;
        large[3][2][0] = 0;
        while (small.length > 0)
            small.pop();
        while (large.length > 0)
            large.pop();
        return (small.length, large.length);
    }
}
// ----
// test() -> 0x02000202
// gas irOptimized: 4559604
// gas legacy: 4481421
// gas legacyOptimized: 4480534
// storageEmpty -> 1
// clear() -> 0, 0
// gas irOptimized: 4485719
// gas legacy: 4411548
// gas legacyOptimized: 4411041
// storageEmpty -> 1
