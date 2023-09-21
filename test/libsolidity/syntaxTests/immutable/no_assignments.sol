contract C {
    uint immutable x;
    constructor() {
        x = 0;
        while (true)
        {}
    }
    function f() external view returns(uint) { return x; }
}
// ====
// optimize-yul: true
// ----
