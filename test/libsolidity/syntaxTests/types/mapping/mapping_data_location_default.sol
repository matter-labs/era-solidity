contract c {
    mapping(uint => uint) y;
    function f() view public {
        mapping(uint => uint) x = y;
        x;
    }
}
// ----
// TypeError 6651: (81-104): Data location must be "storage", "memory", "calldata" or "stack" for variable, but none was given.
