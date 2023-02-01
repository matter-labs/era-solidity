contract test {
    function f() public {
        uint[3.5] a; a;
    }
}
// ----
// TypeError 3208: (55-58): Array with fractional length specified.
// TypeError 6651: (50-61): Data location must be "storage", "memory", "calldata" or "stack" for variable, but none was given.
