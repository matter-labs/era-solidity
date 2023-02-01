contract C {
    function f(mapping(uint => uint) storage) external pure {
    }
}
// ----
// TypeError 6651: (28-57): Data location must be "memory", "calldata" or "stack" for parameter in external function, but "storage" was given.
