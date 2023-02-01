struct S { mapping(uint => uint)[2] a; }
contract C {
    function f(S storage s) public {}
}
// ----
// TypeError 6651: (69-80): Data location must be "memory", "calldata" or "stack" for parameter in function, but "storage" was given.
