contract C {
    function f(uint constant LEN) public {
        uint[LEN] a;
    }
}
// ----
// DeclarationError 1788: (28-45): The "constant" keyword can only be used for state variables or variables at file level.
// TypeError 5462: (69-72): Invalid array length, expected integer literal or constant expression.
// TypeError 6651: (64-75): Data location must be "storage", "memory", "calldata" or "stack" for variable, but none was given.
