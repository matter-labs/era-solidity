contract C {
    function f() public pure {
        assembly {
            pop(difficulty())
            pop(prevrandao())
        }
    }
}
// ====
// EVMVersion: =london
// ----
// Warning 5761: (109-119): "prevrandao" is not supported by the VM version and will be treated like "difficulty".
// TypeError 2527: (79-91): Function declared as pure, but this expression (potentially) reads from the environment or state and thus requires "view".
// TypeError 2527: (109-121): Function declared as pure, but this expression (potentially) reads from the environment or state and thus requires "view".
