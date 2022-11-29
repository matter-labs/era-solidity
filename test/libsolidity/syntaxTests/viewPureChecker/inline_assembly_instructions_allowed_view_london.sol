contract C {
    function f() public view {
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
