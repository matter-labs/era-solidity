contract C {
    function f() public {
        assembly {
            pop(difficulty())
            pop(prevrandao())
        }
    }
}
// ====
// EVMVersion: =london
// ----
// Warning 5761: (104-114): "prevrandao" is not supported by the VM version and will be treated like "difficulty".
// Warning 2018: (17-133): Function state mutability can be restricted to view
