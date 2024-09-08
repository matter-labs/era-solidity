{
	$zk_global_load("memory_pointer")
	$zk_global_store(0xa, 0xa)
	$zk_global_extra_abi_data(0xb)
}
// ====
// dialect: evm
// ----
// TypeError 3083: (3-36): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 5859: (55-58): Function expects string literal.
// TypeError 3083: (66-96): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
