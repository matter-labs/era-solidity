{
	$zk_system_mimic_call(0xa, 0xb, 0xc, 0xd, 0xe)
	$zk_system_mimic_call_byref(0xa, 0xb, 0xc, 0xd)
}
// ====
// dialect: evm
// ----
// TypeError 3083: (3-49): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (51-98): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
