{
	$zk_mimic_call(0xa, 0xb, 0xc)
	$zk_mimic_call_byref(0xa, 0xb)
}
// ====
// dialect: evm
// ----
// TypeError 3083: (3-32): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (34-64): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
