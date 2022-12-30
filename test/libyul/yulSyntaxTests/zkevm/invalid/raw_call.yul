{
	$zk_raw_call(0xa, 0xb, 0xc, 0xd)
	$zk_raw_call_byref(0xa, 0xb, 0xc)
	$zk_static_raw_call(0xa, 0xb, 0xc, 0xd)
	$zk_static_raw_call_byref(0xa, 0xb, 0xc)
	$zk_delegate_raw_call(0xa, 0xb, 0xc, 0xd)
	$zk_delegate_raw_call_byref(0xa, 0xb, 0xc)
}
// ====
// dialect: evm
// ----
// TypeError 3083: (3-35): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (37-70): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (72-111): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (113-153): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (155-196): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (198-240): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
