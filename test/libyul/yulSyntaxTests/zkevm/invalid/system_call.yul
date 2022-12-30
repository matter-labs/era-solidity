{
	$zk_system_call(0xa, 0xb, 0xc, 0xd, 0xe, 0xf)
	$zk_system_call_byref(0xa, 0xb, 0xc, 0xd, 0xe)
	$zk_static_system_call(0xa, 0xb, 0xc, 0xd, 0xe, 0xf)
	$zk_static_system_call_byref(0xa, 0xb, 0xc, 0xd, 0xe)
	$zk_delegate_system_call(0xa, 0xb, 0xc, 0xd, 0xe, 0xf)
	$zk_delegate_system_call_byref(0xa, 0xb, 0xc, 0xd, 0xe)
}
// ====
// dialect: evm
// ----
// TypeError 3083: (3-48): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (50-96): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (98-150): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (152-205): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (207-261): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
// TypeError 3083: (263-318): Top-level expressions are not supposed to return values (this expression returns 1 value). Use ``pop()`` or assign them.
