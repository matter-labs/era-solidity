{
	let a := $zk_get_global("memory_pointer")
	$zk_set_global("memory_pointer", 0xa)
}
// ====
// dialect: evm
// ----
