{
	let a := $zk_global_load("memory_pointer")
	$zk_global_store("memory_pointer", 0xa)
	let b := $zk_global_extra_abi_data(0xb)
}
// ====
// dialect: evm
// ----
