{
	let a := $zk_mimic_call(0xa, 0xb, 0xc)
	let b := $zk_mimic_call_byref(0xa, 0xb)
}
// ====
// dialect: evm
// ----
