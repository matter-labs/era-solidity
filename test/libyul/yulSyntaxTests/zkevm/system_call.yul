{
	let a := $zk_system_call(0xa, 0xb, 0xc, 0xd, 0xe, 0xf)
	let b := $zk_system_call_byref(0xa, 0xb, 0xc, 0xd, 0xe)
	let c := $zk_static_system_call(0xa, 0xb, 0xc, 0xd, 0xe, 0xf)
	let d := $zk_static_system_call_byref(0xa, 0xb, 0xc, 0xd, 0xe)
	let e := $zk_delegate_system_call(0xa, 0xb, 0xc, 0xd, 0xe, 0xf)
	let f := $zk_delegate_system_call_byref(0xa, 0xb, 0xc, 0xd, 0xe)
}
// ====
// dialect: evm
// ----
