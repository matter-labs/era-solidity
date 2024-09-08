{
	let a := $zk_load_calldata_into_active_ptr()
	let b := $zk_load_returndata_into_active_ptr()
}
// ====
// dialect: evm
// ----
// DeclarationError 3812: (3-47): Variable count mismatch for declaration of "a": 1 variables and 0 values.
// DeclarationError 3812: (49-95): Variable count mismatch for declaration of "b": 1 variables and 0 values.
