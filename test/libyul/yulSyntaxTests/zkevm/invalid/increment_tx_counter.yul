{
	let a := $zk_increment_tx_counter()
}
// ====
// dialect: evm
// ----
// DeclarationError 3812: (3-38): Variable count mismatch for declaration of "a": 1 variables and 0 values.
