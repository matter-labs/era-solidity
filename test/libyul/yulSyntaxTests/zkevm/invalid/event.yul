{
	$zk_event_initialize(0xa)
	$zk_event_write(0xa)
}
// ====
// dialect: evm
// ----
// TypeError 7000: (3-23): Function "$zk_event_initialize" expects 2 arguments but got 1.
// TypeError 7000: (30-45): Function "$zk_event_write" expects 2 arguments but got 1.
