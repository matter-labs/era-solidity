contract C {
	mapping(uint => uint) transient y;
}
// ====
// EVMVersion: >=cancun
// ----
// UnimplementedFeatureError 1834: Transient data location is only supported for value types.