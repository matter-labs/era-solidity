contract C {
  struct S {
    uint a;
  }

  function f() external returns (uint) {
    S memory s = S(1);
    return s.a;
  }
}
// ----
// /// @use-src 0:""
// object "C_21" {
//     code {
//         /// @src 0:59:187  "contract C {..."
//         $zk_set_global("memory_pointer", memoryguard(128))
//         if callvalue()
//         {
//             revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb()
//         }
//         constructor_C_21()
//         let _1 := allocate_unbounded()
//         codecopy(_1, dataoffset("C_21_deployed"), datasize("C_21_deployed"))
//         return(_1, datasize("C_21_deployed"))
//         function allocate_unbounded() -> memPtr
//         {
//             memPtr := $zk_get_global("memory_pointer")
//         }
//         function revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb()
//         { revert(0, 0) }
//         function constructor_C_21()
//         { }
//     }
//     /// @use-src 0:""
//     object "C_21_deployed" {
//         code {
//             /// @src 0:59:187  "contract C {..."
//             $zk_set_global("memory_pointer", memoryguard(128))
//             if iszero(lt(calldatasize(), 4))
//             {
//                 let selector := shift_right_224_unsigned(calldataload(0))
//                 switch selector
//                 case 0x26121ff0 { external_fun_f_20() }
//                 default { }
//             }
//             revert_error_42b3090547df1d2001c96683413b8cf91c1b902ef5e3cb8d9f6f304cf7446f74()
//             function shift_right_224_unsigned(value) -> newValue
//             { newValue := shr(224, value) }
//             function allocate_unbounded() -> memPtr
//             {
//                 memPtr := $zk_get_global("memory_pointer")
//             }
//             function revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb()
//             { revert(0, 0) }
//             function revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b()
//             { revert(0, 0) }
//             function abi_decode_tuple_(headStart, dataEnd)
//             {
//                 if slt(sub(dataEnd, headStart), 0)
//                 {
//                     revert_error_dbdddcbe895c83990c08b3492a0e83918d802a52331272ac6fdb6a7c4aea3b1b()
//                 }
//             }
//             function cleanup_t_uint256(value) -> cleaned
//             { cleaned := value }
//             function abi_encode_t_uint256_to_t_uint256_fromStack(value, pos)
//             {
//                 mstore(pos, cleanup_t_uint256(value))
//             }
//             function abi_encode_tuple_t_uint256__to_t_uint256__fromStack(headStart, value0) -> tail
//             {
//                 tail := add(headStart, 32)
//                 abi_encode_t_uint256_to_t_uint256_fromStack(value0, add(headStart, 0))
//             }
//             function external_fun_f_20()
//             {
//                 if callvalue()
//                 {
//                     revert_error_ca66f745a3ce8ff40e2ccaf1ad45db7774001b90d25810abd9040049be7bf4bb()
//                 }
//                 abi_decode_tuple_(4, calldatasize())
//                 let ret_0 := fun_f_20()
//                 let memPos := allocate_unbounded()
//                 let memEnd := abi_encode_tuple_t_uint256__to_t_uint256__fromStack(memPos, ret_0)
//                 return(memPos, sub(memEnd, memPos))
//             }
//             function revert_error_42b3090547df1d2001c96683413b8cf91c1b902ef5e3cb8d9f6f304cf7446f74()
//             { revert(0, 0) }
//             function zero_value_for_split_t_uint256() -> ret
//             { ret := 0 }
//             function round_up_to_mul_of_32(value) -> result
//             {
//                 result := and(add(value, 31), not(31))
//             }
//             function panic_error_0x41()
//             {
//                 mstore(0, 35408467139433450592217433187231851964531694900788300625387963629091585785856)
//                 mstore(4, 0x41)
//                 revert(0, 0x24)
//             }
//             function finalize_allocation(memPtr, size)
//             {
//                 let newFreePtr := add(memPtr, round_up_to_mul_of_32(size))
//                 if or(gt(newFreePtr, 0xffffffffffffffff), lt(newFreePtr, memPtr)) { panic_error_0x41() }
//                 $zk_set_global("memory_pointer", newFreePtr)
//             }
//             function allocate_memory(size) -> memPtr
//             {
//                 memPtr := allocate_unbounded()
//                 finalize_allocation(memPtr, size)
//             }
//             function allocate_memory_struct_t_struct$_S_$4_storage_ptr() -> memPtr
//             { memPtr := allocate_memory(32) }
//             function cleanup_t_rational_1_by_1(value) -> cleaned
//             { cleaned := value }
//             function identity(value) -> ret
//             { ret := value }
//             function convert_t_rational_1_by_1_to_t_uint256(value) -> converted
//             {
//                 converted := cleanup_t_uint256(identity(cleanup_t_rational_1_by_1(value)))
//             }
//             function write_to_memory_t_uint256(memPtr, value)
//             {
//                 mstore(memPtr, cleanup_t_uint256(value))
//             }
//             function read_from_memoryt_uint256(ptr) -> returnValue
//             {
//                 let value := cleanup_t_uint256(mload(ptr))
//                 returnValue := value
//             }
//             /// @ast-id 20 @src 0:104:185  "function f() external returns (uint) {..."
//             function fun_f_20() -> var__7
//             {
//                 /// @src 0:135:139  "uint"
//                 let zero_t_uint256_1 := zero_value_for_split_t_uint256()
//                 var__7 := zero_t_uint256_1
//                 /// @src 0:162:163  "1"
//                 let expr_13 := 0x01
//                 /// @src 0:160:164  "S(1)"
//                 let expr_14_mpos := allocate_memory_struct_t_struct$_S_$4_storage_ptr()
//                 let _2 := convert_t_rational_1_by_1_to_t_uint256(expr_13)
//                 write_to_memory_t_uint256(add(expr_14_mpos, 0), _2)
//                 /// @src 0:147:164  "S memory s = S(1)"
//                 let var_s_11_mpos := expr_14_mpos
//                 /// @src 0:177:178  "s"
//                 let _3_mpos := var_s_11_mpos
//                 let expr_16_mpos := _3_mpos
//                 /// @src 0:177:180  "s.a"
//                 let _4 := add(expr_16_mpos, 0)
//                 let _5 := read_from_memoryt_uint256(_4)
//                 let expr_17 := _5
//                 /// @src 0:170:180  "return s.a"
//                 var__7 := expr_17
//                 leave
//             }
//         }
//         data ".metadata" hex""
//     }
// }
