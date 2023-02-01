/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0

#include <libyul/backends/evm/ZKEVMIntrinsics.h>

using namespace solidity::zkevm;

// clang-format off
const std::array<IntrInfo, 38> solidity::zkevm::intrInfos{{
//	 Name											Args	Rets	SideEffect	Literals
	{"$zk_to_l1",									3,		0,		true},
	{"$zk_code_source",								0,		1,		false},
	{"$zk_precompile",								2,		1,		true},
	{"$zk_meta",									0,		1,		false},
	{"$zk_mimic_call",								3,		1,		true},
	{"$zk_system_mimic_call",						5,		1,		true},
	{"$zk_mimic_call_byref",						2,		1,		true},
	{"$zk_system_mimic_call_byref",					4,		1,		true},
	{"$zk_raw_call",								4,		1,		true},
	{"$zk_raw_call_byref",							3,		1,		true},
	{"$zk_system_call",								6,		1,		true},
	{"$zk_system_call_byref",						5,		1,		true},
	{"$zk_static_raw_call",							4,		1,		true},
	{"$zk_static_raw_call_byref",					3,		1,		true},
	{"$zk_static_system_call",						6,		1,		true},
	{"$zk_static_system_call_byref",				5,		1,		true},
	{"$zk_delegate_raw_call",						4,		1,		true},
	{"$zk_delegate_raw_call_byref",					3,		1,		true},
	{"$zk_delegate_system_call",					6,		1,		true},
	{"$zk_delegate_system_call_byref",				5,		1,		true},
	{"$zk_set_context_u128",						1,		0,		true},
	{"$zk_set_pubdata_price",						1,		0,		true},
	{"$zk_increment_tx_counter",					0,		0,		true},
	{"$zk_event_initialize",						2,		0,		true},
	{"$zk_event_write",								2,		0,		true},
	{"$zk_load_calldata_into_active_ptr",			0,		0,		true},
	{"$zk_load_returndata_into_active_ptr",			0,		0,		true},
	{"$zk_ptr_add_into_active",						1,		0,		true},
	{"$zk_ptr_shrink_into_active",					1,		0,		true},
	{"$zk_ptr_pack_into_active",					1,		0,		true},
	{"$zk_multiplication_high",						2,		1,		false},
	{"$zk_global_load",								1,		1,		false,		{yul::LiteralKind::String}},
	{"$zk_global_store",							2,		0,		true,		{yul::LiteralKind::String, std::nullopt}},
	{"$zk_global_extra_abi_data",					1,		1,		false},
	{"$zk_stack_alloc",								1,		1,		true},
	{"$zk_stack_load",								1,		1,		false},
	{"$zk_stack_store",								2,		0,		true},
	{"$zk_copy_calldata_to_stack",					3,		0,		true},
}};
// clang-format on
