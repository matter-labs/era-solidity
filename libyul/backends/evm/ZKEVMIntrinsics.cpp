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

#include <libyul/ASTForward.h>
#include <libyul/backends/evm/AbstractAssembly.h>
#include <libyul/backends/evm/EVMDialect.h>
#include <libyul/backends/evm/ZKEVMIntrinsics.h>

#include <libevmasm/Instruction.h>

#include <liblangutil/Exceptions.h>

using namespace solidity::evmasm;
using namespace solidity::yul;
using namespace solidity::zkevm;

/// Generates equivalent evmasm for $zk_global_load
static void genGlobalLoad(FunctionCall const& _call, AbstractAssembly& _assembly, BuiltinContext&)
{
	Expression const& arg = _call.arguments.front();
	if (std::get<Literal>(arg).value.str() == "memory_pointer")
	{
		_assembly.appendConstant(64);
		_assembly.appendInstruction(Instruction::MLOAD);
	}
	else
	{
		solUnimplemented("Undefined $zk_global_load argument");
	}
}

/// Generates equivalent evmasm for $zk_global_store
static void genGlobalStore(FunctionCall const& _call, AbstractAssembly& _assembly, BuiltinContext&)
{
	Expression const& arg = _call.arguments.front();
	if (std::get<Literal>(arg).value.str() == "memory_pointer")
	{
		_assembly.appendConstant(64);
		_assembly.appendInstruction(Instruction::MSTORE);
	}
	else
	{
		solUnimplemented("Undefined $zk_global_store argument");
	}
}

/// Literal arguments in $zk_global_{load|store}
static IntrInfo::LiteralKinds globalLoadLit = {LiteralKind::String};
static IntrInfo::LiteralKinds globalStoreLit = {LiteralKind::String, std::nullopt};

// clang-format off
const std::array<IntrInfo, 34> solidity::zkevm::intrInfos{{
//	 Name											Args	Rets	SideEffect	Literals			GenCode
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
	{"$zk_global_load",								1,		1,		false,		globalLoadLit,		genGlobalLoad},
	{"$zk_global_store",							2,		0,		true,		globalStoreLit,		genGlobalStore},
	{"$zk_global_extra_abi_data",					1,		1,		false}
}};
// clang-format on
