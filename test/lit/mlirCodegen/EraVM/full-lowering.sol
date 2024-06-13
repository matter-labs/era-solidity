// RUN: solc --mlir-action=print-llvm-ir --mlir-target=eravm %s | FileCheck %s
// RUN: solc --mlir-action=print-asm --mlir-target=eravm %s | FileCheck --check-prefix=ASM %s

// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.0;

contract C {
	function f() public pure returns (uint) { return 42; }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK-NEXT: source_filename = "LLVMDialectModule"
// CHECK-EMPTY:
// CHECK-NEXT: @ptr_active = private global ptr addrspace(3) undef
// CHECK-NEXT: @ptr_return_data = private global ptr addrspace(3) undef
// CHECK-NEXT: @ptr_calldata = private global ptr addrspace(3) undef
// CHECK-NEXT: @extra_abi_data = private global [10 x i256] zeroinitializer, align 32
// CHECK-NEXT: @call_flags = private global i256 0, align 32
// CHECK-NEXT: @returndatasize = private global i256 0, align 32
// CHECK-NEXT: @calldatasize = private global i256 0, align 32
// CHECK-NEXT: @memory_pointer = private global i256 0, align 32
// CHECK-EMPTY:
// CHECK-NEXT: declare ptr @malloc(i64)
// CHECK-EMPTY:
// CHECK-NEXT: declare void @free(ptr)
// CHECK-EMPTY:
// CHECK-NEXT: declare void @__return(i256, i256, i256)
// CHECK-EMPTY:
// CHECK-NEXT: define private void @.unreachable() {
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: declare void @__revert(i256, i256, i256)
// CHECK-EMPTY:
// CHECK-NEXT: define private void @__deploy() {
// CHECK-NEXT:   store i256 128, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %1 = call i256 @llvm.eravm.getu128()
// CHECK-NEXT:   %2 = icmp ne i256 %1, 0
// CHECK-NEXT:   br i1 %2, label %3, label %4
// CHECK-EMPTY:
// CHECK-NEXT: 3:                                                ; preds = %0
// CHECK-NEXT:   call void @__revert(i256 0, i256 0, i256 2)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   br label %4
// CHECK-EMPTY:
// CHECK-NEXT: 4:                                                ; preds = %3, %0
// CHECK-NEXT:   %5 = load i256, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %6 = inttoptr i256 %5 to ptr addrspace(1)
// CHECK-NEXT:   %7 = load ptr addrspace(3), ptr @ptr_calldata, align 32
// CHECK-NEXT:   %8 = getelementptr i8, ptr addrspace(3) %7, i256 0
// CHECK-NEXT:   call void @llvm.memcpy.p1.p3.i256(ptr addrspace(1) %6, ptr addrspace(3) %8, i256 0, i1 false)
// CHECK-NEXT:   store i256 32, ptr addrspace(2) inttoptr (i256 256 to ptr addrspace(2)), align 1
// CHECK-NEXT:   store i256 0, ptr addrspace(2) inttoptr (i256 288 to ptr addrspace(2)), align 1
// CHECK-NEXT:   call void @__return(i256 256, i256 64, i256 2)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: define private void @__runtime() {
// CHECK-NEXT:   store i256 128, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %1 = load i256, ptr @calldatasize, align 32
// CHECK-NEXT:   %2 = icmp uge i256 %1, 4
// CHECK-NEXT:   br i1 %2, label %3, label %18
// CHECK-EMPTY:
// CHECK-NEXT: 3:                                                ; preds = %0
// CHECK-NEXT:   %4 = load ptr addrspace(3), ptr @ptr_calldata, align 32
// CHECK-NEXT:   %5 = getelementptr i8, ptr addrspace(3) %4, i256 0
// CHECK-NEXT:   %6 = load i256, ptr addrspace(3) %5, align 1
// CHECK-NEXT:   switch i256 %6, label %16 [
// CHECK-NEXT:     i256 638722032, label %7
// CHECK-NEXT:   ]
// CHECK-EMPTY:
// CHECK-NEXT: 7:                                                ; preds = %3
// CHECK-NEXT:   %8 = call i256 @llvm.eravm.getu128()
// CHECK-NEXT:   %9 = icmp ne i256 %8, 0
// CHECK-NEXT:   br i1 %9, label %10, label %11
// CHECK-EMPTY:
// CHECK-NEXT: 10:                                               ; preds = %7
// CHECK-NEXT:   call void @__revert(i256 0, i256 0, i256 0)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   br label %11
// CHECK-EMPTY:
// CHECK-NEXT: 11:                                               ; preds = %10, %7
// CHECK-NEXT:   %12 = call i256 @f_9()
// CHECK-NEXT:   %13 = load i256, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %14 = add i256 %13, 32
// CHECK-NEXT:   %15 = inttoptr i256 %13 to ptr addrspace(1)
// CHECK-NEXT:   store i256 %12, ptr addrspace(1) %15, align 1
// CHECK-NEXT:   call void @__return(i256 %13, i256 32, i256 0)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   br label %17
// CHECK-EMPTY:
// CHECK-NEXT: 16:                                               ; preds = %3
// CHECK-NEXT:   br label %17
// CHECK-EMPTY:
// CHECK-NEXT: 17:                                               ; preds = %16, %11
// CHECK-NEXT:   br label %18
// CHECK-EMPTY:
// CHECK-NEXT: 18:                                               ; preds = %17, %0
// CHECK-NEXT:   call void @__revert(i256 0, i256 0, i256 0)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: define i256 @__entry(ptr addrspace(3) %0, i256 %1, i256 %2, i256 %3, i256 %4, i256 %5, i256 %6, i256 %7, i256 %8, i256 %9, i256 %10, i256 %11) {
// CHECK-NEXT:   store i256 0, ptr @memory_pointer, align 32
// CHECK-NEXT:   store i256 0, ptr @calldatasize, align 32
// CHECK-NEXT:   store i256 0, ptr @returndatasize, align 32
// CHECK-NEXT:   store i256 0, ptr @call_flags, align 32
// CHECK-NEXT:   store <10 x i256> zeroinitializer, ptr @extra_abi_data, align 512
// CHECK-NEXT:   store ptr addrspace(3) %0, ptr @ptr_calldata, align 32
// CHECK-NEXT:   store i256 and (i256 lshr (i256 ptrtoint (ptr @ptr_calldata to i256), i256 96), i256 4294967295), ptr @calldatasize, align 32
// CHECK-NEXT:   %13 = load i256, ptr @calldatasize, align 32
// CHECK-NEXT:   %14 = getelementptr i8, ptr addrspace(3) %0, i256 %13
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr @ptr_return_data, align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr @ptr_active, align 32
// CHECK-NEXT:   store i256 %1, ptr @call_flags, align 32
// CHECK-NEXT:   store i256 %2, ptr @extra_abi_data, align 32
// CHECK-NEXT:   store i256 %3, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 1), align 32
// CHECK-NEXT:   store i256 %4, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 2), align 32
// CHECK-NEXT:   store i256 %5, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 3), align 32
// CHECK-NEXT:   store i256 %6, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 4), align 32
// CHECK-NEXT:   store i256 %7, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 5), align 32
// CHECK-NEXT:   store i256 %8, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 6), align 32
// CHECK-NEXT:   store i256 %9, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 7), align 32
// CHECK-NEXT:   store i256 %10, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 8), align 32
// CHECK-NEXT:   store i256 %11, ptr getelementptr inbounds ([10 x i256], ptr @extra_abi_data, i256 0, i256 9), align 32
// CHECK-NEXT:   %15 = and i256 %1, 1
// CHECK-NEXT:   %16 = icmp eq i256 %15, 1
// CHECK-NEXT:   br i1 %16, label %17, label %18
// CHECK-EMPTY:
// CHECK-NEXT: 17:                                               ; preds = %12
// CHECK-NEXT:   call void @__deploy()
// CHECK-NEXT:   br label %19
// CHECK-EMPTY:
// CHECK-NEXT: 18:                                               ; preds = %12
// CHECK-NEXT:   call void @__runtime()
// CHECK-NEXT:   br label %19
// CHECK-EMPTY:
// CHECK-NEXT: 19:                                               ; preds = %17, %18
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: define private i256 @f_9.0() {
// CHECK-NEXT:   ret i256 42
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: define private i256 @f_9() {
// CHECK-NEXT:   ret i256 42
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nounwind willreturn memory(none)
// CHECK-NEXT: declare i256 @llvm.eravm.getu128() #0
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK-NEXT: declare void @llvm.memcpy.p1.p3.i256(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i256, i1 immarg) #1
// CHECK-EMPTY:
// CHECK-NEXT: attributes #0 = { nounwind willreturn memory(none) }
// CHECK-NEXT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK-EMPTY:
// CHECK-NEXT: !llvm.module.flags = !{!0}
// CHECK-EMPTY:
// CHECK-NEXT: !0 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK-EMPTY:
// ASM: 	.text
// ASM-NEXT: 	.file	{{.*}}
// ASM-NEXT: .unreachable:
// ASM-EMPTY:
// ASM-NEXT: __deploy:
// ASM-NEXT: 	add	128, r0, r1
// ASM-NEXT: 	st.1	64, r1
// ASM-NEXT: 	context.get_context_u128	r1
// ASM-NEXT: 	sub!	r1, r0, r1
// ASM-NEXT: 	jump.eq	@.BB1_2
// ASM-NEXT: 	add	2, r0, r3
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: 	add	r0, r0, r2
// ASM-NEXT: 	near_call	r0, @__revert, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-NEXT: .BB1_2:
// ASM-NEXT: 	add	32, r0, r1
// ASM-NEXT: 	st.2	256, r1
// ASM-NEXT: 	st.2	288, r0
// ASM-NEXT: 	add	256, r0, r1
// ASM-NEXT: 	add	64, r0, r2
// ASM-NEXT: 	add	2, r0, r3
// ASM-NEXT: 	near_call	r0, @__return, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-EMPTY:
// ASM-NEXT: __runtime:
// ASM-NEXT: 	add	128, r0, r1
// ASM-NEXT: 	st.1	64, r1
// ASM-NEXT: 	add	stack[@calldatasize], r0, r1
// ASM-NEXT: 	sub.s!	4, r1, r1
// ASM-NEXT: 	jump.lt	@.BB2_5
// ASM-NEXT: 	ptr.add	stack[@ptr_calldata], r0, r1
// ASM-NEXT: 	ld	r1, r1
// ASM-NEXT: 	sub.s!	@CPI2_0[0], r1, r1
// ASM-NEXT: 	jump.ne	@.BB2_5
// ASM-NEXT: 	context.get_context_u128	r1
// ASM-NEXT: 	sub!	r1, r0, r1
// ASM-NEXT: 	jump.eq	@.BB2_4
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: 	add	r0, r0, r2
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	near_call	r0, @__revert, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-NEXT: .BB2_4:
// ASM-NEXT: 	near_call	r0, @f_9, @DEFAULT_UNWIND
// ASM-NEXT: 	ld.1	64, r3
// ASM-NEXT: 	st.1	r3, r1
// ASM-NEXT: 	add	32, r0, r2
// ASM-NEXT: 	add	r3, r0, r1
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	near_call	r0, @__return, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-NEXT: .BB2_5:
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: 	add	r0, r0, r2
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	near_call	r0, @__revert, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-EMPTY:
// ASM-NEXT: 	.globl	__entry
// ASM-NEXT: __entry:
// ASM-NEXT: 	add	stack[0], r0, r13
// ASM-NEXT: 	shr.s	96, r13, r13
// ASM-NEXT: 	and	@CPI3_0[0], r13, r14
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_return_data]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active]
// ASM-NEXT: 	add	r3, r0, stack[@extra_abi_data]
// ASM-NEXT: 	add	r4, r0, stack[@extra_abi_data+1]
// ASM-NEXT: 	add	r5, r0, stack[@extra_abi_data+2]
// ASM-NEXT: 	add	r6, r0, stack[@extra_abi_data+3]
// ASM-NEXT: 	add	r7, r0, stack[@extra_abi_data+4]
// ASM-NEXT: 	add	r8, r0, stack[@extra_abi_data+5]
// ASM-NEXT: 	add	r9, r0, stack[@extra_abi_data+6]
// ASM-NEXT: 	add	r10, r0, stack[@extra_abi_data+7]
// ASM-NEXT: 	add	r11, r0, stack[@extra_abi_data+8]
// ASM-NEXT: 	add	r12, r0, stack[@extra_abi_data+9]
// ASM-NEXT: 	ptr.add	r1, r0, stack[@ptr_calldata]
// ASM-NEXT: 	and	@CPI3_0[0], r13, stack[@calldatasize]
// ASM-NEXT: 	add	r2, r0, stack[@call_flags]
// ASM-NEXT: 	add	0, r0, stack[@memory_pointer]
// ASM-NEXT: 	add	0, r0, stack[@returndatasize]
// ASM-NEXT: 	and!	1, r2, r1
// ASM-NEXT: 	jump.eq	@.BB3_2
// ASM-NEXT: 	near_call	r0, @__deploy, @DEFAULT_UNWIND
// ASM-NEXT: .BB3_2:
// ASM-NEXT: 	near_call	r0, @__runtime, @DEFAULT_UNWIND
// ASM-EMPTY:
// ASM-NEXT: f_9:
// ASM-NEXT: 	add	42, r0, r1
// ASM-NEXT: 	ret
// ASM-EMPTY:
// ASM-NEXT: 	.data
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: ptr_active:
// ASM-NEXT: 	.zero	32
// ASM-EMPTY:
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: ptr_return_data:
// ASM-NEXT: 	.zero	32
// ASM-EMPTY:
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: ptr_calldata:
// ASM-NEXT: 	.zero	32
// ASM-EMPTY:
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: extra_abi_data:
// ASM-NEXT: 	.zero	320
// ASM-EMPTY:
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: call_flags:
// ASM-NEXT: 	.cell	0
// ASM-EMPTY:
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: returndatasize:
// ASM-NEXT: 	.cell	0
// ASM-EMPTY:
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: calldatasize:
// ASM-NEXT: 	.cell	0
// ASM-EMPTY:
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: memory_pointer:
// ASM-NEXT: 	.cell	0
// ASM-EMPTY:
// ASM-NEXT: 	.note.GNU-stack
// ASM-NEXT: 	.rodata
// ASM-NEXT: CPI2_0:
// ASM-NEXT: 	.cell	638722032
// ASM-NEXT: CPI3_0:
// ASM-NEXT: 	.cell	4294967295
// ASM-EMPTY: