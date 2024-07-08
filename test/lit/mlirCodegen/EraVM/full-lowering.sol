// RUN: solc --mlir-action=print-llvm-ir --mlir-target=eravm %s | FileCheck %s
// RUN: solc --mlir-action=print-asm --mlir-target=eravm %s | FileCheck --check-prefix=ASM %s

contract C {
  string m;
  function f0() public pure returns (uint) { return 42; }
  function f1() public view returns (string memory) {
    return m;
  }
}

// NOTE: Assertions have been autogenerated by test/updFileCheckTest.py
// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK-NEXT: source_filename = "LLVMDialectModule"
// CHECK-NEXT: target datalayout = "E-p:256:256-i256:256:256-S32-a:256:256"
// CHECK-NEXT: target triple = "eravm-unknown-unknown"
// CHECK-EMPTY:
// CHECK-NEXT: @ptr_decommit = private global ptr addrspace(3) undef
// CHECK-NEXT: @ptr_return_data = private global ptr addrspace(3) undef
// CHECK-NEXT: @ptr_calldata = private global ptr addrspace(3) undef
// CHECK-NEXT: @ptr_active = private global [16 x ptr addrspace(3)] undef
// CHECK-NEXT: @extra_abi_data = private global [10 x i256] zeroinitializer
// CHECK-NEXT: @call_flags = private global i256 0
// CHECK-NEXT: @returndatasize = private global i256 0
// CHECK-NEXT: @calldatasize = private global i256 0
// CHECK-NEXT: @memory_pointer = private global i256 0
// CHECK-EMPTY:
// CHECK-NEXT: declare ptr @malloc(i64)
// CHECK-EMPTY:
// CHECK-NEXT: declare void @free(ptr)
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: declare void @__return(i256, i256, i256) #0
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: define private void @.unreachable() #0 personality ptr @__personality {
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: declare void @__revert(i256, i256, i256) #0
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: declare i256 @__sha3(ptr addrspace(1), i256, i1) #0
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: define private void @__deploy() #0 personality ptr @__personality {
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
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: define private void @__runtime() #0 personality ptr @__personality {
// CHECK-NEXT:   store i256 128, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %1 = load i256, ptr @calldatasize, align 32
// CHECK-NEXT:   %2 = icmp uge i256 %1, 4
// CHECK-NEXT:   br i1 %2, label %3, label %40
// CHECK-EMPTY:
// CHECK-NEXT: 3:                                                ; preds = %0
// CHECK-NEXT:   %4 = load ptr addrspace(3), ptr @ptr_calldata, align 32
// CHECK-NEXT:   %5 = getelementptr i8, ptr addrspace(3) %4, i256 0
// CHECK-NEXT:   %6 = load i256, ptr addrspace(3) %5, align 1
// CHECK-NEXT:   switch i256 %6, label %38 [
// CHECK-NEXT:     i256 2776958069, label %7
// CHECK-NEXT:     i256 3263152901, label %16
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
// CHECK-NEXT:   %12 = call i256 @f0_10()
// CHECK-NEXT:   %13 = load i256, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %14 = add i256 %13, 32
// CHECK-NEXT:   %15 = inttoptr i256 %13 to ptr addrspace(1)
// CHECK-NEXT:   store i256 %12, ptr addrspace(1) %15, align 1
// CHECK-NEXT:   call void @__return(i256 %13, i256 32, i256 0)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   br label %39
// CHECK-EMPTY:
// CHECK-NEXT: 16:                                               ; preds = %3
// CHECK-NEXT:   %17 = call i256 @llvm.eravm.getu128()
// CHECK-NEXT:   %18 = icmp ne i256 %17, 0
// CHECK-NEXT:   br i1 %18, label %19, label %20
// CHECK-EMPTY:
// CHECK-NEXT: 19:                                               ; preds = %16
// CHECK-NEXT:   call void @__revert(i256 0, i256 0, i256 0)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   br label %20
// CHECK-EMPTY:
// CHECK-NEXT: 20:                                               ; preds = %19, %16
// CHECK-NEXT:   %21 = call i256 @f1_18()
// CHECK-NEXT:   %22 = load i256, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %23 = add i256 %22, 32
// CHECK-NEXT:   %24 = inttoptr i256 %22 to ptr addrspace(1)
// CHECK-NEXT:   store i256 32, ptr addrspace(1) %24, align 1
// CHECK-NEXT:   %25 = inttoptr i256 %21 to ptr addrspace(1)
// CHECK-NEXT:   %26 = load i256, ptr addrspace(1) %25, align 1
// CHECK-NEXT:   %27 = inttoptr i256 %23 to ptr addrspace(1)
// CHECK-NEXT:   store i256 %26, ptr addrspace(1) %27, align 1
// CHECK-NEXT:   %28 = add i256 %21, 32
// CHECK-NEXT:   %29 = add i256 %23, 32
// CHECK-NEXT:   %30 = inttoptr i256 %29 to ptr addrspace(1)
// CHECK-NEXT:   %31 = inttoptr i256 %28 to ptr addrspace(1)
// CHECK-NEXT:   call void @llvm.memmove.p1.p1.i256(ptr addrspace(1) %30, ptr addrspace(1) %31, i256 %26, i1 false)
// CHECK-NEXT:   %32 = add i256 %29, %26
// CHECK-NEXT:   %33 = inttoptr i256 %32 to ptr addrspace(1)
// CHECK-NEXT:   store i256 0, ptr addrspace(1) %33, align 1
// CHECK-NEXT:   %34 = add i256 %26, 31
// CHECK-NEXT:   %35 = and i256 %34, 4294967264
// CHECK-NEXT:   %36 = add i256 %29, %35
// CHECK-NEXT:   %37 = sub i256 %36, %22
// CHECK-NEXT:   call void @__return(i256 %22, i256 %37, i256 0)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   br label %39
// CHECK-EMPTY:
// CHECK-NEXT: 38:                                               ; preds = %3
// CHECK-NEXT:   br label %39
// CHECK-EMPTY:
// CHECK-NEXT: 39:                                               ; preds = %38, %11, %20
// CHECK-NEXT:   br label %40
// CHECK-EMPTY:
// CHECK-NEXT: 40:                                               ; preds = %39, %0
// CHECK-NEXT:   call void @__revert(i256 0, i256 0, i256 0)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: define i256 @__entry(ptr addrspace(3) %0, i256 %1, i256 %2, i256 %3, i256 %4, i256 %5, i256 %6, i256 %7, i256 %8, i256 %9, i256 %10, i256 %11) #0 personality ptr @__personality {
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
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr @ptr_decommit, align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr @ptr_active, align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 1), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 2), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 3), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 4), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 5), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 6), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 7), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 8), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 9), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 10), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 11), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 12), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 13), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 14), align 32
// CHECK-NEXT:   store ptr addrspace(3) %14, ptr getelementptr ([16 x i256], ptr @ptr_active, i256 0, i256 15), align 32
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
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: define private i256 @f1_18.0() #0 personality ptr @__personality {
// CHECK-NEXT:   store i256 0, ptr addrspace(1) null, align 1
// CHECK-NEXT:   %1 = call i256 @__sha3(ptr addrspace(1) null, i256 32, i1 false)
// CHECK-NEXT:   %2 = load i256, ptr addrspace(5) null, align 1
// CHECK-NEXT:   %3 = add i256 %2, 31
// CHECK-NEXT:   %4 = and i256 %3, 4294967264
// CHECK-NEXT:   %5 = add i256 %4, 32
// CHECK-NEXT:   %6 = load i256, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %7 = add i256 %6, %5
// CHECK-NEXT:   %8 = icmp ugt i256 %7, 18446744073709551615
// CHECK-NEXT:   %9 = icmp ult i256 %7, %6
// CHECK-NEXT:   %10 = or i1 %8, %9
// CHECK-NEXT:   br i1 %10, label %11, label %12
// CHECK-EMPTY:
// CHECK-NEXT: 11:                                               ; preds = %0
// CHECK-NEXT:   store i256 35408467139433450592217433187231851964531694900788300625387963629091585785856, ptr addrspace(1) null, align 1
// CHECK-NEXT:   store i256 65, ptr addrspace(1) inttoptr (i256 4 to ptr addrspace(1)), align 1
// CHECK-NEXT:   call void @__revert(i256 0, i256 24, i256 2)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   br label %12
// CHECK-EMPTY:
// CHECK-NEXT: 12:                                               ; preds = %11, %0
// CHECK-NEXT:   store i256 %7, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %13 = inttoptr i256 %6 to ptr addrspace(1)
// CHECK-NEXT:   store i256 %2, ptr addrspace(1) %13, align 1
// CHECK-NEXT:   %14 = add i256 %2, 31
// CHECK-NEXT:   %15 = and i256 %14, 4294967264
// CHECK-NEXT:   br label %16
// CHECK-EMPTY:
// CHECK-NEXT: 16:                                               ; preds = %19, %12
// CHECK-NEXT:   %17 = phi i256 [ 0, %12 ], [ %26, %19 ]
// CHECK-NEXT:   %18 = icmp slt i256 %17, %15
// CHECK-NEXT:   br i1 %18, label %19, label %27
// CHECK-EMPTY:
// CHECK-NEXT: 19:                                               ; preds = %16
// CHECK-NEXT:   %20 = add i256 %1, %17
// CHECK-NEXT:   %21 = inttoptr i256 %20 to ptr addrspace(5)
// CHECK-NEXT:   %22 = load i256, ptr addrspace(5) %21, align 1
// CHECK-NEXT:   %23 = mul i256 %17, 32
// CHECK-NEXT:   %24 = add i256 %6, %23
// CHECK-NEXT:   %25 = inttoptr i256 %24 to ptr addrspace(1)
// CHECK-NEXT:   store i256 %22, ptr addrspace(1) %25, align 1
// CHECK-NEXT:   %26 = add i256 %17, 1
// CHECK-NEXT:   br label %16
// CHECK-EMPTY:
// CHECK-NEXT: 27:                                               ; preds = %16
// CHECK-NEXT:   ret i256 %6
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: define private i256 @f0_10.0() #0 personality ptr @__personality {
// CHECK-NEXT:   ret i256 42
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: define private i256 @f0_10() #0 personality ptr @__personality {
// CHECK-NEXT:   ret i256 42
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: define private i256 @f1_18() #0 personality ptr @__personality {
// CHECK-NEXT:   store i256 0, ptr addrspace(1) null, align 1
// CHECK-NEXT:   %1 = call i256 @__sha3(ptr addrspace(1) null, i256 32, i1 false)
// CHECK-NEXT:   %2 = load i256, ptr addrspace(5) null, align 1
// CHECK-NEXT:   %3 = add i256 %2, 31
// CHECK-NEXT:   %4 = and i256 %3, 4294967264
// CHECK-NEXT:   %5 = add i256 %4, 32
// CHECK-NEXT:   %6 = load i256, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %7 = add i256 %6, %5
// CHECK-NEXT:   %8 = icmp ugt i256 %7, 18446744073709551615
// CHECK-NEXT:   %9 = icmp ult i256 %7, %6
// CHECK-NEXT:   %10 = or i1 %8, %9
// CHECK-NEXT:   br i1 %10, label %11, label %12
// CHECK-EMPTY:
// CHECK-NEXT: 11:                                               ; preds = %0
// CHECK-NEXT:   store i256 35408467139433450592217433187231851964531694900788300625387963629091585785856, ptr addrspace(1) null, align 1
// CHECK-NEXT:   store i256 65, ptr addrspace(1) inttoptr (i256 4 to ptr addrspace(1)), align 1
// CHECK-NEXT:   call void @__revert(i256 0, i256 24, i256 0)
// CHECK-NEXT:   call void @.unreachable()
// CHECK-NEXT:   br label %12
// CHECK-EMPTY:
// CHECK-NEXT: 12:                                               ; preds = %11, %0
// CHECK-NEXT:   store i256 %7, ptr addrspace(1) inttoptr (i256 64 to ptr addrspace(1)), align 1
// CHECK-NEXT:   %13 = inttoptr i256 %6 to ptr addrspace(1)
// CHECK-NEXT:   store i256 %2, ptr addrspace(1) %13, align 1
// CHECK-NEXT:   %14 = add i256 %2, 31
// CHECK-NEXT:   %15 = and i256 %14, 4294967264
// CHECK-NEXT:   br label %16
// CHECK-EMPTY:
// CHECK-NEXT: 16:                                               ; preds = %19, %12
// CHECK-NEXT:   %17 = phi i256 [ 0, %12 ], [ %26, %19 ]
// CHECK-NEXT:   %18 = icmp slt i256 %17, %15
// CHECK-NEXT:   br i1 %18, label %19, label %27
// CHECK-EMPTY:
// CHECK-NEXT: 19:                                               ; preds = %16
// CHECK-NEXT:   %20 = add i256 %1, %17
// CHECK-NEXT:   %21 = inttoptr i256 %20 to ptr addrspace(5)
// CHECK-NEXT:   %22 = load i256, ptr addrspace(5) %21, align 1
// CHECK-NEXT:   %23 = mul i256 %17, 32
// CHECK-NEXT:   %24 = add i256 %6, %23
// CHECK-NEXT:   %25 = inttoptr i256 %24 to ptr addrspace(1)
// CHECK-NEXT:   store i256 %22, ptr addrspace(1) %25, align 1
// CHECK-NEXT:   %26 = add i256 %17, 1
// CHECK-NEXT:   br label %16
// CHECK-EMPTY:
// CHECK-NEXT: 27:                                               ; preds = %16
// CHECK-NEXT:   ret i256 %6
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nofree null_pointer_is_valid
// CHECK-NEXT: declare i32 @__personality() #0
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nounwind willreturn memory(none)
// CHECK-NEXT: declare i256 @llvm.eravm.getu128() #1
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK-NEXT: declare void @llvm.memcpy.p1.p3.i256(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(3) noalias nocapture readonly, i256, i1 immarg) #2
// CHECK-EMPTY:
// CHECK-NEXT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK-NEXT: declare void @llvm.memmove.p1.p1.i256(ptr addrspace(1) nocapture writeonly, ptr addrspace(1) nocapture readonly, i256, i1 immarg) #2
// CHECK-EMPTY:
// CHECK-NEXT: attributes #0 = { nofree null_pointer_is_valid }
// CHECK-NEXT: attributes #1 = { nounwind willreturn memory(none) }
// CHECK-NEXT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK-EMPTY:
// CHECK-NEXT: !llvm.module.flags = !{!0}
// CHECK-EMPTY:
// CHECK-NEXT: !0 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK-EMPTY:
// ASM: 	.text
// ASM-NEXT: 	.file	{{.*}}
// ASM-NEXT: .unreachable:
// ASM-NEXT: .func_begin0:
// ASM-NEXT: .func_end0:
// ASM-EMPTY:
// ASM-NEXT: __deploy:
// ASM-NEXT: .func_begin1:
// ASM-NEXT: 	add	128, r0, r1
// ASM-NEXT: 	st.1	64, r1
// ASM-NEXT: 	context.get_context_u128	r1
// ASM-NEXT: 	sub!	r1, r0, r0
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
// ASM-NEXT: .func_end1:
// ASM-EMPTY:
// ASM-NEXT: __runtime:
// ASM-NEXT: .func_begin2:
// ASM-NEXT: 	add	128, r0, r1
// ASM-NEXT: 	st.1	64, r1
// ASM-NEXT: 	add	stack[@calldatasize], r0, r1
// ASM-NEXT: 	sub.s!	4, r1, r0
// ASM-NEXT: 	jump.lt	@.BB2_22
// ASM-NEXT: 	ptr.add	stack[@ptr_calldata], r0, r1
// ASM-NEXT: 	ld	r1, r1
// ASM-NEXT: 	sub.s!	@CPI2_0[0], r1, r0
// ASM-NEXT: 	jump.eq	@.BB2_6
// ASM-NEXT: 	sub.s!	@CPI2_1[0], r1, r0
// ASM-NEXT: 	jump.ne	@.BB2_22
// ASM-NEXT: 	context.get_context_u128	r1
// ASM-NEXT: 	sub!	r1, r0, r0
// ASM-NEXT: 	jump.eq	@.BB2_5
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: 	add	r0, r0, r2
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	near_call	r0, @__revert, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-NEXT: .BB2_5:
// ASM-NEXT: 	near_call	r0, @f0_10, @DEFAULT_UNWIND
// ASM-NEXT: 	ld.1	64, r3
// ASM-NEXT: 	st.1	r3, r1
// ASM-NEXT: 	add	32, r0, r2
// ASM-NEXT: 	jump	@.BB2_21
// ASM-NEXT: .BB2_6:
// ASM-NEXT: 	context.get_context_u128	r1
// ASM-NEXT: 	sub!	r1, r0, r0
// ASM-NEXT: 	jump.eq	@.BB2_8
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: 	add	r0, r0, r2
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	near_call	r0, @__revert, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-NEXT: .BB2_8:
// ASM-NEXT: 	near_call	r0, @f1_18, @DEFAULT_UNWIND
// ASM-NEXT: 	add	32, r0, r2
// ASM-NEXT: 	ld.1	64, r3
// ASM-NEXT: 	st.1.inc	r3, r2, r2
// ASM-NEXT: 	ld.1.inc	r1, r1, r4
// ASM-NEXT: 	st.1	r2, r1
// ASM-NEXT: 	and	@CPI2_3[0], r1, r6
// ASM-NEXT: 	and	31, r1, r5
// ASM-NEXT: 	add	64, r3, r2
// ASM-NEXT: 	sub!	r4, r2, r0
// ASM-NEXT: 	jump.ge	@.BB2_14
// ASM-NEXT: 	sub!	r6, r0, r0
// ASM-NEXT: 	jump.eq	@.BB2_10
// ASM-NEXT: 	add	r4, r5, r8
// ASM-NEXT: 	add	r2, r5, r7
// ASM-NEXT: 	sub.s	32, r7, r7
// ASM-NEXT: 	sub.s	32, r8, r8
// ASM-NEXT: .BB2_13:
// ASM-NEXT: 	add	r7, r6, r9
// ASM-NEXT: 	add	r8, r6, r10
// ASM-NEXT: 	ld.1	r10, r10
// ASM-NEXT: 	st.1	r9, r10
// ASM-NEXT: 	sub.s!	32, r6, r6
// ASM-NEXT: 	jump.ne	@.BB2_13
// ASM-NEXT: .BB2_10:
// ASM-NEXT: 	sub!	r5, r0, r0
// ASM-NEXT: 	jump.eq	@.BB2_20
// ASM-NEXT: 	add	r2, r0, r7
// ASM-NEXT: 	jump	@.BB2_19
// ASM-NEXT: .BB2_14:
// ASM-NEXT: 	add	r2, r6, r7
// ASM-NEXT: 	sub!	r6, r0, r0
// ASM-NEXT: 	jump.eq	@.BB2_15
// ASM-NEXT: 	add	r4, r0, r8
// ASM-NEXT: 	add	r2, r0, r9
// ASM-NEXT: .BB2_17:
// ASM-NEXT: 	ld.1.inc	r8, r10, r8
// ASM-NEXT: 	st.1.inc	r9, r10, r9
// ASM-NEXT: 	sub!	r9, r7, r0
// ASM-NEXT: 	jump.ne	@.BB2_17
// ASM-NEXT: .BB2_15:
// ASM-NEXT: 	sub!	r5, r0, r0
// ASM-NEXT: 	jump.eq	@.BB2_20
// ASM-NEXT: 	add	r4, r6, r4
// ASM-NEXT: .BB2_19:
// ASM-NEXT: 	shl.s	3, r5, r5
// ASM-NEXT: 	ld.1	r7, r6
// ASM-NEXT: 	shl	r6, r5, r6
// ASM-NEXT: 	shr	r6, r5, r6
// ASM-NEXT: 	ld.1	r4, r4
// ASM-NEXT: 	sub	256, r5, r5
// ASM-NEXT: 	shr	r4, r5, r4
// ASM-NEXT: 	shl	r4, r5, r4
// ASM-NEXT: 	or	r4, r6, r4
// ASM-NEXT: 	st.1	r7, r4
// ASM-NEXT: .BB2_20:
// ASM-NEXT: 	add	r2, r1, r4
// ASM-NEXT: 	st.1	r4, r0
// ASM-NEXT: 	add	31, r1, r1
// ASM-NEXT: 	and	@CPI2_2[0], r1, r1
// ASM-NEXT: 	add	r2, r1, r1
// ASM-NEXT: 	sub	r1, r3, r2
// ASM-NEXT: .BB2_21:
// ASM-NEXT: 	add	r3, r0, r1
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	near_call	r0, @__return, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-NEXT: .BB2_22:
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: 	add	r0, r0, r2
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	near_call	r0, @__revert, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-NEXT: .func_end2:
// ASM-EMPTY:
// ASM-NEXT: 	.globl	__entry
// ASM-NEXT: __entry:
// ASM-NEXT: .func_begin3:
// ASM-NEXT: 	add	stack[@ptr_calldata + r0], r0, r13
// ASM-NEXT: 	shr.s	96, r13, r13
// ASM-NEXT: 	and	@CPI3_0[0], r13, r14
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_return_data]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_decommit]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 15]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 14]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 13]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 12]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 11]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 10]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 9]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 8]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 7]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 6]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 5]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 4]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 3]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 2]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active + 1]
// ASM-NEXT: 	ptr.add	r1, r14, stack[@ptr_active]
// ASM-NEXT: 	ptr.add	r1, r0, stack[@ptr_calldata]
// ASM-NEXT: 	and	@CPI3_0[0], r13, stack[@calldatasize]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 9]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 8]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 7]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 6]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 5]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 4]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 3]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 2]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data + 1]
// ASM-NEXT: 	add	0, r0, stack[@extra_abi_data]
// ASM-NEXT: 	add	0, r0, stack[@memory_pointer]
// ASM-NEXT: 	add	0, r0, stack[@returndatasize]
// ASM-NEXT: 	add	0, r0, stack[@call_flags]
// ASM-NEXT: 	add	r3, r0, stack[@extra_abi_data]
// ASM-NEXT: 	add	r4, r0, stack[@extra_abi_data + 1]
// ASM-NEXT: 	add	r5, r0, stack[@extra_abi_data + 2]
// ASM-NEXT: 	add	r6, r0, stack[@extra_abi_data + 3]
// ASM-NEXT: 	add	r7, r0, stack[@extra_abi_data + 4]
// ASM-NEXT: 	add	r8, r0, stack[@extra_abi_data + 5]
// ASM-NEXT: 	add	r9, r0, stack[@extra_abi_data + 6]
// ASM-NEXT: 	add	r10, r0, stack[@extra_abi_data + 7]
// ASM-NEXT: 	add	r11, r0, stack[@extra_abi_data + 8]
// ASM-NEXT: 	add	r12, r0, stack[@extra_abi_data + 9]
// ASM-NEXT: 	add	r2, r0, stack[@call_flags]
// ASM-NEXT: 	and!	1, r2, r0
// ASM-NEXT: 	jump.eq	@.BB3_2
// ASM-NEXT: 	near_call	r0, @__deploy, @DEFAULT_UNWIND
// ASM-NEXT: .BB3_2:
// ASM-NEXT: 	near_call	r0, @__runtime, @DEFAULT_UNWIND
// ASM-NEXT: .func_end3:
// ASM-EMPTY:
// ASM-NEXT: f0_10:
// ASM-NEXT: .func_begin4:
// ASM-NEXT: 	add	42, r0, r1
// ASM-NEXT: 	ret
// ASM-NEXT: .func_end4:
// ASM-EMPTY:
// ASM-NEXT: f1_18:
// ASM-NEXT: .func_begin5:
// ASM-NEXT: 	nop	stack+=[5 + r0]
// ASM-NEXT: 	st.1	0, r0
// ASM-NEXT: 	add	32, r0, r2
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	near_call	r0, @__sha3, @DEFAULT_UNWIND
// ASM-NEXT: 	add	r1, r0, r5
// ASM-NEXT: 	sload	r0, r2
// ASM-NEXT: 	add	31, r2, r1
// ASM-NEXT: 	and	@CPI5_0[0], r1, r7
// ASM-NEXT: 	ld.1	64, r6
// ASM-NEXT: 	add	r7, r6, r1
// ASM-NEXT: 	add	32, r1, r3
// ASM-NEXT: 	sub!	r3, r6, r0
// ASM-NEXT: 	add	0, r0, r1
// ASM-NEXT: 	add.lt	1, r0, r1
// ASM-NEXT: 	sub.s!	@CPI5_1[0], r3, r0
// ASM-NEXT: 	jump.gt	@.BB5_2
// ASM-NEXT: 	and!	1, r1, r0
// ASM-NEXT: 	jump.eq	@.BB5_3
// ASM-NEXT: .BB5_2:
// ASM-NEXT: 	add	@CPI5_2[0], r0, r1
// ASM-NEXT: 	st.1	0, r1
// ASM-NEXT: 	add	65, r0, r1
// ASM-NEXT: 	st.1	4, r1
// ASM-NEXT: 	add	r2, r0, stack-[2]
// ASM-NEXT: 	add	24, r0, r2
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: 	add	r3, r0, stack-[1]
// ASM-NEXT: 	add	r0, r0, r3
// ASM-NEXT: 	add	r5, r0, stack-[5]
// ASM-NEXT: 	add	r6, r0, stack-[4]
// ASM-NEXT: 	add	r7, r0, stack-[3]
// ASM-NEXT: 	near_call	r0, @__revert, @DEFAULT_UNWIND
// ASM-NEXT: 	near_call	r0, @.unreachable, @DEFAULT_UNWIND
// ASM-NEXT: 	add	stack-[1], r0, r3
// ASM-NEXT: 	add	stack-[2], r0, r2
// ASM-NEXT: 	add	stack-[3], r0, r7
// ASM-NEXT: 	add	stack-[4], r0, r6
// ASM-NEXT: 	add	stack-[5], r0, r5
// ASM-NEXT: .BB5_3:
// ASM-NEXT: 	st.1	64, r3
// ASM-NEXT: 	st.1	r6, r2
// ASM-NEXT: 	add	r0, r0, r1
// ASM-NEXT: .BB5_4:
// ASM-NEXT: 	sub!	r1, r7, r0
// ASM-NEXT: 	add	r0, r0, r2
// ASM-NEXT: 	add.ge	@CPI5_3[0], r0, r2
// ASM-NEXT: 	and	@CPI5_3[0], r1, r3
// ASM-NEXT: 	sub!	r3, r0, r0
// ASM-NEXT: 	add	r0, r0, r4
// ASM-NEXT: 	add.lt	@CPI5_3[0], r0, r4
// ASM-NEXT: 	sub.s!	@CPI5_3[0], r3, r0
// ASM-NEXT: 	add.ne	r2, r0, r4
// ASM-NEXT: 	sub!	r4, r0, r0
// ASM-NEXT: 	jump.ne	@.BB5_6
// ASM-NEXT: 	shl.s	5, r1, r2
// ASM-NEXT: 	add	r6, r2, r2
// ASM-NEXT: 	add	r5, r1, r3
// ASM-NEXT: 	sload	r3, r3
// ASM-NEXT: 	st.1	r2, r3
// ASM-NEXT: 	add	1, r1, r1
// ASM-NEXT: 	jump	@.BB5_4
// ASM-NEXT: .BB5_6:
// ASM-NEXT: 	add	r6, r0, r1
// ASM-NEXT: 	ret
// ASM-NEXT: .func_end5:
// ASM-EMPTY:
// ASM-NEXT: 	.data
// ASM-NEXT: 	.p2align	5, 0x0
// ASM-NEXT: ptr_decommit:
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
// ASM-NEXT: ptr_active:
// ASM-NEXT: 	.zero	512
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
// ASM-NEXT: 	.cell	3263152901
// ASM-NEXT: CPI2_1:
// ASM-NEXT: 	.cell	2776958069
// ASM-NEXT: CPI2_2:
// ASM-NEXT: CPI5_0:
// ASM-NEXT: 	.cell	4294967264
// ASM-NEXT: CPI2_3:
// ASM-NEXT: 	.cell	-32
// ASM-NEXT: CPI3_0:
// ASM-NEXT: 	.cell	4294967295
// ASM-NEXT: CPI5_1:
// ASM-NEXT: 	.cell	18446744073709551615
// ASM-NEXT: CPI5_2:
// ASM-NEXT: 	.cell	35408467139433450592217433187231851964531694900788300625387963629091585785856
// ASM-NEXT: CPI5_3:
// ASM-NEXT: 	.cell	-57896044618658097711785492504343953926634992332820282019728792003956564819968
// ASM-EMPTY:
