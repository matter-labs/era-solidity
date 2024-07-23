// This file is part of solidity.

// solidity is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// solidity is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with solidity.  If not, see <http://www.gnu.org/licenses/>.

// SPDX-License-Identifier: GPL-3.0

//
// EraVM specific utility
//

#pragma once

#include "libsolidity/codegen/mlir/Util.h"
#include "libsolutil/ErrorCodes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <vector>

namespace eravm {

using AllocSize = int64_t;

//
// FIXME: Is it possible to define enum classes whose members can implicitly
// cast to unsigned? AddrSpace, {Byte, Bit}Len enums are mostly used as unsigned
// in the lowering (for instance, address space arguments in llvm are unsigned).
// For now, we're mimicking the scope with prefixes
//

enum AddrSpace : unsigned {
  AddrSpace_Stack = 0,
  AddrSpace_Heap = 1,
  AddrSpace_HeapAuxiliary = 2,
  AddrSpace_Generic = 3,
  AddrSpace_Code = 4,
  AddrSpace_Storage = 5,
};

enum ByteLen {
  ByteLen_Byte = 1,
  ByteLen_X32 = 4,
  ByteLen_X64 = 8,
  ByteLen_EthAddr = 20,
  ByteLen_Field = 32
};
enum BitLen {
  BitLen_Bool = 1,
  BitLen_Byte = 8,
  BitLen_X32 = BitLen_Byte * ByteLen_X32,
  BitLen_X64 = BitLen_Byte * ByteLen_X64,
  BitLen_EthAddr = BitLen_Byte * ByteLen_EthAddr,
  BitLen_Field = BitLen_Byte * ByteLen_Field
};

enum : unsigned {
  HeapAuxOffsetCtorRetData = ByteLen_Field * 8,
  ExtraABIDataSize = 10,
};
enum RetForwardPageType { UseHeap = 0, ForwardFatPtr = 1, UseAuxHeap = 2 };

enum : uint16_t { Address_EventWriter = 0x800D };

// TODO: Track address space, linkage etc. here (in a struct?) as well?
static const char *GlobHeapMemPtr = "memory_pointer";
static const char *GlobCallDataSize = "calldatasize";
static const char *GlobRetDataSize = "returndatasize";
static const char *GlobCallFlags = "call_flags";
static const char *GlobExtraABIData = "extra_abi_data";
static const char *GlobCallDataPtr = "ptr_calldata";
static const char *GlobRetDataPtr = "ptr_return_data";
static const char *GlobDecommitPtr = "ptr_decommit";
static const char *GlobActivePtr = "ptr_active";

enum EntryInfo {
  ArgIndexCallDataABI = 0,
  ArgIndexCallFlags = 1,
  MandatoryArgCnt = 2,
};

/// Returns the alignment of `addrSpace`
unsigned getAlignment(AddrSpace addrSpace);
/// Returns the alignment of the LLVMPointerType value `ptr`
unsigned getAlignment(mlir::Value ptr);

/// EraVM version of solidity ast's Type::calldataHeadSize.
unsigned getCallDataHeadSize(mlir::Type ty);

/// EraVM version of solidity ast's Type::storageBytes().
unsigned getStorageByteCount(mlir::Type ty);

/// IR Builder for EraVM specific lowering.
class Builder {
  // It's possible to provide a mlirgen::BuilderHelper member with same default
  // location, but then it becomes tricky to keep the default location behaviour
  // consistent.

  mlir::OpBuilder &b;
  mlir::Location defLoc;

public:
  explicit Builder(mlir::OpBuilder &b) : b(b), defLoc(b.getUnknownLoc()) {}

  explicit Builder(mlir::OpBuilder &b, mlir::Location loc)
      : b(b), defLoc(loc) {}

  /// Generates a pointer to the heap address space.
  mlir::Value genHeapPtr(mlir::Value addr,
                         std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the initialization of global variables for EraVM.
  void genGlobalVarsInit(mlir::ModuleOp mod,
                         std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the store of string at address.
  void genStringStore(std::string const &str, mlir::Value addr,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the ABI length for the pointer `ptr`.
  mlir::Value genABILen(mlir::Value ptr,
                        std::optional<mlir::Location> locArg = std::nullopt);

  // FIXME: Refactor ABI APIs to distinguish solidity/evm ones with eravm
  // specfic ones.

  /// Generates the tuple encoding as per ABI for the literal string and return
  /// the "tail" address.
  mlir::Value
  genABITupleEncoding(std::string const &str, mlir::Value headStart,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a revert with message.
  void genRevertWithMsg(std::string const &msg,
                        std::optional<mlir::Location> locArg = std::nullopt);
  void genRevertWithMsg(mlir::Value cond, std::string const &msg,
                        std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a revert without message.
  void genRevert(mlir::Value cond,
                 std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates an assertion that the tuple size should be less than `size`.
  void
  genABITupleSizeAssert(mlir::TypeRange tys, mlir::Value size,
                        std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the tuple encoding as per ABI and return the "tail" address.
  mlir::Value
  genABITupleEncoding(mlir::TypeRange tys, mlir::ValueRange vals,
                      mlir::Value headStart,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the tuple decoder as per ABI and populates the results.
  void genABITupleDecoding(mlir::TypeRange tys, mlir::Value headStart,
                           std::vector<mlir::Value> &results, bool fromMem,
                           std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the ABI data for an external call.
  mlir::Value genABIData(mlir::Value addr, mlir::Value size,
                         eravm::AddrSpace addrSpace, bool isSysCall,
                         std::optional<mlir::Location> locArg = std::nullopt);

  /// Returns an existing or a new (if not found) creation function.
  mlir::sol::FuncOp getOrInsertCreationFuncOp(mlir::StringRef name,
                                              mlir::FunctionType fnTy,
                                              mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) runtime function.
  mlir::sol::FuncOp getOrInsertRuntimeFuncOp(mlir::StringRef name,
                                             mlir::FunctionType fnTy,
                                             mlir::ModuleOp mod);

  /// Returns an existing personality function symbol.
  mlir::FlatSymbolRefAttr getPersonality();

  /// Returns an existing or a new (if not found) personality function symbol.
  mlir::FlatSymbolRefAttr getOrInsertPersonality(mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) return function symbol.
  mlir::FlatSymbolRefAttr getOrInsertReturn(mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) revert function symbol.
  mlir::FlatSymbolRefAttr getOrInsertRevert(mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) sha3 function symbol.
  mlir::FlatSymbolRefAttr getOrInsertSha3(mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) far-call function.
  mlir::sol::FuncOp getOrInsertFarCall(mlir::ModuleOp mod);

  /// Generates the address to the calldatasize global variable (creates the
  /// variable if it doesn't exist).
  mlir::LLVM::AddressOfOp
  genCallDataSizeAddr(mlir::ModuleOp mod,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a load of the calldatasize global variable (creates the variable
  /// if it doesn't exist).
  mlir::LLVM::LoadOp
  genCallDataSizeLoad(mlir::ModuleOp mod,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the address to the ptr_calldata global variable (creates the
  /// variable if it doesn't exist).
  mlir::LLVM::AddressOfOp
  genCallDataPtrAddr(mlir::ModuleOp mod,
                     std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a load of the ptr_calldata global variable (creates the variable
  /// if it doesn't exist).
  mlir::LLVM::LoadOp
  genCallDataPtrLoad(mlir::ModuleOp mod,
                     std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the panic code.
  void genPanic(solidity::util::PanicCode code, mlir::Value cond,
                std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the free pointer.
  mlir::Value genFreePtr(std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the memory allocation code.
  mlir::Value genMemAlloc(mlir::Value size,
                          std::optional<mlir::Location> locArg = std::nullopt);
  mlir::Value genMemAlloc(AllocSize size,
                          std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the memory allocation code for dynamic array.
  mlir::Value
  genMemAllocForDynArray(mlir::Value sizeVar, mlir::Value sizeInBytes,
                         std::optional<mlir::Location> locArg = std::nullopt);
};

} // namespace eravm
