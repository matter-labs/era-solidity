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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace eravm {

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

static const char *GlobHeapMemPtr = "memory_pointer";
static const char *GlobCallDataSize = "calldatasize";
static const char *GlobRetDataSz = "returndatasize";
static const char *GlobCallFlags = "call_flags";
static const char *GlobExtraABIData = "extra_abi_data";
static const char *GlobCallDataPtr = "ptr_calldata";
static const char *GlobRetDataPtr = "ptr_return_data";
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

/// Builder extension for EraVM
class BuilderHelper {
  mlir::OpBuilder &b;
  mlir::Location defLoc;
  solidity::mlirgen::BuilderHelper h;

public:
  explicit BuilderHelper(mlir::OpBuilder &b)
      : b(b), defLoc(b.getUnknownLoc()), h(b) {}

  explicit BuilderHelper(mlir::OpBuilder &b, mlir::Location loc)
      : b(b), defLoc(loc), h(b) {}

  /// Initialize global variables for EraVM
  void initGlobs(mlir::Location loc, mlir::ModuleOp mod);

  /// Generates and return the ABI length for the pointer `ptr`
  mlir::Value getABILen(mlir::Location loc, mlir::Value ptr);

  /// Returns an existing or a new (if not found) creation function.
  mlir::sol::FuncOp getOrInsertCreationFuncOp(mlir::StringRef name,
                                              mlir::FunctionType fnTy,
                                              mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) runtime function.
  mlir::sol::FuncOp getOrInsertRuntimeFuncOp(mlir::StringRef name,
                                             mlir::FunctionType fnTy,
                                             mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) return function symbol.
  mlir::FlatSymbolRefAttr getOrInsertReturn(mlir::ModuleOp mod);

  /// Returns an existing or a new (if not found) revert function symbol.
  mlir::FlatSymbolRefAttr getOrInsertRevert(mlir::ModuleOp mod);

  /// Returns the address to the calldatasize global variable (creates the
  /// variable if it doesn't exist).
  mlir::LLVM::AddressOfOp
  getCallDataSizeAddr(mlir::ModuleOp mod,
                      std::optional<mlir::Location> loc = std::nullopt);

  /// Returns the address to the ptr_calldata global variable (creates the
  /// variable if it doesn't exist).
  mlir::LLVM::AddressOfOp
  getCallDataPtrAddr(mlir::ModuleOp mod,
                     std::optional<mlir::Location> loc = std::nullopt);

  /// Loads the ptr_calldata global variable (creates the variable if it doesn't
  /// exist).
  mlir::LLVM::LoadOp
  loadCallDataPtr(mlir::ModuleOp mod,
                  std::optional<mlir::Location> loc = std::nullopt);
};

} // namespace eravm
