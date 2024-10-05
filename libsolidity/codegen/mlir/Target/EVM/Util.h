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
// EVM specific utility
//

#pragma once

#include "libsolidity/codegen/mlir/Sol/SolOps.h"
#include "libsolutil/ErrorCodes.h"
#include "mlir/IR/Builders.h"

namespace evm {

enum AddrSpace : unsigned {
  AddrSpace_Stack = 0,
  AddrSpace_Heap = 1,
  AddrSpace_CallData = 2,
  AddrSpace_ReturnData = 3,
  AddrSpace_Code = 4,
  AddrSpace_Storage = 5,
  AddrSpace_TransientStorage = 6,
};

enum ByteLen {
  ByteLen_Byte = 1,
  ByteLen_X32 = 4,
  ByteLen_X64 = 8,
  ByteLen_EthAddr = 20,
  ByteLen_Field = 32
};

using AllocSize = int64_t;

/// Returns the alignment of `addrSpace`
unsigned getAlignment(AddrSpace addrSpace);
/// Returns the alignment of the LLVMPointerType value `ptr`
unsigned getAlignment(mlir::Value ptr);

// FIXME: Remove this! The lowering should not expand the ABI encoding/decoding
// code (where this is used) involved in sol.contract, sol.emit etc. lowering.
// Instead it should generate a high level sol op that can have a custom
// lowering for each target.
/// MLIR version of solidity ast's Type::calldataHeadSize.
unsigned getCallDataHeadSize(mlir::Type ty);

/// MLIR version of solidity ast's Type::storageBytes().
unsigned getStorageByteCount(mlir::Type ty);

/// IR Builder for EVM specific lowering.
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

  //
  // TODO? Should we work with the high level types + OpAdaptor for the APIs
  // that work with low level integral type pointers?
  //

  /// Generates a low level integral type pointer to the address holding the
  /// data of a dynamic allocation.
  mlir::Value
  genDataAddrPtr(mlir::Value addr, mlir::sol::DataLocation dataLoc,
                 std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a load from the low level integral type address.
  mlir::Value genLoad(mlir::Value addr, mlir::sol::DataLocation dataLoc,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a store to the low level integral type address.
  void genStore(mlir::Value val, mlir::Value addr,
                mlir::sol::DataLocation dataLoc,
                std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the store of string at address.
  void genStringStore(std::string const &str, mlir::Value addr,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a loop to copy the data. This works for low level integral type
  /// addresses.
  void genCopyLoop(mlir::Value srcAddr, mlir::Value dstAddr,
                   mlir::Value sizeInWords, mlir::sol::DataLocation srcDataLoc,
                   mlir::sol::DataLocation dstDataLoc,
                   std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates an assertion that the tuple size should be less than `size`.
  void
  genABITupleSizeAssert(mlir::TypeRange tys, mlir::Value size,
                        std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the tuple encoder code as per the ABI and returns the address at
  /// the end of the tuple.
  mlir::Value
  genABITupleEncoding(mlir::TypeRange tys, mlir::ValueRange vals,
                      mlir::Value tupleStart,
                      std::optional<mlir::Location> locArg = std::nullopt);
  mlir::Value
  genABITupleEncoding(std::string const &str, mlir::Value headStart,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the tuple decoder code as per the ABI and populates the results.
  void genABITupleDecoding(mlir::TypeRange tys, mlir::Value tupleStart,
                           mlir::Value tupleEnd,
                           std::vector<mlir::Value> &results, bool fromMem,
                           std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the panic code.
  void genPanic(solidity::util::PanicCode code, mlir::Value cond,
                std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a revert with message.
  void genRevertWithMsg(std::string const &msg,
                        std::optional<mlir::Location> locArg = std::nullopt);
  void genRevertWithMsg(mlir::Value cond, std::string const &msg,
                        std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a revert without message.
  void genRevert(mlir::Value cond,
                 std::optional<mlir::Location> locArg = std::nullopt);

  //
  // TODO: Move the following APIs somewhere else.
  //

  /// Returns an existing personality function symbol.
  mlir::FlatSymbolRefAttr getPersonality();
};

} // namespace evm
