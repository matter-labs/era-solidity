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
// Sol operations
//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Sol/SolOpsDialect.h.inc"
#include "Sol/SolOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Sol/SolOpsAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Sol/SolOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Sol/SolOps.h.inc"

namespace mlir {
namespace sol {
/// Returns the data-location of type.
DataLocation getDataLocation(Type ty);
} // namespace sol
} // namespace mlir