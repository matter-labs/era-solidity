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

#include "SolidityOps.h"
#include "Solidity/SolidityOpsDialect.cpp.inc"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::solidity;

void SolidityDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Solidity/SolidityOps.cpp.inc"
      >();
}

void ContractOp::build(OpBuilder &builder, OperationState &state,
                       StringRef name) {
  state.addRegion()->emplaceBlock();
  state.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

void ObjectOp::build(OpBuilder &builder, OperationState &state,
                     StringRef name) {
  state.addRegion()->emplaceBlock();
  state.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

#define GET_OP_CLASSES
#include "Solidity/SolidityOps.cpp.inc"