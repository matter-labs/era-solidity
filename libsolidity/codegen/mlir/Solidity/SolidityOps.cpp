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
#include "Solidity/SolidityOpsEnums.cpp.inc"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sol;

void SolidityDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Solidity/SolidityOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Solidity/SolidityOpsAttributes.cpp.inc"
      >();
}

DictionaryAttr ContractOp::getInterfaceFnAttr(func::FuncOp fn) {
  ArrayAttr interfaceFnsAttr = getInterfaceFnsAttr();
  auto fnSym = SymbolRefAttr::get(fn.getSymNameAttr());
  TypeAttr fnTy = fn.getFunctionTypeAttr();

  for (Attribute interfaceFnAttr : interfaceFnsAttr) {
    auto attr = interfaceFnAttr.cast<DictionaryAttr>();
    assert(attr.contains("sym"));
    if (fnSym == attr.get("sym").cast<SymbolRefAttr>() &&
        fnTy == attr.get("type").cast<TypeAttr>())
      return attr;
  }

  return {};
}

void ObjectOp::build(OpBuilder &builder, OperationState &state,
                     StringRef name) {
  state.addRegion()->emplaceBlock();
  state.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

#define GET_OP_CLASSES
#include "Solidity/SolidityOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Solidity/SolidityOpsAttributes.cpp.inc"
