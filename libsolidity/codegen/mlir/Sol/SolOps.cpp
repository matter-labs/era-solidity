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

#include "SolOps.h"
#include "Sol/SolOpsDialect.cpp.inc"
#include "Sol/SolOpsEnums.cpp.inc"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sol;

void SolDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Sol/SolOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Sol/SolOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Sol/SolOpsAttributes.cpp.inc"
      >();
}

Type mlir::sol::getEltType(Type ty, Index structTyIdx) {
  if (auto ptrTy = dyn_cast<sol::PointerType>(ty)) {
    return ptrTy.getPointeeType();
  }
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    return arrTy.getEltType();
  }
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    return structTy.getMemTypes()[structTyIdx];
  }
  llvm_unreachable("Invalid type");
}

DataLocation mlir::sol::getDataLocation(Type ty) {
  return TypeSwitch<Type, DataLocation>(ty)
      .Case<PointerType>(
          [](sol::PointerType ptrTy) { return ptrTy.getDataLocation(); })
      .Case<ArrayType>(
          [](sol::ArrayType arrTy) { return arrTy.getDataLocation(); })
      .Case<StringType>(
          [](sol::StringType strTy) { return strTy.getDataLocation(); })
      .Case<StructType>(
          [](sol::StructType structTy) { return structTy.getDataLocation(); })
      .Case<MappingType>(
          [](sol::MappingType) { return sol::DataLocation::Storage; })
      .Default([&](Type) { return DataLocation::Stack; });
}

bool mlir::sol::isRefType(Type ty) {
  return isa<ArrayType>(ty) || isa<StringType>(ty) || isa<StructType>(ty) ||
         isa<PointerType>(ty) || isa<MappingType>(ty);
}

bool mlir::sol::isNonPtrRefType(Type ty) {
  return isRefType(ty) && !isa<PointerType>(ty);
}

bool mlir::sol::isLeftAligned(Type ty) {
  if (isa<IntegerType>(ty))
    return false;
  llvm_unreachable("NYI: isLeftAligned of other types");
}

bool mlir::sol::isDynamicallySized(Type ty) {
  if (isa<StringType>(ty))
    return true;

  if (auto arrTy = dyn_cast<ArrayType>(ty))
    return arrTy.isDynSized();

  return false;
}

bool mlir::sol::hasDynamicallySizedElt(Type ty) {
  if (isa<StringType>(ty))
    return true;

  if (auto arrTy = dyn_cast<ArrayType>(ty))
    return arrTy.isDynSized() || hasDynamicallySizedElt(arrTy.getEltType());

  if (auto structTy = dyn_cast<StructType>(ty))
    return llvm::any_of(structTy.getMemTypes(),
                        [](Type ty) { return hasDynamicallySizedElt(ty); });

  return false;
}

static ParseResult parseDataLocation(AsmParser &parser,
                                     DataLocation &dataLocation) {
  StringRef dataLocationTok;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&dataLocationTok))
    return failure();

  auto parsedDataLoc = symbolizeDataLocation(dataLocationTok);
  if (!parsedDataLoc) {
    parser.emitError(loc, "Invalid data-location");
    return failure();
  }

  dataLocation = *parsedDataLoc;
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Parses a sol.array type.
///
///   array-type ::= `<` size `x` elt-ty `,` data-location `>`
///   size ::= fixed-size | `?`
///
Type ArrayType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  int64_t size = -1;
  if (parser.parseOptionalQuestion()) {
    if (parser.parseInteger(size))
      return {};
  }

  if (parser.parseKeyword("x"))
    return {};

  Type eleTy;
  if (parser.parseType(eleTy))
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), size, eleTy, dataLocation);
}

/// Prints a sol.array type.
void ArrayType::print(AsmPrinter &printer) const {
  printer << "<";

  if (getSize() == -1)
    printer << "?";
  else
    printer << getSize();

  printer << " x " << getEltType() << ", "
          << stringifyDataLocation(getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

/// Parses a sol.string type.
///
///   string-type ::= `<` data-location `>`
///
Type StringType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), dataLocation);
}

/// Prints a sol.string type.
void StringType::print(AsmPrinter &printer) const {
  printer << "<" << stringifyDataLocation(this->getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

/// Parses a sol.struct type.
///
///   struct-type ::= `<` `(` member-types `)` `,` data-location `>`
///
Type StructType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  if (parser.parseLParen())
    return {};

  SmallVector<Type, 4> memTys;
  do {
    Type memTy;
    if (parser.parseType(memTy))
      return {};
    memTys.push_back(memTy);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), memTys, dataLocation);
}

/// Prints a sol.array type.
void StructType::print(AsmPrinter &printer) const {
  printer << "<(";
  llvm::interleaveComma(getMemTypes(), printer.getStream(),
                        [&](Type memTy) { printer << memTy; });
  printer << "), " << stringifyDataLocation(getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

/// Parses a sol.ptr type.
///
///   ptr-type ::= `<` pointee-ty, data-location `>`
///
Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  Type pointeeTy;
  if (parser.parseType(pointeeTy))
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), pointeeTy, dataLocation);
}

/// Prints a sol.ptr type.
void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getPointeeType() << ", "
          << stringifyDataLocation(getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// ContractOp
//===----------------------------------------------------------------------===//

DictionaryAttr ContractOp::getInterfaceFnAttr(sol::FuncOp fn) {
  ArrayAttr interfaceFnsAttr = getInterfaceFnsAttr();
  auto fnSym = SymbolRefAttr::get(fn.getSymNameAttr());
  TypeAttr fnTy = fn.getFunctionTypeAttr();

  for (Attribute interfaceFnAttr : interfaceFnsAttr) {
    auto attr = cast<DictionaryAttr>(interfaceFnAttr);
    assert(attr.contains("sym"));
    if (fnSym == cast<SymbolRefAttr>(attr.get("sym")) &&
        fnTy == cast<TypeAttr>(attr.get("type")))
      return attr;
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ObjectOp
//===----------------------------------------------------------------------===//

void ObjectOp::build(OpBuilder &builder, OperationState &state,
                     StringRef name) {
  state.addRegion()->emplaceBlock();
  state.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs,
      /*resultAttrs=*/{}, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name));
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, StateMutability stateMutability,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  build(builder, state, name, type, attrs, argAttrs);
  state.addAttribute(
      getStateMutabilityAttrName(state.name),
      StateMutabilityAttr::get(builder.getContext(), stateMutability));
}

void FuncOp::cloneInto(FuncOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});
  for (const auto &attr : (*this)->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

FuncOp FuncOp::clone(IRMapping &mapper) {
  // Create the new function.
  FuncOp newFunc = cast<FuncOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    FunctionType oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i)
      if (!mapper.contains(getArgument(i)))
        newInputs.push_back(oldType.getInput(i));

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                        oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
        for (unsigned i = 0; i != oldNumArgs; ++i)
          if (!mapper.contains(getArgument(i)))
            newArgAttrs.push_back(argAttrs[i]);
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}
FuncOp FuncOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

/// Parses an allocation op.
///
///   ssa-id = alloc-op ssa-use? attr-dict `:` type
///
static ParseResult parseAllocationOp(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand sizeOperand;
  auto parseRes = parser.parseOptionalOperand(sizeOperand);
  if (parseRes.has_value() && succeeded(parseRes.value())) {
    if (parser.resolveOperand(sizeOperand,
                              parser.getBuilder().getIntegerType(256),
                              result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  Type allocType;
  if (parser.parseType(allocType))
    return failure();
  result.addAttribute("alloc_type", TypeAttr::get(allocType));
  result.addTypes(allocType);

  return success();
}

static void printAllocationOp(Operation *op, OpAsmPrinter &p, Value size = {}) {
  if (size)
    p << ' ' << size;
  p.printOptionalAttrDict(op->getAttrs(), {"alloc_type"});
  p << " : " << op->getResultTypes()[0];
}

ParseResult AllocaOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAllocationOp(parser, result);
}

void AllocaOp::print(OpAsmPrinter &p) { printAllocationOp(*this, p); }

//===----------------------------------------------------------------------===//
// MallocOp
//===----------------------------------------------------------------------===//

ParseResult MallocOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAllocationOp(parser, result);
}

void MallocOp::print(OpAsmPrinter &p) {
  printAllocationOp(*this, p, getSize());
}

//===----------------------------------------------------------------------===//
// EmitOp
//===----------------------------------------------------------------------===//

/// Parses an emit op.
///
///   emit-op signature? (`indexed` `=` `[` indexed-args `]`)?
///           (`non-indexed` `=` `[` non-indexed-args `]`)?
///           attr-dict `:` type(args)
///
ParseResult EmitOp::parse(OpAsmParser &p, OperationState &result) {
  auto b = p.getBuilder();

  // Parse the signature.
  std::string signature;
  if (!p.parseOptionalString(&signature))
    result.addAttribute("signature", b.getStringAttr(signature));

  SmallVector<OpAsmParser::UnresolvedOperand> unresolvedOpnds;

  // Parse indexed args and add the attribute to track its count.
  if (!p.parseOptionalKeyword("indexed")) {
    if (p.parseEqual())
      return failure();

    if (p.parseOperandList(unresolvedOpnds, OpAsmParser::Delimiter::Square))
      return failure();
    assert(unresolvedOpnds.size() <= UINT8_MAX);
  }
  result.addAttribute("indexedArgsCount",
                      b.getI8IntegerAttr(unresolvedOpnds.size()));

  // Parse non-indexed args.
  if (!p.parseOptionalKeyword("non_indexed")) {
    if (p.parseEqual())
      return failure();

    if (p.parseOperandList(unresolvedOpnds, OpAsmParser::Delimiter::Square))
      return failure();
  }

  // Parse other attributes and the type list.
  SmallVector<Type> opndTys;
  if (p.parseOptionalAttrDict(result.attributes) ||
      p.parseOptionalColonTypeList(opndTys))
    return failure();

  // Resolve all the indexed and non-indexed args.
  if (p.resolveOperands(unresolvedOpnds, opndTys, p.getNameLoc(),
                        result.operands))
    return failure();

  return success();
}

void EmitOp::print(OpAsmPrinter &p) {
  if (auto signature = getSignatureAttr()) {
    p << ' ' << signature;
  }

  if (getIndexedArgsCount()) {
    p << " indexed = [";
    p.printOperands(getIndexedArgs());
    p << ']';
  }

  if (!getNonIndexedArgs().empty()) {
    p << " non_indexed = [";
    p.printOperands(getNonIndexedArgs());
    p << ']';
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"indexedArgsCount", "signature"});
  if (!getArgs().empty())
    p << " : " << getArgs().getTypes();
}

#define GET_OP_CLASSES
#include "Sol/SolOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Sol/SolOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Sol/SolOpsTypes.cpp.inc"
