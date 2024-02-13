add_mlir_doc(SolidityDialect SolidityDialect Solidity/ -gen-dialect-doc)
add_mlir_doc(SolidityOps SolidityOps Solidity/ -gen-op-doc)
add_mlir_doc(SolidityAttrs SolidityAttrs Dialects/ -gen-attrdef-doc)
add_mlir_doc(SolidityTypes SolidityTypes Dialects/ -gen-typedef-doc)

# Create MLIRSolidityOpsIncGen
add_mlir_dialect(SolidityOps sol)

set(LLVM_TARGET_DEFINITIONS SolidityOps.td)

# Create MLIRSolidityExtraIncGen
mlir_tablegen(SolidityOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SolidityOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(SolidityOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=sol)
mlir_tablegen(SolidityOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=sol)
add_public_tablegen_target(MLIRSolidityExtraIncGen)
