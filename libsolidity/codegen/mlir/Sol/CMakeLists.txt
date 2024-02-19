add_mlir_doc(SolDialect SolDialect Sol/ -gen-dialect-doc)
add_mlir_doc(SolOps SolOps Sol/ -gen-op-doc)
add_mlir_doc(SolAttrs SolAttrs Dialects/ -gen-attrdef-doc)
add_mlir_doc(SolTypes SolTypes Dialects/ -gen-typedef-doc)

# Create MLIRSolOpsIncGen
add_mlir_dialect(SolOps sol)

set(LLVM_TARGET_DEFINITIONS SolOps.td)

# Create MLIRSolExtraIncGen
mlir_tablegen(SolOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SolOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(SolOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=sol)
mlir_tablegen(SolOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=sol)
add_public_tablegen_target(MLIRSolExtraIncGen)
