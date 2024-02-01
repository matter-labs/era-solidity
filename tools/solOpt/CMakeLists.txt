include_directories(${PROJECT_BINARY_DIR}/libsolidity/codegen/mlir)

add_executable(sol-opt main.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(sol-opt PUBLIC
  ${dialect_libs}
  ${conversion_libs}
  MLIRSupport
  MLIRDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransforms
  MLIRTransformUtils
  MLIROptLib
  # FIXME: We only need the solidity dialect lowering code from libsolidity.
  solidity
)
