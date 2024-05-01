include_directories(${PROJECT_BINARY_DIR}/libsolidity/codegen/mlir)

add_executable(sol-opt main.cpp)

target_link_libraries(sol-opt PUBLIC
  MLIROptLib
  # FIXME: We only need the solidity dialect lowering code from libsolidity.
  solidity
)
