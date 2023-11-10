#pragma once

#include "liblangutil/CharStream.h"
#include "libsolidity/ast/ASTForward.h"
#include "libyul/ASTForward.h"
#include "libyul/Dialect.h"
#include "libyul/Object.h"

namespace solidity::mlirgen {

enum class Action {
  PrintInitStg,
  PrintPostSolcDialLowering,
  PrintLLVMIR,
  LowerToAsm,
};

/// Registers required command line options in the MLIR framework
extern void registerMLIRCLOpts();

/// Parses command line options in `argv` for the MLIR framework
extern bool parseMLIROpts(std::vector<const char *> &_argv);

extern bool runSolidityToMLIRPass(
    std::vector<frontend::ContractDefinition const *> const &contracts,
    langutil::CharStream const &stream, Action);

extern bool runYulToMLIRPass(yul::Object const &, langutil::CharStream const &,
                             yul::Dialect const &, Action);
} // namespace solidity::mlirgen
