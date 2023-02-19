# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'solidity-mlir'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.sol']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.solidity_obj_root, 'test/lit')

solc_bin_dir = config.solidity_obj_root + '/solc'
llvm_config.with_environment('PATH', solc_bin_dir, append_path=True)
tool_dirs = [solc_bin_dir]
tools = ['solc']
llvm_config.add_tool_substitutions(tools, tool_dirs)
