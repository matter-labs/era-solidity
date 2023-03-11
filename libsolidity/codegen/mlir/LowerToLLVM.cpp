/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0
/**
 * LLVM dialect generator
 */

#include "libsolidity/codegen/mlir/Passes.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"

#include "mlir/Support/LogicalResult.h"

#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>

using namespace mlir;

namespace
{
struct LowerToLLVMPass: public PassWrapper<LowerToLLVMPass, OperationPass<ModuleOp>>
{
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLLVMPass)

	void getDependentDialects(DialectRegistry& reg) const override { reg.insert<LLVM::LLVMDialect>(); }

	void runOnOperation() override
	{
		// We only lower till the llvm dialect
		LLVMConversionTarget llConv(getContext());
		llConv.addLegalOp<ModuleOp>();
		LLVMTypeConverter llTyConv(&getContext());

		// Lower arith, memref and func dialects to the llvm dialect
		RewritePatternSet pats(&getContext());
		arith::populateArithmeticToLLVMConversionPatterns(llTyConv, pats);
		populateMemRefToLLVMConversionPatterns(llTyConv, pats);
		populateFuncToLLVMConversionPatterns(llTyConv, pats);

		ModuleOp mod = getOperation();
		if (failed(applyFullConversion(mod, llConv, std::move(pats))))
			signalPassFailure();
	}
};
}

std::unique_ptr<Pass> solidity::createLowerToLLVMPass() { return std::make_unique<LowerToLLVMPass>(); }
