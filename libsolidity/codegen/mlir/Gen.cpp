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

#include <libsolidity/codegen/mlir/Gen.h>

#include <liblangutil/CharStream.h>
#include <liblangutil/SourceLocation.h>
#include <libsolidity/ast/AST.h>
#include <libsolidity/ast/ASTVisitor.h>

#include "Solidity/SolidityOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace solidity::langutil;
using namespace solidity::frontend;

namespace solidity::frontend
{

class MLIRGen: public ASTConstVisitor
{
public:
	explicit MLIRGen(mlir::MLIRContext& _ctx, CharStream const& _stream): m_b(&_ctx), m_stream(_stream)
	{
		mod = mlir::ModuleOp::create(m_b.getUnknownLoc());
		m_b.setInsertionPointToEnd(mod.getBody());
	}

	void run(ContractDefinition const&);

	mlir::ModuleOp mod;

private:
	mlir::OpBuilder m_b;
	CharStream const& m_stream;

	/// Returns the mlir location for the solidity source location `_loc`
	mlir::Location loc(SourceLocation _loc)
	{
		// FIXME: Track _loc.end as well
		LineColumn lineCol = m_stream.translatePositionToLineColumn(_loc.start);
		return mlir::FileLineColLoc::get(m_b.getStringAttr(m_stream.name()), lineCol.line, lineCol.column);
	}

	/// Returns the corresponding mlir type for the solidity type `_ty`
	mlir::Type type(Type const* _ty);

	/// Returns the cast from `_val` to a value having type `_dstTy`
	mlir::Value genCast(mlir::Value _val, mlir::Type _dstTy);

	/// Returns the mlir expression for the literal `_lit`
	mlir::Value genExpr(Literal const* _lit);

	/// Returns the mlir expression from `_expr` and optionally casts it to `_resTy`
	mlir::Value genExpr(Expression const* _expr, std::optional<mlir::Type const> _resTy = std::nullopt);

	bool visit(Return const&) override;
	void run(FunctionDefinition const&);
};

}

mlir::Type MLIRGen::type(Type const* _ty)
{
	// Integer type
	if (auto* i = dynamic_cast<IntegerType const*>(_ty))
	{
		return m_b.getIntegerType(i->numBits());
	}
	// Rational number type
	else if (auto* ratNumTy = dynamic_cast<RationalNumberType const*>(_ty))
	{
		// TODO:
		if (ratNumTy->isFractional())
			solUnimplemented("Unhandled type\n");

		// Integral rational number type
		const IntegerType* intTy = ratNumTy->integerType();
		return m_b.getIntegerType(intTy->numBits());
	}
	// TODO:
	solUnimplemented("Unhandled type\n");
}

mlir::Value MLIRGen::genCast(mlir::Value _val, mlir::Type _dstTy)
{
	mlir::Type srcTy = _val.getType();

	// Don't cast if we're casting to the same type
	if (srcTy == _dstTy)
		return _val;

	// Casting between integers
	auto srcIntTy = srcTy.dyn_cast<mlir::IntegerType>();
	auto dstIntTy = _dstTy.dyn_cast<mlir::IntegerType>();
	if (srcIntTy && dstIntTy)
	{
		// Generate extends
		if (dstIntTy.getWidth() > srcIntTy.getWidth())
		{
			return dstIntTy.isSigned() ? m_b.create<mlir::arith::ExtSIOp>(_val.getLoc(), dstIntTy, _val)->getResult(0)
									   : m_b.create<mlir::arith::ExtUIOp>(_val.getLoc(), dstIntTy, _val)->getResult(0);
		}
		else
		{
			// TODO:
			solUnimplemented("Unhandled cast\n");
		}
	}

	// TODO:
	solUnimplemented("Unhandled cast\n");
}

mlir::Value MLIRGen::genExpr(Expression const* _expr, std::optional<mlir::Type const> _resTy)
{
	mlir::Value val;

	// Generate literals
	if (auto* lit = dynamic_cast<Literal const*>(_expr))
	{
		val = genExpr(lit);
	}

	// Generate cast (Optional)
	if (_resTy)
	{
		return genCast(val, *_resTy);
	}

	return val;
}

mlir::Value MLIRGen::genExpr(Literal const* _lit)
{
	mlir::Location lc = loc(_lit->location());
	Type const* ty = _lit->annotation().type;

	// Rational number literal
	if (auto* ratNumTy = dynamic_cast<RationalNumberType const*>(ty))
	{
		// TODO:
		if (ratNumTy->isFractional())
			solUnimplemented("Unhandled literal\n");

		auto* intTy = ratNumTy->integerType();
		u256 val = ty->literalValue(_lit);
		// TODO: Is there a faster way to convert boost::multiprecision::number to llvm::APInt?
		return m_b.create<
			mlir::arith::
				ConstantOp>(lc, m_b.getIntegerAttr(type(ty), llvm::APInt(intTy->numBits(), val.str(), /*radix=*/10)));
	}
	else
	{
		// TODO:
		solUnimplemented("Unhandled literal\n");
	}
}

bool MLIRGen::visit(Return const& _ret)
{
	auto currFunc = m_b.getBlock()->getParent()->getParentOfType<mlir::func::FuncOp>();

	// The function generator emits `ReturnOp` for empty result
	if (currFunc.getNumResults() == 0)
		return true;

	// Get the return type of the current function so that we can apply any
	// necessary cast operations
	mlir::Type currFuncResTy = currFunc.getFunctionType().getResult(0);
	mlir::Value expr = genExpr(_ret.expression(), currFuncResTy);

	m_b.create<mlir::func::ReturnOp>(loc(_ret.location()), expr);

	return true;
}

void MLIRGen::run(FunctionDefinition const& _func)
{
	std::vector<mlir::Type> inpTys, outTys;

	for (auto const& param: _func.parameters())
	{
		inpTys.push_back(type(param->annotation().type));
	}

	for (auto const& param: _func.returnParameters())
	{
		outTys.push_back(type(param->annotation().type));
	}

	// TODO:
	solUnimplementedAssert(outTys.size() <= 1, "TODO: Impl multivalued return");

	auto funcType = m_b.getFunctionType(inpTys, outTys);
	auto op = m_b.create<mlir::func::FuncOp>(loc(_func.location()), _func.name(), funcType);

	solUnimplementedAssert(inpTys.empty(), "TODO: Add inp args to entry block");
	mlir::Block* entryBlk = m_b.createBlock(&op.getRegion());
	m_b.setInsertionPointToStart(entryBlk);

	_func.accept(*this);

	// Generate empty return
	if (outTys.empty())
		m_b.create<mlir::func::ReturnOp>(loc(_func.location()));

	m_b.setInsertionPointAfter(op);
}

void MLIRGen::run(ContractDefinition const& _cont)
{
	auto op = m_b.create<mlir::solidity::ContractOp>(loc(_cont.location()), _cont.name());
	m_b.setInsertionPointToStart(op.getBody());

	for (auto* f: _cont.definedFunctions())
	{
		run(*f);
	}
	m_b.setInsertionPointAfter(op);
}

bool solidity::frontend::runMLIRGen(std::vector<ContractDefinition const*> const& _contracts, CharStream const& _stream)
{
	mlir::MLIRContext ctx;
	ctx.getOrLoadDialect<mlir::solidity::SolidityDialect>();
	ctx.getOrLoadDialect<mlir::func::FuncDialect>();
	ctx.getOrLoadDialect<mlir::arith::ArithmeticDialect>();

	MLIRGen gen(ctx, _stream);
	for (auto* contract: _contracts)
	{
		gen.run(*contract);
	}

	if (failed(mlir::verify(gen.mod)))
	{
		gen.mod.emitError("Module verification error");
		return false;
	}

	gen.mod.print(llvm::outs(), mlir::OpPrintingFlags().enableDebugInfo());
	llvm::outs() << "\n";
	llvm::outs().flush();
	return true;
}

void solidity::frontend::registerMLIRCLOpts() { mlir::registerAsmPrinterCLOptions(); }

bool solidity::frontend::parseMLIROpts(std::vector<const char*>& _argv)
{
	// ParseCommandLineOptions() expects argv[0] to be the name of a program
	std::vector<const char*> argv{"foo"};
	for (const char* arg: _argv)
	{
		argv.push_back(arg);
	}

	return llvm::cl::ParseCommandLineOptions(argv.size(), argv.data(), "Generic MLIR flags\n");
}
