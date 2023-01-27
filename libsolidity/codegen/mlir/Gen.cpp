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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
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

	mlir::Location loc(int _loc)
	{
		LineColumn lineCol = m_stream.translatePositionToLineColumn(_loc);
		return mlir::FileLineColLoc::get(m_b.getStringAttr(m_stream.name()), lineCol.line, lineCol.column);
	}

	mlir::Type type(Type const* ty)
	{
		if (auto* i = dynamic_cast<IntegerType const*>(ty))
		{
			return m_b.getIntegerType(i->numBits());
		}
		solUnimplemented("Unhandled type\n");
	}

	void run(FunctionDefinition const&);
	void run(Block const&);

	bool visit(Block const&) override;
	bool visit(Assignment const&) override;
	bool visit(BinaryOperation const&) override;
};

}

bool MLIRGen::visit(BinaryOperation const& _binOp) { return true; }

bool MLIRGen::visit(Block const& _blk) { return true; }

bool MLIRGen::visit(Assignment const& _assgn) { return true; }

void MLIRGen::run(Block const& _blk) { _blk.accept(*this); }

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

	auto funcType = m_b.getFunctionType(inpTys, outTys);
	auto op = m_b.create<mlir::func::FuncOp>(loc(_func.location().start), _func.name(), funcType);

	solUnimplementedAssert(inpTys.empty(), "TODO: Add inp args to entry block");
	mlir::Block* entryBlk = m_b.createBlock(&op.getRegion());
	m_b.setInsertionPointToStart(entryBlk);

	run(_func.body());

	if (outTys.empty())
	{
		m_b.create<mlir::func::ReturnOp>(loc(_func.location().end));
		m_b.setInsertionPointAfter(op);
	}
	else
	{
		solUnimplemented("TODO: Return codegen\n");
	}
}

void MLIRGen::run(ContractDefinition const& _cont)
{
	auto op = m_b.create<mlir::solidity::ContractOp>(loc(_cont.location().start), _cont.name());
	m_b.setInsertionPointToStart(op.getBody());

	for (auto* f: _cont.definedFunctions())
	{
		run(*f);
	}
	m_b.setInsertionPointAfter(op);
}

void solidity::frontend::runMLIRGen(std::vector<ContractDefinition const*> const& _contracts, CharStream const& _stream)
{
	mlir::MLIRContext ctx;
	ctx.getOrLoadDialect<mlir::solidity::SolidityDialect>();
	ctx.getOrLoadDialect<mlir::func::FuncDialect>();

	MLIRGen gen(ctx, _stream);
	for (auto* contract: _contracts)
	{
		gen.run(*contract);
	}

	gen.mod.print(llvm::errs(), mlir::OpPrintingFlags().enableDebugInfo());

	if (failed(mlir::verify(gen.mod)))
	{
		gen.mod.emitError("Module verification error");
	}
}
