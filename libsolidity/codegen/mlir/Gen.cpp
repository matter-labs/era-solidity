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

	void run(ContractDefinition const& _contract);

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

	void run(FunctionDefinition const& _function);
	void run(Block const& _block);

	bool visit(Block const& _block) override;
	bool visit(Assignment const& _assignment) override;
	bool visit(BinaryOperation const& _binOp) override;
};

}

bool MLIRGen::visit(BinaryOperation const& _binOp) { return true; }

bool MLIRGen::visit(Block const& _block) { return true; }

bool MLIRGen::visit(Assignment const& _assignment) { return true; }

void MLIRGen::run(Block const& _block) { _block.accept(*this); }

void MLIRGen::run(FunctionDefinition const& _function)
{
	std::vector<mlir::Type> inpTys, outTys;

	for (auto const& param: _function.parameters())
	{
		inpTys.push_back(type(param->annotation().type));
	}
	for (auto const& param: _function.returnParameters())
	{
		outTys.push_back(type(param->annotation().type));
	}

	auto funcType = m_b.getFunctionType(inpTys, outTys);
	auto op = m_b.create<mlir::func::FuncOp>(loc(_function.location().start), _function.name(), funcType);

	solUnimplementedAssert(inpTys.empty(), "TODO: Add inp args to entry block");
	mlir::Block* entryBlk = m_b.createBlock(&op.getRegion());
	m_b.setInsertionPointToStart(entryBlk);

	run(_function.body());

	if (outTys.empty())
	{
		m_b.create<mlir::func::ReturnOp>(loc(_function.location().end));
		m_b.setInsertionPointAfter(op);
	}
	else
	{
		solUnimplemented("TODO: Return codegen\n");
	}
}

void MLIRGen::run(ContractDefinition const& _contract)
{
	auto cont = m_b.create<mlir::solidity::ContractOp>(loc(_contract.location().start), _contract.name());
	m_b.setInsertionPointToStart(cont.getBody());

	for (auto* f: _contract.definedFunctions())
	{
		run(*f);
	}
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
