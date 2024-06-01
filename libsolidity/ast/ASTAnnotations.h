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
/**
 * @author Christian <c@ethdev.com>
 * @date 2015
 * Object containing the type and other annotations for the AST nodes.
 */

#pragma once

#include <libsolidity/ast/ASTForward.h>
#include <libsolidity/ast/ASTEnums.h>
#include <libsolidity/ast/ExperimentalFeatures.h>

#include <libsolutil/SetOnce.h>

#include <libyul/AsmData.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace solidity::yul
{
struct AsmAnalysisInfo;
struct Identifier;
struct Dialect;
}

namespace solidity::frontend
{

class Type;
using TypePointer = Type const*;

struct CallGraph;

struct ASTAnnotation
{
	virtual ~ASTAnnotation() = default;
};

struct DocTag
{
	std::string content;	///< The text content of the tag.
	std::string paramName;	///< Only used for @param, stores the parameter name.
};

struct StructurallyDocumentedAnnotation
{
	virtual ~StructurallyDocumentedAnnotation() = default;
	/// Mapping docstring tag name -> content.
	std::multimap<std::string, DocTag> docTags;
};

struct SourceUnitAnnotation: ASTAnnotation
{
	/// The "absolute" (in the compiler sense) path of this source unit.
	std::string path;
	/// The exported symbols (all global symbols).
	std::map<ASTString, std::vector<Declaration const*>> exportedSymbols;
	/// Experimental features.
	std::set<ExperimentalFeature> experimentalFeatures;
};

struct ScopableAnnotation
{
	/// The scope this declaration resides in. Can be nullptr if it is the global scope.
	/// Available only after name and type resolution step.
	ASTNode const* scope = nullptr;
	/// Pointer to the contract this declaration resides in. Can be nullptr if the current scope
	/// is not part of a contract. Available only after name and type resolution step.
	ContractDefinition const* contract = nullptr;
};

struct DeclarationAnnotation: ASTAnnotation, ScopableAnnotation
{
};

struct ImportAnnotation: DeclarationAnnotation
{
	/// The absolute path of the source unit to import.
	std::string absolutePath;
	/// The actual source unit.
	SourceUnit const* sourceUnit = nullptr;
};

struct TypeDeclarationAnnotation: DeclarationAnnotation
{
	/// The name of this type, prefixed by proper namespaces if globally accessible.
	std::string canonicalName;
};

struct ContractDefinitionAnnotation: TypeDeclarationAnnotation, StructurallyDocumentedAnnotation
{
	/// List of functions without a body. Can also contain functions from base classes.
	std::vector<FunctionDefinition const*> unimplementedFunctions;
	/// List of all (direct and indirect) base contracts in order from derived to
	/// base, including the contract itself.
	std::vector<ContractDefinition const*> linearizedBaseContracts;
	/// List of contracts this contract creates, i.e. which need to be compiled first.
	/// Also includes all contracts from @a linearizedBaseContracts.
	std::set<ContractDefinition const*> contractDependencies;
	/// Mapping containing the nodes that define the arguments for base constructors.
	/// These can either be inheritance specifiers or modifier invocations.
	std::map<FunctionDefinition const*, ASTNode const*> baseConstructorArguments;
	/// A graph with edges representing calls between functions that may happen during contract construction.
	util::SetOnce<std::shared_ptr<CallGraph const>> creationCallGraph;
	/// A graph with edges representing calls between functions that may happen in a deployed contract.
	util::SetOnce<std::shared_ptr<CallGraph const>> deployedCallGraph;
	/// Set of internal functions referenced as function pointers
	std::set<FunctionDefinition const*> intFuncPtrRefs;
};

struct CallableDeclarationAnnotation: DeclarationAnnotation
{
	/// The set of functions/modifiers/events this callable overrides.
	std::set<CallableDeclaration const*> baseFunctions;
};

struct FunctionDefinitionAnnotation: CallableDeclarationAnnotation, StructurallyDocumentedAnnotation
{
};

struct EventDefinitionAnnotation: CallableDeclarationAnnotation, StructurallyDocumentedAnnotation
{
};

struct ModifierDefinitionAnnotation: CallableDeclarationAnnotation, StructurallyDocumentedAnnotation
{
};

struct VariableDeclarationAnnotation: DeclarationAnnotation
{
	/// Type of variable (type of identifier referencing this variable).
	TypePointer type = nullptr;
	/// The set of functions this (public state) variable overrides.
	std::set<CallableDeclaration const*> baseFunctions;
};

struct StatementAnnotation: ASTAnnotation
{
};

struct InlineAssemblyAnnotation: StatementAnnotation
{
	struct ExternalIdentifierInfo
	{
		Declaration const* declaration = nullptr;
		bool isSlot = false; ///< Whether the storage slot of a variable is queried.
		bool isOffset = false; ///< Whether the intra-slot offset of a storage variable is queried.
		size_t valueSize = size_t(-1);
	};

	/// Mapping containing resolved references to external identifiers and their value size
	std::map<yul::Identifier const*, ExternalIdentifierInfo> externalReferences;
	/// Information generated during analysis phase.
	std::shared_ptr<yul::AsmAnalysisInfo> analysisInfo;
	/// The yul block of the InlineAssembly::operations() after optimizations.
	std::shared_ptr<yul::Block> optimizedOperations;
};

struct BlockAnnotation: StatementAnnotation, ScopableAnnotation
{
};

struct TryCatchClauseAnnotation: ASTAnnotation, ScopableAnnotation
{
};

struct ForStatementAnnotation: StatementAnnotation, ScopableAnnotation
{
};

struct ReturnAnnotation: StatementAnnotation
{
	/// Reference to the return parameters of the function.
	ParameterList const* functionReturnParameters = nullptr;
};

struct TypeNameAnnotation: ASTAnnotation
{
	/// Type declared by this type name, i.e. type of a variable where this type name is used.
	/// Set during reference resolution stage.
	TypePointer type = nullptr;
};

struct UserDefinedTypeNameAnnotation: TypeNameAnnotation
{
	/// Referenced declaration, set during reference resolution stage.
	Declaration const* referencedDeclaration = nullptr;
	/// Stores a reference to the current contract.
	/// This is needed because types of base contracts change depending on the context.
	ContractDefinition const* contractScope = nullptr;
};

struct ExpressionAnnotation: ASTAnnotation
{
	/// Inferred type of the expression.
	TypePointer type = nullptr;
	/// Whether the expression is a constant variable
	bool isConstant = false;
	/// Whether the expression is pure, i.e. compile-time constant.
	bool isPure = false;
	/// Whether it is an LValue (i.e. something that can be assigned to).
	bool isLValue = false;
	/// Whether the expression is used in a context where the LValue is actually required.
	bool lValueRequested = false;

	/// Types and - if given - names of arguments if the expr. is a function
	/// that is called, used for overload resoultion
	std::optional<FuncCallArguments> arguments;

	/// True if the expression consists solely of the name of the function and the function is called immediately
	/// instead of being stored or processed. The name may be qualified with the name of a contract, library
	/// module, etc., that clarifies the scope. For example: `m.L.f()`, where `m` is a module, `L` is a library
	/// and `f` is a function is a direct call. This means that the function to be called is known at compilation
	/// time and it's not necessary to rely on any runtime dispatch mechanism to resolve it.
	/// Note that even the simplest expressions, like `(f)()`, result in an indirect call even if they consist of
	/// values known at compilation time.
	bool calledDirectly = false;
};

struct IdentifierAnnotation: ExpressionAnnotation
{
	/// Referenced declaration, set at latest during overload resolution stage.
	Declaration const* referencedDeclaration = nullptr;
	/// What kind of lookup needs to be done (static, virtual, super) find the declaration.
	util::SetOnce<VirtualLookup> requiredLookup;
	/// List of possible declarations it could refer to.
	std::vector<Declaration const*> overloadedDeclarations;
};

struct MemberAccessAnnotation: ExpressionAnnotation
{
	/// Referenced declaration, set at latest during overload resolution stage.
	Declaration const* referencedDeclaration = nullptr;
	/// What kind of lookup needs to be done (static, virtual, super) find the declaration.
	util::SetOnce<VirtualLookup> requiredLookup;
};

struct BinaryOperationAnnotation: ExpressionAnnotation
{
	/// The common type that is used for the operation, not necessarily the result type (which
	/// e.g. for comparisons is bool).
	TypePointer commonType = nullptr;
};

enum class FunctionCallKind
{
	Unset,
	FunctionCall,
	TypeConversion,
	StructConstructorCall
};

struct FunctionCallAnnotation: ExpressionAnnotation
{
	FunctionCallKind kind = FunctionCallKind::Unset;
	/// If true, this is the external call of a try statement.
	bool tryCall = false;

	// HACK!
	// We track the success tag here for the `TryStatement` lowering. This is to avoid the redundant status check and
	// the conditional jump. Such patterns can confuse the zksolc translator.
	//
	// uint32_t since Assembly::new[Push]Tag() asserts that the tag is 32 bits.
	std::optional<uint32_t> tryCallSuccessTag;
};

}
