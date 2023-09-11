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
 * @author Alex Beregszaszi
 * @date 2016
 * Standard JSON compiler interface.
 */

#include <libsolidity/interface/StandardCompiler.h>

#include <libsolidity/ast/ASTJsonConverter.h>
#include <liblangutil/SourceReferenceFormatter.h>
#include <libevmasm/Instruction.h>
#include <libdevcore/JSON.h>
#include <libdevcore/Keccak256.h>

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/optional.hpp>
#include <algorithm>

using namespace std;
using namespace dev;
using namespace langutil;
using namespace dev::solidity;

namespace {

Json::Value formatError(
	bool _warning,
	string const& _type,
	string const& _component,
	string const& _message,
	string const& _formattedMessage = "",
	Json::Value const& _sourceLocation = Json::Value()
)
{
	Json::Value error = Json::objectValue;
	error["type"] = _type;
	error["component"] = _component;
	error["severity"] = _warning ? "warning" : "error";
	error["message"] = _message;
	error["formattedMessage"] = (_formattedMessage.length() > 0) ? _formattedMessage : _message;
	if (_sourceLocation.isObject())
		error["sourceLocation"] = _sourceLocation;
	return error;
}

Json::Value formatFatalError(string const& _type, string const& _message)
{
	Json::Value output = Json::objectValue;
	output["errors"] = Json::arrayValue;
	output["errors"].append(formatError(false, _type, "general", _message));
	return output;
}

Json::Value formatErrorWithException(
	Exception const& _exception,
	bool const& _warning,
	string const& _type,
	string const& _component,
	string const& _message
)
{
	string message;
	string formattedMessage = SourceReferenceFormatter::formatExceptionInformation(_exception, _type);

	// NOTE: the below is partially a copy from SourceReferenceFormatter
	SourceLocation const* location = boost::get_error_info<errinfo_sourceLocation>(_exception);

	if (string const* description = boost::get_error_info<errinfo_comment>(_exception))
		message = ((_message.length() > 0) ? (_message + ":") : "") + *description;
	else
		message = _message;

	Json::Value sourceLocation;
	if (location && location->source && location->source->name() != "")
	{
		sourceLocation["file"] = location->source->name();
		sourceLocation["start"] = location->start;
		sourceLocation["end"] = location->end;
	}

	return formatError(_warning, _type, _component, message, formattedMessage, sourceLocation);
}

set<string> requestedContractNames(Json::Value const& _outputSelection)
{
	set<string> names;
	for (auto const& sourceName: _outputSelection.getMemberNames())
	{
		for (auto const& contractName: _outputSelection[sourceName].getMemberNames())
		{
			/// Consider the "all sources" shortcuts as requesting everything.
			if (contractName == "*" || contractName == "")
				return set<string>();
			names.insert((sourceName == "*" ? "" : sourceName) + ":" + contractName);
		}
	}
	return names;
}

/// Returns true iff @a _hash (hex with 0x prefix) is the Keccak256 hash of the binary data in @a _content.
bool hashMatchesContent(string const& _hash, string const& _content)
{
	try
	{
		return dev::h256(_hash) == dev::keccak256(_content);
	}
	catch (dev::BadHexCharacter const&)
	{
		return false;
	}
}

bool isArtifactRequested(Json::Value const& _outputSelection, string const& _artifact)
{
	for (auto const& artifact: _outputSelection)
		/// @TODO support sub-matching, e.g "evm" matches "evm.assembly"
		if (artifact == "*" || artifact == _artifact)
			return true;
	return false;
}

///
/// @a _outputSelection is a JSON object containining a two-level hashmap, where the first level is the filename,
/// the second level is the contract name and the value is an array of artifact names to be requested for that contract.
/// @a _file is the current file
/// @a _contract is the current contract
/// @a _artifact is the current artifact name
///
/// @returns true if the @a _outputSelection has a match for the requested target in the specific file / contract.
///
/// In @a _outputSelection the use of '*' as a wildcard is permitted.
///
/// @TODO optimise this. Perhaps flatten the structure upfront.
///
bool isArtifactRequested(Json::Value const& _outputSelection, string const& _file, string const& _contract, string const& _artifact)
{
	if (!_outputSelection.isObject())
		return false;

	for (auto const& file: { _file, string("*") })
		if (_outputSelection.isMember(file) && _outputSelection[file].isObject())
		{
			/// For SourceUnit-level targets (such as AST) only allow empty name, otherwise
			/// for Contract-level targets try both contract name and wildcard
			vector<string> contracts{ _contract };
			if (!_contract.empty())
				contracts.push_back("*");
			for (auto const& contract: contracts)
				if (
					_outputSelection[file].isMember(contract) &&
					_outputSelection[file][contract].isArray() &&
					isArtifactRequested(_outputSelection[file][contract], _artifact)
				)
					return true;
		}

	return false;
}

bool isArtifactRequested(Json::Value const& _outputSelection, string const& _file, string const& _contract, vector<string> const& _artifacts)
{
	for (auto const& artifact: _artifacts)
		if (isArtifactRequested(_outputSelection, _file, _contract, artifact))
			return true;
	return false;
}

/// @returns true if any binary was requested, i.e. we actually have to perform compilation.
bool isBinaryRequested(Json::Value const& _outputSelection)
{
	if (!_outputSelection.isObject())
		return false;

	// This does not inculde "evm.methodIdentifiers" on purpose!
	static vector<string> const outputsThatRequireBinaries{
		"*",
		"evm.deployedBytecode", "evm.deployedBytecode.object", "evm.deployedBytecode.opcodes",
		"evm.deployedBytecode.sourceMap", "evm.deployedBytecode.linkReferences",
		"evm.bytecode", "evm.bytecode.object", "evm.bytecode.opcodes", "evm.bytecode.sourceMap",
		"evm.bytecode.linkReferences",
		"evm.gasEstimates", "evm.legacyAssembly", "evm.assembly"
	};

	for (auto const& fileRequests: _outputSelection)
		for (auto const& requests: fileRequests)
			for (auto const& output: outputsThatRequireBinaries)
				if (isArtifactRequested(requests, output))
					return true;
	return false;
}

Json::Value formatLinkReferences(std::map<size_t, std::string> const& linkReferences)
{
	Json::Value ret(Json::objectValue);

	for (auto const& ref: linkReferences)
	{
		string const& fullname = ref.second;
		size_t colon = fullname.rfind(':');
		solAssert(colon != string::npos, "");
		string file = fullname.substr(0, colon);
		string name = fullname.substr(colon + 1);

		Json::Value fileObject = ret.get(file, Json::objectValue);
		Json::Value libraryArray = fileObject.get(name, Json::arrayValue);

		Json::Value entry = Json::objectValue;
		entry["start"] = Json::UInt(ref.first);
		entry["length"] = 20;

		libraryArray.append(entry);
		fileObject[name] = libraryArray;
		ret[file] = fileObject;
	}

	return ret;
}

Json::Value collectEVMObject(eth::LinkerObject const& _object, string const* _sourceMap)
{
	Json::Value output = Json::objectValue;
	output["object"] = _object.toHex();
	output["opcodes"] = solidity::disassemble(_object.bytecode);
	output["sourceMap"] = _sourceMap ? *_sourceMap : "";
	output["linkReferences"] = formatLinkReferences(_object.linkReferences);
	return output;
}

boost::optional<Json::Value> checkKeys(Json::Value const& _input, set<string> const& _keys, string const& _name)
{
	if (!!_input && !_input.isObject())
		return formatFatalError("JSONError", "\"" + _name + "\" must be an object");

	for (auto const& member: _input.getMemberNames())
		if (!_keys.count(member))
			return formatFatalError("JSONError", "Unknown key \"" + member + "\"");

	return boost::none;
}

boost::optional<Json::Value> checkRootKeys(Json::Value const& _input)
{
	static set<string> keys{"auxiliaryInput", "language", "settings", "sources"};
	return checkKeys(_input, keys, "root");
}

boost::optional<Json::Value> checkSourceKeys(Json::Value const& _input, string const& _name)
{
	static set<string> keys{"content", "keccak256", "urls"};
	return checkKeys(_input, keys, "sources." + _name);
}

boost::optional<Json::Value> checkAuxiliaryInputKeys(Json::Value const& _input)
{
	static set<string> keys{"smtlib2responses"};
	return checkKeys(_input, keys, "auxiliaryInput");
}

boost::optional<Json::Value> checkSettingsKeys(Json::Value const& _input)
{
	static set<string> keys{"evmVersion", "libraries", "metadata", "optimizer", "outputSelection", "remappings"};
	return checkKeys(_input, keys, "settings");
}

boost::optional<Json::Value> checkOptimizerKeys(Json::Value const& _input)
{
	static set<string> keys{"details", "enabled", "runs"};
	return checkKeys(_input, keys, "settings.optimizer");
}

boost::optional<Json::Value> checkOptimizerDetailsKeys(Json::Value const& _input)
{
	static set<string> keys{"peephole", "jumpdestRemover", "orderLiterals", "deduplicate", "cse", "constantOptimizer", "yul", "yulDetails"};
	return checkKeys(_input, keys, "settings.optimizer.details");
}

boost::optional<Json::Value> checkOptimizerDetail(Json::Value const& _details, std::string const& _name, bool& _setting)
{
	if (_details.isMember(_name))
	{
		if (!_details[_name].isBool())
			return formatFatalError("JSONError", "\"settings.optimizer.details." + _name + "\" must be Boolean");
		_setting = _details[_name].asBool();
	}
	return {};
}

boost::optional<Json::Value> checkMetadataKeys(Json::Value const& _input)
{
	if (_input.isObject() && _input.isMember("useLiteralContent") && !_input["useLiteralContent"].isBool())
		return formatFatalError("JSONError", "\"settings.metadata.useLiteralContent\" must be Boolean");
	static set<string> keys{"useLiteralContent"};
	return checkKeys(_input, keys, "settings.metadata");
}

boost::optional<Json::Value> checkOutputSelection(Json::Value const& _outputSelection)
{
	if (!!_outputSelection && !_outputSelection.isObject())
		return formatFatalError("JSONError", "\"settings.outputSelection\" must be an object");

	for (auto const& sourceName: _outputSelection.getMemberNames())
	{
		auto const& sourceVal = _outputSelection[sourceName];

		if (!sourceVal.isObject())
			return formatFatalError(
				"JSONError",
				"\"settings.outputSelection." + sourceName + "\" must be an object"
			);

		for (auto const& contractName: sourceVal.getMemberNames())
		{
			auto const& contractVal = sourceVal[contractName];

			if (!contractVal.isArray())
				return formatFatalError(
					"JSONError",
					"\"settings.outputSelection." +
					sourceName +
					"." +
					contractName +
					"\" must be a string array"
				);

			for (auto const& output: contractVal)
				if (!output.isString())
					return formatFatalError(
						"JSONError",
						"\"settings.outputSelection." +
						sourceName +
						"." +
						contractName +
						"\" must be a string array"
					);
		}
	}

	return boost::none;
}

/// Validates the optimizer settings and returns them in a parsed object.
/// On error returns the json-formatted error message.
boost::variant<OptimiserSettings, Json::Value> parseOptimizerSettings(Json::Value const& _jsonInput)
{
	if (auto result = checkOptimizerKeys(_jsonInput))
		return *result;

	OptimiserSettings settings = OptimiserSettings::none();

	if (_jsonInput.isMember("enabled"))
	{
		if (!_jsonInput["enabled"].isBool())
			return formatFatalError("JSONError", "The \"enabled\" setting must be a Boolean.");

		settings = _jsonInput["enabled"].asBool() ? OptimiserSettings::enabled() : OptimiserSettings::minimal();
	}

	if (_jsonInput.isMember("runs"))
	{
		if (!_jsonInput["runs"].isUInt())
			return formatFatalError("JSONError", "The \"runs\" setting must be an unsigned number.");
		settings.expectedExecutionsPerDeployment = _jsonInput["runs"].asUInt();
	}

	if (_jsonInput.isMember("details"))
	{
		Json::Value const& details = _jsonInput["details"];
		if (auto result = checkOptimizerDetailsKeys(details))
			return *result;

		if (auto error = checkOptimizerDetail(details, "peephole", settings.runPeephole))
			return *error;
		if (auto error = checkOptimizerDetail(details, "jumpdestRemover", settings.runJumpdestRemover))
			return *error;
		if (auto error = checkOptimizerDetail(details, "orderLiterals", settings.runOrderLiterals))
			return *error;
		if (auto error = checkOptimizerDetail(details, "deduplicate", settings.runDeduplicate))
			return *error;
		if (auto error = checkOptimizerDetail(details, "cse", settings.runCSE))
			return *error;
		if (auto error = checkOptimizerDetail(details, "constantOptimizer", settings.runConstantOptimiser))
			return *error;
		if (auto error = checkOptimizerDetail(details, "yul", settings.runYulOptimiser))
			return *error;
		if (details.isMember("yulDetails"))
		{
			if (!_jsonInput["yulDetails"].isObject())
				return formatFatalError("JSONError", "The \"yulDetails\" optimizer setting has to be a JSON object.");
			if (!_jsonInput["yulDetails"].getMemberNames().empty())
				return formatFatalError("JSONError", "The \"yulDetails\" optimizer setting cannot have any settings yet.");
		}
	}
	return { std::move(settings) };
}

}

boost::variant<StandardCompiler::InputsAndSettings, Json::Value> StandardCompiler::parseInput(Json::Value const& _input)
{
	InputsAndSettings ret;

	if (!_input.isObject())
		return formatFatalError("JSONError", "Input is not a JSON object.");

	if (auto result = checkRootKeys(_input))
		return *result;

	if (_input["language"] != "Solidity")
		return formatFatalError("JSONError", "Only \"Solidity\" is supported as a language.");

	Json::Value const& sources = _input["sources"];

	if (!sources.isObject() && !sources.isNull())
		return formatFatalError("JSONError", "\"sources\" is not a JSON object.");

	if (sources.empty())
		return formatFatalError("JSONError", "No input sources specified.");

	ret.errors = Json::arrayValue;

	for (auto const& sourceName: sources.getMemberNames())
	{
		string hash;

		if (auto result = checkSourceKeys(sources[sourceName], sourceName))
			return *result;

		if (sources[sourceName]["keccak256"].isString())
			hash = sources[sourceName]["keccak256"].asString();

		if (sources[sourceName]["content"].isString())
		{
			string content = sources[sourceName]["content"].asString();
			if (!hash.empty() && !hashMatchesContent(hash, content))
				ret.errors.append(formatError(
					false,
					"IOError",
					"general",
					"Mismatch between content and supplied hash for \"" + sourceName + "\""
				));
			else
				ret.sources[sourceName] = content;
		}
		else if (sources[sourceName]["urls"].isArray())
		{
			if (!m_readFile)
				return formatFatalError("JSONError", "No import callback supplied, but URL is requested.");

			bool found = false;
			vector<string> failures;

			for (auto const& url: sources[sourceName]["urls"])
			{
				if (!url.isString())
					return formatFatalError("JSONError", "URL must be a string.");
				ReadCallback::Result result = m_readFile(url.asString());
				if (result.success)
				{
					if (!hash.empty() && !hashMatchesContent(hash, result.responseOrErrorMessage))
						ret.errors.append(formatError(
							false,
							"IOError",
							"general",
							"Mismatch between content and supplied hash for \"" + sourceName + "\" at \"" + url.asString() + "\""
						));
					else
					{
						ret.sources[sourceName] =  result.responseOrErrorMessage;
						found = true;
						break;
					}
				}
				else
					failures.push_back("Cannot import url (\"" + url.asString() + "\"): " + result.responseOrErrorMessage);
			}

			for (auto const& failure: failures)
			{
				/// If the import succeeded, let mark all the others as warnings, otherwise all of them are errors.
				ret.errors.append(formatError(
					found ? true : false,
					"IOError",
					"general",
					failure
				));
			}
		}
		else
			return formatFatalError("JSONError", "Invalid input source specified.");
	}

	Json::Value const& auxInputs = _input["auxiliaryInput"];

	if (auto result = checkAuxiliaryInputKeys(auxInputs))
		return *result;

	if (!!auxInputs)
	{
		Json::Value const& smtlib2Responses = auxInputs["smtlib2responses"];
		if (!!smtlib2Responses)
		{
			if (!smtlib2Responses.isObject())
				return formatFatalError("JSONError", "\"auxiliaryInput.smtlib2responses\" must be an object.");

			for (auto const& hashString: smtlib2Responses.getMemberNames())
			{
				h256 hash;
				try
				{
					hash = h256(hashString);
				}
				catch (dev::BadHexCharacter const&)
				{
					return formatFatalError("JSONError", "Invalid hex encoding of SMTLib2 auxiliary input.");
				}

				if (!smtlib2Responses[hashString].isString())
					return formatFatalError(
						"JSONError",
						"\"smtlib2Responses." + hashString + "\" must be a string."
					);

				ret.smtLib2Responses[hash] = smtlib2Responses[hashString].asString();
			}
		}
	}

	Json::Value const& settings = _input.get("settings", Json::Value());

	if (auto result = checkSettingsKeys(settings))
		return *result;

	if (settings.isMember("evmVersion"))
	{
		if (!settings["evmVersion"].isString())
			return formatFatalError("JSONError", "evmVersion must be a string.");
		boost::optional<langutil::EVMVersion> version = langutil::EVMVersion::fromString(settings["evmVersion"].asString());
		if (!version)
			return formatFatalError("JSONError", "Invalid EVM version requested.");
		ret.evmVersion = *version;
	}

	if (settings.isMember("remappings") && !settings["remappings"].isArray())
		return formatFatalError("JSONError", "\"settings.remappings\" must be an array of strings.");

	for (auto const& remapping: settings.get("remappings", Json::Value()))
	{
		if (!remapping.isString())
			return formatFatalError("JSONError", "\"settings.remappings\" must be an array of strings");
		if (auto r = CompilerStack::parseRemapping(remapping.asString()))
			ret.remappings.emplace_back(std::move(*r));
		else
			return formatFatalError("JSONError", "Invalid remapping: \"" + remapping.asString() + "\"");
	}

	if (settings.isMember("optimizer"))
	{
		auto optimiserSettings = parseOptimizerSettings(settings["optimizer"]);
		if (optimiserSettings.type() == typeid(Json::Value))
			return boost::get<Json::Value>(std::move(optimiserSettings)); // was an error
		else
			ret.optimiserSettings = boost::get<OptimiserSettings>(std::move(optimiserSettings));
	}

	Json::Value jsonLibraries = settings.get("libraries", Json::Value(Json::objectValue));
	if (!jsonLibraries.isObject())
		return formatFatalError("JSONError", "\"libraries\" is not a JSON object.");
	for (auto const& sourceName: jsonLibraries.getMemberNames())
	{
		auto const& jsonSourceName = jsonLibraries[sourceName];
		if (!jsonSourceName.isObject())
			return formatFatalError("JSONError", "Library entry is not a JSON object.");
		for (auto const& library: jsonSourceName.getMemberNames())
		{
			if (!jsonSourceName[library].isString())
				return formatFatalError("JSONError", "Library address must be a string.");
			string address = jsonSourceName[library].asString();

			if (!boost::starts_with(address, "0x"))
				return formatFatalError(
					"JSONError",
					"Library address is not prefixed with \"0x\"."
				);

			if (address.length() != 42)
				return formatFatalError(
					"JSONError",
					"Library address is of invalid length."
				);

			try
			{
				// @TODO use libraries only for the given source
				ret.libraries[library] = h160(address);
			}
			catch (dev::BadHexCharacter const&)
			{
				return formatFatalError(
					"JSONError",
					"Invalid library address (\"" + address + "\") supplied."
				);
			}
		}
	}

	Json::Value metadataSettings = settings.get("metadata", Json::Value());

	if (auto result = checkMetadataKeys(metadataSettings))
		return *result;

	ret.metadataLiteralSources = metadataSettings.get("useLiteralContent", Json::Value(false)).asBool();

	Json::Value outputSelection = settings.get("outputSelection", Json::Value());

	if (auto jsonError = checkOutputSelection(outputSelection))
		return *jsonError;

	ret.outputSelection = std::move(outputSelection);

	return { std::move(ret) };
}

Json::Value StandardCompiler::compileSolidity(StandardCompiler::InputsAndSettings _inputsAndSettings)
{
	CompilerStack compilerStack(m_readFile);

	StringMap sourceList = std::move(_inputsAndSettings.sources);
	for (auto const& source: sourceList)
		compilerStack.addSource(source.first, source.second);
	for (auto const& smtLib2Response: _inputsAndSettings.smtLib2Responses)
		compilerStack.addSMTLib2Response(smtLib2Response.first, smtLib2Response.second);
	compilerStack.setEVMVersion(_inputsAndSettings.evmVersion);
	compilerStack.setRemappings(_inputsAndSettings.remappings);
	compilerStack.setOptimiserSettings(std::move(_inputsAndSettings.optimiserSettings));
	compilerStack.setLibraries(_inputsAndSettings.libraries);
	compilerStack.useMetadataLiteralSources(_inputsAndSettings.metadataLiteralSources);
	compilerStack.setRequestedContractNames(requestedContractNames(_inputsAndSettings.outputSelection));

	Json::Value errors = std::move(_inputsAndSettings.errors);

	bool const binariesRequested = isBinaryRequested(_inputsAndSettings.outputSelection);

	try
	{
		if (binariesRequested)
			compilerStack.compile();
		else
			compilerStack.parseAndAnalyze();

		for (auto const& error: compilerStack.errors())
		{
			Error const& err = dynamic_cast<Error const&>(*error);

			errors.append(formatErrorWithException(
				*error,
				err.type() == Error::Type::Warning,
				err.typeName(),
				"general",
				""
			));
		}
	}
	/// This is only thrown in a very few locations.
	catch (Error const& _error)
	{
		errors.append(formatErrorWithException(
			_error,
			false,
			_error.typeName(),
			"general",
			"Uncaught error: "
		));
	}
	/// This should not be leaked from compile().
	catch (FatalError const& _exception)
	{
		errors.append(formatError(
			false,
			"FatalError",
			"general",
			"Uncaught fatal error: " + boost::diagnostic_information(_exception)
		));
	}
	catch (CompilerError const& _exception)
	{
		errors.append(formatErrorWithException(
			_exception,
			false,
			"CompilerError",
			"general",
			"Compiler error (" + _exception.lineInfo() + ")"
		));
	}
	catch (InternalCompilerError const& _exception)
	{
		errors.append(formatErrorWithException(
			_exception,
			false,
			"InternalCompilerError",
			"general",
			"Internal compiler error (" + _exception.lineInfo() + ")"
		));
	}
	catch (UnimplementedFeatureError const& _exception)
	{
		errors.append(formatErrorWithException(
			_exception,
			false,
			"UnimplementedFeatureError",
			"general",
			"Unimplemented feature (" + _exception.lineInfo() + ")"
		));
	}
	catch (Exception const& _exception)
	{
		errors.append(formatError(
			false,
			"Exception",
			"general",
			"Exception during compilation: " + boost::diagnostic_information(_exception)
		));
	}
	catch (...)
	{
		errors.append(formatError(
			false,
			"Exception",
			"general",
			"Unknown exception during compilation."
		));
	}

	bool const analysisSuccess = compilerStack.state() >= CompilerStack::State::AnalysisSuccessful;
	bool const compilationSuccess = compilerStack.state() == CompilerStack::State::CompilationSuccessful;

	/// Inconsistent state - stop here to receive error reports from users
	if (((binariesRequested && !compilationSuccess) || !analysisSuccess) && errors.empty())
		return formatFatalError("InternalCompilerError", "No error reported, but compilation failed.");

	Json::Value output = Json::objectValue;

	if (errors.size() > 0)
		output["errors"] = std::move(errors);

	if (!compilerStack.unhandledSMTLib2Queries().empty())
		for (string const& query: compilerStack.unhandledSMTLib2Queries())
			output["auxiliaryInputRequested"]["smtlib2queries"]["0x" + keccak256(query).hex()] = query;

	output["sources"] = Json::objectValue;
	unsigned sourceIndex = 0;
	for (string const& sourceName: analysisSuccess ? compilerStack.sourceNames() : vector<string>())
	{
		Json::Value sourceResult = Json::objectValue;
		sourceResult["id"] = sourceIndex++;
		if (isArtifactRequested(_inputsAndSettings.outputSelection, sourceName, "", "ast"))
			sourceResult["ast"] = ASTJsonConverter(false, compilerStack.sourceIndices()).toJson(compilerStack.ast(sourceName));
		if (isArtifactRequested(_inputsAndSettings.outputSelection, sourceName, "", "legacyAST"))
			sourceResult["legacyAST"] = ASTJsonConverter(true, compilerStack.sourceIndices()).toJson(compilerStack.ast(sourceName));
		output["sources"][sourceName] = sourceResult;
	}

	Json::Value contractsOutput = Json::objectValue;
	for (string const& contractName: analysisSuccess ? compilerStack.contractNames() : vector<string>())
	{
		size_t colon = contractName.rfind(':');
		solAssert(colon != string::npos, "");
		string file = contractName.substr(0, colon);
		string name = contractName.substr(colon + 1);

		// ABI, documentation and metadata
		Json::Value contractData(Json::objectValue);
		if (isArtifactRequested(_inputsAndSettings.outputSelection, file, name, "abi"))
			contractData["abi"] = compilerStack.contractABI(contractName);
		if (isArtifactRequested(_inputsAndSettings.outputSelection, file, name, "metadata"))
			contractData["metadata"] = compilerStack.metadata(contractName);
		if (isArtifactRequested(_inputsAndSettings.outputSelection, file, name, "userdoc"))
			contractData["userdoc"] = compilerStack.natspecUser(contractName);
		if (isArtifactRequested(_inputsAndSettings.outputSelection, file, name, "devdoc"))
			contractData["devdoc"] = compilerStack.natspecDev(contractName);

		// EVM
		Json::Value evmData(Json::objectValue);
		// @TODO: add ir
		if (compilationSuccess && isArtifactRequested(_inputsAndSettings.outputSelection, file, name, "evm.assembly"))
			evmData["assembly"] = compilerStack.assemblyString(contractName, sourceList);
		if (compilationSuccess && isArtifactRequested(_inputsAndSettings.outputSelection, file, name, "evm.legacyAssembly"))
			evmData["legacyAssembly"] = compilerStack.assemblyJSON(contractName, sourceList);
		if (isArtifactRequested(_inputsAndSettings.outputSelection, file, name, "evm.methodIdentifiers"))
			evmData["methodIdentifiers"] = compilerStack.methodIdentifiers(contractName);
		if (compilationSuccess && isArtifactRequested(_inputsAndSettings.outputSelection, file, name, "evm.gasEstimates"))
			evmData["gasEstimates"] = compilerStack.gasEstimates(contractName);

		Json::Value extraMetadata = compilerStack.extraMetadata(contractName);
		if (compilationSuccess && !extraMetadata.empty())
			evmData["extraMetadata"] = extraMetadata;

		if (compilationSuccess && isArtifactRequested(
			_inputsAndSettings.outputSelection,
			file,
			name,
			{ "evm.bytecode", "evm.bytecode.object", "evm.bytecode.opcodes", "evm.bytecode.sourceMap", "evm.bytecode.linkReferences" }
		))
			evmData["bytecode"] = collectEVMObject(
				compilerStack.object(contractName),
				compilerStack.sourceMapping(contractName)
			);

		if (compilationSuccess && isArtifactRequested(
			_inputsAndSettings.outputSelection,
			file,
			name,
			{ "evm.deployedBytecode", "evm.deployedBytecode.object", "evm.deployedBytecode.opcodes", "evm.deployedBytecode.sourceMap", "evm.deployedBytecode.linkReferences" }
		))
			evmData["deployedBytecode"] = collectEVMObject(
				compilerStack.runtimeObject(contractName),
				compilerStack.runtimeSourceMapping(contractName)
			);

		if (!evmData.empty())
			contractData["evm"] = evmData;

		if (!contractData.empty())
		{
			if (!contractsOutput.isMember(file))
				contractsOutput[file] = Json::objectValue;
			contractsOutput[file][name] = contractData;
		}
	}
	if (!contractsOutput.empty())
		output["contracts"] = contractsOutput;

	return output;
}

Json::Value StandardCompiler::compile(Json::Value const& _input) noexcept
{
	try
	{
		auto parsed = parseInput(_input);
		if (parsed.type() == typeid(InputsAndSettings))
			return compileSolidity(boost::get<InputsAndSettings>(std::move(parsed)));
		else
			return boost::get<Json::Value>(std::move(parsed));
	}
	catch (Json::LogicError const& _exception)
	{
		return formatFatalError("InternalCompilerError", string("JSON logic exception: ") + _exception.what());
	}
	catch (Json::RuntimeError const& _exception)
	{
		return formatFatalError("InternalCompilerError", string("JSON runtime exception: ") + _exception.what());
	}
	catch (Exception const& _exception)
	{
		return formatFatalError("InternalCompilerError", "Internal exception in StandardCompiler::compile: " + boost::diagnostic_information(_exception));
	}
	catch (...)
	{
		return formatFatalError("InternalCompilerError", "Internal exception in StandardCompiler::compile");
	}
}

string StandardCompiler::compile(string const& _input) noexcept
{
	Json::Value input;
	string errors;
	try
	{
		if (!jsonParseStrict(_input, input, &errors))
			return jsonCompactPrint(formatFatalError("JSONError", errors));
	}
	catch (...)
	{
		return "{\"errors\":[{\"type\":\"JSONError\",\"component\":\"general\",\"severity\":\"error\",\"message\":\"Error parsing input JSON.\"}]}";
	}

	// cout << "Input: " << input.toStyledString() << endl;
	Json::Value output = compile(input);
	// cout << "Output: " << output.toStyledString() << endl;

	try
	{
		return jsonCompactPrint(output);
	}
	catch (...)
	{
		return "{\"errors\":[{\"type\":\"JSONError\",\"component\":\"general\",\"severity\":\"error\",\"message\":\"Error writing output JSON.\"}]}";
	}
}
