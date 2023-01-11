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
 * Solidity dialect
 */

#ifndef SOLIDITY_DIALECT
#define SOLIDITY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Solidity dialect definition.
//===----------------------------------------------------------------------===//

def Solidity_Dialect : Dialect {
    let name = "solidity";
    let summary = "Solidity dialect";
    let description = [{
		This dialect has high level ops for representing solidity.
    }];
    let cppNamespace = "::mlir::solidity";

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

//===----------------------------------------------------------------------===//
// Base solidity operation definition.
//===----------------------------------------------------------------------===//

class Solidity_Op<string mnemonic, list<Trait> traits = []> :
        Op<Solidity_Dialect, mnemonic, traits>;

#endif // SOLIDITY_DIALECT
