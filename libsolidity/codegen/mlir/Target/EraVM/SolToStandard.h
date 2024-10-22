// This file is part of solidity.

// solidity is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// solidity is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with solidity.  If not, see <http://www.gnu.org/licenses/>.

// SPDX-License-Identifier: GPL-3.0

//
// EraVM specific lowering of sol dialect
//

#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace eravm {

void populateStage1Pats(mlir::RewritePatternSet &pats,
                        mlir::TypeConverter &tyConv);
void populateStage2Pats(mlir::RewritePatternSet &pats);

/// Adds the conversion patterns of func and related ops in the sol dialect.
void populateFuncPats(mlir::RewritePatternSet &pats,
                      mlir::TypeConverter &tyConv);

} // namespace eravm
