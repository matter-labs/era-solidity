#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# This file is part of solidity.
#
# solidity is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# solidity is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with solidity.  If not, see <http://www.gnu.org/licenses/>
#
# (c) 2019 solidity contributors.
#------------------------------------------------------------------------------

# Disable shellcheck errors on quoted special chars like backticks. Too many false-positives.
# shellcheck disable=SC2016

set -e
# Temporary(?) fix to up the heap limit for node in order to prevent 'out of heap errors'
export NODE_OPTIONS="--max-old-space-size=4096"

source scripts/common.sh
source scripts/externalTests/common.sh

REPO_ROOT=$(realpath "$(dirname "$0")/../..")

verify_input "$@"
BINARY_TYPE="$1"
BINARY_PATH="$(realpath "$2")"
SELECTED_PRESETS="$3"

function compile_fn { npm run compile; }
function test_fn { npm test; }

function zeppelin_test
{
    local repo="https://github.com/OpenZeppelin/openzeppelin-contracts.git"
    local ref="<latest-release>"
    local config_file="hardhat.config.js"

    local compile_only_presets=(
        #ir-no-optimize           # Compilation fails with "Contract initcode size is 49410 bytes and exceeds 49152 bytes (a limit introduced in Shanghai)."
        ir-optimize-evm-only      # FIXME: A few tests fail with "Transaction: ... exited with an error (status 0) after consuming all gas."
)
    local settings_presets=(
        "${compile_only_presets[@]}"
        ir-optimize-evm+yul
        #legacy-no-optimize       # FIXME: Fails with stack too deep
        #legacy-optimize-evm-only # FIXME: Fails with stack too deep
        legacy-optimize-evm+yul
    )

    [[ $SELECTED_PRESETS != "" ]] || SELECTED_PRESETS=$(circleci_select_steps_multiarg "${settings_presets[@]}")
    print_presets_or_exit "$SELECTED_PRESETS"

    setup_solc "$DIR" "$BINARY_TYPE" "$BINARY_PATH"
    download_project "$repo" "$ref" "$DIR"

    # Disable tests that won't pass on the ir presets due to Hardhat heuristics. Note that this also disables
    # them for other presets but that's fine - we want same code run for benchmarks to be comparable.
    # TODO: Remove the lines below when Hardhat adjusts heuristics for IR (https://github.com/nomiclabs/hardhat/issues/3750).
    sed -i "s|it(\('proxy admin cannot call delegated functions',\)|it.skip(\1|g" test/proxy/transparent/TransparentUpgradeableProxy.behaviour.js
    sed -i "s|describe(\('when the given implementation is the zero address'\)|describe.skip(\1|g" test/proxy/transparent/TransparentUpgradeableProxy.behaviour.js
    sed -i "s|describe(\('when the new proposed admin is the zero address'\)|describe.skip(\1|g" test/proxy/transparent/TransparentUpgradeableProxy.behaviour.js
    # In some cases Hardhat does not detect revert reasons properly via IR.
    sed -i "s|it(\('prevent unauthorized maintenance'\)|it.skip(\1|g" test/governance/TimelockController.test.js
    sed -i "s|it(\('cannot cancel invalid operation'\)|it.skip(\1|g" test/governance/TimelockController.test.js
    sed -i "s|it(\('cannot call onlyInitializable function outside the scope of an initializable function'\)|it.skip(\1|g" test/proxy/utils/Initializable.test.js
    sed -i "s|it(\('reverts when sending non-zero amounts'\)|it.skip(\1|g" test/utils/Address.test.js
    sed -i "s|it(\('reverts when sending more than the balance'\)|it.skip(\1|g" test/utils/Address.test.js
    sed -i "s|it(\('fails deploying a contract if the bytecode length is zero'\)|it.skip(\1|g" test/utils/Create2.test.js
    sed -i "s|it(\('fails deploying a contract if factory contract does not have sufficient balance'\)|it.skip(\1|g" test/utils/Create2.test.js
    sed -i "s|it(\('reverts when casting -1'\)|it.skip(\1|g" test/utils/math/SafeCast.test.js
    sed -i 's|it(\(`reverts when casting [^`]\+`\)|it.skip(\1|g' test/utils/math/SafeCast.test.js
    sed -i "s|it(\('reverts if index is greater than supply'\)|it.skip(\1|g" test/token/ERC721/ERC721.behavior.js
    sed -i "s|it(\('burns all tokens'\)|it.skip(\1|g" test/token/ERC721/ERC721.behavior.js
    sed -i "s|it(\('guards transfer against invalid user'\)|it.skip(\1|g" test/access/Ownable2Step.test.js
    sed -i "s|it(\('reverting initialization function'\)|it.skip(\1|g" test/proxy/beacon/BeaconProxy.test.js
    sed -i "s|describe(\('reverting initialization'\)|describe.skip(\1|g" test/proxy/Proxy.behaviour.js
    sed -i "s|it(\('does not allow remote callback'\)|it.skip(\1|g" test/utils/ReentrancyGuard.test.js

    # TODO: Remove when hardhat properly handle reverts of custom errors with via-ir enabled
    # and/or open-zeppelin fix https://github.com/OpenZeppelin/openzeppelin-contracts/issues/4349
    sed -i "s|it(\('cannot nest reinitializers'\)|it.skip(\1|g" test/proxy/utils/Initializable.test.js
    sed -i "s|it(\('prevents re-initialization'\)|it.skip(\1|g" test/proxy/utils/Initializable.test.js
    sed -i "s|it(\('can lock contract after initialization'\)|it.skip(\1|g" test/proxy/utils/Initializable.test.js
    sed -i "s|it(\('calling upgradeTo on the implementation reverts'\)|it.skip(\1|g" test/proxy/utils/UUPSUpgradeable.test.js
    sed -i "s|it(\('calling upgradeToAndCall on the implementation reverts'\)|it.skip(\1|g" test/proxy/utils/UUPSUpgradeable.test.js
    sed -i "s|it(\('calling upgradeTo from a contract that is not an ERC1967 proxy\)|it.skip(\1|g" test/proxy/utils/UUPSUpgradeable.test.js
    sed -i "s|it(\('calling upgradeToAndCall from a contract that is not an ERC1967 proxy\)|it.skip(\1|g" test/proxy/utils/UUPSUpgradeable.test.js
    sed -i "s|it(\('rejects overflow'\)|it.skip(\1|g" test/token/ERC20/ERC20.test.js
    sed -i "s|it(\('decimals overflow'\)|it.skip(\1|g" test/token/ERC20/extensions/ERC4626.test.js

    # Here only the testToInt(248) and testToInt(256) cases fail so change the loop range to skip them
    sed -i "s|range(8, 256, 8)\(.forEach(bits => testToInt(bits));\)|range(8, 240, 8)\1|" test/utils/math/SafeCast.test.js

    neutralize_package_json_hooks
    force_hardhat_compiler_binary "$config_file" "$BINARY_TYPE" "$BINARY_PATH"
    force_hardhat_compiler_settings "$config_file" "$(first_word "$SELECTED_PRESETS")"
    npm install
    # We require to install hardhat 2.20.0 due to support for evm version cancun, otherwise we get the following error:
    # Invalid value {"blockGasLimit":10000000,"allowUnlimitedContractSize":true,"hardfork":"cancun"} for HardhatConfig.networks.hardhat - Expected a value of type HardhatNetworkConfig.
    # See: https://github.com/NomicFoundation/hardhat/issues/4176
    npm install hardhat@2.20.0

    replace_version_pragmas
    for preset in $SELECTED_PRESETS; do
        hardhat_run_test "$config_file" "$preset" "${compile_only_presets[*]}" compile_fn test_fn
        store_benchmark_report hardhat zeppelin "$repo" "$preset"
    done
}

external_test Zeppelin zeppelin_test
