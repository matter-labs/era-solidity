#!/usr/bin/env sh
#

set -e

REPO_ROOT="$(dirname "$0")"/..
(
    cd "$REPO_ROOT"
    version=$(scripts/get_version.sh)
    commithash=$(git rev-parse --short=8 HEAD)
    commitdate=$(git show --format=%ci HEAD | head -n 1 | cut - -b1-10 | sed -e 's/-0?/./' | sed -e 's/-0?/./')

    # file exists and has zero size -> not a prerelease
    if [ -e prerelease.txt ] && [ ! -s prerelease.txt ]
    then
        versionstring="$version"
    else
        versionstring="$version-nightly-$commitdate-$commithash"
    fi

    TEMPDIR=$(mktemp -d)
    SOLDIR="$TEMPDIR/solidity_$versionstring/"
    mkdir "$SOLDIR"
    # Ensure that submodules are initialized.
    git submodule update --init --recursive
    # Store the current source
    git checkout-index -a --prefix="$SOLDIR"
    # shellcheck disable=SC2016
    SOLDIR="$SOLDIR" git submodule foreach 'git checkout-index -a --prefix="$SOLDIR/$sm_path/"'
    # Store the commit hash
    echo "$commithash" > "$SOLDIR/commit_hash.txt"
    if [ -e prerelease.txt ] && [ ! -s prerelease.txt ]
    then
        cp prerelease.txt "$SOLDIR/"
    fi
    mkdir -p "$REPO_ROOT/upload"
    tar --owner 0 --group 0 -czf "$REPO_ROOT/upload/solidity_$versionstring.tar.gz" -C "$TEMPDIR" "solidity_$versionstring"
    rm -r "$TEMPDIR"
)
