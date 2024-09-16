#!/usr/bin/env bash
##############################################################################
## This is used to package .deb packages and upload them to the launchpad
## ppa servers for building.
##
## It will clone the Z3 git from github on the specified version tag,
## create a source archive and push it to the ubuntu ppa servers.
##
## This requires the following entries in /etc/dput.cf:
##
##  [cpp-build-deps]
##  fqdn            = ppa.launchpad.net
##  method          = ftp
##  incoming        = ~ethereum/cpp-build-deps
##  login           = anonymous
##
## To interact with launchpad, you need to set the variables $LAUNCHPAD_EMAIL
## and $LAUNCHPAD_KEYID in the file .release_ppa_auth in the root directory of
## the project to your launchpad email and pgp keyid.
## This could for example look like this:
##
##  LAUNCHPAD_EMAIL=your-launchpad-email@ethereum.org
##  LAUNCHPAD_KEYID=123ABCFFFFFFFF
##
##############################################################################

set -e

packagename=z3-static
version="$1"

REPO_ROOT="$(dirname "$0")/../.."

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/common.sh"

[[ $version != "" ]] || fail "Usage: $0 <version-without-leading-v>"

sourcePPAConfig

# Sanity check
checkDputEntries "\[cpp-build-deps\]"

DISTRIBUTIONS="focal jammy noble oracular"

for distribution in $DISTRIBUTIONS
do
cd /tmp/
rm -rf "$distribution"
mkdir "$distribution"
cd "$distribution"

pparepo=cpp-build-deps
ppafilesurl=https://launchpad.net/~ethereum/+archive/ubuntu/${pparepo}/+files

# Fetch source
git clone --branch "z3-${version}" https://github.com/Z3Prover/z3.git
cd z3

debversion="${version}"

CMAKE_OPTIONS="-DZ3_BUILD_LIBZ3_SHARED=OFF -DCMAKE_BUILD_TYPE=Release"

# gzip will create different tars all the time and we are not allowed
# to upload the same file twice with different contents, so we only
# create it once.
if [ ! -e "/tmp/${packagename}_${debversion}.orig.tar.gz" ]
then
    tar --exclude .git -czf "/tmp/${packagename}_${debversion}.orig.tar.gz" .
fi
cp "/tmp/${packagename}_${debversion}.orig.tar.gz" ../

# Create debian package information

mkdir debian
echo 9 > debian/compat
# TODO: the Z3 packages have different build dependencies
cat <<EOF > debian/control
Source: z3-static
Section: science
Priority: extra
Maintainer: Daniel Kirchner <daniel@ekpyron.org>
Build-Depends: debhelper (>= 9.0.0),
               cmake,
               g++ (>= 5.0),
               git,
               libgmp-dev,
               dh-python,
               python3
Standards-Version: 3.9.6
Homepage: https://github.com/Z3Prover/z3
Vcs-Git: https://github.com/Z3Prover/z3.git
Vcs-Browser: https://github.com/Z3Prover/z3

Package: z3-static
Architecture: any
Breaks: z3
Replaces: z3
Depends: \${misc:Depends}, \${shlibs:Depends}
Description: theorem prover from Microsoft Research
 Z3 is a state-of-the art theorem prover from Microsoft Research. It can be
 used to check the satisfiability of logical formulas over one or more
 theories. Z3 offers a compelling match for software analysis and verification
 tools, since several common software constructs map directly into supported
 theories.
 .
 The Z3 input format is an extension of the one defined by the SMT-LIB 2.0
 standard.


Package: libz3-static-dev
Section: libdevel
Architecture: any-amd64
Breaks: libz3-dev
Replaces: libz3-dev
Multi-Arch: same
Depends: \${shlibs:Depends}, \${misc:Depends}
Description: theorem prover from Microsoft Research - development files (static library)
 Z3 is a state-of-the art theorem prover from Microsoft Research. It can be
 used to check the satisfiability of logical formulas over one or more
 theories. Z3 offers a compelling match for software analysis and verification
 tools, since several common software constructs map directly into supported
 theories.
 .
 This package can be used to invoke Z3 via its C++ API.
EOF
cat <<EOF > debian/rules
#!/usr/bin/make -f
# -*- makefile -*-
# Sample debian/rules that uses debhelper.
#
# This file was originally written by Joey Hess and Craig Small.
# As a special exception, when this file is copied by dh-make into a
# dh-make output file, you may use that output file without restriction.
# This special exception was added by Craig Small in version 0.37 of dh-make.
#
# Modified to make a template file for a multi-binary package with separated
# build-arch and build-indep targets  by Bill Allombert 2001

# Uncomment this to turn on verbose mode.
export DH_VERBOSE=1

# This has to be exported to make some magic below work.
export DH_OPTIONS


%:
	dh \$@ --buildsystem=cmake

override_dh_auto_test:

override_dh_shlibdeps:
	dh_shlibdeps --dpkg-shlibdeps-params=--ignore-missing-info

override_dh_auto_configure:
	dh_auto_configure -- ${CMAKE_OPTIONS}

override_dh_auto_install:
	dh_auto_install --destdir debian/tmp
EOF
cat <<EOF > debian/libz3-static-dev.install
usr/include/*
usr/lib/*/libz3.a
usr/lib/*/cmake/z3/*
EOF
cat <<EOF > debian/z3-static.install
usr/bin/z3
EOF
cat <<EOF > debian/copyright
Format: http://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: z3
Source: https://github.com/Z3Prover/z3

Files: *
Copyright: Microsoft Corporation
License: Expat
 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:
 .
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 .
 THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

Files: debian/*
Copyright: 2019 Ethereum
License: GPL-3.0+
This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 .
 This package is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 .
 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
 .
 On Debian systems, the complete text of the GNU General
 Public License version 3 can be found in "/usr/share/common-licenses/GPL-3".
EOF
cat <<EOF > debian/changelog
z3-static (0.0.1-1ubuntu0) saucy; urgency=low

  * Initial release.

 -- Daniel <daniel@ekpyron.org>  Mon, 03 Jun 2019 14:50:20 +0000
EOF
mkdir debian/source
echo "3.0 (quilt)" > debian/source/format
chmod +x debian/rules

versionsuffix=1ubuntu0~${distribution}
EMAIL="$LAUNCHPAD_EMAIL" dch -v "1:${debversion}-${versionsuffix}" "build of ${version}"

# build source package
# If packages is rejected because original source is already present, add
# -sd to remove it from the .changes file
# -d disables the build dependencies check
debuild -S -d -sa -us -uc

# prepare .changes file for Launchpad
sed -i -e "s/UNRELEASED/${distribution}/" -e s/urgency=medium/urgency=low/ ../*.changes

# check if ubuntu already has the source tarball
(
cd ..
orig="${packagename}_${debversion}.orig.tar.gz"
# shellcheck disable=SC2012
orig_size=$(ls -l "$orig" | cut -d ' ' -f 5)
orig_sha1=$(sha1sum "$orig" | cut -d ' ' -f 1)
orig_sha256=$(sha256sum "$orig" | cut -d ' ' -f 1)
orig_md5=$(md5sum "$orig" | cut -d ' ' -f 1)

if wget --quiet -O "$orig-tmp" "$ppafilesurl/$orig"
then
    echo "[WARN] Original tarball found in Ubuntu archive, using it instead"
    mv "${orig}-tmp" "$orig"
    # shellcheck disable=SC2012
    new_size=$(ls -l ./*.orig.tar.gz | cut -d ' ' -f 5)
    new_sha1=$(sha1sum "$orig" | cut -d ' ' -f 1)
    new_sha256=$(sha256sum "$orig" | cut -d ' ' -f 1)
    new_md5=$(md5sum "$orig" | cut -d ' ' -f 1)
    sed -i -e "s,$orig_sha1,$new_sha1,g" -e "s,$orig_sha256,$new_sha256,g" -e "s,$orig_size,$new_size,g" -e "s,$orig_md5,$new_md5,g" ./*.dsc
    sed -i -e "s,$orig_sha1,$new_sha1,g" -e "s,$orig_sha256,$new_sha256,g" -e "s,$orig_size,$new_size,g" -e "s,$orig_md5,$new_md5,g" ./*.changes
fi
)

# sign the package
debsign --re-sign -k "${LAUNCHPAD_KEYID}" "../${packagename}_${debversion}-${versionsuffix}_source.changes"

# upload
dput "${pparepo}" "../${packagename}_${debversion}-${versionsuffix}_source.changes"

done
