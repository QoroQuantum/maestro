#!/bin/bash
# Build the C/C++ release archive: libmaestro.so, its C headers and every
# non-system library it loads, so users need no Boost, OpenBLAS or OpenMP.
#
# Runs inside a quay.io/pypa/manylinux_2_28_<arch> container (the image the
# Linux wheels use), so the archive runs on glibc 2.28 or newer.
#
# Usage: build_c_artifact.sh SRC_DIR OUT_DIR
# Environment: VERSION (e.g. 0.3.12), COMMIT (source commit),
#              AER (ON or OFF), JOBS (parallel compile jobs, default 2)
#
# Writes OUT_DIR/maestro-c-$VERSION-linux-$ARCH.tar.gz containing:
#   include/maestrolib/{Interface.h,InterfaceTypes.h}
#   lib/libmaestro.so plus bundled dependencies (RUNPATH $ORIGIN)
#   bin/maestro-request (only for versions that build it)
#   LICENSE, BUILD_INFO
set -euo pipefail

src=$(realpath "$1")
out=$2
: "${VERSION:?}" "${COMMIT:?}" "${AER:?}"
jobs=${JOBS:-2}
arch=$(uname -m)
boost_version=1.90.0
name="maestro-c-${VERSION}-linux-${arch}"
build=/tmp/maestro-build
stage="/tmp/stage/${name}"

echo "::group::Install OpenBLAS, CMake, Ninja and Boost ${boost_version}"
dnf install -y openblas-devel
pipx install cmake
pipx install ninja
export PATH="$HOME/.local/bin:/opt/_internal/pipx/venvs/cmake/bin:$PATH"
boost_dir="boost_${boost_version//./_}"
curl -sSfL --retry 5 "https://archives.boost.io/release/${boost_version}/source/${boost_dir}.tar.gz" -o /tmp/boost.tgz
tar -xzf /tmp/boost.tgz -C /tmp
(
    cd "/tmp/${boost_dir}"
    ./bootstrap.sh --prefix=/usr/local
    ./b2 -j"$(nproc)" -d0 --with-program_options --with-json --with-container --with-serialization install
)
echo "::endgroup::"

echo "::group::Build Maestro (Qiskit Aer ${AER})"
# The checkout belongs to the runner's user, not root; without this, git
# refuses to read it and the library reports its source revision as unknown.
git config --global --add safe.directory "$src"
cmake -S "$src" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DCOMPILE_TESTS=OFF \
    -DENABLE_QISKIT_AER="$AER"
# Each job can need several GB (more with Aer); 2 fits the 16 GB runners. If
# the compiler is still killed for memory, finish the remaining files one at a time.
cmake --build "$build" -j"$jobs" || cmake --build "$build" -j1
echo "::endgroup::"

echo "::group::Package ${name}"
rm -rf "$stage"
mkdir -p "$stage/include/maestrolib" "$stage/lib"
cp "$src/maestrolib/Interface.h" "$src/maestrolib/InterfaceTypes.h" "$stage/include/maestrolib/"
cp "$build/libmaestro.so" "$stage/lib/"
if [ -x "$build/bin/maestro-request" ]; then
    mkdir -p "$stage/bin"
    cp "$build/bin/maestro-request" "$stage/bin/"
fi
cp "$src/LICENSE" "$stage/"

# Bundle every library outside the manylinux_2_28 baseline (glibc, libstdc++,
# libgcc_s). ldd lists the full dependency closure, so one pass is enough.
system='^(linux-vdso|ld-linux|libc|libm|libdl|librt|libpthread|libutil|libstdc\+\+|libgcc_s)\.so'
for binary in "$stage"/lib/libmaestro.so "$stage"/bin/*; do
    [ -e "$binary" ] || continue
    ldd "$binary" | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}' |
        while read -r soname path; do
            if [[ ! $soname =~ $system ]] && [ ! -e "$stage/lib/$soname" ]; then
                cp -L "$path" "$stage/lib/$soname"
            fi
        done
done
for library in "$stage"/lib/*.so*; do
    patchelf --set-rpath '$ORIGIN' "$library"
done
for binary in "$stage"/bin/*; do
    [ -e "$binary" ] && patchelf --set-rpath '$ORIGIN/../lib' "$binary"
done

# Everything must now resolve inside the archive or to the system baseline.
for binary in "$stage"/lib/*.so* "$stage"/bin/*; do
    [ -e "$binary" ] || continue
    if ldd "$binary" | grep -q 'not found'; then
        ldd "$binary"
        echo "::error::$(basename "$binary") has unresolved dependencies"
        exit 1
    fi
    ldd "$binary" | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}' |
        while read -r soname path; do
            path=$(realpath "$path")
            if [[ ! $soname =~ $system ]] && [[ $path != "$stage"/lib/* ]]; then
                echo "::error::$(basename "$binary") loads $soname from $path, outside the archive"
                exit 1
            fi
        done
done

cat > "$stage/BUILD_INFO" <<EOF
version=${VERSION}
commit=${COMMIT}
arch=${arch}
qiskit_aer=${AER}
boost=${boost_version}
min_glibc=2.28
bundled=$(cd "$stage/lib" && ls | grep -v '^libmaestro\.so$' | tr '\n' ' ')
EOF
cat "$stage/BUILD_INFO"

mkdir -p "$out"
tar -C "$(dirname "$stage")" -czf "$out/${name}.tar.gz" "$name"
ls -l "$out/${name}.tar.gz"
echo "::endgroup::"
