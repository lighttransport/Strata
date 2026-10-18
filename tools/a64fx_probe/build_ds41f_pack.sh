#!/bin/bash
# tools/a64fx_probe/build_ds41f_pack.sh - build strata-ds41f-pack on an A64FX compute node outside CMake and publish it
# under a content-addressed directory on the shared filesystem.
#
#   GGML_SRC=<dir with ggml/> DS41F_BIN_ROOT=<shared dir> [DS41F_GGML_QUANTS_SHA256=<hex>] tools/a64fx_probe/build_ds41f_pack.sh
#
# The tool needs only ggml-base (quantizers, type traits) and ggml_cpu_init; ggml is built static with OpenMP off, so
# the binary needs no Fujitsu runtime library at run time. Quantizer output depends on ggml's ggml-quants.c: give the
# same ggml tree the x86 builds use and pin it with DS41F_GGML_QUANTS_SHA256 (the build refuses another one).
# Publishes $DS41F_BIN_ROOT/strata-ds41f-pack-<sha12>/ with the binary, BUILD.txt and the PJM job scripts; a published
# directory is never edited (compute nodes may see stale shared files), a new build gets a new directory.
# Prints `SENTINEL DS41F_PACK_BUILD=OK dir=... sha256=...`.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
STRATA=${STRATA:-$(cd "$here/../.." && pwd)}
GGML_SRC=${GGML_SRC:?set GGML_SRC to a directory containing ggml/}
DS41F_BIN_ROOT=${DS41F_BIN_ROOT:?set DS41F_BIN_ROOT to the shared directory for published binaries}
if [ -d /local ]; then B=${BUILD_DIR:-/local/ds41f-pack-build-${PJM_JOBID:-manual}}; else B=${BUILD_DIR:-$STRATA/tmp/ds41f-pack-build}; fi
mkdir -p "$B/tmp"; export TMPDIR=$B/tmp
source "$here/ggml_cmake_args.sh"
GGML_C_FLAGS=${GGML_C_FLAGS/ -fopenmp/}; GGML_CXX_FLAGS=${GGML_CXX_FLAGS/ -fopenmp/}   # static, OpenMP off: no runtime library
quants_sha=$(sha256sum "$GGML_SRC/ggml/src/ggml-quants.c" | cut -d' ' -f1)
if [ -n "${DS41F_GGML_QUANTS_SHA256:-}" ] && [ "$quants_sha" != "$DS41F_GGML_QUANTS_SHA256" ]; then
    echo "SENTINEL DS41F_PACK_BUILD=FAIL reason=ggml-quants.c sha256 $quants_sha != pinned $DS41F_GGML_QUANTS_SHA256"; exit 2
fi
echo "BUILD ggml from $GGML_SRC (ggml-quants.c $quants_sha) into $B"
ggml_configure "$GGML_SRC/ggml" "$B/ggml" -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF > "$B/ggml-cmake.log" 2>&1 \
    || { tail -20 "$B/ggml-cmake.log"; echo "SENTINEL DS41F_PACK_BUILD=FAIL reason=ggml cmake"; exit 1; }
ninja -C "$B/ggml" -j "${JOBS:-24}" ggml-base ggml-cpu > "$B/ggml-build.log" 2>&1 \
    || { tail -30 "$B/ggml-build.log"; echo "SENTINEL DS41F_PACK_BUILD=FAIL reason=ggml build"; exit 1; }
base=$(find "$B/ggml" -name 'libggml-base.a' | head -1); cpu=$(find "$B/ggml" -name 'libggml-cpu.a' | head -1)
[ -n "$base" ] && [ -n "$cpu" ] || { echo "SENTINEL DS41F_PACK_BUILD=FAIL reason=ggml archives not found"; exit 1; }
CXXFLAGS="-Nclang -O2 -std=c++17 -DNDEBUG"
"$A64FX_CC" -Nclang -O2 -std=c11 -I"$STRATA/include" -c "$STRATA/src/host/ds41f_names.c" -o "$B/ds41f_names.o"
"$A64FX_CXX" $CXXFLAGS -I"$STRATA/include" -I"$GGML_SRC/ggml/include" -c "$STRATA/tools/ds41f_pack.cpp" -o "$B/ds41f_pack.o"
link() { "$A64FX_CXX" -Nclang -o "$B/strata-ds41f-pack" "$B/ds41f_pack.o" "$B/ds41f_names.o" "$cpu" "$base" -lpthread -lm "$@"; }
link > "$B/link.log" 2>&1 || link -lstdc++fs > "$B/link.log" 2>&1 || { tail -20 "$B/link.log"; echo "SENTINEL DS41F_PACK_BUILD=FAIL reason=link"; exit 1; }
"$B/strata-ds41f-pack" 2>&1 | grep -q "usage: strata-ds41f-pack" || { echo "SENTINEL DS41F_PACK_BUILD=FAIL reason=binary does not run"; exit 1; }
sha=$(sha256sum "$B/strata-ds41f-pack" | cut -d' ' -f1)
out=$DS41F_BIN_ROOT/strata-ds41f-pack-${sha:0:12}
if [ -e "$out/strata-ds41f-pack" ]; then
    [ "$(sha256sum "$out/strata-ds41f-pack" | cut -d' ' -f1)" = "$sha" ] || { echo "SENTINEL DS41F_PACK_BUILD=FAIL reason=$out exists with other content"; exit 1; }
else
    mkdir -p "$out.partial"
    cp "$B/strata-ds41f-pack" "$out.partial/"
    cp "$STRATA"/tools/ds41f_pjm_*.sh "$out.partial/" 2>/dev/null || true
    { echo "binary_sha256 $sha"; echo "ggml_quants_sha256 $quants_sha"; echo "ggml_src $GGML_SRC"
      echo "compiler $("$A64FX_CXX" --version 2>&1 | head -1)"; echo "cxxflags $CXXFLAGS"; echo "built $(date -Is) on $(hostname) job ${PJM_JOBID:-none}"
      for f in tools/ds41f_pack.cpp src/host/ds41f_names.c include/strata/host/ds41f_names.h include/strata/artifact/json.hpp \
               include/strata/artifact/gguf_reader.hpp include/strata/artifact/fp8_experts.hpp; do echo "source $(sha256sum "$STRATA/$f")"; done
      echo "ldd:"; ldd "$B/strata-ds41f-pack" 2>&1 | sed 's/^/  /'; } > "$out.partial/BUILD.txt"
    mv "$out.partial" "$out"
fi
echo "SENTINEL DS41F_PACK_BUILD=OK dir=$out sha256=$sha"
