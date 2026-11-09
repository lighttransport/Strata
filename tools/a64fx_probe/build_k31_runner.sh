#!/usr/bin/env bash
# Build k31pro_runner (K3.1 Pro decode, MPI + uTofu weight shuffle) and its kernel test for A64FX.
# Usage: build_k31_runner.sh [OUT_DIR]   (default build/k31-a64fx). Site settings come from the environment:
#   K31_CC / K31_CXX (default mpiclang / mpiclang++), K31_CFLAGS, K31_CXXFLAGS, K3_DIR (optional: the K3 reference
#   headers for the kernel test's parity cases).
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-"$repo/build/k31-a64fx"}
cc=${K31_CC:-mpiclang}
cxx=${K31_CXX:-mpiclang++}
read -r -a cflags <<< "${K31_CFLAGS:--O3 -mcpu=a64fx -msve-vector-bits=512 -fopenmp -D_GNU_SOURCE}"
read -r -a cxxflags <<< "${K31_CXXFLAGS:--std=c++20 -stdlib=libc++}"
mkdir -p "$out"
out=$(cd "$out" && pwd)
cd "$repo"
"$cc" "${cflags[@]}" -std=c11 -Iinclude -c src/kernels/cpu/k31_sve.c -o "$out/k31_sve.o"
"$cc" "${cflags[@]}" -std=c11 -Iinclude -c src/kernels/cpu/ds41_sve.c -o "$out/ds41_sve.o"
"$cc" "${cflags[@]}" -std=c11 -c src/cpu_runner/k31_shuffle_utofu.c -o "$out/k31_shuffle.o"
# optional uTofu all-reduce for decode (UTOFU_DIR: the directory containing tp_allreduce.h)
ar=(); arobj=()
if [ -n "${UTOFU_DIR:-}" ]; then
    "$cc" "${cflags[@]}" -std=c11 -Iinclude -I"$UTOFU_DIR" -c src/cpu_runner/comm_utofu.c -o "$out/comm_utofu.o"
    ar=(-DK31_UTOFU_AR); arobj=("$out/comm_utofu.o")
    if grep -q glm_env.h src/cpu_runner/comm_utofu.c; then
        "$cc" "${cflags[@]}" -std=c11 -Iinclude -c src/host/glm_env.c -o "$out/glm_env.o"; arobj+=("$out/glm_env.o")
    fi
fi
ref=()
if [ -n "${K3_DIR:-}" ]; then ref=(-DK31_K3REF "-I$K3_DIR"); fi
"$cc" "${cflags[@]}" -std=c11 -Iinclude "${ref[@]}" tools/a64fx_probe/k31_kernel_test.c "$out/k31_sve.o" -lm \
    -o "$out/k31_kernel_test"
"$cxx" "${cflags[@]}" "${cxxflags[@]}" -Iinclude -DK31_MPI -DK31_UTOFU "${ar[@]}" src/cpu_runner/k31pro_runner.cpp \
    "$out/k31_sve.o" "$out/ds41_sve.o" "$out/k31_shuffle.o" "${arobj[@]}" -ltofucom -o "$out/k31pro_runner"
printf 'Built %s\n' "$out/k31pro_runner"
