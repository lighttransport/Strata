#!/bin/bash
# tools/a64fx_probe/ggml_cmake_args.sh - sourced by the A64FX build scripts: the FCC (Fujitsu, clang mode) settings that
# build ggml-base / ggml-cpu on a compute node.
#
# FCC 4.12 (LLVM 7) aborts on the auto-vectorized FP16 code of ggml-cpu/unary-ops.cpp at -O3 and -fno-vectorize does not
# avoid it; disable_autovec.h (`#pragma clang optimize off`) is force-included into the C++ files only. The quantized
# kernels and quantizers are C and keep -O3.
_A64FX_PROBE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A64FX_CC=${A64FX_CC:-fcc}
A64FX_CXX=${A64FX_CXX:-FCC}
GGML_C_FLAGS="-Nclang -O3 -fopenmp"
GGML_CXX_FLAGS="-Nclang -O3 -fopenmp -include $_A64FX_PROBE_DIR/disable_autovec.h"
GGML_OPTS=(-DGGML_NATIVE=OFF "-DGGML_CPU_ARM_ARCH=armv8.2-a+sve" -DGGML_CPU=ON -DGGML_CCACHE=OFF -DGGML_CPU_HBM=OFF
           -DGGML_CPU_KLEIDIAI=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF)

# Ninja when one runs on this node, else Unix Makefiles (an x86 ninja on the PATH of an A64FX node does not run).
if command -v ninja > /dev/null 2>&1 && ninja --version > /dev/null 2>&1; then GGML_GENERATOR=Ninja; else GGML_GENERATOR="Unix Makefiles"; fi

# ggml_configure SRC BUILD [extra cmake args...]: SRC is a directory with ggml's own CMakeLists.txt (a llama.cpp
# checkout's ggml/ or a copied ggml tree); configures with $GGML_GENERATOR into BUILD.
# A partial ggml tree (CMakeLists.txt, cmake/, include/, src/ only) cannot be configured on its own: standalone ggml adds
# tests/ and examples/ and configures ggml.pc.in. Such a tree is configured through a one-line parent project instead.
ggml_configure() {
    local src=$1 build=$2; shift 2
    if [ ! -f "$src/ggml.pc.in" ]; then
        mkdir -p "$build.wrap"
        printf 'cmake_minimum_required(VERSION 3.14)\nproject(ggml_wrap C CXX ASM)\nadd_subdirectory("%s" ggml)\n' "$(cd "$src" && pwd)" \
            > "$build.wrap/CMakeLists.txt"
        src=$build.wrap
    fi
    cmake -S "$src" -B "$build" -G "$GGML_GENERATOR" -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
        -DCMAKE_C_COMPILER="$A64FX_CC" -DCMAKE_CXX_COMPILER="$A64FX_CXX" -DGIT_EXECUTABLE=/bin/false \
        -DCMAKE_C_FLAGS="$GGML_C_FLAGS" -DCMAKE_CXX_FLAGS="$GGML_CXX_FLAGS" "${GGML_OPTS[@]}" "$@"
}

# ggml_build BUILD -j N TARGET...: builds the targets with the generator ggml_configure used.
ggml_build() {
    local build=$1; shift
    if [ "$GGML_GENERATOR" = Ninja ]; then ninja -C "$build" "$@"; else make -C "$build" "$@"; fi
}
