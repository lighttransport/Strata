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

# ggml_configure SRC BUILD [extra cmake args...]: SRC is a directory with ggml's own CMakeLists.txt (a llama.cpp
# checkout's ggml/ or a copied ggml tree); configures with Ninja into BUILD.
ggml_configure() {
    local src=$1 build=$2; shift 2
    cmake -S "$src" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$A64FX_CC" -DCMAKE_CXX_COMPILER="$A64FX_CXX" -DGIT_EXECUTABLE=/bin/false \
        -DCMAKE_C_FLAGS="$GGML_C_FLAGS" -DCMAKE_CXX_FLAGS="$GGML_CXX_FLAGS" "${GGML_OPTS[@]}" "$@"
}
