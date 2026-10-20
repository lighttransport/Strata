/* tools/a64fx_probe/disable_autovec.h - force-included into ggml's C++ files when building with Fujitsu FCC 4.12
 * (LLVM 7) in clang mode: its SVE instruction selector aborts on the auto-vectorized FP16 code in
 * ggml-cpu/unary-ops.cpp at -O3, and -fno-vectorize/-fno-slp-vectorize do not avoid it. The quantized dot kernels are
 * C with explicit SVE intrinsics and keep -O3; only the C++ side loses optimization. */
#pragma clang optimize off
