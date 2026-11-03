// native_tc_mmvq - quantized matvec for MANY columns (9..64) that reads the weights ONCE: each 32-row x 256-column block of Q4_K / Q5_K / Q6_K
// weights is decoded to fp16 in shared memory and multiplied on tensor cores (WMMA 32x8x16) against fp16 activations with fp32 accumulation, for up
// to 64 columns in the same pass.  The q8_1 matvec (native_mmvq) re-reads the weights for every 8 columns and spends ~12 us per column and matrix;
// this kernel is flat in the column count.
//
// NOT bitwise equal to native_mmvq: the activations are fp16 (not q8_1 integers) and the weights are dequantized to fp16, so results differ at the
// level of the q8_1 quantization noise (about 0.4 %); it is meant for the batched decode path only, never for the single-sequence path.
// CUDA only (WMMA); `native_tc_mmvq_supported` is false elsewhere and for shapes that are not multiples of 32 rows / 256 columns.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

constexpr int kTcMaxCols = 64;

/// GGML type ids as in native_mmvq.hpp: Q4_K = 12, Q5_K = 13, Q6_K = 14.
bool native_tc_mmvq_supported(int ggml_type, int n_in, int n_out, int ncols) noexcept;

/// y[col * n_out + row] = sum_k W[row][k] * x16[col * n_in + k].  x16 holds `ceil(ncols / 8) * 8` columns of n_in fp16 values (the padding columns
/// finite); the weights are the GGUF blocks of the matrix with at least 256 readable bytes after the last one.  Asynchronous on `stream`.
void native_tc_mmvq(int ggml_type, const void* weights, const void* x16, float* y, int n_in, int n_out, int ncols, void* stream);

/// n floats -> n fp16 (round to nearest).
void native_f32_to_f16(const float* src, void* dst, int64_t n, void* stream);

}  // namespace strata::kernels
