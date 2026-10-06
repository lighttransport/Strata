#pragma once

namespace strata::kernels {

// Decode GEMVs over GGUF Q8_0 weights (34-byte blocks: FP16 scale, 32 int8) with FP32 activations. Each
// product uses the exactly dequantized weight, so only the summation order differs from an FP32 copy.

// Per-head rows: y[t * ld_y + r] = W[r] . x[t * ld_x + (r / out) * in + 0..in), for r < heads * out.
// in must be a multiple of 32; tokens 1..8.
void glm_q8_heads_gemv(const void *w, const float *x, float *y, int heads, int in, int out, int tokens, int ld_x,
                       int ld_y, void *stream);

// Few long rows (the 24-row mHC projection): y[t * rows + r] = W[r] . x[t * in + 0..in). Split over 32 slices
// with a fixed-order second pass; scratch holds rows * tokens * 32 floats. in must be a multiple of 32 * 32.
void glm_q8_rows_gemv(const void *w, const float *x, float *y, float *scratch, int rows, int in, int tokens,
                      void *stream);

// FP32 row-major W[rows][in] (cuBLAS OP_T layout): y[t * rows + r] = W[r] . x[t * in + 0..in). One warp per row
// with float4 loads; for the router and other small decode matrices. in % 4 == 0, 16-byte aligned, tokens 1..8.
void glm_f32_rows_gemv(const float *w, const float *x, float *y, int rows, int in, int tokens, void *stream);

// Dequantizes rows * in Q8_0 values to FP32 (for prefill GEMMs that keep their FP32 path).
void glm_q8_dequant(const void *w, float *y, long long values, void *stream);

} // namespace strata::kernels
