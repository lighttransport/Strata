// include/strata/kernels/cpu/nvfp4_avx2.hpp - NVFP4 expert rows in 256-bit lanes, several tokens at once, for CPUs
// without AVX-512 (Zen 2/3, Intel 12th-14th gen).
//
// ggml-cpu's NVFP4 dot product (ggml_vec_dot_nvfp4_q8_0) re-decodes a row's blocks for every token of a window;
// these decode a 64-value block once and apply it to each token. Each token's arithmetic is that dot product's AVX2
// path step for step (the same integer sums, the same FMA order, the same horizontal sum), so a row equals
// ggml-cpu's bit for bit - what an AVX2 CPU computed before these existed. Activations are Q8_0.
// STRATA_NVFP4_128=1 opts into two 128-bit accumulators for nt==1.
// STRATA_NVFP4_OFFSET_DOT=1 opts into biased integer dots for nt==1.
// The switches are independent: enabling both selects 128-bit biased dots.
// Read once on first single-token call; all other values retain the default.
// Raw Q8_0 -128 activations fall back to sign-transfer arithmetic.
// Optional STRATA_NVFP4_CLANG_NT4 build: runtime value 1 selects Clang only for
// ordinary nt=4 GU (clamped or unclamped). Read once; unset/value 2 remain off.
// Optional STRATA_NVFP4_GU_HALF_PAIR_NT1 build pairs ordinary nt=1 gate/up
// halves. No runtime switch; K32, down and narrow/offset paths are unchanged.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Whether rows of `n` values fit the kernels (whole 64-value blocks, at most 64 of them).
bool nvfp4_256_fits(int n);

/// Gate/up rows [r0, r1) for `nt` tokens: ff[t][r] = silu(s_gate * g) * (s_up * u), as nvfp4_512_gu_rows.
void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up);

/// Positive limit matches native SwiGLU: gate <= limit and up in [-limit, limit]
/// after global scaling. A nonpositive limit retains unclamped arithmetic.
/// separate_up selects independent up storage instead of blob + up_off.
void nvfp4_256_gu_rows_clamped(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                              const void* const* act, int nt, float* const* ff,
                              int r0, int r1, float s_gate, float s_up, float limit, const uint8_t* separate_up = nullptr);

/// Plain rows [r0, r1) against `nt` Q8_0 activations of `n` values, times `scale`, as nvfp4_512_rows.
void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale = 1.f);

/// Experimental rows converted by strata_glm_nvfp4_repack_k32; NOT GGML layout.
/// STRATA_NVFP4_K32_OFFSET=1 opts one-token calls into the K32-specific offset dot.
/// Other values keep the original path; STRATA_NVFP4_OFFSET_DOT does not select it.
/// Requires AVX2/FMA/F16C, n=64..4096 in whole K64 blocks, nt=1..32 and
/// valid buffers/row stride. Empty ranges do not access buffers. Invalid
/// geometry/null nonempty buffers throw. The default uses signed 256-bit dots.
/// Legacy STRATA_NVFP4_128/OFFSET_DOT switches do not select this implementation;
/// no automatic production dispatch selects this weight layout.
void nvfp4_256_k32_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                         float* const* out, int r0, int r1, float scale = 1.f);

/// Matching experimental gate/up primitive, with the same K32 geometry/ISA
/// requirements. Clamp semantics match nvfp4_256_gu_rows_clamped. separate_up
/// selects independent packed up rows; otherwise up_off selects them in blob.
void nvfp4_256_k32_gu_rows_clamped(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                                  const void* const* act, int nt, float* const* ff, int r0, int r1,
                                  float s_gate, float s_up, float limit, const uint8_t* separate_up = nullptr);

}  // namespace strata::kernels::cpu
