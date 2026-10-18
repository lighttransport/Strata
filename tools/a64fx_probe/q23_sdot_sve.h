/* tools/a64fx_probe/q23_sdot_sve.h - Q2_K / Q3_K x Q8_K dot products on A64FX (SVE 512, int8 SDOT).
 *
 * ggml's SVE vec_dot for Q2_K/Q3_K runs at 2.0/1.2 GB/s per core (measured 2026-10-18), far below HBM.
 * This kernel exploits the K-quant layout: the 64 qs bytes of a 256-weight block are exactly one 512-bit vector,
 * and the four 2-bit planes (shift 0,2,4,6) each hold 64 weights whose positions are y[32j .. 32j+31] and
 * y[128+32j .. 128+32j+31]. The activation (a ggml Q8_K block) is permuted once per token into that plane order,
 * so a block costs one SDOT per plane per token; per-16 scales enter as int32 lane multipliers (svtbl over the
 * 16 unpacked scales), the Q2_K mins and the Q3_K "-4" offset go through the Q8_K bsums.
 * Results equal ggml's (same integer sums, different float order). Requires a 512-bit SVE vector length. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* one Q8_K block of an activation, permuted to plane order: x[64 j + l] = q[32 j + l], x[64 j + 32 + l] = q[128 + 32 j + l] */
typedef struct {
    int8_t  x[256];
    int32_t bsum[16];
    float   d;
    float   pad[15];
} q23s_act;

int  q23s_vl_bytes(void);
/* q8k_row: `cols/256` ggml block_q8_K (float d, int8 qs[256], int16 bsums[16]) */
void q23s_prepare(const void *q8k_row, int cols, q23s_act *out);
/* natural order (x[k] = q[k]) for the IQ kernels, same struct */
void q23s_prepare_natural(const void *q8k_row, int cols, q23s_act *out);

/* v1: dot of one weight row with nt (1..4) prepared activations; out[t] */
void q2k_sdot_row(int cols, const void *w, const q23s_act *const *acts, int nt, float *out);
void q3k_sdot_row(int cols, const void *w, const q23s_act *const *acts, int nt, float *out);

/* v2: rows `w` and `w + stride` together (nrows 1 or 2), out[r * nt + t]. Q2_K pre-multiplies the 2-bit weights by
 * their 4-bit scales in int8 (<= 45) so a token costs four SDOTs and no integer multiply; both split the SDOT chains
 * and share the activation loads between the two rows. */
void q2k_sdot_rows_v2(int cols, const void *w, size_t stride, int nrows, const q23s_act *const *acts, int nt, float *out);
void q3k_sdot_rows_v2(int cols, const void *w, size_t stride, int nrows, const q23s_act *const *acts, int nt, float *out);

/* IQ2_XXS / IQ1_S x Q8_K (iq_sdot_sve.c), natural-order activations, one row; grid entries by SVE gather.
 * iq2xxs variant 0: signs from popcount parity in registers; variant 1: signs gathered from ksigns64. */
void iq2xxs_sdot_row(int cols, const void *w, const q23s_act *const *acts, int nt, float *out, int variant);
void iq1s_sdot_row(int cols, const void *w, const q23s_act *const *acts, int nt, float *out);

#ifdef __cplusplus
}
#endif
