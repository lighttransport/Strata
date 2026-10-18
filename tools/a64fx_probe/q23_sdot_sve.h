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
/* natural order (x[k] = q[k]) for the IQ gather kernels, same struct */
void q23s_prepare_natural(const void *q8k_row, int cols, q23s_act *out);
/* quad order for the gather-free IQ kernels: lane L of plane j holds weight 4 L + j (x[64 j + L] = q[4 L + j]) */
void q23s_prepare_quad(const void *q8k_row, int cols, q23s_act *out);

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

/* Gather-free IQ (iq_sdot_sve.c), quad-order activations. A block's 32 grid entries become 64 bytes of 2-bit codes
 * (byte L = weights 4L..4L+3), one plane per 2-bit field, values by a 64-byte svtbl.
 *   IQ2_XXS layout 0: the ggml block as stored; the codes come from eight 64-byte svtbl tables (the 256-entry grid
 *                     as 2-bit codes = 512 B) and the signs from ksigns via two svtbl. No repack, no gather.
 *   IQ2_XXS layout 1: repacked at load, 74 B per block (+12 %): d, 8 scale bytes (2s+1), 32 grid indices, 32 sign
 *                     bytes (parity resolved). Codes still from the svtbl tables.
 *   IQ2_XXS layout 2: repacked at load, 106 B (+61 %): codes expanded (the decode floor; a reference, not a format).
 *   IQ1_S rpc:        repacked at load, 82 B (+64 %): 2-bit ternary codes, 8 scale bytes, 8 delta signs. IQ1_S has no
 *                     compact gather-free form: its 2048-entry grid would need 64 svtbl tables. */
enum { IQ2XXS_RP_BYTES = 74, IQ2XXS_RPC_BYTES = 106, IQ1S_RPC_BYTES = 82 };
void iq2xxs_repack(const void *src, void *dst, int nblocks, int layout);   /* layout 1 or 2 */
void iq1s_repack(const void *src, void *dst, int nblocks);
/* v3: software-pipelined single-row kernels (block i+1 decoded, its gathers issued, before block i's token math;
 * per-block scales kept in vector registers). Same activations as v2 / the gather kernels. */
void q2k_sdot_row_v3(int cols, const void *w, const q23s_act *const *acts, int nt, float *out);
void q3k_sdot_row_v3(int cols, const void *w, const q23s_act *const *acts, int nt, float *out);
void iq2xxs_sdot_row_v3(int cols, const void *w, const q23s_act *const *acts, int nt, float *out);
void iq1s_sdot_row_v3(int cols, const void *w, const q23s_act *const *acts, int nt, float *out);
void iq2xxs_sdot_row_tbl(int cols, const void *w, const q23s_act *const *acts, int nt, float *out, int layout);
void iq1s_sdot_row_rpc(int cols, const void *w, const q23s_act *const *acts, int nt, float *out);

#ifdef __cplusplus
}
#endif
