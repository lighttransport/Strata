/* DeepSeek-V4.1-Flash (GGUF) CPU kernels for A64FX (SVE 512): rows of a row-major ggml-quantized matrix against
 * 1-8 tokens, the activation quantizers ggml uses for them, and scalar dequantizers (the reference).
 *
 * Types are ggml type ids. Routed experts: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS (x Q8_K activations); dense: Q5_K
 * (x Q8_K), Q8_0 (x Q8_0), BF16 and F32 (x F32); Engram rows: IQ4_NL (dequantized only). The integer sums equal
 * ggml's vec_dot; only the float order differs (parity: tools/a64fx_probe/ds41_kernel_test.c). */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { DS41_T_F32 = 0, DS41_T_Q8_0 = 8, DS41_T_Q5_K = 13, DS41_T_IQ2_XXS = 16, DS41_T_IQ2_XS = 17, DS41_T_IQ3_XXS = 18,
       DS41_T_IQ4_NL = 20, DS41_T_IQ2_S = 22, DS41_T_BF16 = 30, DS41_T_MXFP4 = 39 };
/* MXFP4 (the DSpark drafter's routed experts): 32 weights per block, E8M0 scale + 16 bytes of e2m1 nibbles (low nibble
   j, high nibble j + 16), x Q8_0 activations; integer sums of the doubled e2m1 values as ggml's vec_dot_mxfp4_q8_0 */
/* IQ2R: a Strata in-memory layout (not a ggml type) for IQ2_XXS / IQ2_XS / IQ2_S rows without grid indices: per 256
 * weights the F16 d, 16 4-bit scales (one per 16 weights), 2-bit magnitude codes (grid bytes 8 / 25 / 43) and the sign
 * bits, 106 bytes. Its integer sums are the source block's, so ds41_rows gives the source kernel's bits (no gathers). */
enum { DS41_T_IQ2R = 1016 };
/* IQ2P: IQ2R's bits in the order the SVE kernel reads them (same 106 bytes: d, scales, codes[64], signs[32]): codes[j]
 * bits 2 v .. 2 v + 1 = weight 64 v + j's magnitude code (one load for all four planes), signs[8 v .. 8 v + 7] = plane v's
 * sign bits in predicate order (bit j % 8 of byte j / 8: one predicate load). Same integer sums as IQ2R (same bits). */
enum { DS41_T_IQ2P = 1017 };
void ds41_iq2r_to_iq2p(void *blocks, size_t nblocks);   /* in place */
/* Q5P: a Q5_K block's bits in the kernel's read order (same 176 bytes: d, dmin, scales, qh[32], qs[128]): qs[64 p + k]
 * low nibble = weight 64 (2 p) + k, high nibble = weight 64 (2 p + 1) + k (one load per two planes, mask / shift), qh[8 v ..
 * 8 v + 7] = plane v's fifth bits in predicate order. Same integer sums as Q5_K (same bits). */
enum { DS41_T_Q5P = 1018 };
void ds41_q5k_to_q5p(void *blocks, size_t nblocks);   /* in place */
/* one row of an IQ2_XXS / IQ2_XS / IQ2_S matrix to IQ2R (cols % 256 == 0); 0 on success, -1 for another type or a
 * grid byte outside 8 / 25 / 43 */
int ds41_repack_iq2r(int type, const void *src, int cols, void *dst);

/* a Q8_K activation block (ggml quantize_row_q8_K_ref: d = max / -127, round half even, bsums per 16), 64-aligned */
typedef struct { int8_t q[256]; int16_t bsum[16]; float d; uint8_t pad[28]; } ds41_q8k;
/* a Q8_0 activation block (ggml quantize_row_q8_0_ref: d = amax / 127 rounded to F16, q = roundf(x / d)) */
typedef struct { int8_t q[32]; float d; } ds41_q80;

enum { DS41_ACT_F32 = 0, DS41_ACT_Q8K = 1, DS41_ACT_Q80 = 2 };
int ds41_act_kind(int type);                       /* the activation a type's rows take; -1: unsupported */
int ds41_type_ok(int type);
size_t ds41_row_bytes(int type, int cols);
void ds41_quant_q8k(const float *x, int n, ds41_q8k *y);   /* n % 256 == 0 */
void ds41_quant_q80(const float *x, int n, ds41_q80 *y);   /* n % 32 == 0 */
void ds41_dequant_row(int type, const void *row, int cols, float *y);

/* y[t * ldy + r] = row r of w (row 0 at w) . token t, for r in [r0, r1), t < nt <= 8 (IQ2P also accepts 9-16). acts[t]: cols/256 ds41_q8k,
 * cols/32 ds41_q80 or cols floats, as ds41_act_kind says. */
void ds41_rows(int type, const void *w, int cols, int r0, int r1, const void *const *acts, int nt, float *y, int ldy);
/* the same in double from ds41_dequant_row and the dequantized activations; absy[t * ldy + r] = sum |w x| */
void ds41_rows_ref(int type, const void *w, int cols, int r0, int r1, const void *const *acts, int nt, double *y,
                   double *absy, int ldy);

#ifdef __cplusplus
}
#endif
