/* Ripple-65B (Ricursive dense) CPU kernels for A64FX (SVE 512): NVFP4 rows against one F32 token vector.
 *
 * NVFP4 as the ModelOpt checkpoint stores it: a row of `cols` weights is cols / 2 bytes (two e2m1 nibbles per byte, the
 * element 2 j in the low nibble, 2 j + 1 in the high nibble) plus cols / 16 F8_E4M3 block scales; a tensor-wide F32
 * weight_scale_2 multiplies every scale. weight = e2m1(nibble) * e4m3(scale) * weight_scale_2.
 * Parity with the scalar reference: tools/a64fx_probe/ripple_kernel_test.c. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The 256-entry E4M3 (fn: no infinity, 0x7f / 0xff are NaN) to float table; ripple_nvfp4_init() fills it, call it once */
void ripple_nvfp4_init(void);
float ripple_e4m3_to_f32(uint8_t b);
float ripple_e2m1_to_f32(unsigned nibble);

/* y[t * ldy + r] = (row r of the matrix) . xs[t] for r in [r0, r1), t < nt <= 8. q: row-major packed nibbles (row stride
 * cols / 2 bytes), sc: row-major block scales (row stride cols / 16), s2: weight_scale_2. cols % 32 == 0, cols <= 8192.
 * One token: the fused decode + FMA loop; several: each row is decoded once into floats, then dotted with every token. */
void ripple_nvfp4_rows(const uint8_t *q, const uint8_t *sc, float s2, int cols, int r0, int r1, const float *const *xs, int nt, float *y, int ldy);
/* the same in double from the scalar decoders (the reference) */
void ripple_nvfp4_rows_ref(const uint8_t *q, const uint8_t *sc, float s2, int cols, int r0, int r1, const float *x, double *y);

/* ---- int8 dot-product path (SVE sdot). The loader repacks every NVFP4 row once; the activations are quantized to int8.
 * Repacked row: groups of 128 weights (a tail group of 32 / 64 / 96 when cols % 128 != 0, call it G, half = G / 2): half
 * (64 for a full group) bytes with byte j = nibble(element j) | nibble(element j + half) << 4, so one 64-byte load gives
 * two 64-lane index vectors (low and high) and each maps by svtbl onto doubled-e2m1 int8 values. The block scales are F16
 * (an E4M3 scale is exact in F16), one per 16 elements in element order. Activations: units of 64 elements (half of
 * the tail group's elements for the tail), each int8 with one F32 scale d = max|x| / 127. A row's result is
 * 0.5 * s2 * sum over units of d * sum over its 16-blocks of scale * sum(w2 * xq) (w2 = 2 * e2m1, an integer). cols % 32 == 0. */
int ripple_q8_units(int cols);
void ripple_q8_quant_unit(const float *x, int cols, int u, int8_t *xq, float *xd);   /* unit u of x (xq: cols int8) */
void ripple_q8_quant(const float *x, int cols, int8_t *xq, float *xd);              /* every unit */
void ripple_fp4_repack_row(const uint8_t *q, const uint8_t *sc, int cols, uint8_t *qo, uint16_t *sfo);
/* y[t * ldy + r] for r in [r0, r1), t < nt: q / sf are the repacked rows (row stride cols / 2 bytes, cols / 16 halfs) */
void ripple_fp4_rows_q8(const uint8_t *q, const uint16_t *sf, int cols, int r0, int r1, const int8_t *const *xq,
                        const float *const *xd, int nt, float s2, float *y, int ldy);
void ripple_fp4_rows_q8_ref(const uint8_t *q, const uint16_t *sf, int cols, int r0, int r1, const int8_t *const *xq,
                            const float *const *xd, int nt, float s2, float *y, int ldy);   /* scalar, same math */

#ifdef __cplusplus
}
#endif
