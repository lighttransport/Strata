/* strata/kernels/cpu/glm_fp4_sve.h - GLM-5.3-Flash FP4-Local weights on A64FX (SVE 512): NVFP4 experts, E4M3 / BF16 trunk.
 *
 * Source formats (the checkpoint, row-major W[N][K], y = W x):
 *   NVFP4  codes [N][K/2] (low nibble = even column), E4M3 scales [N][K/16] (CSF32 expanded), one F32 global;
 *          w = e2m1(code) * e4m3(scale) * global
 *   MXFP8  E4M3 [N][K], UE8M0 scales [N][K/32]: w = e4m3 * 2^(e8m0 - 127)
 *   BFP8   E4M3 [N][K], F32 scale per 128 x 128 block
 *   BF16   [N][K]
 * Packed layouts (repacked at load, the same bytes per weight as the source; N % 16 == 0, K % 64 == 0):
 *   rt16q  NVFP4, 16-row tiles, per K64 superblock 576 B: four K16 groups of two 64-byte code vectors, then the 4 x 16
 *          E4M3 row scales. In group g, byte 4r+i of vector v holds code(r, 16g+8v+i) | code(r, 16g+8v+4+i) << 4, so after
 *          the nibble split int32 lane r of slice j (low/high nibbles of v = j/2) holds row r at columns 16g+4j+i, i = 0..3:
 *          four SDOT-by-lane accumulate one K16 group per row in one int32 lane.
 *   rt16p  NVFP4, same sizes; group g holds 8 pair blocks of 16 B, byte r of pair p = code(r, 16g+2p) | code(r, 16g+2p+1) << 4
 *          (one byte per 32-bit lane after a widening load: two svtbl_f32 decodes per pair).
 *   e8k    E4M3, 16-row tiles, column-major inside the tile (16 bytes per column); MXFP8: then 16 UE8M0 bytes per K32 block
 *          after its 32 columns; BFP8: then one F32 scale per K128 block after its 128 columns.
 *   q8k    Q8 trunk (fast mode): int8, 16-row tiles, per K32 block eight 64-byte vectors (lane r of vector j = row r,
 *          columns 4j..4j+3) and 16 F16 row scales (544 B per 512 weights, the bytes of ggml Q8_0).
 *   b16k   BF16, 16-row tiles, column-major inside the tile (32 bytes per column).
 * Kernels: one 16-row tile (or a range of tiles), nt = 1..4 tokens; y[t * ldy + row].
 *   w4a8   NVFP4 x int8 activations (one F32 scale per 64 values, Strata's glm_fp4_quantize_q8 rounding), SDOT per K16.
 *   w4a16q / w4a16p  NVFP4 x F32 activations, exact products, F32 accumulation in a fixed order (rt16q / rt16p).
 *   mxfp8 / bfp8 / bf16  F32 activations; q8 int8 activations as w4a8.
 * Every kernel has a scalar replica with the same operation order (bitwise equal) in src/kernels/cpu/glm_fp4_sve.c. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { GF4_TILE = 16, GF4_SB_BYTES = 576, GF4_Q8_BLOCK_BYTES = 544 };

/* ---- scalar codecs (exact) */
float gf4_e2m1(int code);                 /* 4-bit code -> value */
float gf4_e4m3(uint8_t b);                /* finite E4M3 -> value (NaN codes -> NaN) */
float gf4_ue8m0(uint8_t b);               /* 2^(b-127) */
float gf4_bf16(uint16_t h);

/* ---- activations */
/* q[cols], scale[cols / group]; group 0 or 64: one scale per 64 values (every kernel), 16: one per K16 group (W4A8 only) */
typedef struct { int cols; int8_t *q; float *scale; int group; } gf4_q8act;
void gf4_quantize_q8(const float *x, int cols, int8_t *q, float *scale);
void gf4_quantize_q8g(const float *x, int cols, int group, int8_t *q, float *scale);   /* group divides cols */

/* ---- packing (portable C) */
size_t gf4_nvfp4_packed_bytes(int rows, int cols);                 /* rows * cols * 9 / 16 */
/* returns 0, or -1 when a scale is negative or NaN (the arithmetic scale decode assumes 0 <= scale <= 448) */
int gf4_pack_rt16q(const uint8_t *codes, const uint8_t *scales, int rows, int cols, uint8_t *dst);
int gf4_pack_rt16p(const uint8_t *codes, const uint8_t *scales, int rows, int cols, uint8_t *dst);
size_t gf4_e8k_packed_bytes(int rows, int cols, int mx);           /* mx: 1 MXFP8, 0 BFP8 */
void gf4_pack_e8k_mx(const uint8_t *w, const uint8_t *e8m0, int rows, int cols, uint8_t *dst);
void gf4_pack_e8k_bfp8(const uint8_t *w, const float *bscale, int rows, int cols, uint8_t *dst);   /* bscale[rows/128][cols/128] */
size_t gf4_q8k_packed_bytes(int rows, int cols);
/* requantize F32 weights to int8 per (row, K32) with an F16 scale max|w|/127 */
void gf4_pack_q8k(const float *w, int rows, int cols, uint8_t *dst);
void gf4_pack_b16k(const uint16_t *w, int rows, int cols, uint8_t *dst);

/* ---- kernels: tiles [t0, t1) of a packed matrix; act pointers per token */
void gf4_w4a8(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy);
void gf4_w4a16q(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
void gf4_w4a16p(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
void gf4_mxfp8(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
void gf4_bfp8(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
void gf4_q8(const uint8_t *m, int cols, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy);
void gf4_bf16k(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
/* the same product over columns [k0, k1) only (multiples of 32): partial sums for splitting a few-tile matrix along K */
void gf4_bf16k_part(const uint8_t *m, int cols, int k0, int k1, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
/* one 16-row tile of an exact float layout (kind 0 MXFP8, 1 BFP8, 2 BF16) for up to 2048 tokens (prefill): the tile
   decoded once per slab of up to 256 columns into F32 scratch, every group of 4 tokens through gf4_mxfp8 / gf4_bfp8 /
   gf4_bf16k's chains over the slab, the running sums kept in F32 between slabs, so the result per token is bitwise
   that of those kernels. m: the tile's first block; token n's rows at y + n * ldy (nt <= 2048) */
void gf4_tile_many(int kind, const uint8_t *m, int cols, const float *const *x, int nt, float *y, int ldy);
/* two tiles (rows of m0 to y0, of m1 to y1), each bitwise gf4_tile_many's */
void gf4_tile_many2(int kind, const uint8_t *m0, const uint8_t *m1, int cols, const float *const *x, int nt, float *y0, float *y1, int ldy);
/* gf4_w4a8 for tiles [t0, t1) and any number of tokens (prefill): 16 tokens per vector against broadcast weight rows,
   each tile decoded once per slab for up to 32 tokens; bitwise equal to gf4_w4a8 over groups of 4 tokens */
void gf4_w4a8_gemm(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy);
/* gf4_w4a8_gemm with the activations transposed once for every caller (an expert use's tokens, shared by its tiles
   over the team): gf4_w4a8_gemm_blocks(nt) blocks of 16 tokens, gf4_w4a8_gemm_prep writes them to xt (blocks * cols * 16
   bytes) and xs (blocks * (cols / group) * 16 floats); gf4_w4a8_gemm_t reads them (act for the tokens past the blocks:
   one scale group for all, bitwise gf4_w4a8_gemm) */
int gf4_w4a8_gemm_blocks(int nt);
void gf4_w4a8_gemm_prep(const gf4_q8act *act, int nt, int cols, int8_t *xt, float *xs);
void gf4_w4a8_gemm_t(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, const int8_t *xt,
                     const float *xs, float *y, int ldy);

/* scalar replicas (same operation order, any ISA) */
void gf4_w4a8_ref(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy);
void gf4_w4a16_ref(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy, int pair);
void gf4_mxfp8_ref(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
void gf4_bfp8_ref(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
void gf4_q8_ref(const uint8_t *m, int cols, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy);
void gf4_bf16k_ref(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy);
void gf4_bf16k_part_ref(const uint8_t *m, int cols, int k0, int k1, int t0, int t1, const float *const *x, int nt, float *y, int ldy);

/* ---- vector transcendental functions (approximate: a few ulp from libm's; STRATA_RUNNER_FAST_EXP) over n values,
   in place allowed. exp: FEXPA 2^(k/64) table times a cubic in the remainder (|r| <= ln2/128), inputs clamped to
   [-87.3, 88]; sigmoid 1/(1+exp(-x)) by FRECPE and two Newton steps; silu x sigmoid(x); swiglu ref::swiglu's
   clamps then g sigmoid(g) u; exp_f64 the same in double with a degree-5 polynomial */
void gf4_vexp(float *y, const float *x, int n);
void gf4_vsigmoid(float *y, const float *x, int n);
void gf4_vsilu(float *y, const float *x, int n);
void gf4_vswiglu(float *y, const float *g, const float *u, int n, float limit);
void gf4_vexp_f64(double *y, const double *x, long n);

int gf4_have_sve512(void);
/* decodes all 256 E4M3 codes with the vector path and compares with gf4_e4m3; 0 = ok (FPCR flush-to-zero breaks it) */
int gf4_selftest_e4m3(void);

#ifdef __cplusplus
}
#endif
