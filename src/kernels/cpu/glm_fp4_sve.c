/* src/kernels/cpu/glm_fp4_sve.c - see strata/kernels/cpu/glm_fp4_sve.h. Packing and scalar replicas are portable C; the kernels need
 * SVE built with -msve-vector-bits=512 (A64FX) and fall back to the replicas elsewhere. No -ffast-math: the vector kernels and the
 * replicas use the same fused multiply-adds in the same order, so their results are bitwise equal. */
#include "strata/kernels/cpu/glm_fp4_sve.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

static const int8_t E2M1X2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};   /* 2 x E2M1 */

float gf4_e2m1(int code) { return 0.5f * E2M1X2[code & 15]; }

float gf4_e4m3(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = e ? ldexpf(1.f + m / 8.f, e - 7) : ldexpf(m / 8.f, -6);
    return s ? -v : v;
}

float gf4_ue8m0(uint8_t b) { return ldexpf(1.f, (int) b - 127); }

float gf4_bf16(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }

static float h2f(uint16_t h) { _Float16 v; memcpy(&v, &h, 2); return (float) v; }
static uint16_t f2h(float f) { _Float16 v = (_Float16) f; uint16_t h; memcpy(&h, &v, 2); return h; }

/* E4M3 scale or weight times 2^-8, exactly (the vector path: lsl 7 into an F16, widen) */
static float e4m3_dec(uint8_t b) { return gf4_e4m3(b) * 0x1p-8f; }

void gf4_quantize_q8g(const float *x, int cols, int group, int8_t *q, float *scale)
{
    for (int b = 0; b < cols / group; ++b) {
        float mag = 0;
        for (int i = 0; i < group; ++i) mag = fmaxf(mag, fabsf(x[group * b + i]));
        float s = mag / 127.f;
        if (mag > 0 && s == 0) s = mag;
        scale[b] = s;
        for (int i = 0; i < group; ++i) {
            const float v = s == 0 ? 0.f : fminf(fmaxf(nearbyintf(x[group * b + i] / s), -127.f), 127.f);
            q[group * b + i] = (int8_t) v;
        }
    }
}

void gf4_quantize_q8(const float *x, int cols, int8_t *q, float *scale) { gf4_quantize_q8g(x, cols, 64, q, scale); }

/* ---------------------------------------------------------------------------------------------------------------- packing */
size_t gf4_nvfp4_packed_bytes(int rows, int cols) { return (size_t) rows * cols / 64 * GF4_SB_BYTES / 16; }

static int code_at(const uint8_t *codes, int cols, int r, int k) { return (codes[(size_t) r * (cols / 2) + k / 2] >> (4 * (k & 1))) & 15; }

static int pack_nvfp4(const uint8_t *codes, const uint8_t *scales, int rows, int cols, uint8_t *dst, int pair)
{
    const int nsb = cols / 64;
    for (int t = 0; t < rows / 16; ++t)
        for (int sb = 0; sb < nsb; ++sb) {
            uint8_t *o = dst + ((size_t) t * nsb + sb) * GF4_SB_BYTES;
            for (int g = 0; g < 4; ++g) {
                const int k0 = 64 * sb + 16 * g;
                for (int r = 0; r < 16; ++r) {
                    const int row = 16 * t + r;
                    const uint8_t s = scales[(size_t) row * (cols / 16) + k0 / 16];
                    if (s & 0x80 || s == 0x7f) return -1;
                    o[512 + 16 * g + r] = s;
                    if (pair) {
                        for (int p = 0; p < 8; ++p)
                            o[128 * g + 16 * p + r] = (uint8_t) (code_at(codes, cols, row, k0 + 2 * p) | code_at(codes, cols, row, k0 + 2 * p + 1) << 4);
                    } else {
                        for (int v = 0; v < 2; ++v)
                            for (int i = 0; i < 4; ++i)
                                o[128 * g + 64 * v + 4 * r + i] = (uint8_t) (code_at(codes, cols, row, k0 + 8 * v + i)
                                                                             | code_at(codes, cols, row, k0 + 8 * v + 4 + i) << 4);
                    }
                }
            }
        }
    return 0;
}

int gf4_pack_rt16q(const uint8_t *codes, const uint8_t *scales, int rows, int cols, uint8_t *dst)
{ return pack_nvfp4(codes, scales, rows, cols, dst, 0); }
int gf4_pack_rt16p(const uint8_t *codes, const uint8_t *scales, int rows, int cols, uint8_t *dst)
{ return pack_nvfp4(codes, scales, rows, cols, dst, 1); }

enum { MX_BLOCK = 32 * 16 + 16, BFP_BLOCK = 128 * 16 + 16 };   /* bytes per tile per K32 (MXFP8) / K128 (BFP8) block */

size_t gf4_e8k_packed_bytes(int rows, int cols, int mx)
{ return (size_t) rows / 16 * (mx ? (size_t) cols / 32 * MX_BLOCK : (size_t) cols / 128 * BFP_BLOCK); }

void gf4_pack_e8k_mx(const uint8_t *w, const uint8_t *e8m0, int rows, int cols, uint8_t *dst)
{
    for (int t = 0; t < rows / 16; ++t)
        for (int b = 0; b < cols / 32; ++b) {
            uint8_t *o = dst + ((size_t) t * (cols / 32) + b) * MX_BLOCK;
            for (int k = 0; k < 32; ++k)
                for (int r = 0; r < 16; ++r) o[16 * k + r] = w[(size_t) (16 * t + r) * cols + 32 * b + k];
            for (int r = 0; r < 16; ++r) o[512 + r] = e8m0[(size_t) (16 * t + r) * (cols / 32) + b];
        }
}

void gf4_pack_e8k_bfp8(const uint8_t *w, const float *bscale, int rows, int cols, uint8_t *dst)
{
    for (int t = 0; t < rows / 16; ++t)
        for (int b = 0; b < cols / 128; ++b) {
            uint8_t *o = dst + ((size_t) t * (cols / 128) + b) * BFP_BLOCK;
            for (int k = 0; k < 128; ++k)
                for (int r = 0; r < 16; ++r) o[16 * k + r] = w[(size_t) (16 * t + r) * cols + 128 * b + k];
            memset(o + 2048, 0, 16);
            memcpy(o + 2048, &bscale[(size_t) (16 * t / 128) * (cols / 128) + b], 4);
        }
}

size_t gf4_q8k_packed_bytes(int rows, int cols) { return (size_t) rows / 16 * cols / 32 * GF4_Q8_BLOCK_BYTES; }

void gf4_pack_q8k(const float *w, int rows, int cols, uint8_t *dst)
{
    for (int t = 0; t < rows / 16; ++t)
        for (int b = 0; b < cols / 32; ++b) {
            uint8_t *o = dst + ((size_t) t * (cols / 32) + b) * GF4_Q8_BLOCK_BYTES;
            for (int r = 0; r < 16; ++r) {
                const float *row = w + (size_t) (16 * t + r) * cols + 32 * b;
                float mag = 0;
                for (int k = 0; k < 32; ++k) mag = fmaxf(mag, fabsf(row[k]));
                const uint16_t hs = f2h(mag / 127.f);
                const float s = h2f(hs);
                memcpy(o + 512 + 2 * r, &hs, 2);
                for (int k = 0; k < 32; ++k) {
                    const float v = s == 0 ? 0.f : fminf(fmaxf(nearbyintf(row[k] / s), -127.f), 127.f);
                    o[64 * (k / 4) + 4 * r + k % 4] = (uint8_t) (int8_t) v;
                }
            }
        }
}

void gf4_pack_b16k(const uint16_t *w, int rows, int cols, uint8_t *dst)
{
    for (int t = 0; t < rows / 16; ++t)
        for (int k = 0; k < cols; ++k)
            for (int r = 0; r < 16; ++r)
                memcpy(dst + (((size_t) t * cols + k) * 16 + r) * 2, &w[(size_t) (16 * t + r) * cols + k], 2);
}

/* ---------------------------------------------------------------------------------------------------- scalar replicas */
static int q_code(const uint8_t *g, int r, int k)          /* rt16q group g (128 B), row r, column k in 0..15 */
{
    const int j = k / 4, i = k % 4, v = j / 2;
    const uint8_t b = g[64 * v + 4 * r + i];
    return j & 1 ? b >> 4 : b & 15;
}

static int p_code(const uint8_t *g, int r, int k)          /* rt16p group g, row r, column k */
{
    const uint8_t b = g[16 * (k / 2) + r];
    return k & 1 ? b >> 4 : b & 15;
}

void gf4_w4a8_ref(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{
    const int nsb = cols / 64;
    const float G = global * 128.f;
    for (int t = t0; t < t1; ++t)
        for (int r = 0; r < 16; ++r)
            for (int n = 0; n < nt; ++n) {
                float facc = 0;
                const int g16 = act[n].group == 16;
                for (int sb = 0; sb < nsb; ++sb) {
                    const uint8_t *o = m + ((size_t) t * nsb + sb) * GF4_SB_BYTES;
                    float bacc = 0;
                    for (int g = 0; g < 4; ++g) {
                        int dot = 0;
                        for (int k = 0; k < 16; ++k) dot += E2M1X2[q_code(o + 128 * g, r, k)] * act[n].q[64 * sb + 16 * g + k];
                        const float ws = e4m3_dec(o[512 + 16 * g + r]);
                        bacc = fmaf((float) dot, g16 ? ws * act[n].scale[4 * sb + g] : ws, bacc);
                    }
                    facc = g16 ? facc + bacc : fmaf(bacc, act[n].scale[sb], facc);
                }
                y[(size_t) n * ldy + 16 * t + r] = facc * G;
            }
}

void gf4_w4a16_ref(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy, int pair)
{
    const int nsb = cols / 64;
    const float G = global * 128.f;
    for (int t = t0; t < t1; ++t)
        for (int r = 0; r < 16; ++r)
            for (int n = 0; n < nt; ++n) {
                float sum = 0;
                for (int sb = 0; sb < nsb; ++sb) {
                    const uint8_t *o = m + ((size_t) t * nsb + sb) * GF4_SB_BYTES;
                    for (int g = 0; g < 4; ++g) {
                        const float *xg = x[n] + 64 * sb + 16 * g;
                        float p[4];
                        for (int j = 0; j < 4; ++j) {
                            p[j] = 0;
                            for (int i = 0; i < 4; ++i) {
                                const int c = pair ? p_code(o + 128 * g, r, 4 * j + i) : q_code(o + 128 * g, r, 4 * j + i);
                                p[j] = fmaf((float) E2M1X2[c], xg[4 * j + i], p[j]);
                            }
                        }
                        sum = fmaf((p[0] + p[1]) + (p[2] + p[3]), e4m3_dec(o[512 + 16 * g + r]), sum);
                    }
                }
                y[(size_t) n * ldy + 16 * t + r] = sum * G;
            }
}

void gf4_mxfp8_ref(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{
    const int nb = cols / 32;
    for (int t = t0; t < t1; ++t)
        for (int r = 0; r < 16; ++r)
            for (int n = 0; n < nt; ++n) {
                float sum = 0;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t *o = m + ((size_t) t * nb + b) * MX_BLOCK;
                    float q[4] = {0, 0, 0, 0};
                    for (int k = 0; k < 32; ++k) q[k / 8] = fmaf(e4m3_dec(o[16 * k + r]), x[n][32 * b + k], q[k / 8]);
                    sum = fmaf((q[0] + q[1]) + (q[2] + q[3]), gf4_ue8m0(o[512 + r]), sum);
                }
                y[(size_t) n * ldy + 16 * t + r] = sum * 256.f;
            }
}

void gf4_bfp8_ref(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{
    const int nb = cols / 128;
    for (int t = t0; t < t1; ++t)
        for (int r = 0; r < 16; ++r)
            for (int n = 0; n < nt; ++n) {
                float sum = 0;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t *o = m + ((size_t) t * nb + b) * BFP_BLOCK;
                    float q[4] = {0, 0, 0, 0}, s;
                    for (int k = 0; k < 128; ++k) q[(k / 8) % 4] = fmaf(e4m3_dec(o[16 * k + r]), x[n][128 * b + k], q[(k / 8) % 4]);
                    memcpy(&s, o + 2048, 4);
                    sum = fmaf((q[0] + q[1]) + (q[2] + q[3]), s, sum);
                }
                y[(size_t) n * ldy + 16 * t + r] = sum * 256.f;
            }
}

void gf4_q8_ref(const uint8_t *m, int cols, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{
    const int nb = cols / 32;
    for (int t = t0; t < t1; ++t)
        for (int r = 0; r < 16; ++r)
            for (int n = 0; n < nt; ++n) {
                float facc = 0, bacc = 0;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t *o = m + ((size_t) t * nb + b) * GF4_Q8_BLOCK_BYTES;
                    int dot = 0;
                    for (int k = 0; k < 32; ++k) dot += (int8_t) o[64 * (k / 4) + 4 * r + k % 4] * act[n].q[32 * b + k];
                    uint16_t hs; memcpy(&hs, o + 512 + 2 * r, 2);
                    bacc = fmaf((float) dot, h2f(hs), bacc);
                    if (b & 1) { facc = fmaf(bacc, act[n].scale[b / 2], facc); bacc = 0; }
                }
                y[(size_t) n * ldy + 16 * t + r] = facc;
            }
}

void gf4_bf16k_ref(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{
    gf4_bf16k_part_ref(m, cols, 0, cols, t0, t1, x, nt, y, ldy);
}

void gf4_bf16k_part_ref(const uint8_t *m, int cols, int k0, int k1, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{
    for (int t = t0; t < t1; ++t)
        for (int r = 0; r < 16; ++r)
            for (int n = 0; n < nt; ++n) {
                float sum = 0;
                for (int b = k0 / 32; b < k1 / 32; ++b) {
                    float q[4] = {0, 0, 0, 0};
                    for (int k = 32 * b; k < 32 * b + 32; ++k) {
                        uint16_t h; memcpy(&h, m + (((size_t) t * cols + k) * 16 + r) * 2, 2);
                        q[(k % 32) / 8] = fmaf(gf4_bf16(h), x[n][k], q[(k % 32) / 8]);
                    }
                    sum += (q[0] + q[1]) + (q[2] + q[3]);
                }
                y[(size_t) n * ldy + 16 * t + r] = sum;
            }
}

/* ---------------------------------------------------------------------------------------------------------- SVE kernels */
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 512

/* fixed-length (-msve-vector-bits=512) types so vectors can live in arrays */
typedef svfloat32_t vf32 __attribute__((arm_sve_vector_bits(512)));
typedef svint8_t vs8 __attribute__((arm_sve_vector_bits(512)));
typedef svint32_t vs32 __attribute__((arm_sve_vector_bits(512)));

int gf4_have_sve512(void) { return svcntb() == 64; }

#define INL static inline __attribute__((always_inline))

/* 16 E4M3 bytes (non-negative scales) -> F32 x 2^-8 */
INL svfloat32_t scale16(const uint8_t *p)
{
    const svbool_t pw = svptrue_b32();
    return svcvt_f32_f16_x(pw, svreinterpret_f16_u32(svlsl_n_u32_x(pw, svld1ub_u32(pw, p), 7)));
}

/* 16 signed E4M3 bytes -> F32 x 2^-8 (sign and subnormals exact) */
INL svfloat32_t e4m3x16(const uint8_t *p)
{
    const svbool_t pw = svptrue_b32();
    const svint32_t v = svand_n_s32_x(pw, svlsl_n_s32_x(pw, svld1sb_s32(pw, (const int8_t *) p), 7), 0xBF80);
    return svcvt_f32_f16_x(pw, svreinterpret_f16_s32(v));
}

int gf4_selftest_e4m3(void)
{
    uint8_t all[256];
    for (int i = 0; i < 256; ++i) all[i] = (uint8_t) i;
    for (int c = 0; c < 256; c += 16) {
        float got[16];
        svst1_f32(svptrue_b32(), got, e4m3x16(all + c));
        for (int i = 0; i < 16; ++i) {
            const float want = gf4_e4m3((uint8_t) (c + i)) * 0x1p-8f;
            if (isnan(want)) continue;
            if (memcmp(&got[i], &want, 4)) return -1;
        }
    }
    return 0;
}

INL void decode_group_q(const uint8_t *g, svint8_t tbl, vs8 w[4])
{
    const svbool_t pb = svptrue_b8();
    const svuint8_t c0 = svld1_u8(pb, g), c1 = svld1_u8(pb, g + 64);
    w[0] = svtbl_s8(tbl, svand_n_u8_x(pb, c0, 15));
    w[1] = svtbl_s8(tbl, svlsr_n_u8_x(pb, c0, 4));
    w[2] = svtbl_s8(tbl, svand_n_u8_x(pb, c1, 15));
    w[3] = svtbl_s8(tbl, svlsr_n_u8_x(pb, c1, 4));
}

/* lane: the indexed SDOT on a quad-replicated activation load (GF4_W4A8_LANE=1); otherwise vector SDOTs on 4-byte
   activation broadcasts (ld1rw): the same integer dots (A64FX issues the indexed form as 2 uops on both pipes) */
INL void w4a8_tile(const uint8_t *m, int nsb, float G, const gf4_q8act *act, const int nt, float *y, int ldy, svint8_t tbl, const int lane)
{
    const svbool_t pw = svptrue_b32();
    vf32 facc[4];
    for (int n = 0; n < nt; ++n) facc[n] = svdup_n_f32(0);
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t *o = m + (size_t) sb * GF4_SB_BYTES;
        vs8 w[4][4];
        vf32 sc[4];
        for (int g = 0; g < 4; ++g) { decode_group_q(o + 128 * g, tbl, w[g]); sc[g] = scale16(o + 512 + 16 * g); }
        for (int n = 0; n < nt; ++n) {
            const int8_t *xq = act[n].q + 64 * sb;
            const int g16 = act[n].group == 16;   /* one activation scale per K16 group instead of per superblock */
            svfloat32_t bacc = svdup_n_f32(0);
            for (int g = 0; g < 4; ++g) {
                svint32_t d = svdup_n_s32(0);
                if (lane) {
                    const svint8_t xv = svld1rq_s8(svptrue_b8(), xq + 16 * g);
                    d = svdot_lane_s32(d, w[g][0], xv, 0);
                    d = svdot_lane_s32(d, w[g][1], xv, 1);
                    d = svdot_lane_s32(d, w[g][2], xv, 2);
                    d = svdot_lane_s32(d, w[g][3], xv, 3);
                } else {
                    int32_t xw[4];
                    memcpy(xw, xq + 16 * g, 16);
                    d = svdot_s32(d, w[g][0], svreinterpret_s8_s32(svdup_n_s32(xw[0])));
                    d = svdot_s32(d, w[g][1], svreinterpret_s8_s32(svdup_n_s32(xw[1])));
                    d = svdot_s32(d, w[g][2], svreinterpret_s8_s32(svdup_n_s32(xw[2])));
                    d = svdot_s32(d, w[g][3], svreinterpret_s8_s32(svdup_n_s32(xw[3])));
                }
                svfloat32_t s = sc[g];
                if (g16) s = svmul_n_f32_x(pw, s, act[n].scale[4 * sb + g]);
                bacc = svmla_f32_x(pw, bacc, svcvt_f32_s32_x(pw, d), s);
            }
            facc[n] = g16 ? svadd_f32_x(pw, facc[n], bacc) : svmla_n_f32_x(pw, facc[n], bacc, act[n].scale[sb]);
        }
    }
    for (int n = 0; n < nt; ++n) svst1_f32(pw, y + (size_t) n * ldy, svmul_n_f32_x(pw, facc[n], G));
}

static svint8_t e2m1_table(void)
{
    int8_t t[64] = {0};
    memcpy(t, E2M1X2, 16);
    return svld1_s8(svptrue_b8(), t);
}

void gf4_w4a8(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{
    const int nsb = cols / 64;
    const svint8_t tbl = e2m1_table();
    const float G = global * 128.f;
    static int lane = -1;
    if (lane < 0) { const char *e = getenv("GF4_W4A8_LANE"); lane = e && e[0] == '1'; }
    for (int t = t0; t < t1; ++t) {
        const uint8_t *mt = m + (size_t) t * nsb * GF4_SB_BYTES;
#define W4A8_CASES(L)                                                                   \
        switch (nt) {   /* constant token counts so the accumulators stay in registers */ \
        case 1: w4a8_tile(mt, nsb, G, act, 1, y + 16 * t, ldy, tbl, L); break;          \
        case 2: w4a8_tile(mt, nsb, G, act, 2, y + 16 * t, ldy, tbl, L); break;          \
        case 3: w4a8_tile(mt, nsb, G, act, 3, y + 16 * t, ldy, tbl, L); break;          \
        default: w4a8_tile(mt, nsb, G, act, 4, y + 16 * t, ldy, tbl, L); break;         \
        }
        if (lane) { W4A8_CASES(1) } else { W4A8_CASES(0) }
#undef W4A8_CASES
    }
}

/* lane r's byte i of an int8 vector as F32 */
INL svfloat32_t byte_f32(svint8_t w, int i)
{
    const svbool_t pw = svptrue_b32();
    const svint32_t v = svreinterpret_s32_s8(w);
    return svcvt_f32_s32_x(pw, svasr_n_s32_x(pw, svlsl_n_s32_x(pw, v, 24 - 8 * i), 24));
}

/* exact NVFP4 group product for nt tokens: p[n][j] chains in column order, then (p0+p1)+(p2+p3) */
INL void w4a16_group(const vf32 wf[16], const float *const *x, int col, const int nt, svfloat32_t sc, vf32 *sum)
{
    const svbool_t pw = svptrue_b32();
    for (int n = 0; n < nt; ++n) {
        vf32 p[4];
        for (int j = 0; j < 4; ++j) {
            const svfloat32_t xv = svld1rq_f32(pw, x[n] + col + 4 * j);
            p[j] = svmla_lane_f32(svdup_n_f32(0), wf[4 * j], xv, 0);
            p[j] = svmla_lane_f32(p[j], wf[4 * j + 1], xv, 1);
            p[j] = svmla_lane_f32(p[j], wf[4 * j + 2], xv, 2);
            p[j] = svmla_lane_f32(p[j], wf[4 * j + 3], xv, 3);
        }
        const svfloat32_t acc = svadd_f32_x(pw, svadd_f32_x(pw, p[0], p[1]), svadd_f32_x(pw, p[2], p[3]));
        sum[n] = svmla_f32_x(pw, sum[n], acc, sc);
    }
}

INL void w4a16_tile(const uint8_t *m, int nsb, float G, const float *const *x, const int nt, float *y, int ldy, int pair)
{
    const svbool_t pw = svptrue_b32();
    vf32 sum[4];
    for (int n = 0; n < nt; ++n) sum[n] = svdup_n_f32(0);
    float t32[16];
    for (int i = 0; i < 16; ++i) t32[i] = (float) E2M1X2[i];
    const svfloat32_t tf = svld1_f32(pw, t32);
    const svint8_t tbl = e2m1_table();
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t *o = m + (size_t) sb * GF4_SB_BYTES;
        for (int g = 0; g < 4; ++g) {
            vf32 wf[16];
            if (pair) {
                for (int p = 0; p < 8; ++p) {
                    const svuint32_t v = svld1ub_u32(pw, o + 128 * g + 16 * p);
                    wf[2 * p] = svtbl_f32(tf, svand_n_u32_x(pw, v, 15));
                    wf[2 * p + 1] = svtbl_f32(tf, svlsr_n_u32_x(pw, v, 4));
                }
            } else {
                vs8 w[4];
                decode_group_q(o + 128 * g, tbl, w);
                for (int j = 0; j < 4; ++j)
                    for (int i = 0; i < 4; ++i) wf[4 * j + i] = byte_f32(w[j], i);
            }
            w4a16_group(wf, x, 64 * sb + 16 * g, nt, scale16(o + 512 + 16 * g), sum);
        }
    }
    for (int n = 0; n < nt; ++n) svst1_f32(pw, y + (size_t) n * ldy, svmul_n_f32_x(pw, sum[n], G));
}

static int getenv_w4a16_old(void)   /* GF4_W4A16_OLD=1: the general tile for one token, 2: two tiles at once (A/B) */
{
    static int v = -1;
    if (v < 0) { const char *e = getenv("GF4_W4A16_OLD"); v = e ? e[0] - '0' : 0; }
    return v;
}

/* one token, rt16p: two K16 groups interleaved, their weight vectors decoded four at a time next to the FMAs that use
   them (w4a16_tile decodes a group's 16 vectors first and ran one group's dependent chains at a time, ~105 cycles per
   group measured); per group the same products, chains and order as w4a16_group, so the bits are unchanged */
INL void w4a16p_chain4(const uint8_t *g, const float *xg, int j, svfloat32_t tf, vf32 *acc)
{
    const svbool_t pw = svptrue_b32();
    const svuint32_t v0 = svld1ub_u32(pw, g + 16 * (2 * j)), v1 = svld1ub_u32(pw, g + 16 * (2 * j + 1));
    const svfloat32_t xv = svld1rq_f32(pw, xg + 4 * j);
    svfloat32_t a = svmla_lane_f32(svdup_n_f32(0), svtbl_f32(tf, svand_n_u32_x(pw, v0, 15)), xv, 0);
    a = svmla_lane_f32(a, svtbl_f32(tf, svlsr_n_u32_x(pw, v0, 4)), xv, 1);
    a = svmla_lane_f32(a, svtbl_f32(tf, svand_n_u32_x(pw, v1, 15)), xv, 2);
    a = svmla_lane_f32(a, svtbl_f32(tf, svlsr_n_u32_x(pw, v1, 4)), xv, 3);
    *acc = a;
}
INL void w4a16p_tile1(const uint8_t *m, int nsb, float G, const float *x, float *y)
{
    const svbool_t pw = svptrue_b32();
    float t32[16];
    for (int i = 0; i < 16; ++i) t32[i] = (float) E2M1X2[i];
    const svfloat32_t tf = svld1_f32(pw, t32);
    svfloat32_t sum = svdup_n_f32(0);
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t *o = m + (size_t) sb * GF4_SB_BYTES;
#pragma GCC unroll 2
        for (int g = 0; g < 4; g += 2) {
            const float *xa = x + 64 * sb + 16 * g;
            vf32 pa[4], pb[4];
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                w4a16p_chain4(o + 128 * g, xa, j, tf, &pa[j]);
                w4a16p_chain4(o + 128 * (g + 1), xa + 16, j, tf, &pb[j]);
            }
            const svfloat32_t aa = svadd_f32_x(pw, svadd_f32_x(pw, pa[0], pa[1]), svadd_f32_x(pw, pa[2], pa[3]));
            const svfloat32_t ab = svadd_f32_x(pw, svadd_f32_x(pw, pb[0], pb[1]), svadd_f32_x(pw, pb[2], pb[3]));
            sum = svmla_f32_x(pw, sum, aa, scale16(o + 512 + 16 * g));
            sum = svmla_f32_x(pw, sum, ab, scale16(o + 512 + 16 * (g + 1)));
        }
    }
    svst1_f32(pw, y, svmul_n_f32_x(pw, sum, G));
}

/* one token, rt16p, two tiles (32 rows) at once: independent chains for both, each x broadcast shared; per row the
   same products, chains and order as w4a16_group (bitwise unchanged) */
INL void w4a16p_chain4x(const uint8_t *g, svfloat32_t xv, int j, svfloat32_t tf, vf32 *acc)
{
    const svbool_t pw = svptrue_b32();
    const svuint32_t v0 = svld1ub_u32(pw, g + 16 * (2 * j)), v1 = svld1ub_u32(pw, g + 16 * (2 * j + 1));
    svfloat32_t a = svmla_lane_f32(svdup_n_f32(0), svtbl_f32(tf, svand_n_u32_x(pw, v0, 15)), xv, 0);
    a = svmla_lane_f32(a, svtbl_f32(tf, svlsr_n_u32_x(pw, v0, 4)), xv, 1);
    a = svmla_lane_f32(a, svtbl_f32(tf, svand_n_u32_x(pw, v1, 15)), xv, 2);
    a = svmla_lane_f32(a, svtbl_f32(tf, svlsr_n_u32_x(pw, v1, 4)), xv, 3);
    *acc = a;
}
/* variant without TBL: a nibble's E2M1 bits as an fp16 (e, m into bits 11..9, sign into 15) converts to its value
   x 2^-14 exactly (the table path uses 2 x E2M1); 2^15 goes into the group scale (exact), so the bits are unchanged */
INL svfloat32_t e2m1_lo(svuint32_t b)
{
    const svbool_t pw = svptrue_b32();
    const svuint32_t h = svorr_u32_x(pw, svlsl_n_u32_x(pw, svand_n_u32_x(pw, b, 7), 9), svlsl_n_u32_x(pw, svand_n_u32_x(pw, b, 8), 12));
    return svcvt_f32_f16_x(pw, svreinterpret_f16_u32(h));
}
INL svfloat32_t e2m1_hi(svuint32_t b)
{
    const svbool_t pw = svptrue_b32();
    const svuint32_t h = svorr_u32_x(pw, svlsl_n_u32_x(pw, svand_n_u32_x(pw, b, 0x70), 5), svlsl_n_u32_x(pw, svand_n_u32_x(pw, b, 0x80), 8));
    return svcvt_f32_f16_x(pw, svreinterpret_f16_u32(h));
}
INL void w4a16p_tile1c(const uint8_t *m, int nsb, float G, const float *x, float *y)
{
    const svbool_t pw = svptrue_b32();
    svfloat32_t sum = svdup_n_f32(0);
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t *o = m + (size_t) sb * GF4_SB_BYTES;
#pragma GCC unroll 2
        for (int g = 0; g < 4; g += 2) {
            const float *xa = x + 64 * sb + 16 * g;
            vf32 pa[4], pb[4];
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j)
                for (int h = 0; h < 2; ++h) {
                    const uint8_t *gg = o + 128 * (g + h);
                    const svuint32_t v0 = svld1ub_u32(pw, gg + 16 * (2 * j)), v1 = svld1ub_u32(pw, gg + 16 * (2 * j + 1));
                    const svfloat32_t xv = svld1rq_f32(pw, xa + 16 * h + 4 * j);
                    svfloat32_t a = svmla_lane_f32(svdup_n_f32(0), e2m1_lo(v0), xv, 0);
                    a = svmla_lane_f32(a, e2m1_hi(v0), xv, 1);
                    a = svmla_lane_f32(a, e2m1_lo(v1), xv, 2);
                    a = svmla_lane_f32(a, e2m1_hi(v1), xv, 3);
                    (h ? pb : pa)[j] = a;
                }
            const svfloat32_t aa = svadd_f32_x(pw, svadd_f32_x(pw, pa[0], pa[1]), svadd_f32_x(pw, pa[2], pa[3]));
            const svfloat32_t ab = svadd_f32_x(pw, svadd_f32_x(pw, pb[0], pb[1]), svadd_f32_x(pw, pb[2], pb[3]));
            sum = svmla_f32_x(pw, sum, aa, svmul_n_f32_x(pw, scale16(o + 512 + 16 * g), 32768.f));
            sum = svmla_f32_x(pw, sum, ab, svmul_n_f32_x(pw, scale16(o + 512 + 16 * (g + 1)), 32768.f));
        }
    }
    svst1_f32(pw, y, svmul_n_f32_x(pw, sum, G));
}

INL void w4a16p_tile2(const uint8_t *m0, const uint8_t *m1, int nsb, float G, const float *x, float *y0, float *y1)
{
    const svbool_t pw = svptrue_b32();
    float t32[16];
    for (int i = 0; i < 16; ++i) t32[i] = (float) E2M1X2[i];
    const svfloat32_t tf = svld1_f32(pw, t32);
    svfloat32_t s0 = svdup_n_f32(0), s1 = svdup_n_f32(0);
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t *o0 = m0 + (size_t) sb * GF4_SB_BYTES, *o1 = m1 + (size_t) sb * GF4_SB_BYTES;
#pragma GCC unroll 4
        for (int g = 0; g < 4; ++g) {
            const float *xg = x + 64 * sb + 16 * g;
            vf32 pa[4], pb[4];
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const svfloat32_t xv = svld1rq_f32(pw, xg + 4 * j);
                w4a16p_chain4x(o0 + 128 * g, xv, j, tf, &pa[j]);
                w4a16p_chain4x(o1 + 128 * g, xv, j, tf, &pb[j]);
            }
            s0 = svmla_f32_x(pw, s0, svadd_f32_x(pw, svadd_f32_x(pw, pa[0], pa[1]), svadd_f32_x(pw, pa[2], pa[3])), scale16(o0 + 512 + 16 * g));
            s1 = svmla_f32_x(pw, s1, svadd_f32_x(pw, svadd_f32_x(pw, pb[0], pb[1]), svadd_f32_x(pw, pb[2], pb[3])), scale16(o1 + 512 + 16 * g));
        }
    }
    svst1_f32(pw, y0, svmul_n_f32_x(pw, s0, G));
    svst1_f32(pw, y1, svmul_n_f32_x(pw, s1, G));
}

static void w4a16(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy, int pair)
{
    const int nsb = cols / 64;
    const float G = global * 128.f;
    for (int t = t0; t < t1; ++t) {
        const uint8_t *mt = m + (size_t) t * nsb * GF4_SB_BYTES;
        if (nt == 1 && pair && getenv_w4a16_old() == 0) { w4a16p_tile1(mt, nsb, G, x[0], y + 16 * t); continue; }
        if (nt == 1 && pair && getenv_w4a16_old() == 3) { w4a16p_tile1c(mt, nsb, G, x[0], y + 16 * t); continue; }
        if (nt == 1 && pair && getenv_w4a16_old() == 2 && t + 1 < t1) {   /* two tiles at once: measured no faster */
            w4a16p_tile2(mt, mt + (size_t) nsb * GF4_SB_BYTES, nsb, G, x[0], y + 16 * t, y + 16 * (t + 1));
            ++t;
            continue;
        }
        switch (nt) {
        case 1: w4a16_tile(mt, nsb, G, x, 1, y + 16 * t, ldy, pair); break;
        case 2: w4a16_tile(mt, nsb, G, x, 2, y + 16 * t, ldy, pair); break;
        case 3: w4a16_tile(mt, nsb, G, x, 3, y + 16 * t, ldy, pair); break;
        default: w4a16_tile(mt, nsb, G, x, 4, y + 16 * t, ldy, pair); break;
        }
    }
}

void gf4_w4a16q(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{ w4a16(m, cols, global, t0, t1, x, nt, y, ldy, 0); }
void gf4_w4a16p(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{ w4a16(m, cols, global, t0, t1, x, nt, y, ldy, 1); }

/* E4M3 columns (16 rows each) [k0, k0 + 32): q[(k % 32) / 8] chains, sum += ((q0+q1)+(q2+q3)) * scale */
/* GF4_FLOAT_LANE=1: the exact float kernels with indexed FMLAs on quad-loaded activations, four dependent FMLAs per
   chain in a row (the first form). Otherwise vector FMLAs on replicated activations (ld1rw), consecutive FMLAs on
   different chains, every chain in its own order: the same fused operations, so the same bits (A64FX issues an
   indexed FMLA as 2 uops on both pipes, and four dependent ones in a row fill its FP queues) */
static int float_lane(void)
{
    static int v = -1;
    if (v < 0) { const char *e = getenv("GF4_FLOAT_LANE"); v = e && e[0] == '1'; }
    return v;
}
INL void e8_block32(const uint8_t *cols16, const float *const *x, int k0, const int nt, svfloat32_t scale, vf32 *sum, const int lane)
{
    const svbool_t pw = svptrue_b32();
    vf32 q[4][4];
    for (int n = 0; n < nt; ++n) for (int c = 0; c < 4; ++c) q[n][c] = svdup_n_f32(0);
    if (!lane) {
#pragma GCC unroll 2
        for (int h = 0; h < 2; ++h)
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i)
#pragma GCC unroll 4
                for (int c = 0; c < 4; ++c) {
                    const int k = 8 * c + 4 * h + i;
                    const svfloat32_t wv = e4m3x16(cols16 + 16 * k);
                    for (int n = 0; n < nt; ++n) q[n][c] = svmla_f32_x(pw, q[n][c], wv, svdup_n_f32(x[n][k0 + k]));
                }
        for (int n = 0; n < nt; ++n)
            sum[n] = svmla_f32_x(pw, sum[n], svadd_f32_x(pw, svadd_f32_x(pw, q[n][0], q[n][1]), svadd_f32_x(pw, q[n][2], q[n][3])), scale);
        return;
    }
    /* chain c takes columns 8c .. 8c + 7; c and h are unrolled so the chains stay in registers (an index computed at
       run time put them in memory: a store and a load around every four FMAs) */
#pragma GCC unroll 4
    for (int c = 0; c < 4; ++c)
#pragma GCC unroll 2
        for (int h = 0; h < 2; ++h) {
            const int k = 8 * c + 4 * h;
            const svfloat32_t w0 = e4m3x16(cols16 + 16 * k), w1 = e4m3x16(cols16 + 16 * (k + 1));
            const svfloat32_t w2 = e4m3x16(cols16 + 16 * (k + 2)), w3 = e4m3x16(cols16 + 16 * (k + 3));
            for (int n = 0; n < nt; ++n) {
                const svfloat32_t xv = svld1rq_f32(pw, x[n] + k0 + k);
                svfloat32_t a = q[n][c];
                a = svmla_lane_f32(a, w0, xv, 0);
                a = svmla_lane_f32(a, w1, xv, 1);
                a = svmla_lane_f32(a, w2, xv, 2);
                a = svmla_lane_f32(a, w3, xv, 3);
                q[n][c] = a;
            }
        }
    for (int n = 0; n < nt; ++n)
        sum[n] = svmla_f32_x(pw, sum[n], svadd_f32_x(pw, svadd_f32_x(pw, q[n][0], q[n][1]), svadd_f32_x(pw, q[n][2], q[n][3])), scale);
}

INL void mxfp8_tile(const uint8_t *m, int nb, const float *const *x, const int nt, float *y, int ldy, const int lane)
{
    const svbool_t pw = svptrue_b32();
    vf32 sum[4];
    for (int n = 0; n < nt; ++n) sum[n] = svdup_n_f32(0);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *o = m + (size_t) b * MX_BLOCK;
        const svfloat32_t s = svreinterpret_f32_u32(svlsl_n_u32_x(pw, svld1ub_u32(pw, o + 512), 23));
        e8_block32(o, x, 32 * b, nt, s, sum, lane);
    }
    for (int n = 0; n < nt; ++n) svst1_f32(pw, y + (size_t) n * ldy, svmul_n_f32_x(pw, sum[n], 256.f));
}

void gf4_mxfp8(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{
    const int nb = cols / 32, lane = float_lane();
    for (int t = t0; t < t1; ++t) {
        const uint8_t *mt = m + (size_t) t * nb * MX_BLOCK;
        switch (nt) {
        case 1: mxfp8_tile(mt, nb, x, 1, y + 16 * t, ldy, lane); break;
        case 2: mxfp8_tile(mt, nb, x, 2, y + 16 * t, ldy, lane); break;
        case 3: mxfp8_tile(mt, nb, x, 3, y + 16 * t, ldy, lane); break;
        default: mxfp8_tile(mt, nb, x, 4, y + 16 * t, ldy, lane); break;
        }
    }
}

/* BFP8: one K128 block = four 32-column steps into the same four chains (q[(k / 8) % 4]), then one scalar scale */
INL void bfp8_tile(const uint8_t *m, int nb, const float *const *x, const int nt, float *y, int ldy, const int lane)
{
    const svbool_t pw = svptrue_b32();
    vf32 sum[4];
    for (int n = 0; n < nt; ++n) sum[n] = svdup_n_f32(0);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *o = m + (size_t) b * BFP_BLOCK;
        vf32 q[4][4];
        for (int n = 0; n < nt; ++n) for (int c = 0; c < 4; ++c) q[n][c] = svdup_n_f32(0);
        if (!lane)
            for (int kb = 0; kb < 128; kb += 32)
#pragma GCC unroll 2
                for (int h = 0; h < 2; ++h)
#pragma GCC unroll 4
                    for (int i = 0; i < 4; ++i)
#pragma GCC unroll 4
                        for (int c = 0; c < 4; ++c) {
                            const int k = kb + 8 * c + 4 * h + i;
                            const svfloat32_t wv = e4m3x16(o + 16 * k);
                            for (int n = 0; n < nt; ++n) q[n][c] = svmla_f32_x(pw, q[n][c], wv, svdup_n_f32(x[n][128 * b + k]));
                        }
        else
        for (int kb = 0; kb < 128; kb += 32)   /* chain c: columns kb + 8c .. kb + 8c + 7 (c, h unrolled: see e8_block32) */
#pragma GCC unroll 4
            for (int c = 0; c < 4; ++c)
#pragma GCC unroll 2
                for (int h = 0; h < 2; ++h) {
                    const int k = kb + 8 * c + 4 * h;
                    const svfloat32_t w0 = e4m3x16(o + 16 * k), w1 = e4m3x16(o + 16 * (k + 1));
                    const svfloat32_t w2 = e4m3x16(o + 16 * (k + 2)), w3 = e4m3x16(o + 16 * (k + 3));
                    for (int n = 0; n < nt; ++n) {
                        const svfloat32_t xv = svld1rq_f32(pw, x[n] + 128 * b + k);
                        svfloat32_t a = q[n][c];
                        a = svmla_lane_f32(a, w0, xv, 0);
                        a = svmla_lane_f32(a, w1, xv, 1);
                        a = svmla_lane_f32(a, w2, xv, 2);
                        a = svmla_lane_f32(a, w3, xv, 3);
                        q[n][c] = a;
                    }
                }
        float s; memcpy(&s, o + 2048, 4);
        for (int n = 0; n < nt; ++n)
            sum[n] = svmla_n_f32_x(pw, sum[n], svadd_f32_x(pw, svadd_f32_x(pw, q[n][0], q[n][1]), svadd_f32_x(pw, q[n][2], q[n][3])), s);
    }
    for (int n = 0; n < nt; ++n) svst1_f32(pw, y + (size_t) n * ldy, svmul_n_f32_x(pw, sum[n], 256.f));
}

void gf4_bfp8(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{
    const int nb = cols / 128, lane = float_lane();
    for (int t = t0; t < t1; ++t) {
        const uint8_t *mt = m + (size_t) t * nb * BFP_BLOCK;
        switch (nt) {
        case 1: bfp8_tile(mt, nb, x, 1, y + 16 * t, ldy, lane); break;
        case 2: bfp8_tile(mt, nb, x, 2, y + 16 * t, ldy, lane); break;
        case 3: bfp8_tile(mt, nb, x, 3, y + 16 * t, ldy, lane); break;
        default: bfp8_tile(mt, nb, x, 4, y + 16 * t, ldy, lane); break;
        }
    }
}

INL void q8_tile(const uint8_t *m, int nb, const gf4_q8act *act, const int nt, float *y, int ldy)
{
    const svbool_t pw = svptrue_b32(), pb = svptrue_b8();
    vf32 facc[4], bacc[4];
    for (int n = 0; n < nt; ++n) facc[n] = bacc[n] = svdup_n_f32(0);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *o = m + (size_t) b * GF4_Q8_BLOCK_BYTES;
        vs8 w[8];
        for (int j = 0; j < 8; ++j) w[j] = svld1_s8(pb, (const int8_t *) o + 64 * j);
        const svfloat32_t s = svcvt_f32_f16_x(pw, svreinterpret_f16_u32(svld1uh_u32(pw, (const uint16_t *) (o + 512))));
        for (int n = 0; n < nt; ++n) {
            const svint8_t x0 = svld1rq_s8(pb, act[n].q + 32 * b), x1 = svld1rq_s8(pb, act[n].q + 32 * b + 16);
            svint32_t d = svdup_n_s32(0);
            d = svdot_lane_s32(d, w[0], x0, 0); d = svdot_lane_s32(d, w[1], x0, 1);
            d = svdot_lane_s32(d, w[2], x0, 2); d = svdot_lane_s32(d, w[3], x0, 3);
            d = svdot_lane_s32(d, w[4], x1, 0); d = svdot_lane_s32(d, w[5], x1, 1);
            d = svdot_lane_s32(d, w[6], x1, 2); d = svdot_lane_s32(d, w[7], x1, 3);
            bacc[n] = svmla_f32_x(pw, bacc[n], svcvt_f32_s32_x(pw, d), s);
            if (b & 1) { facc[n] = svmla_n_f32_x(pw, facc[n], bacc[n], act[n].scale[b / 2]); bacc[n] = svdup_n_f32(0); }
        }
    }
    for (int n = 0; n < nt; ++n) svst1_f32(pw, y + (size_t) n * ldy, facc[n]);
}

void gf4_q8(const uint8_t *m, int cols, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{
    const int nb = cols / 32;
    for (int t = t0; t < t1; ++t) {
        const uint8_t *mt = m + (size_t) t * nb * GF4_Q8_BLOCK_BYTES;
        switch (nt) {
        case 1: q8_tile(mt, nb, act, 1, y + 16 * t, ldy); break;
        case 2: q8_tile(mt, nb, act, 2, y + 16 * t, ldy); break;
        case 3: q8_tile(mt, nb, act, 3, y + 16 * t, ldy); break;
        default: q8_tile(mt, nb, act, 4, y + 16 * t, ldy); break;
        }
    }
}

INL void bf16_tile(const uint8_t *m, int k0, int k1, const float *const *x, const int nt, float *y, int ldy, const int lane)
{
    const svbool_t pw = svptrue_b32();
    vf32 sum[4];
    for (int n = 0; n < nt; ++n) sum[n] = svdup_n_f32(0);
    for (int b = k0 / 32; b < k1 / 32; ++b) {
        vf32 q[4][4];
        for (int n = 0; n < nt; ++n) for (int c = 0; c < 4; ++c) q[n][c] = svdup_n_f32(0);
        if (!lane)
#pragma GCC unroll 2
            for (int h = 0; h < 2; ++h)
#pragma GCC unroll 4
                for (int i = 0; i < 4; ++i)
#pragma GCC unroll 4
                    for (int c = 0; c < 4; ++c) {
                        const int k = 32 * b + 8 * c + 4 * h + i;
                        const svfloat32_t wv = svreinterpret_f32_u32(svlsl_n_u32_x(pw, svld1uh_u32(pw, (const uint16_t *) (m + (size_t) k * 32)), 16));
                        for (int n = 0; n < nt; ++n) q[n][c] = svmla_f32_x(pw, q[n][c], wv, svdup_n_f32(x[n][k]));
                    }
        else
#pragma GCC unroll 4
        for (int c = 0; c < 4; ++c)   /* chain c: columns 8c .. 8c + 7 of the block (unrolled: see e8_block32) */
#pragma GCC unroll 2
            for (int h = 0; h < 2; ++h) {
                const int k = 32 * b + 8 * c + 4 * h;
                vf32 w[4];
                for (int i = 0; i < 4; ++i)
                    w[i] = svreinterpret_f32_u32(svlsl_n_u32_x(pw, svld1uh_u32(pw, (const uint16_t *) (m + (size_t) (k + i) * 32)), 16));
                for (int n = 0; n < nt; ++n) {
                    const svfloat32_t xv = svld1rq_f32(pw, x[n] + k);
                    svfloat32_t a = q[n][c];
                    a = svmla_lane_f32(a, w[0], xv, 0);
                    a = svmla_lane_f32(a, w[1], xv, 1);
                    a = svmla_lane_f32(a, w[2], xv, 2);
                    a = svmla_lane_f32(a, w[3], xv, 3);
                    q[n][c] = a;
                }
            }
        for (int n = 0; n < nt; ++n)
            sum[n] = svadd_f32_x(pw, sum[n], svadd_f32_x(pw, svadd_f32_x(pw, q[n][0], q[n][1]), svadd_f32_x(pw, q[n][2], q[n][3])));
    }
    for (int n = 0; n < nt; ++n) svst1_f32(pw, y + (size_t) n * ldy, sum[n]);
}

void gf4_bf16k_part(const uint8_t *m, int cols, int k0, int k1, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{
    const int lane = float_lane();
    for (int t = t0; t < t1; ++t) {
        const uint8_t *mt = m + (size_t) t * cols * 32;
        switch (nt) {
        case 1: bf16_tile(mt, k0, k1, x, 1, y + 16 * t, ldy, lane); break;
        case 2: bf16_tile(mt, k0, k1, x, 2, y + 16 * t, ldy, lane); break;
        case 3: bf16_tile(mt, k0, k1, x, 3, y + 16 * t, ldy, lane); break;
        default: bf16_tile(mt, k0, k1, x, 4, y + 16 * t, ldy, lane); break;
        }
    }
}

void gf4_bf16k(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{
    gf4_bf16k_part(m, cols, 0, cols, t0, t1, x, nt, y, ldy);
}

/* gf4_w4a8_gemm: tokens in the lanes. Per block of up to 32 tokens their activations are transposed once (zips) to
   [token block][4-column chunk][16 tokens x 4 bytes], their scales to [block][scale][16]; per tile and slab of 8
   superblocks the codes are decoded once to int8 ([sb][g][j] vectors) and F32 row scales. A row's 4 weight bytes go as a
   broadcast (ld1rw) against the token vector, so per (row, token) the integer dots, scale products and fused
   multiply-adds are w4a8_tile's, in its order */
enum { W4G_SLAB = 8, W4G_TB = 2 };
static _Thread_local int8_t w4g_w[W4G_SLAB * 16 * 64] __attribute__((aligned(256)));   /* [sb][g][j][row]: 4 bytes */
static _Thread_local float w4g_s[W4G_SLAB * 4 * 16] __attribute__((aligned(256)));     /* [sb][g][row] */
static _Thread_local float w4g_acc[16 * W4G_TB * 16] __attribute__((aligned(256)));    /* [row][block][token] */
static _Thread_local int8_t *w4g_x;
static _Thread_local float *w4g_xs;
static _Thread_local size_t w4g_xcap, w4g_scap;

/* 16 x 16 words: four rounds of zips transpose (v[k] lane i <- v[i] lane k) */
INL void w4g_transpose(vs32 v[16])
{
    for (int round = 0; round < 4; ++round) {
        vs32 o[16];
        for (int i = 0; i < 8; ++i) o[2 * i] = svzip1_s32(v[i], v[i + 8]), o[2 * i + 1] = svzip2_s32(v[i], v[i + 8]);
        for (int i = 0; i < 16; ++i) v[i] = o[i];
    }
}

/* rows 4 rb .. 4 rb + 3 over the decoded slab (ns superblocks from sb0), token block xa (scales sa), and xb / sb2
   when two */
INL void w4g_rows(int rb, int sb0, int ns, const int two, const int g16, const int8_t *xa, const int8_t *xb, const float *sa,
                  const float *sb2)
{
    const svbool_t pw = svptrue_b32(), pb = svptrue_b8();
    const int32_t *w32 = (const int32_t *) w4g_w;
    vf32 f[4][2];
    for (int r = 0; r < 4; ++r) {
        f[r][0] = svld1_f32(pw, w4g_acc + (4 * rb + r) * W4G_TB * 16);
        if (two) f[r][1] = svld1_f32(pw, w4g_acc + (4 * rb + r) * W4G_TB * 16 + 16);
    }
    for (int sl = 0; sl < ns; ++sl) {
        const int sb = sb0 + sl;
        vf32 b[4][2];
        for (int r = 0; r < 4; ++r) b[r][0] = b[r][1] = svdup_n_f32(0);
        for (int g = 0; g < 4; ++g) {
            vs32 d[4][2];
            for (int r = 0; r < 4; ++r) d[r][0] = d[r][1] = svdup_n_s32(0);
            for (int j = 0; j < 4; ++j) {
                const size_t c = (size_t) (16 * sb + 4 * g + j) * 64;
                const svint8_t a0 = svld1_s8(pb, xa + c), a1 = two ? svld1_s8(pb, xb + c) : a0;
                for (int r = 0; r < 4; ++r) {
                    const svint8_t w = svreinterpret_s8_s32(svdup_n_s32(w32[(sl * 16 + 4 * g + j) * 16 + 4 * rb + r]));
                    d[r][0] = svdot_s32(d[r][0], a0, w);
                    if (two) d[r][1] = svdot_s32(d[r][1], a1, w);
                }
            }
            for (int r = 0; r < 4; ++r) {
                const float ws = w4g_s[(sl * 4 + g) * 16 + 4 * rb + r];
                const svfloat32_t s0 = g16 ? svmul_n_f32_x(pw, svld1_f32(pw, sa + (4 * sb + g) * 16), ws) : svdup_n_f32(ws);
                b[r][0] = svmla_f32_x(pw, b[r][0], svcvt_f32_s32_x(pw, d[r][0]), s0);
                if (two) {
                    const svfloat32_t s1 = g16 ? svmul_n_f32_x(pw, svld1_f32(pw, sb2 + (4 * sb + g) * 16), ws) : s0;
                    b[r][1] = svmla_f32_x(pw, b[r][1], svcvt_f32_s32_x(pw, d[r][1]), s1);
                }
            }
        }
        for (int r = 0; r < 4; ++r) {
            f[r][0] = g16 ? svadd_f32_x(pw, f[r][0], b[r][0]) : svmla_f32_x(pw, f[r][0], b[r][0], svld1_f32(pw, sa + 16 * sb));
            if (two) f[r][1] = g16 ? svadd_f32_x(pw, f[r][1], b[r][1]) : svmla_f32_x(pw, f[r][1], b[r][1], svld1_f32(pw, sb2 + 16 * sb));
        }
    }
    for (int r = 0; r < 4; ++r) {
        svst1_f32(pw, w4g_acc + (4 * rb + r) * W4G_TB * 16, f[r][0]);
        if (two) svst1_f32(pw, w4g_acc + (4 * rb + r) * W4G_TB * 16 + 16, f[r][1]);
    }
}

int gf4_w4a8_gemm_blocks(int nt) { return ((nt % 16 && nt % 16 <= 8 ? nt - nt % 16 : nt) + 15) / 16; }

/* token blocks [tb0, tb1) of act (tokens past nt zero): xt + tb * cols * 16 bytes, xs + tb * (cols / group) * 16 */
static void w4g_prep(const gf4_q8act *act, int nt, int cols, int tb0, int tb1, int8_t *xt, float *xs)
{
    const svbool_t pw = svptrue_b32();
    const int nch = cols / 4, nsc = cols / act[0].group;
    for (int tb = tb0; tb < tb1; ++tb) {
        for (int c16 = 0; c16 < nch / 16; ++c16) {   /* 16 tokens x 16 words */
            vs32 v[16];
            for (int i = 0; i < 16; ++i) {
                const int tok = 16 * tb + i;
                v[i] = tok < nt ? svld1_s32(pw, (const int32_t *) act[tok].q + 16 * c16) : svdup_n_s32(0);
            }
            w4g_transpose(v);
            for (int k = 0; k < 16; ++k) svst1_s32(pw, (int32_t *) (xt + ((size_t) tb * nch + 16 * c16 + k) * 64), v[k]);
        }
        for (int k = 0; k < nsc; ++k)
            for (int i = 0; i < 16; ++i) {
                const int tok = 16 * tb + i;
                xs[((size_t) tb * nsc + k) * 16 + i] = tok < nt ? act[tok].scale[k] : 0.f;
            }
    }
}

void gf4_w4a8_gemm_prep(const gf4_q8act *act, int nt, int cols, int8_t *xt, float *xs)
{
    const int nb = gf4_w4a8_gemm_blocks(nt);
    w4g_prep(act, 16 * nb < nt ? 16 * nb : nt, cols, 0, nb, xt, xs);
}

/* tiles [t0, t1) for the nn (<= 32) tokens of blocks xa (and xb when there are more than 16), rows of token n at
   y + n * ldy */
static void w4g_group(const uint8_t *m, int cols, float G, int t0, int t1, const int8_t *xa, const int8_t *xb, const float *sa,
                      const float *sb2, int g16, int nn, float *y, int ldy)
{
    const svbool_t pw = svptrue_b32(), pb = svptrue_b8();
    const int nsb = cols / 64, ntb = (nn + 15) / 16;
    const svint8_t tbl = e2m1_table();
    for (int t = t0; t < t1; ++t) {
        const uint8_t *mt = m + (size_t) t * nsb * GF4_SB_BYTES;
        for (int i = 0; i < 16 * W4G_TB; ++i) svst1_f32(pw, w4g_acc + 16 * i, svdup_n_f32(0));
        for (int sb0 = 0; sb0 < nsb; sb0 += W4G_SLAB) {
            const int ns = nsb - sb0 < W4G_SLAB ? nsb - sb0 : W4G_SLAB;
            for (int sl = 0; sl < ns; ++sl) {   /* decode the slab once */
                const uint8_t *o = mt + (size_t) (sb0 + sl) * GF4_SB_BYTES;
                for (int g = 0; g < 4; ++g) {
                    vs8 w[4];
                    decode_group_q(o + 128 * g, tbl, w);
                    for (int j = 0; j < 4; ++j) svst1_s8(pb, w4g_w + (size_t) (sl * 16 + g * 4 + j) * 64, w[j]);
                    svst1_f32(pw, w4g_s + (size_t) (sl * 4 + g) * 16, scale16(o + 512 + 16 * g));
                }
            }
            for (int rb = 0; rb < 4; ++rb) {
                if (ntb == 2) {
                    if (g16) w4g_rows(rb, sb0, ns, 1, 1, xa, xb, sa, sb2); else w4g_rows(rb, sb0, ns, 1, 0, xa, xb, sa, sb2);
                } else {
                    if (g16) w4g_rows(rb, sb0, ns, 0, 1, xa, xb, sa, sb2); else w4g_rows(rb, sb0, ns, 0, 0, xa, xb, sa, sb2);
                }
            }
        }
        for (int tb = 0; tb < ntb; ++tb) {   /* back to a row vector per token (scalar stores at ldy = 4096 floats all
                                               fell in one L1 set: the down slices ran at 3.7 MAC/cycle) */
            vs32 v[16];
            for (int r = 0; r < 16; ++r) v[r] = svreinterpret_s32_f32(svmul_n_f32_x(pw, svld1_f32(pw, w4g_acc + (r * W4G_TB + tb) * 16), G));
            w4g_transpose(v);
            for (int i = 0; i < 16 && 16 * tb + i < nn; ++i) svst1_f32(pw, y + (size_t) (16 * tb + i) * ldy + 16 * t, svreinterpret_f32_s32(v[i]));
        }
    }
}

/* the tokens past the blocks (a remainder of 1-8) through gf4_w4a8; returns the tokens in blocks */
static int w4g_rest(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{
    const int nm = nt % 16 && nt % 16 <= 8 ? nt - nt % 16 : nt;
    for (int g = nm; g < nt; g += 4) gf4_w4a8(m, cols, global, t0, t1, act + g, nt - g < 4 ? nt - g : 4, y + (size_t) g * ldy, ldy);
    return nm;
}

void gf4_w4a8_gemm_t(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, const int8_t *xt,
                     const float *xs, float *y, int ldy)
{
    const int nch = cols / 4, nsc = cols / act[0].group, g16 = act[0].group == 16;
    nt = w4g_rest(m, cols, global, t0, t1, act, nt, y, ldy);
    for (int n0 = 0; n0 < nt; n0 += 16 * W4G_TB) {
        const size_t tb = (size_t) n0 / 16;
        w4g_group(m, cols, global * 128.f, t0, t1, xt + tb * nch * 64, xt + (tb + 1) * nch * 64, xs + tb * nsc * 16, xs + (tb + 1) * nsc * 16, g16,
                  nt - n0 < 16 * W4G_TB ? nt - n0 : 16 * W4G_TB, y + (size_t) n0 * ldy, ldy);
    }
}

void gf4_w4a8_gemm(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{
    const int nch = cols / 4, grp = act[0].group;
    for (int n = 1; n < nt; ++n)
        if (act[n].group != grp) {   /* (one scale layout per call) */
            for (int g = 0; g < nt; g += 4) gf4_w4a8(m, cols, global, t0, t1, act + g, nt - g < 4 ? nt - g : 4, y + (size_t) g * ldy, ldy);
            return;
        }
    const int nsc = cols / grp;
    nt = w4g_rest(m, cols, global, t0, t1, act, nt, y, ldy);
    if (w4g_xcap < (size_t) W4G_TB * nch * 64) free(w4g_x), w4g_xcap = (size_t) W4G_TB * nch * 64, w4g_x = aligned_alloc(256, w4g_xcap);
    if (w4g_scap < (size_t) W4G_TB * nsc * 16) free(w4g_xs), w4g_scap = (size_t) W4G_TB * nsc * 16, w4g_xs = aligned_alloc(256, (w4g_scap * 4 + 255) & ~(size_t) 255);
    for (int n0 = 0; n0 < nt; n0 += 16 * W4G_TB) {   /* 32 tokens at a time through the thread's buffers */
        const int nn = nt - n0 < 16 * W4G_TB ? nt - n0 : 16 * W4G_TB;
        w4g_prep(act + n0, nn, cols, 0, (nn + 15) / 16, w4g_x, w4g_xs);
        w4g_group(m, cols, global * 128.f, t0, t1, w4g_x, w4g_x + (size_t) nch * 64, w4g_xs, w4g_xs + (size_t) nsc * 16, grp == 16, nn,
                  y + (size_t) n0 * ldy, ldy);
    }
}

/* gf4_tile_many: a slab's weights as F32 columns of 16 rows (column k at 16 k), its block scales, the token sums */
enum { MANY_KS = 256, MANY_MAXT = 2048 };
static _Thread_local float many_w[MANY_KS * 16] __attribute__((aligned(256)));
static _Thread_local float many_sc[MANY_KS / 32 * 16] __attribute__((aligned(256)));
static _Thread_local float many_acc[MANY_MAXT * 16] __attribute__((aligned(256)));

/* one block of `width` columns from decoded weights: e8_block32 (width 32) / bfp8_tile's block (128) / bf16_tile's
   block (32, unscaled), the same chains and FMA order */
INL void many_block(const float *w, const float *const *x, int k0, const int width, const int nt, svfloat32_t scale, const int scaled, vf32 *sum)
{
    const svbool_t pw = svptrue_b32();
    vf32 q[4][4];
    for (int n = 0; n < nt; ++n) for (int c = 0; c < 4; ++c) q[n][c] = svdup_n_f32(0);
    /* chain (n, c) takes columns 8c .. 8c + 7 of each 32 in order, as e8_block32; consecutive FMLAs go to different
       chains (the per-chain order is unchanged), so A64FX's FP queues hold independent work: the per-token kernels'
       order (four dependent FMLAs per chain in a row) ran at 7 of 32 MACs per cycle. x by replicating loads and vector
       FMLAs: the same fused a + w x per lane as the indexed form */
    for (int kb = 0; kb < width; kb += 32) {
        const float *xb[4];   /* this block's activations: replicating loads at small positive offsets */
        for (int n = 0; n < nt; ++n) xb[n] = x[n] + k0 + kb;
        const float *wb = w + 16 * kb;
#pragma GCC unroll 2
        for (int h = 0; h < 2; ++h)
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i)
#pragma GCC unroll 4
                for (int c = 0; c < 4; ++c) {
                    const int k = 8 * c + 4 * h + i;
                    const svfloat32_t wv = svld1_f32(pw, wb + 16 * k);
                    for (int n = 0; n < nt; ++n) q[n][c] = svmla_f32_x(pw, q[n][c], wv, svdup_n_f32(xb[n][k]));
                }
    }
    for (int n = 0; n < nt; ++n) {
        const svfloat32_t t = svadd_f32_x(pw, svadd_f32_x(pw, q[n][0], q[n][1]), svadd_f32_x(pw, q[n][2], q[n][3]));
        sum[n] = scaled ? svmla_f32_x(pw, sum[n], t, scale) : svadd_f32_x(pw, sum[n], t);
    }
}

/* tokens x[0..nt) (nt <= 4) over the decoded slab [s0, s1): sums loaded from and stored to acc */
INL void many_group(const int kind, int s0, int s1, const float *const *x, const int nt, float *acc)
{
    const svbool_t pw = svptrue_b32();
    const int bw = kind == 1 ? 128 : 32;
    vf32 sum[4];
    for (int n = 0; n < nt; ++n) sum[n] = svld1_f32(pw, acc + 16 * n);
    for (int b = s0; b < s1; b += bw) {
        const svfloat32_t s = kind == 0 ? svld1_f32(pw, many_sc + 16 * ((b - s0) / 32)) : svdup_n_f32(many_sc[(b - s0) / 128]);
        many_block(many_w + 16 * (b - s0), x, b, bw, nt, s, kind != 2, sum);
    }
    for (int n = 0; n < nt; ++n) svst1_f32(pw, acc + 16 * n, sum[n]);
}

void gf4_tile_many(int kind, const uint8_t *m, int cols, const float *const *x, int nt, float *y, int ldy)
{
    const svbool_t pw = svptrue_b32();
    if (nt > MANY_MAXT) abort();
    for (int n = 0; n < nt; ++n) svst1_f32(pw, many_acc + 16 * n, svdup_n_f32(0));
    for (int s0 = 0; s0 < cols; s0 += MANY_KS) {
        const int s1 = cols < s0 + MANY_KS ? cols : s0 + MANY_KS;
        for (int k = s0; k < s1; ++k) {   /* decode the slab once */
            svfloat32_t v;
            if (kind == 0) v = e4m3x16(m + (size_t) (k / 32) * MX_BLOCK + 16 * (k % 32));
            else if (kind == 1) v = e4m3x16(m + (size_t) (k / 128) * BFP_BLOCK + 16 * (k % 128));
            else v = svreinterpret_f32_u32(svlsl_n_u32_x(pw, svld1uh_u32(pw, (const uint16_t *) (m + (size_t) k * 32)), 16));
            svst1_f32(pw, many_w + 16 * (k - s0), v);
        }
        if (kind == 0)
            for (int b = s0; b < s1; b += 32)
                svst1_f32(pw, many_sc + 16 * ((b - s0) / 32), svreinterpret_f32_u32(svlsl_n_u32_x(pw, svld1ub_u32(pw, m + (size_t) (b / 32) * MX_BLOCK + 512), 23)));
        else if (kind == 1)
            for (int b = s0; b < s1; b += 128) memcpy(&many_sc[(b - s0) / 128], m + (size_t) (b / 128) * BFP_BLOCK + 2048, 4);
        for (int g = 0; g < nt; g += 4) {
            const int gn = nt - g < 4 ? nt - g : 4;
            for (int n = g + 4; n < g + 8 && n < nt; ++n)   /* the next group's activations of this slab, from L2 */
                for (int k = s0; k < s1; k += 64) __builtin_prefetch(x[n] + k, 0, 3);
#define MANY_CASE(K)                                                                    \
            switch (gn) {                                                               \
            case 1: many_group(K, s0, s1, x + g, 1, many_acc + 16 * g); break;          \
            case 2: many_group(K, s0, s1, x + g, 2, many_acc + 16 * g); break;          \
            case 3: many_group(K, s0, s1, x + g, 3, many_acc + 16 * g); break;          \
            default: many_group(K, s0, s1, x + g, 4, many_acc + 16 * g); break;         \
            }
            if (kind == 0) { MANY_CASE(0) }
            else if (kind == 1) { MANY_CASE(1) }
            else { MANY_CASE(2) }
#undef MANY_CASE
        }
    }
    for (int n = 0; n < nt; ++n) {
        const svfloat32_t v = svld1_f32(pw, many_acc + 16 * n);
        svst1_f32(pw, y + (size_t) n * ldy, kind == 2 ? v : svmul_n_f32_x(pw, v, 256.f));
    }
}

/* gf4_tile_many2: two tiles at once, 2 tiles x 3 tokens per register block (24 chains, 5 loads per 6 FMLAs instead
   of 5 per 4); every (tile, token) chain as many_block's, so each tile's rows are gf4_tile_many's bits */
static _Thread_local float many_w2[MANY_KS * 16] __attribute__((aligned(256)));
static _Thread_local float many_sc2[MANY_KS / 32 * 16] __attribute__((aligned(256)));
static _Thread_local float many_acc2[MANY_MAXT * 16] __attribute__((aligned(256)));

INL void many_block2(const float *w0, const float *w1, const float *const *x, int k0, const int width, const int nt,
                     svfloat32_t s0, svfloat32_t s1, const int scaled, vf32 *sum0, vf32 *sum1)
{
    const svbool_t pw = svptrue_b32();
    vf32 q0[3][4], q1[3][4];
    for (int n = 0; n < nt; ++n) for (int c = 0; c < 4; ++c) q0[n][c] = q1[n][c] = svdup_n_f32(0);
    for (int kb = 0; kb < width; kb += 32) {
        const float *xb[3];
        for (int n = 0; n < nt; ++n) xb[n] = x[n] + k0 + kb;
        const float *wb0 = w0 + 16 * kb, *wb1 = w1 + 16 * kb;
#pragma GCC unroll 2
        for (int h = 0; h < 2; ++h)
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i)
#pragma GCC unroll 4
                for (int c = 0; c < 4; ++c) {
                    const int k = 8 * c + 4 * h + i;
                    const svfloat32_t wv0 = svld1_f32(pw, wb0 + 16 * k), wv1 = svld1_f32(pw, wb1 + 16 * k);
                    for (int n = 0; n < nt; ++n) {
                        const svfloat32_t xv = svdup_n_f32(xb[n][k]);
                        q0[n][c] = svmla_f32_x(pw, q0[n][c], wv0, xv);
                        q1[n][c] = svmla_f32_x(pw, q1[n][c], wv1, xv);
                    }
                }
    }
    for (int n = 0; n < nt; ++n) {
        const svfloat32_t t0 = svadd_f32_x(pw, svadd_f32_x(pw, q0[n][0], q0[n][1]), svadd_f32_x(pw, q0[n][2], q0[n][3]));
        const svfloat32_t t1 = svadd_f32_x(pw, svadd_f32_x(pw, q1[n][0], q1[n][1]), svadd_f32_x(pw, q1[n][2], q1[n][3]));
        sum0[n] = scaled ? svmla_f32_x(pw, sum0[n], t0, s0) : svadd_f32_x(pw, sum0[n], t0);
        sum1[n] = scaled ? svmla_f32_x(pw, sum1[n], t1, s1) : svadd_f32_x(pw, sum1[n], t1);
    }
}

INL void many_group2(const int kind, int s0, int s1, const float *const *x, const int nt, float *acc0, float *acc1)
{
    const svbool_t pw = svptrue_b32();
    const int bw = kind == 1 ? 128 : 32;
    vf32 sum0[3], sum1[3];
    for (int n = 0; n < nt; ++n) sum0[n] = svld1_f32(pw, acc0 + 16 * n), sum1[n] = svld1_f32(pw, acc1 + 16 * n);
    for (int b = s0; b < s1; b += bw) {
        const int j = kind == 0 ? (b - s0) / 32 : (b - s0) / 128;
        const svfloat32_t a = kind == 0 ? svld1_f32(pw, many_sc + 16 * j) : svdup_n_f32(many_sc[j]);
        const svfloat32_t c = kind == 0 ? svld1_f32(pw, many_sc2 + 16 * j) : svdup_n_f32(many_sc2[j]);
        many_block2(many_w + 16 * (b - s0), many_w2 + 16 * (b - s0), x, b, bw, nt, a, c, kind != 2, sum0, sum1);
    }
    for (int n = 0; n < nt; ++n) svst1_f32(pw, acc0 + 16 * n, sum0[n]), svst1_f32(pw, acc1 + 16 * n, sum1[n]);
}

INL void many_decode(int kind, const uint8_t *m, int s0, int s1, float *w, float *sc)
{
    const svbool_t pw = svptrue_b32();
    for (int k = s0; k < s1; ++k) {
        svfloat32_t v;
        if (kind == 0) v = e4m3x16(m + (size_t) (k / 32) * MX_BLOCK + 16 * (k % 32));
        else if (kind == 1) v = e4m3x16(m + (size_t) (k / 128) * BFP_BLOCK + 16 * (k % 128));
        else v = svreinterpret_f32_u32(svlsl_n_u32_x(pw, svld1uh_u32(pw, (const uint16_t *) (m + (size_t) k * 32)), 16));
        svst1_f32(pw, w + 16 * (k - s0), v);
    }
    if (kind == 0)
        for (int b = s0; b < s1; b += 32)
            svst1_f32(pw, sc + 16 * ((b - s0) / 32), svreinterpret_f32_u32(svlsl_n_u32_x(pw, svld1ub_u32(pw, m + (size_t) (b / 32) * MX_BLOCK + 512), 23)));
    else if (kind == 1)
        for (int b = s0; b < s1; b += 128) memcpy(&sc[(b - s0) / 128], m + (size_t) (b / 128) * BFP_BLOCK + 2048, 4);
}

void gf4_tile_many2(int kind, const uint8_t *m0, const uint8_t *m1, int cols, const float *const *x, int nt, float *y0, float *y1, int ldy)
{
    const svbool_t pw = svptrue_b32();
    if (nt > MANY_MAXT) abort();
    for (int n = 0; n < nt; ++n) svst1_f32(pw, many_acc + 16 * n, svdup_n_f32(0)), svst1_f32(pw, many_acc2 + 16 * n, svdup_n_f32(0));
    for (int s0 = 0; s0 < cols; s0 += MANY_KS) {
        const int s1 = cols < s0 + MANY_KS ? cols : s0 + MANY_KS;
        many_decode(kind, m0, s0, s1, many_w, many_sc);
        many_decode(kind, m1, s0, s1, many_w2, many_sc2);
        for (int g = 0; g < nt; g += 3) {
            const int gn = nt - g < 3 ? nt - g : 3;
            for (int n = g + 3; n < g + 6 && n < nt; ++n)
                for (int k = s0; k < s1; k += 64) __builtin_prefetch(x[n] + k, 0, 3);
#define MANY2_CASE(K)                                                                                       \
            switch (gn) {                                                                                   \
            case 1: many_group2(K, s0, s1, x + g, 1, many_acc + 16 * g, many_acc2 + 16 * g); break;         \
            case 2: many_group2(K, s0, s1, x + g, 2, many_acc + 16 * g, many_acc2 + 16 * g); break;         \
            default: many_group2(K, s0, s1, x + g, 3, many_acc + 16 * g, many_acc2 + 16 * g); break;        \
            }
            if (kind == 0) { MANY2_CASE(0) }
            else if (kind == 1) { MANY2_CASE(1) }
            else { MANY2_CASE(2) }
#undef MANY2_CASE
        }
    }
    for (int n = 0; n < nt; ++n) {
        const svfloat32_t v0 = svld1_f32(pw, many_acc + 16 * n), v1 = svld1_f32(pw, many_acc2 + 16 * n);
        svst1_f32(pw, y0 + (size_t) n * ldy, kind == 2 ? v0 : svmul_n_f32_x(pw, v0, 256.f));
        svst1_f32(pw, y1 + (size_t) n * ldy, kind == 2 ? v1 : svmul_n_f32_x(pw, v1, 256.f));
    }
}

/* exp(x) = 2^(k/64) (1 + p(r)), r = x - (k/64) ln2: z = shift + x / ln2 carries k / 64 in its low mantissa bits
   (ulp 2^-6, plus the bias 127 in bits [13:6]), so FEXPA on z's bits gives 2^(k/64) exactly */
INL svfloat32_t vexpf(svbool_t pg, svfloat32_t x)
{
    x = svmax_n_f32_x(pg, svmin_n_f32_x(pg, x, 88.0f), -87.3f);
    const svfloat32_t shift = svdup_n_f32(0x1.803f8p17f);
    const svfloat32_t z = svmla_n_f32_x(pg, shift, x, 0x1.715476p+0f);
    const svfloat32_t k = svsub_f32_x(pg, z, shift);
    svfloat32_t r = svmls_n_f32_x(pg, x, k, 0x1.62e4p-1f);
    r = svmls_n_f32_x(pg, r, k, 0x1.7f7d1cp-20f);
    const svfloat32_t s = svexpa_f32(svreinterpret_u32_f32(z));
    /* p(r) = r + r^2 / 2 + r^3 / 6 */
    const svfloat32_t p = svmul_f32_x(pg, r, svmla_f32_x(pg, svdup_n_f32(1.f), r, svmla_n_f32_x(pg, svdup_n_f32(0.5f), r, 0x1.555556p-3f)));
    return svmla_f32_x(pg, s, s, p);
}
INL svfloat32_t vrecipf(svbool_t pg, svfloat32_t d)
{
    svfloat32_t r = svrecpe_f32(d);
    r = svmul_f32_x(pg, r, svrecps_f32(d, r));
    r = svmul_f32_x(pg, r, svrecps_f32(d, r));
    return r;
}
INL svfloat32_t vsigmoidf(svbool_t pg, svfloat32_t x)
{
    return vrecipf(pg, svadd_n_f32_x(pg, vexpf(pg, svneg_f32_x(pg, x)), 1.f));
}
void gf4_vexp(float *y, const float *x, int n)
{
    for (int i = 0; i < n; i += 16) { const svbool_t pg = svwhilelt_b32(i, n); svst1_f32(pg, y + i, vexpf(pg, svld1_f32(pg, x + i))); }
}
void gf4_vsigmoid(float *y, const float *x, int n)
{
    for (int i = 0; i < n; i += 16) { const svbool_t pg = svwhilelt_b32(i, n); svst1_f32(pg, y + i, vsigmoidf(pg, svld1_f32(pg, x + i))); }
}
void gf4_vsilu(float *y, const float *x, int n)
{
    for (int i = 0; i < n; i += 16) {
        const svbool_t pg = svwhilelt_b32(i, n);
        const svfloat32_t v = svld1_f32(pg, x + i);
        svst1_f32(pg, y + i, svmul_f32_x(pg, v, vsigmoidf(pg, v)));
    }
}
void gf4_vswiglu(float *y, const float *g, const float *u, int n, float limit)
{
    for (int i = 0; i < n; i += 16) {
        const svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t gv = svld1_f32(pg, g + i), uv = svld1_f32(pg, u + i);
        if (limit > 0) gv = svmin_n_f32_x(pg, gv, limit), uv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, uv, limit), -limit);
        svst1_f32(pg, y + i, svmul_f32_x(pg, svmul_f32_x(pg, gv, vsigmoidf(pg, gv)), uv));
    }
}
/* double: z = shift + x / ln2 with ulp 2^-6 (shift 1.5 * 2^46 + 1023), FEXPA .D on bits [16:6] */
void gf4_vexp_f64(double *y, const double *x, long n)
{
    for (long i = 0; i < n; i += 8) {
        const svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t v = svld1_f64(pg, x + i);
        v = svmax_n_f64_x(pg, svmin_n_f64_x(pg, v, 709.0), -708.0);
        const svfloat64_t shift = svdup_n_f64(0x1.800000000ffc0p46);
        const svfloat64_t z = svmla_n_f64_x(pg, shift, v, 0x1.71547652b82fep0);
        const svfloat64_t k = svsub_f64_x(pg, z, shift);
        svfloat64_t r = svmls_n_f64_x(pg, v, k, 0x1.62e42fefa3800p-1);
        r = svmls_n_f64_x(pg, r, k, 0x1.ef35793c76730p-45);
        const svfloat64_t s = svexpa_f64(svreinterpret_u64_f64(z));
        /* p(r) = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 (|r| <= ln2/128: the next term is below 2^-60) */
        svfloat64_t p = svmla_n_f64_x(pg, svdup_n_f64(1.0 / 24), r, 1.0 / 120);
        p = svmla_f64_x(pg, svdup_n_f64(1.0 / 6), r, p);
        p = svmla_f64_x(pg, svdup_n_f64(0.5), r, p);
        p = svmla_f64_x(pg, svdup_n_f64(1.0), r, p);
        p = svmul_f64_x(pg, r, p);
        svst1_f64(pg, y + i, svmla_f64_x(pg, s, s, p));
    }
}

#else   /* no 512-bit SVE build: the replicas */

int gf4_have_sve512(void) { return 0; }
int gf4_selftest_e4m3(void) { return 0; }
void gf4_w4a8(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{ gf4_w4a8_ref(m, cols, global, t0, t1, act, nt, y, ldy); }
void gf4_w4a16q(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{ gf4_w4a16_ref(m, cols, global, t0, t1, x, nt, y, ldy, 0); }
void gf4_w4a16p(const uint8_t *m, int cols, float global, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{ gf4_w4a16_ref(m, cols, global, t0, t1, x, nt, y, ldy, 1); }
void gf4_mxfp8(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{ gf4_mxfp8_ref(m, cols, t0, t1, x, nt, y, ldy); }
void gf4_bfp8(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{ gf4_bfp8_ref(m, cols, t0, t1, x, nt, y, ldy); }
void gf4_q8(const uint8_t *m, int cols, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{ gf4_q8_ref(m, cols, t0, t1, act, nt, y, ldy); }
void gf4_bf16k(const uint8_t *m, int cols, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{ gf4_bf16k_ref(m, cols, t0, t1, x, nt, y, ldy); }
void gf4_bf16k_part(const uint8_t *m, int cols, int k0, int k1, int t0, int t1, const float *const *x, int nt, float *y, int ldy)
{ gf4_bf16k_part_ref(m, cols, k0, k1, t0, t1, x, nt, y, ldy); }
void gf4_vexp(float *y, const float *x, int n) { for (int i = 0; i < n; ++i) y[i] = expf(x[i]); }
void gf4_vsigmoid(float *y, const float *x, int n) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
void gf4_vsilu(float *y, const float *x, int n) { for (int i = 0; i < n; ++i) { const float v = x[i]; y[i] = v * (1.f / (1.f + expf(-v))); } }
void gf4_vswiglu(float *y, const float *g, const float *u, int n, float limit)
{
    for (int i = 0; i < n; ++i) {
        float gv = g[i], uv = u[i];
        if (limit > 0) gv = gv < limit ? gv : limit, uv = uv < -limit ? -limit : uv > limit ? limit : uv;
        y[i] = gv * (1.f / (1.f + expf(-gv))) * uv;
    }
}
void gf4_vexp_f64(double *y, const double *x, long n) { for (long i = 0; i < n; ++i) y[i] = exp(x[i]); }
void gf4_w4a8_gemm(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, float *y, int ldy)
{
    for (int g = 0; g < nt; g += 4) gf4_w4a8_ref(m, cols, global, t0, t1, act + g, nt - g < 4 ? nt - g : 4, y + (size_t) g * ldy, ldy);
}
int gf4_w4a8_gemm_blocks(int nt) { return ((nt % 16 && nt % 16 <= 8 ? nt - nt % 16 : nt) + 15) / 16; }
void gf4_w4a8_gemm_prep(const gf4_q8act *act, int nt, int cols, int8_t *xt, float *xs)
{
    const int nb = gf4_w4a8_gemm_blocks(nt), nch = cols / 4, nsc = cols / act[0].group;
    for (int tb = 0; tb < nb; ++tb)
        for (int i = 0; i < 16; ++i) {
            const int tok = 16 * tb + i;
            for (int c = 0; c < nch; ++c) memcpy(xt + ((size_t) tb * nch + c) * 64 + 4 * i, tok < nt ? act[tok].q + 4 * c : (const int8_t *) "\0\0\0", 4);
            for (int k = 0; k < nsc; ++k) xs[((size_t) tb * nsc + k) * 16 + i] = tok < nt ? act[tok].scale[k] : 0.f;
        }
}
void gf4_w4a8_gemm_t(const uint8_t *m, int cols, float global, int t0, int t1, const gf4_q8act *act, int nt, const int8_t *xt,
                     const float *xs, float *y, int ldy)
{
    (void) xt, (void) xs;
    gf4_w4a8_gemm(m, cols, global, t0, t1, act, nt, y, ldy);
}
void gf4_tile_many(int kind, const uint8_t *m, int cols, const float *const *x, int nt, float *y, int ldy)
{
    for (int g = 0; g < nt; g += 4) {
        const int gn = nt - g < 4 ? nt - g : 4;
        if (kind == 0) gf4_mxfp8_ref(m, cols, 0, 1, x + g, gn, y + (size_t) g * ldy, ldy);
        else if (kind == 1) gf4_bfp8_ref(m, cols, 0, 1, x + g, gn, y + (size_t) g * ldy, ldy);
        else gf4_bf16k_ref(m, cols, 0, 1, x + g, gn, y + (size_t) g * ldy, ldy);
    }
}
void gf4_tile_many2(int kind, const uint8_t *m0, const uint8_t *m1, int cols, const float *const *x, int nt, float *y0, float *y1, int ldy)
{
    gf4_tile_many(kind, m0, cols, x, nt, y0, ldy);
    gf4_tile_many(kind, m1, cols, x, nt, y1, ldy);
}

#endif
