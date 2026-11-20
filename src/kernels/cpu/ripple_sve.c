#include "strata/kernels/cpu/ripple_sve.h"

#include <math.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
#include <arm_sve.h>
#define RIPPLE_SVE 1
#endif

static float g_e4m3[256];
static int g_inited = 0;

float ripple_e4m3_to_f32(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return s ? -v : v;
}

float ripple_e2m1_to_f32(unsigned n) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[n & 7];
    return (n & 8) ? -v : v;
}

void ripple_nvfp4_init(void) {
    if (g_inited) return;
    for (int i = 0; i < 256; ++i) g_e4m3[i] = ripple_e4m3_to_f32((uint8_t)i);
    g_inited = 1;
}

void ripple_nvfp4_rows_ref(const uint8_t *q, const uint8_t *sc, float s2, int cols, int r0, int r1, const float *x, double *y) {
    const size_t qb = (size_t)cols / 2, sb = (size_t)cols / 16;
    for (int r = r0; r < r1; ++r) {
        const uint8_t *qr = q + (size_t)r * qb, *sr = sc + (size_t)r * sb;
        double acc = 0.0;
        for (int b = 0; b < cols / 16; ++b) {
            double s = 0.0;
            for (int j = 0; j < 16; ++j) {
                const unsigned byte = qr[b * 8 + j / 2], nib = (j & 1) ? byte >> 4 : byte & 15;
                s += (double)ripple_e2m1_to_f32(nib) * (double)x[b * 16 + j];
            }
            acc += s * (double)ripple_e4m3_to_f32(sr[b]);
        }
        y[r - r0] = acc * (double)s2;
    }
}

/* one row decoded to floats (scale and s2 not applied to the sum; applied here per block) */
static void decode_row(const uint8_t *qr, const uint8_t *sr, float s2, int cols, float *w) {
#if RIPPLE_SVE
    const svbool_t all = svptrue_b32(), p8 = svwhilelt_b32(0, 8);
    float lut[16];
    for (int i = 0; i < 16; ++i) lut[i] = ripple_e2m1_to_f32((unsigned)i);
    const svfloat32_t vlut = svld1_f32(all, lut);
    const svuint32_t lane = svindex_u32(0, 1);
    const svuint32_t bidx = svlsr_n_u32_x(all, lane, 1);
    const svuint32_t shift = svlsl_n_u32_x(all, svand_n_u32_x(all, lane, 1), 2);
    for (int b = 0; b < cols / 16; ++b) {
        const svuint32_t by = svld1ub_u32(p8, qr + b * 8);
        const svuint32_t n = svand_n_u32_x(all, svlsr_u32_x(all, svtbl_u32(by, bidx), shift), 15u);
        svst1_f32(all, w + b * 16, svmul_n_f32_x(all, svtbl_f32(vlut, n), g_e4m3[sr[b]] * s2));
    }
#else
    for (int b = 0; b < cols / 16; ++b)
        for (int j = 0; j < 16; ++j) {
            const unsigned byte = qr[b * 8 + j / 2], nib = (j & 1) ? byte >> 4 : byte & 15;
            w[b * 16 + j] = ripple_e2m1_to_f32(nib) * g_e4m3[sr[b]] * s2;
        }
#endif
}

static float dot_f32(const float *a, const float *b, int n) {
#if RIPPLE_SVE
    const svbool_t all = svptrue_b32();
    svfloat32_t s0 = svdup_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    for (int i = 0; i + 64 <= n; i += 64) {
        s0 = svmla_f32_x(all, s0, svld1_f32(all, a + i), svld1_f32(all, b + i));
        s1 = svmla_f32_x(all, s1, svld1_f32(all, a + i + 16), svld1_f32(all, b + i + 16));
        s2 = svmla_f32_x(all, s2, svld1_f32(all, a + i + 32), svld1_f32(all, b + i + 32));
        s3 = svmla_f32_x(all, s3, svld1_f32(all, a + i + 48), svld1_f32(all, b + i + 48));
    }
    float r = svaddv_f32(all, svadd_f32_x(all, svadd_f32_x(all, s0, s1), svadd_f32_x(all, s2, s3)));
    for (int i = n & ~63; i < n; ++i) r += a[i] * b[i];
    return r;
#else
    float r = 0.0f;
    for (int i = 0; i < n; ++i) r += a[i] * b[i];
    return r;
#endif
}

#if RIPPLE_SVE
static void rows_one(const uint8_t *q, const uint8_t *sc, float s2, int cols, int r0, int r1, const float *x, float *y, int ldy) {
    (void)ldy;
    const size_t qb = (size_t)cols / 2, sb = (size_t)cols / 16;
    const svbool_t all = svptrue_b32(), p8 = svwhilelt_b32(0, 8);
    float lut[16];
    for (int i = 0; i < 16; ++i) lut[i] = ripple_e2m1_to_f32((unsigned)i);
    const svfloat32_t vlut = svld1_f32(all, lut);
    const svuint32_t lane = svindex_u32(0, 1);
    const svuint32_t bidx = svlsr_n_u32_x(all, lane, 1);                           /* source byte of lane i: i / 2 */
    const svuint32_t shift = svlsl_n_u32_x(all, svand_n_u32_x(all, lane, 1), 2);   /* 0 (low nibble) or 4 (high) */
    for (int r = r0; r < r1; ++r) {
        const uint8_t *qr = q + (size_t)r * qb, *sr = sc + (size_t)r * sb;
        svfloat32_t a0 = svdup_n_f32(0.0f), a1 = a0;
        for (int b = 0; b + 2 <= cols / 16; b += 2) {
            const svuint32_t by0 = svld1ub_u32(p8, qr + b * 8), by1 = svld1ub_u32(p8, qr + b * 8 + 8);
            const svuint32_t n0 = svand_n_u32_x(all, svlsr_u32_x(all, svtbl_u32(by0, bidx), shift), 15u);
            const svuint32_t n1 = svand_n_u32_x(all, svlsr_u32_x(all, svtbl_u32(by1, bidx), shift), 15u);
            const svfloat32_t w0 = svmul_n_f32_x(all, svtbl_f32(vlut, n0), g_e4m3[sr[b]]);
            const svfloat32_t w1 = svmul_n_f32_x(all, svtbl_f32(vlut, n1), g_e4m3[sr[b + 1]]);
            a0 = svmla_f32_x(all, a0, w0, svld1_f32(all, x + b * 16));
            a1 = svmla_f32_x(all, a1, w1, svld1_f32(all, x + b * 16 + 16));
        }
        y[r] = svaddv_f32(all, svadd_f32_x(all, a0, a1)) * s2;
    }
}
#endif

void ripple_nvfp4_rows(const uint8_t *q, const uint8_t *sc, float s2, int cols, int r0, int r1, const float *const *xs, int nt, float *y, int ldy) {
    ripple_nvfp4_init();
#if RIPPLE_SVE
    if (nt == 1) { rows_one(q, sc, s2, cols, r0, r1, xs[0], y, ldy); return; }
#endif
    const size_t qb = (size_t)cols / 2, sb = (size_t)cols / 16;
    float w[8192] __attribute__((aligned(256)));
    for (int r = r0; r < r1; ++r) {
        decode_row(q + (size_t)r * qb, sc + (size_t)r * sb, s2, cols, w);
        for (int t = 0; t < nt; ++t) y[(size_t)t * ldy + r] = dot_f32(w, xs[t], cols);
    }
}

/* ================================================================ int8 dot-product path */
static const int8_t g_lut8[64] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};   /* 2 * e2m1; lanes 16.. are 0 */

int ripple_q8_units(int cols) { return 2 * ((cols + 127) / 128); }

static void unit_range(int cols, int u, int *start, int *len) {
    const int full = cols / 128, g = u >> 1, rem = cols - full * 128;
    if (g < full) { *start = g * 128 + (u & 1) * 64; *len = 64; }
    else { const int half = rem / 2; *start = full * 128 + (u & 1) * half; *len = half; }
}

void ripple_q8_quant_unit(const float *x, int cols, int u, int8_t *xq, float *xd) {
    int st, n;
    unit_range(cols, u, &st, &n);
    x += st; xq += st;
#if RIPPLE_SVE
    const svbool_t all = svptrue_b32();
    svfloat32_t mx = svdup_n_f32(0.0f);
    for (int i = 0; i < n; i += 16) {
        const svbool_t pg = svwhilelt_b32(i, n);
        mx = svmax_f32_m(pg, mx, svabs_f32_x(pg, svld1_f32(pg, x + i)));
    }
    const float amax = svmaxv_f32(all, mx), d = amax / 127.0f, inv = d > 0.0f ? 1.0f / d : 0.0f;
    for (int i = 0; i < n; i += 16) {
        const svbool_t pg = svwhilelt_b32(i, n);
        const svfloat32_t v = svrintn_f32_x(pg, svmul_n_f32_x(pg, svld1_f32(pg, x + i), inv));
        svst1b_s32(pg, xq + i, svcvt_s32_f32_x(pg, v));
    }
    *xd = d;
#else
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) { const float a = fabsf(x[i]); if (a > amax) amax = a; }
    const float d = amax / 127.0f, inv = d > 0.0f ? 1.0f / d : 0.0f;
    for (int i = 0; i < n; ++i) xq[i] = (int8_t)nearbyintf(x[i] * inv);
    *xd = d;
#endif
}

void ripple_q8_quant(const float *x, int cols, int8_t *xq, float *xd) {
    for (int u = 0; u < ripple_q8_units(cols); ++u) ripple_q8_quant_unit(x, cols, u, xq, xd + u);
}

void ripple_fp4_repack_row(const uint8_t *q, const uint8_t *sc, int cols, uint8_t *qo, uint16_t *sfo) {
    ripple_nvfp4_init();
#define NIB(e) ((q[(e) >> 1] >> (((e) & 1) * 4)) & 15u)
    const int full = cols / 128, rem = cols - full * 128;
    for (int g = 0; g < full; ++g)
        for (int j = 0; j < 64; ++j) qo[g * 64 + j] = (uint8_t)(NIB(g * 128 + j) | (NIB(g * 128 + 64 + j) << 4));
    if (rem) {
        const int half = rem / 2;
        for (int j = 0; j < half; ++j) qo[full * 64 + j] = (uint8_t)(NIB(full * 128 + j) | (NIB(full * 128 + half + j) << 4));
    }
#undef NIB
    for (int b = 0; b < cols / 16; ++b) {
        const _Float16 h = (_Float16)g_e4m3[sc[b]];
        uint16_t u; memcpy(&u, &h, 2);
        sfo[b] = u;
    }
}

static inline float f16_to_f32_s(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

void ripple_fp4_rows_q8_ref(const uint8_t *q, const uint16_t *sf, int cols, int r0, int r1, const int8_t *const *xq,
                            const float *const *xd, int nt, float s2, float *y, int ldy) {
    const size_t qb = (size_t)cols / 2, sb = (size_t)cols / 16;
    const int units = ripple_q8_units(cols);
    for (int r = r0; r < r1; ++r) {
        const uint8_t *qr = q + (size_t)r * qb;
        const uint16_t *sr = sf + (size_t)r * sb;
        for (int t = 0; t < nt; ++t) {
            float acc = 0.0f;
            for (int u = 0; u < units; ++u) {
                int st, n;
                unit_range(cols, u, &st, &n);
                const int g = u >> 1, hi = u & 1, full = cols / 128, half = g < full ? 64 : (cols - full * 128) / 2;
                float sum = 0.0f;
                for (int b = 0; b < n / 16; ++b) {
                    int isum = 0;
                    for (int e = b * 16; e < b * 16 + 16; ++e) {
                        const unsigned byte = qr[g * 64 + e], nib = hi ? byte >> 4 : byte & 15;
                        isum += (int)g_lut8[nib] * (int)xq[t][st + e];
                    }
                    sum += f16_to_f32_s(sr[(st + b * 16) / 16]) * (float)isum;
                }
                acc += sum * xd[t][u];
                (void)half;
            }
            y[(size_t)t * ldy + r] = acc * 0.5f * s2;
        }
    }
}

#if RIPPLE_SVE
static float dot_row_q8(const uint8_t *qr, const uint16_t *sf, const int8_t *xq, const float *xd, int cols) {
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svint8_t lut = svld1_s8(p8, g_lut8);
    const svuint32_t i0 = svlsr_n_u32_x(p32, svindex_u32(0, 1), 2), i1 = svadd_n_u32_x(p32, i0, 4), i2 = svadd_n_u32_x(p32, i0, 8),
                     i3 = svadd_n_u32_x(p32, i0, 12);
    const svint32_t zero = svdup_n_s32(0);
    svfloat32_t acc = svdup_n_f32(0.0f);
    const int full = cols / 128;
    int g = 0;
    for (; g + 2 <= full; g += 2) {
        const svuint8_t b0 = svld1_u8(p8, qr + g * 64), b1 = svld1_u8(p8, qr + g * 64 + 64);
        const int8_t *x = xq + g * 128;
        const svint32_t d0 = svdot_s32(zero, svtbl_s8(lut, svand_n_u8_x(p8, b0, 15)), svld1_s8(p8, x));
        const svint32_t d1 = svdot_s32(zero, svtbl_s8(lut, svlsr_n_u8_x(p8, b0, 4)), svld1_s8(p8, x + 64));
        const svint32_t d2 = svdot_s32(zero, svtbl_s8(lut, svand_n_u8_x(p8, b1, 15)), svld1_s8(p8, x + 128));
        const svint32_t d3 = svdot_s32(zero, svtbl_s8(lut, svlsr_n_u8_x(p8, b1, 4)), svld1_s8(p8, x + 192));
        const svfloat32_t s16 = svcvt_f32_f16_x(p32, svreinterpret_f16_u32(svld1uh_u32(p32, sf + g * 8)));
        acc = svmla_n_f32_x(p32, acc, svmul_f32_x(p32, svcvt_f32_s32_x(p32, d0), svtbl_f32(s16, i0)), xd[2 * g]);
        acc = svmla_n_f32_x(p32, acc, svmul_f32_x(p32, svcvt_f32_s32_x(p32, d1), svtbl_f32(s16, i1)), xd[2 * g + 1]);
        acc = svmla_n_f32_x(p32, acc, svmul_f32_x(p32, svcvt_f32_s32_x(p32, d2), svtbl_f32(s16, i2)), xd[2 * g + 2]);
        acc = svmla_n_f32_x(p32, acc, svmul_f32_x(p32, svcvt_f32_s32_x(p32, d3), svtbl_f32(s16, i3)), xd[2 * g + 3]);
    }
    if (g < full) {
        const svuint8_t b0 = svld1_u8(p8, qr + g * 64);
        const int8_t *x = xq + g * 128;
        const svint32_t d0 = svdot_s32(zero, svtbl_s8(lut, svand_n_u8_x(p8, b0, 15)), svld1_s8(p8, x));
        const svint32_t d1 = svdot_s32(zero, svtbl_s8(lut, svlsr_n_u8_x(p8, b0, 4)), svld1_s8(p8, x + 64));
        const svfloat32_t s8 = svcvt_f32_f16_x(p32, svreinterpret_f16_u32(svld1uh_u32(svwhilelt_b32(0, 8), sf + g * 8)));
        acc = svmla_n_f32_x(p32, acc, svmul_f32_x(p32, svcvt_f32_s32_x(p32, d0), svtbl_f32(s8, i0)), xd[2 * g]);
        acc = svmla_n_f32_x(p32, acc, svmul_f32_x(p32, svcvt_f32_s32_x(p32, d1), svtbl_f32(s8, i1)), xd[2 * g + 1]);
        ++g;
    }
    const int rem = cols - full * 128;
    if (rem) {
        const int half = rem / 2;
        const svbool_t pg8 = svwhilelt_b8(0, half), pb = svwhilelt_b32(0, half / 16);
        const svuint8_t b0 = svld1_u8(pg8, qr + full * 64);
        const int8_t *x = xq + full * 128;
        const svint32_t dl = svdot_s32(zero, svtbl_s8(lut, svand_n_u8_x(pg8, b0, 15)), svld1_s8(pg8, x));
        const svint32_t dh = svdot_s32(zero, svtbl_s8(lut, svlsr_n_u8_x(pg8, b0, 4)), svld1_s8(pg8, x + half));
        const svfloat32_t sl = svcvt_f32_f16_x(p32, svreinterpret_f16_u32(svld1uh_u32(pb, sf + full * 8)));
        const svfloat32_t sh = svcvt_f32_f16_x(p32, svreinterpret_f16_u32(svld1uh_u32(pb, sf + full * 8 + half / 16)));
        acc = svmla_n_f32_x(p32, acc, svmul_f32_x(p32, svcvt_f32_s32_x(p32, dl), svtbl_f32(sl, i0)), xd[2 * full]);
        acc = svmla_n_f32_x(p32, acc, svmul_f32_x(p32, svcvt_f32_s32_x(p32, dh), svtbl_f32(sh, i0)), xd[2 * full + 1]);
    }
    return svaddv_f32(p32, acc);
}

void ripple_fp4_rows_q8(const uint8_t *q, const uint16_t *sf, int cols, int r0, int r1, const int8_t *const *xq,
                        const float *const *xd, int nt, float s2, float *y, int ldy) {
    const size_t qb = (size_t)cols / 2, sb = (size_t)cols / 16;
    const float k = 0.5f * s2;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) y[(size_t)t * ldy + r] = dot_row_q8(q + (size_t)r * qb, sf + (size_t)r * sb, xq[t], xd[t], cols) * k;
}
#else
void ripple_fp4_rows_q8(const uint8_t *q, const uint16_t *sf, int cols, int r0, int r1, const int8_t *const *xq,
                        const float *const *xd, int nt, float s2, float *y, int ldy) {
    ripple_fp4_rows_q8_ref(q, sf, cols, r0, r1, xq, xd, nt, s2, y, ldy);
}
#endif
