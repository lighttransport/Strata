/* DS4.1-Flash GGUF kernels for A64FX (SVE 512): see include/strata/kernels/cpu/ds41_sve.h.
 *
 * IQ types: a 256-weight block is four 64-weight planes W0..W3 (natural order), each built from eight 8-weight grid
 * entries fetched with one 64-bit gather (IQ3_XXS: sixteen 4-weight entries, 32-bit gather) and signed with the
 * gathered byte masks (IQ2_XXS / IQ2_XS: ksigns64) or by a predicated negate from each lane's 8 sign bits (IQ3_XXS:
 * 7 + parity; IQ2_S: the explicit sign byte) spread to bytes by a tbl and a bit mask. One SDOT per
 * plane per token against the Q8_K activation; the per-16 / per-32 scales 2s+1 enter as int32 lane multipliers.
 * Q5_K: planes from the low/high nibbles plus the qh bit, 6-bit scales as lane multipliers, mins through bsums.
 * Grids and sign tables from the vendored ggml-common.h. */
#include "strata/kernels/cpu/ds41_sve.h"

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "../../../third_party/ggml/ggml-common.h"

typedef struct { uint16_t d; uint8_t qs[64]; } b_iq2xxs;
typedef struct { uint16_t d; uint16_t qs[32]; uint8_t scales[8]; } b_iq2xs;
typedef struct { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t scales[8]; } b_iq2s;
typedef struct { uint16_t d; uint8_t qs[96]; } b_iq3xxs;
/* IQ2R: weight 64 v + k (plane v, k < 64): magnitude code in codes[16 v + k % 16] bits 2 (k / 16), sign in
   signs[8 v + k % 8] bit k / 8; scales[j]: the 16-weight groups 2 j (low nibble) and 2 j + 1 */
typedef struct { uint16_t d; uint8_t scales[8]; uint8_t codes[64]; uint8_t signs[32]; } b_iq2r;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } b_q5k;
typedef struct { uint16_t d; int8_t qs[32]; } b_q80;
typedef struct { uint16_t d; uint8_t qs[16]; } b_iq4nl;
typedef struct { uint8_t e; uint8_t qs[16]; } b_mxfp4;
_Static_assert(sizeof(b_iq2xxs) == 66 && sizeof(b_iq2xs) == 74 && sizeof(b_iq2s) == 82 && sizeof(b_iq3xxs) == 98 && sizeof(b_iq2r) == 106, "iq");
_Static_assert(sizeof(b_q5k) == 176 && sizeof(b_q80) == 34 && sizeof(b_iq4nl) == 18 && sizeof(b_mxfp4) == 17 && sizeof(ds41_q8k) == 320, "k");

/* ------------------------------------------------------------------------------------------- common */
static float f16(uint16_t h)
{
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   /* subnormal */
            float f = (float) m * 0x1p-24f;
            memcpy(&u, &f, 4);
            u |= s;
        }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* E8M0 scale / 2 as a float (ggml GGML_E8M0_TO_FP32_HALF: the e2m1 values are stored doubled) */
static float e8m0_half(uint8_t e)
{
    const uint32_t u = e < 2 ? 0x00200000u << e : (uint32_t) (e - 1) << 23;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint16_t to_f16(float f)   /* round to nearest even, as the F16 conversion ggml uses */
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    const int e = (int) ((x >> 23) & 255) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 255) == 255) return (uint16_t) (s | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31) return (uint16_t) (s | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t) s;
        m |= 0x800000;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t) (s | r);
    }
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    return (uint16_t) (s | r);
}

int ds41_type_ok(int t)
{
    return t == DS41_T_F32 || t == DS41_T_Q8_0 || t == DS41_T_Q5_K || t == DS41_T_IQ2_XXS || t == DS41_T_IQ2_XS ||
           t == DS41_T_IQ3_XXS || t == DS41_T_IQ4_NL || t == DS41_T_IQ2_S || t == DS41_T_BF16 || t == DS41_T_MXFP4;
}

int ds41_act_kind(int t)
{
    switch (t) {
    case DS41_T_F32: case DS41_T_BF16: return DS41_ACT_F32;
    case DS41_T_Q8_0: case DS41_T_IQ4_NL: case DS41_T_MXFP4: return DS41_ACT_Q80;
    case DS41_T_Q5_K: case DS41_T_Q5P: case DS41_T_IQ2_XXS: case DS41_T_IQ2_XS: case DS41_T_IQ3_XXS: case DS41_T_IQ2_S: case DS41_T_IQ2R: case DS41_T_IQ2P:
        return DS41_ACT_Q8K;
    }
    return -1;
}

size_t ds41_row_bytes(int t, int cols)
{
    switch (t) {
    case DS41_T_F32: return (size_t) cols * 4;
    case DS41_T_BF16: return (size_t) cols * 2;
    case DS41_T_Q8_0: return (size_t) cols / 32 * 34;
    case DS41_T_IQ4_NL: return (size_t) cols / 32 * 18;
    case DS41_T_MXFP4: return (size_t) cols / 32 * 17;
    case DS41_T_Q5_K: case DS41_T_Q5P: return (size_t) cols / 256 * 176;
    case DS41_T_IQ2_XXS: return (size_t) cols / 256 * 66;
    case DS41_T_IQ2_XS: return (size_t) cols / 256 * 74;
    case DS41_T_IQ2_S: return (size_t) cols / 256 * 82;
    case DS41_T_IQ3_XXS: return (size_t) cols / 256 * 98;
    case DS41_T_IQ2R: case DS41_T_IQ2P: return (size_t) cols / 256 * 106;
    }
    return 0;
}

static inline int nearest_int(float fval)
{
    const float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

void ds41_quant_q8k(const float *x, int n, ds41_q8k *y)
{
    for (int i = 0; i < n / 256; ++i, x += 256) {
        float max = 0, amax = 0;
        for (int j = 0; j < 256; ++j) {
            const float ax = fabsf(x[j]);
            if (ax > amax) amax = ax, max = x[j];
        }
        if (!amax) {
            memset(&y[i], 0, sizeof y[i]);
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < 256; ++j) {
            const int v = nearest_int(iscale * x[j]);
            y[i].q[j] = (int8_t) (v < 127 ? v : 127);
        }
        for (int j = 0; j < 16; ++j) {
            int s = 0;
            for (int k = 0; k < 16; ++k) s += y[i].q[16 * j + k];
            y[i].bsum[j] = (int16_t) s;
        }
        y[i].d = 1 / iscale;
    }
}

void ds41_quant_q80(const float *x, int n, ds41_q80 *y)
{
    for (int i = 0; i < n / 32; ++i, x += 32) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[j]));
        const float d = amax / 127, id = d ? 1.0f / d : 0.0f;
        y[i].d = f16(to_f16(d));
        for (int j = 0; j < 32; ++j) y[i].q[j] = (int8_t) roundf(x[j] * id);
    }
}

/* ------------------------------------------------------------------------------- scalar dequantizers */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) *d = q[j] & 63, *m = q[j + 4] & 63;
    else *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}

void ds41_dequant_row(int t, const void *row, int cols, float *y)
{
    const int nb = cols / 256;
    switch (t) {
    case DS41_T_F32: memcpy(y, row, (size_t) cols * 4); return;
    case DS41_T_BF16:
        for (int k = 0; k < cols; ++k) {
            const uint32_t u = (uint32_t) ((const uint16_t *) row)[k] << 16;
            memcpy(&y[k], &u, 4);
        }
        return;
    case DS41_T_Q8_0:
        for (int i = 0; i < cols / 32; ++i) {
            const b_q80 *b = (const b_q80 *) row + i;
            const float d = f16(b->d);
            for (int j = 0; j < 32; ++j) y[32 * i + j] = d * b->qs[j];
        }
        return;
    case DS41_T_MXFP4:
        for (int i = 0; i < cols / 32; ++i) {
            const b_mxfp4 *b = (const b_mxfp4 *) row + i;
            const float d = e8m0_half(b->e);
            for (int j = 0; j < 16; ++j) y[32 * i + j] = d * kvalues_mxfp4[b->qs[j] & 15], y[32 * i + j + 16] = d * kvalues_mxfp4[b->qs[j] >> 4];
        }
        return;
    case DS41_T_IQ4_NL:
        for (int i = 0; i < cols / 32; ++i) {
            const b_iq4nl *b = (const b_iq4nl *) row + i;
            const float d = f16(b->d);
            for (int j = 0; j < 16; ++j) y[32 * i + j] = d * kvalues_iq4nl[b->qs[j] & 15], y[32 * i + j + 16] = d * kvalues_iq4nl[b->qs[j] >> 4];
        }
        return;
    case DS41_T_Q5P:
        for (int i = 0; i < nb; ++i) {
            const b_q5k *b = (const b_q5k *) row + i;   /* (Q5_K's fields, Q5P's bit order) */
            const float d = f16(b->d), mn = f16(b->dmin);
            for (int k = 0; k < 256; ++k) {
                const int v = k / 64, kk = k % 64, is = k / 32;
                uint8_t sc, m;
                get_scale_min_k4(is, b->scales, &sc, &m);
                const int q = ((b->qs[64 * (v / 2) + kk] >> (4 * (v % 2))) & 15) | ((b->qh[8 * v + kk / 8] >> (kk % 8) & 1) << 4);
                y[256 * i + k] = d * sc * q - mn * m;
            }
        }
        return;
    case DS41_T_Q5_K:
        for (int i = 0; i < nb; ++i) {
            const b_q5k *b = (const b_q5k *) row + i;
            const float d = f16(b->d), mn = f16(b->dmin);
            const uint8_t *ql = b->qs;
            float *o = y + 256 * i;
            uint8_t u1 = 1, u2 = 2, sc, m;
            for (int j = 0, is = 0; j < 256; j += 64, is += 2, ql += 32, u1 <<= 2, u2 <<= 2) {
                get_scale_min_k4(is, b->scales, &sc, &m);
                const float d1 = d * sc, m1 = mn * m;
                get_scale_min_k4(is + 1, b->scales, &sc, &m);
                const float d2 = d * sc, m2 = mn * m;
                for (int l = 0; l < 32; ++l) o[j + l] = d1 * ((ql[l] & 0xF) + (b->qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) o[j + 32 + l] = d2 * ((ql[l] >> 4) + (b->qh[l] & u2 ? 16 : 0)) - m2;
            }
        }
        return;
    case DS41_T_IQ2_XXS:
        for (int i = 0; i < nb; ++i) {
            const b_iq2xxs *b = (const b_iq2xxs *) row + i;
            const float d = f16(b->d);
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t a[2];
                memcpy(a, b->qs + 8 * ib, 8);
                const float db = d * (0.5f + (a[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t *g = (const uint8_t *) (iq2xxs_grid + ((const uint8_t *) a)[l]);
                    const uint8_t s = ksigns_iq2xs[(a[1] >> 7 * l) & 127];
                    for (int j = 0; j < 8; ++j) y[256 * i + 32 * ib + 8 * l + j] = db * g[j] * (s & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
        }
        return;
    case DS41_T_IQ2_XS:
        for (int i = 0; i < nb; ++i) {
            const b_iq2xs *b = (const b_iq2xs *) row + i;
            const float d = f16(b->d);
            for (int ib = 0; ib < 8; ++ib) {
                const float db0 = d * (0.5f + (b->scales[ib] & 0xf)) * 0.25f, db1 = d * (0.5f + (b->scales[ib] >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint16_t q = b->qs[4 * ib + l];
                    const uint8_t *g = (const uint8_t *) (iq2xs_grid + (q & 511));
                    const uint8_t s = ksigns_iq2xs[q >> 9];
                    for (int j = 0; j < 8; ++j) y[256 * i + 32 * ib + 8 * l + j] = (l < 2 ? db0 : db1) * g[j] * (s & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
        }
        return;
    case DS41_T_IQ2_S:
        for (int i = 0; i < nb; ++i) {
            const b_iq2s *b = (const b_iq2s *) row + i;
            const float d = f16(b->d);
            for (int ib = 0; ib < 8; ++ib) {
                const float db0 = d * (0.5f + (b->scales[ib] & 0xf)) * 0.25f, db1 = d * (0.5f + (b->scales[ib] >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t *g = (const uint8_t *) (iq2s_grid + (b->qs[4 * ib + l] | ((b->qh[ib] << (8 - 2 * l)) & 0x300)));
                    const uint8_t s = b->qs[32 + 4 * ib + l];
                    for (int j = 0; j < 8; ++j) y[256 * i + 32 * ib + 8 * l + j] = (l < 2 ? db0 : db1) * g[j] * (s & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
        }
        return;
    case DS41_T_IQ2P:
        for (int i = 0; i < nb; ++i) {
            const b_iq2r *b = (const b_iq2r *) row + i;   /* (the same fields, IQ2P's bit order) */
            const float d = f16(b->d);
            static const float mags[4] = {8, 25, 43, 0};
            for (int k = 0; k < 256; ++k) {
                const int v = k / 64, kk = k % 64, g = k / 16, sc = (b->scales[g / 2] >> (4 * (g % 2))) & 15;
                const int code = (b->codes[kk] >> (2 * v)) & 3, neg = b->signs[8 * v + kk / 8] >> (kk % 8) & 1;
                y[256 * i + k] = d * (0.5f + sc) * 0.25f * mags[code] * (neg ? -1.f : 1.f);
            }
        }
        return;
    case DS41_T_IQ2R:
        for (int i = 0; i < nb; ++i) {
            const b_iq2r *b = (const b_iq2r *) row + i;
            const float d = f16(b->d);
            static const float mags[4] = {8, 25, 43, 0};
            for (int k = 0; k < 256; ++k) {
                const int v = k / 64, kk = k % 64, g = k / 16, sc = (b->scales[g / 2] >> (4 * (g % 2))) & 15;
                const int code = (b->codes[16 * v + kk % 16] >> (2 * (kk / 16))) & 3, neg = b->signs[8 * v + kk % 8] >> (kk / 8) & 1;
                y[256 * i + k] = d * (0.5f + sc) * 0.25f * mags[code] * (neg ? -1.f : 1.f);
            }
        }
        return;
    case DS41_T_IQ3_XXS:
        for (int i = 0; i < nb; ++i) {
            const b_iq3xxs *b = (const b_iq3xxs *) row + i;
            const float d = f16(b->d);
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t a;
                memcpy(&a, b->qs + 64 + 4 * ib, 4);
                const float db = d * (0.5f + (a >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t s = ksigns_iq2xs[(a >> 7 * l) & 127];
                    const uint8_t *g1 = (const uint8_t *) (iq3xxs_grid + b->qs[8 * ib + 2 * l]), *g2 = (const uint8_t *) (iq3xxs_grid + b->qs[8 * ib + 2 * l + 1]);
                    for (int j = 0; j < 4; ++j) {
                        y[256 * i + 32 * ib + 8 * l + j] = db * g1[j] * (s & kmask_iq2xs[j] ? -1.f : 1.f);
                        y[256 * i + 32 * ib + 8 * l + j + 4] = db * g2[j] * (s & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                    }
                }
            }
        }
        return;
    }
}

void ds41_q5k_to_q5p(void *blocks, size_t nblocks)
{
    for (size_t i = 0; i < nblocks; ++i) {
        b_q5k *b = (b_q5k *) blocks + i;
        uint8_t qs[128] = {0}, qh[32] = {0};
        for (int k = 0; k < 256; ++k) {   /* Q5_K: weight 64 j + l (l < 32) low nibble of qs[32 j + l], 64 j + 32 + l high; bit 4 qh[l] bit 2 j (+1) */
            const int j = k / 64, w = k % 64, l = w % 32, hi = w >= 32;
            const int q = ((b->qs[32 * j + l] >> (4 * hi)) & 15) | ((b->qh[l] >> (2 * j + hi) & 1) << 4);
            qs[64 * (j / 2) + w] |= (uint8_t) ((q & 15) << (4 * (j % 2)));
            if (q & 16) qh[8 * j + w / 8] |= (uint8_t) (1u << (w % 8));
        }
        memcpy(b->qs, qs, 128), memcpy(b->qh, qh, 32);
    }
}

void ds41_iq2r_to_iq2p(void *blocks, size_t nblocks)
{
    for (size_t i = 0; i < nblocks; ++i) {
        b_iq2r *b = (b_iq2r *) blocks + i;
        uint8_t codes[64] = {0}, signs[32] = {0};
        for (int k = 0; k < 256; ++k) {
            const int v = k / 64, kk = k % 64;
            const int code = (b->codes[16 * v + kk % 16] >> (2 * (kk / 16))) & 3, neg = b->signs[8 * v + kk % 8] >> (kk / 8) & 1;
            codes[kk] |= (uint8_t) (code << (2 * v));
            if (neg) signs[8 * v + kk / 8] |= (uint8_t) (1u << (kk % 8));
        }
        memcpy(b->codes, codes, 64), memcpy(b->signs, signs, 32);
    }
}

int ds41_repack_iq2r(int t, const void *src, int cols, void *dst)
{
    if (t != DS41_T_IQ2_XXS && t != DS41_T_IQ2_XS && t != DS41_T_IQ2_S) return -1;
    const int nb = cols / 256;
    for (int i = 0; i < nb; ++i) {
        uint8_t mag[256], neg[256], sc[16];
        uint16_t d = 0;
        for (int ib = 0; ib < 8; ++ib)
            for (int l = 0; l < 4; ++l) {   /* the source block's grid bytes and signs of weights 32 ib + 8 l + j */
                const uint8_t *g;
                uint8_t s;
                if (t == DS41_T_IQ2_XXS) {
                    const b_iq2xxs *b = (const b_iq2xxs *) src + i;
                    uint32_t a[2];
                    memcpy(a, b->qs + 8 * ib, 8);
                    d = b->d, g = (const uint8_t *) (iq2xxs_grid + ((const uint8_t *) a)[l]), s = ksigns_iq2xs[(a[1] >> 7 * l) & 127];
                    sc[2 * ib] = sc[2 * ib + 1] = (uint8_t) (a[1] >> 28);
                } else if (t == DS41_T_IQ2_XS) {
                    const b_iq2xs *b = (const b_iq2xs *) src + i;
                    const uint16_t q = b->qs[4 * ib + l];
                    d = b->d, g = (const uint8_t *) (iq2xs_grid + (q & 511)), s = ksigns_iq2xs[q >> 9];
                    sc[2 * ib] = b->scales[ib] & 15, sc[2 * ib + 1] = b->scales[ib] >> 4;
                } else {
                    const b_iq2s *b = (const b_iq2s *) src + i;
                    d = b->d, g = (const uint8_t *) (iq2s_grid + (b->qs[4 * ib + l] | ((b->qh[ib] << (8 - 2 * l)) & 0x300))), s = b->qs[32 + 4 * ib + l];
                    sc[2 * ib] = b->scales[ib] & 15, sc[2 * ib + 1] = b->scales[ib] >> 4;
                }
                for (int j = 0; j < 8; ++j) mag[32 * ib + 8 * l + j] = g[j], neg[32 * ib + 8 * l + j] = (s & kmask_iq2xs[j]) != 0;
            }
        b_iq2r *o = (b_iq2r *) dst + i;
        memset(o, 0, sizeof *o);
        o->d = d;
        for (int j = 0; j < 8; ++j) o->scales[j] = (uint8_t) (sc[2 * j] | sc[2 * j + 1] << 4);
        for (int k = 0; k < 256; ++k) {
            const int code = mag[k] == 8 ? 0 : mag[k] == 25 ? 1 : mag[k] == 43 ? 2 : -1;
            if (code < 0) return -1;
            const int v = k / 64, kk = k % 64;
            o->codes[16 * v + kk % 16] |= (uint8_t) (code << (2 * (kk / 16)));
            if (neg[k]) o->signs[8 * v + kk % 8] |= (uint8_t) (1u << (kk / 8));
        }
    }
    return 0;
}

static void act_dequant(int kind, const void *a, int cols, double *x)
{
    if (kind == DS41_ACT_F32) for (int k = 0; k < cols; ++k) x[k] = ((const float *) a)[k];
    else if (kind == DS41_ACT_Q8K) for (int k = 0; k < cols; ++k) x[k] = (double) ((const ds41_q8k *) a)[k / 256].d * ((const ds41_q8k *) a)[k / 256].q[k % 256];
    else for (int k = 0; k < cols; ++k) x[k] = (double) ((const ds41_q80 *) a)[k / 32].d * ((const ds41_q80 *) a)[k / 32].q[k % 32];
}

#include <stdlib.h>
void ds41_rows_ref(int t, const void *w, int cols, int r0, int r1, const void *const *acts, int nt, double *y, double *absy, int ldy)
{
    float *row = malloc(sizeof(float) * cols);
    double *x = malloc(sizeof(double) * cols * nt);
    for (int n = 0; n < nt; ++n) act_dequant(ds41_act_kind(t), acts[n], cols, x + (size_t) n * cols);
    const size_t rb = ds41_row_bytes(t, cols);
    for (int r = r0; r < r1; ++r) {
        ds41_dequant_row(t, (const uint8_t *) w + rb * r, cols, row);
        for (int n = 0; n < nt; ++n) {
            double s = 0, a = 0;
            for (int k = 0; k < cols; ++k) s += (double) row[k] * x[(size_t) n * cols + k], a += fabs((double) row[k] * x[(size_t) n * cols + k]);
            y[(size_t) n * ldy + r] = s;
            if (absy) absy[(size_t) n * ldy + r] = a;
        }
    }
    free(row);
    free(x);
}

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 512
#include <arm_sve.h>
/* the hardware F16 -> F32 conversion (exact; the same value as f16 for every non-NaN input) */
static inline __attribute__((always_inline)) float f16h(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float) x; }
#define INL static inline __attribute__((always_inline))
typedef svint8_t vs8 __attribute__((arm_sve_vector_bits(512)));
typedef svfloat32_t vf32 __attribute__((arm_sve_vector_bits(512)));

static uint8_t GIDX[4][64], AIDX[4][64], SCIDX[64], LO32[64];
static uint64_t SHV[8], HIDX2[8], SH2[8], AIDX3[4][8];
static uint32_t LS32[4][16], LS16[4][16];
/* Q5_K: get_scale_min_k4 for j = 0..7 as lanes (0-7 scales, 8-15 mins), the mins expanded per bsum lane, nibble shifts */
static uint32_t Q5IA[16], Q5IB[16], Q5SA[16], Q5MA[16], Q5MB[16], Q5MI[16];
static uint8_t HI32[64], SH4[64], SH1[64];
/* sign bits to byte masks: each 64-bit lane's byte 0 spread over its 8 bytes, then byte j's bit j */
static uint8_t BCAST8[64], BITS8[64];
/* IQ2R planes: code shifts 2 (k / 16), the magnitudes, sign bit k / 8 */
static uint8_t SHC2[64], VALS3[64], BITS2[64];
#include <pthread.h>
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void init_once(void)
{
    memset(GIDX, 0xff, sizeof GIDX), memset(AIDX, 0xff, sizeof AIDX), memset(SCIDX, 0xff, sizeof SCIDX);
    for (int v = 0; v < 4; ++v)
        for (int l = 0; l < 8; ++l) {
            const int sb = 2 * v + l / 4, ll = l % 4;
            GIDX[v][8 * l] = (uint8_t) (8 * sb + ll);
            for (int b = 0; b < 4; ++b) AIDX[v][8 * l + b] = (uint8_t) (8 * sb + 4 + b);
            AIDX3[v][l] = (uint64_t) sb;
        }
    for (int l = 0; l < 8; ++l) SHV[l] = (uint64_t) (7 * (l % 4)), HIDX2[l] = (uint64_t) (l / 4), SH2[l] = (uint64_t) (2 * (l % 4));
    for (int k = 0; k < 8; ++k) SCIDX[4 * k] = (uint8_t) (8 * k + 7);
    for (int v = 0; v < 4; ++v)
        for (int k = 0; k < 16; ++k) LS32[v][k] = (uint32_t) (2 * v + k / 8), LS16[v][k] = (uint32_t) (4 * v + k / 4);
    for (int k = 0; k < 64; ++k) LO32[k] = (uint8_t) (k & 31), HI32[k] = (uint8_t) (32 + (k & 31)), SH1[k] = k >= 32, SH4[k] = 4 * (k >= 32);
    for (int k = 0; k < 64; ++k) BCAST8[k] = (uint8_t) (8 * (k / 8)), BITS8[k] = (uint8_t) (1u << (k % 8));
    for (int k = 0; k < 64; ++k) SHC2[k] = (uint8_t) (2 * (k / 16)), VALS3[k] = (uint8_t) (k == 0 ? 8 : k == 1 ? 25 : k == 2 ? 43 : 0), BITS2[k] = (uint8_t) (1u << (k / 8));
    for (int l = 0; l < 16; ++l) {
        const int j = l & 7, mn = l >= 8, hi = j >= 4;
        Q5IA[l] = (uint32_t) (mn || hi ? j + 4 : j), Q5SA[l] = (uint32_t) (mn && hi ? 4 : 0), Q5MA[l] = !mn && hi ? 15 : 63;
        Q5IB[l] = (uint32_t) (hi ? (mn ? j : j - 4) : 0), Q5MB[l] = hi ? 0x30 : 0;
        Q5MI[l] = (uint32_t) (8 + l / 2);
    }
}
static void init_tables(void) { pthread_once(&once, init_once); }

INL svint8_t signed_plane(svbool_t p8, svuint64_t g, svuint64_t m)
{
    const svuint8_t mb = svreinterpret_u8_u64(m);
    return svreinterpret_s8_u8(svsub_u8_x(p8, sveor_u8_x(p8, svreinterpret_u8_u64(g), mb), mb));
}
/* the same plane from the lanes' sign bits s8 (bit j: byte j negative) by a predicated negate, without the mask
   gather (the grid bytes are positive). Faster for IQ2_S (explicit sign bytes) and IQ3_XXS; IQ2_XXS / IQ2_XS keep
   the ksigns64 gather (the parity bits cost more than it there) */
INL svint8_t sign_bits(svbool_t p8, svuint64_t g, svuint64_t s8, svuint8_t Bc, svuint8_t Bt)
{
    const svbool_t neg = svcmpne_n_u8(p8, svand_u8_x(p8, svtbl_u8(svreinterpret_u8_u64(s8), Bc), Bt), 0);
    const svint8_t v = svreinterpret_s8_u64(g);
    return svneg_s8_m(v, neg, v);
}
/* ksigns_iq2xs: 7 sign bits plus their parity in bit 7 */
INL svuint64_t parity7(svbool_t p64, svuint64_t s7)
{
    return svorr_u64_x(p64, s7, svlsl_n_u64_x(p64, svand_n_u64_x(p64, svcnt_u64_x(p64, s7), 1), 7));
}

/* the four plane dot products of one Q8_K token block, each times its lane scales, summed */
#define TOK(k)                                                                                                   \
    if (NT > k) {                                                                                                \
        const ds41_q8k *x = a##k + i;                                                                            \
        svint32_t s = svmul_s32_x(p32, svdot_s32(z, W0, svld1_s8(p8, x->q)), S0);                                \
        s = svmla_s32_x(p32, s, svdot_s32(z, W1, svld1_s8(p8, x->q + 64)), S1);                                  \
        svint32_t u = svmul_s32_x(p32, svdot_s32(z, W2, svld1_s8(p8, x->q + 128)), S2);                          \
        u = svmla_s32_x(p32, u, svdot_s32(z, W3, svld1_s8(p8, x->q + 192)), S3);                                 \
        A##k = svmla_n_f32_x(p32, A##k, svcvt_f32_s32_x(p32, svadd_s32_x(p32, s, u)), d * x->d);                 \
    }
#define TOKS TOK(0) TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7)
#define ROW_HEAD(BT)                                                                                             \
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p64 = svptrue_b64();                                  \
    const svint32_t z = svdup_s32(0);                                                                            \
    const svuint8_t Bc = svld1_u8(p8, BCAST8), Bt = svld1_u8(p8, BITS8);                                         \
    (void) Bc, (void) Bt;                                                                                        \
    const ds41_q8k *a0 = acts[0], *a1 = NT > 1 ? acts[1] : a0, *a2 = NT > 2 ? acts[2] : a0, *a3 = NT > 3 ? acts[3] : a0; \
    const ds41_q8k *a4 = NT > 4 ? acts[4] : a0, *a5 = NT > 5 ? acts[5] : a0, *a6 = NT > 6 ? acts[6] : a0, *a7 = NT > 7 ? acts[7] : a0; \
    (void) p64;
/* the row loop: tables and constants stay in registers across rows (a down chunk's rows are a single block, where
   per-row setup cost as much as the block) */
#define ROW_BEGIN(BT)                                                                                            \
    _Pragma("clang loop unroll_count(2)")                                                                        \
    for (int r = r0; r < r1; ++r) {                                                                              \
        const BT *wr = (const BT *) ((const uint8_t *) w + rb * (size_t) r);                                     \
        svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0, A4 = A0, A5 = A0, A6 = A0, A7 = A0;
#define ROW_STORE                                                                                                \
        for (int k = 0; k < NT; ++k) y[(size_t) k * ldy + r] = out[k];                                           \
    }
#define ROW_TAIL                                                                                                 \
    float out[8];                                                                                                \
    out[0] = svaddv_f32(p32, A0);                                                                                \
    if (NT > 1) out[1] = svaddv_f32(p32, A1);                                                                    \
    if (NT > 2) out[2] = svaddv_f32(p32, A2);                                                                    \
    if (NT > 3) out[3] = svaddv_f32(p32, A3);                                                                    \
    if (NT > 4) out[4] = svaddv_f32(p32, A4);                                                                    \
    if (NT > 5) out[5] = svaddv_f32(p32, A5);                                                                    \
    if (NT > 6) out[6] = svaddv_f32(p32, A6);                                                                    \
    if (NT > 7) out[7] = svaddv_f32(p32, A7);

INL void row_iq2xxs(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_iq2xxs)
    const svuint8_t Sci = svld1_u8(p8, SCIDX);
    const svuint32_t L0 = svld1_u32(p32, LS32[0]), L1 = svld1_u32(p32, LS32[1]), L2 = svld1_u32(p32, LS32[2]), L3 = svld1_u32(p32, LS32[3]);
    const svuint64_t Shv = svld1_u64(p64, SHV);
    ROW_BEGIN(b_iq2xxs)
    for (int i = 0; i < nb; ++i) {
        const b_iq2xxs *b = wr + i;
        const svuint8_t Q = svld1_u8(p8, b->qs);
        const svuint32_t s4 = svlsr_n_u32_x(p32, svreinterpret_u32_u8(svtbl_u8(Q, Sci)), 4);
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s4, 1), 1));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        vs8 W[4];
        for (int v = 0; v < 4; ++v) {
            const svuint64_t gi = svreinterpret_u64_u8(svtbl_u8(Q, svld1_u8(p8, GIDX[v])));
            const svuint64_t g = svld1_gather_u64index_u64(p64, iq2xxs_grid, gi);
            const svuint64_t a1 = svreinterpret_u64_u8(svtbl_u8(Q, svld1_u8(p8, AIDX[v])));
            const svuint64_t s7 = svand_n_u64_x(p64, svlsr_u64_x(p64, a1, Shv), 127);
            W[v] = signed_plane(p8, g, svld1_gather_u64index_u64(p64, ksigns64, s7));
        }
        const svint8_t W0 = W[0], W1 = W[1], W2 = W[2], W3 = W[3];
        const float d = f16h(b->d) * 0.125f;
        TOKS
    }
    ROW_TAIL
    ROW_STORE
}

INL void row_iq2xs(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_iq2xs)
    const svbool_t p8l = svwhilelt_b32(0, 8);
    const svuint32_t L0 = svld1_u32(p32, LS16[0]), L1 = svld1_u32(p32, LS16[1]), L2 = svld1_u32(p32, LS16[2]), L3 = svld1_u32(p32, LS16[3]);
    ROW_BEGIN(b_iq2xs)
    for (int i = 0; i < nb; ++i) {
        const b_iq2xs *b = wr + i;
        const svuint32_t sv = svld1ub_u32(p8l, b->scales);
        const svuint32_t s16 = svzip1_u32(svand_n_u32_x(p32, sv, 15), svlsr_n_u32_x(p32, sv, 4));
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s16, 1), 1));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        vs8 W[4];
        for (int v = 0; v < 4; ++v) {
            const svuint64_t q = svld1uh_u64(p64, b->qs + 8 * v);
            const svuint64_t g = svld1_gather_u64index_u64(p64, iq2xs_grid, svand_n_u64_x(p64, q, 511));
            W[v] = signed_plane(p8, g, svld1_gather_u64index_u64(p64, ksigns64, svlsr_n_u64_x(p64, q, 9)));
        }
        const svint8_t W0 = W[0], W1 = W[1], W2 = W[2], W3 = W[3];
        const float d = f16h(b->d) * 0.125f;
        TOKS
    }
    ROW_TAIL
    ROW_STORE
}

INL void row_iq2s(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_iq2s)
    const svbool_t p8l = svwhilelt_b32(0, 8);
    const svuint32_t L0 = svld1_u32(p32, LS16[0]), L1 = svld1_u32(p32, LS16[1]), L2 = svld1_u32(p32, LS16[2]), L3 = svld1_u32(p32, LS16[3]);
    const svuint64_t Hi = svld1_u64(p64, HIDX2), Sh = svld1_u64(p64, SH2);
    ROW_BEGIN(b_iq2s)
    for (int i = 0; i < nb; ++i) {
        const b_iq2s *b = wr + i;
        const svuint32_t sv = svld1ub_u32(p8l, b->scales);
        const svuint32_t s16 = svzip1_u32(svand_n_u32_x(p32, sv, 15), svlsr_n_u32_x(p32, sv, 4));
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s16, 1), 1));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        vs8 W[4];
        for (int v = 0; v < 4; ++v) {
            const svuint64_t lo = svld1ub_u64(p64, b->qs + 8 * v);
            const svuint64_t h = svtbl_u64(svld1ub_u64(p64, b->qh + 2 * v), Hi);
            const svuint64_t idx = svorr_u64_x(p64, lo, svlsl_n_u64_x(p64, svand_n_u64_x(p64, svlsr_u64_x(p64, h, Sh), 3), 8));
            const svuint64_t g = svld1_gather_u64index_u64(p64, iq2s_grid, idx);
            W[v] = sign_bits(p8, g, svld1ub_u64(p64, b->qs + 32 + 8 * v), Bc, Bt);
        }
        const svint8_t W0 = W[0], W1 = W[1], W2 = W[2], W3 = W[3];
        const float d = f16h(b->d) * 0.125f;
        TOKS
    }
    ROW_TAIL
    ROW_STORE
}

INL void row_iq3xxs(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_iq3xxs)
    const svbool_t p8l = svwhilelt_b32(0, 8);
    const svuint32_t L0 = svld1_u32(p32, LS32[0]), L1 = svld1_u32(p32, LS32[1]), L2 = svld1_u32(p32, LS32[2]), L3 = svld1_u32(p32, LS32[3]);
    const svuint64_t Shv = svld1_u64(p64, SHV);
    ROW_BEGIN(b_iq3xxs)
    for (int i = 0; i < nb; ++i) {
        const b_iq3xxs *b = wr + i;
        uint32_t aux[8] __attribute__((aligned(64)));
        memcpy(aux, b->qs + 64, 32);
        const svuint32_t a32 = svld1_u32(p8l, aux);
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, svlsr_n_u32_x(p32, a32, 28), 1), 1));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        vs8 W[4];
        for (int v = 0; v < 4; ++v) {
            const svuint32_t g = svld1_gather_u32index_u32(p32, iq3xxs_grid, svld1ub_u32(p32, b->qs + 16 * v));
            const svuint64_t a = svld1uw_gather_u64index_u64(p64, aux, svld1_u64(p64, AIDX3[v]));
            const svuint64_t s7 = svand_n_u64_x(p64, svlsr_u64_x(p64, a, Shv), 127);
            W[v] = sign_bits(p8, svreinterpret_u64_u32(g), parity7(p64, s7), Bc, Bt);
        }
        const svint8_t W0 = W[0], W1 = W[1], W2 = W[2], W3 = W[3];
        const float d = f16h(b->d) * 0.25f;
        TOKS
    }
    ROW_TAIL
    ROW_STORE
}

INL void row_iq2r(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_iq2r)
    const svbool_t p8l = svwhilelt_b32(0, 8);
    const svuint32_t L0 = svld1_u32(p32, LS16[0]), L1 = svld1_u32(p32, LS16[1]), L2 = svld1_u32(p32, LS16[2]), L3 = svld1_u32(p32, LS16[3]);
    const svuint8_t Shc = svld1_u8(p8, SHC2), Vals = svld1_u8(p8, VALS3), Bs = svld1_u8(p8, BITS2);
    ROW_BEGIN(b_iq2r)
    for (int i = 0; i < nb; ++i) {
        const b_iq2r *b = wr + i;
        const svuint32_t sv = svld1ub_u32(p8l, b->scales);
        const svuint32_t s16 = svzip1_u32(svand_n_u32_x(p32, sv, 15), svlsr_n_u32_x(p32, sv, 4));
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s16, 1), 1));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        vs8 W[4];
        for (int v = 0; v < 4; ++v) {
            const svuint8_t C = svld1rq_u8(p8, b->codes + 16 * v);
            const svint8_t m = svreinterpret_s8_u8(svtbl_u8(Vals, svand_n_u8_x(p8, svlsr_u8_x(p8, C, Shc), 3)));
            uint64_t sw;
            memcpy(&sw, b->signs + 8 * v, 8);
            const svbool_t neg = svcmpne_n_u8(p8, svand_u8_x(p8, svreinterpret_u8_u64(svdup_n_u64(sw)), Bs), 0);
            W[v] = svneg_s8_m(m, neg, m);
        }
        const svint8_t W0 = W[0], W1 = W[1], W2 = W[2], W3 = W[3];
        const float d = f16h(b->d) * 0.125f;
        TOKS
    }
    ROW_TAIL
    ROW_STORE
}

/* IQ2P: one 64-byte code load for the four planes (shift, mask, magnitude table), each plane's signs one predicate load */
INL void row_iq2p(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_iq2r)
    const svbool_t p8l = svwhilelt_b32(0, 8);
    const svuint32_t L0 = svld1_u32(p32, LS16[0]), L1 = svld1_u32(p32, LS16[1]), L2 = svld1_u32(p32, LS16[2]), L3 = svld1_u32(p32, LS16[3]);
    const svuint8_t Vals = svld1_u8(p8, VALS3);
    ROW_BEGIN(b_iq2r)
    for (int i = 0; i < nb; ++i) {
        const b_iq2r *b = wr + i;
        const svuint32_t sv = svld1ub_u32(p8l, b->scales);
        const svuint32_t s16 = svzip1_u32(svand_n_u32_x(p32, sv, 15), svlsr_n_u32_x(p32, sv, 4));
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s16, 1), 1));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        const svuint8_t C = svld1_u8(p8, b->codes);
        svbool_t g0, g1, g2, g3;
        __asm__("ldr %0, [%1]" : "=Upa"(g0) : "r"(b->signs));
        __asm__("ldr %0, [%1]" : "=Upa"(g1) : "r"(b->signs + 8));
        __asm__("ldr %0, [%1]" : "=Upa"(g2) : "r"(b->signs + 16));
        __asm__("ldr %0, [%1]" : "=Upa"(g3) : "r"(b->signs + 24));
        const svint8_t m0 = svreinterpret_s8_u8(svtbl_u8(Vals, svand_n_u8_x(p8, C, 3)));
        const svint8_t m1 = svreinterpret_s8_u8(svtbl_u8(Vals, svand_n_u8_x(p8, svlsr_n_u8_x(p8, C, 2), 3)));
        const svint8_t m2 = svreinterpret_s8_u8(svtbl_u8(Vals, svand_n_u8_x(p8, svlsr_n_u8_x(p8, C, 4), 3)));
        const svint8_t m3 = svreinterpret_s8_u8(svtbl_u8(Vals, svlsr_n_u8_x(p8, C, 6)));
        const svint8_t W0 = svneg_s8_m(m0, g0, m0), W1 = svneg_s8_m(m1, g1, m1), W2 = svneg_s8_m(m2, g2, m2), W3 = svneg_s8_m(m3, g3, m3);
        const float d = f16h(b->d) * 0.125f;
        TOKS
    }
    ROW_TAIL
    ROW_STORE
}

/* Opt-in 16-token IQ2P tile: identical per-lane FMA chains and final reduction. */
INL void row_iq2p16(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_iq2r)
    const ds41_q8k *a8 = NT > 8 ? acts[8] : a0;
    const ds41_q8k *a9 = NT > 9 ? acts[9] : a0;
    const ds41_q8k *a10 = NT > 10 ? acts[10] : a0;
    const ds41_q8k *a11 = NT > 11 ? acts[11] : a0;
    const ds41_q8k *a12 = NT > 12 ? acts[12] : a0;
    const ds41_q8k *a13 = NT > 13 ? acts[13] : a0;
    const ds41_q8k *a14 = NT > 14 ? acts[14] : a0;
    const ds41_q8k *a15 = NT > 15 ? acts[15] : a0;
    const svbool_t p8l = svwhilelt_b32(0, 8);
    const svuint32_t L0 = svld1_u32(p32, LS16[0]), L1 = svld1_u32(p32, LS16[1]), L2 = svld1_u32(p32, LS16[2]), L3 = svld1_u32(p32, LS16[3]);
    const svuint8_t Vals = svld1_u8(p8, VALS3);
    ROW_BEGIN(b_iq2r)
    svfloat32_t A8 = A0, A9 = A0, A10 = A0, A11 = A0, A12 = A0, A13 = A0, A14 = A0, A15 = A0;
    for (int i = 0; i < nb; ++i) {
        const b_iq2r *b = wr + i;
        const svuint32_t sv = svld1ub_u32(p8l, b->scales);
        const svuint32_t s16 = svzip1_u32(svand_n_u32_x(p32, sv, 15), svlsr_n_u32_x(p32, sv, 4));
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s16, 1), 1));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        const svuint8_t C = svld1_u8(p8, b->codes);
        svbool_t g0, g1, g2, g3;
        __asm__("ldr %0, [%1]" : "=Upa"(g0) : "r"(b->signs));
        __asm__("ldr %0, [%1]" : "=Upa"(g1) : "r"(b->signs + 8));
        __asm__("ldr %0, [%1]" : "=Upa"(g2) : "r"(b->signs + 16));
        __asm__("ldr %0, [%1]" : "=Upa"(g3) : "r"(b->signs + 24));
        const svint8_t m0 = svreinterpret_s8_u8(svtbl_u8(Vals, svand_n_u8_x(p8, C, 3)));
        const svint8_t m1 = svreinterpret_s8_u8(svtbl_u8(Vals, svand_n_u8_x(p8, svlsr_n_u8_x(p8, C, 2), 3)));
        const svint8_t m2 = svreinterpret_s8_u8(svtbl_u8(Vals, svand_n_u8_x(p8, svlsr_n_u8_x(p8, C, 4), 3)));
        const svint8_t m3 = svreinterpret_s8_u8(svtbl_u8(Vals, svlsr_n_u8_x(p8, C, 6)));
        const svint8_t W0 = svneg_s8_m(m0, g0, m0), W1 = svneg_s8_m(m1, g1, m1), W2 = svneg_s8_m(m2, g2, m2), W3 = svneg_s8_m(m3, g3, m3);
        const float d = f16h(b->d) * 0.125f;
        TOKS
        TOK(8) TOK(9) TOK(10) TOK(11) TOK(12) TOK(13) TOK(14) TOK(15)
    }
    float out[16];
    if (NT > 0) out[0] = svaddv_f32(p32, A0);
    if (NT > 1) out[1] = svaddv_f32(p32, A1);
    if (NT > 2) out[2] = svaddv_f32(p32, A2);
    if (NT > 3) out[3] = svaddv_f32(p32, A3);
    if (NT > 4) out[4] = svaddv_f32(p32, A4);
    if (NT > 5) out[5] = svaddv_f32(p32, A5);
    if (NT > 6) out[6] = svaddv_f32(p32, A6);
    if (NT > 7) out[7] = svaddv_f32(p32, A7);
    if (NT > 8) out[8] = svaddv_f32(p32, A8);
    if (NT > 9) out[9] = svaddv_f32(p32, A9);
    if (NT > 10) out[10] = svaddv_f32(p32, A10);
    if (NT > 11) out[11] = svaddv_f32(p32, A11);
    if (NT > 12) out[12] = svaddv_f32(p32, A12);
    if (NT > 13) out[13] = svaddv_f32(p32, A13);
    if (NT > 14) out[14] = svaddv_f32(p32, A14);
    if (NT > 15) out[15] = svaddv_f32(p32, A15);
    ROW_STORE
}

INL void row_q5k(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_q5k)
    const svbool_t p12 = svwhilelt_b32(0, 12), plo = svwhilelt_b8(0, 32);
    const svuint32_t L0 = svld1_u32(p32, LS32[0]), L1 = svld1_u32(p32, LS32[1]), L2 = svld1_u32(p32, LS32[2]), L3 = svld1_u32(p32, LS32[3]);
    const svuint32_t Ia = svld1_u32(p32, Q5IA), Ib = svld1_u32(p32, Q5IB), Sa = svld1_u32(p32, Q5SA), Ma = svld1_u32(p32, Q5MA),
                     Mb = svld1_u32(p32, Q5MB), Mi = svld1_u32(p32, Q5MI);
    const svuint8_t Lo = svld1_u8(p8, LO32), Hi = svld1_u8(p8, HI32), Sh4 = svld1_u8(p8, SH4), Sh1 = svld1_u8(p8, SH1);
    ROW_BEGIN(b_q5k)
    float M[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < nb; ++i) {
        const b_q5k *b = wr + i;
        /* scales and mins in vector lanes (no scalar stores feeding vector loads) */
        const svuint32_t V = svld1ub_u32(p12, b->scales);
        const svuint32_t sa = svand_u32_x(p32, svlsr_u32_x(p32, svtbl_u32(V, Ia), Sa), Ma);
        const svint32_t ls = svreinterpret_s32_u32(svorr_u32_x(p32, sa, svand_u32_x(p32, svlsr_n_u32_x(p32, svtbl_u32(V, Ib), 2), Mb)));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        const svint32_t mx = svtbl_s32(ls, Mi);
        /* plane j: low nibbles of qs[32j..] then high nibbles, bit 4 from qh bit 2j (2j + 1 for the high half) */
        const svuint8_t H = svlsr_u8_x(p8, svtbl_u8(svld1_u8(plo, b->qh), Lo), Sh1);
        vs8 W[4];
        for (int jp = 0; jp < 2; ++jp) {
            const svuint8_t Q = svld1_u8(p8, b->qs + 64 * jp);
            for (int h = 0; h < 2; ++h) {
                const int j = 2 * jp + h;
                const svuint8_t nib = svand_n_u8_x(p8, svlsr_u8_x(p8, svtbl_u8(Q, h ? Hi : Lo), Sh4), 15);
                const svuint8_t hb = j < 3 ? svlsl_n_u8_x(p8, H, 4 - 2 * j) : svlsr_n_u8_x(p8, H, 2);
                W[j] = svreinterpret_s8_u8(svorr_u8_x(p8, nib, svand_n_u8_x(p8, hb, 16)));
            }
        }
        const svint8_t W0 = W[0], W1 = W[1], W2 = W[2], W3 = W[3];
        const float d = f16h(b->d), dm = f16h(b->dmin);
        TOKS
#define MINS(k)                                                                                                  \
        if (NT > k) {                                                                                            \
            const ds41_q8k *x = a##k + i;                                                                        \
            const int s = (int) svaddv_s32(p32, svmul_s32_x(p32, mx, svld1sh_s32(p32, x->bsum)));                 \
            M[k] += dm * x->d * (float) s;                                                                       \
        }
        MINS(0) MINS(1) MINS(2) MINS(3) MINS(4) MINS(5) MINS(6) MINS(7)
#undef MINS
    }
    ROW_TAIL
    for (int k = 0; k < NT; ++k) out[k] -= M[k];
    ROW_STORE
}

/* Q5P: two planes per 64-byte load (mask / shift), each plane's fifth bits one predicate load (or 16 where set) */
INL void row_q5p(int nb, const void *w, size_t rb, int r0, int r1, const ds41_q8k *const *acts, const int NT, float *y, int ldy)
{
    ROW_HEAD(b_q5k)
    const svbool_t p12 = svwhilelt_b32(0, 12);
    const svuint32_t L0 = svld1_u32(p32, LS32[0]), L1 = svld1_u32(p32, LS32[1]), L2 = svld1_u32(p32, LS32[2]), L3 = svld1_u32(p32, LS32[3]);
    const svuint32_t Ia = svld1_u32(p32, Q5IA), Ib = svld1_u32(p32, Q5IB), Sa = svld1_u32(p32, Q5SA), Ma = svld1_u32(p32, Q5MA),
                     Mb = svld1_u32(p32, Q5MB), Mi = svld1_u32(p32, Q5MI);
    const svbool_t p16h = svwhilelt_b16(0, 16);
    const svint64_t z64 = svdup_s64(0);
    const uint64_t abase[8] = {(uint64_t) &a0->d, (uint64_t) &a1->d, (uint64_t) &a2->d, (uint64_t) &a3->d,
                               (uint64_t) &a4->d, (uint64_t) &a5->d, (uint64_t) &a6->d, (uint64_t) &a7->d};
    const svuint64_t Ab = svld1_u64(p64, abase);
    (void) p16h, (void) z64, (void) Ab;
    ROW_BEGIN(b_q5k)
    float M[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    svfloat32_t MV = svdup_f32(0);
    for (int i = 0; i < nb; ++i) {
        const b_q5k *b = wr + i;
        const svuint32_t V = svld1ub_u32(p12, b->scales);
        const svuint32_t sa = svand_u32_x(p32, svlsr_u32_x(p32, svtbl_u32(V, Ia), Sa), Ma);
        const svint32_t ls = svreinterpret_s32_u32(svorr_u32_x(p32, sa, svand_u32_x(p32, svlsr_n_u32_x(p32, svtbl_u32(V, Ib), 2), Mb)));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        const svint32_t mx = svtbl_s32(ls, Mi);
        const svuint8_t Q0 = svld1_u8(p8, b->qs), Q1 = svld1_u8(p8, b->qs + 64);
        svbool_t g0, g1, g2, g3;
        __asm__("ldr %0, [%1]" : "=Upa"(g0) : "r"(b->qh));
        __asm__("ldr %0, [%1]" : "=Upa"(g1) : "r"(b->qh + 8));
        __asm__("ldr %0, [%1]" : "=Upa"(g2) : "r"(b->qh + 16));
        __asm__("ldr %0, [%1]" : "=Upa"(g3) : "r"(b->qh + 24));
        const svint8_t W0 = svreinterpret_s8_u8(svorr_n_u8_m(g0, svand_n_u8_x(p8, Q0, 15), 16));
        const svint8_t W1 = svreinterpret_s8_u8(svorr_n_u8_m(g1, svlsr_n_u8_x(p8, Q0, 4), 16));
        const svint8_t W2 = svreinterpret_s8_u8(svorr_n_u8_m(g2, svand_n_u8_x(p8, Q1, 15), 16));
        const svint8_t W3 = svreinterpret_s8_u8(svorr_n_u8_m(g3, svlsr_n_u8_x(p8, Q1, 4), 16));
        const float d = f16h(b->d), dm = f16h(b->dmin);
        TOKS
#define MINS(k)                                                                                                  \
        if (NT > k) {                                                                                            \
            const ds41_q8k *x = a##k + i;                                                                        \
            const int s = (int) svaddv_s32(p32, svmul_s32_x(p32, mx, svld1sh_s32(p32, x->bsum)));                 \
            M[k] += dm * x->d * (float) s;                                                                       \
        }
        if (NT >= 4) {
            /* 4-8 tokens: every token's mins . bsums (an exact integer, so any order gives the same s) at once: one
               16-bit dot per token pair, then two unzip-add steps leave token k's sum in 64-bit lane k */
            const svint16_t m16 = svuzp1_s16(svreinterpret_s16_s32(mx), svreinterpret_s16_s32(mx));
#define BPAIR(u, v) svdot_s64(z64, svsplice_s16(p16h, svld1_s16(p16h, (a##u + i)->bsum), svld1_s16(p16h, (a##v + i)->bsum)), m16)
            const svint64_t d01 = BPAIR(0, 1), d23 = BPAIR(2, 3), d45 = BPAIR(4, 5), d67 = BPAIR(6, 7);
#undef BPAIR
            const svint64_t e0 = svadd_s64_x(p64, svuzp1_s64(d01, d23), svuzp2_s64(d01, d23));
            const svint64_t e1 = svadd_s64_x(p64, svuzp1_s64(d45, d67), svuzp2_s64(d45, d67));
            const svint64_t sk = svadd_s64_x(p64, svuzp1_s64(e0, e1), svuzp2_s64(e0, e1));
            if (nb != 1) {   /* (KDISP: nb is the constant 1 or known != 1, so this folds) token k's chain in 64-bit lane k (its float in the low half): M_k = fma(dm * d_k,
                               (float) s_k, M_k), the scalar code's fmul + fmadd per lane */
                const svfloat32_t df = svreinterpret_f32_u64(svld1uw_gather_u64base_u64(p64, svadd_n_u64_x(p64, Ab, (uint64_t) i * sizeof(ds41_q8k))));
                MV = svmla_f32_x(p32, MV, svmul_n_f32_x(p32, df, dm), svcvt_f32_s64_x(p64, sk));
            } else {   /* (one-block rows: the vector chain's per-row setup costs more than the scalar fmas) */
                int64_t sv[8];
                svst1_s64(p64, sv, sk);
#define MINS2(k) if (NT > k) { const int s = (int) sv[k]; M[k] += dm * (a##k + i)->d * (float) s; }
                MINS2(0) MINS2(1) MINS2(2) MINS2(3) MINS2(4) MINS2(5) MINS2(6) MINS2(7)
#undef MINS2
            }
        } else {
            MINS(0) MINS(1) MINS(2) MINS(3)
        }
#undef MINS
    }
    ROW_TAIL
    if (NT >= 4 && nb != 1) {
        float mv[16];
        svst1_f32(p32, mv, MV);
        for (int k = 0; k < NT; ++k) M[k] = mv[2 * k];
    }
    for (int k = 0; k < NT; ++k) out[k] -= M[k];
    ROW_STORE
}

INL void row_q80(int nb, const void *w, const ds41_q80 *const *acts, const int NT, float *out)
{
    const svbool_t pb = svwhilelt_b8(0, 32), p32 = svptrue_b32();
    const svint32_t z = svdup_s32(0);
    vf32 A[8];
    for (int k = 0; k < 8; ++k) A[k] = svdup_f32(0);
    const b_q80 *wr = (const b_q80 *) w;
    for (int i = 0; i < nb; ++i) {
        const svint8_t q = svld1_s8(pb, wr[i].qs);
        const float d = f16h(wr[i].d);
        for (int k = 0; k < NT; ++k)
            A[k] = svmla_n_f32_x(p32, A[k], svcvt_f32_s32_x(p32, svdot_s32(z, q, svld1_s8(pb, acts[k][i].q))), d * acts[k][i].d);
    }
    for (int k = 0; k < NT; ++k) out[k] = svaddv_f32(p32, A[k]);
}

/* MXFP4 x Q8_0: one 32-weight block per step (nibbles to 32 int8 lanes: low nibbles, then high, through the e2m1 table) */
INL void row_mxfp4(int nb, const void *w, const ds41_q80 *const *acts, const int NT, float *out)
{
    const svbool_t pb = svwhilelt_b8(0, 32), p16 = svwhilelt_b8(0, 16), p32 = svptrue_b32();
    const svint32_t z = svdup_s32(0);
    const svint8_t Kv = svld1_s8(p16, kvalues_mxfp4);
    vf32 A[8];
    for (int k = 0; k < 8; ++k) A[k] = svdup_f32(0);
    const b_mxfp4 *wr = (const b_mxfp4 *) w;
    for (int i = 0; i < nb; ++i) {
        const svuint8_t qs = svld1_u8(p16, wr[i].qs);
        const svuint8_t idx = svsplice_u8(p16, svand_n_u8_x(p16, qs, 15), svlsr_n_u8_x(p16, qs, 4));
        const svint8_t q = svtbl_s8(Kv, idx);
        const float d = e8m0_half(wr[i].e);
        for (int k = 0; k < NT; ++k)
            A[k] = svmla_n_f32_x(p32, A[k], svcvt_f32_s32_x(p32, svdot_s32(z, q, svld1_s8(pb, acts[k][i].q))), d * acts[k][i].d);
    }
    for (int k = 0; k < NT; ++k) out[k] = svaddv_f32(p32, A[k]);
}

INL void row_f(const int bf, int cols, const void *w, const float *const *acts, const int NT, float *out)
{
    const svbool_t p32 = svptrue_b32();
    vf32 A[8][2];
    for (int k = 0; k < 8; ++k) A[k][0] = A[k][1] = svdup_f32(0);
    for (int c = 0; c < cols; c += 32) {
        vf32 wv[2];
        for (int h = 0; h < 2; ++h) {
            const svbool_t p = svwhilelt_b32(c + 16 * h, cols);
            wv[h] = bf ? svreinterpret_f32_u32(svlsl_n_u32_x(p, svld1uh_u32(p, (const uint16_t *) w + c + 16 * h), 16))
                       : svld1_f32(p, (const float *) w + c + 16 * h);
            for (int k = 0; k < NT; ++k) A[k][h] = svmla_f32_m(p, A[k][h], wv[h], svld1_f32(p, acts[k] + c + 16 * h));
        }
    }
    for (int k = 0; k < NT; ++k) out[k] = svaddv_f32(p32, svadd_f32_x(p32, A[k][0], A[k][1]));
}

#define DISPATCH(F, ...)                                                                                         \
    switch (nt) { case 1: F(__VA_ARGS__, 1, o); break; case 2: F(__VA_ARGS__, 2, o); break;                      \
                  case 3: F(__VA_ARGS__, 3, o); break; case 4: F(__VA_ARGS__, 4, o); break;                      \
                  case 5: F(__VA_ARGS__, 5, o); break; case 6: F(__VA_ARGS__, 6, o); break;                      \
                  case 7: F(__VA_ARGS__, 7, o); break; default: F(__VA_ARGS__, 8, o); }

/* one-block rows (an expert chunk's down) with a constant block count: the row body is straight-line code and the
   row loop's 2x unroll lets two rows' gathers and dots overlap (each row's operations unchanged) */
#define KDISP_NB(F, NB)                                                                                          \
    switch (nt) { case 1: F(NB, w, rb, r0, r1, q, 1, y, ldy); break; case 2: F(NB, w, rb, r0, r1, q, 2, y, ldy); break; \
                  case 3: F(NB, w, rb, r0, r1, q, 3, y, ldy); break; case 4: F(NB, w, rb, r0, r1, q, 4, y, ldy); break; \
                  case 5: F(NB, w, rb, r0, r1, q, 5, y, ldy); break; case 6: F(NB, w, rb, r0, r1, q, 6, y, ldy); break; \
                  case 7: F(NB, w, rb, r0, r1, q, 7, y, ldy); break; default: F(NB, w, rb, r0, r1, q, 8, y, ldy); }
#define KDISP(F) if (nb == 1) { KDISP_NB(F, 1) } else { KDISP_NB(F, nb) }

void ds41_rows(int t, const void *w, int cols, int r0, int r1, const void *const *acts, int nt, float *y, int ldy)
{
    init_tables();
    const size_t rb = ds41_row_bytes(t, cols);
    const int nb = cols / 256;
    if (t == DS41_T_IQ2P && nt > 8 && nt <= 16) {
        const ds41_q8k *const *q = (const ds41_q8k *const *) acts;
        if (nt == 16) row_iq2p16(nb, w, rb, r0, r1, q, 16, y, ldy);
        else row_iq2p16(nb, w, rb, r0, r1, q, nt, y, ldy);
        return;
    }
    {   /* the Q8_K types: the row loop inside the kernel */
        const ds41_q8k *const *q = (const ds41_q8k *const *) acts;
        switch (t) {
        case DS41_T_IQ2_XXS: KDISP(row_iq2xxs) return;
        case DS41_T_IQ2_XS: KDISP(row_iq2xs) return;
        case DS41_T_IQ2_S: KDISP(row_iq2s) return;
        case DS41_T_IQ3_XXS: KDISP(row_iq3xxs) return;
        case DS41_T_Q5_K: KDISP(row_q5k) return;
        case DS41_T_Q5P: KDISP(row_q5p) return;
        case DS41_T_IQ2R: KDISP(row_iq2r) return;
        case DS41_T_IQ2P: KDISP(row_iq2p) return;
        }
    }
    for (int r = r0; r < r1; ++r) {
        const void *row = (const uint8_t *) w + rb * r;
        float o[8];
        switch (t) {
        case DS41_T_Q8_0: DISPATCH(row_q80, cols / 32, row, (const ds41_q80 *const *) acts) break;
        case DS41_T_MXFP4: DISPATCH(row_mxfp4, cols / 32, row, (const ds41_q80 *const *) acts) break;
        case DS41_T_BF16: DISPATCH(row_f, 1, cols, row, (const float *const *) acts) break;
        case DS41_T_F32: DISPATCH(row_f, 0, cols, row, (const float *const *) acts) break;
        default: for (int k = 0; k < nt; ++k) o[k] = NAN;
        }
        for (int k = 0; k < nt; ++k) y[(size_t) k * ldy + r] = o[k];
    }
}
#else   /* no 512-bit SVE: the double reference, rounded */
void ds41_rows(int t, const void *w, int cols, int r0, int r1, const void *const *acts, int nt, float *y, int ldy)
{
    double *d = malloc(sizeof(double) * (size_t) nt * ldy);
    ds41_rows_ref(t, w, cols, r0, r1, acts, nt, d, NULL, ldy);
    for (int n = 0; n < nt; ++n)
        for (int r = r0; r < r1; ++r) y[(size_t) n * ldy + r] = (float) d[(size_t) n * ldy + r];
    free(d);
}
#endif
