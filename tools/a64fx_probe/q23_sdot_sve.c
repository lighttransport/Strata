/* tools/a64fx_probe/q23_sdot_sve.c - see q23_sdot_sve.h. Build: fcc -Nclang -O3 -march=armv8.2-a+sve -c */
#include "q23_sdot_sve.h"

#include <arm_sve.h>
#include <string.h>

typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; } blk_q2k;      /* 84 B */
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; } blk_q3k;   /* 110 B */
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } blk_q8k;                           /* 292 B */

_Static_assert(sizeof(blk_q2k) == 84, "block_q2_K");
_Static_assert(sizeof(blk_q3k) == 110, "block_q3_K");
_Static_assert(sizeof(blk_q8k) == 292, "block_q8_K");

static inline float h2f(uint16_t h) { __fp16 v; memcpy(&v, &h, 2); return (float) v; }

int q23s_vl_bytes(void) { return (int) svcntb(); }

void q23s_prepare(const void *q8k_row, int cols, q23s_act *out)
{
    const blk_q8k *b = (const blk_q8k *) q8k_row;
    for (int i = 0; i < cols / 256; ++i) {
        for (int j = 0; j < 4; ++j) {
            memcpy(out[i].x + 64 * j,      b[i].qs + 32 * j,       32);
            memcpy(out[i].x + 64 * j + 32, b[i].qs + 128 + 32 * j, 32);
        }
        for (int g = 0; g < 16; ++g) out[i].bsum[g] = b[i].bsums[g];
        out[i].d = b[i].d;
    }
}

/* int32 lane k of an SDOT result sums bytes 4k..4k+3: lanes 0-3 -> group 2j, 4-7 -> 2j+1, 8-11 -> 8+2j, 12-15 -> 9+2j */
static const uint32_t IDX[4][16] = {
    {0,0,0,0, 1,1,1,1,  8, 8, 8, 8,  9, 9, 9, 9},
    {2,2,2,2, 3,3,3,3, 10,10,10,10, 11,11,11,11},
    {4,4,4,4, 5,5,5,5, 12,12,12,12, 13,13,13,13},
    {6,6,6,6, 7,7,7,7, 14,14,14,14, 15,15,15,15},
};

#define TOKEN_Q2(k)                                                                                     \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        svint32_t v = svmul_s32_x(p32, svdot_s32(z, p0, svld1_s8(p8, x->x)), s0);                       \
        v = svmla_s32_x(p32, v, svdot_s32(z, p1, svld1_s8(p8, x->x + 64)), s1);                         \
        v = svmla_s32_x(p32, v, svdot_s32(z, p2, svld1_s8(p8, x->x + 128)), s2);                        \
        v = svmla_s32_x(p32, v, svdot_s32(z, p3, svld1_s8(p8, x->x + 192)), s3);                        \
        const svint32_t m = svmul_s32_x(p32, mn, svld1_s32(p32, x->bsum));                              \
        A##k = svmla_n_f32_x(p32, A##k, svcvt_f32_s32_x(p32, v), d * x->d);                              \
        B##k = svmla_n_f32_x(p32, B##k, svcvt_f32_s32_x(p32, m), dm * x->d);                             \
    }

static inline __attribute__((always_inline)) void q2k_row(int nb, const blk_q2k *w, const q23s_act *const *acts,
                                                          const int NT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svuint32_t I0 = svld1_u32(p32, IDX[0]), I1 = svld1_u32(p32, IDX[1]), I2 = svld1_u32(p32, IDX[2]), I3 = svld1_u32(p32, IDX[3]);
    const svint32_t z = svdup_s32(0);
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0, B0 = A0, B1 = A0, B2 = A0, B3 = A0;
    for (int i = 0; i < nb; ++i) {
        const blk_q2k *b = w + i;
        const svuint8_t q = svld1_u8(p8, b->qs);
        const svint8_t p0 = svreinterpret_s8_u8(svand_n_u8_x(p8, q, 3));
        const svint8_t p1 = svreinterpret_s8_u8(svand_n_u8_x(p8, svlsr_n_u8_x(p8, q, 2), 3));
        const svint8_t p2 = svreinterpret_s8_u8(svand_n_u8_x(p8, svlsr_n_u8_x(p8, q, 4), 3));
        const svint8_t p3 = svreinterpret_s8_u8(svlsr_n_u8_x(p8, q, 6));
        const svuint32_t sb = svld1ub_u32(p32, b->scales);
        const svint32_t sc = svreinterpret_s32_u32(svand_n_u32_x(p32, sb, 15));
        const svint32_t mn = svreinterpret_s32_u32(svlsr_n_u32_x(p32, sb, 4));
        const svint32_t s0 = svtbl_s32(sc, I0), s1 = svtbl_s32(sc, I1), s2 = svtbl_s32(sc, I2), s3 = svtbl_s32(sc, I3);
        const float d = h2f(b->d), dm = h2f(b->dmin);
        TOKEN_Q2(0) TOKEN_Q2(1) TOKEN_Q2(2) TOKEN_Q2(3)
    }
    out[0] = svaddv_f32(p32, A0) - svaddv_f32(p32, B0);
    if (NT > 1) out[1] = svaddv_f32(p32, A1) - svaddv_f32(p32, B1);
    if (NT > 2) out[2] = svaddv_f32(p32, A2) - svaddv_f32(p32, B2);
    if (NT > 3) out[3] = svaddv_f32(p32, A3) - svaddv_f32(p32, B3);
}

void q2k_sdot_row(int cols, const void *w, const q23s_act *const *acts, int nt, float *out)
{
    const int nb = cols / 256;
    switch (nt) {
    case 1: q2k_row(nb, (const blk_q2k *) w, acts, 1, out); break;
    case 2: q2k_row(nb, (const blk_q2k *) w, acts, 2, out); break;
    case 3: q2k_row(nb, (const blk_q2k *) w, acts, 3, out); break;
    default: q2k_row(nb, (const blk_q2k *) w, acts, 4, out); break;
    }
}

static const uint32_t SC_L[16]  = {0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7};
static const uint32_t SC_LS[16] = {0,0,0,0,0,0,0,0, 4,4,4,4,4,4,4,4};
static const uint32_t SC_H[16]  = {8,9,10,11, 8,9,10,11, 8,9,10,11, 8,9,10,11};
static const uint32_t SC_HS[16] = {0,0,0,0, 2,2,2,2, 4,4,4,4, 6,6,6,6};

/* Q3_K: weight = d * (scale - 32) * ((q2 | hbit << 2) - 4); the "-4" goes through the bsums. hmask bit j of byte l
 * belongs to y[32 j + l] (l < 32, first half) and bit 4 + j to y[128 + 32 j + l]. */
#define TOKEN_Q3(k)                                                                                     \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        svint32_t v = svmul_s32_x(p32, svdot_s32(z, w0, svld1_s8(p8, x->x)), s0);                       \
        v = svmla_s32_x(p32, v, svdot_s32(z, w1, svld1_s8(p8, x->x + 64)), s1);                         \
        v = svmla_s32_x(p32, v, svdot_s32(z, w2, svld1_s8(p8, x->x + 128)), s2);                        \
        v = svmla_s32_x(p32, v, svdot_s32(z, w3, svld1_s8(p8, x->x + 192)), s3);                        \
        const svint32_t c = svmul_s32_x(p32, sc, svld1_s32(p32, x->bsum));                              \
        A##k = svmla_n_f32_x(p32, A##k, svcvt_f32_s32_x(p32, v), d * x->d);                              \
        B##k = svmla_n_f32_x(p32, B##k, svcvt_f32_s32_x(p32, c), 4.0f * d * x->d);                       \
    }

static inline __attribute__((always_inline)) void q3k_row(int nb, const blk_q3k *w, const q23s_act *const *acts,
                                                          const int NT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svbool_t plo = svwhilelt_b8_s32(0, 32), phi = svnot_b_z(p8, plo);
    const svuint32_t I0 = svld1_u32(p32, IDX[0]), I1 = svld1_u32(p32, IDX[1]), I2 = svld1_u32(p32, IDX[2]), I3 = svld1_u32(p32, IDX[3]);
    const svuint8_t dup = svand_n_u8_x(p8, svindex_u8(0, 1), 31);
    const svbool_t p12 = svwhilelt_b32_s32(0, 12);
    const svuint32_t L = svld1_u32(p32, SC_L), LS = svld1_u32(p32, SC_LS), H = svld1_u32(p32, SC_H), HS = svld1_u32(p32, SC_HS);
    const svint32_t z = svdup_s32(0);
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0, B0 = A0, B1 = A0, B2 = A0, B3 = A0;
    for (int i = 0; i < nb; ++i) {
        const blk_q3k *b = w + i;
        /* 6-bit scales, ggml's packing, unpacked in 16 int32 lanes: low nibble from byte L[g] >> LS[g], high two bits
         * from byte 8 + g % 4 >> 2 (g / 4); minus 32 */
        const svuint32_t raw = svld1ub_u32(p12, b->scales);
        const svuint32_t lo = svand_n_u32_x(p32, svlsr_u32_x(p32, svtbl_u32(raw, L), LS), 15);
        const svuint32_t hi = svand_n_u32_x(p32, svlsr_u32_x(p32, svtbl_u32(raw, H), HS), 3);
        const svint32_t sc = svsub_n_s32_x(p32, svreinterpret_s32_u32(svorr_u32_x(p32, lo, svlsl_n_u32_x(p32, hi, 4))), 32);
        const svint32_t s0 = svtbl_s32(sc, I0), s1 = svtbl_s32(sc, I1), s2 = svtbl_s32(sc, I2), s3 = svtbl_s32(sc, I3);
        const svuint8_t q = svld1_u8(p8, b->qs);
        svuint8_t h = svtbl_u8(svld1_u8(plo, b->hmask), dup);   /* hmask in both halves */
        h = svlsr_n_u8_m(phi, h, 4);                             /* upper half uses bits 4..7 */
        const svuint8_t four = svdup_u8(4);
        const svint8_t w0 = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, q, 3),                        svand_u8_x(p8, svlsl_n_u8_x(p8, h, 2), four)));
        const svint8_t w1 = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q, 2), 3),   svand_u8_x(p8, svlsl_n_u8_x(p8, h, 1), four)));
        const svint8_t w2 = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q, 4), 3),   svand_u8_x(p8, h, four)));
        const svint8_t w3 = svreinterpret_s8_u8(svorr_u8_x(p8, svlsr_n_u8_x(p8, q, 6),                        svand_u8_x(p8, svlsr_n_u8_x(p8, h, 1), four)));
        const float d = h2f(b->d);
        TOKEN_Q3(0) TOKEN_Q3(1) TOKEN_Q3(2) TOKEN_Q3(3)
    }
    out[0] = svaddv_f32(p32, A0) - svaddv_f32(p32, B0);
    if (NT > 1) out[1] = svaddv_f32(p32, A1) - svaddv_f32(p32, B1);
    if (NT > 2) out[2] = svaddv_f32(p32, A2) - svaddv_f32(p32, B2);
    if (NT > 3) out[3] = svaddv_f32(p32, A3) - svaddv_f32(p32, B3);
}

void q3k_sdot_row(int cols, const void *w, const q23s_act *const *acts, int nt, float *out)
{
    const int nb = cols / 256;
    switch (nt) {
    case 1: q3k_row(nb, (const blk_q3k *) w, acts, 1, out); break;
    case 2: q3k_row(nb, (const blk_q3k *) w, acts, 2, out); break;
    case 3: q3k_row(nb, (const blk_q3k *) w, acts, 3, out); break;
    default: q3k_row(nb, (const blk_q3k *) w, acts, 4, out); break;
    }
}
