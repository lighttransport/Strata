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

void q23s_prepare_natural(const void *q8k_row, int cols, q23s_act *out)
{
    const blk_q8k *b = (const blk_q8k *) q8k_row;
    for (int i = 0; i < cols / 256; ++i) {
        memcpy(out[i].x, b[i].qs, 256);
        for (int g = 0; g < 16; ++g) out[i].bsum[g] = b[i].bsums[g];
        out[i].d = b[i].d;
    }
}

void q23s_prepare_quad(const void *q8k_row, int cols, q23s_act *out)
{
    const blk_q8k *b = (const blk_q8k *) q8k_row;
    for (int i = 0; i < cols / 256; ++i) {
        for (int j = 0; j < 4; ++j)
            for (int L = 0; L < 64; ++L) out[i].x[64 * j + L] = b[i].qs[4 * L + j];
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

/* ============================================================================================================ v2 */

/* byte lanes of a plane -> its 16-weight group: 0-15 -> 2j, 16-31 -> 2j+1, 32-47 -> 8+2j, 48-63 -> 9+2j */
static uint8_t JB[4][64];
static int jb_ready;
static void jb_init(void)
{
    if (jb_ready) return;
    for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 64; ++k) JB[j][k] = (uint8_t) ((k < 32 ? 0 : 8) + 2 * j + ((k & 31) >= 16));
    jb_ready = 1;
}

/* Q2_K v2, one or two rows. a_j = q_j * sc (int8, <= 45): isum = SDOT chains only. */
#define Q2V2_DEC(r, wr)                                                                                 \
    const blk_q2k *b##r = (wr) + i;                                                                     \
    const svuint8_t q##r = svld1_u8(p8, b##r->qs);                                                      \
    const svuint8_t sc##r = svand_n_u8_x(p8, svld1_u8(p16, b##r->scales), 15);                          \
    const svint8_t a0##r = svreinterpret_s8_u8(svmul_u8_x(p8, svand_n_u8_x(p8, q##r, 3), svtbl_u8(sc##r, J0)));                      \
    const svint8_t a1##r = svreinterpret_s8_u8(svmul_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q##r, 2), 3), svtbl_u8(sc##r, J1))); \
    const svint8_t a2##r = svreinterpret_s8_u8(svmul_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q##r, 4), 3), svtbl_u8(sc##r, J2))); \
    const svint8_t a3##r = svreinterpret_s8_u8(svmul_u8_x(p8, svlsr_n_u8_x(p8, q##r, 6), svtbl_u8(sc##r, J3)));                     \
    const svint32_t mn##r = svreinterpret_s32_u32(svlsr_n_u32_x(p32, svld1ub_u32(p32, b##r->scales), 4)); \
    const float d##r = h2f(b##r->d), dm##r = h2f(b##r->dmin);

#define Q2V2_ROW(r, k)                                                                                  \
    {                                                                                                   \
        const svint32_t s01 = svdot_s32(svdot_s32(z, a0##r, X0), a1##r, X1);                            \
        const svint32_t s23 = svdot_s32(svdot_s32(z, a2##r, X2), a3##r, X3);                            \
        const svint32_t m = svmul_s32_x(p32, mn##r, BS);                                                \
        A##r##k = svmla_n_f32_x(p32, A##r##k, svcvt_f32_s32_x(p32, svadd_s32_x(p32, s01, s23)), d##r * xd); \
        B##r##k = svmla_n_f32_x(p32, B##r##k, svcvt_f32_s32_x(p32, m), dm##r * xd);                     \
    }

#define Q2V2_TOK(k)                                                                                     \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        const svint8_t X0 = svld1_s8(p8, x->x), X1 = svld1_s8(p8, x->x + 64);                           \
        const svint8_t X2 = svld1_s8(p8, x->x + 128), X3 = svld1_s8(p8, x->x + 192);                    \
        const svint32_t BS = svld1_s32(p32, x->bsum);                                                   \
        const float xd = x->d;                                                                          \
        Q2V2_ROW(0, k)                                                                                  \
        if (R > 1) Q2V2_ROW(1, k)                                                                       \
    }

#define DECL_ACC                                                                                        \
    svfloat32_t A00 = svdup_f32(0), A01 = A00, A02 = A00, A03 = A00, B00 = A00, B01 = A00, B02 = A00, B03 = A00; \
    svfloat32_t A10 = A00, A11 = A00, A12 = A00, A13 = A00, B10 = A00, B11 = A00, B12 = A00, B13 = A00;

#define STORE_ACC                                                                                       \
    out[0] = svaddv_f32(p32, svsub_f32_x(p32, A00, B00));                                               \
    if (NT > 1) out[1] = svaddv_f32(p32, svsub_f32_x(p32, A01, B01));                                   \
    if (NT > 2) out[2] = svaddv_f32(p32, svsub_f32_x(p32, A02, B02));                                   \
    if (NT > 3) out[3] = svaddv_f32(p32, svsub_f32_x(p32, A03, B03));                                   \
    if (R > 1) {                                                                                        \
        out[NT + 0] = svaddv_f32(p32, svsub_f32_x(p32, A10, B10));                                      \
        if (NT > 1) out[NT + 1] = svaddv_f32(p32, svsub_f32_x(p32, A11, B11));                          \
        if (NT > 2) out[NT + 2] = svaddv_f32(p32, svsub_f32_x(p32, A12, B12));                          \
        if (NT > 3) out[NT + 3] = svaddv_f32(p32, svsub_f32_x(p32, A13, B13));                          \
    }

#define ACTS                                                                                            \
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];

static inline __attribute__((always_inline)) void q2k_v2(int nb, const blk_q2k *w0, const blk_q2k *w1,
                                                         const q23s_act *const *acts, const int NT, const int R, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p16 = svwhilelt_b8_s32(0, 16);
    const svuint8_t J0 = svld1_u8(p8, JB[0]), J1 = svld1_u8(p8, JB[1]), J2 = svld1_u8(p8, JB[2]), J3 = svld1_u8(p8, JB[3]);
    const svint32_t z = svdup_s32(0);
    ACTS
    DECL_ACC
    for (int i = 0; i < nb; ++i) {
        Q2V2_DEC(0, w0)
        Q2V2_DEC(1, R > 1 ? w1 : w0)
        Q2V2_TOK(0) Q2V2_TOK(1) Q2V2_TOK(2) Q2V2_TOK(3)
    }
    STORE_ACC
}

void q2k_sdot_rows_v2(int cols, const void *w, size_t stride, int nrows, const q23s_act *const *acts, int nt, float *out)
{
    jb_init();
    const int nb = cols / 256;
    const blk_q2k *w0 = (const blk_q2k *) w, *w1 = (const blk_q2k *) ((const uint8_t *) w + stride);
    if (nrows > 1) {
        switch (nt) { case 1: q2k_v2(nb, w0, w1, acts, 1, 2, out); break; case 2: q2k_v2(nb, w0, w1, acts, 2, 2, out); break;
                      case 3: q2k_v2(nb, w0, w1, acts, 3, 2, out); break; default: q2k_v2(nb, w0, w1, acts, 4, 2, out); }
    } else {
        switch (nt) { case 1: q2k_v2(nb, w0, w0, acts, 1, 1, out); break; case 2: q2k_v2(nb, w0, w0, acts, 2, 1, out); break;
                      case 3: q2k_v2(nb, w0, w0, acts, 3, 1, out); break; default: q2k_v2(nb, w0, w0, acts, 4, 1, out); }
    }
}

/* Q3_K v2: v1's math, two rows sharing the activation loads, split chains (scales stay int32 lane multipliers:
 * (q3 - 4) x (s - 32) reaches 128 and does not fit int8). */
#define Q3V2_DEC(r, wr)                                                                                 \
    const blk_q3k *b##r = (wr) + i;                                                                     \
    const svint32_t sc##r = svsub_n_s32_x(p32, svreinterpret_s32_u32(svorr_u32_x(p32,                  \
        svand_n_u32_x(p32, svlsr_u32_x(p32, svtbl_u32(svld1ub_u32(p12, b##r->scales), L), LS), 15),    \
        svlsl_n_u32_x(p32, svand_n_u32_x(p32, svlsr_u32_x(p32, svtbl_u32(svld1ub_u32(p12, b##r->scales), H), HS), 3), 4))), 32); \
    const svint32_t s0##r = svtbl_s32(sc##r, I0), s1##r = svtbl_s32(sc##r, I1), s2##r = svtbl_s32(sc##r, I2), s3##r = svtbl_s32(sc##r, I3); \
    const svuint8_t q##r = svld1_u8(p8, b##r->qs);                                                      \
    const svuint8_t h##r = svlsr_n_u8_m(phi, svtbl_u8(svld1_u8(plo, b##r->hmask), dup), 4);             \
    const svint8_t w0##r = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, q##r, 3), svand_u8_x(p8, svlsl_n_u8_x(p8, h##r, 2), four))); \
    const svint8_t w1##r = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q##r, 2), 3), svand_u8_x(p8, svlsl_n_u8_x(p8, h##r, 1), four))); \
    const svint8_t w2##r = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q##r, 4), 3), svand_u8_x(p8, h##r, four))); \
    const svint8_t w3##r = svreinterpret_s8_u8(svorr_u8_x(p8, svlsr_n_u8_x(p8, q##r, 6), svand_u8_x(p8, svlsr_n_u8_x(p8, h##r, 1), four))); \
    const float d##r = h2f(b##r->d);

#define Q3V2_ROW(r, k)                                                                                  \
    {                                                                                                   \
        const svint32_t v01 = svmla_s32_x(p32, svmul_s32_x(p32, svdot_s32(z, w0##r, X0), s0##r), svdot_s32(z, w1##r, X1), s1##r); \
        const svint32_t v23 = svmla_s32_x(p32, svmul_s32_x(p32, svdot_s32(z, w2##r, X2), s2##r), svdot_s32(z, w3##r, X3), s3##r); \
        const svint32_t c = svmul_s32_x(p32, sc##r, BS);                                                \
        A##r##k = svmla_n_f32_x(p32, A##r##k, svcvt_f32_s32_x(p32, svadd_s32_x(p32, v01, v23)), d##r * xd); \
        B##r##k = svmla_n_f32_x(p32, B##r##k, svcvt_f32_s32_x(p32, c), 4.0f * d##r * xd);               \
    }

#define Q3V2_TOK(k)                                                                                     \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        const svint8_t X0 = svld1_s8(p8, x->x), X1 = svld1_s8(p8, x->x + 64);                           \
        const svint8_t X2 = svld1_s8(p8, x->x + 128), X3 = svld1_s8(p8, x->x + 192);                    \
        const svint32_t BS = svld1_s32(p32, x->bsum);                                                   \
        const float xd = x->d;                                                                          \
        Q3V2_ROW(0, k)                                                                                  \
        if (R > 1) Q3V2_ROW(1, k)                                                                       \
    }

static inline __attribute__((always_inline)) void q3k_v2(int nb, const blk_q3k *wa, const blk_q3k *wb,
                                                         const q23s_act *const *acts, const int NT, const int R, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svbool_t plo = svwhilelt_b8_s32(0, 32), phi = svnot_b_z(p8, plo), p12 = svwhilelt_b32_s32(0, 12);
    const svuint32_t I0 = svld1_u32(p32, IDX[0]), I1 = svld1_u32(p32, IDX[1]), I2 = svld1_u32(p32, IDX[2]), I3 = svld1_u32(p32, IDX[3]);
    const svuint32_t L = svld1_u32(p32, SC_L), LS = svld1_u32(p32, SC_LS), H = svld1_u32(p32, SC_H), HS = svld1_u32(p32, SC_HS);
    const svuint8_t dup = svand_n_u8_x(p8, svindex_u8(0, 1), 31), four = svdup_u8(4);
    const svint32_t z = svdup_s32(0);
    ACTS
    DECL_ACC
    for (int i = 0; i < nb; ++i) {
        Q3V2_DEC(0, wa)
        Q3V2_DEC(1, R > 1 ? wb : wa)
        Q3V2_TOK(0) Q3V2_TOK(1) Q3V2_TOK(2) Q3V2_TOK(3)
    }
    STORE_ACC
}

void q3k_sdot_rows_v2(int cols, const void *w, size_t stride, int nrows, const q23s_act *const *acts, int nt, float *out)
{
    const int nb = cols / 256;
    const blk_q3k *w0 = (const blk_q3k *) w, *w1 = (const blk_q3k *) ((const uint8_t *) w + stride);
    if (nrows > 1) {
        switch (nt) { case 1: q3k_v2(nb, w0, w1, acts, 1, 2, out); break; case 2: q3k_v2(nb, w0, w1, acts, 2, 2, out); break;
                      case 3: q3k_v2(nb, w0, w1, acts, 3, 2, out); break; default: q3k_v2(nb, w0, w1, acts, 4, 2, out); }
    } else {
        switch (nt) { case 1: q3k_v2(nb, w0, w0, acts, 1, 1, out); break; case 2: q3k_v2(nb, w0, w0, acts, 2, 1, out); break;
                      case 3: q3k_v2(nb, w0, w0, acts, 3, 1, out); break; default: q3k_v2(nb, w0, w0, acts, 4, 1, out); }
    }
}

/* ============================================================================================================ v3 */
/* Software pipelining: the decoded state of block i+1 is produced before block i's token math, so the core's small
 * scheduling queues always hold independent work; scales stay in vector registers (fp16 broadcast + convert). */
static inline __attribute__((always_inline)) svfloat32_t h2v(svbool_t p32, const uint16_t *h)
{
    return svcvt_f32_f16_x(p32, svreinterpret_f16_u16(svdup_n_u16(*h)));
}

#define Q2V3_DEC(b, A0_, A1_, A2_, A3_, MN_, D_, DM_)                                                   \
    {                                                                                                   \
        const svuint8_t q = svld1_u8(p8, (b)->qs);                                                      \
        const svuint8_t sc = svand_n_u8_x(p8, svld1_u8(p16, (b)->scales), 15);                          \
        A0_ = svreinterpret_s8_u8(svmul_u8_x(p8, svand_n_u8_x(p8, q, 3), svtbl_u8(sc, J0)));            \
        A1_ = svreinterpret_s8_u8(svmul_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q, 2), 3), svtbl_u8(sc, J1))); \
        A2_ = svreinterpret_s8_u8(svmul_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q, 4), 3), svtbl_u8(sc, J2))); \
        A3_ = svreinterpret_s8_u8(svmul_u8_x(p8, svlsr_n_u8_x(p8, q, 6), svtbl_u8(sc, J3)));           \
        MN_ = svreinterpret_s32_u32(svlsr_n_u32_x(p32, svld1ub_u32(p32, (b)->scales), 4));              \
        D_ = h2v(p32, &(b)->d); DM_ = h2v(p32, &(b)->dmin);                                             \
    }

#define Q2V3_TOK(k)                                                                                     \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        const svint32_t s01 = svdot_s32(svdot_s32(z, c0, svld1_s8(p8, x->x)), c1, svld1_s8(p8, x->x + 64)); \
        const svint32_t s23 = svdot_s32(svdot_s32(z, c2, svld1_s8(p8, x->x + 128)), c3, svld1_s8(p8, x->x + 192)); \
        const svint32_t m = svmul_s32_x(p32, cmn, svld1_s32(p32, x->bsum));                             \
        const svfloat32_t xd = svdup_n_f32(x->d);                                                       \
        A##k = svmla_f32_x(p32, A##k, svcvt_f32_s32_x(p32, svadd_s32_x(p32, s01, s23)), svmul_f32_x(p32, cd, xd)); \
        B##k = svmla_f32_x(p32, B##k, svcvt_f32_s32_x(p32, m), svmul_f32_x(p32, cdm, xd));              \
    }

static inline __attribute__((always_inline)) void q2k_v3(int nb, const blk_q2k *w, const q23s_act *const *acts,
                                                         const int NT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p16 = svwhilelt_b8_s32(0, 16);
    const svuint8_t J0 = svld1_u8(p8, JB[0]), J1 = svld1_u8(p8, JB[1]), J2 = svld1_u8(p8, JB[2]), J3 = svld1_u8(p8, JB[3]);
    const svint32_t z = svdup_s32(0);
    ACTS
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0, B0 = A0, B1 = A0, B2 = A0, B3 = A0;
    svint8_t c0, c1, c2, c3; svint32_t cmn; svfloat32_t cd, cdm;
    Q2V3_DEC(w, c0, c1, c2, c3, cmn, cd, cdm)
    for (int i = 0; i < nb; ++i) {
        svint8_t n0 = c0, n1 = c1, n2 = c2, n3 = c3; svint32_t nmn = cmn; svfloat32_t nd = cd, ndm = cdm;
        if (i + 1 < nb) Q2V3_DEC(w + i + 1, n0, n1, n2, n3, nmn, nd, ndm)
        Q2V3_TOK(0) Q2V3_TOK(1) Q2V3_TOK(2) Q2V3_TOK(3)
        c0 = n0; c1 = n1; c2 = n2; c3 = n3; cmn = nmn; cd = nd; cdm = ndm;
    }
    out[0] = svaddv_f32(p32, svsub_f32_x(p32, A0, B0));
    if (NT > 1) out[1] = svaddv_f32(p32, svsub_f32_x(p32, A1, B1));
    if (NT > 2) out[2] = svaddv_f32(p32, svsub_f32_x(p32, A2, B2));
    if (NT > 3) out[3] = svaddv_f32(p32, svsub_f32_x(p32, A3, B3));
}

void q2k_sdot_row_v3(int cols, const void *w, const q23s_act *const *acts, int nt, float *out)
{
    jb_init();
    const int nb = cols / 256; const blk_q2k *b = (const blk_q2k *) w;
    switch (nt) { case 1: q2k_v3(nb, b, acts, 1, out); break; case 2: q2k_v3(nb, b, acts, 2, out); break;
                  case 3: q2k_v3(nb, b, acts, 3, out); break; default: q2k_v3(nb, b, acts, 4, out); }
}

#define Q3V3_DEC(b, W0_, W1_, W2_, W3_, SC_, D_)                                                       \
    {                                                                                                   \
        const svuint32_t raw = svld1ub_u32(p12, (b)->scales);                                           \
        SC_ = svsub_n_s32_x(p32, svreinterpret_s32_u32(svorr_u32_x(p32,                                \
            svand_n_u32_x(p32, svlsr_u32_x(p32, svtbl_u32(raw, L), LS), 15),                           \
            svlsl_n_u32_x(p32, svand_n_u32_x(p32, svlsr_u32_x(p32, svtbl_u32(raw, H), HS), 3), 4))), 32); \
        const svuint8_t q = svld1_u8(p8, (b)->qs);                                                      \
        const svuint8_t hh = svlsr_n_u8_m(phi, svtbl_u8(svld1_u8(plo, (b)->hmask), dup), 4);            \
        W0_ = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, q, 3), svand_u8_x(p8, svlsl_n_u8_x(p8, hh, 2), four))); \
        W1_ = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q, 2), 3), svand_u8_x(p8, svlsl_n_u8_x(p8, hh, 1), four))); \
        W2_ = svreinterpret_s8_u8(svorr_u8_x(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, q, 4), 3), svand_u8_x(p8, hh, four))); \
        W3_ = svreinterpret_s8_u8(svorr_u8_x(p8, svlsr_n_u8_x(p8, q, 6), svand_u8_x(p8, svlsr_n_u8_x(p8, hh, 1), four))); \
        D_ = h2v(p32, &(b)->d);                                                                         \
    }

#define Q3V3_TOK(k)                                                                                     \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        const svint32_t v01 = svmla_s32_x(p32, svmul_s32_x(p32, svdot_s32(z, c0, svld1_s8(p8, x->x)), svtbl_s32(csc, I0)), \
                                          svdot_s32(z, c1, svld1_s8(p8, x->x + 64)), svtbl_s32(csc, I1)); \
        const svint32_t v23 = svmla_s32_x(p32, svmul_s32_x(p32, svdot_s32(z, c2, svld1_s8(p8, x->x + 128)), svtbl_s32(csc, I2)), \
                                          svdot_s32(z, c3, svld1_s8(p8, x->x + 192)), svtbl_s32(csc, I3)); \
        const svint32_t cc = svmul_s32_x(p32, csc, svld1_s32(p32, x->bsum));                            \
        const svfloat32_t dd = svmul_f32_x(p32, cd, svdup_n_f32(x->d));                                 \
        A##k = svmla_f32_x(p32, A##k, svcvt_f32_s32_x(p32, svadd_s32_x(p32, v01, v23)), dd);            \
        B##k = svmla_f32_x(p32, B##k, svcvt_f32_s32_x(p32, cc), svmul_n_f32_x(p32, dd, 4.0f));          \
    }

static inline __attribute__((always_inline)) void q3k_v3(int nb, const blk_q3k *w, const q23s_act *const *acts,
                                                         const int NT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svbool_t plo = svwhilelt_b8_s32(0, 32), phi = svnot_b_z(p8, plo), p12 = svwhilelt_b32_s32(0, 12);
    const svuint32_t I0 = svld1_u32(p32, IDX[0]), I1 = svld1_u32(p32, IDX[1]), I2 = svld1_u32(p32, IDX[2]), I3 = svld1_u32(p32, IDX[3]);
    const svuint32_t L = svld1_u32(p32, SC_L), LS = svld1_u32(p32, SC_LS), H = svld1_u32(p32, SC_H), HS = svld1_u32(p32, SC_HS);
    const svuint8_t dup = svand_n_u8_x(p8, svindex_u8(0, 1), 31), four = svdup_u8(4);
    const svint32_t z = svdup_s32(0);
    ACTS
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0, B0 = A0, B1 = A0, B2 = A0, B3 = A0;
    svint8_t c0, c1, c2, c3; svint32_t csc; svfloat32_t cd;
    Q3V3_DEC(w, c0, c1, c2, c3, csc, cd)
    for (int i = 0; i < nb; ++i) {
        svint8_t n0 = c0, n1 = c1, n2 = c2, n3 = c3; svint32_t nsc = csc; svfloat32_t nd = cd;
        if (i + 1 < nb) Q3V3_DEC(w + i + 1, n0, n1, n2, n3, nsc, nd)
        Q3V3_TOK(0) Q3V3_TOK(1) Q3V3_TOK(2) Q3V3_TOK(3)
        c0 = n0; c1 = n1; c2 = n2; c3 = n3; csc = nsc; cd = nd;
    }
    out[0] = svaddv_f32(p32, svsub_f32_x(p32, A0, B0));
    if (NT > 1) out[1] = svaddv_f32(p32, svsub_f32_x(p32, A1, B1));
    if (NT > 2) out[2] = svaddv_f32(p32, svsub_f32_x(p32, A2, B2));
    if (NT > 3) out[3] = svaddv_f32(p32, svsub_f32_x(p32, A3, B3));
}

void q3k_sdot_row_v3(int cols, const void *w, const q23s_act *const *acts, int nt, float *out)
{
    const int nb = cols / 256; const blk_q3k *b = (const blk_q3k *) w;
    switch (nt) { case 1: q3k_v3(nb, b, acts, 1, out); break; case 2: q3k_v3(nb, b, acts, 2, out); break;
                  case 3: q3k_v3(nb, b, acts, 3, out); break; default: q3k_v3(nb, b, acts, 4, out); }
}
