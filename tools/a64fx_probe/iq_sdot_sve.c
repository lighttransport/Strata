/* tools/a64fx_probe/iq_sdot_sve.c - IQ2_XXS and IQ1_S x Q8_K on A64FX (SVE 512, gathers + int8 SDOT).
 *
 * ggml's SVE paths for these codebook formats run at 0.17-0.23 GB/s per core (measured 2026-10-18). Here a
 * 512-bit vector holds 64 weights = two 32-weight sub-blocks = eight 8-weight grid entries, fetched with one
 * 64-bit-element gather; the activation stays in natural order (q23s_prepare_natural).
 *   IQ2_XXS: grid bytes {8, 25, 43}; signs per 8 weights = 7 stored bits + parity. Variant 0 rebuilds the sign mask
 *            in registers (popcount parity, byte replicate, bit test, negate); variant 1 gathers it from ksigns64.
 *            Per sub-block scale (2s+1)/8 as an int32 lane multiplier after the SDOT.
 *   IQ1_S:   11-bit grid index (qs byte + 3 qh bits) into the 2048-entry {-1,0,1} grid; the +-1/8 delta is folded
 *            into the weights as 8g +- 1 (int8), so it costs no per-token work; scale 2s+1 per sub-block.
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -I<ggml/src> -c (ggml-common.h for the grids). Needs VL = 512. */
#include "q23_sdot_sve.h"

#include <arm_sve.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

typedef struct { uint16_t d; uint8_t qs[64]; } blk_iq2xxs;               /* 66 B */
typedef struct { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; } blk_iq1s;  /* 50 B */
_Static_assert(sizeof(blk_iq2xxs) == 66, "block_iq2_xxs");
_Static_assert(sizeof(blk_iq1s) == 50, "block_iq1_s");

static inline float h2f(uint16_t h) { __fp16 v; memcpy(&v, &h, 2); return (float) v; }

static uint8_t GIDX[4][64], AIDX[4][64], REP[64], BITS[64], SCIDX[64], DIDX[4][64];
static uint64_t SHV[8], SH3[8], HIDX[4][8];
static uint32_t LSIDX[4][16];
static int ready;

static void init_tables(void)
{
    if (ready) return;
    memset(GIDX, 0xff, sizeof GIDX); memset(AIDX, 0xff, sizeof AIDX); memset(SCIDX, 0xff, sizeof SCIDX);
    for (int v = 0; v < 4; ++v)
        for (int l = 0; l < 8; ++l) {
            const int sb = 2 * v + l / 4, ll = l % 4;
            GIDX[v][8 * l] = (uint8_t) (8 * sb + ll);                       /* index byte ll of sub-block sb */
            for (int b = 0; b < 4; ++b) AIDX[v][8 * l + b] = (uint8_t) (8 * sb + 4 + b);   /* aux32[1] of sb */
            HIDX[v][l] = (uint64_t) sb;
        }
    for (int l = 0; l < 8; ++l) { SHV[l] = (uint64_t) (7 * (l % 4)); SH3[l] = (uint64_t) (3 * (l % 4)); }
    for (int k = 0; k < 64; ++k) { REP[k] = (uint8_t) (k & ~7); BITS[k] = (uint8_t) (1u << (k & 7)); }
    for (int k = 0; k < 8; ++k) SCIDX[4 * k] = (uint8_t) (8 * k + 7);      /* top byte of aux32[1]: scale in bits 4-7 */
    for (int v = 0; v < 4; ++v) {
        for (int k = 0; k < 16; ++k) LSIDX[v][k] = (uint32_t) (2 * v + (k >= 8));
        for (int k = 0; k < 64; ++k) DIDX[v][k] = (uint8_t) (4 * (2 * v + (k >= 32)));
    }
    ready = 1;
}

/* ------------------------------------------------------------------------------------------------ IQ2_XXS */
#define IQ2_TOK(k)                                                                                      \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        svint32_t v = svmul_s32_x(p32, svdot_s32(z, W0, svld1_s8(p8, x->x)), S0);                       \
        v = svmla_s32_x(p32, v, svdot_s32(z, W1, svld1_s8(p8, x->x + 64)), S1);                         \
        svint32_t u = svmul_s32_x(p32, svdot_s32(z, W2, svld1_s8(p8, x->x + 128)), S2);                 \
        u = svmla_s32_x(p32, u, svdot_s32(z, W3, svld1_s8(p8, x->x + 192)), S3);                        \
        A##k = svmla_n_f32_x(p32, A##k, svcvt_f32_s32_x(p32, svadd_s32_x(p32, v, u)), d * x->d);         \
    }

static inline __attribute__((always_inline)) svint8_t iq2_plane(svbool_t p8, svbool_t p64, svuint8_t Q, int v,
                                                                const int VAR, svuint8_t Rep, svuint8_t Bits)
{
    const svuint64_t gi = svreinterpret_u64_u8(svtbl_u8(Q, svld1_u8(p8, GIDX[v])));
    const svuint64_t g = svld1_gather_u64index_u64(p64, iq2xxs_grid, gi);
    const svuint64_t a1 = svreinterpret_u64_u8(svtbl_u8(Q, svld1_u8(p8, AIDX[v])));
    const svuint64_t s7 = svand_n_u64_x(p64, svlsr_u64_x(p64, a1, svld1_u64(p64, SHV)), 127);
    if (VAR == 1) {
        const svuint8_t m = svreinterpret_u8_u64(svld1_gather_u64index_u64(p64, ksigns64, s7));
        return svreinterpret_s8_u8(svsub_u8_x(p8, sveor_u8_x(p8, svreinterpret_u8_u64(g), m), m));
    }
    const svuint64_t par = svand_n_u64_x(p64, svcnt_u64_x(p64, s7), 1);
    const svuint8_t s8 = svreinterpret_u8_u64(svorr_u64_x(p64, s7, svlsl_n_u64_x(p64, par, 7)));
    const svbool_t neg = svcmpne_n_u8(p8, svand_u8_x(p8, svtbl_u8(s8, Rep), Bits), 0);
    const svint8_t w = svreinterpret_s8_u64(g);
    return svneg_s8_m(w, neg, w);
}

static inline __attribute__((always_inline)) void iq2xxs_row(int nb, const blk_iq2xxs *wr, const q23s_act *const *acts,
                                                             const int NT, const int VAR, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p64 = svptrue_b64();
    const svuint8_t Rep = svld1_u8(p8, REP), Bits = svld1_u8(p8, BITS), Sci = svld1_u8(p8, SCIDX);
    const svuint32_t L0 = svld1_u32(p32, LSIDX[0]), L1 = svld1_u32(p32, LSIDX[1]), L2 = svld1_u32(p32, LSIDX[2]), L3 = svld1_u32(p32, LSIDX[3]);
    const svint32_t z = svdup_s32(0);
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0;
    for (int i = 0; i < nb; ++i) {
        const blk_iq2xxs *b = wr + i;
        const svuint8_t Q = svld1_u8(p8, b->qs);
        /* per sub-block scale 2s+1 in int32 lanes 0..7 */
        const svuint32_t s4 = svlsr_n_u32_x(p32, svreinterpret_u32_u8(svtbl_u8(Q, Sci)), 4);
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s4, 1), 1));
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        const svint8_t W0 = iq2_plane(p8, p64, Q, 0, VAR, Rep, Bits), W1 = iq2_plane(p8, p64, Q, 1, VAR, Rep, Bits);
        const svint8_t W2 = iq2_plane(p8, p64, Q, 2, VAR, Rep, Bits), W3 = iq2_plane(p8, p64, Q, 3, VAR, Rep, Bits);
        const float d = h2f(b->d) * 0.125f;
        IQ2_TOK(0) IQ2_TOK(1) IQ2_TOK(2) IQ2_TOK(3)
    }
    out[0] = svaddv_f32(p32, A0);
    if (NT > 1) out[1] = svaddv_f32(p32, A1);
    if (NT > 2) out[2] = svaddv_f32(p32, A2);
    if (NT > 3) out[3] = svaddv_f32(p32, A3);
}

void iq2xxs_sdot_row(int cols, const void *w, const q23s_act *const *acts, int nt, float *out, int variant)
{
    init_tables();
    const int nb = cols / 256; const blk_iq2xxs *b = (const blk_iq2xxs *) w;
    if (variant == 1) {
        switch (nt) { case 1: iq2xxs_row(nb, b, acts, 1, 1, out); break; case 2: iq2xxs_row(nb, b, acts, 2, 1, out); break;
                      case 3: iq2xxs_row(nb, b, acts, 3, 1, out); break; default: iq2xxs_row(nb, b, acts, 4, 1, out); }
    } else {
        switch (nt) { case 1: iq2xxs_row(nb, b, acts, 1, 0, out); break; case 2: iq2xxs_row(nb, b, acts, 2, 0, out); break;
                      case 3: iq2xxs_row(nb, b, acts, 3, 0, out); break; default: iq2xxs_row(nb, b, acts, 4, 0, out); }
    }
}

/* ------------------------------------------------------------------------------------------------- IQ1_S */
static inline __attribute__((always_inline)) svint8_t iq1_plane(svbool_t p8, svbool_t p64, const blk_iq1s *b, int v,
                                                                svuint64_t QH, svuint64_t Sh3, svuint8_t Dsb)
{
    const svuint64_t qs = svld1ub_u64(p64, b->qs + 8 * v);
    const svuint64_t hv = svtbl_u64(QH, svld1_u64(p64, HIDX[v]));
    const svuint64_t idx = svorr_u64_x(p64, qs, svlsl_n_u64_x(p64, svand_n_u64_x(p64, svlsr_u64_x(p64, hv, Sh3), 7), 8));
    const svint8_t g = svreinterpret_s8_u64(svld1_gather_u64index_u64(p64, iq1s_grid, idx));
    const svint8_t ds = svreinterpret_s8_u8(svtbl_u8(Dsb, svld1_u8(p8, DIDX[v])));
    return svadd_s8_x(p8, svlsl_n_s8_x(p8, g, 3), ds);                     /* 8 (g + delta) = 8g +- 1 */
}

#define IQ1_TOK(k) IQ2_TOK(k)

static inline __attribute__((always_inline)) void iq1s_row(int nb, const blk_iq1s *wr, const q23s_act *const *acts,
                                                           const int NT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p64 = svptrue_b64(), p32h = svwhilelt_b32_s32(0, 8);
    const svuint64_t Sh3 = svld1_u64(p64, SH3);
    const svuint32_t L0 = svld1_u32(p32, LSIDX[0]), L1 = svld1_u32(p32, LSIDX[1]), L2 = svld1_u32(p32, LSIDX[2]), L3 = svld1_u32(p32, LSIDX[3]);
    const svint32_t z = svdup_s32(0);
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0;
    for (int i = 0; i < nb; ++i) {
        const blk_iq1s *b = wr + i;
        const svuint64_t QH = svld1uh_u64(p64, b->qh);
        const svuint32_t qh32 = svld1uh_u32(p32h, b->qh);
        const svint32_t ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, svand_n_u32_x(p32, svlsr_n_u32_x(p32, qh32, 12), 7), 1), 1));
        const svint32_t dsg = svsub_s32_x(p32, svdup_s32(1), svreinterpret_s32_u32(svlsl_n_u32_x(p32, svlsr_n_u32_x(p32, qh32, 15), 1)));
        const svuint8_t Dsb = svreinterpret_u8_s32(dsg);
        const svint32_t S0 = svtbl_s32(ls, L0), S1 = svtbl_s32(ls, L1), S2 = svtbl_s32(ls, L2), S3 = svtbl_s32(ls, L3);
        const svint8_t W0 = iq1_plane(p8, p64, b, 0, QH, Sh3, Dsb), W1 = iq1_plane(p8, p64, b, 1, QH, Sh3, Dsb);
        const svint8_t W2 = iq1_plane(p8, p64, b, 2, QH, Sh3, Dsb), W3 = iq1_plane(p8, p64, b, 3, QH, Sh3, Dsb);
        const float d = h2f(b->d) * 0.125f;
        IQ1_TOK(0) IQ1_TOK(1) IQ1_TOK(2) IQ1_TOK(3)
    }
    out[0] = svaddv_f32(p32, A0);
    if (NT > 1) out[1] = svaddv_f32(p32, A1);
    if (NT > 2) out[2] = svaddv_f32(p32, A2);
    if (NT > 3) out[3] = svaddv_f32(p32, A3);
}

void iq1s_sdot_row(int cols, const void *w, const q23s_act *const *acts, int nt, float *out)
{
    init_tables();
    const int nb = cols / 256; const blk_iq1s *b = (const blk_iq1s *) w;
    switch (nt) { case 1: iq1s_row(nb, b, acts, 1, out); break; case 2: iq1s_row(nb, b, acts, 2, out); break;
                  case 3: iq1s_row(nb, b, acts, 3, out); break; default: iq1s_row(nb, b, acts, 4, out); }
}
