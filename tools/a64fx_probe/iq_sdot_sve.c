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

/* ================================================================================================ gather-free */
typedef struct { uint16_t d; uint8_t ls[8]; uint8_t idx[32]; uint8_t sgn[32]; } iq2rp;     /* 74 B */
typedef struct { uint16_t d; uint8_t ls[8]; uint8_t sgn[32]; uint8_t codes[64]; } iq2rpc;  /* 106 B */
typedef struct { uint16_t d; uint8_t ls[8]; int8_t dsg[8]; uint8_t codes[64]; } iq1rpc;    /* 82 B */
_Static_assert(sizeof(iq2rp) == IQ2XXS_RP_BYTES, "iq2rp");
_Static_assert(sizeof(iq2rpc) == IQ2XXS_RPC_BYTES, "iq2rpc");
_Static_assert(sizeof(iq1rpc) == IQ1S_RPC_BYTES, "iq1rpc");

/* T2[k][h][m]: the 2-bit codes (8 -> 0, 25 -> 1, 43 -> 2) of grid entry 64 k + m, weights 4 h .. 4 h + 3 */
static uint8_t T2[4][2][64], LIDX2[64], SLO[64], SHI[64], SHR[64], SHL[64], KSLO[64], KSHI[64], H4[64], HALF[64],
               SUB8[64], VAL2[64], VAL1[64];
static uint32_t SL2[16];
static int ready2;

static uint8_t code_byte(uint64_t entry, int h, int ternary)
{
    uint8_t c = 0;
    for (int j = 0; j < 4; ++j) {
        const int8_t v = (int8_t) (entry >> (8 * (4 * h + j)));
        const int code = ternary ? v + 1 : (v == 8 ? 0 : v == 25 ? 1 : 2);
        c |= (uint8_t) (code << (2 * j));
    }
    return c;
}

static void init_tables2(void)
{
    if (ready2) return;
    init_tables();
    for (int k = 0; k < 4; ++k)
        for (int h = 0; h < 2; ++h)
            for (int m = 0; m < 64; ++m) T2[k][h][m] = code_byte(iq2xxs_grid[64 * k + m], h, 0);
    static const int lo_off[4] = {0, 0, 1, 2}, rsh[4] = {0, 7, 6, 5};
    for (int L = 0; L < 64; ++L) {
        const int g = L >> 1, sb = g >> 2, ll = g & 3;
        LIDX2[L] = (uint8_t) (8 * sb + ll);
        SLO[L] = (uint8_t) (8 * sb + 4 + lo_off[ll]); SHI[L] = (uint8_t) (SLO[L] + 1);
        SHR[L] = (uint8_t) rsh[ll]; SHL[L] = (uint8_t) (8 - rsh[ll]);
        KSLO[L] = ksigns_iq2xs[L]; KSHI[L] = ksigns_iq2xs[64 + L];
        H4[L] = (uint8_t) ((L & 1) * 4); HALF[L] = (uint8_t) g; SUB8[L] = (uint8_t) (L >> 3);
        VAL2[L] = L == 0 ? 8 : L == 1 ? 25 : L == 2 ? 43 : 0;
        VAL1[L] = (uint8_t) (L == 0 ? -8 : L == 2 ? 8 : 0);
    }
    for (int k = 0; k < 16; ++k) SL2[k] = (uint32_t) (k >> 1);
    ready2 = 1;
}

void iq2xxs_repack(const void *src, void *dst, int nblocks, int layout)
{
    init_tables2();
    const blk_iq2xxs *b = (const blk_iq2xxs *) src;
    for (int i = 0; i < nblocks; ++i) {
        uint8_t ls[8], idx[32], sgn[32];
        for (int sb = 0; sb < 8; ++sb) {
            uint32_t aux[2]; memcpy(aux, b[i].qs + 8 * sb, 8);
            ls[sb] = (uint8_t) (2 * (aux[1] >> 28) + 1);
            for (int l = 0; l < 4; ++l) {
                idx[4 * sb + l] = (uint8_t) (aux[0] >> (8 * l));
                sgn[4 * sb + l] = ksigns_iq2xs[(aux[1] >> (7 * l)) & 127];
            }
        }
        if (layout == 1) {
            iq2rp *o = (iq2rp *) dst + i; o->d = b[i].d;
            memcpy(o->ls, ls, 8); memcpy(o->idx, idx, 32); memcpy(o->sgn, sgn, 32);
        } else {
            iq2rpc *o = (iq2rpc *) dst + i; o->d = b[i].d;
            memcpy(o->ls, ls, 8); memcpy(o->sgn, sgn, 32);
            for (int L = 0; L < 64; ++L) o->codes[L] = code_byte(iq2xxs_grid[idx[L >> 1]], L & 1, 0);
        }
    }
}

void iq1s_repack(const void *src, void *dst, int nblocks)
{
    init_tables2();
    const blk_iq1s *b = (const blk_iq1s *) src;
    for (int i = 0; i < nblocks; ++i) {
        iq1rpc *o = (iq1rpc *) dst + i; o->d = b[i].d;
        for (int sb = 0; sb < 8; ++sb) {
            const uint16_t qh = b[i].qh[sb];
            o->ls[sb] = (uint8_t) (2 * ((qh >> 12) & 7) + 1);
            o->dsg[sb] = (int8_t) (qh & 0x8000 ? -1 : 1);
        }
        for (int L = 0; L < 64; ++L) {
            const int g = L >> 1, sb = g >> 2, l = g & 3;
            const int gi = b[i].qs[g] | (((b[i].qh[sb] >> (3 * l)) & 7) << 8);
            o->codes[L] = code_byte(iq1s_grid[gi], L & 1, 1);
        }
    }
}

/* the shared tail: 2-bit codes r (byte L = weights 4L..4L+3), sign nibbles (or none), int32 scale lanes S */
#define QUAD_TOK(k)                                                                                     \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        const svint32_t v = svadd_s32_x(p32, svdot_s32(svdot_s32(z, W0, svld1_s8(p8, x->x)), W1, svld1_s8(p8, x->x + 64)), \
                                              svdot_s32(svdot_s32(z, W2, svld1_s8(p8, x->x + 128)), W3, svld1_s8(p8, x->x + 192))); \
        A##k = svmla_n_f32_x(p32, A##k, svcvt_f32_s32_x(p32, svmul_s32_x(p32, v, S)), d * x->d);        \
    }

static inline __attribute__((always_inline)) svint8_t iq2_signed(svbool_t p8, svuint8_t c, svuint8_t nib, int j, svuint8_t Val)
{
    const svint8_t v = svreinterpret_s8_u8(svtbl_u8(Val, c));
    const svbool_t neg = svcmpne_n_u8(p8, svand_n_u8_x(p8, nib, (uint8_t) (1u << j)), 0);
    return svneg_s8_m(v, neg, v);
}

static inline __attribute__((always_inline)) void iq2xxs_tbl_row(int nb, const void *wr, const q23s_act *const *acts,
                                                                 const int NT, const int LAYOUT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p8l = svwhilelt_b32_s32(0, 8);
    const svbool_t podd = svcmpne_n_u8(p8, svand_n_u8_x(p8, svindex_u8(0, 1), 1), 0);
    const svuint8_t Val = svld1_u8(p8, VAL2), H4v = svld1_u8(p8, H4);
    const svuint32_t Sl2 = svld1_u32(p32, SL2);
    const svint32_t z = svdup_s32(0);
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0;
    for (int i = 0; i < nb; ++i) {
        svuint8_t r, nib; svint32_t ls; float d;
        if (LAYOUT == 2) {
            const iq2rpc *b = (const iq2rpc *) wr + i;
            r = svld1_u8(p8, b->codes);
            nib = svlsr_u8_x(p8, svtbl_u8(svld1_u8(svwhilelt_b8_s32(0, 32), b->sgn), svld1_u8(p8, HALF)), H4v);
            ls = svreinterpret_s32_u32(svld1ub_u32(p8l, b->ls));
            d = h2f(b->d) * 0.125f;
        } else {
            svuint8_t idx, s8;
            if (LAYOUT == 1) {
                const iq2rp *b = (const iq2rp *) wr + i;
                const svbool_t p32b = svwhilelt_b8_s32(0, 32);
                const svuint8_t half = svld1_u8(p8, HALF);
                idx = svtbl_u8(svld1_u8(p32b, b->idx), half);
                s8 = svtbl_u8(svld1_u8(p32b, b->sgn), half);
                ls = svreinterpret_s32_u32(svld1ub_u32(p8l, b->ls));
                d = h2f(b->d) * 0.125f;
            } else {
                const blk_iq2xxs *b = (const blk_iq2xxs *) wr + i;
                const svuint8_t Q = svld1_u8(p8, b->qs);
                idx = svtbl_u8(Q, svld1_u8(p8, LIDX2));
                const svuint8_t s7 = svand_n_u8_x(p8, svorr_u8_x(p8,
                    svlsr_u8_x(p8, svtbl_u8(Q, svld1_u8(p8, SLO)), svld1_u8(p8, SHR)),
                    svlsl_u8_x(p8, svtbl_u8(Q, svld1_u8(p8, SHI)), svld1_u8(p8, SHL))), 127);
                s8 = svorr_u8_x(p8, svtbl_u8(svld1_u8(p8, KSLO), s7), svtbl_u8(svld1_u8(p8, KSHI), svsub_n_u8_x(p8, s7, 64)));
                const svuint32_t s4 = svlsr_n_u32_x(p32, svreinterpret_u32_u8(svtbl_u8(Q, svld1_u8(p8, SCIDX))), 4);
                ls = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s4, 1), 1));
                d = h2f(b->d) * 0.125f;
            }
            /* 256-entry grid as 2-bit codes: OR of four 64-entry svtbl per half (out-of-range lanes read 0) */
            svuint8_t r0 = svtbl_u8(svld1_u8(p8, T2[0][0]), idx), r1 = svtbl_u8(svld1_u8(p8, T2[0][1]), idx);
            for (int k = 1; k < 4; ++k) {
                const svuint8_t ik = svsub_n_u8_x(p8, idx, (uint8_t) (64 * k));
                r0 = svorr_u8_x(p8, r0, svtbl_u8(svld1_u8(p8, T2[k][0]), ik));
                r1 = svorr_u8_x(p8, r1, svtbl_u8(svld1_u8(p8, T2[k][1]), ik));
            }
            r = svsel_u8(podd, r1, r0);
            nib = svlsr_u8_x(p8, s8, H4v);
        }
        const svint32_t S = svtbl_s32(ls, Sl2);
        const svint8_t W0 = iq2_signed(p8, svand_n_u8_x(p8, r, 3), nib, 0, Val);
        const svint8_t W1 = iq2_signed(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, r, 2), 3), nib, 1, Val);
        const svint8_t W2 = iq2_signed(p8, svand_n_u8_x(p8, svlsr_n_u8_x(p8, r, 4), 3), nib, 2, Val);
        const svint8_t W3 = iq2_signed(p8, svlsr_n_u8_x(p8, r, 6), nib, 3, Val);
        QUAD_TOK(0) QUAD_TOK(1) QUAD_TOK(2) QUAD_TOK(3)
    }
    out[0] = svaddv_f32(p32, A0);
    if (NT > 1) out[1] = svaddv_f32(p32, A1);
    if (NT > 2) out[2] = svaddv_f32(p32, A2);
    if (NT > 3) out[3] = svaddv_f32(p32, A3);
}

#define DISPATCH_NT(fn, ...)                                                                            \
    switch (nt) { case 1: fn(__VA_ARGS__, 1, LAYOUT_ARG out); break; case 2: fn(__VA_ARGS__, 2, LAYOUT_ARG out); break; \
                  case 3: fn(__VA_ARGS__, 3, LAYOUT_ARG out); break; default: fn(__VA_ARGS__, 4, LAYOUT_ARG out); }

void iq2xxs_sdot_row_tbl(int cols, const void *w, const q23s_act *const *acts, int nt, float *out, int layout)
{
    init_tables2();
    const int nb = cols / 256;
#define LAYOUT_ARG 0,
    if (layout == 0) { DISPATCH_NT(iq2xxs_tbl_row, nb, w, acts) }
#undef LAYOUT_ARG
#define LAYOUT_ARG 1,
    else if (layout == 1) { DISPATCH_NT(iq2xxs_tbl_row, nb, w, acts) }
#undef LAYOUT_ARG
#define LAYOUT_ARG 2,
    else { DISPATCH_NT(iq2xxs_tbl_row, nb, w, acts) }
#undef LAYOUT_ARG
}

static inline __attribute__((always_inline)) void iq1s_rpc_row(int nb, const iq1rpc *wr, const q23s_act *const *acts,
                                                               const int NT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p8l = svwhilelt_b32_s32(0, 8), p8b = svwhilelt_b8_s32(0, 8);
    const svuint8_t Val = svld1_u8(p8, VAL1), Sub8 = svld1_u8(p8, SUB8);
    const svuint32_t Sl2 = svld1_u32(p32, SL2);
    const svint32_t z = svdup_s32(0);
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0;
    for (int i = 0; i < nb; ++i) {
        const iq1rpc *b = wr + i;
        const svuint8_t r = svld1_u8(p8, b->codes);
        const svint8_t ds = svreinterpret_s8_u8(svtbl_u8(svld1_u8(p8b, (const uint8_t *) b->dsg), Sub8));
        const svint32_t S = svtbl_s32(svreinterpret_s32_u32(svld1ub_u32(p8l, b->ls)), Sl2);
        const float d = h2f(b->d) * 0.125f;
        /* 8 (g + delta) = 8 g +- 1: value table {-8, 0, 8} plus the sub-block's delta sign */
        const svint8_t W0 = svadd_s8_x(p8, svreinterpret_s8_u8(svtbl_u8(Val, svand_n_u8_x(p8, r, 3))), ds);
        const svint8_t W1 = svadd_s8_x(p8, svreinterpret_s8_u8(svtbl_u8(Val, svand_n_u8_x(p8, svlsr_n_u8_x(p8, r, 2), 3))), ds);
        const svint8_t W2 = svadd_s8_x(p8, svreinterpret_s8_u8(svtbl_u8(Val, svand_n_u8_x(p8, svlsr_n_u8_x(p8, r, 4), 3))), ds);
        const svint8_t W3 = svadd_s8_x(p8, svreinterpret_s8_u8(svtbl_u8(Val, svlsr_n_u8_x(p8, r, 6))), ds);
        QUAD_TOK(0) QUAD_TOK(1) QUAD_TOK(2) QUAD_TOK(3)
    }
    out[0] = svaddv_f32(p32, A0);
    if (NT > 1) out[1] = svaddv_f32(p32, A1);
    if (NT > 2) out[2] = svaddv_f32(p32, A2);
    if (NT > 3) out[3] = svaddv_f32(p32, A3);
}

void iq1s_sdot_row_rpc(int cols, const void *w, const q23s_act *const *acts, int nt, float *out)
{
    init_tables2();
    const int nb = cols / 256; const iq1rpc *b = (const iq1rpc *) w;
    switch (nt) { case 1: iq1s_rpc_row(nb, b, acts, 1, out); break; case 2: iq1s_rpc_row(nb, b, acts, 2, out); break;
                  case 3: iq1s_rpc_row(nb, b, acts, 3, out); break; default: iq1s_rpc_row(nb, b, acts, 4, out); }
}

/* ===================================================================================== v3: pipelined gathers */
#define IQV3_TOK(k)                                                                                     \
    if (NT > k) {                                                                                       \
        const q23s_act *x = a##k + i;                                                                   \
        const svint32_t v = svmla_s32_x(p32, svmul_s32_x(p32, svdot_s32(z, c0, svld1_s8(p8, x->x)), svtbl_s32(cls, L0)), \
                                        svdot_s32(z, c1, svld1_s8(p8, x->x + 64)), svtbl_s32(cls, L1)); \
        const svint32_t u = svmla_s32_x(p32, svmul_s32_x(p32, svdot_s32(z, c2, svld1_s8(p8, x->x + 128)), svtbl_s32(cls, L2)), \
                                        svdot_s32(z, c3, svld1_s8(p8, x->x + 192)), svtbl_s32(cls, L3)); \
        A##k = svmla_f32_x(p32, A##k, svcvt_f32_s32_x(p32, svadd_s32_x(p32, v, u)), svmul_f32_x(p32, cd, svdup_n_f32(x->d))); \
    }

#define IQV3_STORE                                                                                      \
    out[0] = svaddv_f32(p32, A0);                                                                       \
    if (NT > 1) out[1] = svaddv_f32(p32, A1);                                                           \
    if (NT > 2) out[2] = svaddv_f32(p32, A2);                                                           \
    if (NT > 3) out[3] = svaddv_f32(p32, A3);

static inline __attribute__((always_inline)) svfloat32_t h2v8(svbool_t p32, const uint16_t *h)
{
    return svmul_n_f32_x(p32, svcvt_f32_f16_x(p32, svreinterpret_f16_u16(svdup_n_u16(*h))), 0.125f);
}

#define IQ2V3_DEC(b, W0_, W1_, W2_, W3_, LS_, D_)                                                      \
    {                                                                                                   \
        const svuint8_t Q = svld1_u8(p8, (b)->qs);                                                      \
        const svuint32_t s4 = svlsr_n_u32_x(p32, svreinterpret_u32_u8(svtbl_u8(Q, Sci)), 4);            \
        LS_ = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, s4, 1), 1));                  \
        W0_ = iq2_plane(p8, p64, Q, 0, 1, Rep, Bits); W1_ = iq2_plane(p8, p64, Q, 1, 1, Rep, Bits);     \
        W2_ = iq2_plane(p8, p64, Q, 2, 1, Rep, Bits); W3_ = iq2_plane(p8, p64, Q, 3, 1, Rep, Bits);     \
        D_ = h2v8(p32, &(b)->d);                                                                        \
    }

static inline __attribute__((always_inline)) void iq2xxs_v3(int nb, const blk_iq2xxs *w, const q23s_act *const *acts,
                                                            const int NT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p64 = svptrue_b64();
    const svuint8_t Rep = svld1_u8(p8, REP), Bits = svld1_u8(p8, BITS), Sci = svld1_u8(p8, SCIDX);
    const svuint32_t L0 = svld1_u32(p32, LSIDX[0]), L1 = svld1_u32(p32, LSIDX[1]), L2 = svld1_u32(p32, LSIDX[2]), L3 = svld1_u32(p32, LSIDX[3]);
    const svint32_t z = svdup_s32(0);
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0;
    svint8_t c0, c1, c2, c3; svint32_t cls; svfloat32_t cd;
    IQ2V3_DEC(w, c0, c1, c2, c3, cls, cd)
    for (int i = 0; i < nb; ++i) {
        svint8_t n0 = c0, n1 = c1, n2 = c2, n3 = c3; svint32_t nls = cls; svfloat32_t nd = cd;
        if (i + 1 < nb) IQ2V3_DEC(w + i + 1, n0, n1, n2, n3, nls, nd)
        IQV3_TOK(0) IQV3_TOK(1) IQV3_TOK(2) IQV3_TOK(3)
        c0 = n0; c1 = n1; c2 = n2; c3 = n3; cls = nls; cd = nd;
    }
    IQV3_STORE
}

void iq2xxs_sdot_row_v3(int cols, const void *w, const q23s_act *const *acts, int nt, float *out)
{
    init_tables();
    const int nb = cols / 256; const blk_iq2xxs *b = (const blk_iq2xxs *) w;
    switch (nt) { case 1: iq2xxs_v3(nb, b, acts, 1, out); break; case 2: iq2xxs_v3(nb, b, acts, 2, out); break;
                  case 3: iq2xxs_v3(nb, b, acts, 3, out); break; default: iq2xxs_v3(nb, b, acts, 4, out); }
}

#define IQ1V3_DEC(b, W0_, W1_, W2_, W3_, LS_, D_)                                                      \
    {                                                                                                   \
        const svuint64_t QH = svld1uh_u64(p64, (b)->qh);                                                \
        const svuint32_t qh32 = svld1uh_u32(p32h, (b)->qh);                                             \
        LS_ = svreinterpret_s32_u32(svadd_n_u32_x(p32, svlsl_n_u32_x(p32, svand_n_u32_x(p32, svlsr_n_u32_x(p32, qh32, 12), 7), 1), 1)); \
        const svuint8_t Dsb = svreinterpret_u8_s32(svsub_s32_x(p32, svdup_s32(1),                       \
                                  svreinterpret_s32_u32(svlsl_n_u32_x(p32, svlsr_n_u32_x(p32, qh32, 15), 1)))); \
        W0_ = iq1_plane(p8, p64, b, 0, QH, Sh3, Dsb); W1_ = iq1_plane(p8, p64, b, 1, QH, Sh3, Dsb);     \
        W2_ = iq1_plane(p8, p64, b, 2, QH, Sh3, Dsb); W3_ = iq1_plane(p8, p64, b, 3, QH, Sh3, Dsb);     \
        D_ = h2v8(p32, &(b)->d);                                                                        \
    }

static inline __attribute__((always_inline)) void iq1s_v3(int nb, const blk_iq1s *w, const q23s_act *const *acts,
                                                          const int NT, float *out)
{
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p64 = svptrue_b64(), p32h = svwhilelt_b32_s32(0, 8);
    const svuint64_t Sh3 = svld1_u64(p64, SH3);
    const svuint32_t L0 = svld1_u32(p32, LSIDX[0]), L1 = svld1_u32(p32, LSIDX[1]), L2 = svld1_u32(p32, LSIDX[2]), L3 = svld1_u32(p32, LSIDX[3]);
    const svint32_t z = svdup_s32(0);
    const q23s_act *a0 = acts[0], *a1 = NT > 1 ? acts[1] : acts[0], *a2 = NT > 2 ? acts[2] : acts[0], *a3 = NT > 3 ? acts[3] : acts[0];
    svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0;
    svint8_t c0, c1, c2, c3; svint32_t cls; svfloat32_t cd;
    IQ1V3_DEC(w, c0, c1, c2, c3, cls, cd)
    for (int i = 0; i < nb; ++i) {
        svint8_t n0 = c0, n1 = c1, n2 = c2, n3 = c3; svint32_t nls = cls; svfloat32_t nd = cd;
        if (i + 1 < nb) IQ1V3_DEC(w + i + 1, n0, n1, n2, n3, nls, nd)
        IQV3_TOK(0) IQV3_TOK(1) IQV3_TOK(2) IQV3_TOK(3)
        c0 = n0; c1 = n1; c2 = n2; c3 = n3; cls = nls; cd = nd;
    }
    IQV3_STORE
}

void iq1s_sdot_row_v3(int cols, const void *w, const q23s_act *const *acts, int nt, float *out)
{
    init_tables();
    const int nb = cols / 256; const blk_iq1s *b = (const blk_iq1s *) w;
    switch (nt) { case 1: iq1s_v3(nb, b, acts, 1, out); break; case 2: iq1s_v3(nb, b, acts, 2, out); break;
                  case 3: iq1s_v3(nb, b, acts, 3, out); break; default: iq1s_v3(nb, b, acts, 4, out); }
}
