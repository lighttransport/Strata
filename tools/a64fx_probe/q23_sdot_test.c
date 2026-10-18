/* tools/a64fx_probe/q23_sdot_test.c - every SDOT kernel (q23_sdot_sve.c, iq_sdot_sve.c) against a scalar double
 * reference written from ggml's dequantize_row_{q2_K,q3_K,iq2_xxs,iq1_s}, on random blocks and random Q8_K
 * activations, 1..4 tokens, one and two rows. Needs only ggml-common.h (grid tables), so it also runs under
 * qemu-aarch64 (-cpu max,sve512=on) on an x86 host:
 *   clang --target=aarch64-linux-gnu -fuse-ld=lld -march=armv8.2-a+sve -O2 -I<ggml-common.h dir> \
 *       q23_sdot_test.c q23_sdot_sve.c iq_sdot_sve.c -lm -o t && qemu-aarch64 -cpu max,sve512=on -L /usr/aarch64-linux-gnu t */
#include "q23_sdot_sve.h"

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } blk_q8k;

static uint64_t S = 0x1234567;
static uint64_t rnd(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return S; }
static uint16_t f2h(float f) { __fp16 h = (__fp16) f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float) h; }

static void deq_q2k(const uint8_t *b, double *y)          /* ggml dequantize_row_q2_K, one block */
{
    const uint8_t *sc = b, *q = b + 16; uint16_t hd, hm; memcpy(&hd, b + 80, 2); memcpy(&hm, b + 82, 2);
    const float d = h2f(hd), mn = h2f(hm); int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t s = sc[is++]; float dl = d * (s & 0xF), ml = mn * (s >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l] >> shift) & 3) - ml;
            s = sc[is++]; dl = d * (s & 0xF); ml = mn * (s >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
        }
        q += 32;
    }
}

static void deq_q3k(const uint8_t *b, double *y)          /* ggml dequantize_row_q3_K, one block */
{
    const uint8_t *hm = b, *q = b + 32; uint16_t hd; memcpy(&hd, b + 108, 2); const float d_all = h2f(hd);
    uint32_t aux[4] = {0}; memcpy(aux, b + 96, 12);
    const uint32_t k1 = 0x03030303, k2 = 0x0f0f0f0f, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    const int8_t *scales = (const int8_t *) aux; int is = 0; uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2; m <<= 1;
        }
        q += 32;
    }
}

static void deq_iq2xxs(const uint8_t *b, double *y)       /* ggml dequantize_row_iq2_xxs, one block */
{
    uint16_t hd; memcpy(&hd, b, 2); const float d = h2f(hd);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t aux32[2]; memcpy(aux32, b + 2 + 8 * ib32, 8); const uint8_t *aux8 = (const uint8_t *) aux32;
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t *grid = (const uint8_t *) (iq2xxs_grid + aux8[l]);
            const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
            for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}

static void deq_iq1s(const uint8_t *b, double *y)          /* ggml dequantize_row_iq1_s, one block */
{
    uint16_t hd; memcpy(&hd, b, 2); const float d = h2f(hd);
    const uint8_t *qs = b + 2; uint16_t qh[8]; memcpy(qh, b + 34, 16);
    for (int ib = 0; ib < 8; ++ib) {
        const float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
        const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 4; ++l) {
            const int8_t *grid = (const int8_t *) (iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
            for (int j = 0; j < 8; ++j) y[j] = dl * (grid[j] + delta);
            y += 8;
        }
        qs += 4;
    }
}

/* kernel ids: 0 q2 v1, 1 q3 v1, 2 q2 v2 (1 row), 3 q3 v2 (1 row), 4 q2 v2 (2 rows), 5 q3 v2 (2 rows),
 * 6 iq2xxs var0, 7 iq2xxs var1, 8 iq1s */
static const char *NAMES[] = {"q2_K v1", "q3_K v1", "q2_K v2 1row", "q3_K v2 1row", "q2_K v2 2rows", "q3_K v2 2rows",
                              "iq2_xxs regsign", "iq2_xxs gathersign", "iq1_s", "iq2_xxs tbl", "iq2_xxs rp", "iq2_xxs rpc",
                              "iq1_s rpc"};

int main(void)
{
    enum { COLS = 4096, NB = COLS / 256, ROWS = 32 };
    int fails = 0;
    static blk_q8k a8[4][NB]; static q23s_act as[4][NB]; static double xf[4][COLS];
    for (int t = 0; t < 4; ++t) {
        for (int i = 0; i < NB; ++i) {
            a8[t][i].d = 0.01f + (rnd() % 1000) * 1e-5f;
            for (int g = 0; g < 16; ++g) {
                int s = 0;
                for (int l = 0; l < 16; ++l) { int8_t v = (int8_t) ((int) (rnd() % 255) - 127); a8[t][i].qs[16 * g + l] = v; s += v; }
                a8[t][i].bsums[g] = (int16_t) s;
            }
            for (int k = 0; k < 256; ++k) xf[t][256 * i + k] = (double) a8[t][i].d * a8[t][i].qs[k];
        }
        q23s_prepare(a8[t], COLS, as[t]);
    }
    static q23s_act an[4][NB], aq[4][NB];
    for (int t = 0; t < 4; ++t) { q23s_prepare_natural(a8[t], COLS, an[t]); q23s_prepare_quad(a8[t], COLS, aq[t]); }
    const q23s_act *ap[4] = {as[0], as[1], as[2], as[3]}, *apn[4] = {an[0], an[1], an[2], an[3]}, *apq[4] = {aq[0], aq[1], aq[2], aq[3]};
    static uint8_t rp[NB * 106];
    for (int kid = 0; kid < 13; ++kid) {
        const int fmt = (kid == 8 || kid == 12) ? 3 : (kid >= 6) ? 2 : (kid & 1);      /* 0 q2 1 q3 2 iq2xxs 3 iq1s */
        const size_t bs = fmt == 0 ? 84 : fmt == 1 ? 110 : fmt == 2 ? 66 : 50;
        const int rows_per_call = (kid == 4 || kid == 5) ? 2 : 1;
        uint8_t *w = malloc(bs * NB * ROWS); double y[256];
        for (size_t i = 0; i < bs * NB * ROWS; ++i) w[i] = (uint8_t) rnd();
        for (int r = 0; r < ROWS * NB; ++r) {
            uint8_t *b = w + (size_t) r * bs; const float d = 0.002f + (rnd() % 1000) * 1e-5f;
            if (fmt == 0) { uint16_t h = f2h(d), h2 = f2h(d * 0.5f); memcpy(b + 80, &h, 2); memcpy(b + 82, &h2, 2); }
            else if (fmt == 1) { uint16_t h = f2h(d); memcpy(b + 108, &h, 2); }
            else { uint16_t h = f2h(d); memcpy(b, &h, 2); }
        }
        for (int nt = 1; nt <= 4; ++nt) {
            double num = 0, den = 0;
            for (int r = 0; r < ROWS; r += rows_per_call) {
                const uint8_t *row = w + (size_t) r * NB * bs; float got[8];
                switch (kid) {
                case 0: q2k_sdot_row(COLS, row, ap, nt, got); break;
                case 1: q3k_sdot_row(COLS, row, ap, nt, got); break;
                case 2: q2k_sdot_rows_v2(COLS, row, NB * bs, 1, ap, nt, got); break;
                case 3: q3k_sdot_rows_v2(COLS, row, NB * bs, 1, ap, nt, got); break;
                case 4: q2k_sdot_rows_v2(COLS, row, NB * bs, 2, ap, nt, got); break;
                case 5: q3k_sdot_rows_v2(COLS, row, NB * bs, 2, ap, nt, got); break;
                case 6: iq2xxs_sdot_row(COLS, row, apn, nt, got, 0); break;
                case 7: iq2xxs_sdot_row(COLS, row, apn, nt, got, 1); break;
                case 8: iq1s_sdot_row(COLS, row, apn, nt, got); break;
                case 9: iq2xxs_sdot_row_tbl(COLS, row, apq, nt, got, 0); break;
                case 10: iq2xxs_repack(row, rp, NB, 1); iq2xxs_sdot_row_tbl(COLS, rp, apq, nt, got, 1); break;
                case 11: iq2xxs_repack(row, rp, NB, 2); iq2xxs_sdot_row_tbl(COLS, rp, apq, nt, got, 2); break;
                default: iq1s_repack(row, rp, NB); iq1s_sdot_row_rpc(COLS, rp, apq, nt, got); break;
                }
                for (int rr = 0; rr < rows_per_call; ++rr)
                    for (int t = 0; t < nt; ++t) {
                        const uint8_t *rw = row + (size_t) rr * NB * bs; double ref = 0;
                        for (int i = 0; i < NB; ++i) {
                            if (fmt == 0) deq_q2k(rw + i * bs, y); else if (fmt == 1) deq_q3k(rw + i * bs, y);
                            else if (fmt == 2) deq_iq2xxs(rw + i * bs, y); else deq_iq1s(rw + i * bs, y);
                            for (int k = 0; k < 256; ++k) ref += y[k] * xf[t][256 * i + k];
                        }
                        const double g = got[rr * nt + t];
                        num += (g - ref) * (g - ref); den += ref * ref;
                    }
            }
            const double rel = sqrt(num / den); const int ok = rel < 1e-5;
            printf("%-18s tokens=%d rel %.2e %s\n", NAMES[kid], nt, rel, ok ? "PASS" : "FAIL");
            fails += !ok;
        }
        free(w);
    }
    return fails != 0;
}
