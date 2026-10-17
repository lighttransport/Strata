/* tools/a64fx_probe/q23_sdot_test.c - q23_sdot_sve.c against a scalar double reference written from ggml's
 * dequantize_row_q2_K / q3_K, on random blocks and random Q8_K activations, 1..4 tokens. No ggml needed, so it
 * also runs under qemu-aarch64 (-cpu max,sve512=on) on an x86 host:
 *   clang --target=aarch64-linux-gnu -march=armv8.2-a+sve -O2 q23_sdot_test.c q23_sdot_sve.c -o t && qemu-aarch64 -cpu max,sve512=on -L /usr/aarch64-linux-gnu t */
#include "q23_sdot_sve.h"

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
    const q23s_act *ap[4] = {as[0], as[1], as[2], as[3]};
    for (int fmt = 0; fmt < 2; ++fmt) {
        const size_t bs = fmt == 0 ? 84 : 110;
        uint8_t *w = malloc(bs * NB * ROWS); double y[256];
        for (size_t i = 0; i < bs * NB * ROWS; ++i) w[i] = (uint8_t) rnd();
        for (int r = 0; r < ROWS * NB; ++r) {
            uint8_t *b = w + (size_t) r * bs; const float d = 0.002f + (rnd() % 1000) * 1e-5f;
            if (fmt == 0) { uint16_t h = f2h(d), h2 = f2h(d * 0.5f); memcpy(b + 80, &h, 2); memcpy(b + 82, &h2, 2); }
            else { uint16_t h = f2h(d); memcpy(b + 108, &h, 2); }
        }
        for (int nt = 1; nt <= 4; ++nt) {
            double num = 0, den = 0;
            for (int r = 0; r < ROWS; ++r) {
                const uint8_t *row = w + (size_t) r * NB * bs; float got[4];
                (fmt == 0 ? q2k_sdot_row : q3k_sdot_row)(COLS, row, ap, nt, got);
                for (int t = 0; t < nt; ++t) {
                    double ref = 0;
                    for (int i = 0; i < NB; ++i) { (fmt == 0 ? deq_q2k : deq_q3k)(row + i * bs, y); for (int k = 0; k < 256; ++k) ref += y[k] * xf[t][256 * i + k]; }
                    num += (got[t] - ref) * (got[t] - ref); den += ref * ref;
                }
            }
            const double rel = sqrt(num / den); const int ok = rel < 1e-5;
            printf("%s tokens=%d rel %.2e %s\n", fmt == 0 ? "q2_K" : "q3_K", nt, rel, ok ? "PASS" : "FAIL");
            fails += !ok;
        }
        free(w);
    }
    return fails != 0;
}
