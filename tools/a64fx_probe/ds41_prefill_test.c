/* tools/a64fx_probe/ds41_prefill_test.c - 16-token IQ2P tiles against the established
 * <=8-token kernel, bit for bit for nt=1..64; partial row ranges and output padding
 * are checked too. Single-thread timings alternate tile8/tile16 in two rounds.
 * Build: clang -O3 -mcpu=a64fx -msve-vector-bits=512 -fopenmp -Iinclude
 *        src/kernels/cpu/ds41_sve.c tools/a64fx_probe/ds41_prefill_test.c -lm -lpthread */
#include "strata/kernels/cpu/ds41_sve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned long long S = 88172645463325252ull;
static unsigned rnd(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return (unsigned) (S >> 16); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint16_t h16(float f) { __fp16 h = (__fp16) f; uint16_t u; memcpy(&u, &h, 2); return u; }

/* Compare each 16-token tile with the established <=8-token kernel, including
 * nonzero row ranges and padded token strides. nt=1..64 covers every tail. */
static void rows(int tile, const void *w, int cols, int r0, int r1,
                 const void *const *a, int nt, float *y, int stride)
{
    for (int g = 0; g < nt; g += tile)
        ds41_rows(DS41_T_IQ2P, w, cols, r0, r1, a + g,
                  nt - g < tile ? nt - g : tile, y + (size_t) g * stride, stride);
}
int main(void)
{
    int fails = 0;
    const int shapes[][2] = {{37, 256}, {37, 5120}, {256, 5120}, {512, 256}};
    for (unsigned si = 0; si < sizeof shapes / sizeof shapes[0]; ++si) {
        const int nr = shapes[si][0], cols = shapes[si][1], stride = nr + 7;
        const size_t rb = ds41_row_bytes(DS41_T_IQ2P, cols);
        uint8_t *w = malloc(rb * nr);
        for (size_t j = 0; j < rb * nr; ++j) w[j] = (uint8_t) rnd();
        for (size_t j = 0; j < rb * nr; j += 106) {
            uint16_t d = h16(0.001f + 0.01f * (rnd() % 1000) / 1000.f);
            memcpy(w + j, &d, 2);
        }
        ds41_q8k *a[64];
        float *x = malloc(sizeof(float) * cols);
        for (int t = 0; t < 64; ++t) {
            for (int c = 0; c < cols; ++c) x[c] = (rnd() % 20001) / 10000.f - 1.f;
            a[t] = aligned_alloc(64, sizeof(ds41_q8k) * (cols / 256));
            ds41_quant_q8k(x, cols, a[t]);
        }
        float *ref = malloc(sizeof(float) * 64 * stride), *test = malloc(sizeof(float) * 64 * stride);
        int bad = 0;
        for (int nt = 1; nt <= 64; ++nt) {
            for (int r0 = 0; r0 <= 3; r0 += 3) {
                const int r1 = nr - r0;
                for (int j = 0; j < 64 * stride; ++j) ref[j] = test[j] = -12345.f;
                rows(8, w, cols, r0, r1, (const void *const *) a, nt, ref, stride);
                rows(16, w, cols, r0, r1, (const void *const *) a, nt, test, stride);
                if (memcmp(ref, test, sizeof(float) * 64 * stride)) {
                    printf("MISMATCH rows=%d cols=%d nt=%d r0=%d\n", nr, cols, nt, r0);
                    ++bad;
                }
            }
        }
        printf("TILE16 rows=%d cols=%d nt=1..64 fails=%d\n", nr, cols, bad);
        fails += bad;
        for (int nt = 16; nt <= 64; nt *= 2) {
            double elapsed[2] = {0};
            const int reps = 8;
            for (int round = 0; round < 2; ++round) {
                for (int which = 0; which < 2; ++which) {
                    const int k = which ^ round;
                    double start = now();
                    for (int rep = 0; rep < reps; ++rep)
                        rows(k ? 16 : 8, w, cols, 0, nr, (const void *const *) a, nt, test, stride);
                    elapsed[k] += now() - start;
                }
            }
            printf("TILE16_RATE rows=%d cols=%d nt=%d tile8_ms=%.3f tile16_ms=%.3f speedup=%.3f\n",
                   nr, cols, nt, 1e3 * elapsed[0] / (2 * reps), 1e3 * elapsed[1] / (2 * reps), elapsed[0] / elapsed[1]);
        }
        fflush(stdout);
        for (int t = 0; t < 64; ++t) free(a[t]);
        free(w); free(x); free(ref); free(test);
    }
    printf("PREFILL_KERNEL fails=%d\n", fails);
    return fails != 0;
}
