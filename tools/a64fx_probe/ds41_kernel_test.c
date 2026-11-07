/* tools/a64fx_probe/ds41_kernel_test.c - ds41_sve kernels against the double reference (parity, rel <= 1e-5 of
 * sum |w x|) and their speed on one A64FX node (weight GB/s, 1 thread and all threads), on random blocks.
 *   ds41_kernel_test [quick]     build: clang -O3 -mcpu=a64fx -msve-vector-bits=512 -fopenmp -Iinclude ... -lm */
#include "strata/kernels/cpu/ds41_sve.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned long long S = 88172645463325252ull;
static unsigned rnd(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return (unsigned) (S >> 16); }
static float frand(void) { return (rnd() % 20001) / 10000.f - 1.f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint16_t h16(float f) { __fp16 h = (__fp16) f; uint16_t u; memcpy(&u, &h, 2); return u; }

/* random bytes, then valid F16 scale fields */
static void fill(int t, uint8_t *w, size_t bytes, int rows, int cols)
{
    for (size_t i = 0; i < bytes; ++i) w[i] = (uint8_t) rnd();
    if (t == DS41_T_F32) { for (size_t i = 0; i < bytes / 4; ++i) ((float *) w)[i] = frand(); return; }
    if (t == DS41_T_BF16) { for (size_t i = 0; i < bytes / 2; ++i) { float f = frand(); uint32_t u; memcpy(&u, &f, 4); ((uint16_t *) w)[i] = (uint16_t) (u >> 16); } return; }
    const size_t bb = ds41_row_bytes(t, cols) / (t == DS41_T_Q8_0 || t == DS41_T_IQ4_NL || t == DS41_T_MXFP4 ? cols / 32 : cols / 256);
    if (t == DS41_T_MXFP4) {   /* E8M0 scales 2^-9 .. 2^-4 (the drafter's range), random nibbles */
        for (size_t o = 0; o < bytes; o += bb) w[o] = (uint8_t) (118 + rnd() % 6);
        return;
    }
    for (size_t o = 0; o < bytes; o += bb) {
        const uint16_t d = h16(0.001f + 0.01f * (rnd() % 1000) / 1000.f);
        memcpy(w + o, &d, 2);
        if (t == DS41_T_Q5_K) { const uint16_t m = h16(0.0005f * (rnd() % 1000) / 1000.f); memcpy(w + o + 2, &m, 2); }
    }
    (void) rows;
}

int main(int argc, char **argv)
{
    const int quick = argc > 1 && !strcmp(argv[1], "quick");
    struct { int t; const char *name; int rows, cols; } cases[] = {
        {DS41_T_IQ2_XXS, "iq2_xxs", 2304, 5120}, {DS41_T_IQ2_XXS, "iq2_xxs", 5120, 2304},
        {DS41_T_IQ2_XS, "iq2_xs", 2304, 5120}, {DS41_T_IQ2_XS, "iq2_xs", 5120, 2304},
        {DS41_T_IQ2_S, "iq2_s", 2304, 5120}, {DS41_T_IQ2_S, "iq2_s", 5120, 2304},
        {DS41_T_IQ3_XXS, "iq3_xxs", 5120, 2304},
        {DS41_T_IQ2_XXS, "iq2_xxs_d", 5120, 256}, {DS41_T_IQ2_XS, "iq2_xs_d", 5120, 256}, {DS41_T_IQ2_S, "iq2_s_d", 5120, 256},
        {DS41_T_Q5_K, "q5_K", 1024, 5120}, {DS41_T_Q8_0, "q8_0", 1024, 5120}, {DS41_T_BF16, "bf16", 1024, 5120},
        {DS41_T_F32, "f32", 24, 20480}, {DS41_T_MXFP4, "mxfp4", 2304, 5120}, {DS41_T_MXFP4, "mxfp4_d", 5120, 2304}};
    int fails = 0;
    const int nth = omp_get_max_threads();
    for (unsigned ci = 0; ci < sizeof cases / sizeof cases[0]; ++ci) {
        if (getenv("DS41K_ONLY") && strcmp(getenv("DS41K_ONLY"), cases[ci].name)) continue;
        const int t = cases[ci].t, rows = cases[ci].rows, cols = cases[ci].cols, kind = ds41_act_kind(t);
        const size_t rb = ds41_row_bytes(t, cols), bytes = rb * rows;
        uint8_t *w = aligned_alloc(256, (bytes + 255) & ~(size_t) 255);
        fill(t, w, bytes, rows, cols);
        void *acts[8];
        float *xf = malloc(sizeof(float) * cols);
        for (int n = 0; n < 8; ++n) {
            for (int k = 0; k < cols; ++k) xf[k] = frand() * (1 + (k % 7 == 0) * 4);
            if (kind == DS41_ACT_Q8K) { acts[n] = aligned_alloc(64, sizeof(ds41_q8k) * (cols / 256)); ds41_quant_q8k(xf, cols, acts[n]); }
            else if (kind == DS41_ACT_Q80) { acts[n] = malloc(sizeof(ds41_q80) * (cols / 32)); ds41_quant_q80(xf, cols, acts[n]); }
            else { acts[n] = aligned_alloc(64, sizeof(float) * cols); memcpy(acts[n], xf, sizeof(float) * cols); }
        }
        float *y = malloc(sizeof(float) * 8 * rows), *y1 = malloc(sizeof(float) * rows);
        double *yr = malloc(sizeof(double) * 4 * rows), *ya = malloc(sizeof(double) * 4 * rows);
        const int rcheck = rows < (quick ? 64 : 512) ? rows : (quick ? 64 : 512);
        double worst = 0;
        unsigned long long hash = 1469598103934665603ull;   /* FNV-1a of the kernel outputs: bitwise A/B between builds */
        for (int nt = 1; nt <= 4; ++nt) {
            ds41_rows(t, w, cols, 0, rcheck, (const void *const *) acts, nt, y, rows);
            ds41_rows_ref(t, w, cols, 0, rcheck, (const void *const *) acts, nt, yr, ya, rows);
            for (int n = 0; n < nt; ++n)
                for (int r = 0; r < rcheck; ++r) {
                    const double e = fabs(y[(size_t) n * rows + r] - yr[(size_t) n * rows + r]) / (ya[(size_t) n * rows + r] + 1e-30);
                    uint32_t u; memcpy(&u, &y[(size_t) n * rows + r], 4); hash = (hash ^ u) * 1099511628211ull;
                    if (!(e <= worst)) worst = isnan(e) ? INFINITY : e;
                }
        }
        /* batch invariance: each token's row of an nt-token call (nt up to 8) equals its 1-token call, bit for bit */
        long bad = 0;
        for (int nt = 2; nt <= 8; ++nt) {
            ds41_rows(t, w, cols, 0, rcheck, (const void *const *) acts, nt, y, rows);
            for (int n = 0; n < nt; ++n) {
                ds41_rows(t, w, cols, 0, rcheck, (const void *const *) (acts + n), 1, y1, rows);
                bad += memcmp(y1, y + (size_t) n * rows, sizeof(float) * (size_t) rcheck) != 0;
            }
        }
        const int ok = worst <= 1e-5 && !bad;
        fails += !ok;
        double gb1[8] = {0}, gbn[8] = {0};
        if (!quick)
            for (int nt = 1; nt <= 8; ++nt) {
                const int reps = 20;
                double t0 = now();
                for (int i = 0; i < reps / 4 + 1; ++i) ds41_rows(t, w, cols, 0, rows / 16, (const void *const *) acts, nt, y, rows);
                gb1[nt - 1] = (double) rb * (rows / 16) * (reps / 4 + 1) / (now() - t0) / 1e9;
                t0 = now();
                for (int i = 0; i < reps; ++i) {
#pragma omp parallel
                    {
                        const int th = omp_get_thread_num(), n = omp_get_num_threads();
                        ds41_rows(t, w, cols, rows * th / n, rows * (th + 1) / n, (const void *const *) acts, nt, y, rows);
                    }
                }
                gbn[nt - 1] = (double) bytes * reps / (now() - t0) / 1e9;
            }
        printf("DS41K %-8s %5dx%-5d parity=%s worst_rel=%.2e inv_bad=%ld hash=%016llx  GB/s 1thr %.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f  %dthr %.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f (nt 1-8)\n",
               cases[ci].name, rows, cols, ok ? "ok" : "FAIL", worst, bad, hash, gb1[0], gb1[1], gb1[2], gb1[3], gb1[4], gb1[5], gb1[6], gb1[7], nth, gbn[0], gbn[1], gbn[2],
               gbn[3], gbn[4], gbn[5], gbn[6], gbn[7]);
        fflush(stdout);
        for (int n = 0; n < 8; ++n) free(acts[n]);
        free(xf), free(y), free(y1), free(yr), free(ya), free(w);
    }
    printf("DS41K fails=%d\n", fails);
    return fails != 0;
}
