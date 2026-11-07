/* tools/a64fx_probe/ds41_iq2r_test.c - IQ2R (ds41_repack_iq2r) against its source kernels: the same bits for 1-4
 * tokens on random IQ2_XXS / IQ2_XS / IQ2_S blocks, and the 1-thread weight GB/s of both (IQ2R counted at the source
 * type's bytes, the rate the model sees); IQ2P (ds41_iq2r_to_iq2p) the same way.   build: clang -O3 -mcpu=a64fx -msve-vector-bits=512 -Iinclude ... -lm */
#include "strata/kernels/cpu/ds41_sve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned long long S = 88172645463325252ull;
static unsigned rnd(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return (unsigned) (S >> 16); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint16_t h16(float f) { __fp16 h = (__fp16) f; uint16_t u; memcpy(&u, &h, 2); return u; }

int main(void)
{
    struct { int t; const char *name; int rows, cols; } cases[] = {
        {DS41_T_IQ2_XXS, "iq2_xxs", 2304, 5120}, {DS41_T_IQ2_XXS, "iq2_xxs_d", 5120, 256}, {DS41_T_IQ2_XS, "iq2_xs", 2304, 5120},
        {DS41_T_IQ2_XS, "iq2_xs_d", 5120, 256}, {DS41_T_IQ2_S, "iq2_s", 2304, 5120}, {DS41_T_IQ2_S, "iq2_s_d", 5120, 256}};
    int fails = 0;
    for (unsigned ci = 0; ci < sizeof cases / sizeof cases[0]; ++ci) {
        const int t = cases[ci].t, rows = cases[ci].rows, cols = cases[ci].cols;
        const size_t rb = ds41_row_bytes(t, cols), rr = ds41_row_bytes(DS41_T_IQ2R, cols), bb = rb / (cols / 256);
        uint8_t *w = malloc(rb * rows), *r = malloc(rr * rows), *q = malloc(rr * rows);
        for (size_t i = 0; i < rb * rows; ++i) w[i] = (uint8_t) rnd();
        for (size_t o = 0; o < rb * rows; o += bb) { const uint16_t d = h16(0.001f + 0.01f * (rnd() % 1000) / 1000.f); memcpy(w + o, &d, 2); }
        int bad = 0;
        for (int i = 0; i < rows; ++i) bad |= ds41_repack_iq2r(t, w + rb * i, cols, r + rr * i);
        memcpy(q, r, rr * rows);
        ds41_iq2r_to_iq2p(q, (size_t) rows * (cols / 256));
        int badp = 0;
        ds41_q8k *acts[8];
        for (int n = 0; n < 8; ++n) {
            float *x = malloc(sizeof(float) * cols);
            for (int k = 0; k < cols; ++k) x[k] = (rnd() % 20001) / 10000.f - 1.f;
            acts[n] = aligned_alloc(64, sizeof(ds41_q8k) * (cols / 256));
            ds41_quant_q8k(x, cols, acts[n]);
            free(x);
        }
        float *y0 = malloc(sizeof(float) * 8 * rows), *y1 = malloc(sizeof(float) * 8 * rows), *y2 = malloc(sizeof(float) * 8 * rows);
        double g0[8], g1[8], g2[8];
        for (int nt = 1; nt <= 8; ++nt) {
            ds41_rows(t, w, cols, 0, rows, (const void *const *) acts, nt, y0, rows);
            ds41_rows(DS41_T_IQ2R, r, cols, 0, rows, (const void *const *) acts, nt, y1, rows);
            bad |= memcmp(y0, y1, sizeof(float) * nt * rows) != 0;
            ds41_rows(DS41_T_IQ2P, q, cols, 0, rows, (const void *const *) acts, nt, y2, rows);
            badp |= memcmp(y0, y2, sizeof(float) * nt * rows) != 0;
            const int reps = 4;
            double t0 = now();
            for (int k = 0; k < reps; ++k) ds41_rows(t, w, cols, 0, rows, (const void *const *) acts, nt, y0, rows);
            g0[nt - 1] = (double) rb * rows * reps / (now() - t0) / 1e9;
            t0 = now();
            for (int k = 0; k < reps; ++k) ds41_rows(DS41_T_IQ2R, r, cols, 0, rows, (const void *const *) acts, nt, y1, rows);
            g1[nt - 1] = (double) rb * rows * reps / (now() - t0) / 1e9;
            t0 = now();
            for (int k = 0; k < reps; ++k) ds41_rows(DS41_T_IQ2P, q, cols, 0, rows, (const void *const *) acts, nt, y2, rows);
            g2[nt - 1] = (double) rb * rows * reps / (now() - t0) / 1e9;
        }
        fails += bad + badp;
        printf("IQ2R %-10s %5dx%-5d same_bits=%s/%s  1thr GB/s (source bytes) %s %.2f/%.2f/%.2f/%.2f  iq2r %.2f/%.2f/%.2f/%.2f  iq2p %.2f/%.2f/%.2f/%.2f  nt5-8 iq2p %.2f/%.2f/%.2f/%.2f\n",
               cases[ci].name, rows, cols, bad ? "NO" : "yes", badp ? "NO" : "yes", cases[ci].name, g0[0], g0[1], g0[2], g0[3], g1[0], g1[1], g1[2], g1[3],
               g2[0], g2[1], g2[2], g2[3], g2[4], g2[5], g2[6], g2[7]);
        fflush(stdout);
        for (int n = 0; n < 8; ++n) free(acts[n]);
        free(w), free(r), free(q), free(y0), free(y1), free(y2);
    }
    {   /* Q5P (ds41_q5k_to_q5p) against Q5_K: the same bits for 1-4 tokens; 1-thread GB/s of both */
        const int shapes[2][2] = {{1024, 5120}, {5120, 256}};
        for (int si = 0; si < 2; ++si) {
            const int rows = shapes[si][0], cols = shapes[si][1];
            const size_t rb = ds41_row_bytes(DS41_T_Q5_K, cols);
            uint8_t *w = malloc(rb * rows), *q = malloc(rb * rows);
            for (size_t i = 0; i < rb * rows; ++i) w[i] = (uint8_t) rnd();
            for (size_t o = 0; o < rb * rows; o += 176) {
                const uint16_t d = h16(0.001f + 0.01f * (rnd() % 1000) / 1000.f), m = h16(0.0005f * (rnd() % 1000) / 1000.f);
                memcpy(w + o, &d, 2), memcpy(w + o + 2, &m, 2);
            }
            memcpy(q, w, rb * rows);
            ds41_q5k_to_q5p(q, (size_t) rows * (cols / 256));
            ds41_q8k *acts[8];
            for (int n = 0; n < 8; ++n) {
                float *x = malloc(sizeof(float) * cols);
                for (int k = 0; k < cols; ++k) x[k] = (rnd() % 20001) / 10000.f - 1.f;
                acts[n] = aligned_alloc(64, sizeof(ds41_q8k) * (cols / 256));
                ds41_quant_q8k(x, cols, acts[n]);
                free(x);
            }
            float *y0 = malloc(sizeof(float) * 8 * rows), *y1 = malloc(sizeof(float) * 8 * rows);
            int bad = 0;
            double g0[8], g1[8];
            for (int nt = 1; nt <= 8; ++nt) {
                ds41_rows(DS41_T_Q5_K, w, cols, 0, rows, (const void *const *) acts, nt, y0, rows);
                ds41_rows(DS41_T_Q5P, q, cols, 0, rows, (const void *const *) acts, nt, y1, rows);
                bad |= memcmp(y0, y1, sizeof(float) * nt * rows) != 0;
                const int reps = 4;
                double t0 = now();
                for (int k = 0; k < reps; ++k) ds41_rows(DS41_T_Q5_K, w, cols, 0, rows, (const void *const *) acts, nt, y0, rows);
                g0[nt - 1] = (double) rb * rows * reps / (now() - t0) / 1e9;
                t0 = now();
                for (int k = 0; k < reps; ++k) ds41_rows(DS41_T_Q5P, q, cols, 0, rows, (const void *const *) acts, nt, y1, rows);
                g1[nt - 1] = (double) rb * rows * reps / (now() - t0) / 1e9;
            }
            fails += bad;
            printf("Q5P  q5_K       %5dx%-5d same_bits=%s  1thr GB/s q5_K %.2f/%.2f/%.2f/%.2f  q5p %.2f/%.2f/%.2f/%.2f  nt5-8 q5p %.2f/%.2f/%.2f/%.2f\n", rows, cols, bad ? "NO" : "yes",
                   g0[0], g0[1], g0[2], g0[3], g1[0], g1[1], g1[2], g1[3], g1[4], g1[5], g1[6], g1[7]);
            for (int n = 0; n < 8; ++n) free(acts[n]);
            free(w), free(q), free(y0), free(y1);
        }
    }
    printf("IQ2R fails=%d\n", fails);
    return fails != 0;
}
