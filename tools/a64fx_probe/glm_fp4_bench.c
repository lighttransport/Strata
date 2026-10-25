/* tools/a64fx_probe/glm_fp4_bench.c - node GB/s of the glm_fp4_sve.c kernels on A64FX (48 threads, first-touched copies).
 *
 *   glm_fp4_bench [--kernels w4a8,w4a16q,w4a16p,mxfp8,bfp8,q8,bf16] [--rows 2048] [--cols 4096] [--tokens 1,2,3,4]
 *                 [--threads N] [--mb 2048] [--reps 3] [--mode stream|phase] [--tag TEXT] [--gemm M]
 *
 * stream: every thread runs its 16-row tiles of every copy without a barrier (the kernel rate, as q23_sdot_bench);
 * phase:  an OpenMP barrier after every matrix (one projection as an in-model phase, barrier included).
 * The copies together exceed the caches; GBps counts the packed bytes (= the source format's bytes per weight).
 * --gemm M: prefill shape, M tokens per call in blocks of 4 over every 16-row tile (the tile is decoded once per block,
 * read from HBM once per call and from the caches for the next blocks); prints TOPS = 2 x MACs / s. */
#include "strata/kernels/cpu/glm_fp4_sve.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t S = 0x9e3779b97f4a7c15ull;
static uint64_t rnd(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return S; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

typedef struct { const char *name; size_t (*bytes)(int, int); } kinfo;
static size_t b_nvfp4(int r, int c) { return gf4_nvfp4_packed_bytes(r, c); }
static size_t b_mx(int r, int c) { return gf4_e8k_packed_bytes(r, c, 1); }
static size_t b_bfp(int r, int c) { return gf4_e8k_packed_bytes(r, c, 0); }
static size_t b_q8(int r, int c) { return gf4_q8k_packed_bytes(r, c); }
static size_t b_bf16(int r, int c) { return (size_t) r * c * 2; }

/* fill one tile range of a packed matrix with valid random content (codes random, scales in range) */
static void fill(const char *k, uint8_t *m, int cols, int t0, int t1, uint64_t seed)
{
    uint64_t s = seed | 1;
#define R() (s ^= s << 13, s ^= s >> 7, s ^= s << 17, s)
    if (!strcmp(k, "w4a8") || !strcmp(k, "w4a16q") || !strcmp(k, "w4a16p")) {
        const size_t per = (size_t) cols / 64 * GF4_SB_BYTES;
        for (size_t o = t0 * per; o < t1 * per; o += GF4_SB_BYTES) {
            for (int i = 0; i < 512; i += 8) { const uint64_t v = R(); memcpy(m + o + i, &v, 8); }
            for (int i = 0; i < 64; ++i) m[o + 512 + i] = (uint8_t) (0x20 + R() % 0x40);
        }
    } else if (!strcmp(k, "mxfp8") || !strcmp(k, "bfp8")) {
        const int mx = !strcmp(k, "mxfp8"), blk = mx ? 528 : 2064, w = mx ? 512 : 2048;
        const size_t per = (size_t) cols / (mx ? 32 : 128) * blk;
        for (size_t o = t0 * per; o < t1 * per; o += blk) {
            for (int i = 0; i < w; ++i) { uint8_t b = (uint8_t) R(); if ((b & 0x7f) == 0x7f) b = 0x3f; m[o + i] = b; }
            if (mx) for (int i = 0; i < 16; ++i) m[o + 512 + i] = (uint8_t) (115 + R() % 16);
            else { const float f = 1e-3f; memset(m + o + 2048, 0, 16); memcpy(m + o + 2048, &f, 4); }
        }
    } else if (!strcmp(k, "q8")) {
        const size_t per = (size_t) cols / 32 * GF4_Q8_BLOCK_BYTES;
        for (size_t o = t0 * per; o < t1 * per; o += GF4_Q8_BLOCK_BYTES) {
            for (int i = 0; i < 512; i += 8) { const uint64_t v = R(); memcpy(m + o + i, &v, 8); }
            for (int i = 0; i < 16; ++i) { const uint16_t h = 0x1c00 + (uint16_t) (R() % 0x400); memcpy(m + o + 512 + 2 * i, &h, 2); }
        }
    } else {
        const size_t per = (size_t) cols * 32;
        for (size_t o = t0 * per; o < t1 * per; o += 2) {
            const uint64_t v = R();
            const uint16_t h = (uint16_t) (0x3c00 + v % 0x0400) ^ (uint16_t) ((v >> 32) & 0x8000);
            memcpy(m + o, &h, 2);
        }
    }
#undef R
}

static void run(const char *k, const uint8_t *m, int cols, int t0, int t1, const gf4_q8act *act, const float *const *x, int nt, float *y, int ldy)
{
    if (!strcmp(k, "w4a8")) gf4_w4a8(m, cols, 1e-4f, t0, t1, act, nt, y, ldy);
    else if (!strcmp(k, "w4a16q")) gf4_w4a16q(m, cols, 1e-4f, t0, t1, x, nt, y, ldy);
    else if (!strcmp(k, "w4a16p")) gf4_w4a16p(m, cols, 1e-4f, t0, t1, x, nt, y, ldy);
    else if (!strcmp(k, "mxfp8")) gf4_mxfp8(m, cols, t0, t1, x, nt, y, ldy);
    else if (!strcmp(k, "bfp8")) gf4_bfp8(m, cols, t0, t1, x, nt, y, ldy);
    else if (!strcmp(k, "q8")) gf4_q8(m, cols, t0, t1, act, nt, y, ldy);
    else gf4_bf16k(m, cols, t0, t1, x, nt, y, ldy);
}

int main(int argc, char **argv)
{
    const char *kernels = "w4a8,w4a16q,w4a16p,mxfp8,bfp8,q8,bf16", *tokens = "1,2,3,4", *mode = "stream", *tag = "";
    int rows = 2048, cols = 4096, threads = omp_get_max_threads(), mb = 2048, reps = 3, gemm = 0;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--kernels")) kernels = v, ++i;
        else if (!strcmp(a, "--rows")) rows = atoi(v), ++i;
        else if (!strcmp(a, "--cols")) cols = atoi(v), ++i;
        else if (!strcmp(a, "--tokens")) tokens = v, ++i;
        else if (!strcmp(a, "--threads")) threads = atoi(v), ++i;
        else if (!strcmp(a, "--mb")) mb = atoi(v), ++i;
        else if (!strcmp(a, "--reps")) reps = atoi(v), ++i;
        else if (!strcmp(a, "--mode")) mode = v, ++i;
        else if (!strcmp(a, "--tag")) tag = v, ++i;
        else if (!strcmp(a, "--gemm")) gemm = atoi(v), ++i;
        else { fprintf(stderr, "unknown argument %s\n", a); return 2; }
    }
    if (rows % 16 || cols % 128) { fprintf(stderr, "rows %% 16 and cols %% 128 must be 0\n"); return 2; }
    omp_set_num_threads(threads);
    const int phase = !strcmp(mode, "phase"), tiles = rows / 16;
    if (!gf4_have_sve512()) fprintf(stderr, "warning: no 512-bit SVE build, timing the scalar replicas\n");
    if (gf4_selftest_e4m3()) { fprintf(stderr, "E4M3 vector decode self-test failed (FPCR?)\n"); return 1; }

    const int nx = gemm > 4 ? gemm : 4;
    if (gemm % 4) { fprintf(stderr, "--gemm M must be a multiple of 4\n"); return 2; }
    float **x = malloc(sizeof(float *) * nx); int8_t **q = malloc(sizeof(int8_t *) * nx); float **qs = malloc(sizeof(float *) * nx);
    gf4_q8act *act = malloc(sizeof(gf4_q8act) * nx);
    for (int n = 0; n < nx; ++n) {
        x[n] = malloc(sizeof(float) * cols); q[n] = malloc(cols); qs[n] = malloc(sizeof(float) * cols / 64);
        for (int k = 0; k < cols; ++k) x[n][k] = (float) ((int) (rnd() % 2001) - 1000) * 1e-3f;
        gf4_quantize_q8(x[n], cols, q[n], qs[n]);
        act[n] = (gf4_q8act) {cols, q[n], qs[n]};
    }
    float *y = aligned_alloc(256, sizeof(float) * nx * rows);

    char klist[256]; snprintf(klist, sizeof klist, "%s", kernels);
    for (char *k = strtok(klist, ","); k; k = strtok(NULL, ",")) {
        size_t bytes = !strncmp(k, "w4", 2) ? b_nvfp4(rows, cols) : !strcmp(k, "mxfp8") ? b_mx(rows, cols)
                     : !strcmp(k, "bfp8") ? b_bfp(rows, cols) : !strcmp(k, "q8") ? b_q8(rows, cols) : b_bf16(rows, cols);
        int copies = (int) ((size_t) mb * 1048576 / bytes); if (copies < 1) copies = 1;
        uint8_t *buf = aligned_alloc(256, ((bytes * copies + 255) / 256) * 256);
#pragma omp parallel
        {   /* first touch with the timing schedule: every thread's tiles live on its own CMG */
            const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
            const int t0 = tid * tiles / nth, t1 = (tid + 1) * tiles / nth;
            for (int c = 0; c < copies; ++c) fill(k, buf + (size_t) c * bytes, cols, t0, t1, 77 + 1000003ull * c + tid);
        }
        if (gemm) {
            double best = 1e30;
            for (int rep = 0; rep <= reps; ++rep) {
                const double t = now();
#pragma omp parallel
                {
                    const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
                    const int t0 = tid * tiles / nth, t1 = (tid + 1) * tiles / nth;
                    for (int c = 0; c < copies; ++c)
                        for (int tt = t0; tt < t1; ++tt)
                            for (int b = 0; b < gemm; b += 4)
                                run(k, buf + (size_t) c * bytes, cols, tt, tt + 1, act + b, (const float *const *) x + b, 4, y + (size_t) b * rows, rows);
                }
                const double dt = now() - t;
                if (rep && dt < best) best = dt;
            }
            const double per = best / copies;
            printf("A64FX_FP4_GEMM kernel=%s%s threads=%d rows=%d cols=%d M=%d bytes=%zu copies=%d us=%.2f TOPS=%.3f GBps=%.1f\n",
                   k, tag, threads, rows, cols, gemm, bytes, copies, per * 1e6, 2.0 * rows * cols * gemm / per * 1e-12, bytes / per * 1e-9);
            fflush(stdout);
            free(buf);
            continue;
        }
        char tl[64]; snprintf(tl, sizeof tl, "%s", tokens);
        char *save = NULL;
        for (char *tk = strtok_r(tl, ",", &save); tk; tk = strtok_r(NULL, ",", &save)) {
            const int nt = atoi(tk);
            double best = 1e30;
            for (int rep = 0; rep <= reps; ++rep) {   /* rep 0 warms up */
                const double t = now();
#pragma omp parallel
                {
                    const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
                    const int t0 = tid * tiles / nth, t1 = (tid + 1) * tiles / nth;
                    for (int c = 0; c < copies; ++c) {
                        run(k, buf + (size_t) c * bytes, cols, t0, t1, act, (const float *const *) x, nt, y, rows);
                        if (phase) {
#pragma omp barrier
                        }
                    }
                }
                const double dt = now() - t;
                if (rep && dt < best) best = dt;
            }
            const double per = best / copies;
            printf("A64FX_FP4 kernel=%s%s threads=%d rows=%d cols=%d tokens=%d mode=%s bytes=%zu copies=%d us=%.2f GBps=%.1f GMACs=%.1f\n",
                   k, tag, threads, rows, cols, nt, mode, bytes, copies, per * 1e6, bytes / per * 1e-9,
                   (double) rows * cols * nt / per * 1e-9);
            fflush(stdout);
        }
        free(buf);
    }
    return 0;
}
