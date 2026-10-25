/* tools/a64fx_probe/glm_fp4_test.c - every src/kernels/cpu/glm_fp4_sve.c kernel against its scalar replica (bitwise) and against a
 * double-precision product of the source-format values (relative L2 <= 1e-5), 1..4 tokens, random weights with
 * subnormal and zero scales and outlier activations. Also runs under qemu-aarch64 on an x86 host:
 *   clang --target=aarch64-linux-gnu -fuse-ld=lld -march=armv8.2-a+sve -msve-vector-bits=512 -O2 -Iinclude \
 *     tools/a64fx_probe/glm_fp4_test.c src/kernels/cpu/glm_fp4_sve.c -lm -o t \
 *     && qemu-aarch64 -cpu max,sve512=on -L /usr/aarch64-linux-gnu ./t
 * Prints A64FX_FP4_CHECK lines; the exit code is the number of failures. */
#include "strata/kernels/cpu/glm_fp4_sve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t S = 0x5eed1234;
static uint64_t rnd(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return S; }
static float unif(void) { return (float) (rnd() >> 40) / 16777216.f; }
static float gauss(void) { float a = unif() + 1e-7f, b = unif(); return sqrtf(-2.f * logf(a)) * cosf(6.2831853f * b); }

static int failures;

static void report(const char *kernel, int nt, const float *got, const float *rep, const double *dref, int n)
{
    int bitwise = memcmp(got, rep, sizeof(float) * n) == 0;
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { const double e = got[i] - dref[i]; num += e * e; den += dref[i] * dref[i]; }
    const double rel = sqrt(num / (den > 0 ? den : 1e-300));
    const int ok = bitwise && rel <= 1e-5;
    printf("A64FX_FP4_CHECK kernel=%s tokens=%d bitwise=%d rel=%.2e %s\n", kernel, nt, bitwise, rel, ok ? "PASS" : "FAIL");
    failures += !ok;
}

static uint8_t rand_e4m3(int scale)     /* scale: non-negative, mostly normal, some subnormal and zero */
{
    for (;;) {
        uint8_t b = (uint8_t) rnd();
        if (scale) {
            const int u = (int) (rnd() % 100);
            b = u < 3 ? 0 : u < 10 ? (uint8_t) (1 + rnd() % 7) : (uint8_t) (0x20 + rnd() % 0x40);
        }
        if ((b & 0x7f) != 0x7f) return b;
    }
}

int main(void)
{
    const int rows = 32, cols = 512, T = 4;
    printf("A64FX_FP4_TEST sve512=%d\n", gf4_have_sve512());
    if (gf4_selftest_e4m3()) { printf("A64FX_FP4_CHECK kernel=e4m3_decode FAIL (FPCR flush-to-zero?)\n"); return 1; }
    printf("A64FX_FP4_CHECK kernel=e4m3_decode PASS\n");

    /* activations: gaussian with a few large outliers */
    float *x[4]; int8_t *q[4]; float *qs[4]; gf4_q8act act[4];
    for (int n = 0; n < T; ++n) {
        x[n] = malloc(sizeof(float) * cols); q[n] = malloc(cols); qs[n] = malloc(sizeof(float) * cols / 64);
        for (int k = 0; k < cols; ++k) x[n][k] = gauss() * (rnd() % 97 == 0 ? 40.f : 1.f);
        gf4_quantize_q8(x[n], cols, q[n], qs[n]);
        act[n] = (gf4_q8act) {cols, q[n], qs[n]};
    }
    int8_t *q16[4]; float *qs16[4]; gf4_q8act act16[4];
    for (int n = 0; n < T; ++n) { q16[n] = malloc(cols); qs16[n] = malloc(sizeof(float) * cols / 16); }
    const float *const *xp = (const float *const *) x;
    float *got = malloc(sizeof(float) * rows * T), *rep = malloc(sizeof(float) * rows * T);
    double *dref = malloc(sizeof(double) * rows * T);

    /* ---- NVFP4 */
    uint8_t *codes = malloc(rows * cols / 2), *scales = malloc(rows * cols / 16);
    for (int i = 0; i < rows * cols / 2; ++i) codes[i] = (uint8_t) rnd();
    for (int i = 0; i < rows * cols / 16; ++i) scales[i] = rand_e4m3(1);
    const float global = 3.1e-4f;
    double *wtrue = malloc(sizeof(double) * rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < cols; ++k) {
            const int c = (codes[(size_t) r * cols / 2 + k / 2] >> (4 * (k & 1))) & 15;
            wtrue[(size_t) r * cols + k] = (double) gf4_e2m1(c) * gf4_e4m3(scales[(size_t) r * cols / 16 + k / 16]) * global;
        }
    uint8_t *pq = malloc(gf4_nvfp4_packed_bytes(rows, cols)), *pp = malloc(gf4_nvfp4_packed_bytes(rows, cols));
    if (gf4_pack_rt16q(codes, scales, rows, cols, pq) || gf4_pack_rt16p(codes, scales, rows, cols, pp)) { puts("pack failed"); return 1; }
    for (int nt = 1; nt <= T; ++nt) {
        for (int n = 0; n < nt; ++n)
            for (int r = 0; r < rows; ++r) {
                double s = 0;
                for (int k = 0; k < cols; ++k) s += wtrue[(size_t) r * cols + k] * ((double) q[n][k] * qs[n][k / 64]);
                dref[n * rows + r] = s;
            }
        gf4_w4a8(pq, cols, global, 0, rows / 16, act, nt, got, rows);
        gf4_w4a8_ref(pq, cols, global, 0, rows / 16, act, nt, rep, rows);
        report("nvfp4_w4a8", nt, got, rep, dref, nt * rows);
        /* one activation scale per K16 group */
        for (int n = 0; n < nt; ++n) {
            gf4_quantize_q8g(x[n], cols, 16, q16[n], qs16[n]);
            act16[n] = (gf4_q8act) {cols, q16[n], qs16[n], 16};
            for (int r = 0; r < rows; ++r) {
                double s = 0;
                for (int k = 0; k < cols; ++k) s += wtrue[(size_t) r * cols + k] * ((double) q16[n][k] * qs16[n][k / 16]);
                dref[n * rows + r] = s;
            }
        }
        gf4_w4a8(pq, cols, global, 0, rows / 16, act16, nt, got, rows);
        gf4_w4a8_ref(pq, cols, global, 0, rows / 16, act16, nt, rep, rows);
        report("nvfp4_w4a8_g16", nt, got, rep, dref, nt * rows);
        for (int n = 0; n < nt; ++n)
            for (int r = 0; r < rows; ++r) {
                double s = 0;
                for (int k = 0; k < cols; ++k) s += wtrue[(size_t) r * cols + k] * x[n][k];
                dref[n * rows + r] = s;
            }
        gf4_w4a16q(pq, cols, global, 0, rows / 16, xp, nt, got, rows);
        gf4_w4a16_ref(pq, cols, global, 0, rows / 16, xp, nt, rep, rows, 0);
        report("nvfp4_w4a16q", nt, got, rep, dref, nt * rows);
        gf4_w4a16p(pp, cols, global, 0, rows / 16, xp, nt, got, rows);
        gf4_w4a16_ref(pp, cols, global, 0, rows / 16, xp, nt, rep, rows, 1);
        report("nvfp4_w4a16p", nt, got, rep, dref, nt * rows);
    }

    /* ---- MXFP8, BFP8 (E4M3 weights) and the Q8 requantization of the MXFP8 weights */
    const int frows = 128, fcols = 512;
    uint8_t *e4 = malloc((size_t) frows * fcols), *e8 = malloc((size_t) frows * fcols / 32);
    float *bsc = malloc(sizeof(float) * (frows / 128) * (fcols / 128));
    for (int i = 0; i < frows * fcols; ++i) e4[i] = rand_e4m3(0);
    for (int i = 0; i < frows * fcols / 32; ++i) e8[i] = (uint8_t) (112 + rnd() % 24);
    for (int i = 0; i < (frows / 128) * (fcols / 128); ++i) bsc[i] = 1e-3f * (0.5f + unif());
    float *fx[4];
    for (int n = 0; n < T; ++n) { fx[n] = malloc(sizeof(float) * fcols); for (int k = 0; k < fcols; ++k) fx[n][k] = gauss(); }
    const float *const *fxp = (const float *const *) fx;
    float *fgot = malloc(sizeof(float) * frows * T), *frep = malloc(sizeof(float) * frows * T);
    double *fref = malloc(sizeof(double) * frows * T);
    uint8_t *pmx = malloc(gf4_e8k_packed_bytes(frows, fcols, 1)), *pbf = malloc(gf4_e8k_packed_bytes(frows, fcols, 0));
    gf4_pack_e8k_mx(e4, e8, frows, fcols, pmx);
    gf4_pack_e8k_bfp8(e4, bsc, frows, fcols, pbf);
    float *wmx = malloc(sizeof(float) * frows * fcols);
    for (int r = 0; r < frows; ++r)
        for (int k = 0; k < fcols; ++k) wmx[(size_t) r * fcols + k] = gf4_e4m3(e4[(size_t) r * fcols + k]) * gf4_ue8m0(e8[(size_t) r * fcols / 32 + k / 32]);
    uint8_t *pq8 = malloc(gf4_q8k_packed_bytes(frows, fcols));
    gf4_pack_q8k(wmx, frows, fcols, pq8);
    int8_t *fq[4]; float *fqs[4]; gf4_q8act fact[4];
    for (int n = 0; n < T; ++n) {
        fq[n] = malloc(fcols); fqs[n] = malloc(sizeof(float) * fcols / 64);
        gf4_quantize_q8(fx[n], fcols, fq[n], fqs[n]);
        fact[n] = (gf4_q8act) {fcols, fq[n], fqs[n]};
    }
    uint16_t *b16 = malloc(2 * (size_t) frows * fcols);
    for (int i = 0; i < frows * fcols; ++i) { const float v = gauss() * 0.02f; uint32_t u; memcpy(&u, &v, 4); b16[i] = (uint16_t) (u >> 16); }
    uint8_t *pb16 = malloc(2 * (size_t) frows * fcols);
    gf4_pack_b16k(b16, frows, fcols, pb16);
    for (int nt = 1; nt <= T; ++nt) {
        for (int n = 0; n < nt; ++n)
            for (int r = 0; r < frows; ++r) {
                double s = 0;
                for (int k = 0; k < fcols; ++k) s += (double) wmx[(size_t) r * fcols + k] * fx[n][k];
                fref[n * frows + r] = s;
            }
        gf4_mxfp8(pmx, fcols, 0, frows / 16, fxp, nt, fgot, frows);
        gf4_mxfp8_ref(pmx, fcols, 0, frows / 16, fxp, nt, frep, frows);
        report("mxfp8", nt, fgot, frep, fref, nt * frows);
        /* Q8: bitwise against its replica; its error against the exact MXFP8 product is the fast-mode requantization */
        gf4_q8(pq8, fcols, 0, frows / 16, fact, nt, fgot, frows);
        gf4_q8_ref(pq8, fcols, 0, frows / 16, fact, nt, frep, frows);
        {
            double num = 0, den = 0;
            for (int i = 0; i < nt * frows; ++i) { const double e = fgot[i] - fref[i]; num += e * e; den += fref[i] * fref[i]; }
            const int bitwise = memcmp(fgot, frep, sizeof(float) * nt * frows) == 0;
            printf("A64FX_FP4_CHECK kernel=q8 tokens=%d bitwise=%d requant_rel=%.2e %s\n", nt, bitwise, sqrt(num / den), bitwise ? "PASS" : "FAIL");
            failures += !bitwise;
        }
        for (int n = 0; n < nt; ++n)
            for (int r = 0; r < frows; ++r) {
                double s = 0;
                for (int k = 0; k < fcols; ++k)
                    s += (double) gf4_e4m3(e4[(size_t) r * fcols + k]) * bsc[(r / 128) * (fcols / 128) + k / 128] * fx[n][k];
                fref[n * frows + r] = s;
            }
        gf4_bfp8(pbf, fcols, 0, frows / 16, fxp, nt, fgot, frows);
        gf4_bfp8_ref(pbf, fcols, 0, frows / 16, fxp, nt, frep, frows);
        report("bfp8", nt, fgot, frep, fref, nt * frows);
        for (int n = 0; n < nt; ++n)
            for (int r = 0; r < frows; ++r) {
                double s = 0;
                for (int k = 0; k < fcols; ++k) s += (double) gf4_bf16(b16[(size_t) r * fcols + k]) * fx[n][k];
                fref[n * frows + r] = s;
            }
        gf4_bf16k(pb16, fcols, 0, frows / 16, fxp, nt, fgot, frows);
        gf4_bf16k_ref(pb16, fcols, 0, frows / 16, fxp, nt, frep, frows);
        report("bf16", nt, fgot, frep, fref, nt * frows);
        {   /* column ranges: each chunk bitwise against its replica, the chunk sum against the double product */
            float *pg = malloc(sizeof(float) * frows * T), *pr = malloc(sizeof(float) * frows * T);
            for (int i = 0; i < nt * frows; ++i) fgot[i] = frep[i] = 0;
            int bit = 1;
            for (int k0 = 0; k0 < fcols; k0 += 128) {
                gf4_bf16k_part(pb16, fcols, k0, k0 + 128, 0, frows / 16, fxp, nt, pg, frows);
                gf4_bf16k_part_ref(pb16, fcols, k0, k0 + 128, 0, frows / 16, fxp, nt, pr, frows);
                bit &= memcmp(pg, pr, sizeof(float) * nt * frows) == 0;
                for (int i = 0; i < nt * frows; ++i) fgot[i] += pg[i], frep[i] += pr[i];
            }
            if (!bit) frep[0] += 1;   /* report() then shows the chunk mismatch as bitwise=0 */
            report("bf16_part", nt, fgot, frep, fref, nt * frows);
            free(pg), free(pr);
        }
    }
    printf("A64FX_FP4_TEST %s failures=%d\n", failures ? "FAIL" : "PASS", failures);
    return failures;
}
