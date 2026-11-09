/* k31_mv_bench: HBM throughput of the K3.1 decode matvecs on one A64FX node. MXFP4 expert-shaped matrices
 * (3072 x 3584) through k31_mxfp4_rows and ds41_rows, BF16 (128 x 7168 and 7168 x 128) through ds41_rows; pages
 * first-touched by one thread (touch=1) or by the threads that read them (touch=par). Prints GB/s.
 *   mpiclang -O3 -mcpu=a64fx -msve-vector-bits=512 -fopenmp -Iinclude k31_mv_bench.c k31_sve.o ds41_sve.o -lm */
#include "strata/kernels/cpu/ds41_sve.h"
#include "strata/kernels/cpu/k31_sve.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now(void) { return omp_get_wtime(); }

static void bench_mx(int nm, int par_touch, int chunk, int use_k31) {
    const int R = 3072, C = 3584, nb = C / 32;
    const size_t mb = (size_t)R * nb * 17;
    uint8_t *w = aligned_alloc(256, mb * nm);
    if (par_touch) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < mb * nm; i += 4096) memset(w + i, 0x5a, (mb * nm - i) < 4096 ? mb * nm - i : 4096);
    } else memset(w, 0x5a, mb * nm);
    for (size_t i = 0; i < mb * nm; i += 17) w[i] = 120;
    int8_t *aq = malloc(C); float *ad = malloc(sizeof(float) * nb), *y = malloc(sizeof(float) * R);
    float *x = malloc(sizeof(float) * C);
    for (int i = 0; i < C; ++i) { aq[i] = (int8_t)(i % 7); x[i] = 0.01f * (i % 13); }
    for (int i = 0; i < nb; ++i) ad[i] = 0.01f;
    ds41_q80 *q8 = malloc(sizeof(ds41_q80) * nb);
    ds41_quant_q80(x, C, q8);
    const void *acts[1] = {q8};
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now();
        for (int m = 0; m < nm; ++m) {
            const uint8_t *wm = w + mb * m;
#pragma omp parallel for schedule(dynamic, 1)
            for (int r0 = 0; r0 < R; r0 += chunk) {
                const int r1 = r0 + chunk < R ? r0 + chunk : R;
                if (use_k31) k31_mxfp4_rows(wm, C, r0, r1, aq, ad, y);
                else ds41_rows(DS41_T_MXFP4, wm, C, r0, r1, acts, 1, y, R);
            }
        }
        const double t = now() - t0;
        if (t < best) best = t;
    }
    printf("MV mxfp4 %s touch=%s chunk=%d mats=%d us_per_mat=%.1f GB/s=%.1f\n", use_k31 ? "k31" : "ds41",
           par_touch ? "par" : "1", chunk, nm, 1e6 * best / nm, mb * nm / best / 1e9);
    free(w); free(aq); free(ad); free(y); free(x); free(q8);
}

static void bench_bf16(int R, int C, int nm, int par_touch, int chunk) {
    const size_t mb = (size_t)R * C * 2;
    uint8_t *w = aligned_alloc(256, mb * nm);
    if (par_touch) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < mb * nm; i += 4096) memset(w + i, 0x3c, (mb * nm - i) < 4096 ? mb * nm - i : 4096);
    } else memset(w, 0x3c, mb * nm);
    float *x = malloc(sizeof(float) * C), *y = malloc(sizeof(float) * R);
    for (int i = 0; i < C; ++i) x[i] = 0.01f;
    const void *acts[1] = {x};
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now();
        for (int m = 0; m < nm; ++m) {
            const uint8_t *wm = w + mb * m;
#pragma omp parallel for schedule(dynamic, 1)
            for (int r0 = 0; r0 < R; r0 += chunk) ds41_rows(DS41_T_BF16, wm, C, r0, r0 + chunk < R ? r0 + chunk : R, acts, 1, y, R);
        }
        const double t = now() - t0;
        if (t < best) best = t;
    }
    printf("MV bf16 %dx%d touch=%s chunk=%d mats=%d us_per_mat=%.1f GB/s=%.1f\n", R, C, par_touch ? "par" : "1", chunk, nm,
           1e6 * best / nm, mb * nm / best / 1e9);
    free(w); free(x); free(y);
}

int main(void) {
    printf("MV threads=%d\n", omp_get_max_threads());
    for (int pt = 0; pt < 2; ++pt) {
        bench_mx(16, pt, 16, 1);
        bench_mx(16, pt, 16, 0);
        bench_mx(16, pt, 4, 1);
        bench_mx(16, pt, 64, 1);
        bench_bf16(128, 7168, 64, pt, 2);
        bench_bf16(128, 7168, 64, pt, 8);
        bench_bf16(7168, 128, 64, pt, 32);
    }
    /* empty parallel regions: fork / join cost */
    double t0 = now();
    for (int i = 0; i < 10000; ++i) {
#pragma omp parallel
        { __asm__ __volatile__("" ::: "memory"); }
    }
    printf("MV omp_region_us=%.2f\n", (now() - t0) / 10000 * 1e6);
    return 0;
}
