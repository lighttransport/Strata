/* Ripple kernel test: ripple_nvfp4_rows against the double reference on random rows. Prints "RIPPLEK fails=N". */
#include "strata/kernels/cpu/ripple_sve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long long s = 88172645463325252ull;
static unsigned rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (unsigned)(s >> 16); }

int main(void) {
    int fails = 0;
    ripple_nvfp4_init();
    /* decoder spot checks */
    if (ripple_e4m3_to_f32(0x38) != 1.0f) { ++fails; printf("e4m3 0x38 = %g\n", ripple_e4m3_to_f32(0x38)); }
    if (ripple_e4m3_to_f32(0x7e) != 448.0f) { ++fails; printf("e4m3 0x7e = %g\n", ripple_e4m3_to_f32(0x7e)); }
    if (ripple_e4m3_to_f32(0x01) != ldexpf(1.0f, -9)) { ++fails; printf("e4m3 0x01\n"); }
    if (ripple_e2m1_to_f32(7) != 6.0f || ripple_e2m1_to_f32(0xb) != -1.5f) { ++fails; printf("e2m1\n"); }
    const int colsv[3] = {32, 2560, 5120};
    for (int ci = 0; ci < 3; ++ci) {
        const int cols = colsv[ci], rows = 37;
        unsigned char *q = malloc((size_t)rows * cols / 2), *sc = malloc((size_t)rows * cols / 16);
        float *x = malloc((size_t)cols * sizeof(float)), *y = malloc((size_t)rows * sizeof(float));
        double *ref = malloc((size_t)rows * sizeof(double));
        for (size_t i = 0; i < (size_t)rows * cols / 2; ++i) q[i] = (unsigned char)rnd();
        for (size_t i = 0; i < (size_t)rows * cols / 16; ++i) { unsigned char b; do b = (unsigned char)(0x20 + rnd() % 0x30); while ((b & 0x7f) == 0x7f); sc[i] = b; }
        for (int i = 0; i < cols; ++i) x[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
        const float *xs1[1] = {x};
        ripple_nvfp4_rows(q, sc, 0.0123f, cols, 0, rows, xs1, 1, y, rows);
        ripple_nvfp4_rows_ref(q, sc, 0.0123f, cols, 0, rows, x, ref);
        double worst = 0;
        for (int r = 0; r < rows; ++r) {
            const double e = fabs((double)y[r] - ref[r]) / (fabs(ref[r]) + 1e-3);
            if (e > worst) worst = e;
        }
        printf("nvfp4_rows cols=%d worst_rel=%.3g\n", cols, worst);
        if (worst > 2e-4) ++fails;
        /* several tokens: each against the one-token result */
        { float *xb = malloc((size_t)5 * cols * sizeof(float)), *yb = malloc((size_t)5 * rows * sizeof(float)), *y1 = malloc((size_t)rows * sizeof(float));
          const float *xs5[5];
          for (int t = 0; t < 5; ++t) { xs5[t] = xb + (size_t)t * cols; for (int i = 0; i < cols; ++i) xb[(size_t)t * cols + i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f; }
          ripple_nvfp4_rows(q, sc, 0.0123f, cols, 0, rows, xs5, 5, yb, rows);
          double wb = 0;
          for (int t = 0; t < 5; ++t) {
              const float *x1[1] = {xs5[t]};
              ripple_nvfp4_rows(q, sc, 0.0123f, cols, 0, rows, x1, 1, y1, rows);
              ripple_nvfp4_rows_ref(q, sc, 0.0123f, cols, 0, rows, xs5[t], ref);
              for (int r = 0; r < rows; ++r) { const double e = fabs((double)yb[(size_t)t * rows + r] - ref[r]) / (fabs(ref[r]) + 1e-3); if (e > wb) wb = e; }
          }
          printf("nvfp4_rows nt=5 cols=%d worst_rel=%.3g\n", cols, wb);
          if (wb > 2e-4) ++fails;
          free(xb); free(yb); free(y1); }
        free(q); free(sc); free(x); free(y); free(ref);
    }
    /* int8 path: repack + quantize + sdot against (a) the double reference on the dequantized activations (tight: same
     * integers, float order only) and (b) the unquantized reference (the quantization error). cols include non-multiples of 128. */
    {
        const int cq[6] = {32, 96, 128, 2560, 5120, 6816};
        for (int ci = 0; ci < 6; ++ci) {
            const int cols = cq[ci], rows = 29, NT = 3, units = ripple_q8_units(cols);
            unsigned char *q = malloc((size_t)rows * cols / 2), *sc = malloc((size_t)rows * cols / 16);
            unsigned char *qo = malloc((size_t)rows * cols / 2); unsigned short *so = malloc((size_t)rows * cols / 16 * 2);
            for (size_t i = 0; i < (size_t)rows * cols / 2; ++i) q[i] = (unsigned char)rnd();
            for (size_t i = 0; i < (size_t)rows * cols / 16; ++i) { unsigned char b; do b = (unsigned char)(0x20 + rnd() % 0x30); while ((b & 0x7f) == 0x7f); sc[i] = b; }
            for (int r = 0; r < rows; ++r) ripple_fp4_repack_row(q + (size_t)r * cols / 2, sc + (size_t)r * cols / 16, cols, qo + (size_t)r * cols / 2, so + (size_t)r * cols / 16);
            float *x = malloc((size_t)NT * cols * sizeof(float)), *xdq = malloc((size_t)cols * sizeof(float)), *y = malloc((size_t)NT * rows * sizeof(float)), *y2 = malloc((size_t)NT * rows * sizeof(float));
            signed char *xq = malloc((size_t)NT * cols); float *xd = malloc((size_t)NT * units * sizeof(float));
            const signed char *xqs[3]; const float *xds[3];
            for (int t = 0; t < NT; ++t) {
                for (int i = 0; i < cols; ++i) x[(size_t)t * cols + i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f * (i % 97 == 5 ? 12.0f : 1.0f);
                ripple_q8_quant(x + (size_t)t * cols, cols, xq + (size_t)t * cols, xd + (size_t)t * units);
                xqs[t] = xq + (size_t)t * cols; xds[t] = xd + (size_t)t * units;
            }
            ripple_fp4_rows_q8(qo, so, cols, 0, rows, xqs, xds, NT, 0.0123f, y, rows);
            ripple_fp4_rows_q8_ref(qo, so, cols, 0, rows, xqs, xds, NT, 0.0123f, y2, rows);
            double wk = 0, nq = 0, nr = 0, wr = 0;
            for (int t = 0; t < NT; ++t) {
                /* dequantized activation, original layout, double reference */
                for (int u = 0; u < units; ++u) {
                    int st, n; { const int full = cols / 128, g = u >> 1, rem = cols - full * 128;
                      if (g < full) { st = g * 128 + (u & 1) * 64; n = 64; } else { n = rem / 2; st = full * 128 + (u & 1) * n; } }
                    for (int i = 0; i < n; ++i) xdq[st + i] = (float)xqs[t][st + i] * xds[t][u];
                }
                double *rd = malloc((size_t)rows * sizeof(double)), *ru = malloc((size_t)rows * sizeof(double));
                ripple_nvfp4_rows_ref(q, sc, 0.0123f, cols, 0, rows, xdq, rd);
                ripple_nvfp4_rows_ref(q, sc, 0.0123f, cols, 0, rows, x + (size_t)t * cols, ru);
                for (int r = 0; r < rows; ++r) {
                    const double e = fabs((double)y[(size_t)t * rows + r] - rd[r]) / (fabs(rd[r]) + 1e-3);
                    if (e > wk) wk = e;
                    const double e2 = fabs((double)y[(size_t)t * rows + r] - (double)y2[(size_t)t * rows + r]) / (fabs(rd[r]) + 1e-3);
                    if (e2 > wr) wr = e2;
                    nq += (y[(size_t)t * rows + r] - ru[r]) * (y[(size_t)t * rows + r] - ru[r]); nr += ru[r] * ru[r];
                }
                free(rd); free(ru);
            }
            printf("q8 cols=%d kernel_vs_dequantized worst_rel=%.3g simd_vs_scalar worst_rel=%.3g quantization rel_l2=%.3g\n", cols, wk, wr, sqrt(nq / nr));
            if (wk > 2e-4 || wr > 2e-4 || sqrt(nq / nr) > 2e-2) ++fails;
            free(q); free(sc); free(qo); free(so); free(x); free(xdq); free(y); free(y2); free(xq); free(xd);
        }
    }
    printf("RIPPLEK fails=%d\n", fails);
    return fails != 0;
}
