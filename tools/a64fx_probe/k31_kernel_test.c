/* k31_kernel_test: K3.1 Pro kernels (src/kernels/cpu/k31_sve.c) against double oracles and, with -DK31_K3REF and the
 * K3 experiment headers on the include path, against the K3 reference functions. Prints "K31K <case> ..." lines and
 * "K31K fails=N". No model, no GPU.
 *   cc -O2 -std=c11 -Iinclude [-DK31_K3REF -I<k3 dir>] tools/a64fx_probe/k31_kernel_test.c src/kernels/cpu/k31_sve.c -lm */
#include "strata/kernels/cpu/k31_sve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(K31_K3REF)
#include "k3_kernels.h"
#endif

static int fails = 0;
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static float frand(void) {   /* [-1, 1) */
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float)((rs >> 40) & 0xffffff) / 8388608.0f - 1.0f;
}
static void fill(float *x, size_t n, float s) { for (size_t i = 0; i < n; ++i) x[i] = frand() * s; }

static void report(const char *name, double worst, double tol) {
    const int ok = worst <= tol && worst == worst;
    printf("K31K %s worst=%.3e tol=%.1e %s\n", name, worst, tol, ok ? "ok" : "FAIL");
    if (!ok) ++fails;
}
static double relerr(const float *a, const double *b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { const double d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}
static double relerr_f(const float *a, const float *b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { const double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}
static double sig_d(double x) { return 1.0 / (1.0 + exp(-x)); }

static void t_norms(void) {
    enum { N = K31_HIDDEN };
    static float x[N], w[N], b[N], g[N], y[N];
    static double r[N];
    fill(x, N, 3.0f); fill(w, N, 1.0f); fill(b, N, 0.5f); fill(g, N, 4.0f);
    k31_rmsnorm(y, x, w, N, 1e-5f);
    double ss = 0; for (int i = 0; i < N; ++i) ss += (double)x[i] * x[i];
    for (int i = 0; i < N; ++i) r[i] = x[i] / sqrt(ss / N + 1e-5) * w[i];
    report("rmsnorm", relerr(y, r, N), 1e-6);
    k31_layernorm(y, x, w, b, N, 1e-6f);
    double m = 0, v = 0; for (int i = 0; i < N; ++i) m += x[i];
    m /= N; for (int i = 0; i < N; ++i) v += (x[i] - m) * (x[i] - m);
    for (int i = 0; i < N; ++i) r[i] = (x[i] - m) / sqrt(v / N + 1e-6) * w[i] + b[i];
    report("layernorm", relerr(y, r, N), 1e-6);
    k31_gated_rmsnorm(y, x, g, w, 128, 1e-5f);
    ss = 0; for (int i = 0; i < 128; ++i) ss += (double)x[i] * x[i];
    for (int i = 0; i < 128; ++i) r[i] = x[i] / sqrt(ss / 128 + 1e-5) * w[i] * sig_d(g[i]);
    report("gated_rmsnorm", relerr(y, r, 128), 1e-6);
#if defined(K31_K3REF)
    float y3[128];
    k3_gated_rmsnorm_ref(y3, x, g, w, 128, 1e-5f);
    report("gated_rmsnorm_vs_k3", relerr_f(y, y3, 128), 0.0);
#endif
}

static void t_situ(void) {
    enum { N = 4096 };
    static float g[N], u[N], y[N], yf[N];
    static double r[N];
    fill(g, N, 12.0f); fill(u, N, 60.0f);
    k31_situ(y, g, u, N);
    for (int i = 0; i < N; ++i) r[i] = 4.0 * tanh(g[i] / 4.0) * sig_d(g[i]) * 25.0 * tanh(u[i] / 25.0);
    report("situ", relerr(y, r, N), 1e-6);
    k31_situ_fast(yf, g, u, N);
    report("situ_fast", relerr(yf, r, N), 2e-3);   /* K3's [situ-fexpa] gate: 2e-3 */
#if defined(K31_K3REF)
    float y3[N];
    k3_situ_ref(y3, g, u, N);
    report("situ_vs_k3", relerr_f(y, y3, N), 0.0);
#endif
}

static void t_conv_kda(void) {
    enum { H = 4, D = K31_HEAD_DIM, C = H * D, STEPS = 5 };
    static float w[C * K31_CONV], st[C * (K31_CONV - 1)], st3[C * (K31_CONV - 1)], x[C], y[C], y3[C];
    (void)st3; (void)y3;
    fill(w, C * K31_CONV, 0.5f); memset(st, 0, sizeof st); memset(st3, 0, sizeof st3);
    double worst = 0;
    for (int s = 0; s < STEPS; ++s) {
        fill(x, C, 2.0f);
        k31_conv_step(y, x, st, w, C, K31_CONV);
        /* oracle: keep history in a double ring */
        static double hist[STEPS][C];
        for (int c = 0; c < C; ++c) hist[s][c] = x[c];
        static double ref[C];
        for (int c = 0; c < C; ++c) {
            double acc = 0;
            for (int j = 0; j < K31_CONV; ++j) { const int t = s - (K31_CONV - 1) + j; if (t >= 0) acc += w[c * K31_CONV + j] * hist[t][c]; }
            ref[c] = acc;
        }
        worst = fmax(worst, relerr(y, ref, C));
#if defined(K31_K3REF)
        k3_conv_step_ref(y3, x, st3, w, NULL, C, K31_CONV);
        const double e3 = relerr_f(y, y3, C); if (e3 > 0) worst = fmax(worst, 1.0 + e3);   /* must be bit-equal */
#endif
    }
    report("conv_step", worst, 1e-5);

    static float g[C], a[H], dt[C], ld[C], dec[C], q[C], k[C], v[C], beta[H], out[C];
    static float S[H * D * D];
    static double Sd[H * D * D], od[C];
    fill(a, H, 1.0f); for (int h = 0; h < H; ++h) a[h] = fabsf(a[h]) + 0.1f;
    fill(dt, C, 0.5f); memset(S, 0, sizeof S); memset(Sd, 0, sizeof Sd);
#if defined(K31_K3REF)
    static float S3[H * D * D], out3[C], ld3[C];
    memset(S3, 0, sizeof S3);
#endif
    double wk = 0, wk3 = 0, wld = 0;
    for (int s = 0; s < STEPS; ++s) {
        fill(g, C, 3.0f); fill(q, C, 1.0f); fill(k, C, 1.0f); fill(v, C, 1.0f); fill(beta, H, 1.0f);
        for (int h = 0; h < H; ++h) { k31_l2_normalize(q + h * D, D, 1e-6f); k31_l2_normalize(k + h * D, D, 1e-6f); beta[h] = k31_sigmoid(beta[h]); }
        k31_kda_log_decay(ld, g, a, dt, H, D, -5.0f);
        for (int i = 0; i < C; ++i) { const double z = -5.0 * sig_d(-(double)a[i / D] * (g[i] + dt[i])); wld = fmax(wld, fabs(ld[i] - z)); }
        for (int i = 0; i < C; ++i) dec[i] = expf(ld[i]);
        k31_kda_step(out, q, k, v, dec, beta, S, H, D, D);
        for (int h = 0; h < H; ++h)   /* double oracle of the delta rule */
            for (int j = 0; j < D; ++j) {
                double *row = Sd + ((size_t)h * D + j) * D, pred = 0, o = 0;
                for (int d = 0; d < D; ++d) { row[d] *= dec[h * D + d]; pred += k[h * D + d] * row[d]; }
                const double delta = beta[h] * (v[h * D + j] - pred);
                for (int d = 0; d < D; ++d) { row[d] += k[h * D + d] * delta; o += q[h * D + d] * row[d]; }
                od[h * D + j] = o / sqrt((double)D);
            }
        wk = fmax(wk, relerr(out, od, C));
#if defined(K31_K3REF)
        k3_kda_log_decay(ld3, g, a, dt, H, D);
        wk3 = fmax(wk3, relerr_f(ld, ld3, C));
        k3_kda_step_ref(out3, q, k, v, ld, beta, S3, H, D, D);
        wk3 = fmax(wk3, relerr_f(out, out3, C));
#endif
    }
    report("kda_log_decay", wld, 1e-6);
    report("kda_step", wk, 1e-5);
#if defined(K31_K3REF)
    report("kda_vs_k3", wk3, 1e-6);
#endif
    (void)wk3;
}

static void t_attn_res(void) {
    enum { N = K31_HIDDEN, B = 7 };
    static float blocks[B * N], prefix[N], nw[N], pw[N], y[N];
    static double r[N];
    fill(blocks, (size_t)B * N, 2.0f); fill(prefix, N, 2.0f); fill(nw, N, 1.0f); fill(pw, N, 0.05f);
    k31_attn_res(y, prefix, blocks, B, nw, pw, N, 1e-5f);
    double sc[B + 1], mx = -INFINITY, z = 0;
    for (int c = 0; c <= B; ++c) {
        const float *v = c == B ? prefix : blocks + (size_t)c * N;
        double ss = 0, s = 0; for (int j = 0; j < N; ++j) ss += (double)v[j] * v[j];
        const double inv = 1.0 / sqrt(ss / N + 1e-5);
        for (int j = 0; j < N; ++j) s += v[j] * inv * nw[j] * pw[j];
        sc[c] = s; mx = fmax(mx, s);
    }
    for (int c = 0; c <= B; ++c) z += exp(sc[c] - mx);
    for (int j = 0; j < N; ++j) { double v = 0; for (int c = 0; c <= B; ++c) v += (c == B ? prefix : blocks + (size_t)c * N)[j] * exp(sc[c] - mx) / z; r[j] = v; }
    report("attn_res", relerr(y, r, N), 1e-5);
    k31_attn_res(y, prefix, blocks, 0, nw, pw, N, 1e-5f);
    report("attn_res_first_block", relerr_f(y, prefix, N), 0.0);
}

static void t_router(void) {
    static float lg[K31_EXPERTS], bias[K31_EXPERTS], w[K31_TOPK];
    static int idx[K31_TOPK];
    fill(lg, K31_EXPERTS, 4.0f); fill(bias, K31_EXPERTS, 0.1f);
    lg[10] = lg[20] = 3.9999f; bias[10] = bias[20] = 0.0f;   /* a tie */
    k31_router_topk(lg, bias, K31_EXPERTS, K31_TOPK, 1.0f, idx, w);
    /* oracle */
    double sc[K31_EXPERTS]; int used[K31_EXPERTS] = {0}, oi[K31_TOPK]; double ow[K31_TOPK], sum = 0;
    for (int e = 0; e < K31_EXPERTS; ++e) sc[e] = (float)k31_sigmoid(lg[e]) + bias[e];
    for (int k = 0; k < K31_TOPK; ++k) {
        int b = -1; for (int e = 0; e < K31_EXPERTS; ++e) if (!used[e] && (b < 0 || (float)sc[e] > (float)sc[b])) b = e;
        used[b] = 1; oi[k] = b; sum += (ow[k] = sig_d(lg[b]));
    }
    int bad = 0; for (int k = 0; k < K31_TOPK; ++k) { bad += idx[k] != oi[k]; ow[k] /= sum; }
    report("router_idx", bad, 0);
    report("router_w", relerr(w, ow, K31_TOPK), 1e-6);
#if defined(K31_K3REF)
    int i3[K31_TOPK]; float w3[K31_TOPK];
    k3_router_topk(lg, bias, K31_EXPERTS, K31_TOPK, i3, w3);
    int bad3 = 0; for (int k = 0; k < K31_TOPK; ++k) bad3 += idx[k] != i3[k];
    report("router_idx_vs_k3", bad3, 0);
    report("router_w_vs_k3", relerr_f(w, w3, K31_TOPK), 1e-7);
#endif
}

static int cmp_desc(const void *a, const void *b) {
    const float *x = (const float *)a, *y = (const float *)b;   /* (score, index) pairs */
    if (x[0] != y[0]) return x[0] < y[0] ? 1 : -1;
    return x[1] < y[1] ? -1 : 1;
}
static int cmp_i(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

static void t_indexer(void) {
    enum { H = K31_IDX_HEADS, D = K31_IDX_DIM, T = 5000, K = K31_IDX_TOPK };
    static float q[H * D], w[H], sc[T], kf[T * D];
    static uint16_t keys[T * D];
    static double r[T];
    fill(q, H * D, 1.0f); fill(w, H, 1.0f); fill(kf, (size_t)T * D, 1.0f);
    for (size_t i = 0; i < (size_t)T * D; ++i) keys[i] = k31_f32_to_f16(kf[i]);
    const float ws = 1.0f / sqrtf((float)(D * H));
    k31_index_scores(q, w, H, D, keys, T, ws, sc);
    for (int t = 0; t < T; ++t) {
        double s = 0;
        for (int h = 0; h < H; ++h) { double a = 0; for (int d = 0; d < D; ++d) a += (double)q[h * D + d] * k31_f16_to_f32(keys[(size_t)t * D + d]); s += fmax(a, 0.0) * w[h] * ws; }
        r[t] = s;
    }
    report("index_scores", relerr(sc, r, T), 1e-5);
    /* top-k with ties: force a block of equal scores around the threshold */
    for (int t = 100; t < 140; ++t) sc[t] = sc[99];
    static int got[K], want[K];
    static float pairs[T][2];
    for (int t = 0; t < T; ++t) { pairs[t][0] = sc[t]; pairs[t][1] = (float)t; }
    qsort(pairs, T, sizeof pairs[0], cmp_desc);
    for (int i = 0; i < K; ++i) want[i] = (int)pairs[i][1];
    qsort(want, K, sizeof(int), cmp_i);
    const int n = k31_topk_indices(sc, T, K, got);
    int bad = n != K; for (int i = 0; i < K && !bad; ++i) bad += got[i] != want[i];
    report("topk", bad, 0);
    const int m = k31_topk_indices(sc, 100, K, got);
    report("topk_short", m != 100 || got[99] != 99, 0);
}

static void t_latent_attention(void) {
    /* absorbed MLA (latent cache) against decompressed per-head keys/values (the K3 runner's form) */
    enum { H = 3, T = 300, NOPE = K31_HEAD_DIM, V = K31_HEAD_DIM, QK = NOPE + K31_ROPE };
    static float wkv[H * (NOPE + V) * K31_KV_LORA], c[T * K31_LAT], q[H * QK], qa[H * K31_LAT], ol[H * K31_KV_LORA];
    static uint16_t cache[T * K31_LAT];
    static float keys[H][T * QK], vals[H][T * V], ref[H * V], out[H * V];
    fill(wkv, sizeof wkv / sizeof wkv[0], 0.05f); fill(c, (size_t)T * K31_LAT, 1.0f); fill(q, H * QK, 1.0f);
    for (size_t i = 0; i < (size_t)T * K31_LAT; ++i) { cache[i] = k31_f32_to_f16(c[i]); c[i] = k31_f16_to_f32(cache[i]); }
    for (int h = 0; h < H; ++h) {
        const float *wk = wkv + (size_t)h * (NOPE + V) * K31_KV_LORA, *wv = wk + (size_t)NOPE * K31_KV_LORA;
        for (int t = 0; t < T; ++t) {
            const float *ct = c + (size_t)t * K31_LAT;
            for (int r = 0; r < NOPE; ++r) { double a = 0; for (int d = 0; d < K31_KV_LORA; ++d) a += (double)wk[r * K31_KV_LORA + d] * ct[d]; keys[h][t * QK + r] = (float)a; }
            for (int r = 0; r < K31_ROPE; ++r) keys[h][t * QK + NOPE + r] = ct[K31_KV_LORA + r];
            for (int r = 0; r < V; ++r) { double a = 0; for (int d = 0; d < K31_KV_LORA; ++d) a += (double)wv[r * K31_KV_LORA + d] * ct[d]; vals[h][t * V + r] = (float)a; }
        }
        /* reference attention (double softmax) */
        const float *qh = q + h * QK;
        double sc[T], mx = -INFINITY, z = 0;
        for (int t = 0; t < T; ++t) { double a = 0; for (int d = 0; d < QK; ++d) a += (double)qh[d] * keys[h][t * QK + d]; sc[t] = a / sqrt((double)QK); mx = fmax(mx, sc[t]); }
        for (int t = 0; t < T; ++t) z += exp(sc[t] - mx);
        for (int r = 0; r < V; ++r) { double a = 0; for (int t = 0; t < T; ++t) a += vals[h][t * V + r] * exp(sc[t] - mx) / z; ref[h * V + r] = (float)a; }
        /* absorb: qa = W_UK^T q_nope, q_rope */
        for (int d = 0; d < K31_KV_LORA; ++d) { double a = 0; for (int r = 0; r < NOPE; ++r) a += (double)wk[r * K31_KV_LORA + d] * qh[r]; qa[h * K31_LAT + d] = (float)a; }
        for (int r = 0; r < K31_ROPE; ++r) qa[h * K31_LAT + K31_KV_LORA + r] = qh[NOPE + r];
    }
    k31_latent_attention(qa, H, cache, NULL, T, 1.0f / sqrtf((float)QK), ol);
    for (int h = 0; h < H; ++h) {
        const float *wv = wkv + (size_t)h * (NOPE + V) * K31_KV_LORA + (size_t)NOPE * K31_KV_LORA;
        for (int r = 0; r < V; ++r) { double a = 0; for (int d = 0; d < K31_KV_LORA; ++d) a += (double)wv[r * K31_KV_LORA + d] * ol[h * K31_KV_LORA + d]; out[h * V + r] = (float)a; }
    }
    report("latent_attention_vs_decompressed", relerr_f(out, ref, H * V), 1e-5);
#if defined(K31_K3REF)
    double w3 = 0;
    for (int h = 0; h < H; ++h) {
        float o3[V];
        k3_attention_ref(o3, q + h * QK, keys[h], vals[h], T, QK, V);
        w3 = fmax(w3, relerr_f(out + h * V, o3, V));
    }
    report("latent_attention_vs_k3", w3, 1e-5);
#endif
    /* selected subset = full attention over just those tokens */
    int sel[64]; for (int i = 0; i < 64; ++i) sel[i] = i * 4 + 3;
    static uint16_t sub[64 * K31_LAT];
    for (int i = 0; i < 64; ++i) memcpy(sub + (size_t)i * K31_LAT, cache + (size_t)sel[i] * K31_LAT, K31_LAT * 2);
    static float o1[H * K31_KV_LORA], o2[H * K31_KV_LORA];
    k31_latent_attention(qa, H, cache, sel, 64, 0.07f, o1);
    k31_latent_attention(qa, H, sub, NULL, 64, 0.07f, o2);
    report("latent_attention_selected", relerr_f(o1, o2, H * K31_KV_LORA), 0.0);
}

static void t_bf16(void) {
    enum { R = 256, Cc = K31_HIDDEN };
    static uint16_t w[R * Cc];
    static float x[Cc], y[R];
    static double r[R];
    for (size_t i = 0; i < (size_t)R * Cc; ++i) w[i] = k31_f32_to_bf16(frand() * 0.05f);
    fill(x, Cc, 1.0f);
    k31_bf16_matvec(w, R, Cc, x, y);
    for (int i = 0; i < R; ++i) { double a = 0; for (int c = 0; c < Cc; ++c) a += (double)k31_bf16_to_f32(w[(size_t)i * Cc + c]) * x[c]; r[i] = a; }
    report("bf16_matvec", relerr(y, r, R), 1e-5);
    {
        static float ya[1000], yb[1000]; static double yr[1000];
        fill(ya, 1000, 1.0f); memcpy(yb, ya, sizeof ya);
        for (int i = 0; i < 1000; ++i) yr[i] = ya[i] + (double)k31_bf16_to_f32(w[i]) * 0.37;
        k31_bf16_axpy(ya, w, 0.37f, 1000);
        report("bf16_axpy", relerr(ya, yr, 1000), 1e-6);
        (void)yb;
    }
    report("bf16_round", k31_f32_to_bf16(1.00390625f) != 0x3f80 || k31_f32_to_bf16(1.01171875f) != 0x3f82, 0);
}

/* MXFP4 rows (odd and even block counts) against a double oracle of the same e2m1 / E8M0 values */
static void t_mxfp4(void) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int colss[2] = {3584, 3072 + 32}, R = 37;
    for (int t = 0; t < 2; ++t) {
        const int C = colss[t], nb = C / 32;
        uint8_t *w = malloc((size_t)R * nb * 17);
        int8_t *aq = malloc((size_t)C);
        float *ad = malloc(sizeof(float) * nb), *y = malloc(sizeof(float) * R);
        double *r = malloc(sizeof(double) * R);
        for (size_t i = 0; i < (size_t)R * nb * 17; ++i) w[i] = (uint8_t)(frand() * 128 + 128);
        for (int i = 0; i < R * nb; ++i) w[(size_t)i * 17] = (uint8_t)(120 + (int)((frand() + 1) * 4));
        for (int i = 0; i < C; ++i) aq[i] = (int8_t)(frand() * 127);
        for (int i = 0; i < nb; ++i) ad[i] = 0.01f + 0.001f * i;
        for (int rr = 0; rr < R; ++rr) {
            double a = 0;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *bl = w + ((size_t)rr * nb + b) * 17;
                const double d = ldexp(1.0, bl[0] - 128) * ad[b];
                for (int j = 0; j < 16; ++j) a += d * (kv[bl[1 + j] & 15] * aq[32 * b + j] + kv[bl[1 + j] >> 4] * aq[32 * b + j + 16]);
            }
            r[rr] = a;
        }
        k31_mxfp4_rows(w, C, 0, R, aq, ad, y);
        report(t ? "mxfp4_rows_odd" : "mxfp4_rows", relerr(y, r, R), 1e-6);
        if (nb % 4 == 0) {   /* planar copy of the same rows */
            const size_t st = ((size_t)nb * 17 + 63) & ~(size_t)63;
            uint8_t *wp = aligned_alloc(64, st * R);
            int8_t *aqp = malloc((size_t)C); float *ad4 = malloc(sizeof(float) * nb * 4);
            memset(wp, 0, st * R);
            for (int rr = 0; rr < R; ++rr)
                for (int b = 0; b < nb; ++b) {
                    const uint8_t *bl = w + ((size_t)rr * nb + b) * 17;
                    wp[st * rr + (size_t)nb * 16 + b] = bl[0];
                    memcpy(wp + st * rr + (size_t)b * 16, bl + 1, 16);
                }
            k31_mxfp4p_act(aq, ad, nb, aqp, ad4);
            k31_mxfp4p_rows(wp, st, C, 0, R, aqp, ad4, y);
            report("mxfp4p_rows", relerr(y, r, R), 1e-6);
            free(wp); free(aqp); free(ad4);
        }
        free(w); free(aq); free(ad); free(y); free(r);
    }
}

static void t_tmatvec(void) {
    const int R = 7168 + 48, K = 128;
    uint16_t *wt = malloc(sizeof(uint16_t) * (size_t)R * K);
    float *x = malloc(sizeof(float) * K), *y = malloc(sizeof(float) * R);
    double *r = malloc(sizeof(double) * R);
    for (size_t i = 0; i < (size_t)R * K; ++i) wt[i] = k31_f32_to_bf16(frand());
    fill(x, K, 1.0f);
    for (int i = 0; i < R; ++i) { double a = 0; for (int c = 0; c < K; ++c) a += (double)k31_bf16_to_f32(wt[(size_t)c * R + i]) * x[c]; r[i] = a; }
    k31_bf16_tmatvec(wt, R, K, x, y, 0, 64 * 50);
    k31_bf16_tmatvec(wt, R, K, x, y, 64 * 50, R);
    report("bf16_tmatvec", relerr(y, r, R), 1e-6);
    free(wt); free(x); free(y); free(r);
}

int main(void) {
    t_mxfp4();
    t_tmatvec();
    t_norms();
    t_situ();
    t_conv_kda();
    t_attn_res();
    t_router();
    t_indexer();
    t_latent_attention();
    t_bf16();
    printf("K31K fails=%d\n", fails);
    return fails != 0;
}
