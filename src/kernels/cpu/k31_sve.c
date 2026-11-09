/* K3.1 Pro CPU kernels; see include/strata/kernels/cpu/k31_sve.h. KDA, residual, router and SiTU follow the K3
 * A64FX experiments (k3_kernels.h), including where a sum is kept in double. */
#include "strata/kernels/cpu/k31_sve.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

float k31_sigmoid(float x) {
    if (x >= 0.0f) { float z = expf(-x); return 1.0f / (1.0f + z); }
    float z = expf(x);
    return z / (1.0f + z);
}

uint16_t k31_f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);   /* quiet NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
float k31_bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

uint16_t k31_f32_to_f16(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
float k31_f16_to_f32(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

static double dot_d(const float *a, const float *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * b[i];
    return s;
}

void k31_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    const float inv = (float)(1.0 / sqrt(dot_d(x, x, n) / n + eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

void k31_layernorm(float *out, const float *x, const float *w, const float *b, int n, float eps) {
    double m = 0.0, v = 0.0;
    for (int i = 0; i < n; ++i) m += x[i];
    m /= n;
    for (int i = 0; i < n; ++i) { const double d = x[i] - m; v += d * d; }
    const float inv = (float)(1.0 / sqrt(v / n + eps)), mf = (float)m;
    for (int i = 0; i < n; ++i) out[i] = (x[i] - mf) * inv * w[i] + (b ? b[i] : 0.0f);
}

void k31_gated_rmsnorm(float *out, const float *x, const float *gate, const float *w, int n, float eps) {
    k31_rmsnorm(out, x, w, n, eps);
    for (int i = 0; i < n; ++i) out[i] *= k31_sigmoid(gate[i]);
}

void k31_l2_normalize(float *x, int n, float eps) {
    const float inv = 1.0f / sqrtf((float)dot_d(x, x, n) + eps);
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

void k31_silu_inplace(float *x, int n) {
    for (int i = 0; i < n; ++i) x[i] = x[i] * k31_sigmoid(x[i]);
}

void k31_situ(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = 4.0f * tanhf(gate[i] * 0.25f) * k31_sigmoid(gate[i]) * 25.0f * tanhf(up[i] * 0.04f);
}

#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t exp2_fexpa(svbool_t pg, svfloat32_t x) {
    const svfloat32_t shift = svdup_f32(204927.0f);
    const svfloat32_t z = svadd_f32_x(pg, x, shift);
    const svfloat32_t residual = svsub_f32_x(pg, x, svsub_f32_x(pg, z, shift));
    const svfloat32_t scale = svexpa_f32(svreinterpret_u32_f32(z));
    return svmul_f32_x(pg, scale, svmla_n_f32_x(pg, svdup_f32(1.0f), residual, 0.6931471805599453f));
}
static inline svfloat32_t sigmoid_fast(svbool_t pg, svfloat32_t x) {
    svfloat32_t t = svmul_n_f32_x(pg, x, -1.4426950408889634f);
    t = svmax_n_f32_x(pg, svmin_n_f32_x(pg, t, 80.0f), -80.0f);
    const svfloat32_t den = svadd_n_f32_x(pg, exp2_fexpa(pg, t), 1.0f);
    const svfloat32_t inv = svrecpe_f32(den);
    return svmul_f32_x(pg, inv, svrecps_f32(den, inv));
}
#endif

void k31_situ_fast(float *out, const float *gate, const float *up, int n) {
#if defined(__ARM_FEATURE_SVE)
    const int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        const svbool_t pg = svwhilelt_b32(i, n);
        const svfloat32_t g = svld1(pg, gate + i), u = svld1(pg, up + i);
        const svfloat32_t tg = svsub_n_f32_x(pg, svmul_n_f32_x(pg, sigmoid_fast(pg, svmul_n_f32_x(pg, g, 0.5f)), 2.0f), 1.0f);
        const svfloat32_t su = svsub_n_f32_x(pg, svmul_n_f32_x(pg, sigmoid_fast(pg, svmul_n_f32_x(pg, u, 0.08f)), 2.0f), 1.0f);
        svfloat32_t y = svmul_f32_x(pg, svmul_f32_x(pg, tg, sigmoid_fast(pg, g)), su);
        svst1(pg, out + i, svmul_n_f32_x(pg, y, 100.0f));
    }
#else
    k31_situ(out, gate, up, n);
#endif
}

void k31_conv_step(float *out, const float *x, float *state, const float *weight, int channels, int kernel) {
    for (int c = 0; c < channels; ++c) {
        float *s = state + (size_t)c * (kernel - 1);
        const float *w = weight + (size_t)c * kernel;
        float y = 0.0f;
        for (int j = 0; j < kernel - 1; ++j) y += w[j] * s[j];
        y += w[kernel - 1] * x[c];
        out[c] = y;
        for (int j = 0; j + 1 < kernel - 1; ++j) s[j] = s[j + 1];
        if (kernel > 1) s[kernel - 2] = x[c];
    }
}

void k31_kda_log_decay(float *out, const float *g_raw, const float *a, const float *dt_bias, int heads, int key_dim,
                       float lower_bound) {
    for (int h = 0; h < heads; ++h)
        for (int d = 0; d < key_dim; ++d) {
            const int i = h * key_dim + d;
            out[i] = lower_bound * k31_sigmoid((-a[h]) * (g_raw[i] + dt_bias[i]));
        }
}

void k31_kda_row(float *out, const float *q, const float *k, const float *v, const float *decay, const float *beta,
                 float *state, int key_dim, int value_dim, int task) {
    const int h = task / value_dim, j = task % value_dim;
    const float *qh = q + (size_t)h * key_dim, *kh = k + (size_t)h * key_dim, *dh = decay + (size_t)h * key_dim;
    float *row = state + ((size_t)h * value_dim + j) * key_dim;
#if defined(__ARM_FEATURE_SVE)
    const int vl = (int)svcntw();
    for (int d = 0; d < key_dim; d += vl) {
        const svbool_t pg = svwhilelt_b32(d, key_dim);
        svst1(pg, row + d, svmul_f32_x(pg, svld1(pg, row + d), svld1(pg, dh + d)));
    }
#else
    for (int d = 0; d < key_dim; ++d) row[d] *= dh[d];
#endif
    const float delta = beta[h] * (v[(size_t)h * value_dim + j] - (float)dot_d(kh, row, key_dim));
#if defined(__ARM_FEATURE_SVE)
    for (int d = 0; d < key_dim; d += vl) {
        const svbool_t pg = svwhilelt_b32(d, key_dim);
        svst1(pg, row + d, svmla_n_f32_x(pg, svld1(pg, row + d), svld1(pg, kh + d), delta));
    }
#else
    for (int d = 0; d < key_dim; ++d) row[d] += kh[d] * delta;
#endif
    out[(size_t)h * value_dim + j] = (float)dot_d(qh, row, key_dim) * (1.0f / sqrtf((float)key_dim));
}

void k31_kda_step(float *out, const float *q, const float *k, const float *v, const float *decay, const float *beta,
                  float *state, int heads, int key_dim, int value_dim) {
    const int tasks = heads * value_dim;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < tasks; ++t) k31_kda_row(out, q, k, v, decay, beta, state, key_dim, value_dim, t);
}

void k31_attn_res(float *out, const float *prefix, const float *blocks, int block_count, const float *norm_w,
                  const float *proj_w, int hidden, float eps) {
    if (block_count <= 0) { if (out != prefix) memcpy(out, prefix, (size_t)hidden * sizeof(float)); return; }
    const int count = block_count + 1;
    float scores[K31_LAYERS / K31_RES_BLOCK + 2], mx = -INFINITY;
    for (int c = 0; c < count; ++c) {
        const float *v = c == block_count ? prefix : blocks + (size_t)c * hidden;
        const float inv = 1.0f / sqrtf((float)(dot_d(v, v, hidden) / hidden) + eps);
        double s = 0.0;
        for (int j = 0; j < hidden; ++j) s += (double)v[j] * inv * norm_w[j] * (proj_w ? proj_w[j] : 1.0f);
        scores[c] = (float)s;
        if (scores[c] > mx) mx = scores[c];
    }
    float p[K31_LAYERS / K31_RES_BLOCK + 2], z = 0.0f;
    for (int c = 0; c < count; ++c) z += (p[c] = expf(scores[c] - mx));
    for (int c = 0; c < count; ++c) p[c] /= z;
    for (int j = 0; j < hidden; ++j) {
        double v = 0.0;
        for (int c = 0; c < count; ++c) v += (double)(c == block_count ? prefix : blocks + (size_t)c * hidden)[j] * p[c];
        out[j] = (float)v;
    }
}

void k31_router_topk(const float *logits, const float *bias, int experts, int topk, float scale, int *idx, float *w) {
    float *prob = (float *)malloc((size_t)experts * sizeof(float) * 2), *score = prob + experts;
    for (int e = 0; e < experts; ++e) { prob[e] = k31_sigmoid(logits[e]); score[e] = prob[e] + (bias ? bias[e] : 0.0f); }
    double sum = 0.0;
    for (int k = 0; k < topk; ++k) {
        int best = -1; float bs = -INFINITY;
        for (int e = 0; e < experts; ++e) if (score[e] > bs) { best = e; bs = score[e]; }
        idx[k] = best; score[best] = -INFINITY;
        sum += (w[k] = prob[best]);
    }
    for (int k = 0; k < topk; ++k) w[k] = (float)(w[k] / sum) * scale;
    free(prob);
}

void k31_index_scores(const float *q, const float *w, int heads, int dim, const uint16_t *keys, int n, float w_scale,
                      float *scores) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    if (heads == 32 && dim <= 1024 && dim % 16 == 0) {
        /* q transposed to [dim][32]: per key element one broadcast FMA into two 16-head accumulators, no reduction until the
         * end; the key row's F16 values widened 16 at a time (low halves of 32-bit containers). Head sums in head order. */
        float qt[1024 * 32], ww[32];
        for (int h = 0; h < 32; ++h) { ww[h] = w[h] * w_scale; for (int d = 0; d < dim; ++d) qt[d * 32 + h] = q[(size_t)h * dim + d]; }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int t = 0; t < n; ++t) {
            const uint16_t *k = keys + (size_t)t * dim;
            const svbool_t all = svptrue_b32();
            svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
            float kr[1024];
            for (int d = 0; d < dim; d += 16)
                svst1(all, kr + d, svcvt_f32_f16_x(all, svreinterpret_f16_u32(svld1uh_u32(all, k + d))));
            for (int d = 0; d < dim; ++d) {
                a0 = svmla_n_f32_x(all, a0, svld1(all, qt + d * 32), kr[d]);
                a1 = svmla_n_f32_x(all, a1, svld1(all, qt + d * 32 + 16), kr[d]);
            }
            a0 = svmul_f32_x(all, svmax_n_f32_x(all, a0, 0.0f), svld1(all, ww));
            a1 = svmul_f32_x(all, svmax_n_f32_x(all, a1, 0.0f), svld1(all, ww + 16));
            float o0[16], o1[16];
            svst1(all, o0, a0); svst1(all, o1, a1);
            float s = 0.0f;
            for (int h = 0; h < 16; ++h) s += o0[h];
            for (int h = 0; h < 16; ++h) s += o1[h];
            scores[t] = s;
        }
        return;
    }
#endif
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; ++t) {
        float kr[1024];
        const uint16_t *k = keys + (size_t)t * dim;
        for (int d = 0; d < dim; ++d) kr[d] = k31_f16_to_f32(k[d]);
        float s = 0.0f;
        for (int h = 0; h < heads; ++h) {
            const float *qh = q + (size_t)h * dim;
            float a = 0.0f;
            for (int d = 0; d < dim; ++d) a += qh[d] * kr[d];
            s += (a > 0.0f ? a : 0.0f) * (w[h] * w_scale);
        }
        scores[t] = s;
    }
}

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

int k31_topk_indices(const float *scores, int n, int k, int *out) {
    if (k >= n) { for (int i = 0; i < n; ++i) out[i] = i; return n; }
    /* threshold by partial selection: keep a min-heap of k (score, -index) */
    int m = 0;
    for (int t = 0; t < n; ++t) {
        if (m < k) {   /* sift up */
            int i = m++; out[i] = t;
            while (i > 0) {
                const int p = (i - 1) / 2;
                const float a = scores[out[i]], b = scores[out[p]];
                if (a < b || (a == b && out[i] > out[p])) { const int x = out[i]; out[i] = out[p]; out[p] = x; i = p; }
                else break;
            }
        } else {
            const float s = scores[t];
            const float r = scores[out[0]];
            if (!(s > r)) continue;   /* equal score: the lower (earlier) index stays */
            out[0] = t;
            int i = 0;
            for (;;) {   /* sift down */
                const int l = 2 * i + 1, rr = l + 1;
                int lo = i;
                if (l < k && (scores[out[l]] < scores[out[lo]] || (scores[out[l]] == scores[out[lo]] && out[l] > out[lo]))) lo = l;
                if (rr < k && (scores[out[rr]] < scores[out[lo]] || (scores[out[rr]] == scores[out[lo]] && out[rr] > out[lo]))) lo = rr;
                if (lo == i) break;
                const int x = out[i]; out[i] = out[lo]; out[lo] = x; i = lo;
            }
        }
    }
    qsort(out, (size_t)k, sizeof(int), cmp_int);
    return k;
}

void k31_latent_attention(const float *qa, int heads, const uint16_t *cache, const int *sel, int count, float scale,
                          float *out) {
    float *s = (float *)malloc((size_t)count * sizeof(float)), row[K31_LAT];
    for (int h = 0; h < heads; ++h) {
        const float *q = qa + (size_t)h * K31_LAT;
        float *o = out + (size_t)h * K31_KV_LORA;
        float mx = -INFINITY;
        for (int i = 0; i < count; ++i) {
            const uint16_t *c = cache + (size_t)(sel ? sel[i] : i) * K31_LAT;
            double a = 0.0;
            for (int d = 0; d < K31_LAT; ++d) a += (double)q[d] * k31_f16_to_f32(c[d]);
            s[i] = (float)a * scale;
            if (s[i] > mx) mx = s[i];
        }
        double z = 0.0;
        for (int i = 0; i < count; ++i) z += (s[i] = expf(s[i] - mx));
        for (int d = 0; d < K31_KV_LORA; ++d) o[d] = 0.0f;
        for (int i = 0; i < count; ++i) {
            const uint16_t *c = cache + (size_t)(sel ? sel[i] : i) * K31_LAT;
            const float p = (float)(s[i] / z);
            for (int d = 0; d < K31_KV_LORA; ++d) { row[d] = k31_f16_to_f32(c[d]); o[d] += p * row[d]; }
        }
    }
    free(s);
}

void k31_bf16_matvec(const uint16_t *w, int rows, int cols, const float *x, float *y) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const uint16_t *wr = w + (size_t)r * cols;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        int c = 0;
        for (; c + 4 <= cols; c += 4) {
            a0 += k31_bf16_to_f32(wr[c]) * x[c];
            a1 += k31_bf16_to_f32(wr[c + 1]) * x[c + 1];
            a2 += k31_bf16_to_f32(wr[c + 2]) * x[c + 2];
            a3 += k31_bf16_to_f32(wr[c + 3]) * x[c + 3];
        }
        for (; c < cols; ++c) a0 += k31_bf16_to_f32(wr[c]) * x[c];
        y[r] = (a0 + a1) + (a2 + a3);
    }
}

void k31_bf16_axpy(float *y, const uint16_t *w, float alpha, int n) {
#if defined(__ARM_FEATURE_SVE)
    const int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        const svbool_t pg = svwhilelt_b32(i, n);
        const svfloat32_t wv = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, w + i), 16));
        svst1(pg, y + i, svmla_n_f32_x(pg, svld1(pg, y + i), wv, alpha));
    }
#else
    for (int i = 0; i < n; ++i) y[i] += k31_bf16_to_f32(w[i]) * alpha;
#endif
}

/* ------------------------------------------------------------------------------------------------ MXFP4 x Q8_0 */
static const int8_t k31_kv_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float k31_e8m0h[256];
static int k31_e8m0h_ready;
static void k31_e8m0h_init(void)
{
    for (int e = 0; e < 256; ++e) {   /* E8M0 / 2 (the e2m1 values are stored doubled), as ggml */
        const uint32_t u = e < 2 ? 0x00200000u << e : (uint32_t) (e - 1) << 23;
        memcpy(&k31_e8m0h[e], &u, 4);
    }
    k31_e8m0h_ready = 1;
}

void k31_mxfp4_rows(const uint8_t *w, int cols, int r0, int r1, const int8_t *aq, const float *ad, float *y)
{
    if (!k31_e8m0h_ready) k31_e8m0h_init();
    const int nb = cols / 32;
    const size_t rb = (size_t) nb * 17;
#if defined(__ARM_FEATURE_SVE)
    if (svcntb() == 64) {
        static const uint8_t idx[64] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33};
        static const uint8_t hi[64] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        const svbool_t pall = svptrue_b8(), p34 = svwhilelt_b8(0, 34), p32 = svptrue_b32(), plo8 = svwhilelt_b32(0, 8);
        const svuint8_t IDX = svld1_u8(pall, idx);
        const svbool_t PHI = svcmpne_n_u8(pall, svld1_u8(pall, hi), 0);
        const svint8_t KV = svld1_s8(svwhilelt_b8(0, 16), k31_kv_mxfp4);
        const svint32_t z = svdup_s32(0);
        const int np = nb & ~1;
        for (int r = r0; r < r1; ++r) {
            const uint8_t *row = w + rb * (size_t) r;
            svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0;
#define K31_MX_STEP(ACC, I) do { \
                const svuint8_t raw = svld1_u8(p34, row + 17 * (I)); \
                const svuint8_t X = svtbl_u8(raw, IDX); \
                const svuint8_t v = svsel_u8(PHI, svlsr_n_u8_x(pall, X, 4), svand_n_u8_x(pall, X, 15)); \
                const svint32_t dt = svdot_s32(z, svtbl_s8(KV, v), svld1_s8(pall, aq + 32 * (I))); \
                const svfloat32_t s = svsel_f32(plo8, svdup_f32(k31_e8m0h[row[17 * (I)]] * ad[I]), \
                                                svdup_f32(k31_e8m0h[row[17 * ((I) + 1)]] * ad[(I) + 1])); \
                ACC = svmla_f32_x(p32, ACC, svcvt_f32_s32_x(p32, dt), s); } while (0)
            int i = 0;
            for (; i + 8 <= np; i += 8) { K31_MX_STEP(A0, i); K31_MX_STEP(A1, i + 2); K31_MX_STEP(A2, i + 4); K31_MX_STEP(A3, i + 6); }
            for (; i < np; i += 2) K31_MX_STEP(A0, i);
#undef K31_MX_STEP
            float t = svaddv_f32(p32, svadd_f32_x(p32, svadd_f32_x(p32, A0, A1), svadd_f32_x(p32, A2, A3)));
            for (; i < nb; ++i) {   /* odd block count: the last block scalar */
                const uint8_t *b = row + 17 * (size_t) i;
                int s = 0;
                for (int j = 0; j < 16; ++j) s += k31_kv_mxfp4[b[1 + j] & 15] * aq[32 * i + j] + k31_kv_mxfp4[b[1 + j] >> 4] * aq[32 * i + j + 16];
                t += (float) s * k31_e8m0h[b[0]] * ad[i];
            }
            y[r] = t;
        }
        return;
    }
#endif
    for (int r = r0; r < r1; ++r) {
        const uint8_t *row = w + rb * (size_t) r;
        float t = 0;
        for (int i = 0; i < nb; ++i) {
            const uint8_t *b = row + 17 * (size_t) i;
            int s = 0;
            for (int j = 0; j < 16; ++j) s += k31_kv_mxfp4[b[1 + j] & 15] * aq[32 * i + j] + k31_kv_mxfp4[b[1 + j] >> 4] * aq[32 * i + j + 16];
            t += (float) s * k31_e8m0h[b[0]] * ad[i];
        }
        y[r] = t;
    }
}

/* ------------------------------------------------------------------------------------------------ BF16 transposed */
void k31_bf16_tmatvec(const uint16_t *wt, int rows, int k, const float *x, float *y, int r0, int r1)
{
#if defined(__ARM_FEATURE_SVE)
    if (svcntw() == 16) {
        const svbool_t pt = svptrue_b32();
        int r = r0;
        for (; r + 64 <= r1; r += 64) {
            svfloat32_t A0 = svdup_f32(0), A1 = A0, A2 = A0, A3 = A0;
            for (int c = 0; c < k; ++c) {
                const uint16_t *wr = wt + (size_t) c * rows + r;
                const svfloat32_t xc = svdup_f32(x[c]);
#define K31_BFV(o) svreinterpret_f32_u32(svlsl_n_u32_x(pt, svld1uh_u32(pt, wr + (o)), 16))
                A0 = svmla_f32_x(pt, A0, K31_BFV(0), xc);
                A1 = svmla_f32_x(pt, A1, K31_BFV(16), xc);
                A2 = svmla_f32_x(pt, A2, K31_BFV(32), xc);
                A3 = svmla_f32_x(pt, A3, K31_BFV(48), xc);
            }
            svst1_f32(pt, y + r, A0); svst1_f32(pt, y + r + 16, A1); svst1_f32(pt, y + r + 32, A2); svst1_f32(pt, y + r + 48, A3);
        }
        for (; r < r1; r += 16) {
            const svbool_t p = svwhilelt_b32(r, r1);
            svfloat32_t A = svdup_f32(0);
            for (int c = 0; c < k; ++c)
                A = svmla_f32_x(p, A, svreinterpret_f32_u32(svlsl_n_u32_x(p, svld1uh_u32(p, wt + (size_t) c * rows + r), 16)), svdup_f32(x[c]));
            svst1_f32(p, y + r, A);
        }
#undef K31_BFV
        return;
    }
#endif
    for (int r = r0; r < r1; ++r) {
        float a = 0;
        for (int c = 0; c < k; ++c) a += k31_bf16_to_f32(wt[(size_t) c * rows + r]) * x[c];
        y[r] = a;
    }
}

/* ------------------------------------------------------------------------------------------------ planar MXFP4 */
void k31_mxfp4p_act(const int8_t *aq, const float *ad, int nb, int8_t *aqp, float *ad4)
{
    for (int g = 0; g < nb / 4; ++g)
        for (int b = 0; b < 4; ++b)
            for (int j = 0; j < 16; ++j) {
                aqp[128 * g + 16 * b + j] = aq[32 * (4 * g + b) + j];
                aqp[128 * g + 64 + 16 * b + j] = aq[32 * (4 * g + b) + 16 + j];
            }
    for (int b = 0; b < nb; ++b) for (int l = 0; l < 4; ++l) ad4[4 * b + l] = ad[b];
}

void k31_mxfp4p_rows(const uint8_t *w, size_t stride, int cols, int r0, int r1, const int8_t *aqp, const float *ad4, float *y)
{
    if (!k31_e8m0h_ready) k31_e8m0h_init();
    const int nb = cols / 32, ng = nb / 4;
#if defined(__ARM_FEATURE_SVE)
    if (svcntb() == 64) {
        static const uint32_t i4[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        const svbool_t pall = svptrue_b8(), p32 = svptrue_b32(), p4 = svwhilelt_b32(0, 4);
        const svuint32_t IDX4 = svld1_u32(p32, i4);
        const svint8_t KV = svld1_s8(svwhilelt_b8(0, 16), k31_kv_mxfp4);
        const svint32_t z = svdup_s32(0);
        for (int r = r0; r < r1; ++r) {
            const uint8_t *qs = w + stride * (size_t) r, *e = qs + (size_t) nb * 16;
            svfloat32_t A0 = svdup_f32(0), A1 = A0;
#define K31_MXP(ACC, G) do { \
                const svuint8_t v = svld1_u8(pall, qs + 64 * (G)); \
                const svint8_t ql = svtbl_s8(KV, svand_n_u8_x(pall, v, 15)), qh = svtbl_s8(KV, svlsr_n_u8_x(pall, v, 4)); \
                const svint32_t d = svdot_s32(svdot_s32(z, ql, svld1_s8(pall, aqp + 128 * (G))), qh, svld1_s8(pall, aqp + 128 * (G) + 64)); \
                const svuint32_t ev = svtbl_u32(svld1ub_u32(p4, e + 4 * (G)), IDX4); \
                const svfloat32_t fe = svreinterpret_f32_u32(svsel_u32(svcmpge_n_u32(p32, ev, 2), \
                                         svlsl_n_u32_x(p32, svsub_n_u32_x(p32, ev, 1), 23), svdup_u32(0))); \
                ACC = svmla_f32_x(p32, ACC, svcvt_f32_s32_x(p32, d), svmul_f32_x(p32, fe, svld1_f32(p32, ad4 + 16 * (G)))); } while (0)
            int g = 0;
            for (; g + 2 <= ng; g += 2) { K31_MXP(A0, g); K31_MXP(A1, g + 1); }
            for (; g < ng; ++g) K31_MXP(A0, g);
#undef K31_MXP
            y[r] = svaddv_f32(p32, svadd_f32_x(p32, A0, A1));
        }
        return;
    }
#endif
    for (int r = r0; r < r1; ++r) {
        const uint8_t *qs = w + stride * (size_t) r, *e = qs + (size_t) nb * 16;
        float t = 0;
        for (int b = 0; b < nb; ++b) {
            const int g = b / 4, bb = b % 4;
            int s = 0;
            for (int j = 0; j < 16; ++j)
                s += k31_kv_mxfp4[qs[16 * b + j] & 15] * aqp[128 * g + 16 * bb + j] + k31_kv_mxfp4[qs[16 * b + j] >> 4] * aqp[128 * g + 64 + 16 * bb + j];
            t += (float) s * (e[b] >= 2 ? k31_e8m0h[e[b]] : 0.0f) * ad4[4 * b];
        }
        y[r] = t;
    }
}
