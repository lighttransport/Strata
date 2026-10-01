// GLM-5.3-Flash equations follow the Transformers glm5_next reference.
// https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm5_next
#include "strata/kernels/glm.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::kernels {
namespace {
__device__ float sig(float x) {
    return 1.f / (1.f + expf(-x));
}
__global__ void norm(const float *x, const float *w, const float *b, float *y, int n, float eps,
                     bool centered) {
    __shared__ float sums[256];
    const int t = threadIdx.x, row = blockIdx.x;
    x += (size_t)row * n;
    y += (size_t)row * n;
    float mean = 0, sum = 0;
    if (centered) {
        for (int i = t; i < n; i += 256)
            mean += x[i];
        sums[t] = mean;
        __syncthreads();
        for (int s = 128; s; s /= 2) {
            if (t < s)
                sums[t] += sums[t + s];
            __syncthreads();
        }
        mean = sums[0] / n;
        __syncthreads();
    }
    for (int i = t; i < n; i += 256) {
        const float v = x[i] - mean;
        sum += v * v;
    }
    sums[t] = sum;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s)
            sums[t] += sums[t + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / n + eps);
    for (int i = t; i < n; i += 256)
        y[i] = (x[i] - mean) * inv * (w ? w[i] : 1.f) + (b ? b[i] : 0.f);
}
__global__ void swiglu(const float *g, const float *u, float *y, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = limit > 0 ? fminf(g[i], limit) : g[i];
        const float b = limit > 0 ? fminf(limit, fmaxf(-limit, u[i])) : u[i];
        y[i] = a * sig(a) * b;
    }
}
__global__ void route(const float *logits, const float *bias, int *ids, float *weights, int ne, int k,
                      float scale) {
    if (threadIdx.x || blockIdx.x)
        return;
    float sum = 0;
    for (int j = 0; j < k; ++j) {
        float best = -INFINITY;
        int id = -1;
        for (int e = 0; e < ne; ++e) {
            bool used = false;
            for (int i = 0; i < j; ++i)
                used |= ids[i] == e;
            const float p = sig(logits[e]);
            const float score = p + bias[e];
            if (!used && score > best) {
                best = score;
                id = e;
            }
        }
        ids[j] = id;
        weights[j] = sig(logits[id]);
        sum += weights[j];
    }
    for (int j = 0; j < k; ++j)
        weights[j] *= scale / (sum + 1e-20f);
}
// One warp caches each expert score once. Ties choose the first expert, as
// in the serial scan; lane zero retains the original normalization order.
__global__ void route_warp(const float *logits, const float *bias, int *ids, float *weights,
                           int ne, int k, float scale) {
    const int lane = threadIdx.x;
    float scores[16];
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        scores[i] = e < ne ? sig(logits[e]) + bias[e] : -INFINITY;
    }
    float sum = 0;
    for (int j = 0; j < k; ++j) {
        float best = -INFINITY;
        int id = 0x7fffffff;
        for (int i = 0; i < 16; ++i) {
            const int e = lane + 32 * i;
            if (e < ne && scores[i] > best) { best = scores[i]; id = e; }
        }
        for (int offset = 16; offset; offset /= 2) {
            const float other = __shfl_down_sync(0xffffffff, best, offset);
            const int other_id = __shfl_down_sync(0xffffffff, id, offset);
            if (other > best || (other == best && other_id < id)) {
                best = other; id = other_id;
            }
        }
        const int selected = __shfl_sync(0xffffffff, id, 0);
        if (lane == 0) {
            ids[j] = selected;
            weights[j] = sig(logits[selected]);
            sum += weights[j];
        }
        for (int i = 0; i < 16; ++i)
            if (lane + 32 * i == selected) scores[i] = -INFINITY;
    }
    if (lane == 0)
        for (int j = 0; j < k; ++j) weights[j] *= scale / (sum + 1e-20f);
}

__global__ void mhc_coeff(const float *p, const float *base, const float *scale, float *c, int iters,
                          float eps) {
    if (threadIdx.x)
        return;
    for (int i = 0; i < 4; ++i) {
        c[i] = sig(p[i] * scale[0] + base[i]) + eps;
        c[4 + i] = 2 * sig(p[4 + i] * scale[1] + base[4 + i]);
    }
    for (int i = 0; i < 4; ++i) {
        float mx = -INFINITY, sum = 0;
        for (int j = 0; j < 4; ++j)
            mx = fmaxf(mx, p[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]);
        for (int j = 0; j < 4; ++j) {
            c[8 + i * 4 + j] = expf(p[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j] - mx);
            sum += c[8 + i * 4 + j];
        }
        for (int j = 0; j < 4; ++j)
            c[8 + i * 4 + j] = c[8 + i * 4 + j] / sum + eps;
    }
    for (int n = 0; n < iters; ++n) {
        if (n > 0)
            for (int i = 0; i < 4; ++i) {
                float sum = eps;
                for (int j = 0; j < 4; ++j)
                    sum += c[8 + i * 4 + j];
                for (int j = 0; j < 4; ++j)
                    c[8 + i * 4 + j] /= sum;
            }
        for (int j = 0; j < 4; ++j) {
            float sum = eps;
            for (int i = 0; i < 4; ++i)
                sum += c[8 + i * 4 + j];
            for (int i = 0; i < 4; ++i)
                c[8 + i * 4 + j] /= sum;
        }
    }
}
__global__ void hc_read(const float *r, const float *c, float *x, int n) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < n) {
        float sum = 0;
        for (int i = 0; i < 4; ++i)
            sum += c[i] * r[i * n + d];
        x[d] = sum;
    }
}
__global__ void hc_write(const float *r, const float *c, const float *y, float *out, int n) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n)
        return;
    // Keep all input streams in registers so in-place updates are safe.
    float inputs[4];
    for (int i = 0; i < 4; ++i)
        inputs[i] = r[i * n + d];
    for (int j = 0; j < 4; ++j) {
        float sum = c[4 + j] * y[d];
        for (int i = 0; i < 4; ++i)
            sum += c[8 + i * 4 + j] * inputs[i];
        out[j * n + d] = sum;
    }
}
__global__ void hyper_head(const float *r, float *out, int n) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < n)
        out[d] = (r[d] + r[n + d] + r[2 * n + d] + r[3 * n + d]) * .25f;
}
__global__ void conv(const float *x, const float *w, float *history, float *y, int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float sum = x[i] * w[i * k + k - 1];
    for (int j = 0; j < k - 1; ++j)
        sum += history[i * (k - 1) + j] * w[i * k + j];
    for (int j = 0; j < k - 2; ++j)
        history[i * (k - 1) + j] = history[i * (k - 1) + j + 1];
    if (k > 1)
        history[i * (k - 1) + k - 2] = x[i];
    y[i] = sum * sig(sum);
}
__global__ void kda_gate(const float *x, const float *bias, const float *a, float *out, int h, int d,
                         float lower) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < h * d)
        out[i] = lower * sig(-a[i / d] * (x[i] + bias[i]));
}
__global__ void kda_step(float *state, const float *q, const float *k, const float *v, const float *g,
                         const float *beta, float *out, int d) {
    const int h = blockIdx.x, j = threadIdx.x;
    if (j >= d)
        return;
    q += h * d;
    k += h * d;
    v += h * d;
    g += h * d;
    out += h * d;
    state += (size_t)h * d * d;
    float qs = 1e-6f, ks = 1e-6f, mem = 0;
    for (int i = 0; i < d; ++i) {
        qs += q[i] * q[i];
        ks += k[i] * k[i];
    }
    const float ki = rsqrtf(ks), qi = rsqrtf(qs) * rsqrtf((float)d);
    for (int i = 0; i < d; ++i) {
        state[i * d + j] *= expf(g[i]);
        mem += state[i * d + j] * (k[i] * ki);
    }
    const float delta = (v[j] - mem) * sig(beta[h]);
    float sum = 0;
    for (int i = 0; i < d; ++i) {
        const float s = state[i * d + j] + k[i] * ki * delta;
        state[i * d + j] = s;
        sum += s * q[i] * qi;
    }
    out[j] = sum;
}
__global__ void kda_output(const float *x, const float *gate, const float *w, float *out, int d, float eps) {
    const int h = blockIdx.x, j = threadIdx.x;
    if (j >= d)
        return;
    float sum = 0;
    for (int i = 0; i < d; ++i)
        sum += x[h * d + i] * x[h * d + i];
    out[h * d + j] = x[h * d + j] * rsqrtf(sum / d + eps) * w[j] * sig(gate[h * d + j]);
}
__global__ void pool(const float *keys, const float *gates, const float *ape, float *out, int p, int dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim)
        return;
    float mx = -INFINITY, sum = 0, value = 0;
    for (int i = 0; i < p; ++i)
        mx = fmaxf(mx, gates[i * dim + d] + ape[i * dim + d]);
    for (int i = 0; i < p; ++i) {
        const float w = expf(gates[i * dim + d] + ape[i * dim + d] - mx);
        sum += w;
        value += w * keys[i * dim + d];
    }
    out[d] = value / sum;
}
__global__ void index_score(const float *q, const float *w, const float *keys, float *scores, int heads,
                            int dim, int pools) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pools)
        return;
    float sum = 0;
    for (int h = 0; h < heads; ++h) {
        float dot = 0;
        for (int d = 0; d < dim; ++d)
            dot += q[h * dim + d] * keys[p * dim + d];
        sum += w[h] * fmaxf(dot * rsqrtf((float)dim), 0.f);
    }
    scores[p] = sum * rsqrtf((float)heads);
}
__device__ unsigned long long score_key(float score, int index) {
    // Ordered float bits, then inverse index: equal scores select lower IDs.
    const unsigned bits = __float_as_uint(score == 0 ? 0.f : score);
    const unsigned ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    return ((unsigned long long)ordered << 32) | (0xffffffffu - (unsigned)index);
}
__global__ void index_select(const float *scores, int *out, int *count, int tokens, int pool, int topk) {
    const int pools = tokens / pool, selected = min(topk / pool, pools);
    __shared__ unsigned histogram[256], rank;
    __shared__ unsigned long long prefix, mask;
    if (!threadIdx.x) {
        prefix = 0;
        mask = 0;
        rank = selected;
    }
    __syncthreads();
    // Eight radix passes over pools, rather than rescanning every prior winner.
    if (selected && selected < pools)
        for (int shift = 56; shift >= 0; shift -= 8) {
            histogram[threadIdx.x] = 0;
            __syncthreads();
            for (int p = threadIdx.x; p < pools; p += blockDim.x) {
                const auto key = score_key(scores[p], p);
                if ((key & mask) == prefix)
                    atomicAdd(&histogram[(key >> shift) & 255], 1u);
            }
            __syncthreads();
            if (!threadIdx.x) {
                for (int bin = 255; bin >= 0; --bin) {
                    if (rank > histogram[bin])
                        rank -= histogram[bin];
                    else {
                        prefix |= (unsigned long long)bin << shift;
                        break;
                    }
                }
                mask |= 255ull << shift;
            }
            __syncthreads();
        }
    if (threadIdx.x)
        return;
    int used = 0;
    for (int p = 0; p < pools; ++p)
        if (selected && score_key(scores[p], p) >= prefix) {
            // Sort only the winners (at most 512 for GLM), using output as scratch.
            int slot = used++;
            while (slot > 0 && score_key(scores[p], p) >
                                   score_key(scores[out[(slot - 1) * pool]], out[(slot - 1) * pool])) {
                out[slot * pool] = out[(slot - 1) * pool];
                --slot;
            }
            out[slot * pool] = p;
        }
    for (int i = 0; i < used; ++i) {
        const int p = out[i * pool];
        for (int j = 0; j < pool; ++j)
            out[i * pool + j] = p * pool + j;
    }
    int n = used * pool;
    for (int t = pools * pool; t < tokens; ++t)
        out[n++] = t;
    *count = n;
}
__global__ void mla(const float *q, const float *cache, const int *ids, const int *count, float *out, int dim,
                    float scale) {
    const int h = blockIdx.x;
    extern __shared__ float weights[];
    const int n = *count;
    // One block per head, streamed softmax, with no context-sized scratch.
    __shared__ float maximum, denominator, alpha, weight;
    if (!threadIdx.x) {
        maximum = -INFINITY;
        denominator = 0;
    }
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[h * dim + d] = 0;
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        float dot = 0;
        const float *kv = cache + (size_t)ids[i] * dim;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            dot += q[h * dim + d] * kv[d];
        weights[threadIdx.x] = dot;
        __syncthreads();
        for (int s = blockDim.x / 2; s; s /= 2) {
            if (threadIdx.x < s)
                weights[threadIdx.x] += weights[threadIdx.x + s];
            __syncthreads();
        }
        if (!threadIdx.x) {
            const float score = weights[0] * scale, next = fmaxf(maximum, score);
            alpha = expf(maximum - next);
            weight = expf(score - next);
            denominator = denominator * alpha + weight;
            maximum = next;
        }
        __syncthreads();
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            out[h * dim + d] = out[h * dim + d] * alpha + kv[d] * weight;
        __syncthreads();
    }
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[h * dim + d] = n ? out[h * dim + d] / denominator : 0.f;
}
void check() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void width(int n) {
    if (n < 1)
        throw std::invalid_argument("GLM: non-positive width");
}
} // namespace

void glm_rms_norm(const float *x, const float *w, float *y, int n, int tokens, float eps, void *s) {
    width(n);
    width(tokens);
    norm<<<tokens, 256, 0, (cudaStream_t)s>>>(x, w, nullptr, y, n, eps, false);
    check();
}
void glm_layer_norm(const float *x, const float *w, const float *b, float *y, int n, float eps, void *s) {
    width(n);
    norm<<<1, 256, 0, (cudaStream_t)s>>>(x, w, b, y, n, eps, true);
    check();
}
void glm_layer_norm_batch(const float *x, const float *w, const float *b, float *y, int n, int tokens, float eps, void *s) {
    width(n); width(tokens);
    norm<<<tokens, 256, 0, (cudaStream_t)s>>>(x, w, b, y, n, eps, true);
    check();
}
void glm_swiglu(const float *g, const float *u, float *y, int n, float limit, void *s) {
    width(n);
    swiglu<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(g, u, y, n, limit);
    check();
}
void glm_router(const float *l, const float *b, int *ids, float *w, int ne, int k, float scale, void *s) {
    if (ne < 1 || k < 1 || k > ne)
        throw std::invalid_argument("GLM: invalid routing geometry");
    if (ne <= 512)
        route_warp<<<1, 32, 0, (cudaStream_t)s>>>(l, b, ids, w, ne, k, scale);
    else
        route<<<1, 1, 0, (cudaStream_t)s>>>(l, b, ids, w, ne, k, scale);
    check();
}
void glm_mhc_read(const float *r, const float *p, const float *b, const float *scale, float *c, float *x,
                  int n, int it, float eps, void *s) {
    width(n);
    width(it);
    mhc_coeff<<<1, 1, 0, (cudaStream_t)s>>>(p, b, scale, c, it, eps);
    hc_read<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, c, x, n);
    check();
}
void glm_mhc_write(const float *r, const float *c, const float *y, float *out, int n, void *s) {
    width(n);
    hc_write<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, c, y, out, n);
    check();
}
void glm_hyper_head(const float *r, float *out, int n, void *s) {
    width(n);
    hyper_head<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, out, n);
    check();
}
void glm_conv(const float *x, const float *w, float *history, float *y, int n, int k, void *s) {
    width(n);
    width(k);
    conv<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, w, history, y, n, k);
    check();
}
void glm_kda_gate(const float *x, const float *b, const float *a, float *y, int h, int d, float lower,
                  void *s) {
    width(h);
    width(d);
    kda_gate<<<(h * d + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, b, a, y, h, d, lower);
    check();
}
void glm_kda_step(float *state, const float *q, const float *k, const float *v, const float *g,
                  const float *b, float *out, int h, int d, void *s) {
    width(h);
    if (d < 1 || d > 1024)
        throw std::invalid_argument("GLM: invalid KDA dimension");
    kda_step<<<h, d, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, d);
    check();
}
void glm_kda_output(const float *x, const float *gate, const float *w, float *out, int h, int d, float eps,
                    void *s) {
    width(h);
    if (d < 1 || d > 1024)
        throw std::invalid_argument("GLM: invalid KDA dimension");
    kda_output<<<h, d, 0, (cudaStream_t)s>>>(x, gate, w, out, d, eps);
    check();
}
void glm_index_pool(const float *keys, const float *gates, const float *ape, float *out, int p, int d,
                    void *s) {
    width(p);
    width(d);
    pool<<<(d + 255) / 256, 256, 0, (cudaStream_t)s>>>(keys, gates, ape, out, p, d);
    check();
}
void glm_index_score(const float *q, const float *w, const float *keys, float *scores, int h, int d, int n,
                     void *s) {
    width(h);
    width(d);
    if (n < 0)
        throw std::invalid_argument("GLM: invalid pool count");
    if (n)
        index_score<<<(n + 63) / 64, 64, 0, (cudaStream_t)s>>>(q, w, keys, scores, h, d, n);
    check();
}
void glm_index_select(const float *scores, int *ids, int *count, int n, int p, int k, void *s) {
    width(n);
    width(p);
    if (k < p)
        throw std::invalid_argument("GLM: invalid sparse budget");
    index_select<<<1, 256, 0, (cudaStream_t)s>>>(scores, ids, count, n, p, k);
    check();
}
void glm_mla(const float *q, const float *cache, const int *ids, const int *count, float *out, int h, int d,
             float scale, void *s) {
    width(h);
    width(d);
    mla<<<h, 256, 256 * sizeof(float), (cudaStream_t)s>>>(q, cache, ids, count, out, d, scale);
    check();
}
} // namespace strata::kernels
