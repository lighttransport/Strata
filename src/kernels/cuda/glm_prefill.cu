#include "strata/kernels/glm_prefill.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
namespace strata::kernels {
namespace {
__device__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
void check() {
    auto e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void half_cast(const float *x, uint16_t *y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        y[i] = __half_as_ushort(__float2half(x[i]));
}
__global__ void convolution(const float *x, const float *w, const float *history, float *y, int n, int kernel,
                            int tokens) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, c = i % n;
    float sum = x[i] * w[c * kernel + kernel - 1];
    for (int j = 0; j < kernel - 1; ++j) {
        int previous = t - (kernel - 1) + j;
        float v = previous < 0 ? history[c * (kernel - 1) + previous + kernel - 1] : x[previous * n + c];
        sum += v * w[c * kernel + j];
    }
    y[i] = sum * sigmoid(sum);
}
__global__ void conv_history(const float *x, float *history, int n, int kernel, int tokens) {
    int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= n)
        return;
    for (int j = 0; j < kernel - 1; ++j) {
        int t = tokens - (kernel - 1) + j;
        history[c * (kernel - 1) + j] = t < 0 ? history[c * (kernel - 1) + t + kernel - 1] : x[t * n + c];
    }
}
__global__ void routing(const float *l, const float *bias, int *ids, float *w, int ne, int top, float scale) {
    int t = blockIdx.x;
    l += t * ne;
    ids += t * top;
    w += t * top;
    float sum = 0;
    for (int j = 0; j < top; ++j) {
        float best = -INFINITY;
        int chosen = -1;
        for (int e = 0; e < ne; ++e) {
            bool used = false;
            for (int z = 0; z < j; ++z)
                used |= ids[z] == e;
            float score = sigmoid(l[e]) + bias[e];
            if (!used && score > best) {
                best = score;
                chosen = e;
            }
        }
        ids[j] = chosen;
        w[j] = sigmoid(l[chosen]);
        sum += w[j];
    }
    for (int j = 0; j < top; ++j)
        w[j] *= scale / (sum + 1e-20f);
}
__global__ void count_routes(const int *ids, int *b, int count) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < count)
        atomicAdd(b + ids[i] + 1, 1);
}
__global__ void prefix_routes(int *b, int *c, int ne) {
    for (int e = 1; e <= ne; ++e)
        b[e] += b[e - 1];
    for (int e = 0; e < ne; ++e)
        c[e] = b[e];
}
__global__ void scatter_routes(const int *ids, int *c, int *d, int *src, int top, int count) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < count) {
        int at = atomicAdd(c + ids[i], 1);
        d[at] = i;
        src[at] = i / top;
    }
}
__global__ void slice_bounds(const int *global, int *local, int start, int count, int offset, int rows) {
    int i = threadIdx.x;
    if (i <= count)
        local[i] = max(0, min(rows, global[start + i] - offset));
}
__global__ void route_sum(const float *r, const float *w, float *out, int n, int top, int tokens) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, d = i % n;
    float sum = 0;
    for (int j = 0; j < top; ++j)
        sum += w[t * top + j] * r[(t * top + j) * n + d];
    out[i] += sum;
}
__global__ void mhc(const float *p, const float *base, const float *scale, float *c, int iterations,
                    float eps) {
    p += blockIdx.x * 24;
    c += blockIdx.x * 24;
    for (int i = 0; i < 4; ++i) {
        c[i] = sigmoid(p[i] * scale[0] + base[i]) + eps;
        c[4 + i] = 2 * sigmoid(p[4 + i] * scale[1] + base[4 + i]);
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
    for (int it = 0; it < iterations; ++it) {
        if (it > 0)
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
__global__ void mhc_read(const float *r, const float *c, float *x, int n, int tokens) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, d = i % n;
    float sum = 0;
    for (int j = 0; j < 4; ++j)
        sum += c[t * 24 + j] * r[t * 4 * n + j * n + d];
    x[i] = sum;
}
__global__ void mhc_write(const float *r, const float *c, const float *y, float *out, int n, int tokens) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * tokens)
        return;
    int t = i / n, d = i % n;
    float in[4];
    for (int j = 0; j < 4; ++j)
        in[j] = r[t * 4 * n + j * n + d];
    for (int j = 0; j < 4; ++j) {
        float sum = c[t * 24 + 4 + j] * y[i];
        for (int z = 0; z < 4; ++z)
            sum += c[t * 24 + 8 + z * 4 + j] * in[z];
        out[t * 4 * n + j * n + d] = sum;
    }
}
__global__ void decay(const float *x, const float *b, const float *a, float *y, int heads, int dim,
                      int tokens, float lower) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < heads * dim * tokens) {
        int d = i % (heads * dim);
        y[i] = lower * sigmoid(-a[d / dim] * (x[i] + b[d]));
    }
}
__global__ void kda_chunk(float *state, const float *q, const float *k, const float *v, const float *g,
                          const float *beta, float *out, int heads, int tokens) {
    constexpr int dim = 128;
    int h = blockIdx.x, j = threadIdx.x;
    state += (size_t)h * dim * dim;
    float values[dim];
#pragma unroll
    for (int i = 0; i < dim; ++i)
        values[i] = state[i * dim + j];
    __shared__ float qr[dim], kn[dim], decay[dim], qi, ki, b;
    for (int t = 0; t < tokens; ++t) {
        int offset = (t * heads + h) * dim;
        if (!j) {
            float qs = 1e-6f, ks = 1e-6f;
            for (int i = 0; i < dim; ++i) {
                qs += q[offset + i] * q[offset + i];
                ks += k[offset + i] * k[offset + i];
            }
            ki = rsqrtf(ks);
            qi = rsqrtf(qs) * rsqrtf((float)dim);
            b = sigmoid(beta[t * heads + h]);
        }
        __syncthreads();
        qr[j] = q[offset + j];
        kn[j] = k[offset + j] * ki;
        decay[j] = expf(g[offset + j]);
        __syncthreads();
        float mem = 0;
#pragma unroll
        for (int i = 0; i < dim; ++i) {
            values[i] *= decay[i];
            mem += values[i] * kn[i];
        }
        float delta = (v[offset + j] - mem) * b, sum = 0;
#pragma unroll
        for (int i = 0; i < dim; ++i) {
            values[i] = values[i] + kn[i] * delta;
            sum += values[i] * qr[i] * qi;
        }
        out[offset + j] = sum;
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < dim; ++i)
        state[i * dim + j] = values[i];
}
__global__ void kda_output(const float *x, const float *g, const float *w, float *out, int dim, float eps) {
    int h = blockIdx.x, j = threadIdx.x;
    float sum = 0;
    for (int i = 0; i < dim; ++i)
        sum += x[h * dim + i] * x[h * dim + i];
    out[h * dim + j] = x[h * dim + j] * rsqrtf(sum / dim + eps) * w[j] * sigmoid(g[h * dim + j]);
}
__global__ void pooling(const float *keys, const float *gates, const float *ape, const float *pk,
                        const float *pg, float *out, int pos, int tokens, int pool, int dim) {
    int d = blockIdx.x * 256 + threadIdx.x, first = pos / pool, last = (pos + tokens) / pool;
    if (d >= (last - first) * dim)
        return;
    int p = first + d / dim, c = d % dim;
    float mx = -INFINITY, sum = 0, value = 0;
    for (int j = 0; j < pool; ++j) {
        int t = p * pool + j - pos;
        float g = t < 0 ? pg[(p * pool + j) % pool * dim + c] : gates[t * dim + c];
        mx = fmaxf(mx, g + ape[j * dim + c]);
    }
    for (int j = 0; j < pool; ++j) {
        int t = p * pool + j - pos;
        float g = t < 0 ? pg[(p * pool + j) % pool * dim + c] : gates[t * dim + c];
        float k = t < 0 ? pk[(p * pool + j) % pool * dim + c] : keys[t * dim + c];
        float w = expf(g + ape[j * dim + c] - mx);
        sum += w;
        value += w * k;
    }
    out[p * dim + c] = value / sum;
}
__global__ void pending(const float *keys, const float *gates, float *pk, float *pg, int pos, int tokens,
                        int pool, int dim) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= pool * dim)
        return;
    int j = i / dim, last = pos + tokens - 1;
    int absolute = last - ((last - j + pool) % pool), t = absolute - pos;
    if (t >= 0) {
        pk[i] = keys[t * dim + i % dim];
        pg[i] = gates[t * dim + i % dim];
    }
}
__global__ void index_reduce(const float *dots, const float *w, float *scores, int heads, int pools,
                             int queries, int pos, int pool, int dim) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= pools * queries)
        return;
    int q = i / pools, p = i % pools;
    if (p >= (pos + q + 1) / pool) {
        scores[i] = -INFINITY;
        return;
    }
    float sum = 0;
    for (int h = 0; h < heads; ++h)
        sum += w[q * heads + h] * fmaxf(dots[(q * heads + h) * pools + p] * rsqrtf((float)dim), 0.f);
    scores[i] = sum * rsqrtf((float)heads);
}
__device__ unsigned long long score_key(float score, int index) {
    unsigned bits = __float_as_uint(score == 0 ? 0.f : score),
             ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    return ((unsigned long long)ordered << 32) | (0xffffffffu - index);
}
__global__ void selection(const float *scores, int *ids, int *counts, int pools_total, int pos, int pool,
                          int top, int stride) {
    int q = blockIdx.x, tid = threadIdx.x, tokens = pos + q + 1, pools = tokens / pool,
        n = min(top / pool, pools);
    scores += q * pools_total;
    ids += q * stride;
    __shared__ unsigned histogram[256], rank;
    __shared__ unsigned long long prefix, mask, keys[512];
    __shared__ int used;
    if (!tid) {
        prefix = mask = 0;
        rank = n;
        used = 0;
    }
    __syncthreads();
    if (n && n < pools)
        for (int shift = 56; shift >= 0; shift -= 8) {
            histogram[tid] = 0;
            __syncthreads();
            for (int p = tid; p < pools; p += 256) {
                auto key = score_key(scores[p], p);
                if ((key & mask) == prefix)
                    atomicAdd(&histogram[(key >> shift) & 255], 1u);
            }
            __syncthreads();
            if (!tid) {
                for (int bin = 255; bin >= 0; --bin)
                    if (rank > histogram[bin])
                        rank -= histogram[bin];
                    else {
                        prefix |= (unsigned long long)bin << shift;
                        break;
                    }
                mask |= 255ull << shift;
            }
            __syncthreads();
        }
    for (int i = tid; i < 512; i += 256)
        keys[i] = 0;
    __syncthreads();
    for (int p = tid; p < pools; p += 256) {
        auto key = score_key(scores[p], p);
        if (n && key >= prefix) {
            int slot = atomicAdd(&used, 1);
            keys[slot] = key;
        }
    }
    __syncthreads();
    for (int size = 2; size <= 512; size *= 2)
        for (int step = size / 2; step; step /= 2) {
            for (int i = tid; i < 512; i += 256) {
                int other = i ^ step;
                if (other > i) {
                    auto a = keys[i], b = keys[other];
                    bool descending = (i & size) == 0;
                    if ((descending && a < b) || (!descending && a > b)) {
                        keys[i] = b;
                        keys[other] = a;
                    }
                }
            }
            __syncthreads();
        }
    for (int i = tid; i < n * pool; i += 256) {
        int p = 0xffffffffu - (unsigned)keys[i / pool];
        ids[i] = p * pool + i % pool;
    }
    for (int i = tid; i < tokens % pool; i += 256)
        ids[n * pool + i] = tokens - tokens % pool + i;
    if (!tid)
        counts[q] = n * pool + tokens % pool;
}
__global__ void gather(const float *cache, const int *ids, const int *counts, float *out, int queries,
                       int stride, int latent) {
    int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= (int64_t)queries * stride * latent)
        return;
    int q = i / (stride * latent), key = i / latent % stride, d = i % latent;
    out[i] = key < counts[q] ? cache[(int64_t)ids[q * stride + key] * latent + d] : 0.f;
}
__global__ void softmax(float *scores, const int *counts, int heads, int stride, float scale) {
    int q = blockIdx.x / heads;
    float *row = scores + (int64_t)blockIdx.x * stride;
    __shared__ float reduce[256];
    int tid = threadIdx.x, n = counts[q];
    float mx = -INFINITY;
    for (int i = tid; i < n; i += 256)
        mx = fmaxf(mx, row[i] * scale);
    reduce[tid] = mx;
    __syncthreads();
    for (int z = 128; z; z /= 2) {
        if (tid < z)
            reduce[tid] = fmaxf(reduce[tid], reduce[tid + z]);
        __syncthreads();
    }
    mx = reduce[0];
    float sum = 0;
    for (int i = tid; i < stride; i += 256) {
        float p = i < n ? expf(row[i] * scale - mx) : 0;
        row[i] = p;
        sum += p;
    }
    reduce[tid] = sum;
    __syncthreads();
    for (int z = 128; z; z /= 2) {
        if (tid < z)
            reduce[tid] += reduce[tid + z];
        __syncthreads();
    }
    for (int i = tid; i < stride; i += 256)
        row[i] /= reduce[0];
}
} // namespace
void glm_f16(const float *x, uint16_t *y, int64_t n, void *s) {
    half_cast<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, y, n);
    check();
}
void glm_conv_batch(const float *x, const float *w, float *h, float *y, int n, int kernel, int tokens,
                    void *s) {
    convolution<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, w, h, y, n, kernel, tokens);
    conv_history<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, h, n, kernel, tokens);
    check();
}
void glm_route_batch(const float *l, const float *b, int *ids, float *w, int ne, int top, float scale,
                     int tokens, void *s) {
    routing<<<tokens, 1, 0, (cudaStream_t)s>>>(l, b, ids, w, ne, top, scale);
    check();
}
void glm_group_routes(const int *ids, int *b, int *d, int *src, int *cursor, int ne, int top, int tokens,
                      void *s) {
    cudaMemsetAsync(b, 0, (ne + 1) * 4, (cudaStream_t)s);
    count_routes<<<(tokens * top + 255) / 256, 256, 0, (cudaStream_t)s>>>(ids, b, tokens * top);
    prefix_routes<<<1, 1, 0, (cudaStream_t)s>>>(b, cursor, ne);
    scatter_routes<<<(tokens * top + 255) / 256, 256, 0, (cudaStream_t)s>>>(ids, cursor, d, src, top,
                                                                            tokens * top);
    check();
}
void glm_group_bounds(const int *g, int *l, int start, int n, int offset, int rows, void *s) {
    slice_bounds<<<1, 32, 0, (cudaStream_t)s>>>(g, l, start, n, offset, rows);
    check();
}
void glm_route_sum(const float *r, const float *w, float *out, int n, int top, int tokens, void *s) {
    route_sum<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, w, out, n, top, tokens);
    check();
}
void glm_mhc_read_batch(const float *r, const float *p, const float *b, const float *scale, float *c,
                        float *x, int n, int it, float eps, int tokens, void *s) {
    mhc<<<tokens, 1, 0, (cudaStream_t)s>>>(p, b, scale, c, it, eps);
    mhc_read<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, c, x, n, tokens);
    check();
}
void glm_mhc_write_batch(const float *r, const float *c, const float *y, float *out, int n, int tokens,
                         void *s) {
    mhc_write<<<(n * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(r, c, y, out, n, tokens);
    check();
}
void glm_kda_gate_batch(const float *x, const float *b, const float *a, float *y, int h, int d, float lower,
                        int tokens, void *s) {
    decay<<<(h * d * tokens + 255) / 256, 256, 0, (cudaStream_t)s>>>(x, b, a, y, h, d, tokens, lower);
    check();
}
void glm_kda_chunk(float *state, const float *q, const float *k, const float *v, const float *g,
                   const float *b, float *out, int heads, int dim, int tokens, void *s) {
    if (dim != 128 || tokens < 1 || tokens > 64)
        throw std::invalid_argument("GLM: invalid KDA chunk");
    kda_chunk<<<heads, 128, 0, (cudaStream_t)s>>>(state, q, k, v, g, b, out, heads, tokens);
    check();
}
void glm_kda_output_batch(const float *x, const float *g, const float *w, float *y, int h, int d, float eps,
                          int tokens, void *s) {
    kda_output<<<h * tokens, d, 0, (cudaStream_t)s>>>(x, g, w, y, d, eps);
    check();
}
void glm_index_prepare(const float *k, const float *g, const float *a, float *pk, float *pg, float *out,
                       int pos, int tokens, int pool, int dim, void *s) {
    int n = ((pos + tokens) / pool - pos / pool) * dim;
    if (n)
        pooling<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(k, g, a, pk, pg, out, pos, tokens, pool, dim);
    pending<<<(pool * dim + 255) / 256, 256, 0, (cudaStream_t)s>>>(k, g, pk, pg, pos, tokens, pool, dim);
    check();
}
void glm_index_reduce(const float *d, const float *w, float *out, int h, int pools, int q, int pos, int pool,
                      int dim, void *s) {
    if (pools)
        index_reduce<<<(pools * q + 255) / 256, 256, 0, (cudaStream_t)s>>>(d, w, out, h, pools, q, pos, pool,
                                                                           dim);
    check();
}
void glm_index_select_batch(const float *scores, int *ids, int *counts, int pools, int pos, int queries,
                            int pool, int top, int stride, void *s) {
    if (top / pool > 512)
        throw std::invalid_argument("GLM: selection exceeds 512 pools");
    selection<<<queries, 256, 0, (cudaStream_t)s>>>(scores, ids, counts, pools, pos, pool, top, stride);
    check();
}
void glm_mla_gather(const float *c, const int *ids, const int *counts, float *out, int q, int stride,
                    int latent, void *s) {
    int64_t n = (int64_t)q * stride * latent;
    gather<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(c, ids, counts, out, q, stride, latent);
    check();
}
void glm_mla_softmax(float *scores, const int *counts, int heads, int stride, int q, float scale, void *s) {
    softmax<<<heads * q, 256, 0, (cudaStream_t)s>>>(scores, counts, heads, stride, scale);
    check();
}
} // namespace strata::kernels
