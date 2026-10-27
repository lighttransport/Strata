// src/kernels/cuda/ds41.cu - DeepSeek-V4.1-Flash decode kernels (strata/kernels/ds41.hpp).
// Equations: llama.cpp PR #30817 (src/models/deepseek4.cpp build_hc_mixes / build_hc_sinkhorn / build_hc_post,
// deepseek41.cpp build_attention41 / build_engram).
#include "strata/kernels/ds41.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>   // (hip_compat's under HIP)
#if defined(__CUDACC__) && !defined(__HIP_PLATFORM_AMD__)
#include <mma.h>
#define STRATA_DS41_WMMA 1
#endif
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 ") + what + ": " + cudaGetErrorString(e));
}

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ inline float warp_sum(float v) {
    for (int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}

// one thread: 24 coefficients, ggml's operation order
// one token per block, a thread per comb element: every sum is taken in the serial order (the same bits)
__global__ void hc_coeff_kernel(const float* m, const float* scale, const float* base, float* c, int iterations, float eps) {
    __shared__ float k[16];
    const int t = threadIdx.x;   // 16 threads: element d + 4 s
    m += size_t(blockIdx.x) * 24;
    c += size_t(blockIdx.x) * 24;
    if (t < 4) c[t] = sigmoid(m[t] * scale[0] + base[t]) + eps;
    else if (t < 8) c[t] = sigmoid(m[t] * scale[1] + base[t]) * 2.0f;
    k[t] = m[8 + t] * scale[2] + base[8 + t];
    __syncthreads();
    const int d = t % 4, sc = t / 4;
    {   // comb[dst + 4 src]: softmax over dst for each src
        float mx = -INFINITY;
        for (int i = 0; i < 4; ++i) mx = fmaxf(mx, k[i + 4 * sc]);
        float e[4], sum = 0;
        for (int i = 0; i < 4; ++i) { e[i] = expf(k[i + 4 * sc] - mx); sum += e[i]; }
        const float inv = 1.0f / sum;
        const float v = e[d] * inv + eps;
        __syncthreads();
        k[t] = v;
        __syncthreads();
    }
    auto cols = [&] {   // for each dst: divide by (sum over src + eps)
        float sum = 0;
        for (int i = 0; i < 4; ++i) sum += k[d + 4 * i];
        sum += eps;
        const float v = k[t] / sum;
        __syncthreads();
        k[t] = v;
        __syncthreads();
    };
    auto rows = [&] {   // for each src: divide by (sum over dst + eps)
        float sum = 0;
        for (int i = 0; i < 4; ++i) sum += k[i + 4 * sc];
        sum += eps;
        const float v = k[t] / sum;
        __syncthreads();
        k[t] = v;
        __syncthreads();
    };
    cols();
    for (int i = 1; i < iterations; ++i) { rows(); cols(); }
    c[8 + t] = k[t];
}

// mixes() in one launch per token (1024 threads): glm_rms_norm_rows of the 4 streams (no weight; its float4 sums and
// 32-lane tree) into flat, then the 24 F32 rows of fn as f32_gemv_kernel's threads 0..255 would sum them, then
// hc_coeff_kernel's sinkhorn by threads 0..15: the same bits as the three launches
__global__ void __launch_bounds__(1024) hc_mixes_kernel(const float* h, const float* fn, const float* scale, const float* base, float* flat,
                                                        float* mixes, float* c, int n4, int iterations, float eps, float rms_eps) {
    __shared__ float warps[32];
    __shared__ float red[8];
    __shared__ float k[16];
    const int t = threadIdx.x, lane = t & 31;
    const float4* x = reinterpret_cast<const float4*>(h) + size_t(blockIdx.x) * n4;
    float4* y = reinterpret_cast<float4*>(flat) + size_t(blockIdx.x) * n4;
    float sum = 0;
    for (int i = t; i < n4; i += 1024) {
        const float4 v = x[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (!lane) warps[t / 32] = sum;
    __syncthreads();
    if (t < 32) {
        sum = warps[t];
        for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (!t) warps[0] = rsqrtf(sum / (4.f * n4) + rms_eps);
    }
    __syncthreads();
    const float inv = warps[0];
    for (int i = t; i < n4; i += 1024) {
        float4 v = x[i];
        v.x = v.x * inv * 1.f; v.y = v.y * inv * 1.f; v.z = v.z * inv * 1.f; v.w = v.w * inv * 1.f;
        y[i] = v;
    }
    __syncthreads();
    // the 24 rows, each by threads 0..255 (f32_gemv_kernel's float4 path: in % 4 == 0 and 16-byte aligned rows)
    float* m = mixes + size_t(blockIdx.x) * 24;
    const int in = 4 * n4;
    for (int r = 0; r < 24; ++r) {
        if (t < 256) {
            const float4* r4 = reinterpret_cast<const float4*>(fn + size_t(r) * in);
            float s = 0;
            for (int i = t; i < n4; i += 256) {
                const float4 a = r4[i], b = y[i];
                s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            s = warp_sum(s);
            if (lane == 0) red[t / 32] = s;
        }
        __syncthreads();
        if (t == 0) {
            float tt = 0;
            for (int i = 0; i < 8; ++i) tt += red[i];
            m[r] = tt;
        }
        __syncthreads();
    }
    // hc_coeff_kernel, by threads 0..15 (element d + 4 s); the block's other threads wait at its barriers
    c += size_t(blockIdx.x) * 24;
    const bool in16 = t < 16;
    if (in16) {
        if (t < 4) c[t] = sigmoid(m[t] * scale[0] + base[t]) + eps;
        else if (t < 8) c[t] = sigmoid(m[t] * scale[1] + base[t]) * 2.0f;
        k[t] = m[8 + t] * scale[2] + base[8 + t];
    }
    __syncthreads();
    const int d = t % 4, sc = (t / 4) % 4;
    {
        float v = 0;
        if (in16) {
            float mx = -INFINITY;
            for (int i = 0; i < 4; ++i) mx = fmaxf(mx, k[i + 4 * sc]);
            float e[4], esum = 0;
            for (int i = 0; i < 4; ++i) { e[i] = expf(k[i + 4 * sc] - mx); esum += e[i]; }
            const float einv = 1.0f / esum;
            v = e[d] * einv + eps;
        }
        __syncthreads();
        if (in16) k[t] = v;
        __syncthreads();
    }
    auto cols = [&] {
        float v = 0;
        if (in16) {
            float ss = 0;
            for (int i = 0; i < 4; ++i) ss += k[d + 4 * i];
            ss += eps;
            v = k[t] / ss;
        }
        __syncthreads();
        if (in16) k[t] = v;
        __syncthreads();
    };
    auto rows = [&] {
        float v = 0;
        if (in16) {
            float ss = 0;
            for (int i = 0; i < 4; ++i) ss += k[i + 4 * sc];
            ss += eps;
            v = k[t] / ss;
        }
        __syncthreads();
        if (in16) k[t] = v;
        __syncthreads();
    };
    cols();
    for (int i = 1; i < iterations; ++i) { rows(); cols(); }
    if (in16) c[8 + t] = k[t];
}

__global__ void hc_pre_kernel(const float* h, const float* w, float* y, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    h += size_t(blockIdx.y) * 4 * n;
    w += size_t(blockIdx.y) * 24;
    y += size_t(blockIdx.y) * n;
    float v = __fmul_rn(h[j], w[0]);
    for (int s = 1; s < 4; ++s) v = __fadd_rn(v, __fmul_rn(h[size_t(s) * n + j], w[s]));
    y[j] = v;
}

__global__ void hc_post_kernel(const float* x, const float* res, const float* post, const float* comb, float* out, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    x += size_t(blockIdx.y) * n;
    res += size_t(blockIdx.y) * 4 * n;
    out += size_t(blockIdx.y) * 4 * n;
    post += size_t(blockIdx.y) * 24;
    comb += size_t(blockIdx.y) * 24;
    const float xj = x[j];
    for (int d = 0; d < 4; ++d) {
        float v = __fmul_rn(xj, post[d]);
        for (int s = 0; s < 4; ++s) v = __fadd_rn(v, __fmul_rn(res[size_t(s) * n + j], comb[d + 4 * s]));
        out[size_t(d) * n + j] = v;
    }
}

// STRATA_DS41_DEFER_RANK: a layer's deferred expert sum y added to the streams one sublayer late, as if it had been
// in that layer's output: through its post-mix (post, 4 per token) and then the next attention's stream mix (comb,
// 4x4 per token; null after the last layer): h[d] += y * sum_s post[s] comb[d + 4 s]
// STRATA_DS41_GPU_ROUTE: the router's top-k on the GPU and the table of the GPU-held routes for the grouped expert
// kernels (ds41_decode.cpp GroupedRunner layout, capacity cap = T * K: ptr u64[cap], start[cap + 1], n, dst[cap],
// tok[cap], w[cap]), so the GPU's experts start without the host. One block: per token, probs = sqrt(softplus(logit)),
// top-k of probs + bias (ties to the lower index, as the host's top_k), weights from probs (normalised over the
// picks kept, times scale). cpu_topk >= 0: picks of rank >= cpu_topk not held on the GPU are dropped (weight 0).
__global__ void route_table_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                   const unsigned long long* __restrict__ slots, int T, int E, int K, int normalize,
                                   float scale, int cpu_topk, int cap, int* __restrict__ ids, float* __restrict__ wts,
                                   unsigned char* __restrict__ table) {
    __shared__ float sel[1024], prob[1024];
    __shared__ float bv[256];
    __shared__ int bi[256];
    __shared__ int pick[16];
    __shared__ int count;
    unsigned long long* ptr = reinterpret_cast<unsigned long long*>(table);
    int* start = reinterpret_cast<int*>(table + size_t(cap) * 8);
    int* ng = start + cap + 1;
    int* dst = ng + 1;
    int* tok = dst + cap;
    float* w = reinterpret_cast<float*>(tok + cap);
    if (threadIdx.x == 0) count = 0;
    for (int t = 0; t < T; ++t) {
        for (int e = threadIdx.x; e < E; e += blockDim.x) {
            const float x = logits[size_t(t) * E + e];
            const float p = sqrtf(x > 20.0f ? x : log1pf(expf(x)));
            prob[e] = p;
            sel[e] = p + bias[e];
        }
        __syncthreads();
        for (int k = 0; k < K; ++k) {
            float best = -INFINITY;
            int at = E;
            for (int e = threadIdx.x; e < E; e += blockDim.x)
                if (sel[e] > best || (sel[e] == best && e < at)) { best = sel[e]; at = e; }
            bv[threadIdx.x] = best;
            bi[threadIdx.x] = at;
            __syncthreads();
            for (int o = blockDim.x / 2; o > 0; o >>= 1) {
                if (threadIdx.x < o) {
                    const float b2 = bv[threadIdx.x + o];
                    const int i2 = bi[threadIdx.x + o];
                    if (b2 > bv[threadIdx.x] || (b2 == bv[threadIdx.x] && i2 < bi[threadIdx.x])) { bv[threadIdx.x] = b2; bi[threadIdx.x] = i2; }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) { pick[k] = bi[0]; sel[bi[0]] = -INFINITY; }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            bool cut[16];
            for (int k = 0; k < K; ++k) {
                cut[k] = cpu_topk >= 0 && k >= cpu_topk && slots[pick[k]] == 0ull;
                if (!cut[k]) sum += prob[pick[k]];
            }
            for (int k = 0; k < K; ++k) {
                float v = prob[pick[k]];
                if (cut[k]) v = 0.0f;
                else {
                    if (normalize && K > 1) v /= sum + 1e-20f;
                    v *= scale;
                }
                ids[t * K + k] = pick[k];
                wts[t * K + k] = v;
                if (!cut[k] && slots[pick[k]]) {
                    const int g = count++;
                    ptr[g] = slots[pick[k]];
                    start[g] = g;
                    dst[g] = g;
                    tok[g] = t;
                    w[g] = v;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { start[count] = count; *ng = count; }
}

// scatter_axpy_kernel with the row count on the device (the GPU-routed table's n)
__global__ void scatter_axpy_dev_kernel(const float* src, const int* idx, const float* w, const int* m, int n, float* dst) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int rows = *m;
    for (int i = 0; i < rows; ++i) {
        float* d = dst + size_t(idx[i]) * n + j;
        *d = __fadd_rn(*d, __fmul_rn(w[i], src[size_t(i) * n + j]));
    }
}

__global__ void defer_add_kernel(const float* y, float* h, const float* post, const float* comb, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    y += size_t(blockIdx.y) * n;
    h += size_t(blockIdx.y) * 4 * n;
    post += size_t(blockIdx.y) * 24;
    if (comb) comb += size_t(blockIdx.y) * 24;
    const float yj = y[j];
    for (int d = 0; d < 4; ++d) {
        float v = 0.0f;
        if (comb) for (int s2 = 0; s2 < 4; ++s2) v += post[s2] * comb[d + 4 * s2];
        else v = post[d];
        h[size_t(d) * n + j] += v * yj;
    }
}

// the FFN tail in one launch: x = ((moe + peer) + gsum) + y (each add as ds41_add, in that order; peer / gsum may be
// null), y = x, then hc_post
__global__ void hc_post_sum_kernel(const float* moe, const float* peer, const float* gsum, float* y, const float* res, const float* post,
                                   const float* comb, float* out, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const size_t at = size_t(blockIdx.y) * n + j;
    float xj = moe[at];
    if (peer) xj = xj + peer[at];
    if (gsum) xj = xj + gsum[at];
    xj = xj + y[at];
    y[at] = xj;
    res += size_t(blockIdx.y) * 4 * n;
    out += size_t(blockIdx.y) * 4 * n;
    post += size_t(blockIdx.y) * 24;
    comb += size_t(blockIdx.y) * 24;
    for (int d = 0; d < 4; ++d) {
        float v = __fmul_rn(xj, post[d]);
        for (int s = 0; s < 4; ++s) v = __fadd_rn(v, __fmul_rn(res[size_t(s) * n + j], comb[d + 4 * s]));
        out[size_t(d) * n + j] = v;
    }
}

__global__ void broadcast_kernel(const float* x, float* h, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    x += size_t(blockIdx.y) * n;
    h += size_t(blockIdx.y) * 4 * n;
    const float v = x[j];
    for (int s = 0; s < 4; ++s) h[size_t(s) * n + j] = v;
}

__global__ void rope_kernel(float* x, int rows, int stride, int offset, int pairs, const float* cs, int group, int cs_step) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * pairs) return;
    const int r = i / pairs, p = i % pairs;
    cs += size_t(r / group) * cs_step;
    float* v = x + size_t(r) * stride + offset + 2 * p;
    const float c = cs[2 * p], s = cs[2 * p + 1];
    const float x0 = v[0], x1 = v[1];
    v[0] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    v[1] = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
}

constexpr int kMaxRows = 4096;   // window + selected rows
constexpr int kHeadGroup = 8;    // heads sharing one read of a key row
constexpr int kRowChunk = 64;    // rows per scores block
constexpr int kDimChunk = 128;   // output dims per block

struct Ring { int first, size; };   // window row r is slot (first + r) % size
__device__ inline const float* attn_row(const float* window, Ring ring, int wrows, const float* comp, int comp_stride,
                                        const int* sel, int dim, int r) {
    return r < wrows ? window + size_t((ring.first + r) % ring.size) * dim : comp + size_t(sel[r - wrows]) * comp_stride;
}

// scores[h][r] = scale q_h.k_r for kHeadGroup heads x kRowChunk rows; a warp per row, the group's queries in LDS
__global__ void __launch_bounds__(256) attn_scores_kernel(const float* q, int dim, const float* window, Ring ring, int wrows,
        const float* comp, int comp_stride, const int* sel, int rows, float scale, float* scores, const int* nsel_dev) {
    extern __shared__ float qs[];   // kHeadGroup x dim
    const int h0 = blockIdx.x * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int n = nsel_dev ? min(rows, wrows + *nsel_dev) : rows;   // the selected rows' count on the device: rows is the bound
    if (blockIdx.y * kRowChunk >= n) return;
    for (int i = tid; i < kHeadGroup * dim; i += 256) qs[i] = q[size_t(h0) * dim + i];
    __syncthreads();
    for (int r = blockIdx.y * kRowChunk + warp; r < min(n, (blockIdx.y + 1) * kRowChunk); r += 8) {
        const float* k = attn_row(window, ring, wrows, comp, comp_stride, sel, dim, r);
        float acc[kHeadGroup];
#pragma unroll
        for (int h = 0; h < kHeadGroup; ++h) acc[h] = 0;
        for (int i = lane; i < dim; i += 32) {
            const float kv = k[i];
#pragma unroll
            for (int h = 0; h < kHeadGroup; ++h) acc[h] += qs[h * dim + i] * kv;
        }
#pragma unroll
        for (int h = 0; h < kHeadGroup; ++h) {
            const float v = warp_sum(acc[h]);
            if (!lane) scores[size_t(h0 + h) * kMaxRows + r] = v * scale;
        }
    }
}

// softmax with the sink per head, then out[h][d] for kHeadGroup heads x kDimChunk dims
__global__ void __launch_bounds__(256) attn_out_kernel(const float* scores, int dim, const float* window, Ring ring, int wrows,
        const float* comp, int comp_stride, const int* sel, int rows, const float* sinks, float* out, const int* nsel_dev) {
    extern __shared__ float p[];    // kHeadGroup x rows (rows: the bound; n of them used)
    __shared__ float inv[kHeadGroup];
    const int h0 = blockIdx.x * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int n = nsel_dev ? min(rows, wrows + *nsel_dev) : rows;
    // warp h: head h0 + h's max, exponentials and sum
    {
        const int h = warp;   // 8 warps, kHeadGroup heads
        const float* sc = scores + size_t(h0 + h) * kMaxRows;
        float mx = sinks[h0 + h];
        for (int r = lane; r < n; r += 32) mx = fmaxf(mx, sc[r]);
        for (int d = 16; d; d /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
        float sum = 0;
        for (int r = lane; r < n; r += 32) { const float e = expf(sc[r] - mx); p[h * rows + r] = e; sum += e; }
        sum = warp_sum(sum);
        if (!lane) inv[h] = 1.0f / (sum + expf(sinks[h0 + h] - mx));
    }
    __syncthreads();
    const int d = blockIdx.y * kDimChunk + tid % kDimChunk, hs = tid / kDimChunk;   // 2 head subsets of 4
    if (d >= dim) return;
    float acc[kHeadGroup / 2];
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] = 0;
    for (int r = 0; r < n; ++r) {
        const float v = attn_row(window, ring, wrows, comp, comp_stride, sel, dim, r)[d];
#pragma unroll
        for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] += p[(hs + 2 * j) * rows + r] * v;
    }
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j) out[size_t(h0 + hs + 2 * j) * dim + d] = acc[j] * inv[hs + 2 * j];
}


// ds41_attention for a few tokens in one launch: token t = blockIdx.z, info[t] = {first slot, window rows, selected
// rows (the bound when the count is on the device), -}; q, out at t heads dim, sel at t sel_stride, scores at
// t heads kMaxRows
__global__ void __launch_bounds__(256) attn_scores_multi_kernel(const float* q, int heads, int dim, const float* window, int ring_size,
        const int4* info, const float* comp, int comp_stride, const int* sel, int sel_stride, const int* nsel_dev, float scale,
        float* scores, size_t wss, size_t css) {
    extern __shared__ float qs[];
    const int t = blockIdx.z;
    window += size_t(t) * wss;
    comp += size_t(t) * css;
    const int4 in = info[t];
    const Ring ring{in.x, ring_size};
    const int n = in.y + (nsel_dev ? min(in.z, nsel_dev[t]) : in.z);
    if (blockIdx.y * kRowChunk >= n) return;
    q += size_t(t) * heads * dim;
    sel += size_t(t) * sel_stride;
    scores += size_t(t) * heads * kMaxRows;
    const int h0 = blockIdx.x * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    for (int i = tid; i < kHeadGroup * dim; i += 256) qs[i] = q[size_t(h0) * dim + i];
    __syncthreads();
    for (int r = blockIdx.y * kRowChunk + warp; r < min(n, (blockIdx.y + 1) * kRowChunk); r += 8) {
        const float* k = attn_row(window, ring, in.y, comp, comp_stride, sel, dim, r);
        float acc[kHeadGroup];
#pragma unroll
        for (int h = 0; h < kHeadGroup; ++h) acc[h] = 0;
        for (int i = lane; i < dim; i += 32) {
            const float kv = k[i];
#pragma unroll
            for (int h = 0; h < kHeadGroup; ++h) acc[h] += qs[h * dim + i] * kv;
        }
#pragma unroll
        for (int h = 0; h < kHeadGroup; ++h) {
            const float v = warp_sum(acc[h]);
            if (!lane) scores[size_t(h0 + h) * kMaxRows + r] = v * scale;
        }
    }
}

__global__ void __launch_bounds__(256) attn_out_multi_kernel(const float* scores, int heads, int dim, const float* window, int ring_size,
        const int4* info, const float* comp, int comp_stride, const int* sel, int sel_stride, const int* nsel_dev, int bound,
        const float* sinks, float* out, size_t wss, size_t css) {
    extern __shared__ float p[];    // kHeadGroup x bound
    __shared__ float inv[kHeadGroup];
    const int t = blockIdx.z;
    window += size_t(t) * wss;
    comp += size_t(t) * css;
    const int4 in = info[t];
    const Ring ring{in.x, ring_size};
    const int n = in.y + (nsel_dev ? min(in.z, nsel_dev[t]) : in.z);
    sel += size_t(t) * sel_stride;
    scores += size_t(t) * heads * kMaxRows;
    out += size_t(t) * heads * dim;
    const int h0 = blockIdx.x * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    {
        const int h = warp;
        const float* sc = scores + size_t(h0 + h) * kMaxRows;
        float mx = sinks[h0 + h];
        for (int r = lane; r < n; r += 32) mx = fmaxf(mx, sc[r]);
        for (int d = 16; d; d /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
        float sum = 0;
        for (int r = lane; r < n; r += 32) { const float e = expf(sc[r] - mx); p[h * bound + r] = e; sum += e; }
        sum = warp_sum(sum);
        if (!lane) inv[h] = 1.0f / (sum + expf(sinks[h0 + h] - mx));
    }
    __syncthreads();
    const int d = blockIdx.y * kDimChunk + tid % kDimChunk, hs = tid / kDimChunk;
    if (d >= dim) return;
    float acc[kHeadGroup / 2];
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] = 0;
    for (int r = 0; r < n; ++r) {
        const float v = attn_row(window, ring, in.y, comp, comp_stride, sel, dim, r)[d];
#pragma unroll
        for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] += p[(hs + 2 * j) * bound + r] * v;
    }
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j) out[size_t(h0 + hs + 2 * j) * dim + d] = acc[j] * inv[hs + 2 * j];
}


// attn_out_kernel with the rows split over blockIdx.z (`splits` parts): each block its rows' weighted sum (the
// softmax statistics over all rows), into part[z][heads][dim]; attn_combine_kernel adds the parts in order
__global__ void __launch_bounds__(256) attn_out_split_kernel(const float* scores, int heads, int dim, const float* window, Ring ring, int wrows,
        const float* comp, int comp_stride, const int* sel, int rows, const float* sinks, float* part, const int* nsel_dev) {
    extern __shared__ float p[];    // kHeadGroup x chunk
    __shared__ float inv[kHeadGroup];
    const int h0 = blockIdx.x * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int n = nsel_dev ? min(rows, wrows + *nsel_dev) : rows;
    const int chunk = (rows + gridDim.z - 1) / gridDim.z, r0 = blockIdx.z * chunk, r1 = min(n, r0 + chunk);
    {
        const int h = warp;
        const float* sc = scores + size_t(h0 + h) * kMaxRows;
        float mx = sinks[h0 + h];
        for (int r = lane; r < n; r += 32) mx = fmaxf(mx, sc[r]);
        for (int d = 16; d; d /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
        float sum = 0;
        for (int r = lane; r < n; r += 32) {
            const float e = expf(sc[r] - mx);
            sum += e;
            if (r >= r0 && r < r1) p[h * chunk + (r - r0)] = e;
        }
        sum = warp_sum(sum);
        if (!lane) inv[h] = 1.0f / (sum + expf(sinks[h0 + h] - mx));
    }
    __syncthreads();
    const int d = blockIdx.y * kDimChunk + tid % kDimChunk, hs = tid / kDimChunk;
    if (d >= dim) return;
    float acc[kHeadGroup / 2];
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] = 0;
    for (int r = r0; r < r1; ++r) {
        const float v = attn_row(window, ring, wrows, comp, comp_stride, sel, dim, r)[d];
#pragma unroll
        for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] += p[(hs + 2 * j) * chunk + (r - r0)] * v;
    }
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j)
        part[(size_t(blockIdx.z) * heads + h0 + hs + 2 * j) * dim + d] = acc[j] * inv[hs + 2 * j];
}
__global__ void attn_combine_kernel(const float* part, int splits, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0;
    for (int z = 0; z < splits; ++z) s += part[size_t(z) * n + i];
    out[i] = s;
}


// attn_out_multi_kernel with each token's rows split: blockIdx.z = token * splits + part; into part[token][split]
__global__ void __launch_bounds__(256) attn_out_multi_split_kernel(const float* scores, int heads, int dim, const float* window, int ring_size,
        const int4* info, const float* comp, int comp_stride, const int* sel, int sel_stride, const int* nsel_dev, int bound, int splits,
        const float* sinks, float* part, size_t wss, size_t css) {
    extern __shared__ float p[];    // kHeadGroup x chunk
    __shared__ float inv[kHeadGroup];
    const int t = blockIdx.z / splits, z = blockIdx.z % splits;
    window += size_t(t) * wss;
    comp += size_t(t) * css;
    const int4 in = info[t];
    const Ring ring{in.x, ring_size};
    const int n = in.y + (nsel_dev ? min(in.z, nsel_dev[t]) : in.z);
    const int chunk = (bound + splits - 1) / splits, r0 = z * chunk, r1 = min(n, r0 + chunk);
    sel += size_t(t) * sel_stride;
    scores += size_t(t) * heads * kMaxRows;
    const int h0 = blockIdx.x * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    {
        const int h = warp;
        const float* sc = scores + size_t(h0 + h) * kMaxRows;
        float mx = sinks[h0 + h];
        for (int r = lane; r < n; r += 32) mx = fmaxf(mx, sc[r]);
        for (int d = 16; d; d /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
        float sum = 0;
        for (int r = lane; r < n; r += 32) {
            const float e = expf(sc[r] - mx);
            sum += e;
            if (r >= r0 && r < r1) p[h * chunk + (r - r0)] = e;
        }
        sum = warp_sum(sum);
        if (!lane) inv[h] = 1.0f / (sum + expf(sinks[h0 + h] - mx));
    }
    __syncthreads();
    const int d = blockIdx.y * kDimChunk + tid % kDimChunk, hs = tid / kDimChunk;
    if (d >= dim) return;
    float acc[kHeadGroup / 2];
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] = 0;
    for (int r = r0; r < r1; ++r) {
        const float v = attn_row(window, ring, in.y, comp, comp_stride, sel, dim, r)[d];
#pragma unroll
        for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] += p[(hs + 2 * j) * chunk + (r - r0)] * v;
    }
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j)
        part[((size_t(t) * splits + z) * heads + h0 + hs + 2 * j) * dim + d] = acc[j] * inv[hs + 2 * j];
}
// out[t][i] = sum over z of part[t][z][i] (n = heads x dim per token)
__global__ void attn_combine_multi_kernel(const float* part, int splits, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float s = 0;
    for (int z = 0; z < splits; ++z) s += part[(size_t(t) * splits + z) * n + i];
    out[size_t(t) * n + i] = s;
}

// Batched attention (prefill): block (token t, 8-head group). Key tiles of 16 rows in LDS, scores per head by one
// warp, an online softmax per head, the weighted sum by every thread over 2 of the 512 dims; the sink joins the
// denominator at the end (soft_max_ext with sinks). info[t] = {first slot, window rows, selected rows}.
constexpr int kTile = 16;
__global__ void __launch_bounds__(256) attn_batched_kernel(const float* q, int heads, int dim, const float* window, int ring,
        const int4* info, const float* comp, int comp_stride, const int* sel, int sel_stride, const float* sinks, float scale,
        float* out) {
    extern __shared__ float lds[];
    float* qs = lds;                          // kHeadGroup x dim
    float* ks = qs + kHeadGroup * dim;        // kTile x dim
    float* ps = ks + kTile * dim;             // kHeadGroup x kTile
    __shared__ float m[kHeadGroup], l[kHeadGroup], sc[kHeadGroup];
    const int t = blockIdx.x, h0 = blockIdx.y * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int4 in = info[t];
    const int first = in.x, wrows = in.y, rows = in.y + in.z;
    const int* tsel = sel + size_t(t) * sel_stride;
    const float* qt = q + (size_t(t) * heads + h0) * dim;
    for (int i = tid; i < kHeadGroup * dim; i += 256) qs[i] = qt[i];
    if (tid < kHeadGroup) { m[tid] = -INFINITY; l[tid] = 0; }
    float acc[kHeadGroup][2];
#pragma unroll
    for (int h = 0; h < kHeadGroup; ++h) acc[h][0] = acc[h][1] = 0;
    __syncthreads();
    for (int r0 = 0; r0 < rows; r0 += kTile) {
        for (int i = tid; i < kTile * dim; i += 256) {
            const int r = r0 + i / dim, d = i % dim;
            float v = 0;
            if (r < rows) {
                const float* row = r < wrows ? window + size_t((first + r) % ring) * dim : comp + size_t(tsel[r - wrows]) * comp_stride;
                v = row[d];
            }
            ks[i] = v;
        }
        __syncthreads();
        {   // warp w: head h0 + w, the tile's scores
            for (int r = 0; r < kTile; ++r) {
                float sdot = 0;
                for (int d = lane; d < dim; d += 32) sdot += qs[warp * dim + d] * ks[r * dim + d];
                sdot = warp_sum(sdot);
                if (!lane) ps[warp * kTile + r] = r0 + r < rows ? sdot * scale : -INFINITY;
            }
        }
        __syncthreads();
        if (tid < kHeadGroup) {   // online softmax for head tid
            float mt = -INFINITY;
            for (int r = 0; r < kTile; ++r) mt = fmaxf(mt, ps[tid * kTile + r]);
            const float mn = fmaxf(m[tid], mt);
            const float f = m[tid] == -INFINITY ? 0.0f : expf(m[tid] - mn);
            float sum = 0;
            for (int r = 0; r < kTile; ++r) {
                const float e = ps[tid * kTile + r] == -INFINITY ? 0.0f : expf(ps[tid * kTile + r] - mn);
                ps[tid * kTile + r] = e;
                sum += e;
            }
            l[tid] = l[tid] * f + sum;
            m[tid] = mn;
            sc[tid] = f;
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < kHeadGroup; ++h) {
            float a0 = acc[h][0] * sc[h], a1 = acc[h][1] * sc[h];
            for (int r = 0; r < kTile; ++r) {
                const float p = ps[h * kTile + r];
                a0 += p * ks[r * dim + tid];
                if (tid + 256 < dim) a1 += p * ks[r * dim + tid + 256];
            }
            acc[h][0] = a0;
            acc[h][1] = a1;
        }
        __syncthreads();
    }
    float* ot = out + (size_t(t) * heads + h0) * dim;
#pragma unroll
    for (int h = 0; h < kHeadGroup; ++h) {
        const float sink = sinks[h0 + h], M = fmaxf(m[h], sink);
        const float f = m[h] == -INFINITY ? 0.0f : expf(m[h] - M);
        const float inv = 1.0f / (l[h] * f + expf(sink - M));
        ot[size_t(h) * dim + tid] = acc[h][0] * f * inv;
        if (tid + 256 < dim) ot[size_t(h) * dim + tid + 256] = acc[h][1] * f * inv;
    }
}

constexpr int kTileDim = 512;
// Batched attention, register-tiled: block (token t, 16-head group), 256 threads. Key tiles of 64 rows go through LDS
// in 64-dim chunks; a thread's 4 scores (head th / 16, rows th % 16 + 16 j) accumulate in registers, the online softmax
// reduces over the head's 16 threads by shuffles, and the weighted sum gives each thread 1 dim x 4 heads of every
// chunk (the 32 accumulators of 16 heads x 512 dims / 256 threads). Same arithmetic as attn_batched_kernel up to
// the summation order.
constexpr int kTG = 16, kTR = 64, kDC = 64, kDCP = kDC + 4;
__device__ inline const float* attn_tile_row(const float* window, int ring, int first, int wrows, const float* comp, int comp_stride,
                                             const int* tsel, int r) {
    return r < wrows ? window + size_t((first + r) % ring) * kTileDim : comp + size_t(tsel[r - wrows]) * comp_stride;
}
// 4 values of one row at `col`: the window is FP32, the compressed rows are CT (FP16 when the cache is, converted here so
// the gather to a float buffer per launch is not needed)
template <class CT>
__device__ inline float4 attn_tile_load4(const float* window, int ring, int first, int wrows, const CT* comp, int comp_stride,
                                         const int* tsel, int r, int col) {
    if (r < wrows) return *reinterpret_cast<const float4*>(window + size_t((first + r) % ring) * kTileDim + col);
    const CT* p = comp + size_t(tsel[r - wrows]) * comp_stride + col;
    if constexpr (sizeof(CT) == 4) {
        return *reinterpret_cast<const float4*>(p);
    } else {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
        return make_float4(__half2float(__ushort_as_half(h[0])), __half2float(__ushort_as_half(h[1])),
                           __half2float(__ushort_as_half(h[2])), __half2float(__ushort_as_half(h[3])));
    }
}
template <class CT>
__global__ void __launch_bounds__(256) attn_tiled_kernel(const float* q, int heads, const float* window, int ring, const int4* info,
        const CT* comp, int comp_stride, const int* sel, int sel_stride, const float* sinks, float scale, float* out) {
    constexpr int dim = kTileDim, chunks = kTileDim / kDC;
    extern __shared__ float lds[];
    float* qs = lds;                  // kTG x dim
    float* kc = qs + kTG * dim;       // kTR x kDCP: one chunk of the tile's rows
    float* pt = kc + kTR * kDCP;      // kTR x kTG: the tile's probabilities, [row][head]
    __shared__ float scs[kTG], ms[kTG], ls[kTG];
    const int t = blockIdx.x, h0 = blockIdx.y * kTG, th = threadIdx.x;
    const int4 in = info[t];
    const int first = in.x, wrows = in.y, rows = in.y + in.z;
    const int* tsel = sel + size_t(t) * sel_stride;
    const float4* qt = reinterpret_cast<const float4*>(q + (size_t(t) * heads + h0) * dim);
    for (int i = th; i < kTG * dim / 4; i += 256) reinterpret_cast<float4*>(qs)[i] = qt[i];
    const int sh = th / 16, sr = th % 16;           // scores: head sh, rows sr + 16 j
    const int pd = th % kDC, ph = (th / kDC) * 4;   // weighted sum: dim pd of each chunk, heads ph..ph+3
    float m_run = -INFINITY, l_run = 0;
    float acc[chunks][4];
#pragma unroll
    for (int c = 0; c < chunks; ++c)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[c][k] = 0;
    auto load_chunk = [&](int r0, int c0) {
        for (int i = th; i < kTR * kDC / 4; i += 256) {
            const int r = i / (kDC / 4), c4 = i % (kDC / 4), rr = r0 + r;
            float4 v = make_float4(0, 0, 0, 0);
            if (rr < rows) v = attn_tile_load4(window, ring, first, wrows, comp, comp_stride, tsel, rr, c0 + 4 * c4);
            *reinterpret_cast<float4*>(kc + r * kDCP + 4 * c4) = v;
        }
    };
    __syncthreads();
    for (int r0 = 0; r0 < rows; r0 += kTR) {
        float s[4] = {0, 0, 0, 0};
        for (int c = 0; c < chunks; ++c) {
            load_chunk(r0, c * kDC);
            __syncthreads();
            const float* qh = qs + sh * dim + c * kDC;
#pragma unroll 4
            for (int d = 0; d < kDC; d += 4) {
                const float4 qv = *reinterpret_cast<const float4*>(qh + d);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float4 kv = *reinterpret_cast<const float4*>(kc + (sr + 16 * j) * kDCP + d);
                    s[j] = fmaf(qv.x, kv.x, s[j]);
                    s[j] = fmaf(qv.y, kv.y, s[j]);
                    s[j] = fmaf(qv.z, kv.z, s[j]);
                    s[j] = fmaf(qv.w, kv.w, s[j]);
                }
            }
            if (c + 1 < chunks) __syncthreads();   // the last chunk stays for the weighted sum
        }
        // online softmax over the tile, per head (its 16 threads)
        float mt = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s[j] = r0 + sr + 16 * j < rows ? s[j] * scale : -INFINITY;
            mt = fmaxf(mt, s[j]);
        }
        for (int o = 8; o; o /= 2) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, o, 16));
        const float mn = fmaxf(m_run, mt);
        const float f = m_run == -INFINITY ? 0.0f : expf(m_run - mn);
        float sum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float e = s[j] == -INFINITY ? 0.0f : expf(s[j] - mn);
            pt[(sr + 16 * j) * kTG + sh] = e;
            sum += e;
        }
        for (int o = 8; o; o /= 2) sum += __shfl_xor_sync(0xffffffff, sum, o, 16);
        l_run = l_run * f + sum;
        m_run = mn;
        if (!sr) scs[sh] = f;
        __syncthreads();
        const float4 f4 = *reinterpret_cast<const float4*>(scs + ph);
#pragma unroll
        for (int cc = 0; cc < chunks; ++cc) {   // the resident (last) chunk first
            const int c = chunks - 1 - cc;
            if (cc) {
                __syncthreads();
                load_chunk(r0, c * kDC);
                __syncthreads();
            }
            float a0 = acc[c][0] * f4.x, a1 = acc[c][1] * f4.y, a2 = acc[c][2] * f4.z, a3 = acc[c][3] * f4.w;
#pragma unroll 8
            for (int r = 0; r < kTR; ++r) {
                const float v = kc[r * kDCP + pd];
                const float4 p = *reinterpret_cast<const float4*>(pt + r * kTG + ph);
                a0 = fmaf(p.x, v, a0);
                a1 = fmaf(p.y, v, a1);
                a2 = fmaf(p.z, v, a2);
                a3 = fmaf(p.w, v, a3);
            }
            acc[c][0] = a0; acc[c][1] = a1; acc[c][2] = a2; acc[c][3] = a3;
        }
        __syncthreads();
    }
    if (!sr) { ms[sh] = m_run; ls[sh] = l_run; }
    __syncthreads();
    float* ot = out + (size_t(t) * heads + h0) * dim;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int h = ph + k;
        const float sink = sinks[h0 + h], M = fmaxf(ms[h], sink);
        const float f = ms[h] == -INFINITY ? 0.0f : expf(ms[h] - M);
        const float inv = 1.0f / (ls[h] * f + expf(sink - M));
#pragma unroll
        for (int c = 0; c < chunks; ++c) ot[size_t(h) * dim + c * kDC + pd] = acc[c][k] * f * inv;
    }
}

#ifdef STRATA_DS41_WMMA
// Batched attention on tensor cores (Volta+, STRATA_DS41_ATTN_WMMA=1): block (token, 16-head group), 4 warps. Q and each
// 32-row key tile in shared memory as FP16; S = Q K^T and O += P V with WMMA 16x16x16 (FP32 accumulation, O kept in
// shared memory so the online softmax can rescale its rows); scores, softmax and the sink in FP32 as the tiled kernel.
namespace wm = nvcuda::wmma;
constexpr int kWG = 16, kWR = 32, kWD = 512, kWLQ = kWD + 8, kWLO = kWD + 4, kWLS = kWR + 4, kWLP = kWR + 8;
constexpr size_t kWmmaShared = size_t(kWG * kWLQ + kWR * kWLQ) * 2 + size_t(kWG * kWLO + 2 * kWG * kWLS) * 4 + size_t(kWG * kWLP) * 2;
__global__ void __launch_bounds__(128) attn_wmma_kernel(const float* q, int heads, const float* window, int ring, const int4* info,
        const float* comp, int comp_stride, const int* sel, int sel_stride, const float* sinks, float scale, float* out) {
#if __CUDA_ARCH__ >= 700
    extern __shared__ __align__(32) unsigned char wsm[];
    half* qs = reinterpret_cast<half*>(wsm);                 // [16][kWLQ]
    half* ks = qs + kWG * kWLQ;                              // [32][kWLQ]: keys = values
    float* os = reinterpret_cast<float*>(ks + kWR * kWLQ);   // [16][kWLO]
    float* s0 = os + kWG * kWLO;                             // [16][kWLS] two K halves of S
    float* s1 = s0 + kWG * kWLS;
    half* ps = reinterpret_cast<half*>(s1 + kWG * kWLS);     // [16][kWLP]
    __shared__ float m_run[kWG], l_run[kWG], fac[kWG];
    const int t = blockIdx.x, h0 = blockIdx.y * kWG, th = threadIdx.x, warp = th / 32;
    const int4 in = info[t];
    const int first = in.x, wrows = in.y, rows = in.y + in.z;
    const int* tsel = sel + size_t(t) * sel_stride;
    const float* qt = q + (size_t(t) * heads + h0) * kWD;
    for (int i = th; i < kWG * kWD; i += 128) qs[(i / kWD) * kWLQ + i % kWD] = __float2half(qt[i]);
    for (int i = th; i < kWG * kWLO; i += 128) os[i] = 0.0f;
    if (th < kWG) { m_run[th] = -INFINITY; l_run[th] = 0.0f; }
    __syncthreads();
    for (int r0 = 0; r0 < rows; r0 += kWR) {
        for (int i = th; i < kWR * kWD / 4; i += 128) {   // the tile's rows, FP16
            const int r = i / (kWD / 4), c4 = i % (kWD / 4), rr = r0 + r;
            float4 v = make_float4(0, 0, 0, 0);
            if (rr < rows) {
                const float* row = rr < wrows ? window + size_t((first + rr) % ring) * kWD : comp + size_t(tsel[rr - wrows]) * comp_stride;
                v = *reinterpret_cast<const float4*>(row + 4 * c4);
            }
            half* d = ks + r * kWLQ + 4 * c4;
            d[0] = __float2half(v.x); d[1] = __float2half(v.y); d[2] = __float2half(v.z); d[3] = __float2half(v.w);
        }
        __syncthreads();
        {   // S: warp w takes columns (w % 2) 16 of the tile over half (w / 2) of the dims
            const int n0 = (warp % 2) * 16, k0 = (warp / 2) * (kWD / 2);
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::fill_fragment(acc, 0.0f);
            for (int k = k0; k < k0 + kWD / 2; k += 16) {
                wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> a;
                wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::col_major> b;
                wm::load_matrix_sync(a, qs + k, kWLQ);
                wm::load_matrix_sync(b, ks + n0 * kWLQ + k, kWLQ);
                wm::mma_sync(acc, a, b, acc);
            }
            wm::store_matrix_sync((warp / 2 ? s1 : s0) + n0, acc, kWLS, wm::mem_row_major);
        }
        __syncthreads();
        {   // online softmax: 8 threads per head, 4 columns each
            const int h = th / 8, c0 = (th % 8) * 4;
            float sv[4], mt = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = c0 + j;
                sv[j] = r0 + c < rows ? (s0[h * kWLS + c] + s1[h * kWLS + c]) * scale : -INFINITY;
                mt = fmaxf(mt, sv[j]);
            }
            for (int o = 4; o; o /= 2) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, o, 8));
            const float mo = m_run[h], mn = fmaxf(mo, mt);
            const float f = mo == -INFINITY ? 0.0f : expf(mo - mn);
            float sum = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = sv[j] == -INFINITY ? 0.0f : expf(sv[j] - mn);
                ps[h * kWLP + c0 + j] = __float2half(e);
                sum += e;
            }
            for (int o = 4; o; o /= 2) sum += __shfl_xor_sync(0xffffffff, sum, o, 8);
            __syncwarp();
            if (th % 8 == 0) { l_run[h] = l_run[h] * f + sum; m_run[h] = mn; fac[h] = f; }
        }
        __syncthreads();
        for (int i = th; i < kWG * kWD; i += 128) os[(i / kWD) * kWLO + i % kWD] *= fac[i / kWD];
        __syncthreads();
        // O += P V: warp w takes 8 of the 32 column tiles
        for (int tile = warp; tile < kWD / 16; tile += 4) {
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::load_matrix_sync(acc, os + tile * 16, kWLO, wm::mem_row_major);
#pragma unroll
            for (int k = 0; k < kWR; k += 16) {
                wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> a;
                wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::row_major> b;
                wm::load_matrix_sync(a, ps + k, kWLP);
                wm::load_matrix_sync(b, ks + k * kWLQ + tile * 16, kWLQ);
                wm::mma_sync(acc, a, b, acc);
            }
            wm::store_matrix_sync(os + tile * 16, acc, kWLO, wm::mem_row_major);
        }
        __syncthreads();
    }
    float* ot = out + (size_t(t) * heads + h0) * kWD;
    for (int i = th; i < kWG * kWD; i += 128) {
        const int h = i / kWD;
        const float sink = sinks[h0 + h], M = fmaxf(m_run[h], sink);
        const float f = m_run[h] == -INFINITY ? 0.0f : expf(m_run[h] - M);
        ot[i] = os[h * kWLO + i % kWD] * f / (l_run[h] * f + expf(sink - M));
    }
#endif
}
#endif

// an FP16 cache value (IEEE half bits) as float
__device__ inline float h2f(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu;
    if (e == 0) {   // zero or subnormal
        const float v = float(m) * 5.9604644775390625e-8f;   // 2^-24
        return s ? -v : v;
    }
    if (e == 31) return __uint_as_float(s | 0x7f800000u | (m << 13));
    return __uint_as_float(s | ((e + 112) << 23) | (m << 13));
}
// float -> IEEE half bits, round to nearest even (portable: no cuda_fp16.h)
__device__ inline uint16_t f2h(float f) {
    const uint32_t u = __float_as_uint(f), sign = (u >> 16) & 0x8000u;
    const int e = int((u >> 23) & 0xff) - 127 + 15;
    if (((u >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00u | ((u & 0x7fffff) ? 0x200u : 0u));
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    if (e > 0) {
        const uint32_t m = u & 0x7fffff, rest = m & 0x1fff;
        uint32_t hv = (uint32_t(e) << 10) | (m >> 13);
        if (rest > 0x1000 || (rest == 0x1000 && (hv & 1))) ++hv;
        return uint16_t(sign | hv);
    }
    if (e <= -11) return uint16_t(sign);
    const uint32_t m = (u & 0x7fffff) | 0x800000;
    const int shift = 14 - e;
    uint32_t hv = m >> shift;
    const uint32_t rest = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rest > half || (rest == half && (hv & 1))) ++hv;
    return uint16_t(sign | hv);
}
__device__ inline float kv_ld(const float* p) { return *p; }
__device__ inline float kv_ld(const uint16_t* p) { return h2f(*p); }
// a cache row's value i by storage kind: 0 float, 1 FP16, 2 Q8_0 blocks (FP16 scale + 32 int8, 34 bytes)
template <int KIND> __device__ inline float kv_at(const uint8_t* row, int i) {
    if constexpr (KIND == 0) return reinterpret_cast<const float*>(row)[i];
    else if constexpr (KIND == 1) return h2f(reinterpret_cast<const uint16_t*>(row)[i]);
    else {
        const uint8_t* b = row + size_t(i / 32) * 34;
        return h2f(uint16_t(b[0] | (b[1] << 8))) * float(int8_t(b[2 + i % 32]));
    }
}

// one warp per key row
template <class KT>
__global__ void index_score_kernel(const float* q, int heads, int dim, const KT* keys, int key_stride, int rows,
                                   const float* w, float w_scale, float* score) {
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const KT* k = keys + size_t(r) * key_stride;
    float total = 0;
    for (int h = 0; h < heads; ++h) {
        float s = 0;
        for (int i = lane; i < dim; i += 32) s += kv_ld(k + i) * q[size_t(h) * dim + i];
        s = warp_sum(s);
        s = __shfl_sync(0xffffffff, s, 0);
        total += fmaxf(s, 0.0f) * (w[h] * w_scale);
    }
    if (!lane) score[r] = total;
}

// index scores of a pass: token t (blockIdx.y) scores its (pos0 + t + 1) / ratio visible rows
template <class KT>
__global__ void index_score_batched_kernel(const float* q, int heads, int dim, const KT* keys, int key_stride, int pos0, int ratio,
                                           const float* w, float w_scale, float* score, int score_stride, const int* tpos, size_t kss) {
    const int t = blockIdx.y, rows = ((tpos ? tpos[t] : pos0 + t) + 1) / ratio;
    keys += size_t(t) * kss;
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    q += size_t(t) * heads * dim;
    w += size_t(t) * heads;
    const KT* k = keys + size_t(r) * key_stride;
    float total = 0;
    for (int h = 0; h < heads; ++h) {
        float s = 0;
        for (int i = lane; i < dim; i += 32) s += kv_ld(k + i) * q[size_t(h) * dim + i];
        s = warp_sum(s);
        s = __shfl_sync(0xffffffff, s, 0);
        total += fmaxf(s, 0.0f) * (w[h] * w_scale);
    }
    if (!lane) score[size_t(t) * score_stride + r] = total;
}

// the index scores over a Q8_0 cache (row_bytes per row, keys from value key_off): the batched kernel's work
__global__ void index_score_q8_kernel(const float* q, int heads, int dim, const uint8_t* cache, size_t row_bytes, int key_off, int pos0,
                                      int ratio, const float* w, float w_scale, float* score, int score_stride, int fixed_rows, const int* tpos,
                                      size_t css) {
    const int t = blockIdx.y, rows = tpos ? (tpos[t] + 1) / ratio : fixed_rows > 0 ? fixed_rows : (pos0 + t + 1) / ratio;
    cache += size_t(t) * css;
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    q += size_t(t) * heads * dim;
    w += size_t(t) * heads;
    const uint8_t* row = cache + size_t(r) * row_bytes;
    float kv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) kv[j] = lane + 32 * j < dim ? kv_at<2>(row, key_off + lane + 32 * j) : 0.0f;
    float total = 0;
    for (int h = 0; h < heads; ++h) {
        float s = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) if (lane + 32 * j < dim) s += kv[j] * q[size_t(h) * dim + lane + 32 * j];
        s = warp_sum(s);
        s = __shfl_sync(0xffffffff, s, 0);
        total += fmaxf(s, 0.0f) * (w[h] * w_scale);
    }
    if (!lane) score[size_t(t) * score_stride + r] = total;
}

// STRATA_DS41_INDEX_FAST: the index scores with a thread per cache row and a block per (256 rows, token): the token's
// queries [heads][dim] and head weights in LDS, the row's key decoded 32 values at a time (one Q8_0 block, or 32 floats /
// halves), all heads' dots in registers. heads <= 32, dim % 32 == 0, heads * dim <= 4096. (The warp-per-row kernels
// reduce across the warp once per head.) Same products, another summation order.
template <int KIND>
__global__ void __launch_bounds__(256) index_score_fast_kernel(const float* q, int heads, int dim, const uint8_t* cache, size_t row_bytes,
                                                               int key_off, int pos0, int ratio, const float* w, float w_scale, float* score,
                                                               int score_stride, int fixed_rows, const int* tpos, size_t css) {
    __shared__ float qs[4096];
    __shared__ float ws[32];
    const int t = blockIdx.y;
    const int rows = tpos ? (tpos[t] + 1) / ratio : fixed_rows > 0 ? fixed_rows : (pos0 + t + 1) / ratio;
    if (int(blockIdx.x) * 256 >= rows) return;
    cache += size_t(t) * css;
    q += size_t(t) * heads * dim;
    for (int i = threadIdx.x; i < heads * dim; i += 256) qs[i] = q[i];
    if (threadIdx.x < heads) ws[threadIdx.x] = w[size_t(t) * heads + threadIdx.x] * w_scale;
    __syncthreads();
    const int r = blockIdx.x * 256 + threadIdx.x;
    if (r >= rows) return;
    const uint8_t* row = cache + size_t(r) * row_bytes;
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.0f;
    for (int c0 = 0; c0 < dim; c0 += 32) {
        float k[32];
        if constexpr (KIND == 2) {
            const uint8_t* b = row + size_t((key_off + c0) / 32) * 34;
            const float d = h2f(uint16_t(b[0] | (b[1] << 8)));
#pragma unroll
            for (int j = 0; j < 32; ++j) k[j] = d * float(int8_t(b[2 + j]));
        } else {
#pragma unroll
            for (int j = 0; j < 32; ++j) k[j] = kv_at<KIND>(row, key_off + c0 + j);
        }
#pragma unroll
        for (int h = 0; h < 32; ++h) {
            if (h < heads) {
                const float* qh = qs + h * dim + c0;
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    const float4 qv = *reinterpret_cast<const float4*>(qh + j);
                    s += k[j] * qv.x + k[j + 1] * qv.y + k[j + 2] * qv.z + k[j + 3] * qv.w;
                }
                acc[h] += s;
            }
        }
    }
    float total = 0.0f;
#pragma unroll
    for (int h = 0; h < 32; ++h) if (h < heads) total += fmaxf(acc[h], 0.0f) * ws[h];
    score[size_t(t) * score_stride + r] = total;
}
// a Q8_0 cache's selected rows as floats (see gather_sel_h_kernel)
__global__ void gather_sel_q8_kernel(const uint8_t* cache, size_t row_bytes, int rw, const int* sel, int topk, int pos0, int ratio, float* out,
                                     const int* tpos, size_t css) {
    const int j = blockIdx.x, t = blockIdx.y, visible = ((tpos ? tpos[t] : pos0 + t) + 1) / ratio;
    cache += size_t(t) * css;
    if (j >= min(topk, visible)) return;
    const int s = min(max(sel[size_t(t) * topk + j], 0), visible - 1);
    const uint8_t* src = cache + size_t(s) * row_bytes;
    float* dst = out + (size_t(t) * topk + j) * rw;
    for (int i = threadIdx.x; i < rw; i += blockDim.x) dst[i] = kv_at<2>(src, i);
}
// float rows -> Q8_0 rows (block (row, 32-value block); max |x| / 127, round to nearest)
__global__ void quant_rows_q8_kernel(const float* x, int rw, uint8_t* out, size_t row_bytes) {
    const int r = blockIdx.x, b = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (b * 32 >= rw) return;
    const float v = x[size_t(r) * rw + b * 32 + lane];
    float a = fabsf(v);
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    const float d = a / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
    uint8_t* blk = out + size_t(r) * row_bytes + size_t(b) * 34;
    blk[2 + lane] = uint8_t(int8_t(__float2int_rn(v * id)));
    if (lane == 0) {   // FP16 bits of d
        const uint16_t h = f2h(d);
        blk[0] = uint8_t(h & 0xff);
        blk[1] = uint8_t(h >> 8);
    }
}


// GEMM index scoring (STRATA_DS41_INDEX_GEMM): a Q8_0 cache's keys [row0, row0 + n) as FP16 [n][dim]
__global__ void keys_q8_f16_kernel(const uint8_t* cache, size_t row_bytes, int key_off, int dim, int row0, uint16_t* out) {
    const int r = blockIdx.x;
    const uint8_t* row = cache + size_t(row0 + r) * row_bytes;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = kv_at<2>(row, key_off + i);
        out[size_t(r) * dim + i] = f2h(v);
    }
}
// scores[t][row0 + r] = w_scale sum_h w[t][h] relu(c[(t heads + h) n + r]) for the token's visible rows
__global__ void index_reduce_kernel(const float* c, int n, int heads, const float* w, float w_scale, int pos0, int ratio, int row0,
                                    float* score, int score_stride) {
    const int t = blockIdx.y, r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n || row0 + r >= (pos0 + t + 1) / ratio) return;
    const float* wt = w + size_t(t) * heads;
    float total = 0;
    for (int h = 0; h < heads; ++h) total += fmaxf(c[(size_t(t) * heads + h) * n + r], 0.0f) * (wt[h] * w_scale);
    score[size_t(t) * score_stride + row0 + r] = total;
}

// an FP16 cache's selected rows as floats: out[(t topk + j) rw ..] = cache[sel[t topk + j] rw ..] for the token's first
// min(topk, visible) picks (an index past the visible rows - a stale slot - is clamped; attention never reads it)
__global__ void gather_sel_h_kernel(const uint16_t* cache, int rw, const int* sel, int topk, int pos0, int ratio, float* out, const int* tpos,
                                    size_t css) {
    const int j = blockIdx.x, t = blockIdx.y, visible = ((tpos ? tpos[t] : pos0 + t) + 1) / ratio;
    cache += size_t(t) * css;
    if (j >= min(topk, visible)) return;
    const int s = min(max(sel[size_t(t) * topk + j], 0), visible - 1);
    const uint16_t* src = cache + size_t(s) * rw;
    float* dst = out + (size_t(t) * topk + j) * rw;
    for (int i = threadIdx.x; i < rw; i += blockDim.x) dst[i] = h2f(src[i]);
}

// top-k (radix select), one block of kTopkThreads per token
constexpr int kTopkThreads = 256;

__device__ inline uint32_t order_key(float v) {   // ascending in v
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The key at rank k (1-based, from the largest) of n keys and how many keys equal to it the top k takes (the
// lowest indices first). get(i) gives key i. Every thread returns both.
template <class Get>
__device__ void radix_threshold(Get get, int n, int k, uint32_t* hist, uint32_t* shared2, uint32_t& thr, int& take, uint32_t* whist = nullptr) {
    uint32_t prefix = 0, mask = 0;
    int remaining = k;
    // whist (8 x 256, with 256 threads): a sub-histogram per warp, summed into hist - the high-byte passes put most
    // keys in a few bins, where one shared histogram serializes the block's atomics (the same counts either way)
    const bool split = whist != nullptr && blockDim.x == 256 && n > 4096;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (split) {
            for (int b = threadIdx.x; b < 8 * 256; b += blockDim.x) whist[b] = 0;
            __syncthreads();
            uint32_t* wh = whist + (threadIdx.x / 32) * 256;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const uint32_t key = get(i);
                if ((key & mask) == prefix) atomicAdd(&wh[(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            for (int b = threadIdx.x; b < 256; b += blockDim.x) {
                uint32_t sum = 0;
                for (int w = 0; w < 8; ++w) sum += whist[w * 256 + b];
                hist[b] = sum;
            }
        } else {
            for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const uint32_t key = get(i);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int above = 0, b = 255;
            for (; b > 0; --b) {
                if (above + int(hist[b]) >= remaining) break;
                above += int(hist[b]);
            }
            shared2[0] = prefix | (uint32_t(b) << shift);
            shared2[1] = uint32_t(remaining - above);
        }
        __syncthreads();
        prefix = shared2[0];
        remaining = int(shared2[1]);
        mask |= 255u << shift;
        __syncthreads();
    }
    thr = prefix;
    take = remaining;
}

// exclusive block scan of v (blockDim.x values); returns the prefix, *total the sum
__device__ inline int block_scan(int v, int* buf, int* total) {
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int d = 1; d < blockDim.x; d *= 2) {
        const int add = threadIdx.x >= d ? buf[threadIdx.x - d] : 0;
        __syncthreads();
        buf[threadIdx.x] += add;
        __syncthreads();
    }
    const int incl = buf[threadIdx.x];
    *total = buf[blockDim.x - 1];
    __syncthreads();
    return incl - v;
}

// select_sorted with each thread on a contiguous run of the keys (ascending index order kept across threads): the
// equal keys' ranks and the picks' positions from one block scan each, instead of two scans per 256 keys (a 106K-key
// selection had ~13K barriers). The same picks in the same order.
template <class Get, class Keep>
__device__ int select_sorted_chunked(Get get, Keep keep, int n, int k, uint32_t* hist, uint32_t* shared2, int* scan, int* out,
                                     uint32_t* whist = nullptr) {
    const bool all = n <= k;
    uint32_t thr = 0;
    int take = 0;
    if (!all) radix_threshold(get, n, k, hist, shared2, thr, take, whist);
    const int per = (n + int(blockDim.x) - 1) / int(blockDim.x);
    const int i0 = min(n, int(threadIdx.x) * per), i1 = min(n, i0 + per);
    int eq = 0;
    if (!all)
        for (int i = i0; i < i1; ++i) eq += get(i) == thr ? 1 : 0;
    int eq_total;
    const int eq_base = block_scan(eq, scan, &eq_total);
    int cnt = 0, er = eq_base;
    for (int i = i0; i < i1; ++i) {
        bool picked = true;
        if (!all) {
            const uint32_t key = get(i);
            picked = key > thr || (key == thr && er < take);
            if (key == thr) ++er;
        }
        if (picked && keep(i)) ++cnt;
    }
    int total;
    const int base = block_scan(cnt, scan, &total);
    int at = base;
    er = eq_base;
    for (int i = i0; i < i1; ++i) {
        bool picked = true;
        if (!all) {
            const uint32_t key = get(i);
            picked = key > thr || (key == thr && er < take);
            if (key == thr) ++er;
        }
        if (picked && keep(i)) out[at++] = i;
    }
    return total;
}

// The k largest of n keys (ties to the lowest index), written in ascending index order where keep(i); the count kept.
template <class Get, class Keep>
__device__ int select_sorted(Get get, Keep keep, int n, int k, uint32_t* hist, uint32_t* shared2, int* scan, int* out) {
    if (n <= k) {
        int base = 0;
        for (int i0 = 0; i0 < n; i0 += blockDim.x) {
            const int i = i0 + threadIdx.x;
            const int f = i < n && keep(i) ? 1 : 0;
            int total;
            const int at = block_scan(f, scan, &total);
            if (f) out[base + at] = i;
            base += total;
        }
        return base;
    }
    uint32_t thr;
    int take;
    radix_threshold(get, n, k, hist, shared2, thr, take);
    int eq_base = 0, base = 0;
    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const uint32_t key = i < n ? get(i) : 0u;
        const int eq = i < n && key == thr ? 1 : 0;
        int eq_total;
        const int eq_rank = eq_base + block_scan(eq, scan, &eq_total);
        eq_base += eq_total;
        const bool picked = i < n && (key > thr || (eq && eq_rank < take));
        const int f = picked && keep(i) ? 1 : 0;
        int total;
        const int at = block_scan(f, scan, &total);
        if (f) out[base + at] = i;
        base += total;
    }
    return base;
}

// mode 0: plain; 1: candidate source (block maxima, the newest block pinned, top cand_k blocks into cand); 2: only rows
// in cand's blocks. sel [T][k], count [T].
__global__ void index_topk_kernel(const float* score, int score_stride, int pos0, int ratio, int k, int* sel, int* count, int mode,
                                  uint8_t* cand, int cand_stride, int bs, int cand_k, const int* tpos, int chunked) {
    __shared__ uint32_t hist[256], shared2[2];
    __shared__ uint32_t whist[8 * 256];   // the chunked selection's per-warp histograms
    __shared__ int scan[kTopkThreads];
    extern __shared__ float best[];   // mode 1: [blocks] maxima, then [cand_k] picks (as ints)
    const int t = blockIdx.x, visible = ((tpos ? tpos[t] : pos0 + t) + 1) / ratio;
    const float* s = score + size_t(t) * score_stride;
    int* out = sel + size_t(t) * k;
    if (visible == 0) {
        if (threadIdx.x == 0) count[t] = 0;
        return;
    }
    const uint8_t* keep_blocks = cand + size_t(t) * cand_stride;
    if (mode == 1) {
        const int nb = (visible + bs - 1) / bs;
        for (int b = threadIdx.x; b < nb; b += blockDim.x) {
            float m = -INFINITY;
            for (int r = b * bs; r < min(visible, (b + 1) * bs); ++r) m = fmaxf(m, s[r]);
            if (b == nb - 1) m += 1e30f;
            best[b] = m;
        }
        uint8_t* flags = cand + size_t(t) * cand_stride;
        for (int b = threadIdx.x; b < nb; b += blockDim.x) flags[b] = 0;
        __syncthreads();
        int* picks = reinterpret_cast<int*>(best + nb);
        auto best_get = [&](int i) { return order_key(best[i]); };
        auto all_keep = [](int) { return true; };
        const int n = chunked ? select_sorted_chunked(best_get, all_keep, nb, cand_k, hist, shared2, scan, picks)
                              : select_sorted(best_get, all_keep, nb, cand_k, hist, shared2, scan, picks);
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) flags[picks[j]] = 1;
        __syncthreads();
    }
    const bool masked = mode == 2;
    const uint32_t ninf = order_key(-INFINITY);
    auto get = [&](int i) { return masked && !keep_blocks[i / bs] ? ninf : order_key(s[i]); };
    auto keep = [&](int i) { return !(masked && !keep_blocks[i / bs]) && !(isinf(s[i]) && s[i] < 0); };
    const int n = chunked ? select_sorted_chunked(get, keep, visible, k, hist, shared2, scan, out, whist)
                          : select_sorted(get, keep, visible, k, hist, shared2, scan, out);
    if (threadIdx.x == 0) count[t] = n;
}


// STRATA_DS41_TOPK_2STAGE: the selection of index_topk_kernel (modes 0 and 2) in two stages. Stage 1, block (chunk,
// token): the top k of keys [c kChunk2, +kChunk2) of the token's visible rows, kept picks as global indices in
// ascending order (cand[t][c][..], cnt[t][c]). Stage 2, a block per token: the top k over the concatenated
// candidates. Exact: a key in the global top k (key desc, index asc) has fewer than k keys above it, so it is in its
// chunk's top k; the concatenation keeps index order (the lowest-index tie rule), and masked / -inf keys sort below
// every kept key in both.
constexpr int kChunk2 = 8192;
__global__ void index_topk2_stage1(const float* score, int score_stride, int pos0, int ratio, int k, int mode, const uint8_t* cand, int cand_stride,
                                   int bs, const int* tpos, int nchunks, int* cbuf, int* ccnt) {
    __shared__ uint32_t hist[256], shared2[2];
    __shared__ uint32_t whist[8 * 256];
    __shared__ int scan[kTopkThreads];
    const int c = blockIdx.x, t = blockIdx.y, visible = ((tpos ? tpos[t] : pos0 + t) + 1) / ratio;
    const int c0 = c * kChunk2, n = max(0, min(visible, c0 + kChunk2) - c0);
    int* out = cbuf + (size_t(t) * nchunks + c) * k;
    if (n == 0) { if (threadIdx.x == 0) ccnt[size_t(t) * nchunks + c] = 0; return; }
    const float* s = score + size_t(t) * score_stride + c0;
    const uint8_t* keep_blocks = cand + size_t(t) * cand_stride;
    const bool masked = mode == 2;
    const uint32_t ninf = order_key(-INFINITY);
    auto get = [&](int i) { return masked && !keep_blocks[(c0 + i) / bs] ? ninf : order_key(s[i]); };
    auto keep = [&](int i) { return !(masked && !keep_blocks[(c0 + i) / bs]) && !(isinf(s[i]) && s[i] < 0); };
    const int m = select_sorted_chunked(get, keep, n, k, hist, shared2, scan, out, whist);
    __syncthreads();
    for (int j = threadIdx.x; j < m; j += blockDim.x) out[j] += c0;
    if (threadIdx.x == 0) ccnt[size_t(t) * nchunks + c] = m;
}
__global__ void index_topk2_stage2(const float* score, int score_stride, int pos0, int ratio, int k, const int* tpos, int nchunks, const int* cbuf,
                                   const int* ccnt, int* sel, int* count) {
    __shared__ uint32_t hist[256], shared2[2];
    __shared__ uint32_t whist[8 * 256];
    __shared__ int scan[kTopkThreads];
    __shared__ int offs[1024 + 1];
    const int t = blockIdx.x, visible = ((tpos ? tpos[t] : pos0 + t) + 1) / ratio;
    int* out = sel + size_t(t) * k;
    if (visible == 0) { if (threadIdx.x == 0) count[t] = 0; return; }
    if (threadIdx.x == 0) {
        offs[0] = 0;
        for (int c = 0; c < nchunks; ++c) offs[c + 1] = offs[c] + ccnt[size_t(t) * nchunks + c];
    }
    __syncthreads();
    const int M = offs[nchunks];
    const float* s = score + size_t(t) * score_stride;
    const int* cb = cbuf + size_t(t) * nchunks * k;
    auto at = [&](int j) {   // candidate j -> its global row
        int lo = 0, hi = nchunks - 1;
        while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (offs[mid] <= j) lo = mid; else hi = mid - 1; }
        return cb[size_t(lo) * k + (j - offs[lo])];
    };
    auto get = [&](int j) { return order_key(s[at(j)]); };
    auto keep = [](int) { return true; };   // (stage 1 kept only the kept rows)
    const int m = select_sorted_chunked(get, keep, M, k, hist, shared2, scan, out, whist);
    __syncthreads();
    // local candidate positions -> rows, in place (each thread its own slots, read before written)
    for (int j = threadIdx.x; j < m; j += blockDim.x) out[j] = at(out[j]);
    if (threadIdx.x == 0) count[t] = m;
}

// the batched attention's per-token selected-row counts from the GPU indexer
__global__ void patch_info_kernel(int* info, const int* count, int tokens, int pos0, int ratio) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;
    info[4 * t + 2] = (pos0 + t + 1) / ratio > 0 ? count[t] : 0;
}

// compress2 over consecutive pairs: out row r from (2r, 2r + 1) of v and s ([rows * 2][n])
__global__ void compress2_rows_kernel(const float* v, const float* s, float* out, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= n) return;
    const size_t a = size_t(2 * r) * n + c, b = a + n;
    const float mx = fmaxf(s[a], s[b]);
    const float e0 = expf(s[a] - mx), e1 = expf(s[b] - mx);
    const float inv = 1.0f / (e0 + e1);
    out[size_t(r) * n + c] = __fadd_rn(__fmul_rn(v[a], e0 * inv), __fmul_rn(v[b], e1 * inv));
}

// count blocks of rows x width bytes: block c's row r from src + c sstep + r spitch to dst + c dstep + r dpitch
__global__ void copy_rows_kernel(uint8_t* dst, size_t dpitch, size_t dstep, const uint8_t* src, size_t spitch, size_t sstep,
                                 int width, int rows) {
    const int c = blockIdx.y;
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        uint8_t* d = dst + size_t(c) * dstep + size_t(r) * dpitch;
        const uint8_t* s = src + size_t(c) * sstep + size_t(r) * spitch;
        for (int i = threadIdx.x; i < width; i += blockDim.x) d[i] = s[i];
    }
}

__global__ void compress2_kernel(const float* v0, const float* s0, const float* v1, const float* s1, float* out, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const float mx = fmaxf(s0[c], s1[c]);
    const float e0 = expf(s0[c] - mx), e1 = expf(s1[c] - mx);
    const float inv = 1.0f / (e0 + e1);
    out[c] = __fadd_rn(__fmul_rn(v0[c], e0 * inv), __fmul_rn(v1[c], e1 * inv));
}


// sessions (token t of a pass is session t at position tpos[t]): the compressor's step. Ratio 1: out[t] = v[t].
// Ratio 2: an even position stashes (v, s) as its group's first; an odd one merges with the stash into out[t].
__global__ void compress_sessions_kernel(const float* v, const float* s, float* stash_v, float* stash_s, int n, const int* tpos, int ratio,
                                         float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= n) return;
    const size_t i = size_t(t) * n + c;
    if (ratio == 1) { out[i] = v[i]; return; }
    if (tpos[t] % 2 == 0) { stash_v[i] = v[i]; stash_s[i] = s[i]; out[i] = v[i]; return; }   // (out: a finite row, not stored)
    const float mx = fmaxf(stash_s[i], s[i]);
    const float e0 = expf(stash_s[i] - mx), e1 = expf(s[i] - mx);
    const float inv = 1.0f / (e0 + e1);
    out[i] = __fadd_rn(__fmul_rn(stash_v[i], e0 * inv), __fmul_rn(v[i], e1 * inv));
}
// row t of src [rows][width] to dst + dst_row[t] * dst_stride (skipped when dst_row[t] < 0)
__global__ void scatter_rows_kernel(const float* src, int width, const int* dst_row, float* dst, size_t dst_stride) {
    const int t = blockIdx.y, r = dst_row[t];
    if (r < 0) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x)
        dst[size_t(r) * dst_stride + i] = src[size_t(t) * width + i];
}

// row t of src [rows][width bytes] to dst + dst_row[t] dst_pitch (skipped when dst_row[t] < 0); 4-byte words
__global__ void scatter_words_kernel(const uint32_t* src, int words, const int* dst_row, uint32_t* dst, size_t dst_pitch_words) {
    const int t = blockIdx.y, r = dst_row[t];
    if (r < 0) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x)
        dst[size_t(r) * dst_pitch_words + i] = src[size_t(t) * words + i];
}

// one block per stream
__global__ void engram_kernel(float* h, const float* kv, const float* g, int n, float eps) {
    __shared__ float red[3][32];
    h += size_t(blockIdx.y) * 4 * n;
    kv += size_t(blockIdx.y) * 5 * n;
    const int s = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32, warps = blockDim.x / 32;
    float* x = h + size_t(s) * n;
    const float* key = kv + size_t(s) * n;
    const float* value = kv + size_t(4) * n;
    float xx = 0, kk = 0;
    for (int j = tid; j < n; j += blockDim.x) { xx += x[j] * x[j]; kk += key[j] * key[j]; }
    xx = warp_sum(xx);
    kk = warp_sum(kk);
    if (!lane) { red[0][warp] = xx; red[1][warp] = kk; }
    __syncthreads();
    xx = kk = 0;
    for (int w = 0; w < warps; ++w) { xx += red[0][w]; kk += red[1][w]; }
    const float rx = 1.0f / sqrtf(xx / n + eps), rk = 1.0f / sqrtf(kk / n + eps);
    float dot = 0;
    for (int j = tid; j < n; j += blockDim.x) dot += (x[j] * rx) * (key[j] * rk) * g[size_t(s) * n + j];
    dot = warp_sum(dot);
    __syncthreads();
    if (!lane) red[2][warp] = dot;
    __syncthreads();
    dot = 0;
    for (int w = 0; w < warps; ++w) dot += red[2][w];
    dot *= 1.0f / sqrtf(float(n));
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float gate = sigmoid(dot < 0 ? -mag : mag);
    for (int j = tid; j < n; j += blockDim.x) x[j] = __fadd_rn(x[j], __fmul_rn(value[j], gate));
}

__global__ void axpy_kernel(const float* x, float a, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __fadd_rn(y[i], __fmul_rn(a, x[i]));
}

__global__ void add_kernel(const float* a, const float* b, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

// one 256-thread block per row, float4 loads (in % 4 == 0 and 16-byte aligned rows) else scalar
__global__ void __launch_bounds__(256) f32_gemv_kernel(const float* w, const float* x, float* y, int rows, int in) {
    __shared__ float red[8];
    const int r = blockIdx.x, tid = threadIdx.x;
    const float* row = w + size_t(r) * in;
    x += size_t(blockIdx.y) * in;
    y += size_t(blockIdx.y) * rows;
    float s = 0;
    if ((in & 3) == 0 && ((reinterpret_cast<uintptr_t>(row) | reinterpret_cast<uintptr_t>(x)) & 15) == 0) {
        const float4* r4 = reinterpret_cast<const float4*>(row);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int i = tid; i < in / 4; i += 256) {
            const float4 a = r4[i], b = x4[i];
            s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = tid; i < in; i += 256) s += row[i] * x[i];
    }
    s = warp_sum(s);
    if (tid % 32 == 0) red[tid / 32] = s;
    __syncthreads();
    if (tid == 0) {
        float t = 0;
        for (int i = 0; i < 8; ++i) t += red[i];
        y[r] = t;
    }
}

// y[t][r] = W_r . x[t]: one 256-thread block per (row, token), BF16 weights
__global__ void __launch_bounds__(256) bf16_gemv_kernel(const uint16_t* w, const float* x, float* y, int rows, int in) {
    __shared__ float red[8];
    const int r = blockIdx.x, tid = threadIdx.x;
    const uint16_t* row = w + size_t(r) * in;
    x += size_t(blockIdx.y) * in;
    float s = 0;
    for (int i = tid; i < in; i += 256) s += __uint_as_float(unsigned(row[i]) << 16) * x[i];
    s = warp_sum(s);
    if (tid % 32 == 0) red[tid / 32] = s;
    __syncthreads();
    if (tid == 0) {
        float t = 0;
        for (int i = 0; i < 8; ++i) t += red[i];
        y[size_t(blockIdx.y) * rows + r] = t;
    }
}

__global__ void gather_kernel(const float* src, const int* idx, int n, float* dst) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) dst[size_t(blockIdx.y) * n + j] = src[size_t(idx[blockIdx.y]) * n + j];
}

// rows in order, so a destination row receiving several sources adds them in source order
__global__ void scatter_axpy_kernel(const float* src, const int* idx, const float* w, int m, int n, float* dst) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    for (int i = 0; i < m; ++i) {
        float* d = dst + size_t(idx[i]) * n + j;
        *d = __fadd_rn(*d, __fmul_rn(w[i], src[size_t(i) * n + j]));
    }
}

__global__ void to_bf16_kernel(const float* x, uint16_t* y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t u = __float_as_uint(x[i]);
    y[i] = uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);   // round to nearest even (finite inputs)
}

unsigned blocks(int n, int per) { return unsigned((n + per - 1) / per); }



// dst[b][a][j] = src[a][b][j] for a < na, b < nb, j < width (floats)
__global__ void swap_blocks_kernel(const float* src, float* dst, int na, int nb, int width) {
    const int a = blockIdx.y, b = blockIdx.z;
    const float* s = src + (size_t(a) * nb + b) * width;
    float* d = dst + (size_t(b) * na + a) * width;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x) d[j] = s[j];
}
// ---- DSpark drafter: few-token GEMVs that read each weight row once ----
constexpr int kMtMax = 8;   // tokens per launch
struct MtRows { int n; int row[kMtMax]; };   // the x rows of the launch's tokens

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }

// 8 rows per block (a warp each) over K tiles of x staged in LDS for the n tokens: y[t ldy + r] = W[r] . x[rows.row[t] ldx]
constexpr int kMtTile = 512;
// each lane: 16 BF16 of the tile (two 16-byte loads), so a warp has 1 KiB of the row in flight
__global__ void __launch_bounds__(256) bf16_gemv_mt_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x, int ldx, MtRows rows,
                                                           float* y, int ldy, int out, int in, size_t w_gstride, int x_goff, int y_goff) {
    __shared__ float xs[kMtMax][kMtTile];
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    w += size_t(blockIdx.y) * w_gstride;   // group g (ds41_bf16_gemv_mt_groups): its matrix, columns and rows
    x += size_t(blockIdx.y) * size_t(x_goff);
    y += size_t(blockIdx.y) * size_t(y_goff);
    const uint16_t* wr = w + size_t(min(r, out - 1)) * in;
    float acc[kMtMax] = {};
    for (int k0 = 0; k0 < in; k0 += kMtTile) {
        const int kn = min(kMtTile, in - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < rows.n * kn; i += 256) xs[i / kn][i % kn] = x[size_t(rows.row[i / kn]) * ldx + k0 + i % kn];
        __syncthreads();
        const int k = lane * 16;
        if (k < kn) {
            const uint4* src = reinterpret_cast<const uint4*>(wr + k0 + k);
            const uint4 u0 = src[0], u1 = src[1];
            const uint32_t q[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
            for (int t = 0; t < kMtMax; ++t)
                if (t < rows.n) {
                    float s2 = 0;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        s2 += __uint_as_float(q[j] << 16) * xs[t][k + 2 * j] + __uint_as_float(q[j] & 0xffff0000u) * xs[t][k + 2 * j + 1];
                    acc[t] += s2;
                }
        }
    }
    if (r >= out) return;
#pragma unroll
    for (int t = 0; t < kMtMax; ++t) {
        float v = acc[t];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && t < rows.n) y[size_t(t) * ldy + r] = v;
    }
}

// MXFP4 (ggml block_mxfp4: an E8M0 scale and 16 bytes, the low nibbles values 0..15, the high ones 16..31): each warp
// loads its row's tile (16 blocks, 272 bytes) with word loads into LDS, then two lanes per block decode it
__constant__ float kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
__global__ void __launch_bounds__(256) mxfp4_gemv_mt_kernel(const uint8_t* __restrict__ w, size_t row_bytes, const float* __restrict__ x, int ldx,
                                                            MtRows rows, float* y, int ldy, int out, int in) {
    __shared__ float xs[kMtMax][kMtTile];
    __shared__ uint32_t wt[8][68];   // 16 blocks x 17 bytes per warp
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const uint8_t* wr = w + size_t(min(r, out - 1)) * row_bytes;
    const int half = lane & 1, bl = lane / 2;
    const uint8_t* wb = reinterpret_cast<const uint8_t*>(wt[warp]);
    float acc[kMtMax] = {};
    for (int k0 = 0; k0 < in; k0 += kMtTile) {
        const int kn = min(kMtTile, in - k0), nb = kn / 32;
        __syncthreads();
        for (int i = threadIdx.x; i < rows.n * kn; i += 256) xs[i / kn][i % kn] = x[size_t(rows.row[i / kn]) * ldx + k0 + i % kn];
        const uint8_t* tile = wr + size_t(k0 / 32) * 17;
        if ((reinterpret_cast<uintptr_t>(tile) & 3) == 0) {
            for (int i = lane; i < (nb * 17 + 3) / 4; i += 32) wt[warp][i] = reinterpret_cast<const uint32_t*>(tile)[i];
        } else {
            for (int i = lane; i < nb * 17; i += 32) reinterpret_cast<uint8_t*>(wt[warp])[i] = tile[i];
        }
        __syncthreads();
        if (bl < nb) {
            const uint8_t* blk = wb + bl * 17;
            const uint32_t e = blk[0];
            const float d = e < 2 ? __uint_as_float(0x00200000u << e) : __uint_as_float((e - 1) << 23);   // 2^(e - 128)
            float v[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = blk[1 + j];
                v[j] = kFp4[half ? q >> 4 : q & 15] * d;
            }
            const int base = bl * 32 + half * 16;
#pragma unroll
            for (int t = 0; t < kMtMax; ++t)
                if (t < rows.n) {
                    float s2 = 0;
#pragma unroll
                    for (int j = 0; j < 16; ++j) s2 += v[j] * xs[t][base + j];
                    acc[t] += s2;
                }
        }
    }
    if (r >= out) return;
#pragma unroll
    for (int t = 0; t < kMtMax; ++t) {
        float v = acc[t];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && t < rows.n) y[size_t(t) * ldy + r] = v;
    }
}

// DSpark MoE on the device: per token (block) sqrt(softplus) scores, top-k on score + bias, normalized weights x scale
__global__ void moe_route_kernel(const float* logits, const float* bias, int experts, int k, float scale, int normalize, int* ids, float* w) {
    extern __shared__ float sc[];   // experts scores, then experts selection keys
    const int t = blockIdx.x;
    float* prob = sc;
    float* key = sc + experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        const float x = logits[size_t(t) * experts + e];
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        prob[e] = sqrtf(sp);
        key[e] = prob[e] + bias[e];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        int pick[16];
        for (int j = 0; j < k; ++j) {
            int best = 0;
            for (int e = 1; e < experts; ++e) if (key[e] > key[best]) best = e;
            pick[j] = best;
            key[best] = -INFINITY;
            sum += prob[best];
        }
        for (int j = 0; j < k; ++j) {
            float v = prob[pick[j]];
            if (normalize && k > 1) v /= sum + 1e-20f;
            ids[t * k + j] = pick[j];
            w[t * k + j] = v * scale;
        }
    }
}

// pair p: y[p ldy + r] = W_{ids[p]}[r] . x[(p / div) ldx] (MXFP4 experts expert_bytes apart), 8 rows per block
__global__ void __launch_bounds__(256) mxfp4_moe_kernel(const uint8_t* __restrict__ w, size_t expert_bytes, size_t row_bytes, const int* ids,
                                                        const float* __restrict__ x, int ldx, int div, float* y, int ldy, int out, int in) {
    // (each lane had read its 17-byte block from global memory a byte at a time: the tile now goes through LDS with
    // word loads, and the products are mxfp4_gemv_mt_kernel's, so a device-routed draft is bitwise the host-routed one)
    __shared__ float xs[kMtTile];
    __shared__ uint32_t wt[8][68];
    const int p = blockIdx.y, r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const uint8_t* wr = w + size_t(ids[p]) * expert_bytes + size_t(min(r, out - 1)) * row_bytes;
    const float* xp = x + size_t(p / div) * ldx;
    const int half = lane & 1, bl = lane / 2;
    const uint8_t* wb = reinterpret_cast<const uint8_t*>(wt[warp]);
    float acc = 0;
    for (int k0 = 0; k0 < in; k0 += kMtTile) {
        const int kn = min(kMtTile, in - k0), nb = kn / 32;
        __syncthreads();
        for (int i = threadIdx.x; i < kn; i += 256) xs[i] = xp[k0 + i];
        const uint8_t* tile = wr + size_t(k0 / 32) * 17;
        if ((reinterpret_cast<uintptr_t>(tile) & 3) == 0) {
            for (int i = lane; i < (nb * 17 + 3) / 4; i += 32) wt[warp][i] = reinterpret_cast<const uint32_t*>(tile)[i];
        } else {
            for (int i = lane; i < nb * 17; i += 32) reinterpret_cast<uint8_t*>(wt[warp])[i] = tile[i];
        }
        __syncthreads();
        if (bl < nb) {
            const uint8_t* blk = wb + bl * 17;
            const uint32_t e = blk[0];
            const float d = e < 2 ? __uint_as_float(0x00200000u << e) : __uint_as_float((e - 1) << 23);   // 2^(e - 128)
            float v[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = blk[1 + j];
                v[j] = kFp4[half ? q >> 4 : q & 15] * d;
            }
            const int base = bl * 32 + half * 16;
            float s2 = 0;
#pragma unroll
            for (int j = 0; j < 16; ++j) s2 += v[j] * xs[base + j];
            acc += s2;
        }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0 && r < out) y[size_t(p) * ldy + r] = acc;
}

// DSpark's Markov chain on the device: row = w1[tok[i]] (BF16 -> f32); the argmax of v (the first of equal maxima)
// into tok[i + 1]; conf[i] = sigmoid(cw . [x; row])
__global__ void markov_row_kernel(const uint16_t* w1, int dim, const int* tok, int i, float* row) {
    const uint16_t* r = w1 + size_t(tok[i]) * dim;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) row[j] = __uint_as_float(uint32_t(r[j]) << 16);
}
__global__ void __launch_bounds__(1024) argmax_kernel(const float* v, int n, int* tok, int i) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int j = threadIdx.x; j < n; j += 1024)
        if (v[j] > best) { best = v[j]; at = j; }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, at, o);
        if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
    }
    if (threadIdx.x % 32 == 0) { bv[threadIdx.x / 32] = best; bi[threadIdx.x / 32] = at; }
    __syncthreads();
    if (threadIdx.x < 32) {
        best = bv[threadIdx.x];
        at = bi[threadIdx.x];
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, at, o);
            if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
        }
        if (threadIdx.x == 0) tok[i + 1] = at;
    }
}
__global__ void conf_kernel(const float* cw, const float* x, int hidden, const float* row, int dim, float* conf, int i) {
    __shared__ float part[32];
    float s = 0;
    for (int j = threadIdx.x; j < hidden + dim; j += blockDim.x) s += cw[j] * (j < hidden ? x[j] : row[j - hidden]);
    s = warp_sum(s);
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0;
        s = warp_sum(s);
        if (threadIdx.x == 0) conf[i] = 1.0f / (1.0f + expf(-s));
    }
}
// dst[t][j] = mean over the 4 streams of h[t][s][j]
__global__ void stream_mean_kernel(const float* __restrict__ h, float* dst, size_t ld_dst, int n, int tokens) {
    const int t = blockIdx.y;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const float* ht = h + size_t(t) * 4 * n;
        dst[size_t(t) * ld_dst + j] = 0.25f * (ht[j] + ht[n + j] + ht[2 * n + j] + ht[3 * n + j]);
    }
}
}  // namespace

void ds41_hc_coeff(const float* mixes, const float* scale, const float* base, float* coeff, int iterations, float eps,
                   void* stream, int tokens) {
    hc_coeff_kernel<<<unsigned(tokens), 16, 0, (cudaStream_t)stream>>>(mixes, scale, base, coeff, iterations, eps);
    check("hc_coeff");
}

bool ds41_hc_mixes(const float* h, const float* fn, const float* scale, const float* base, float* flat, float* mixes, float* coeff, int n,
                   int iterations, float eps, float rms_eps, void* stream, int tokens) {
    const auto aligned = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    if (n < 1 || !aligned(h) || !aligned(fn) || !aligned(flat)) return false;
    hc_mixes_kernel<<<unsigned(tokens), 1024, 0, (cudaStream_t)stream>>>(h, fn, scale, base, flat, mixes, coeff, n, iterations, eps, rms_eps);
    check("hc_mixes");
    return true;
}

void ds41_hc_pre(const float* h, const float* w, float* y, int n, void* stream, int tokens) {
    hc_pre_kernel<<<dim3(blocks(n, 256), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(h, w, y, n);
    check("hc_pre");
}

void ds41_hc_post(const float* x, const float* res, const float* post, const float* comb, float* out, int n, void* stream,
                  int tokens) {
    hc_post_kernel<<<dim3(blocks(n, 256), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(x, res, post, comb, out, n);
    check("hc_post");
}

void ds41_route_table(const float* logits, const float* bias, const unsigned long long* slots, int T, int E, int K, bool normalize,
                      float scale, int cpu_topk, int cap, int* ids, float* wts, void* table, void* stream) {
    if (E > 1024 || K > 16) throw std::invalid_argument("ds41_route_table: experts <= 1024, top_k <= 16");
    route_table_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(logits, bias, slots, T, E, K, normalize ? 1 : 0, scale, cpu_topk, cap, ids,
                                                           wts, static_cast<unsigned char*>(table));
}

void ds41_scatter_axpy_dev(const float* src, const int* idx, const float* w, const int* m, int n, float* dst, void* stream) {
    scatter_axpy_dev_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t)stream>>>(src, idx, w, m, n, dst);
}

void ds41_defer_add(const float* y, float* h, const float* post, const float* comb, int n, void* stream, int tokens) {
    defer_add_kernel<<<dim3(blocks(n, 256), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(y, h, post, comb, n);
    check("defer_add");
}

void ds41_hc_post_sum(const float* moe, const float* peer, const float* gsum, float* y, const float* res, const float* post, const float* comb,
                      float* out, int n, void* stream, int tokens) {
    hc_post_sum_kernel<<<dim3(blocks(n, 256), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(moe, peer, gsum, y, res, post, comb, out, n);
    check("hc_post_sum");
}

void ds41_broadcast_streams(const float* x, float* h, int n, void* stream, int tokens) {
    broadcast_kernel<<<dim3(blocks(n, 256), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(x, h, n);
    check("broadcast");
}

void ds41_rope(float* x, int rows, int stride, int offset, int pairs, const float* cs, void* stream, int group, int cs_step) {
    if (group <= 0) group = rows;
    rope_kernel<<<blocks(rows * pairs, 256), 256, 0, (cudaStream_t)stream>>>(x, rows, stride, offset, pairs, cs, group, cs_step);
    check("rope");
}

void ds41_attention(const float* q, int heads, int dim, const float* window, int window_first, int window_ring,
                    int window_rows, const float* comp, int comp_stride, const int* sel, int nsel, const float* sinks,
                    float scale, float* scores, float* out, void* stream, const int* nsel_dev) {
    const Ring ring{window_first, window_ring};
    const int rows = window_rows + nsel;
    if (rows > kMaxRows || rows < 1 || heads % kHeadGroup || dim % kDimChunk) throw std::invalid_argument("ds41_attention: shape");
    const dim3 g1(unsigned(heads / kHeadGroup), unsigned((rows + kRowChunk - 1) / kRowChunk));
    attn_scores_kernel<<<g1, 256, size_t(kHeadGroup) * dim * sizeof(float), (cudaStream_t)stream>>>(
        q, dim, window, ring, window_rows, comp, comp_stride, sel, rows, scale, scores, nsel_dev);
    check("attention scores");
    // STRATA_DS41_ATTN_SPLIT=n: the output's rows over n blocks each (a partial buffer after the scores, then a sum)
    static const int split = std::getenv("STRATA_DS41_ATTN_SPLIT") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_ATTN_SPLIT"))) : 1;
    if (split > 1 && rows >= 2 * split) {
        float* part = scores + size_t(heads) * kMaxRows;   // the caller's scratch holds more than one token's scores
        const int chunk = (rows + split - 1) / split;
        const dim3 g3(unsigned(heads / kHeadGroup), unsigned(dim / kDimChunk), unsigned(split));
        attn_out_split_kernel<<<g3, 256, size_t(kHeadGroup) * chunk * sizeof(float), (cudaStream_t)stream>>>(
            scores, heads, dim, window, ring, window_rows, comp, comp_stride, sel, rows, sinks, part, nsel_dev);
        check("attention out split");
        attn_combine_kernel<<<unsigned((heads * dim + 255) / 256), 256, 0, (cudaStream_t)stream>>>(part, split, heads * dim, out);
        check("attention combine");
        return;
    }
    const dim3 g2(unsigned(heads / kHeadGroup), unsigned(dim / kDimChunk));
    attn_out_kernel<<<g2, 256, size_t(kHeadGroup) * rows * sizeof(float), (cudaStream_t)stream>>>(
        scores, dim, window, ring, window_rows, comp, comp_stride, sel, rows, sinks, out, nsel_dev);
    check("attention out");
}

#if defined(__HIP_PLATFORM_AMD__)
// this ROCm reports the device's arch as "gfx1201" (with an optional ":sramecc" suffix)
static bool ds41_device_is_gfx12() {
    static const bool v = [] {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return false;
        const std::string a(p.gcnArchName);
        return a.rfind("gfx1200", 0) == 0 || a.rfind("gfx1201", 0) == 0;
    }();
    return v;
}
#endif

// ---- gfx12 (RDNA4) WMMA attention, opt-in with STRATA_DS41_ATTN_WMMA=1 -------------------------------------------
// One block per (token, 16-head group); 256 threads = 8 waves. QK and PV on the gfx12 WMMA engine
// (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12: wave32, lane l -> half = l/16, id = l%16; A[i] = A[id][half*8+i],
// B[i] = B[half*8+i][id], C[i] = C[half*8+i][id]), which sustains ~40 TF/s at these shapes against ~10 for the CUDA-core
// kernel (tools/hip/wmma_attn_probe.hip). Tiles are 32 rows: this card has 64 KB of LDS per block, and the tile's K
// (32 x 512 FP16, 32 KB) plus Q (16 x 512 FP16, 16 KB) plus the scores/probabilities fit at 51 KB.
// QK: S[16 heads][32 rows] = Q[16][512] . K[32][512]^T, the 8 waves each summing 4 of the 32 k-steps (LDS atomics).
// PV: O[16][512] = P[16][32] . V[32][512], the 8 waves each owning a 64-dim output slice, V staged transposed.
#if defined(__HIP_PLATFORM_AMD__)
namespace wm12 {
constexpr int kDim = 512, kHeads = 16, kRows = 32, kWaves = 8;
constexpr size_t kLds = size_t(kHeads * kDim + kRows * kDim) * 2 + size_t(kHeads * kRows) * 4 + size_t(kHeads * kRows) * 2 +
                        size_t(4 * kHeads) * 4;   // Q + K/VT + S + P + (m, l, f, sink)
using f8 = float __attribute__((ext_vector_type(8)));
using h8 = _Float16 __attribute__((ext_vector_type(8)));
__device__ inline _Float16 bits_h(uint16_t u) { _Float16 f; __builtin_memcpy(&f, &u, 2); return f; }
__device__ inline float f16_to_f32(_Float16 f) { float r; __builtin_memcpy(&r, &f, 4); return r; }   // (not used: see below)
}  // namespace wm12

template <class CT>
__global__ __launch_bounds__(256) void attn_wmma_gfx12_kernel(
        const float* q, int heads, const float* window, int ring, const int4* info, const CT* comp, int comp_stride,
        const int* sel, int sel_stride, const float* sinks, float scale, float* out) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
    using namespace wm12;
    extern __shared__ __align__(16) unsigned char wsm[];
    _Float16* smQ = reinterpret_cast<_Float16*>(wsm);                       // [16][512]
    _Float16* smK = smQ + kHeads * kDim;                                    // [32][512], then V transposed [512][32]
    float* smS = reinterpret_cast<float*>(smK + kRows * kDim);              // [16][32]
    _Float16* smP = reinterpret_cast<_Float16*>(smS + kHeads * kRows);      // [16][32]
    float* smF = reinterpret_cast<float*>(smP + kHeads * kRows);            // [16] rescale factor of the max
    float* smM = smF + kHeads;                                             // [16] running max
    float* smL = smM + kHeads;                                             // [16] running sum
    float* smK0 = smL + kHeads;                                            // [16] sink
    const int t = blockIdx.x, h0 = blockIdx.y * kHeads;
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31, hf = lid >> 4, id = lid & 15;
    const int4 in = info[t];
    const int first = in.x, wrows = in.y, rows = in.y + in.z;
    const int* tsel = sel + size_t(t) * sel_stride;
    // Q for this 16-head group, FP32 -> FP16
    for (int i = tid; i < kHeads * kDim; i += 256) {
        const int h = i / kDim, d = i % kDim;
        smQ[i] = (_Float16) q[(size_t(t) * heads + h0 + h) * size_t(kDim) + d];
    }
    if (tid < kHeads) { smM[tid] = -INFINITY; smL[tid] = 0.f; smK0[tid] = sinks[h0 + tid]; }
    // one row of the tile, dim `d` (window rows from the FP32 ring, selected rows from the cache)
    auto tile_val = [&](int rr, int d) -> _Float16 {
        if (rr < wrows) return (_Float16) window[size_t((first + rr) % ring) * kDim + d];
        const CT* p = comp + size_t(tsel[rr - wrows]) * comp_stride + d;
        if constexpr (sizeof(CT) == 4) return (_Float16) (*reinterpret_cast<const float*>(p));
        else return bits_h(*reinterpret_cast<const uint16_t*>(p));
    };
    f8 o[4];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[c][i] = 0.f;
    __syncthreads();
    for (int r0 = 0; r0 < rows; r0 += kRows) {
        // ---- K for this tile
        for (int i = tid; i < kRows * kDim; i += 256) {
            const int r = i / kDim, d = i % kDim, rr = r0 + r;
            smK[i] = rr < rows ? tile_val(rr, d) : (_Float16) 0;
        }
        for (int i = tid; i < kHeads * kRows; i += 256) smS[i] = 0.f;   // the 8 waves' partial sums
        __syncthreads();
        // ---- QK: this wave sums its 4 k-steps (64 dims) into smS
        for (int c = 0; c < 2; ++c) {   // two n-tiles of 16 rows
            f8 s;
#pragma unroll
            for (int i = 0; i < 8; ++i) s[i] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                h8 a, b;
#pragma unroll
                for (int i = 0; i < 8; ++i) a[i] = smQ[id * kDim + wid * 64 + ks * 16 + hf * 8 + i];
#pragma unroll
                for (int i = 0; i < 8; ++i) b[i] = smK[(c * 16 + id) * kDim + wid * 64 + ks * 16 + hf * 8 + i];
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = hf * 8 + i, n = c * 16 + id;   // head, row of the tile
                if (r0 + n < rows) atomicAdd(&smS[m * kRows + n], s[i]);   // (8 waves, same 512 cells)
            }
        }
        __syncthreads();
        // ---- softmax over the tile, per head (16 threads of a head: thread th -> head th/16, rows (th%16)*2 + 0..1)
        {
            const int h = tid / 16, j = tid % 16;
            float mx = -INFINITY;
#pragma unroll
            for (int q2 = 0; q2 < 2; ++q2) {
                const int n = j * 2 + q2;
                const int rr = r0 + n;
                float v = rr < rows ? smS[h * kRows + n] * scale : -INFINITY;
                smS[h * kRows + n] = v;   // (scaled in place, -inf off the end)
                mx = fmaxf(mx, v);
            }
            for (int o2 = 8; o2; o2 /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o2, 16));
            const float nm = fmaxf(smM[h], mx);
            const float f = smM[h] == -INFINITY ? 0.f : expf(smM[h] - nm);
            float sum = 0;
#pragma unroll
            for (int q2 = 0; q2 < 2; ++q2) {
                const int n = j * 2 + q2;
                const float e = smS[h * kRows + n] == -INFINITY ? 0.f : expf(smS[h * kRows + n] - nm);
                smP[h * kRows + n] = (_Float16) e;   // A layout: P[head][row]
                sum += e;
            }
            for (int o2 = 8; o2; o2 /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o2, 16);
            if (j == 0) {
                smF[h] = f;
                smL[h] = smL[h] * f + sum;
                smM[h] = nm;
            }
        }
        __syncthreads();
        // ---- V transposed into the K buffer: smK[dim][row]
        for (int i = tid; i < kRows * kDim; i += 256) {
            const int r = i / kDim, d = i % kDim, rr = r0 + r;
            smK[size_t(d) * kRows + r] = rr < rows ? tile_val(rr, d) : (_Float16) 0;
        }
        __syncthreads();
        // ---- PV: each wave owns a 64-dim output slice; rescale the running sum by this tile's factor first
        const float f4 = smF[hf * 8];   // (the 16 lanes of a half share the same 8 heads' factors)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float fc[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) fc[i] = smF[hf * 8 + i];
            for (int i = 0; i < 8; ++i) o[c][i] *= fc[i];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {   // 2 k-steps of 16 rows
                h8 a, b;
#pragma unroll
                for (int i = 0; i < 8; ++i) a[i] = smP[id * kRows + ks * 16 + hf * 8 + i];
#pragma unroll
                for (int i = 0; i < 8; ++i) b[i] = smK[(wid * 64 + c * 16 + id) * kRows + ks * 16 + hf * 8 + i];
                o[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, o[c]);
            }
        }
        (void) f4;
        __syncthreads();   // before the next tile's K overwrites the V rows
    }
    // ---- epilogue: divide by the running sum (+ sink), write FP32 out
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = hf * 8 + i, d = wid * 64 + c * 16 + id;
            const float sink = smK0[h], M = fmaxf(smM[h], sink);
            const float fs = smM[h] == -INFINITY ? 0.f : expf(smM[h] - M);
            const float inv = 1.f / (smL[h] * fs + expf(sink - M));
            out[(size_t(t) * heads + h0 + h) * size_t(kDim) + d] = o[c][i] * inv;
        }
#endif
}
#endif  // __HIP_PLATFORM_AMD__

void ds41_attention_batched(const float* q, int tokens, int heads, int dim, const float* window, int ring, const int* info,
                            const float* comp, int comp_stride, const int* sel, int sel_stride, const float* sinks, float scale,
                            float* out, void* stream, bool comp_half) {
    static const bool v1 = std::getenv("STRATA_DS41_ATTN_V1") != nullptr;
#ifdef STRATA_DS41_WMMA
    static const bool wmma = std::getenv("STRATA_DS41_ATTN_WMMA") && std::string(std::getenv("STRATA_DS41_ATTN_WMMA")) == "1";
    if (wmma && dim == kWD && heads % kWG == 0) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(attn_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kWmmaShared));
            configured = true;
        }
        attn_wmma_kernel<<<dim3(unsigned(tokens), unsigned(heads / kWG)), 128, kWmmaShared, (cudaStream_t)stream>>>(
            q, heads, window, ring, reinterpret_cast<const int4*>(info), comp, comp_stride, sel, sel_stride, sinks, scale, out);
        check("attention wmma");
        return;
    }
#endif
#if defined(__HIP_PLATFORM_AMD__)
    static const bool wmma12 = std::getenv("STRATA_DS41_ATTN_WMMA") && std::string(std::getenv("STRATA_DS41_ATTN_WMMA")) == "1";
    if (wmma12 && !v1 && dim == wm12::kDim && heads % wm12::kHeads == 0 && ds41_device_is_gfx12()) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(attn_wmma_gfx12_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(wm12::kLds));
            cudaFuncSetAttribute(attn_wmma_gfx12_kernel<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(wm12::kLds));
            configured = true;
        }
        if (comp_half)
            attn_wmma_gfx12_kernel<uint16_t><<<dim3(unsigned(tokens), unsigned(heads / wm12::kHeads)), 256, wm12::kLds, (cudaStream_t)stream>>>(
                q, heads, window, ring, reinterpret_cast<const int4*>(info), reinterpret_cast<const uint16_t*>(comp), comp_stride, sel,
                sel_stride, sinks, scale, out);
        else
            attn_wmma_gfx12_kernel<float><<<dim3(unsigned(tokens), unsigned(heads / wm12::kHeads)), 256, wm12::kLds, (cudaStream_t)stream>>>(
                q, heads, window, ring, reinterpret_cast<const int4*>(info), comp, comp_stride, sel, sel_stride, sinks, scale, out);
        check("attention wmma gfx12");
        return;
    }
#endif
    if (!v1 && dim == kTileDim && heads % kTG == 0) {
        const size_t shared = size_t(kTG * kTileDim + kTR * kDCP + kTR * kTG) * sizeof(float);
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(attn_tiled_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared));
            cudaFuncSetAttribute(attn_tiled_kernel<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared));
            configured = true;
        }
        if (comp_half)
            attn_tiled_kernel<uint16_t><<<dim3(unsigned(tokens), unsigned(heads / kTG)), 256, shared, (cudaStream_t)stream>>>(
                q, heads, window, ring, reinterpret_cast<const int4*>(info), reinterpret_cast<const uint16_t*>(comp), comp_stride, sel,
                sel_stride, sinks, scale, out);
        else
            attn_tiled_kernel<float><<<dim3(unsigned(tokens), unsigned(heads / kTG)), 256, shared, (cudaStream_t)stream>>>(
                q, heads, window, ring, reinterpret_cast<const int4*>(info), comp, comp_stride, sel, sel_stride, sinks, scale, out);
        check("attention tiled");
        return;
    }
    if (heads % kHeadGroup || dim > 512 || dim % 256) throw std::invalid_argument("ds41_attention_batched: shape");
    const size_t shared = size_t(kHeadGroup * dim + kTile * dim + kHeadGroup * kTile) * sizeof(float);
    attn_batched_kernel<<<dim3(unsigned(tokens), unsigned(heads / kHeadGroup)), 256, shared, (cudaStream_t)stream>>>(
        q, heads, dim, window, ring, reinterpret_cast<const int4*>(info), comp, comp_stride, sel, sel_stride, sinks, scale, out);
    check("attention batched");
}


size_t ds41_attention_scratch(int heads) { return size_t(heads) * kMaxRows * sizeof(float); }

void ds41_index_score(const float* q, int heads, int dim, const float* keys, int key_stride, int rows, const float* w,
                      float w_scale, float* score, void* stream, bool half) {
    if (rows < 1) return;
    if (half)
        index_score_kernel<uint16_t><<<blocks(rows, 8), 256, 0, (cudaStream_t)stream>>>(q, heads, dim, reinterpret_cast<const uint16_t*>(keys),
                                                                                        key_stride, rows, w, w_scale, score);
    else
        index_score_kernel<float><<<blocks(rows, 8), 256, 0, (cudaStream_t)stream>>>(q, heads, dim, keys, key_stride, rows, w, w_scale, score);
    check("index_score");
}

void ds41_index_score_batched(const float* q, int heads, int dim, const float* keys, int key_stride, int pos0, int ratio, int tokens,
                              const float* w, float w_scale, float* score, int score_stride, void* stream, bool half) {
    const int rows = (pos0 + tokens) / ratio;
    if (rows < 1 || tokens < 1) return;
    if (half)
        index_score_batched_kernel<uint16_t><<<dim3(unsigned(blocks(rows, 8)), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(
            q, heads, dim, reinterpret_cast<const uint16_t*>(keys), key_stride, pos0, ratio, w, w_scale, score, score_stride, nullptr, 0);
    else
        index_score_batched_kernel<float><<<dim3(unsigned(blocks(rows, 8)), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(
            q, heads, dim, keys, key_stride, pos0, ratio, w, w_scale, score, score_stride, nullptr, 0);
    check("index_score_batched");
}

void ds41_index_topk(const float* score, int score_stride, int tokens, int pos0, int ratio, int k, int* sel, int* count, int mode,
                     uint8_t* cand, int cand_stride, int block, int cand_k, void* stream, const int* tpos) {
    if (tokens < 1) return;
    static const bool two_stage = std::getenv("STRATA_DS41_TOPK_2STAGE") && std::string(std::getenv("STRATA_DS41_TOPK_2STAGE")) == "1";
    const int max_visible = (tpos ? score_stride : pos0 + tokens) / std::max(ratio, 1);
    if (two_stage && mode != 1 && max_visible > 2 * kChunk2) {
        const int nchunks = (max_visible + kChunk2 - 1) / kChunk2;
        if (nchunks <= 1024) {
            static void* buf = nullptr;
            static size_t buf_bytes = 0;
            const size_t need = size_t(tokens) * nchunks * (size_t(k) + 1) * 4;
            if (need > buf_bytes) {
                if (buf) cudaFree(buf);
                if (cudaMalloc(&buf, need * 2) != cudaSuccess) throw std::runtime_error("ds41_index_topk: two-stage buffer");
                buf_bytes = need * 2;
            }
            int* cbuf = static_cast<int*>(buf);
            int* ccnt = cbuf + size_t(tokens) * nchunks * k;
            index_topk2_stage1<<<dim3(unsigned(nchunks), unsigned(tokens)), kTopkThreads, 0, (cudaStream_t)stream>>>(
                score, score_stride, pos0, ratio, k, mode, cand, cand_stride, block, tpos, nchunks, cbuf, ccnt);
            check("index_topk stage 1");
            index_topk2_stage2<<<unsigned(tokens), kTopkThreads, 0, (cudaStream_t)stream>>>(score, score_stride, pos0, ratio, k, tpos, nchunks, cbuf,
                                                                                         ccnt, sel, count);
            check("index_topk stage 2");
            return;
        }
    }
    // STRATA_DS41_TOPK_CHUNKED=1: select_sorted_chunked (two block scans per selection; the same picks)
    static const bool chunked = std::getenv("STRATA_DS41_TOPK_CHUNKED") && std::string(std::getenv("STRATA_DS41_TOPK_CHUNKED")) == "1";
    const int nb = mode == 1 ? (score_stride + block - 1) / block : 0;
    const size_t smem = mode == 1 ? size_t(nb + cand_k) * 4 : 0;
    if (smem > 48 * 1024) {   // mode 1's block maxima scale with the context: raise the kernel's limit as far as the device goes
        int dev = 0, opt = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (smem > size_t(opt)) throw std::invalid_argument("ds41_index_topk: candidate-block maxima exceed shared memory (context too long for mode 1)");
        cudaFuncSetAttribute(index_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    }
    index_topk_kernel<<<unsigned(tokens), kTopkThreads, smem, (cudaStream_t)stream>>>(score, score_stride, pos0, ratio, k, sel, count, mode,
                                                                                     cand, cand_stride, block, cand_k, tpos, chunked ? 1 : 0);
    check("index_topk");
}

void ds41_patch_info(int* info, const int* count, int tokens, int pos0, int ratio, void* stream) {
    if (tokens < 1) return;
    patch_info_kernel<<<blocks(tokens, 256), 256, 0, (cudaStream_t)stream>>>(info, count, tokens, pos0, ratio);
    check("patch_info");
}

void ds41_compress2_rows(const float* v, const float* s, float* out, int rows, int n, void* stream) {
    if (rows < 1) return;
    compress2_rows_kernel<<<dim3(blocks(n, 256), unsigned(rows)), 256, 0, (cudaStream_t)stream>>>(v, s, out, n);
    check("compress2_rows");
}

void ds41_copy_rows(uint8_t* dst, size_t dpitch, size_t dstep, const uint8_t* src, size_t spitch, size_t sstep, int width, int rows,
                    int count, void* stream) {
    if (count < 1 || rows < 1 || width < 1) return;
    copy_rows_kernel<<<dim3(unsigned(std::min(rows, 256)), unsigned(count)), 128, 0, (cudaStream_t)stream>>>(dst, dpitch, dstep, src, spitch,
                                                                                                         sstep, width, rows);
    check("copy_rows");
}

void ds41_compress2(const float* v0, const float* s0, const float* v1, const float* s1, float* out, int n, void* stream) {
    compress2_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t)stream>>>(v0, s0, v1, s1, out, n);
    check("compress2");
}

void ds41_engram(float* h, const float* kv, const float* g, int n, float eps, void* stream, int tokens) {
    engram_kernel<<<dim3(4, unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(h, kv, g, n, eps);
    check("engram");
}

void ds41_axpy(const float* x, float a, float* y, int n, void* stream) {
    axpy_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t)stream>>>(x, a, y, n);
    check("axpy");
}

void ds41_add(const float* a, const float* b, float* y, int n, void* stream) {
    add_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t)stream>>>(a, b, y, n);
    check("add");
}

void ds41_f32_gemv(const float* w, const float* x, float* y, int rows, int in, void* stream, int tokens) {
    f32_gemv_kernel<<<dim3(unsigned(rows), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(w, x, y, rows, in);
    check("f32_gemv");
}

void ds41_gather(const float* src, const int* idx, int m, int n, float* dst, void* stream) {
    if (m < 1) return;
    gather_kernel<<<dim3(blocks(n, 256), unsigned(m)), 256, 0, (cudaStream_t)stream>>>(src, idx, n, dst);
    check("gather");
}

void ds41_scatter_axpy(const float* src, const int* idx, const float* w, int m, int n, float* dst, void* stream) {
    if (m < 1) return;
    scatter_axpy_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t)stream>>>(src, idx, w, m, n, dst);
    check("scatter_axpy");
}

void ds41_to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    to_bf16_kernel<<<unsigned((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, y, n);
    check("to_bf16");
}

void ds41_bf16_gemv(const uint16_t* w, const float* x, float* y, int rows, int in, void* stream, int tokens) {
    bf16_gemv_kernel<<<dim3(unsigned(rows), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(w, x, y, rows, in);
    check("bf16_gemv");
}

void ds41_bf16_gemv_mt(const uint16_t* w, const float* x, int ldx, const int* rows, int n, float* y, int ldy, int out, int in,
                       void* stream) {
    if (n < 1 || n > kMtMax || in % 64) throw std::invalid_argument("ds41_bf16_gemv_mt: shape");
    MtRows r{n, {}};
    for (int t = 0; t < n; ++t) r.row[t] = rows[t];
    bf16_gemv_mt_kernel<<<unsigned((out + 7) / 8), 256, 0, (cudaStream_t)stream>>>(w, x, ldx, r, y, ldy, out, in, 0, 0, 0);
    check("bf16_gemv_mt");
}
void ds41_bf16_gemv_mt_groups(const uint16_t* w, size_t w_gstride, const float* x, int ldx, int x_goff, const int* rows, int n, float* y, int ldy,
                              int y_goff, int out, int in, int groups, void* stream) {
    if (n < 1 || n > kMtMax || in % 64 || groups < 1) throw std::invalid_argument("ds41_bf16_gemv_mt_groups: shape");
    MtRows r{n, {}};
    for (int t = 0; t < n; ++t) r.row[t] = rows[t];
    bf16_gemv_mt_kernel<<<dim3(unsigned((out + 7) / 8), unsigned(groups)), 256, 0, (cudaStream_t)stream>>>(w, x, ldx, r, y, ldy, out, in, w_gstride,
                                                                                                          x_goff, y_goff);
    check("bf16_gemv_mt_groups");
}
// rows [t][width] of src into ring slot (pos + t) % ring of dst
__global__ void ring_rows_kernel(const float* src, int width, int pos, int ring, float* dst) {
    const int t = blockIdx.y;
    float* d = dst + size_t((pos + t) % ring) * width;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x) d[i] = src[size_t(t) * width + i];
}
void ds41_ring_rows(const float* src, int rows, int width, int pos, int ring, float* dst, void* stream) {
    if (rows < 1) return;
    ring_rows_kernel<<<dim3(std::min(unsigned(blocks(width, 256)), 4u), unsigned(rows)), 256, 0, (cudaStream_t)stream>>>(src, width, pos, ring, dst);
    check("ring_rows");
}

void ds41_mxfp4_gemv_mt(const uint8_t* w, size_t row_bytes, const float* x, int ldx, const int* rows, int n, float* y, int ldy,
                        int out, int in, void* stream) {
    if (n < 1 || n > kMtMax || in % 32) throw std::invalid_argument("ds41_mxfp4_gemv_mt: shape");
    MtRows r{n, {}};
    for (int t = 0; t < n; ++t) r.row[t] = rows[t];
    mxfp4_gemv_mt_kernel<<<unsigned((out + 7) / 8), 256, 0, (cudaStream_t)stream>>>(w, row_bytes, x, ldx, r, y, ldy, out, in);
    check("mxfp4_gemv_mt");
}

void ds41_stream_mean(const float* h, float* dst, size_t ld_dst, int n, int tokens, void* stream) {
    stream_mean_kernel<<<dim3(unsigned((n + 255) / 256), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(h, dst, ld_dst, n, tokens);
    check("stream_mean");
}

void ds41_attention_multi(const float* q, int tokens, int heads, int dim, const float* window, int ring, const int* info, int bound,
                          const float* comp, int comp_stride, const int* sel, int sel_stride, const int* nsel_dev, const float* sinks,
                          float scale, float* scores, float* out, void* stream, size_t wss, size_t css) {
    if (bound > kMaxRows || bound < 1 || heads % kHeadGroup || dim % kDimChunk || tokens < 1) throw std::invalid_argument("ds41_attention_multi: shape");
    const dim3 g1(unsigned(heads / kHeadGroup), unsigned((bound + kRowChunk - 1) / kRowChunk), unsigned(tokens));
    attn_scores_multi_kernel<<<g1, 256, size_t(kHeadGroup) * dim * sizeof(float), (cudaStream_t)stream>>>(
        q, heads, dim, window, ring, reinterpret_cast<const int4*>(info), comp, comp_stride, sel, sel_stride, nsel_dev, scale, scores, wss, css);
    check("attention multi scores");
    static const int split = std::getenv("STRATA_DS41_ATTN_SPLIT") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_ATTN_SPLIT"))) : 1;
    if (split > 1 && bound >= 2 * split) {
        float* part = scores + size_t(tokens) * heads * kMaxRows;   // the caller's scratch: tokens' scores, then the parts
        const int chunk = (bound + split - 1) / split;
        const dim3 g3(unsigned(heads / kHeadGroup), unsigned(dim / kDimChunk), unsigned(tokens * split));
        attn_out_multi_split_kernel<<<g3, 256, size_t(kHeadGroup) * chunk * sizeof(float), (cudaStream_t)stream>>>(
            scores, heads, dim, window, ring, reinterpret_cast<const int4*>(info), comp, comp_stride, sel, sel_stride, nsel_dev, bound, split, sinks, part, wss, css);
        check("attention multi out split");
        attn_combine_multi_kernel<<<dim3(unsigned((heads * dim + 255) / 256), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(part, split, heads * dim, out);
        check("attention multi combine");
        return;
    }
    const dim3 g2(unsigned(heads / kHeadGroup), unsigned(dim / kDimChunk), unsigned(tokens));
    attn_out_multi_kernel<<<g2, 256, size_t(kHeadGroup) * bound * sizeof(float), (cudaStream_t)stream>>>(
        scores, heads, dim, window, ring, reinterpret_cast<const int4*>(info), comp, comp_stride, sel, sel_stride, nsel_dev, bound, sinks, out, wss, css);
    check("attention multi out");
}

void ds41_swap_blocks(const float* src, float* dst, int na, int nb, int width, void* stream) {
    swap_blocks_kernel<<<dim3(unsigned((width + 255) / 256 < 8 ? (width + 255) / 256 : 8), unsigned(na), unsigned(nb)), 256, 0,
                         (cudaStream_t)stream>>>(src, dst, na, nb, width);
    check("swap_blocks");
}

void ds41_moe_route(const float* logits, const float* bias, int tokens, int experts, int k, float scale, bool normalize, int* ids, float* w,
                    void* stream) {
    if (k > 16) throw std::invalid_argument("ds41_moe_route: k");
    moe_route_kernel<<<unsigned(tokens), 128, size_t(2 * experts) * sizeof(float), (cudaStream_t)stream>>>(logits, bias, experts, k, scale,
                                                                                                       normalize ? 1 : 0, ids, w);
    check("moe_route");
}

void ds41_mxfp4_moe(const uint8_t* w, size_t expert_bytes, size_t row_bytes, const int* ids, int pairs, const float* x, int ldx, int div,
                    float* y, int ldy, int out, int in, void* stream) {
    if (in % 32) throw std::invalid_argument("ds41_mxfp4_moe: shape");
    mxfp4_moe_kernel<<<dim3(unsigned((out + 7) / 8), unsigned(pairs)), 256, 0, (cudaStream_t)stream>>>(w, expert_bytes, row_bytes, ids, x, ldx,
                                                                                                    div, y, ldy, out, in);
    check("mxfp4_moe");
}

void ds41_markov_row(const uint16_t* w1, int dim, const int* tok, int i, float* row, void* stream) {
    markov_row_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(w1, dim, tok, i, row);
    check("markov_row");
}
void ds41_argmax_into(const float* v, int n, int* tok, int i, void* stream) {
    argmax_kernel<<<1, 1024, 0, (cudaStream_t)stream>>>(v, n, tok, i);
    check("argmax");
}
void ds41_confidence(const float* cw, const float* x, int hidden, const float* row, int dim, float* conf, int i, void* stream) {
    conf_kernel<<<1, 512, 0, (cudaStream_t)stream>>>(cw, x, hidden, row, dim, conf, i);
    check("confidence");
}

void ds41_gather_sel_h(const uint16_t* cache, int rw, const int* sel, int topk, int tokens, int pos0, int ratio, float* out, void* stream,
                       const int* tpos, size_t css) {
    if (tokens < 1) return;
    gather_sel_h_kernel<<<dim3(unsigned(topk), unsigned(tokens)), 128, 0, (cudaStream_t)stream>>>(cache, rw, sel, topk, pos0, ratio, out, tpos, css);
    check("gather_sel_h");
}

void ds41_index_score_q8(const float* q, int heads, int dim, const uint8_t* cache, size_t row_bytes, int key_off, int pos0, int ratio, int tokens,
                         int fixed_rows, const float* w, float w_scale, float* score, int score_stride, void* stream,
                         const int* tpos, size_t css) {
    const int rows = fixed_rows > 0 ? fixed_rows : (pos0 + tokens) / ratio;
    if (rows < 1 || tokens < 1 || dim > 128) {
        if (dim > 128) throw std::invalid_argument("ds41_index_score_q8: dim");
        return;
    }
    index_score_q8_kernel<<<dim3(unsigned(blocks(rows, 8)), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(
        q, heads, dim, cache, row_bytes, key_off, pos0, ratio, w, w_scale, score, score_stride, fixed_rows, tpos, css);
    check("index_score_q8");
}
void ds41_gather_sel_q8(const uint8_t* cache, size_t row_bytes, int rw, const int* sel, int topk, int tokens, int pos0, int ratio, float* out,
                        void* stream, const int* tpos, size_t css) {
    if (tokens < 1) return;
    gather_sel_q8_kernel<<<dim3(unsigned(topk), unsigned(tokens)), 128, 0, (cudaStream_t)stream>>>(cache, row_bytes, rw, sel, topk, pos0, ratio, out,
                                                                                                  tpos, css);
    check("gather_sel_q8");
}
void ds41_quant_rows_q8(const float* x, int rows, int rw, uint8_t* out, size_t row_bytes, void* stream) {
    if (rows < 1) return;
    if (rw % 32 || rw > 1024) throw std::invalid_argument("ds41_quant_rows_q8: width");
    quant_rows_q8_kernel<<<unsigned(rows), unsigned(rw), 0, (cudaStream_t)stream>>>(x, rw, out, row_bytes);
    check("quant_rows_q8");
}

void ds41_keys_q8_f16(const uint8_t* cache, size_t row_bytes, int key_off, int dim, int row0, int rows, uint16_t* out, void* stream) {
    if (rows < 1) return;
    keys_q8_f16_kernel<<<unsigned(rows), 128, 0, (cudaStream_t)stream>>>(cache, row_bytes, key_off, dim, row0, out);
    check("keys_q8_f16");
}
void ds41_index_reduce(const float* c, int rows, int tokens, int heads, const float* w, float w_scale, int pos0, int ratio, int row0,
                       float* score, int score_stride, void* stream) {
    if (rows < 1 || tokens < 1) return;
    index_reduce_kernel<<<dim3(unsigned(blocks(rows, 256)), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(c, rows, heads, w, w_scale, pos0, ratio,
                                                                                                          row0, score, score_stride);
    check("index_reduce");
}

void ds41_index_score_sessions(const float* q, int heads, int dim, const float* keys, int key_stride, size_t kss, const int* tpos, int max_rows,
                               int ratio, int tokens, const float* w, float w_scale, float* score, int score_stride, void* stream, bool half) {
    if (max_rows < 1 || tokens < 1) return;
    if (half) {
        index_score_batched_kernel<uint16_t><<<dim3(unsigned(blocks(max_rows, 8)), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(
            q, heads, dim, reinterpret_cast<const uint16_t*>(keys), key_stride, 0, ratio, w, w_scale, score, score_stride, tpos, kss);
        check("index_score_sessions");
        return;
    }
    index_score_batched_kernel<float><<<dim3(unsigned(blocks(max_rows, 8)), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(
        q, heads, dim, keys, key_stride, 0, ratio, w, w_scale, score, score_stride, tpos, kss);
    check("index_score_sessions");
}
void ds41_compress_sessions(const float* v, const float* s, float* stash_v, float* stash_s, int n, const int* tpos, int ratio, int tokens,
                            float* out, void* stream) {
    if (tokens < 1) return;
    compress_sessions_kernel<<<dim3(blocks(n, 256), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(v, s, stash_v, stash_s, n, tpos, ratio, out);
    check("compress_sessions");
}
void ds41_scatter_bytes(const void* src, int rows, size_t row_bytes, const int* dst_row, void* dst, size_t dst_pitch, void* stream) {
    if (rows < 1) return;
    if (row_bytes % 4 || dst_pitch % 4) throw std::invalid_argument("ds41_scatter_bytes: alignment");
    const int words = int(row_bytes / 4);
    scatter_words_kernel<<<dim3(std::min(unsigned(blocks(words, 256)), 4u), unsigned(rows)), 256, 0, (cudaStream_t)stream>>>(
        static_cast<const uint32_t*>(src), words, dst_row, static_cast<uint32_t*>(dst), dst_pitch / 4);
    check("scatter_bytes");
}
void ds41_scatter_rows(const float* src, int rows, int width, const int* dst_row, float* dst, size_t dst_stride, void* stream) {
    if (rows < 1) return;
    scatter_rows_kernel<<<dim3(std::min(unsigned(blocks(width, 256)), 4u), unsigned(rows)), 256, 0, (cudaStream_t)stream>>>(src, width, dst_row, dst,
                                                                                                                         dst_stride);
    check("scatter_rows");
}

bool ds41_index_score_fast(int kind, const float* q, int heads, int dim, const void* cache, size_t row_bytes, int key_off, int pos0, int ratio,
                           int tokens, int max_rows, int fixed_rows, const float* w, float w_scale, float* score, int score_stride, void* stream,
                           const int* tpos, size_t css) {
    if (heads > 32 || dim % 32 || heads * dim > 4096 || max_rows < 1 || tokens < 1) return false;
    const dim3 grid(unsigned((max_rows + 255) / 256), unsigned(tokens));
    const auto* c = static_cast<const uint8_t*>(cache);
    auto* st = (cudaStream_t)stream;
    if (kind == 2) index_score_fast_kernel<2><<<grid, 256, 0, st>>>(q, heads, dim, c, row_bytes, key_off, pos0, ratio, w, w_scale, score, score_stride, fixed_rows, tpos, css);
    else if (kind == 1) index_score_fast_kernel<1><<<grid, 256, 0, st>>>(q, heads, dim, c, row_bytes, key_off, pos0, ratio, w, w_scale, score, score_stride, fixed_rows, tpos, css);
    else index_score_fast_kernel<0><<<grid, 256, 0, st>>>(q, heads, dim, c, row_bytes, key_off, pos0, ratio, w, w_scale, score, score_stride, fixed_rows, tpos, css);
    check("index_score_fast");
    return true;
}

}  // namespace strata::kernels
