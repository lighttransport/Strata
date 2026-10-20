// src/kernels/cuda/ds41.cu - DeepSeek-V4.1-Flash decode kernels (strata/kernels/ds41.hpp).
// Equations: llama.cpp PR #30817 (src/models/deepseek4.cpp build_hc_mixes / build_hc_sinkhorn / build_hc_post,
// deepseek41.cpp build_attention41 / build_engram).
#include "strata/kernels/ds41.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
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
__global__ void hc_coeff_kernel(const float* m, const float* scale, const float* base, float* c, int iterations, float eps) {
    m += size_t(blockIdx.x) * 24;
    c += size_t(blockIdx.x) * 24;
    for (int i = 0; i < 4; ++i) c[i] = sigmoid(m[i] * scale[0] + base[i]) + eps;
    for (int i = 0; i < 4; ++i) c[4 + i] = sigmoid(m[4 + i] * scale[1] + base[4 + i]) * 2.0f;
    float k[16];
    for (int i = 0; i < 16; ++i) k[i] = m[8 + i] * scale[2] + base[8 + i];
    // comb[dst + 4 src]: softmax over dst for each src
    for (int s = 0; s < 4; ++s) {
        float mx = -INFINITY;
        for (int d = 0; d < 4; ++d) mx = fmaxf(mx, k[d + 4 * s]);
        float sum = 0;
        for (int d = 0; d < 4; ++d) { k[d + 4 * s] = expf(k[d + 4 * s] - mx); sum += k[d + 4 * s]; }
        const float inv = 1.0f / sum;
        for (int d = 0; d < 4; ++d) k[d + 4 * s] = k[d + 4 * s] * inv + eps;
    }
    auto cols = [&] {   // for each dst: divide by (sum over src + eps)
        for (int d = 0; d < 4; ++d) {
            float sum = 0;
            for (int s = 0; s < 4; ++s) sum += k[d + 4 * s];
            sum += eps;
            for (int s = 0; s < 4; ++s) k[d + 4 * s] /= sum;
        }
    };
    auto rows = [&] {   // for each src: divide by (sum over dst + eps)
        for (int s = 0; s < 4; ++s) {
            float sum = 0;
            for (int d = 0; d < 4; ++d) sum += k[d + 4 * s];
            sum += eps;
            for (int d = 0; d < 4; ++d) k[d + 4 * s] /= sum;
        }
    };
    cols();
    for (int i = 1; i < iterations; ++i) { rows(); cols(); }
    for (int i = 0; i < 16; ++i) c[8 + i] = k[i];
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
        const float* comp, int comp_stride, const int* sel, int rows, float scale, float* scores) {
    extern __shared__ float qs[];   // kHeadGroup x dim
    const int h0 = blockIdx.x * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    for (int i = tid; i < kHeadGroup * dim; i += 256) qs[i] = q[size_t(h0) * dim + i];
    __syncthreads();
    for (int r = blockIdx.y * kRowChunk + warp; r < min(rows, (blockIdx.y + 1) * kRowChunk); r += 8) {
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
        const float* comp, int comp_stride, const int* sel, int rows, const float* sinks, float* out) {
    extern __shared__ float p[];    // kHeadGroup x rows
    __shared__ float inv[kHeadGroup];
    const int h0 = blockIdx.x * kHeadGroup, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    // warp h: head h0 + h's max, exponentials and sum
    {
        const int h = warp;   // 8 warps, kHeadGroup heads
        const float* sc = scores + size_t(h0 + h) * kMaxRows;
        float mx = sinks[h0 + h];
        for (int r = lane; r < rows; r += 32) mx = fmaxf(mx, sc[r]);
        for (int d = 16; d; d /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
        float sum = 0;
        for (int r = lane; r < rows; r += 32) { const float e = expf(sc[r] - mx); p[h * rows + r] = e; sum += e; }
        sum = warp_sum(sum);
        if (!lane) inv[h] = 1.0f / (sum + expf(sinks[h0 + h] - mx));
    }
    __syncthreads();
    const int d = blockIdx.y * kDimChunk + tid % kDimChunk, hs = tid / kDimChunk;   // 2 head subsets of 4
    if (d >= dim) return;
    float acc[kHeadGroup / 2];
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] = 0;
    for (int r = 0; r < rows; ++r) {
        const float v = attn_row(window, ring, wrows, comp, comp_stride, sel, dim, r)[d];
#pragma unroll
        for (int j = 0; j < kHeadGroup / 2; ++j) acc[j] += p[(hs + 2 * j) * rows + r] * v;
    }
#pragma unroll
    for (int j = 0; j < kHeadGroup / 2; ++j) out[size_t(h0 + hs + 2 * j) * dim + d] = acc[j] * inv[hs + 2 * j];
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
__global__ void __launch_bounds__(256) attn_tiled_kernel(const float* q, int heads, const float* window, int ring, const int4* info,
        const float* comp, int comp_stride, const int* sel, int sel_stride, const float* sinks, float scale, float* out) {
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
            if (rr < rows)
                v = *reinterpret_cast<const float4*>(attn_tile_row(window, ring, first, wrows, comp, comp_stride, tsel, rr) + c0 + 4 * c4);
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

// one warp per key row
__global__ void index_score_kernel(const float* q, int heads, int dim, const float* keys, int key_stride, int rows,
                                   const float* w, float w_scale, float* score) {
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const float* k = keys + size_t(r) * key_stride;
    float total = 0;
    for (int h = 0; h < heads; ++h) {
        float s = 0;
        for (int i = lane; i < dim; i += 32) s += k[i] * q[size_t(h) * dim + i];
        s = warp_sum(s);
        s = __shfl_sync(0xffffffff, s, 0);
        total += fmaxf(s, 0.0f) * (w[h] * w_scale);
    }
    if (!lane) score[r] = total;
}

// index scores of a pass: token t (blockIdx.y) scores its (pos0 + t + 1) / ratio visible rows
__global__ void index_score_batched_kernel(const float* q, int heads, int dim, const float* keys, int key_stride, int pos0, int ratio,
                                           const float* w, float w_scale, float* score, int score_stride) {
    const int t = blockIdx.y, rows = (pos0 + t + 1) / ratio;
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    q += size_t(t) * heads * dim;
    w += size_t(t) * heads;
    const float* k = keys + size_t(r) * key_stride;
    float total = 0;
    for (int h = 0; h < heads; ++h) {
        float s = 0;
        for (int i = lane; i < dim; i += 32) s += k[i] * q[size_t(h) * dim + i];
        s = warp_sum(s);
        s = __shfl_sync(0xffffffff, s, 0);
        total += fmaxf(s, 0.0f) * (w[h] * w_scale);
    }
    if (!lane) score[size_t(t) * score_stride + r] = total;
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
__device__ void radix_threshold(Get get, int n, int k, uint32_t* hist, uint32_t* shared2, uint32_t& thr, int& take) {
    uint32_t prefix = 0, mask = 0;
    int remaining = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t key = get(i);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
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
                                  uint8_t* cand, int cand_stride, int bs, int cand_k) {
    __shared__ uint32_t hist[256], shared2[2];
    __shared__ int scan[kTopkThreads];
    extern __shared__ float best[];   // mode 1: [blocks] maxima, then [cand_k] picks (as ints)
    const int t = blockIdx.x, visible = (pos0 + t + 1) / ratio;
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
        const int n = select_sorted([&](int i) { return order_key(best[i]); }, [](int) { return true; }, nb, cand_k, hist, shared2,
                                    scan, picks);
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) flags[picks[j]] = 1;
        __syncthreads();
    }
    const bool masked = mode == 2;
    const uint32_t ninf = order_key(-INFINITY);
    auto get = [&](int i) { return masked && !keep_blocks[i / bs] ? ninf : order_key(s[i]); };
    auto keep = [&](int i) { return !(masked && !keep_blocks[i / bs]) && !(isinf(s[i]) && s[i] < 0); };
    const int n = select_sorted(get, keep, visible, k, hist, shared2, scan, out);
    if (threadIdx.x == 0) count[t] = n;
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

}  // namespace

void ds41_hc_coeff(const float* mixes, const float* scale, const float* base, float* coeff, int iterations, float eps,
                   void* stream, int tokens) {
    hc_coeff_kernel<<<unsigned(tokens), 1, 0, (cudaStream_t)stream>>>(mixes, scale, base, coeff, iterations, eps);
    check("hc_coeff");
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
                    float scale, float* scores, float* out, void* stream) {
    const Ring ring{window_first, window_ring};
    const int rows = window_rows + nsel;
    if (rows > kMaxRows || rows < 1 || heads % kHeadGroup || dim % kDimChunk) throw std::invalid_argument("ds41_attention: shape");
    const dim3 g1(unsigned(heads / kHeadGroup), unsigned((rows + kRowChunk - 1) / kRowChunk));
    attn_scores_kernel<<<g1, 256, size_t(kHeadGroup) * dim * sizeof(float), (cudaStream_t)stream>>>(
        q, dim, window, ring, window_rows, comp, comp_stride, sel, rows, scale, scores);
    check("attention scores");
    const dim3 g2(unsigned(heads / kHeadGroup), unsigned(dim / kDimChunk));
    attn_out_kernel<<<g2, 256, size_t(kHeadGroup) * rows * sizeof(float), (cudaStream_t)stream>>>(
        scores, dim, window, ring, window_rows, comp, comp_stride, sel, rows, sinks, out);
    check("attention out");
}

void ds41_attention_batched(const float* q, int tokens, int heads, int dim, const float* window, int ring, const int* info,
                            const float* comp, int comp_stride, const int* sel, int sel_stride, const float* sinks, float scale,
                            float* out, void* stream) {
    static const bool v1 = std::getenv("STRATA_DS41_ATTN_V1") != nullptr;
    if (!v1 && dim == kTileDim && heads % kTG == 0) {
        const size_t shared = size_t(kTG * kTileDim + kTR * kDCP + kTR * kTG) * sizeof(float);
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(attn_tiled_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared));
            configured = true;
        }
        attn_tiled_kernel<<<dim3(unsigned(tokens), unsigned(heads / kTG)), 256, shared, (cudaStream_t)stream>>>(
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
                      float w_scale, float* score, void* stream) {
    if (rows < 1) return;
    index_score_kernel<<<blocks(rows, 8), 256, 0, (cudaStream_t)stream>>>(q, heads, dim, keys, key_stride, rows, w, w_scale,
                                                                          score);
    check("index_score");
}

void ds41_index_score_batched(const float* q, int heads, int dim, const float* keys, int key_stride, int pos0, int ratio, int tokens,
                              const float* w, float w_scale, float* score, int score_stride, void* stream) {
    const int rows = (pos0 + tokens) / ratio;
    if (rows < 1 || tokens < 1) return;
    index_score_batched_kernel<<<dim3(unsigned(blocks(rows, 8)), unsigned(tokens)), 256, 0, (cudaStream_t)stream>>>(
        q, heads, dim, keys, key_stride, pos0, ratio, w, w_scale, score, score_stride);
    check("index_score_batched");
}

void ds41_index_topk(const float* score, int score_stride, int tokens, int pos0, int ratio, int k, int* sel, int* count, int mode,
                     uint8_t* cand, int cand_stride, int block, int cand_k, void* stream) {
    if (tokens < 1) return;
    const int nb = mode == 1 ? (score_stride + block - 1) / block : 0;
    const size_t smem = mode == 1 ? size_t(nb + cand_k) * 4 : 0;
    index_topk_kernel<<<unsigned(tokens), kTopkThreads, smem, (cudaStream_t)stream>>>(score, score_stride, pos0, ratio, k, sel, count, mode,
                                                                                     cand, cand_stride, block, cand_k);
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

}  // namespace strata::kernels
