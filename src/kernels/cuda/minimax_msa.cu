// Initial MiniMax MSA score and block-selection kernels; the contract is in minimax_msa.hpp.
#include "strata/kernels/minimax_msa.hpp"

#include <cuda_fp16.h>
#if !defined(__HIP_PLATFORM_AMD__)
#include <mma.h>
#endif
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxTopK = 64;
constexpr int kMaxAttentionSlots = 4096;

__global__ void block_scores_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                   const int32_t* __restrict__ query_positions,
                                   const int32_t* __restrict__ key_positions, int queries,
                                   int index_heads, int dim, int keys, int block_size,
                                   int n_blocks, size_t score_stride, float* __restrict__ scores) {
    __shared__ float partial[kThreads];
    const int block = (int)blockIdx.x;
    const int head = (int)blockIdx.y;
    const int query = (int)blockIdx.z;
    if (block >= n_blocks || head >= index_heads || query >= queries) return;

    const int tid = (int)threadIdx.x;
    const int32_t qpos = query_positions[query];
    const float* qh = q + ((size_t)query * index_heads + head) * dim;
    float best = -INFINITY;
    for (int slot = block * block_size + tid; slot < (block + 1) * block_size && slot < keys;
         slot += kThreads) {
        const int32_t pos = key_positions[slot];
        if (pos < 0 || pos > qpos || pos / block_size != block) continue;
        const float* kh = k + (size_t)slot * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) dot = fmaf(qh[d], kh[d], dot);
        best = fmaxf(best, dot);
    }
    partial[tid] = best;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) {
        const size_t row = ((size_t)query * index_heads + head) * score_stride;
        scores[row + block] = partial[0];
    }
}

// ---- dim == 128 (MiniMax M3.1): one warp per key slot, each lane holds a float4 of the 128-wide vector, so the loads
// are coalesced and a dot product is a warp reduction.  The generic kernels below stay for other widths and as the
// STRATA_MSA_SLOW=1 A/B path.  Same arithmetic up to float summation order.
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}


// K / V rows of 128 elements as four floats per lane, from a float or a half cache (halves load 8 bytes per lane)
__device__ __forceinline__ float4 load4(const float* p, int lane) { return reinterpret_cast<const float4*>(p)[lane]; }
__device__ __forceinline__ float4 load4(const __half* p, int lane) {
    const uint2 u = reinterpret_cast<const uint2*>(p)[lane];
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

#if !defined(__HIP_PLATFORM_AMD__)
/// Prefill block scores with tensor cores (FP16 index keys, 4 index heads, 128-key blocks): a 16-row tile is 4 queries x
/// 4 heads, a warp takes one key block; the score is the masked maximum of q.k over the block, as in the scalar kernel.
__global__ void __launch_bounds__(128)
block_scores_tc_kernel(const float* __restrict__ q, const __half* __restrict__ k, const int32_t* __restrict__ query_positions,
                       const int32_t* __restrict__ key_positions, int queries, int keys, int n_blocks, size_t score_stride,
                       float* __restrict__ scores) {
    using namespace nvcuda;
    __shared__ __half qs[16 * 128];
    __shared__ float tile[4][16 * 16];
    const int qg = (int)blockIdx.y, warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    for (int i = (int)threadIdx.x; i < 2048; i += 128) {
        const int r = i >> 7, d = i & 127, qu = qg * 4 + (r >> 2);
        qs[i] = qu < queries ? __float2half(q[((size_t)qu * 4 + (r & 3)) * 128 + d]) : __float2half(0.f);
    }
    __syncthreads();
    const int block = (int)blockIdx.x * 4 + warp;
    if (block >= n_blocks) return;
    const int pos0 = block * 128, row = lane & 15, qu = qg * 4 + (row >> 2);
    const int qpos = qu < queries ? query_positions[qu] : -1;
    float best = -INFINITY;
    // a key block that starts after every query of the group sees no valid key (all masked: best stays -inf): skip its matrix products
    int group_max = -1;
    for (int u = 0; u < 4; ++u)
        if (qg * 4 + u < queries) group_max = max(group_max, query_positions[qg * 4 + u]);
    const int kt_end = pos0 > group_max ? 0 : 8;
    for (int kt = 0; kt < kt_end; ++kt) {
        if (pos0 + kt * 16 >= keys) break;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.f);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::load_matrix_sync(a, qs + ks * 16, 128);
            wmma::load_matrix_sync(b, k + (size_t)(pos0 + kt * 16) * 128 + ks * 16, 128);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(tile[warp], c, 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = (lane >> 4) * 8; j < (lane >> 4) * 8 + 8; ++j) {
            const int key = pos0 + kt * 16 + j;
            if (key < keys) {
                const int kp = key_positions[key];
                if (kp >= 0 && kp <= qpos && kp / 128 == block) best = fmaxf(best, tile[warp][row * 16 + j]);
            }
        }
        __syncwarp();
    }
    best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, 16));
    if (lane < 16 && qu < queries) scores[((size_t)qu * 4 + (row & 3)) * score_stride + block] = best;
}
#endif

template <class KT>
__global__ void block_scores_d128_kernel(const float* __restrict__ q, const KT* __restrict__ k,
                                         const int32_t* __restrict__ query_positions,
                                         const int32_t* __restrict__ key_positions, int index_heads, int keys,
                                         int block_size, int n_blocks, size_t score_stride, float* __restrict__ scores) {
    __shared__ float partial[kThreads / 32];
    const int block = (int)blockIdx.x, head = (int)blockIdx.y, query = (int)blockIdx.z;
    const int tid = (int)threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int32_t qpos = query_positions[query];
    const float4 qv = reinterpret_cast<const float4*>(q + ((size_t)query * index_heads + head) * 128)[lane];
    float best = -INFINITY;
    for (int slot = block * block_size + warp; slot < (block + 1) * block_size && slot < keys; slot += kThreads / 32) {
        const int32_t pos = key_positions[slot];
        if (pos < 0 || pos > qpos || pos / block_size != block) continue;
        const float4 kv = load4(k + (size_t)slot * 128, lane);
        const float dot = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w);
        best = fmaxf(best, dot);
    }
    if (lane == 0) partial[warp] = best;
    __syncthreads();
    if (tid == 0) {
        float b = partial[0];
        for (int w = 1; w < kThreads / 32; ++w) b = fmaxf(b, partial[w]);
        scores[((size_t)query * index_heads + head) * score_stride + block] = b;
    }
}


// ---- FP8 (E4M3) indexer keys: one float scale per key row of 128 values (the strata-m3 --idx-fp8 cache); one-token and window steps
__device__ __forceinline__ float e4m3_to_float(uint32_t b) {
    const int ef = (int)((b >> 3) & 15), mi = (int)(b & 7);
    const float v = ef == 0 ? (float)mi * (1.f / 512.f) : ldexpf(1.f + (float)mi * 0.125f, ef - 7);
    return (b & 0x80) ? -v : v;
}
__global__ void block_scores_q8_kernel(const float* __restrict__ q, const uint8_t* __restrict__ k8, const float* __restrict__ kscale,
                                       const int32_t* __restrict__ query_positions, const int32_t* __restrict__ key_positions,
                                       int index_heads, int keys, int block_size, int n_blocks, size_t score_stride,
                                       float* __restrict__ scores) {
    __shared__ float partial[kThreads / 32];
    __shared__ float lut[256];   // the 256 E4M3 values, filled with e4m3_to_float: the same floats as decoding each byte
    const int block = (int)blockIdx.x, head = (int)blockIdx.y, query = (int)blockIdx.z;
    const int tid = (int)threadIdx.x, warp = tid / 32, lane = tid % 32;
    for (int i = tid; i < 256; i += kThreads) lut[i] = e4m3_to_float((uint32_t)i);
    __syncthreads();
    const int32_t qpos = query_positions[query];
    const float4 qv = reinterpret_cast<const float4*>(q + ((size_t)query * index_heads + head) * 128)[lane];
    float best = -INFINITY;
    for (int slot = block * block_size + warp; slot < (block + 1) * block_size && slot < keys; slot += kThreads / 32) {
        const int32_t pos = key_positions[slot];
        if (pos < 0 || pos > qpos || pos / block_size != block) continue;
        const uint32_t w = reinterpret_cast<const uint32_t*>(k8 + (size_t)slot * 128)[lane];
        const float sc = kscale[slot];
        const float dot = warp_sum(qv.x * lut[w & 255] + qv.y * lut[(w >> 8) & 255] + qv.z * lut[(w >> 16) & 255] + qv.w * lut[w >> 24]) * sc;
        best = fmaxf(best, dot);
    }
    if (lane == 0) partial[warp] = best;
    __syncthreads();
    if (tid == 0) {
        float b = partial[0];
        for (int w = 1; w < kThreads / 32; ++w) b = fmaxf(b, partial[w]);
        scores[((size_t)query * index_heads + head) * score_stride + block] = b;
    }
}


// ---- block scores for a few queries, all index heads of a query in one thread block: each key row is read once for every head (the index keys
// have one head shared by the 4 index-query heads).  Per (key, head) the same lane products and warp sum as block_scores_d128_kernel, the same
// warp-to-slot assignment, and a max: the scores are bitwise the same.
template <class KT, bool Q8>
__global__ void block_scores_heads_kernel(const float* __restrict__ q, const KT* __restrict__ k, const float* __restrict__ kscale,
                                          const int32_t* __restrict__ query_positions, const int32_t* __restrict__ key_positions,
                                          int index_heads, int keys, int block_size, int n_blocks, size_t score_stride,
                                          float* __restrict__ scores) {
    __shared__ float partial[kThreads / 32][4];
    __shared__ float lut[Q8 ? 256 : 1];   // Q8: the 256 E4M3 values (same floats as e4m3_to_float per byte)
    const int block = (int)blockIdx.x, query = (int)blockIdx.y;
    const int tid = (int)threadIdx.x, warp = tid / 32, lane = tid % 32;
    if constexpr (Q8) {
        for (int i = tid; i < 256; i += kThreads) lut[i] = e4m3_to_float((uint32_t)i);
        __syncthreads();
    }
    const int32_t qpos = query_positions[query];
    float4 qv[4];
    for (int h = 0; h < 4; ++h)
        qv[h] = h < index_heads ? reinterpret_cast<const float4*>(q + ((size_t)query * index_heads + h) * 128)[lane] : make_float4(0.f, 0.f, 0.f, 0.f);
    float best[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (int slot = block * block_size + warp; slot < (block + 1) * block_size && slot < keys; slot += kThreads / 32) {
        const int32_t pos = key_positions[slot];
        if (pos < 0 || pos > qpos || pos / block_size != block) continue;
        float4 kv;
        float sc = 1.f;
        if constexpr (Q8) {
            const uint32_t w = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(k) + (size_t)slot * 128)[lane];
            kv = make_float4(lut[w & 255], lut[(w >> 8) & 255], lut[(w >> 16) & 255], lut[w >> 24]);
            sc = kscale[slot];
        } else {
            kv = load4(k + (size_t)slot * 128, lane);
        }
        for (int h = 0; h < 4; ++h) {
            if (h >= index_heads) break;
            float dot = warp_sum(qv[h].x * kv.x + qv[h].y * kv.y + qv[h].z * kv.z + qv[h].w * kv.w);
            if constexpr (Q8) dot *= sc;
            best[h] = fmaxf(best[h], dot);
        }
    }
    if (lane == 0)
        for (int h = 0; h < 4; ++h) partial[warp][h] = best[h];
    __syncthreads();
    if (tid < index_heads) {
        float b = partial[0][tid];
        for (int w = 1; w < kThreads / 32; ++w) b = fmaxf(b, partial[w][tid]);
        scores[((size_t)query * index_heads + tid) * score_stride + block] = b;
    }
}

// Measured on the V100 (7000 cached positions, top-16 of 55 blocks): 1589 us generic -> 449 us warp-per-slot ->
// 133 us with 1024 threads and 4 slots in flight per warp (unroll 8 spilled: 164 us).  A block per (query, kv head)
// serving its 16 query heads read K/V 16x less but ran on only 4 blocks of an 80-SM GPU: 2100 us.  It would need the
// slots split across blocks with a partial-softmax merge.
constexpr int kAttThreads = 1024;   // 32 warps per (query, head): more slots in flight per SM
constexpr int kAttWarps = kAttThreads / 32;
constexpr int kUnroll = 4;           // independent slots a warp keeps in flight (8 spilled: 164 us against 133)

template <class KT>
__global__ void __launch_bounds__(kAttThreads)
attention_d128_kernel(const float* __restrict__ q, const KT* __restrict__ k,
                      const KT* __restrict__ v, const int32_t* __restrict__ query_positions,
                      const int32_t* __restrict__ key_positions,
                      const int32_t* __restrict__ selected_blocks, int query_heads, int kv_heads,
                      int keys, int block_size, int top_k, size_t ids_stride, float* __restrict__ out) {
    __shared__ float logits[kMaxAttentionSlots];
    __shared__ float red[kAttWarps];
    __shared__ float4 accs[kAttWarps][32];
    const int row = (int)blockIdx.x, query = row / query_heads, qh = row % query_heads;
    const int kh = qh / (query_heads / kv_heads);
    const int tid = (int)threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int nslots = top_k * block_size;
    const int qpos = query_positions[query];
    const int32_t* blocks = selected_blocks + ((size_t)query * kv_heads + kh) * ids_stride;
    const float4 qv = reinterpret_cast<const float4*>(q + (size_t)row * 128)[lane];
    const float scale = rsqrtf(128.0f);

    // logits: kUnroll slots per warp at a time, the position checks first and then the key loads, all independent
    for (int base = warp; base < nslots; base += kAttWarps * kUnroll) {
        int pos[kUnroll];
        bool ok[kUnroll];
        float4 kv[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int slot = base + u * kAttWarps;
            ok[u] = false;
            pos[u] = 0;
            if (slot < nslots) {
                const int block = blocks[slot / block_size];
                pos[u] = block * block_size + slot % block_size;
                if (pos[u] < keys) {
                    const int kp = key_positions[pos[u]];
                    ok[u] = kp >= 0 && kp <= qpos && kp / block_size == block;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
            if (ok[u]) kv[u] = load4(k + ((size_t)pos[u] * kv_heads + kh) * 128, lane);
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int slot = base + u * kAttWarps;
            if (slot >= nslots) continue;
            float d = -INFINITY;
            if (ok[u]) d = warp_sum(qv.x * kv[u].x + qv.y * kv[u].y + qv.z * kv[u].z + qv.w * kv[u].w) * scale;
            if (lane == 0) logits[slot] = d;
        }
    }
    __syncthreads();
    float m = -INFINITY;
    for (int slot = tid; slot < nslots; slot += kAttThreads) m = fmaxf(m, logits[slot]);
    m = warp_max(m);
    if (lane == 0) red[warp] = m;
    __syncthreads();
    float max_score = red[0];
    for (int w = 1; w < kAttWarps; ++w) max_score = fmaxf(max_score, red[w]);
    __syncthreads();
    float sum = 0.0f;
    for (int slot = tid; slot < nslots; slot += kAttThreads) {
        const float sc = logits[slot];
        const float w = isfinite(max_score) && isfinite(sc) ? expf(sc - max_score) : 0.0f;
        logits[slot] = w;
        sum += w;
    }
    sum = warp_sum(sum);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    float denom = 0.0f;
    for (int w = 0; w < kAttWarps; ++w) denom += red[w];
    __syncthreads();
    // weighted values: the same, kUnroll slots in flight (a zero weight is an invalid or underflowed slot: its load is skipped)
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int base = warp; base < nslots; base += kAttWarps * kUnroll) {
        float w[kUnroll];
        int pos[kUnroll];
        float4 vv[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int slot = base + u * kAttWarps;
            w[u] = slot < nslots ? logits[slot] : 0.0f;
            pos[u] = slot < nslots ? blocks[slot / block_size] * block_size + slot % block_size : 0;
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
            if (w[u] != 0.0f) vv[u] = load4(v + ((size_t)pos[u] * kv_heads + kh) * 128, lane);
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
            if (w[u] != 0.0f) {
                acc.x += w[u] * vv[u].x; acc.y += w[u] * vv[u].y; acc.z += w[u] * vv[u].z; acc.w += w[u] * vv[u].w;
            }
    }
    accs[warp][lane] = acc;
    __syncthreads();
    if (tid < 32) {
        float4 t = accs[0][tid];
        for (int w = 1; w < kAttWarps; ++w) {
            t.x += accs[w][tid].x; t.y += accs[w][tid].y; t.z += accs[w][tid].z; t.w += accs[w][tid].w;
        }
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        reinterpret_cast<float4*>(out + (size_t)row * 128)[tid] = make_float4(t.x * inv, t.y * inv, t.z * inv, t.w * inv);
    }
}

__device__ __forceinline__ bool forced(int block, int qpos, int block_size, int local_blocks) {
    const int current = qpos / block_size;
    return block <= current && current - block < local_blocks;
}

__device__ __forceinline__ bool better(float score, int id, int is_forced,
                                       float other_score, int other_id, int other_forced) {
    if (is_forced != other_forced) return is_forced > other_forced;
    if (score != other_score) return score > other_score;
    return id < other_id;
}

// ---- split-K attention for dim == 128 (decode: few rows): each selected block of keys is its own thread block with a
// local softmax, writing (max, sum, value accumulator) to a scratch row; a merge kernel combines the partials.
// Measured at 7000 positions: 133 us (one block per row) -> 99 us.  Tried and dropped: serving the 16 query heads of a
// kv group from one K/V read per thread block (94 us at 7000 positions, 85 us instead of 29 at 270: 16 warp reductions
// per key and 64 accumulator registers outweigh the traffic saved).  A row of
// top_k blocks becomes top_k thread blocks instead of one, which a 80-SM GPU needs when only 64 rows exist.
constexpr int kSplitThreads = 256;
constexpr int kSplitWarps = kSplitThreads / 32;
constexpr int kSplitMaxBlockSize = 256;   // keys per selected block the shared logits hold
constexpr int kSplitMaxTopK = 32;
constexpr int kSplitMaxRows = 256;
constexpr int kPartialStride = 132;       // m, l, 2 pad, acc[128]: float4-aligned


/// Prefill attention: one block per (query, KV head); each warp is one of the group's query heads, so the group's
/// shared K / V rows are read once from memory (the other warps hit L1) instead of once per head.  Online softmax, no
/// logit buffer.  blockDim = 32 * (query_heads / kv_heads).
template <class KT>
__global__ void attention_group_kernel(const float* __restrict__ q, const KT* __restrict__ k, const KT* __restrict__ v,
                                       const int32_t* __restrict__ query_positions, const int32_t* __restrict__ key_positions,
                                       const int32_t* __restrict__ selected_blocks, int query_heads, int kv_heads, int keys,
                                       int block_size, int top_k, size_t ids_stride, float* __restrict__ out) {
    const int query = (int)blockIdx.x, kh = (int)blockIdx.y, warp = (int)threadIdx.x / 32, lane = (int)threadIdx.x % 32;
    const int group = query_heads / kv_heads, qh = kh * group + warp;
    const size_t row = (size_t)query * query_heads + qh;
    const int nslots = top_k * block_size, qpos = query_positions[query];
    const int32_t* blocks = selected_blocks + ((size_t)query * kv_heads + kh) * ids_stride;
    const float4 qv = reinterpret_cast<const float4*>(q + row * 128)[lane];
    const float scale = rsqrtf(128.0f);
    float m = -INFINITY, l = 0.0f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int slot0 = 0; slot0 < nslots; slot0 += 4) {
        float sc[4];
        int pos[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int slot = slot0 + u;
            sc[u] = -INFINITY;
            pos[u] = 0;
            if (slot < nslots) {
                const int block = blocks[slot / block_size];
                pos[u] = block * block_size + slot % block_size;
                if (pos[u] < keys) {
                    const int kp = key_positions[pos[u]];
                    if (kp >= 0 && kp <= qpos && kp / block_size == block) {
                        const float4 kv = load4(k + ((size_t)pos[u] * kv_heads + kh) * 128, lane);
                        sc[u] = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w) * scale;
                    }
                }
            }
        }
        const float mx = fmaxf(fmaxf(m, fmaxf(sc[0], sc[1])), fmaxf(sc[2], sc[3]));
        if (!isfinite(mx)) continue;
        const float alpha = isfinite(m) ? expf(m - mx) : 0.0f;
        acc.x *= alpha; acc.y *= alpha; acc.z *= alpha; acc.w *= alpha;
        l *= alpha;
        m = mx;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (!isfinite(sc[u])) continue;
            const float w = expf(sc[u] - mx);
            l += w;
            const float4 vv = load4(v + ((size_t)pos[u] * kv_heads + kh) * 128, lane);
            acc.x += w * vv.x; acc.y += w * vv.y; acc.z += w * vv.z; acc.w += w * vv.w;
        }
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    reinterpret_cast<float4*>(out + row * 128)[lane] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
}

#if !defined(__HIP_PLATFORM_AMD__)
// ---- tensor-core prefill attention (FP16 caches, 16 query heads per KV head, 128-key blocks, up to 16 blocks).
// One thread block per (query, KV head), 8 warps; a warp takes one selected key block at a time: S = Q K^T (16 x 128) with
// WMMA, a row softmax in shared memory, O = P V with WMMA, and the warps' partials merge in shared memory.
constexpr int kTcWarps = 8;
constexpr size_t kTcSmem = 4096 + (size_t)kTcWarps * 8192 + 8192 + 2 * kTcWarps * 16 * 4 + 2 * 16 * 4 + kTcWarps * 128;

__global__ void __launch_bounds__(256)
attention_tc_kernel(const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                    const int32_t* __restrict__ query_positions, const int32_t* __restrict__ key_positions,
                    const int32_t* __restrict__ selected_blocks, int query_heads, int kv_heads, int keys, int top_k,
                    size_t ids_stride, float* __restrict__ out) {
    using namespace nvcuda;
    extern __shared__ __align__(16) unsigned char smem[];
    __half* qs = reinterpret_cast<__half*>(smem);                         // [16][128]
    float* sb = reinterpret_cast<float*>(smem + 4096);                    // 8 warps x [16][128]
    float* run = sb + kTcWarps * 2048;                                    // merged acc [16][128]
    float* mw = run + 2048;                                               // [8][16]
    float* lw = mw + kTcWarps * 16;                                       // [8][16]
    float* mrun = lw + kTcWarps * 16;                                     // [16]
    float* lrun = mrun + 16;                                              // [16]
    unsigned char* vflag = reinterpret_cast<unsigned char*>(lrun + 16);   // [8][128]
    const int query = (int)blockIdx.x, kh = (int)blockIdx.y, tid = (int)threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qpos = query_positions[query];
    const int32_t* blocks = selected_blocks + ((size_t)query * kv_heads + kh) * ids_stride;
    const float* qsrc = q + ((size_t)query * query_heads + (size_t)kh * 16) * 128;
    for (int i = tid; i < 2048; i += 256) qs[i] = __float2half(qsrc[i]);
    for (int i = tid; i < 2048; i += 256) run[i] = 0.f;
    if (tid < 16) { mrun[tid] = -INFINITY; lrun[tid] = 0.f; }
    __syncthreads();
    const float scale = rsqrtf(128.0f);
    float* sbw = sb + warp * 2048;
    unsigned char* vf = vflag + warp * 128;
    for (int round = 0; round < (top_k + kTcWarps - 1) / kTcWarps; ++round) {
        const int chunk = round * kTcWarps + warp;
        bool active = false;
        int pos0 = 0;
        if (chunk < top_k) {
            const int block = blocks[chunk];
            if (block >= 0) {
                pos0 = block * 128;
                int any = 0;
                for (int j = lane; j < 128; j += 32) {
                    const int pos = pos0 + j;
                    bool ok = false;
                    if (pos < keys) {
                        const int kp = key_positions[pos];
                        ok = kp >= 0 && kp <= qpos && kp / 128 == block;
                    }
                    vf[j] = ok ? 1 : 0;
                    any |= ok ? 1 : 0;
                }
                active = __any_sync(0xffffffffu, any) != 0;
            }
        }
        if (lane < 16) { mw[warp * 16 + lane] = -INFINITY; lw[warp * 16 + lane] = 0.f; }
        __syncwarp();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
        if (active) {
            // S = Q K^T over the key tiles that start below `keys`
            for (int kt = 0; kt < 8; ++kt) {
                if (pos0 + kt * 16 >= keys) break;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.f);
                for (int ks = 0; ks < 8; ++ks) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                    wmma::load_matrix_sync(a, qs + ks * 16, 128);
                    wmma::load_matrix_sync(b, k + ((size_t)(pos0 + kt * 16) * kv_heads + kh) * 128 + ks * 16, kv_heads * 128);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(sbw + kt * 16, c, 128, wmma::mem_row_major);
            }
            __syncwarp();
            if (lane < 16) {   // row softmax, P written in place as halves (element j lands below the floats not yet read)
                float* row = sbw + lane * 128;
                __half* prow = reinterpret_cast<__half*>(row);
                float m = -INFINITY;
                for (int j = 0; j < 128; ++j)
                    if (vf[j]) m = fmaxf(m, row[j] * scale);
                float l = 0.f;
                for (int j = 0; j < 128; ++j) {
                    const float pv = vf[j] ? __expf(row[j] * scale - m) : 0.f;
                    l += pv;
                    prow[j] = __float2half(pv);
                }
                mw[warp * 16 + lane] = m;
                lw[warp * 16 + lane] = l;
            }
            __syncwarp();
            for (int dt = 0; dt < 8; ++dt) wmma::fill_fragment(acc[dt], 0.f);
            for (int kt = 0; kt < 8; ++kt) {
                if (pos0 + kt * 16 >= keys) break;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::load_matrix_sync(a, reinterpret_cast<const __half*>(sbw) + kt * 16, 256);
                for (int dt = 0; dt < 8; ++dt) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                    wmma::load_matrix_sync(b, v + ((size_t)(pos0 + kt * 16) * kv_heads + kh) * 128 + dt * 16, kv_heads * 128);
                    wmma::mma_sync(acc[dt], a, b, acc[dt]);
                }
            }
            __syncwarp();
            for (int dt = 0; dt < 8; ++dt) wmma::store_matrix_sync(sbw + dt * 16, acc[dt], 128, wmma::mem_row_major);
        }
        __syncthreads();
        // merge this round's warps into the running result
        for (int idx = tid; idx < 2048; idx += 256) {
            const int h = idx >> 7;
            float M = mrun[h];
            for (int w = 0; w < kTcWarps; ++w) M = fmaxf(M, mw[w * 16 + h]);
            if (!isfinite(M)) continue;
            const float sr = isfinite(mrun[h]) ? __expf(mrun[h] - M) : 0.f;
            float a = run[idx] * sr;
            for (int w = 0; w < kTcWarps; ++w) {
                const float mwh = mw[w * 16 + h];
                if (isfinite(mwh)) a += __expf(mwh - M) * sb[w * 2048 + idx];
            }
            run[idx] = a;
        }
        __syncthreads();
        if (tid < 16) {
            float M = mrun[tid];
            for (int w = 0; w < kTcWarps; ++w) M = fmaxf(M, mw[w * 16 + tid]);
            if (isfinite(M)) {
                float l = isfinite(mrun[tid]) ? lrun[tid] * __expf(mrun[tid] - M) : 0.f;
                for (int w = 0; w < kTcWarps; ++w) {
                    const float mwh = mw[w * 16 + tid];
                    if (isfinite(mwh)) l += __expf(mwh - M) * lw[w * 16 + tid];
                }
                mrun[tid] = M;
                lrun[tid] = l;
            }
        }
        __syncthreads();
    }
    float* o = out + ((size_t)query * query_heads + (size_t)kh * 16) * 128;
    for (int idx = tid; idx < 2048; idx += 256) {
        const float l = lrun[idx >> 7];
        o[idx] = l > 0.f ? run[idx] / l : 0.f;
    }
}

// ---- v2 of the tensor-core prefill attention (opt-in, STRATA_MSA_TC2): about 47 KB of shared memory instead of 80 KB, so two thread blocks
// share an SM.  A warp takes a half block of 64 keys (S 16 x 64 floats = 4 KB a warp); the softmax runs on all 32 lanes (two per row); the
// merge uses scale factors computed once per round, and the 16 x 128 accumulators leave the warps in two column halves.
constexpr size_t kTc2Smem = 4096 + (size_t)kTcWarps * 4096 + 8192 + 2 * kTcWarps * 16 * 4 + 2 * 16 * 4 + kTcWarps * 64 + 16 * 4 + kTcWarps * 16 * 4;

__global__ void __launch_bounds__(256, 2)
attention_tc2_kernel(const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                     const int32_t* __restrict__ query_positions, const int32_t* __restrict__ key_positions,
                     const int32_t* __restrict__ selected_blocks, int query_heads, int kv_heads, int keys, int top_k,
                     size_t ids_stride, float* __restrict__ out) {
    using namespace nvcuda;
    extern __shared__ __align__(16) unsigned char smem[];
    __half* qs = reinterpret_cast<__half*>(smem);                         // [16][128]
    float* sb = reinterpret_cast<float*>(smem + 4096);                    // 8 warps x [16][64]
    float* run = sb + kTcWarps * 1024;                                    // merged acc [16][128]
    float* mw = run + 2048;                                               // [8][16]
    float* lw = mw + kTcWarps * 16;                                       // [8][16]
    float* mrun = lw + kTcWarps * 16;                                     // [16]
    float* lrun = mrun + 16;                                              // [16]
    float* sr = lrun + 16;                                                // [16] scale of the running result
    float* swt = sr + 16;                                                 // [8][16] scale of each warp's piece
    unsigned char* vflag = reinterpret_cast<unsigned char*>(swt + kTcWarps * 16);   // [8][64]
    const int query = (int)blockIdx.x, kh = (int)blockIdx.y, tid = (int)threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qpos = query_positions[query];
    const int32_t* blocks = selected_blocks + ((size_t)query * kv_heads + kh) * ids_stride;
    const float* qsrc = q + ((size_t)query * query_heads + (size_t)kh * 16) * 128;
    for (int i = tid; i < 2048; i += 256) qs[i] = __float2half(qsrc[i]);
    for (int i = tid; i < 2048; i += 256) run[i] = 0.f;
    if (tid < 16) { mrun[tid] = -INFINITY; lrun[tid] = 0.f; }
    __syncthreads();
    const float scale = rsqrtf(128.0f);
    float* sbw = sb + warp * 1024;
    unsigned char* vf = vflag + warp * 64;
    const int units = 2 * top_k;
    for (int round = 0; round < (units + kTcWarps - 1) / kTcWarps; ++round) {
        const int unit = round * kTcWarps + warp;
        const int chunk = unit >> 1, half = unit & 1;
        bool active = false;
        int pos0 = 0, blk = -1;
        if (chunk < top_k) {
            blk = blocks[chunk];
            if (blk >= 0) {
                pos0 = blk * 128 + half * 64;
                int any = 0;
                for (int j = lane; j < 64; j += 32) {
                    const int pos = pos0 + j;
                    bool ok = false;
                    if (pos < keys) {
                        const int kp = key_positions[pos];
                        ok = kp >= 0 && kp <= qpos && kp / 128 == blk;
                    }
                    vf[j] = ok ? 1 : 0;
                    any |= ok ? 1 : 0;
                }
                active = __any_sync(0xffffffffu, any) != 0;
            }
        }
        if (lane < 16) { mw[warp * 16 + lane] = -INFINITY; lw[warp * 16 + lane] = 0.f; }
        __syncwarp();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
        if (active) {
            for (int kt = 0; kt < 4; ++kt) {
                if (pos0 + kt * 16 >= keys) break;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.f);
#pragma unroll
                for (int ks = 0; ks < 8; ++ks) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                    wmma::load_matrix_sync(a, qs + ks * 16, 128);
                    wmma::load_matrix_sync(b, k + ((size_t)(pos0 + kt * 16) * kv_heads + kh) * 128 + ks * 16, kv_heads * 128);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(sbw + kt * 16, c, 64, wmma::mem_row_major);
            }
            __syncwarp();
            {   // row softmax, two lanes a row (32 keys each); P is written back in place as halves after every lane has read its floats
                const int r = lane >> 1, h2 = lane & 1;
                float* row = sbw + r * 64 + h2 * 32;
                const unsigned char* vfr = vf + h2 * 32;
                float x[32];
                float m = -INFINITY;
#pragma unroll
                for (int j = 0; j < 32; ++j) {
                    x[j] = row[j] * scale;
                    if (vfr[j]) m = fmaxf(m, x[j]);
                }
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                float l = 0.f;
#pragma unroll
                for (int j = 0; j < 32; ++j) {
                    x[j] = vfr[j] ? __expf(x[j] - m) : 0.f;
                    l += x[j];
                }
                l += __shfl_xor_sync(0xffffffffu, l, 1);
                __syncwarp();
                __half* prow = reinterpret_cast<__half*>(sbw) + r * 128 + h2 * 32;
#pragma unroll
                for (int j = 0; j < 32; ++j) prow[j] = __float2half(x[j]);
                if (h2 == 0) { mw[warp * 16 + r] = m; lw[warp * 16 + r] = l; }
            }
            __syncwarp();
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) wmma::fill_fragment(acc[dt], 0.f);
            for (int kt = 0; kt < 4; ++kt) {
                if (pos0 + kt * 16 >= keys) break;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::load_matrix_sync(a, reinterpret_cast<const __half*>(sbw) + kt * 16, 128);
#pragma unroll
                for (int dt = 0; dt < 8; ++dt) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                    wmma::load_matrix_sync(b, v + ((size_t)(pos0 + kt * 16) * kv_heads + kh) * 128 + dt * 16, kv_heads * 128);
                    wmma::mma_sync(acc[dt], a, b, acc[dt]);
                }
            }
            __syncwarp();
        }
        __syncthreads();
        if (tid < 16) {   // scale factors of this round and the new running max / sum
            float M = mrun[tid];
            for (int w = 0; w < kTcWarps; ++w) M = fmaxf(M, mw[w * 16 + tid]);
            if (isfinite(M)) {
                const float s0 = isfinite(mrun[tid]) ? __expf(mrun[tid] - M) : 0.f;
                float l = lrun[tid] * s0;
                sr[tid] = s0;
                for (int w = 0; w < kTcWarps; ++w) {
                    const float mwh = mw[w * 16 + tid];
                    const float sw = isfinite(mwh) ? __expf(mwh - M) : 0.f;
                    swt[w * 16 + tid] = sw;
                    l += sw * lw[w * 16 + tid];
                }
                mrun[tid] = M;
                lrun[tid] = l;
            } else {
                sr[tid] = 1.f;
                for (int w = 0; w < kTcWarps; ++w) swt[w * 16 + tid] = 0.f;
            }
        }
        __syncthreads();
        for (int c = 0; c < 2; ++c) {   // column halves: the warps' accumulators go through their 4 KB buffers
            if (active) {
#pragma unroll
                for (int dt = 0; dt < 4; ++dt) wmma::store_matrix_sync(sbw + dt * 16, acc[c * 4 + dt], 64, wmma::mem_row_major);
            }
            __syncthreads();
            for (int idx = tid; idx < 1024; idx += 256) {
                const int h = idx >> 6, cc = idx & 63;
                float a = run[h * 128 + c * 64 + cc] * sr[h];
#pragma unroll
                for (int w = 0; w < kTcWarps; ++w) {
                    const float sw = swt[w * 16 + h];
                    if (sw != 0.f) a += sw * sb[w * 1024 + h * 64 + cc];
                }
                run[h * 128 + c * 64 + cc] = a;
            }
            __syncthreads();
        }
    }
    float* o = out + ((size_t)query * query_heads + (size_t)kh * 16) * 128;
    for (int idx = tid; idx < 2048; idx += 256) {
        const float l = lrun[idx >> 7];
        o[idx] = l > 0.f ? run[idx] / l : 0.f;
    }
}
#endif

template <class KT>
__global__ void __launch_bounds__(kSplitThreads)
attention_split_partial_kernel(const float* __restrict__ q, const KT* __restrict__ k, const KT* __restrict__ v,
                               const int32_t* __restrict__ query_positions, const int32_t* __restrict__ key_positions,
                               const int32_t* __restrict__ selected_blocks, int query_heads, int kv_heads, int keys,
                               int block_size, size_t ids_stride, float* __restrict__ partial) {
    __shared__ float logits[kSplitMaxBlockSize];
    __shared__ float red[kSplitWarps];
    __shared__ float4 accs[kSplitWarps][32];
    const int row = (int)blockIdx.x, split = (int)blockIdx.y, query = row / query_heads, qh = row % query_heads;
    const int kh = qh / (query_heads / kv_heads);
    const int tid = (int)threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int qpos = query_positions[query];
    const int block = selected_blocks[((size_t)query * kv_heads + kh) * ids_stride + split];
    const float4 qv = reinterpret_cast<const float4*>(q + (size_t)row * 128)[lane];
    const float scale = rsqrtf(128.0f);
    for (int base = warp; base < block_size; base += kSplitWarps * 4) {
        int pos[4];
        bool ok[4];
        float4 kv[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int i = base + u * kSplitWarps;
            ok[u] = false;
            pos[u] = block * block_size + i;
            if (i < block_size && pos[u] < keys) {
                const int kp = key_positions[pos[u]];
                ok[u] = kp >= 0 && kp <= qpos && kp / block_size == block;
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (ok[u]) kv[u] = load4(k + ((size_t)pos[u] * kv_heads + kh) * 128, lane);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int i = base + u * kSplitWarps;
            if (i >= block_size) continue;
            float d = -INFINITY;
            if (ok[u]) d = warp_sum(qv.x * kv[u].x + qv.y * kv[u].y + qv.z * kv[u].z + qv.w * kv[u].w) * scale;
            if (lane == 0) logits[i] = d;
        }
    }
    __syncthreads();
    float m = -INFINITY;
    for (int i = tid; i < block_size; i += kSplitThreads) m = fmaxf(m, logits[i]);
    m = warp_max(m);
    if (lane == 0) red[warp] = m;
    __syncthreads();
    float bm = red[0];
    for (int w = 1; w < kSplitWarps; ++w) bm = fmaxf(bm, red[w]);
    __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i < block_size; i += kSplitThreads) {
        const float sc = logits[i];
        const float w = isfinite(bm) && isfinite(sc) ? expf(sc - bm) : 0.0f;
        logits[i] = w;
        sum += w;
    }
    sum = warp_sum(sum);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int base = warp; base < block_size; base += kSplitWarps * 4) {
        float w[4];
        float4 vv[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) w[u] = base + u * kSplitWarps < block_size ? logits[base + u * kSplitWarps] : 0.0f;
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (w[u] != 0.0f) vv[u] = load4(v + ((size_t)(block * block_size + base + u * kSplitWarps) * kv_heads + kh) * 128, lane);
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (w[u] != 0.0f) { acc.x += w[u] * vv[u].x; acc.y += w[u] * vv[u].y; acc.z += w[u] * vv[u].z; acc.w += w[u] * vv[u].w; }
    }
    accs[warp][lane] = acc;
    __syncthreads();
    float* out = partial + ((size_t)row * gridDim.y + split) * kPartialStride;
    if (tid < 32) {
        float4 t = accs[0][tid];
        for (int w = 1; w < kSplitWarps; ++w) { t.x += accs[w][tid].x; t.y += accs[w][tid].y; t.z += accs[w][tid].z; t.w += accs[w][tid].w; }
        reinterpret_cast<float4*>(out + 4)[tid] = t;
        if (tid == 0) {
            float l = 0.0f;
            for (int w = 0; w < kSplitWarps; ++w) l += red[w];
            out[0] = bm;
            out[1] = l;
        }
    }
}

__global__ void attention_split_merge_kernel(const float* __restrict__ partial, int splits, float* __restrict__ out) {
    const int row = (int)blockIdx.x, lane = (int)threadIdx.x;
    const float* base = partial + (size_t)row * splits * kPartialStride;
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, base[(size_t)s * kPartialStride]);
    float L = 0.0f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (isfinite(M)) {
        for (int s = 0; s < splits; ++s) {
            const float* ps = base + (size_t)s * kPartialStride;
            const float ms = ps[0];
            if (!isfinite(ms)) continue;
            const float f = expf(ms - M);
            L += f * ps[1];
            const float4 a = reinterpret_cast<const float4*>(ps + 4)[lane];
            acc.x += f * a.x; acc.y += f * a.y; acc.z += f * a.z; acc.w += f * a.w;
        }
    }
    const float inv = L > 0.0f ? 1.0f / L : 0.0f;
    reinterpret_cast<float4*>(out + (size_t)row * 128)[lane] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv);
}

// ---- selection for n_blocks <= 64 with one warp per (query, head) row: each lane holds two blocks, a round is a warp
// argmax over (forced, score, lower id) with shuffles, no block barriers.  Same ordering as select_blocks_kernel.
__device__ __forceinline__ void pick_better(float& s, int& id, int& f, float os, int oid, int of) {
    if (oid != INT_MAX && (id == INT_MAX || better(os, oid, of, s, id, f))) { s = os; id = oid; f = of; }
}

__global__ void select_blocks_warp_kernel(const float* __restrict__ scores, const int32_t* __restrict__ query_positions,
                                          int index_heads, int n_blocks, int block_size, int top_k, int local_blocks,
                                          size_t score_stride, size_t id_stride, int32_t* __restrict__ ids) {
    const int row = (int)blockIdx.x, query = row / index_heads, lane = (int)threadIdx.x;
    const int qpos = query_positions[query];
    const float* row_scores = scores + (size_t)row * score_stride;
    int32_t* row_ids = ids + (size_t)row * id_stride;
    float sc[2];
    int fo[2];
    bool taken[2] = {false, false};
    for (int j = 0; j < 2; ++j) {
        const int b = lane + 32 * j;
        sc[j] = b < n_blocks ? row_scores[b] : 0.0f;
        fo[j] = b < n_blocks && forced(b, qpos, block_size, local_blocks) ? 1 : 0;
    }
    int chosen[kMaxTopK];
    for (int rank = 0; rank < top_k; ++rank) {
        float bs = -INFINITY;
        int bid = INT_MAX, bf = 0;
        for (int j = 0; j < 2; ++j) {
            const int b = lane + 32 * j;
            if (b < n_blocks && !taken[j]) pick_better(bs, bid, bf, sc[j], b, fo[j]);
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float os = __shfl_xor_sync(0xffffffffu, bs, o);
            const int oid = __shfl_xor_sync(0xffffffffu, bid, o);
            const int of = __shfl_xor_sync(0xffffffffu, bf, o);
            pick_better(bs, bid, bf, os, oid, of);
        }
        chosen[rank] = bid;   // the same in every lane
        if (bid != INT_MAX && (bid & 31) == lane) taken[bid >> 5] = true;
    }
    if (lane == 0) {
        // canonical ascending IDs, as in the generic kernel
        for (int i = 1; i < top_k; ++i) {
            const int32_t value = chosen[i];
            int j = i;
            while (j > 0 && chosen[j - 1] > value) { chosen[j] = chosen[j - 1]; --j; }
            chosen[j] = value;
        }
        for (int i = 0; i < top_k; ++i) row_ids[i] = chosen[i];
    }
}

// ---- selection for 64 < n_blocks <= kSelMaxBlocks (1M positions with 128-key blocks).  The order of `better` (forced first, then score, then the
// lower id) is packed into one 64-bit key: bit 63 forced, bits 62..31 the score as an ordered integer (-0 counted as +0, as `!=` does), bits 30..0
// the inverted id; ids are unique, so the keys are, and the top-k by key is the top-k set of select_blocks_kernel (output sorted ascending:
// bitwise the same).  Each warp picks the top-k of its blocks (thread t owns blocks t, t + kThreads, ...; rounds of shuffle max, no block
// barriers, a rescan of shared-memory scores only in the lane that owned the winner), then one warp picks the top-k of the 8 warps' lists.
constexpr int kSelMaxBlocks = 8192;
__device__ __forceinline__ unsigned long long sel_key(float score, int id, int is_forced) {
    uint32_t u = __float_as_uint(score + 0.0f);   // -0 -> +0
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)is_forced << 63) | ((unsigned long long)u << 31) | (unsigned long long)(0x7fffffffu - (uint32_t)id);
}
__device__ __forceinline__ unsigned long long warp_max_u64(unsigned long long k) {
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long other = __shfl_xor_sync(0xffffffffu, k, o);
        k = other > k ? other : k;
    }
    return k;
}
__global__ void __launch_bounds__(kThreads) select_blocks_reg_kernel(const float* __restrict__ scores, const int32_t* __restrict__ query_positions,
                                         int index_heads, int n_blocks, int block_size, int top_k, int local_blocks,
                                         size_t score_stride, size_t id_stride, int32_t* __restrict__ ids) {
    __shared__ float s_sc[kSelMaxBlocks];
    __shared__ unsigned long long cand[(kThreads / 32) * kMaxTopK];
    __shared__ int32_t chosen[kMaxTopK];
    const int row = (int)blockIdx.x, query = row / index_heads, tid = (int)threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int qpos = query_positions[query];
    const float* row_scores = scores + (size_t)row * score_stride;
    for (int b = tid; b < n_blocks; b += kThreads) s_sc[b] = row_scores[b];   // each thread later reads only the entries it wrote
    const int mine = n_blocks > tid ? (n_blocks - tid + kThreads - 1) / kThreads : 0;
    uint32_t taken = 0;
    unsigned long long best = 0;   // 0: nothing left (every real key is nonzero)
    auto rescan = [&]() {
        best = 0;
        for (int j = 0; j < mine; ++j) {
            if (taken & (1u << j)) continue;
            const int b = tid + j * kThreads;
            const unsigned long long k = sel_key(s_sc[b], b, forced(b, qpos, block_size, local_blocks) ? 1 : 0);
            best = k > best ? k : best;
        }
    };
    rescan();
    for (int r = 0; r < top_k; ++r) {
        const unsigned long long w = warp_max_u64(best);
        if (lane == 0) cand[warp * top_k + r] = w;
        if (w != 0 && w == best) {
            const int b = (int)(0x7fffffffu - (uint32_t)(w & 0x7fffffffu));
            taken |= 1u << ((b - tid) / kThreads);
            rescan();
        }
    }
    __syncthreads();
    if (warp == 0) {
        const int nc = (kThreads / 32) * top_k;   // <= 512: at most 16 a lane
        uint32_t used = 0;
        auto lbest = [&]() {
            unsigned long long m = 0;
            for (int i = 0; lane + 32 * i < nc; ++i)
                if (!(used & (1u << i))) { const unsigned long long k = cand[lane + 32 * i]; m = k > m ? k : m; }
            return m;
        };
        unsigned long long m = lbest();
        for (int r = 0; r < top_k; ++r) {
            const unsigned long long w = warp_max_u64(m);
            if (lane == 0) chosen[r] = (int32_t)(0x7fffffffu - (uint32_t)(w & 0x7fffffffu));
            if (w != 0 && w == m) {
                for (int i = 0; lane + 32 * i < nc; ++i)
                    if (cand[lane + 32 * i] == w) { used |= 1u << i; break; }
                m = lbest();
            }
        }
        __syncwarp();
        if (lane == 0) {
            int32_t* row_ids = ids + (size_t)row * id_stride;
            for (int i = 1; i < top_k; ++i) {
                const int32_t value = chosen[i];
                int j = i;
                while (j > 0 && chosen[j - 1] > value) { chosen[j] = chosen[j - 1]; --j; }
                chosen[j] = value;
            }
            for (int i = 0; i < top_k; ++i) row_ids[i] = chosen[i];
        }
    }
}

__global__ void select_blocks_kernel(const float* __restrict__ scores,
                                    const int32_t* __restrict__ query_positions,
                                    int index_heads, int n_blocks, int block_size,
                                    int top_k, int local_blocks, size_t score_stride,
                                    size_t id_stride, int32_t* __restrict__ ids) {
    __shared__ float candidate_score[kThreads];
    __shared__ int candidate_id[kThreads];
    __shared__ int candidate_forced[kThreads];
    __shared__ int32_t selected[kMaxTopK];

    const int row = (int)blockIdx.x;
    const int query = row / index_heads;
    const int tid = (int)threadIdx.x;
    const int qpos = query_positions[query];
    const float* row_scores = scores + (size_t)row * score_stride;
    int32_t* row_ids = ids + (size_t)row * id_stride;

    for (int rank = 0; rank < top_k; ++rank) {
        float best_score = -INFINITY;
        int best_id = INT_MAX;
        int best_forced = 0;
        for (int block = tid; block < n_blocks; block += kThreads) {
            bool already = false;
            for (int i = 0; i < rank; ++i) already |= selected[i] == block;
            if (already) continue;
            const float score = row_scores[block];
            const int local = forced(block, qpos, block_size, local_blocks) ? 1 : 0;
            if (best_id == INT_MAX || better(score, block, local, best_score, best_id, best_forced)) {
                best_score = score;
                best_id = block;
                best_forced = local;
            }
        }
        candidate_score[tid] = best_score;
        candidate_id[tid] = best_id;
        candidate_forced[tid] = best_forced;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride && candidate_id[tid + stride] != INT_MAX &&
                (candidate_id[tid] == INT_MAX ||
                 better(candidate_score[tid + stride], candidate_id[tid + stride], candidate_forced[tid + stride],
                        candidate_score[tid], candidate_id[tid], candidate_forced[tid]))) {
                candidate_score[tid] = candidate_score[tid + stride];
                candidate_id[tid] = candidate_id[tid + stride];
                candidate_forced[tid] = candidate_forced[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            selected[rank] = candidate_id[0];
            row_ids[rank] = candidate_id[0];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // The attention mask consumes a set; canonical ascending IDs make captures and diagnostics stable.
        for (int i = 1; i < top_k; ++i) {
            const int32_t value = row_ids[i];
            int j = i;
            while (j > 0 && row_ids[j - 1] > value) { row_ids[j] = row_ids[j - 1]; --j; }
            row_ids[j] = value;
        }
    }
}

__global__ void attention_f32_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                    const float* __restrict__ v,
                                    const int32_t* __restrict__ query_positions,
                                    const int32_t* __restrict__ key_positions,
                                    const int32_t* __restrict__ selected_blocks,
                                    int query_heads, int kv_heads, int dim, int keys,
                                    int block_size, int top_k, size_t ids_stride,
                                    float* __restrict__ out) {
    __shared__ float logits[kMaxAttentionSlots];
    __shared__ float partial[kThreads];
    const int row = (int)blockIdx.x;
    const int query = row / query_heads;
    const int qh = row % query_heads;
    const int group = query_heads / kv_heads;
    const int kh = qh / group;
    const int tid = (int)threadIdx.x;
    const int nslots = top_k * block_size;
    const int qpos = query_positions[query];
    const float* qrow = q + (size_t)row * dim;
    const int32_t* blocks = selected_blocks + ((size_t)query * kv_heads + kh) * ids_stride;
    const float scale = rsqrtf((float)dim);

    for (int slot = tid; slot < nslots; slot += kThreads) {
        const int block = blocks[slot / block_size];
        const int pos = block * block_size + slot % block_size;
        if (pos >= keys || key_positions[pos] < 0 || key_positions[pos] > qpos ||
            key_positions[pos] / block_size != block) {
            logits[slot] = -INFINITY;
            continue;
        }
        const float* key = k + ((size_t)pos * kv_heads + kh) * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) dot = fmaf(qrow[d], key[d], dot);
        logits[slot] = dot * scale;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int slot = tid; slot < nslots; slot += kThreads) local_max = fmaxf(local_max, logits[slot]);
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float max_score = partial[0];

    float local_sum = 0.0f;
    for (int slot = tid; slot < nslots; slot += kThreads) {
        const float score = logits[slot];
        const float weight = isfinite(max_score) && isfinite(score) ? expf(score - max_score) : 0.0f;
        logits[slot] = weight;
        local_sum += weight;
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float denom = partial[0];
    if (tid < dim) {
        float result = 0.0f;
        if (denom > 0.0f) {
            for (int slot = 0; slot < nslots; ++slot) {
                const int block = blocks[slot / block_size];
                const int pos = block * block_size + slot % block_size;
                if (pos >= keys || key_positions[pos] < 0 || key_positions[pos] > qpos ||
                    key_positions[pos] / block_size != block) continue;
                const float probability = logits[slot] / denom;
                result += probability * v[((size_t)pos * kv_heads + kh) * dim + tid];
            }
        }
        out[(size_t)row * dim + tid] = result;
    }
}

bool slow_msa() {
    static const bool slow = std::getenv("STRATA_MSA_SLOW") != nullptr;
    return slow;
}
bool select_old() {   // A/B: the block-barrier selection kernel for every n_blocks > 64
    static const bool old = std::getenv("STRATA_MSA_SELECT_OLD") != nullptr;
    return old;
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("MiniMax MSA ") + operation + ": " + cudaGetErrorString(error));
}

} // namespace

void minimax_msa_block_scores(const float* q, const float* k, const int32_t* query_positions,
                              const int32_t* key_positions, int queries, int index_heads, int dim,
                              int keys, int block_size, int n_blocks, float* scores,
                              size_t score_stride, void* stream) {
    if (!q || !k || !query_positions || !key_positions || !scores || queries < 1 || index_heads < 1 ||
        dim < 1 || keys < 1 || block_size < 1 || block_size > 1024 || n_blocks < 1 ||
        score_stride < (size_t)n_blocks || n_blocks > 65535 || index_heads > 65535 || queries > 65535)
        throw std::invalid_argument("MiniMax MSA block scores: invalid geometry or buffer");
    const dim3 grid((unsigned)n_blocks, (unsigned)index_heads, (unsigned)queries);
    if (dim == 128 && !slow_msa())
        block_scores_d128_kernel<float><<<grid, kThreads, 0, (cudaStream_t)stream>>>(q, k, query_positions, key_positions,
            index_heads, keys, block_size, n_blocks, score_stride, scores);
    else
    block_scores_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(q, k, query_positions, key_positions,
        queries, index_heads, dim, keys, block_size, n_blocks, score_stride, scores);
    check(cudaGetLastError(), "block score launch");
}

void minimax_msa_select_blocks(const float* scores, const int32_t* query_positions,
                               int queries, int index_heads, int n_blocks, int block_size,
                               int top_k, int local_blocks, size_t score_stride,
                               int32_t* ids, size_t id_stride, void* stream) {
    if (!scores || !query_positions || !ids || queries < 1 || index_heads < 1 || n_blocks < 1 ||
        block_size < 1 || top_k < 1 || top_k > n_blocks || top_k > kMaxTopK || local_blocks < 1 ||
        local_blocks > top_k || score_stride < (size_t)n_blocks || id_stride < (size_t)top_k ||
        (size_t)queries * (size_t)index_heads > 2147483647u)
        throw std::invalid_argument("MiniMax MSA block selection: invalid geometry or buffer");
    const int rows = queries * index_heads;
    if (n_blocks <= 64 && !slow_msa())
        select_blocks_warp_kernel<<<rows, 32, 0, (cudaStream_t)stream>>>(scores, query_positions, index_heads, n_blocks,
            block_size, top_k, local_blocks, score_stride, id_stride, ids);
    else if (n_blocks <= kSelMaxBlocks && !slow_msa() && !select_old())
        select_blocks_reg_kernel<<<rows, kThreads, 0, (cudaStream_t)stream>>>(scores, query_positions, index_heads, n_blocks,
            block_size, top_k, local_blocks, score_stride, id_stride, ids);
    else
    select_blocks_kernel<<<rows, kThreads, 0, (cudaStream_t)stream>>>(scores, query_positions, index_heads,
        n_blocks, block_size, top_k, local_blocks, score_stride, id_stride, ids);
    check(cudaGetLastError(), "block selection launch");
}

void minimax_msa_attention_f32(const float* q, const float* k, const float* v,
                               const int32_t* query_positions, const int32_t* key_positions,
                               const int32_t* selected_blocks, int queries, int query_heads,
                               int kv_heads, int dim, int keys, int block_size, int top_k,
                               size_t ids_stride, float* out, void* stream) {
    if (!q || !k || !v || !query_positions || !key_positions || !selected_blocks || !out ||
        queries < 1 || query_heads < 1 || kv_heads < 1 || query_heads % kv_heads != 0 ||
        dim < 1 || keys < 1 || block_size < 1 || top_k < 1 ||
        (int64_t)top_k * block_size > kMaxAttentionSlots || ids_stride < (size_t)top_k ||
        (size_t)queries * (size_t)query_heads > 2147483647u)
        throw std::invalid_argument("MiniMax MSA attention: invalid geometry or buffer");
    const int rows = queries * query_heads;
    static const bool no_split = std::getenv("STRATA_MSA_NOSPLIT") != nullptr;   // A/B: one thread block per row
    // opt-in (different summation order from the per-head kernel): the grouped prefill kernel for FLOAT caches
    static const bool group_f32 = std::getenv("STRATA_MSA_GROUP") != nullptr;
    if (dim == 128 && !slow_msa() && group_f32 && queries >= 16 && query_heads / kv_heads <= 32) {
        attention_group_kernel<float><<<dim3((unsigned)queries, (unsigned)kv_heads), 32 * (query_heads / kv_heads), 0, (cudaStream_t)stream>>>(
            q, k, v, query_positions, key_positions, selected_blocks, query_heads, kv_heads, keys, block_size, top_k, ids_stride, out);
    } else if (dim == 128 && !slow_msa() && !no_split && rows <= kSplitMaxRows && block_size <= kSplitMaxBlockSize &&
        top_k <= kSplitMaxTopK && ids_stride >= (size_t)top_k) {
        static float* partial = nullptr;   // one scratch for every decode call: kSplitMaxRows x kSplitMaxTopK partial rows
        if (!partial)
            check(cudaMalloc(&partial, (size_t)kSplitMaxRows * kSplitMaxTopK * kPartialStride * sizeof(float)), "split attention scratch");
        attention_split_partial_kernel<float><<<dim3((unsigned)rows, (unsigned)top_k), kSplitThreads, 0, (cudaStream_t)stream>>>(
            q, k, v, query_positions, key_positions, selected_blocks, query_heads, kv_heads, keys, block_size, ids_stride, partial);
        attention_split_merge_kernel<<<rows, 32, 0, (cudaStream_t)stream>>>(partial, top_k, out);
    } else if (dim == 128 && !slow_msa())
        attention_d128_kernel<float><<<rows, kAttThreads, 0, (cudaStream_t)stream>>>(q, k, v, query_positions, key_positions,
            selected_blocks, query_heads, kv_heads, keys, block_size, top_k, ids_stride, out);
    else
    attention_f32_kernel<<<rows, kThreads, 0, (cudaStream_t)stream>>>(q, k, v, query_positions,
        key_positions, selected_blocks, query_heads, kv_heads, dim, keys,
        block_size, top_k, ids_stride, out);
    check(cudaGetLastError(), "sparse attention launch");
}


// ---- half-precision caches: the K / V / indexer-key rows are stored as FP16 (the caller converts on write); only the
// 128-wide fast kernels take them.
void minimax_msa_block_scores_h(const float* q, const uint16_t* k, const int32_t* query_positions,
                                const int32_t* key_positions, int queries, int index_heads, int dim, int keys,
                                int block_size, int n_blocks, float* scores, size_t score_stride, void* stream) {
    if (!q || !k || !query_positions || !key_positions || !scores || queries < 1 || index_heads < 1 || dim != 128 ||
        keys < 1 || block_size < 1 || block_size > 1024 || n_blocks < 1 || score_stride < (size_t)n_blocks ||
        n_blocks > 65535 || index_heads > 65535 || queries > 65535)
        throw std::invalid_argument("MiniMax MSA block scores (half): invalid geometry or buffer");
    const dim3 grid((unsigned)n_blocks, (unsigned)index_heads, (unsigned)queries);
#if !defined(__HIP_PLATFORM_AMD__)
    static const bool no_tc_scores = std::getenv("STRATA_MSA_NOTC") != nullptr;
    if (!no_tc_scores && index_heads == 4 && block_size == 128 && queries >= 16 && (size_t)queries * 4 <= 2147483647u) {
        block_scores_tc_kernel<<<dim3((unsigned)((n_blocks + 3) / 4), (unsigned)((queries + 3) / 4)), 128, 0, (cudaStream_t)stream>>>(
            q, (const __half*)k, query_positions, key_positions, queries, keys, n_blocks, score_stride, scores);
        check(cudaGetLastError(), "block score (tensor core) launch");
        return;
    }
#endif
    static const bool per_head = std::getenv("STRATA_MSA_SCORES_PER_HEAD") != nullptr;   // A/B: one thread block per (block, head)
    // all heads in one thread block pays from about 16K cached positions (128 blocks); below that the per-head grid keeps more of the GPU busy
    if (!per_head && index_heads <= 4 && n_blocks >= 128)
        block_scores_heads_kernel<__half, false><<<dim3((unsigned)n_blocks, (unsigned)queries), kThreads, 0, (cudaStream_t)stream>>>(
            q, (const __half*)k, nullptr, query_positions, key_positions, index_heads, keys, block_size, n_blocks, score_stride, scores);
    else
    block_scores_d128_kernel<__half><<<grid, kThreads, 0, (cudaStream_t)stream>>>(q, (const __half*)k, query_positions,
        key_positions, index_heads, keys, block_size, n_blocks, score_stride, scores);
    check(cudaGetLastError(), "block score (half) launch");
}

void minimax_msa_block_scores_q8(const float* q, const uint8_t* k8, const float* kscale, const int32_t* query_positions,
                                 const int32_t* key_positions, int queries, int index_heads, int dim, int keys,
                                 int block_size, int n_blocks, float* scores, size_t score_stride, void* stream) {
    if (!q || !k8 || !kscale || !query_positions || !key_positions || !scores || queries < 1 || index_heads < 1 || dim != 128 ||
        keys < 1 || block_size < 1 || block_size > 1024 || n_blocks < 1 || score_stride < (size_t)n_blocks ||
        n_blocks > 65535 || index_heads > 65535 || queries > 65535)
        throw std::invalid_argument("MiniMax MSA block scores (fp8): invalid geometry or buffer");
    const dim3 grid((unsigned)n_blocks, (unsigned)index_heads, (unsigned)queries);
    if (index_heads <= 4 && n_blocks >= 128)
        block_scores_heads_kernel<uint8_t, true><<<dim3((unsigned)n_blocks, (unsigned)queries), kThreads, 0, (cudaStream_t)stream>>>(
            q, k8, kscale, query_positions, key_positions, index_heads, keys, block_size, n_blocks, score_stride, scores);
    else
    block_scores_q8_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(q, k8, kscale, query_positions, key_positions, index_heads, keys,
                                                                       block_size, n_blocks, score_stride, scores);
    check(cudaGetLastError(), "block score (fp8) launch");
}

void minimax_msa_attention_h(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* query_positions,
                             const int32_t* key_positions, const int32_t* selected_blocks, int queries,
                             int query_heads, int kv_heads, int dim, int keys, int block_size, int top_k,
                             size_t ids_stride, float* out, void* stream) {
    if (!q || !k || !v || !query_positions || !key_positions || !selected_blocks || !out || queries < 1 ||
        query_heads < 1 || kv_heads < 1 || query_heads % kv_heads != 0 || dim != 128 || keys < 1 || block_size < 1 ||
        top_k < 1 || (int64_t)top_k * block_size > kMaxAttentionSlots || ids_stride < (size_t)top_k ||
        (size_t)queries * (size_t)query_heads > 2147483647u)
        throw std::invalid_argument("MiniMax MSA attention (half): invalid geometry or buffer");
    const int rows = queries * query_heads;
    static const bool no_group_h = std::getenv("STRATA_MSA_NOGROUP") != nullptr;
#if !defined(__HIP_PLATFORM_AMD__)
    static const bool no_tc = std::getenv("STRATA_MSA_NOTC") != nullptr;
    if (!no_tc && !no_group_h && queries >= 16 && query_heads / kv_heads == 16 && block_size == 128 && top_k <= 16) {
        static bool attr = false;
        if (!attr) {
            check(cudaFuncSetAttribute(attention_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kTcSmem), "tc attention smem");
            attr = true;
        }
        static const bool tc2 = std::getenv("STRATA_MSA_TC2") != nullptr;
        if (tc2) {
            static bool attr2 = false;
            if (!attr2) {
                check(cudaFuncSetAttribute(attention_tc2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kTc2Smem), "tc2 attention smem");
                attr2 = true;
            }
            attention_tc2_kernel<<<dim3((unsigned)queries, (unsigned)kv_heads), 256, kTc2Smem, (cudaStream_t)stream>>>(
                q, (const __half*)k, (const __half*)v, query_positions, key_positions, selected_blocks, query_heads, kv_heads, keys,
                top_k, ids_stride, out);
        } else
        attention_tc_kernel<<<dim3((unsigned)queries, (unsigned)kv_heads), 256, kTcSmem, (cudaStream_t)stream>>>(
            q, (const __half*)k, (const __half*)v, query_positions, key_positions, selected_blocks, query_heads, kv_heads, keys,
            top_k, ids_stride, out);
    } else
#endif
    if (!no_group_h && queries >= 16 && query_heads / kv_heads <= 32) {
        attention_group_kernel<__half><<<dim3((unsigned)queries, (unsigned)kv_heads), 32 * (query_heads / kv_heads), 0, (cudaStream_t)stream>>>(
            q, (const __half*)k, (const __half*)v, query_positions, key_positions, selected_blocks, query_heads, kv_heads, keys,
            block_size, top_k, ids_stride, out);
    } else if (rows <= kSplitMaxRows && block_size <= kSplitMaxBlockSize && top_k <= kSplitMaxTopK) {
        static float* partial = nullptr;
        if (!partial)
            check(cudaMalloc(&partial, (size_t)kSplitMaxRows * kSplitMaxTopK * kPartialStride * sizeof(float)), "split attention scratch");
        attention_split_partial_kernel<__half><<<dim3((unsigned)rows, (unsigned)top_k), kSplitThreads, 0, (cudaStream_t)stream>>>(
            q, (const __half*)k, (const __half*)v, query_positions, key_positions, selected_blocks, query_heads, kv_heads,
            keys, block_size, ids_stride, partial);
        attention_split_merge_kernel<<<rows, 32, 0, (cudaStream_t)stream>>>(partial, top_k, out);
    } else {
        attention_d128_kernel<__half><<<rows, kAttThreads, 0, (cudaStream_t)stream>>>(q, (const __half*)k, (const __half*)v,
            query_positions, key_positions, selected_blocks, query_heads, kv_heads, keys, block_size, top_k, ids_stride, out);
    }
    check(cudaGetLastError(), "sparse attention (half) launch");
}

} // namespace strata::kernels
