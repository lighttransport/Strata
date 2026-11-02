// src/minimax/m3_engine.cu - strata-m3: the engine path for MiniMax M3 / M3.1 Flash (GGUF architecture minimax-m3).
// A driver of its own beside the other families' (their paths stay untouched), built from Strata's kernels:
//
//   * weights straight from the GGUF: the dense matrices (attention, the three leading dense FFNs, the shared
//     experts, the head) in their GGUF blocks on the GPU (native MMVQ, 8 columns per launch); the 1-D tensors and the
//     F32 projections (the MSA indexer, the router) as f32; the token embedding in its blocks (iq_embed_rows);
//   * the routed experts stay in the mapped GGUF in host memory and are copied into a VRAM slot cache (LRU) when a
//     layer routes to them; each expert runs as three MMVQ products over the rows routed to it;
//   * the plain pre-norm residual: x += attn(rms(x) * attn_norm); x += ffn(rms(x) * ffn_norm);
//   * attention: per-head Q / K RMS norm, NEOX RoPE over the first rope.dimension_count dims, 64 query heads over
//     4 KV heads of 128; MiniMax Sparse Attention on every layer (the HF config's sparse_attention_freq is 1 on all
//     60): the indexer (4 heads of 128, one per GQA group, normed and roped like q / k) scores each 128-token block by
//     its max q.k, the current block is forced, the top 16 blocks are attended (kernels/minimax_msa.hpp);
//   * FFN: swigluoai (min(g, 7) * sigmoid(1.702 * min(g, 7)) * (1 + clamp(u, -7, 7))) for the dense layers, the
//     routed experts and the shared expert; the router is sigmoid, exp_probs_b added for the choice only, the 4 chosen
//     weights normalized and scaled by expert_weights_scale (2.0);
//   * the prompt in chunks of --chunk tokens (default: the context, so a prompt reads each layer's experts once),
//     then greedy decoding one token at a time.  K / V and the indexer keys are kept in f32.
//
// --preload-experts (or STRATA_M3_PRELOAD=1): the routed experts are read once at load, with many threads and large
// sequential reads, into one block of process memory (page-locked when the driver allows), and a slot miss is then one
// host-to-device copy from that block. Without it a miss reads the memory-mapped GGUF, whose page faults measured
// 0.12 GB/s on the V100 node (cold sequential reads there reach 0.42 GB/s, pinned H2D 12.3 GB/s).
//
// Modes: --tokens IDS [--max-new N] [--dump-logits F].  STRATA_M3_TRACE=1 prints each layer's last-row hidden
// statistics; STRATA_M3_PREFAULT=1 reads every expert page once at load with --prefault-threads threads.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/host/glm_env.h"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/minimax_msa.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/minimax/engine.hpp"
#if defined(STRATA_M3_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda_profiler_api.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cerrno>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

namespace K = strata::kernels;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-m3: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

#if defined(__HIP_PLATFORM_AMD__)
constexpr auto kGemmAlgo = CUBLAS_GEMM_DEFAULT;
#else
constexpr auto kGemmAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#endif
constexpr int kThreads = 256;
constexpr int kHeadDim = 128;
constexpr int kMaxExperts = 512;
constexpr int kMaxWin = 8;   // tokens of a decode / verify window
const char* const kArch = "minimax-m3.";

// ------------------------------------------------------------------------------------------------ small kernels
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}

/// Per-head RMS norm (one 128-wide weight for every head) then NEOX RoPE over the first n_rot dims, for T rows of
/// nh heads of src [T][nh][128] at positions pos0 + t.  dst row: (pos0 + t) * nh + h when by_pos (a cache), else
/// t * nh + h.  grid (nh, T), block 128.
__device__ __forceinline__ void store_out(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store_out(__half* p, size_t i, float v) { p[i] = __float2half(v); }

template <class OT>
__global__ void k_head_prep(const float* src, const float* w, OT* dst, int nh, int n_rot, float theta_scale,
                            float eps, int pos0, int by_pos) {
    __shared__ float sh[kHeadDim];
    __shared__ float buf[kHeadDim];
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = pos0 + t;
    const float v0 = src[((size_t) t * nh + h) * kHeadDim + i];
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) kHeadDim + eps);
    buf[i] = v0 * r * w[i];
    __syncthreads();
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const float theta = (float) pos * powf(theta_scale, (float) p);   // llama.cpp's rope: pos * base^(-2p/n_rot)
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    const size_t row = by_pos ? (size_t) pos * nh + h : (size_t) t * nh + h;
    store_out(dst, row * kHeadDim + i, v);
}

__global__ void k_f32_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// swigluoai: g = min(gate, limit), u = clamp(up, -limit, limit), out = g * sigmoid(alpha * g) * (1 + u).
__global__ void k_swiglu_oai(const float* gate, const float* up, float* out, int64_t n, float alpha, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);
    const float u = fmaxf(fminf(up[i], limit), -limit);
    out[i] = g / (1.0f + expf(-alpha * g)) * (1.0f + u);
}

__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/// rows idx.t[i] of src (n floats each) -> dst rows i, for up to kMaxWin rows given by value
struct RowIdx { int t[kMaxWin]; float w[kMaxWin]; int n; };
__global__ void k_gather_rows_i(const float* src, float* dst, int n, RowIdx ix) {
    const float* s = src + (size_t) ix.t[blockIdx.x] * n;
    float* d = dst + (size_t) blockIdx.x * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) d[j] = s[j];
}
/// the q8_1 images of rows ix.t[i] of src (rows of `bytes` bytes, a multiple of 4) back to back into dst: the quantization is per 32 values,
/// so these are the bytes quantizing the gathered float rows would write
__global__ void k_gather_q8(const uint8_t* src, uint8_t* dst, int bytes, RowIdx ix) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src + (size_t) ix.t[blockIdx.x] * bytes);
    uint32_t* d = reinterpret_cast<uint32_t*>(dst + (size_t) blockIdx.x * bytes);
    for (int j = threadIdx.x; j < bytes / 4; j += blockDim.x) d[j] = s[j];
}
/// The resident experts of a layer, each with its own output rows (job j's entry i = row i of eo + j * eo_stride): the adds into
/// `moe` in job order, entry order within a job - the same sequence of operations on every element as the one-job-at-a-time
/// k_scatter_add_i calls (a token is in a job at most once)
constexpr int kJobMax = 16;   // jobs one k_scatter_jobs launch adds (its argument struct is passed by value)
constexpr int kJobCap = 64;   // top_k x kMaxWin: every distinct expert a window layer can need
struct JobList { int n; int nt[kJobMax]; int tok[kJobMax][kMaxWin]; float w[kJobMax][kMaxWin]; const float* eo[kJobMax]; };
__global__ void k_scatter_jobs(float* moe, int N, JobList jl) {
    const int t = blockIdx.x;
    for (int e = threadIdx.x; e < N; e += blockDim.x) {
        float acc = moe[(size_t) t * N + e];
        for (int j = 0; j < jl.n; ++j)
            for (int i = 0; i < jl.nt[j]; ++i)
                if (jl.tok[j][i] == t) acc += jl.w[j][i] * jl.eo[j][(size_t) i * N + e];
        moe[(size_t) t * N + e] = acc;
    }
}
/// out rows ix.t[i] += ix.w[i] * src row i
__global__ void k_scatter_add_i(float* out, const float* src, int n, RowIdx ix) {
    float* d = out + (size_t) ix.t[blockIdx.x] * n;
    const float* s = src + (size_t) blockIdx.x * n;
    const float w = ix.w[blockIdx.x];
    for (int j = threadIdx.x; j < n; j += blockDim.x) d[j] += w * s[j];
}

/// argmax of each of `rows` rows of n floats (one block of 1024 threads per row; ties go to the lowest index)
__global__ void k_argmax_rows(const float* x, int n, int32_t* out) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    const float* r = x + (size_t) blockIdx.x * n;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = r[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if ((int) threadIdx.x < s2) {
            const float ov = sv[threadIdx.x + s2];
            const int oi = si[threadIdx.x + s2];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = si[0];
}

/// y[i] += a * x[i]
__global__ void k_axpy(float* y, const float* x, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

/// argmax of n floats, one block of 1024 threads; ties go to the lowest index (like std::max_element).
__global__ void k_argmax1(const float* x, int n, int32_t* out) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s];
            const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = si[0];
}

__global__ void k_iota(int32_t* p, int n, int base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = base + i;
}

/// dst[i] = src[idx[i]] (rows of n floats), grid (rows), block 256.
__global__ void k_gather_rows(const float* src, const int32_t* idx, float* dst, int n) {
    const float* s = src + (size_t) idx[blockIdx.x] * n;
    float* d = dst + (size_t) blockIdx.x * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) d[j] = s[j];
}

/// out[idx[i]] += w[i] * src[i] (rows of n floats), grid (rows), block 256.  An expert has each token once, so the
/// rows of one launch never share a destination.
__global__ void k_scatter_add(float* out, const float* src, const int32_t* idx, const float* w, int n) {
    const float wi = w[blockIdx.x];
    float* d = out + (size_t) idx[blockIdx.x] * n;
    const float* s = src + (size_t) blockIdx.x * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) d[j] += wi * s[j];
}

/// y [T][n_out] = W [n_out][n_in] x [T][n_in] (f32), a warp per output row: grid (ceil(n_out / 8), T), block 256.
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = blockIdx.x * 8 + warp, t = blockIdx.y;
    if (r >= n_out) return;
    const float* wr = W + (size_t) r * n_in;
    const float* xr = x + (size_t) t * n_in;
    float a = 0.f;
    for (int i = lane; i < n_in; i += 32) a += wr[i] * xr[i];
    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0) y[(size_t) t * n_out + r] = a;
}

/// The same product for a few rows (T <= 8) with a block per output row: the router and the indexer have only 128-512
/// output rows, which a warp-per-row launch spreads over 16-64 blocks of a 80-SM GPU.  n_in a multiple of 4, rows and x
/// 16-byte aligned.  Each thread strides over float4s; the T dot products share the W loads.
__global__ void k_gemv_f32_blk(const float* W, const float* x, float* y, int n_in, int n_out, int T) {
    __shared__ float red[8][8];
    const int r = blockIdx.x;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t) r * n_in);
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int i = threadIdx.x; i < n_in / 4; i += blockDim.x) {
        const float4 w = wr[i];
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (t < T) {
                const float4 v = reinterpret_cast<const float4*>(x + (size_t) t * n_in)[i];
                acc[t] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) red[warp][t] = a;
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float a = 0.f;
        for (int w = 0; w < (int) (blockDim.x / 32); ++w) a += red[w][threadIdx.x];
        y[(size_t) threadIdx.x * n_out + r] = a;
    }
}

/// The router for one token per block (E <= 512 experts, block 256): p = sigmoid(logit); the top K by p + bias (ties:
/// the lower id), weights p normalized over the K (when norm) and times scale.
__global__ void k_route(const float* lg, const float* bias, int32_t* ids, float* wts, int E, int K_, float scale,
                        int norm, int32_t* xids = nullptr, float* xwts = nullptr, int NX = 0) {
    __shared__ float p[kMaxExperts], sel[kMaxExperts];
    const int t = blockIdx.x;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        p[e] = 1.0f / (1.0f + expf(-lg[(size_t) t * E + e]));
        sel[e] = p[e] + bias[e];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    float sum = 0.f;
    for (int k = 0; k < K_; ++k) {
        int bi = 0;
        float best = -INFINITY;
        for (int e = 0; e < E; ++e)
            if (sel[e] > best) {
                best = sel[e];
                bi = e;
            }
        sel[bi] = -INFINITY;
        ids[t * K_ + k] = bi;
        wts[t * K_ + k] = p[bi];
        sum += p[bi];
    }
    sum = fmaxf(sum, 6.103515625e-5f);   // llama.cpp clamps the weight sum the same way
    for (int k = 0; k < K_; ++k) {
        float w = wts[t * K_ + k];
        if (norm) w /= sum;
        wts[t * K_ + k] = w * scale;
    }
    // the next-best candidates (cache-aware substitution), weighted as if they had been selected
    for (int x = 0; x < NX; ++x) {
        int bi = 0;
        float best = -INFINITY;
        for (int e = 0; e < E; ++e)
            if (sel[e] > best) {
                best = sel[e];
                bi = e;
            }
        sel[bi] = -INFINITY;
        xids[t * NX + x] = bi;
        xwts[t * NX + x] = (norm ? p[bi] / sum : p[bi]) * scale;
    }
}

/// STRATA_M3_PROF=1: host time spent enqueuing each part of a decode step (the GPU runs behind it).
struct Prof {
    bool on = false;
    std::vector<std::pair<const char*, double>> acc;
    void add(const char* n, double ms) {
        for (auto& a : acc)
            if (a.first == n) { a.second += ms; return; }
        acc.emplace_back(n, ms);
    }
};
Prof g_prof;
struct Scope {
    const char* n;
    Clock::time_point t;
    explicit Scope(const char* name) : n(name) { if (g_prof.on) t = Clock::now(); }
    ~Scope() { if (g_prof.on) g_prof.add(n, ms_since(t)); }
};

unsigned grid1(int64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct FileRef {
    std::string path;
    uint64_t off = 0;
};

struct Layer {
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    const float *iq_w = nullptr, *ik_w = nullptr, *iq_norm = nullptr, *ik_norm = nullptr;
    Mat wq, wk, wv, wo;
    bool dense = false;
    Mat fg, fu, fd;   // the leading dense FFN
    Mat sg, su, sd;   // the shared expert
    const float *router = nullptr, *bias = nullptr;
    // the routed experts in the mapped GGUF: expert e's gate / up at src_g / src_u + e * gsz, down at src_d + e * dsz
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t gsz = 0, dsz = 0;
    int gu_type = -1, d_type = -1;
    // where the three expert tensors live in their files (for the preload's pread)
    FileRef file_g, file_u, file_d;
    // the preloaded pack: n_expert blobs [gate | up | down] of 2 * gsz + dsz bytes, nullptr = read the mapped file
    uint8_t* pack = nullptr;
};

struct Model {
    std::shared_ptr<strata::GgufModel> file;
    int n_layers = 0, N = 0, H = 0, HKV = 0, n_vocab = 0, n_rot = 0;
    int n_expert = 0, top_k = 0, FF = 0, FS = 0, DFF = 0;
    int idx_heads = 0, idx_top_k = 0, idx_block = 0, idx_local = 0;
    float eps = 1e-6f, rope_base = 1e7f, w_scale = 1.f, alpha = 1.702f, limit = 7.f;
    bool w_norm = true;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    size_t dev_bytes = 0;
    // the preloaded expert pack (one block), see preload()
    uint8_t* pack_base = nullptr;
    size_t pack_bytes = 0, pack_pinned = 0;
    double pack_read_s = 0, pack_pin_s = 0;
};

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

int64_t meta_int(const strata::GgufFile& g, const std::string& key) {
    const strata::MetaValue* v = g.get(key);
    if (!v || !v->is_num()) throw std::runtime_error("missing or invalid " + key);
    return (int64_t) v->u;
}
double meta_num(const strata::GgufFile& g, const std::string& key, double fallback, bool required) {
    const strata::MetaValue* v = g.get(key);
    if (!v) {
        if (required) throw std::runtime_error("missing " + key);
        return fallback;
    }
    if (!v->is_num()) throw std::runtime_error("invalid " + key);
    return v->num();
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    const strata::GgufFile& meta = gm.meta();
    if (strata::model_family(meta) != strata::ModelFamily::MiniMaxM3) {
        err = "strata-m3 runs MiniMax M3 / M3.1 (minimax-m3) files only";
        return false;
    }
    err = strata::check_architecture_minimax_m3(meta);
    if (err.empty()) err = strata::check_minimax_m3_indexers(gm);
    if (!err.empty()) return false;
    const std::string p = kArch;
    m.n_layers = (int) meta_int(meta, p + "block_count");
    m.N = (int) meta_int(meta, p + "embedding_length");
    m.H = (int) meta_int(meta, p + "attention.head_count");
    m.HKV = (int) meta_int(meta, p + "attention.head_count_kv");
    m.n_rot = (int) meta_int(meta, p + "rope.dimension_count");
    m.n_expert = (int) meta_int(meta, p + "expert_count");
    m.top_k = (int) meta_int(meta, p + "expert_used_count");
    m.FF = (int) meta_int(meta, p + "expert_feed_forward_length");
    m.FS = m.FF * (int) meta_int(meta, p + "expert_shared_count");
    m.DFF = (int) meta_int(meta, p + "feed_forward_length");
    m.idx_heads = (int) meta_int(meta, p + "attention.indexer.head_count");
    m.idx_top_k = (int) meta_int(meta, p + "attention.indexer.top_k");
    m.idx_block = (int) meta_int(meta, p + "attention.indexer.block_size");
    m.idx_local = (int) meta_int(meta, p + "attention.indexer.local_blocks");
    const int dense_lead = (int) meta_int(meta, p + "leading_dense_block_count");
    m.eps = (float) meta_num(meta, p + "attention.layer_norm_rms_epsilon", 0, true);
    m.rope_base = (float) meta_num(meta, p + "rope.freq_base", 1e7, false);
    m.w_scale = (float) meta_num(meta, p + "expert_weights_scale", 1.0, false);
    // not stored by the released GGUFs: the HF config's swiglu_alpha / swiglu_limit (ModelDescriptor reads the same)
    m.alpha = (float) meta_num(meta, p + "swiglu_alpha", 1.702, false);
    m.limit = (float) meta_num(meta, p + "swiglu_limit", 7.0, false);
    if (const strata::MetaValue* v = meta.get(p + "expert_weights_norm")) m.w_norm = v->u != 0;
    if (meta_int(meta, p + "expert_gating_func") != 2) {
        err = "only the sigmoid router (expert_gating_func 2) is implemented";
        return false;
    }
    if (m.HKV != m.idx_heads || m.H % m.HKV != 0 || m.n_rot < 2 || m.n_rot > kHeadDim || m.n_rot % 2 ||
        m.n_expert > kMaxExperts || m.top_k > 16 || m.idx_top_k * m.idx_block > 4096 || m.idx_local < 1 ||
        meta_int(meta, p + "attention.key_length") != kHeadDim || m.N % 256 || m.FF % 256 || m.DFF % 256) {
        err = "a geometry this driver's kernels do not take";
        return false;
    }
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) throw std::runtime_error("missing " + name);
        return t;
    };
    auto f32 = [&](const std::string& name, uint64_t elements) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        if (t->type != 0 || (uint64_t) t->elements() != elements)
            throw std::runtime_error(name + " is " + t->type_name() + " of " + std::to_string(t->elements()) +
                                     " values; expected F32 of " + std::to_string(elements));
        float* d = nullptr;
        CK(cudaMalloc(&d, elements * 4));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), elements * 4, cudaMemcpyHostToDevice));
        m.dev_bytes += elements * 4;
        return d;
    };
    auto mat = [&](const std::string& name, int n_in, int n_out) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (t->shape.size() != 2 || w.n_in != n_in || w.n_out != n_out)
            throw std::runtime_error(name + " has an unexpected shape");
        if (!K::native_mmvq_supported(w.type))
            throw std::runtime_error(name + " is " + t->type_name() + ", which the native MMVQ does not take");
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb + 256));
        CK(cudaMemset(d + nb, 0, 256));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.dev_bytes += nb;
        w.d = d;
        return w;
    };
    auto L = [](int l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    const int N = m.N, QD = m.H * kHeadDim, KD = m.HKV * kHeadDim, ID = m.idx_heads * kHeadDim;
    m.layers.resize((size_t) m.n_layers);
    for (int l = 0; l < m.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        y.attn_norm = f32(L(l, "attn_norm.weight"), N);
        y.ffn_norm = f32(L(l, "ffn_norm.weight"), N);
        y.q_norm = f32(L(l, "attn_q_norm.weight"), kHeadDim);
        y.k_norm = f32(L(l, "attn_k_norm.weight"), kHeadDim);
        y.iq_w = f32(L(l, "indexer.q_proj.weight"), (uint64_t) N * ID);
        y.ik_w = f32(L(l, "indexer.k_proj.weight"), (uint64_t) N * kHeadDim);
        y.iq_norm = f32(L(l, "indexer.q_norm.weight"), kHeadDim);
        y.ik_norm = f32(L(l, "indexer.k_norm.weight"), kHeadDim);
        y.wq = mat(L(l, "attn_q.weight"), N, QD);
        y.wk = mat(L(l, "attn_k.weight"), N, KD);
        y.wv = mat(L(l, "attn_v.weight"), N, KD);
        y.wo = mat(L(l, "attn_output.weight"), QD, N);
        y.dense = l < dense_lead;
        if (y.dense) {
            y.fg = mat(L(l, "ffn_gate.weight"), N, m.DFF);
            y.fu = mat(L(l, "ffn_up.weight"), N, m.DFF);
            y.fd = mat(L(l, "ffn_down.weight"), m.DFF, N);
            continue;
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"), (uint64_t) N * m.n_expert);
        y.bias = f32(L(l, "exp_probs_b.bias"), (uint64_t) m.n_expert);
        y.sg = mat(L(l, "ffn_gate_shexp.weight"), N, m.FS);
        y.su = mat(L(l, "ffn_up_shexp.weight"), N, m.FS);
        y.sd = mat(L(l, "ffn_down_shexp.weight"), m.FS, N);
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type || !K::native_mmvq_supported((int) tg->type) ||
            !K::native_mmvq_supported((int) td->type) || tg->shape.size() != 3 || td->shape.size() != 3 ||
            tg->shape[0] != (uint64_t) N || tg->shape[1] != (uint64_t) m.FF || tg->shape[2] != (uint64_t) m.n_expert ||
            tu->shape != tg->shape || td->shape[0] != (uint64_t) m.FF || td->shape[1] != (uint64_t) N ||
            td->shape[2] != (uint64_t) m.n_expert) {
            err = L(l, "experts: types ") + tg->type_name() + "/" + tu->type_name() + "/" + td->type_name() +
                  " or shapes this driver does not take";
            return false;
        }
        y.gu_type = (int) tg->type;
        y.d_type = (int) td->type;
        y.gsz = (size_t) m.FF * row_bytes(y.gu_type, N);
        y.dsz = (size_t) N * row_bytes(y.d_type, m.FF);
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.file_g = {gm.shard(sg).path(), gm.shard(sg).data_start() + tg->offset};
        y.file_u = {gm.shard(su).path(), gm.shard(su).data_start() + tu->offset};
        y.file_d = {gm.shard(sd).path(), gm.shard(sd).data_start() + td->offset};
    }
    m.out_norm = f32("output_norm.weight", N);
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("output.weight", s);
        m.n_vocab = (int) t->shape[1];
    }
    m.head = mat("output.weight", N, m.n_vocab);
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        if (t->shape[0] != (uint64_t) N || !K::embed_type_supported((int) t->type))
            throw std::runtime_error(std::string("token_embd.weight is ") + t->type_name() + ", not a supported table");
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, N);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        // The 0.85 GB table stays in pinned host memory: the embedding kernel reads only the rows of the tokens it
        // looks up (4 KB each, over PCIe), and the VRAM it would hold goes to expert slots.  STRATA_M3_EMBD_GPU=1
        // keeps it on the device as before.
        static const bool embd_gpu = strata_env_truthy_one("STRATA_M3_EMBD_GPU") != 0;
        if (embd_gpu) {
            CK(cudaMalloc(&d, nb));
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            m.dev_bytes += nb;
        } else {
            CK(cudaHostAlloc((void**) &d, nb, cudaHostAllocMapped));
            std::memcpy(d, gm.shard(s).tensor_data(*t), nb);
        }
        m.embd = d;
    }
    std::fprintf(stderr, "strata-m3: %d layers, %.2f GiB of dense weights on the GPU in %.1f s; routed experts stay in "
                 "the mapped file\n", m.n_layers, (double) m.dev_bytes / (1 << 30), ms_since(t0) / 1000.0);
    return true;
}

#if defined(__linux__)
/// The first logical CPU of every sibling group, in id order, among the CPUs this process may run on.
std::vector<int> physical_cpus() {
    std::vector<int> out;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return out;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        char path[128];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        int first = -1;
        if (std::fscanf(f, "%d", &first) != 1) first = c;
        std::fclose(f);
        if (first == c) out.push_back(c);
    }
    return out;
}
#else
std::vector<int> physical_cpus() { return {}; }
#endif

/// Read every expert page once (many threads), so the first prompt does not wait on the file system.
void prefault(const Model& m, int threads) {
    const auto t0 = Clock::now();
    struct Range { const uint8_t* p; size_t n; };
    std::vector<Range> ranges;
    for (const Layer& y : m.layers) {
        if (y.dense) continue;
        for (int e = 0; e < m.n_expert; ++e) {
            ranges.push_back({y.src_g + (size_t) e * y.gsz, y.gsz});
            ranges.push_back({y.src_u + (size_t) e * y.gsz, y.gsz});
            ranges.push_back({y.src_d + (size_t) e * y.dsz, y.dsz});
        }
    }
    std::atomic<size_t> next{0};
    std::atomic<unsigned long long> sink{0}, bytes{0};
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i)
        pool.emplace_back([&]() {
            unsigned long long acc = 0;
            for (size_t r; (r = next.fetch_add(1)) < ranges.size();) {
                const volatile uint8_t* q = ranges[r].p;
                for (size_t o = 0; o < ranges[r].n; o += 4096) acc += q[o];
                bytes += ranges[r].n;
            }
            sink += acc;
        });
    for (auto& t : pool) t.join();
    const double s = ms_since(t0) / 1000.0;
    std::fprintf(stderr, "strata-m3: prefault %.1f GiB of experts in %.1f s (%.2f GiB/s, %d threads)\n",
                 (double) bytes.load() / (1 << 30), s, (double) bytes.load() / (1 << 30) / s, threads);
    (void) sink.load();
}

#if !defined(_WIN32)
/// MemAvailable from /proc/meminfo in bytes (0 when unreadable).
size_t mem_available_bytes() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    std::fclose(f);
    return kb * 1024;
}

/// Reads every routed expert into one block of memory, laid out per layer as n_expert blobs [gate | up | down] (the
/// slot layout), with `threads` readers issuing large sequential preads (the file system caps one node near the same
/// rate for any stream count, so the thread count only has to keep it busy); each range is dropped from the page
/// cache after it is read so the cache does not hold a second copy. Then page-locks it layer by layer when
/// `pin` (a layer the driver refuses stays pageable). Returns false with `err` set when it cannot be done; the
/// engine then keeps reading the mapped file.
bool preload(Model& m, int threads, bool pin, bool interleave, std::string& err) {
    constexpr size_t kPage = 4096;
    auto up = [](size_t v) { return (v + kPage - 1) / kPage * kPage; };
    size_t total = 0;   // each layer's block starts on a page, so it can be page-locked on its own
    for (const Layer& y : m.layers)
        if (!y.dense) total += up((size_t) m.n_expert * (2 * y.gsz + y.dsz));
    const size_t avail = mem_available_bytes();
    const size_t headroom = (size_t) 12 << 30;   // the process itself, the page cache the dense load used, the driver
    if (avail && avail < total + headroom) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "the expert pack needs %.1f GiB and %.1f GiB are available (keeping %.0f GiB free)",
                      (double) total / (1 << 30), (double) avail / (1 << 30), (double) headroom / (1 << 30));
        err = buf;
        return false;
    }
    void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) {
        err = "mmap of the expert pack failed";
        return false;
    }
#ifdef MADV_HUGEPAGE
    madvise(base, total, MADV_HUGEPAGE);   // best effort: fewer page-table entries to pin
#endif
#if defined(__linux__) && defined(SYS_mbind)
    if (interleave) {
        // Spread the pack's pages over every NUMA node (bound to the pack only).  Measured on the 2-socket, 4-node V100
        // host: the CPU expert kernels reach 65 GB/s on interleaved memory and 31 GB/s when the pages sit on one node.
        unsigned long mask = 0;
        int nodes = 0;
        if (FILE* f = std::fopen("/sys/devices/system/node/online", "r")) {   // "0-3" or "0,2-3"
            char buf[128] = {};
            if (std::fgets(buf, sizeof buf, f)) {
                for (char* tok = std::strtok(buf, ",\n"); tok; tok = std::strtok(nullptr, ",\n")) {
                    int a = 0, b = 0;
                    if (std::sscanf(tok, "%d-%d", &a, &b) == 2) { for (int n = a; n <= b && n < 64; ++n) mask |= 1ul << n; }
                    else if (std::sscanf(tok, "%d", &a) == 1 && a < 64) mask |= 1ul << a;
                }
            }
            std::fclose(f);
        }
        for (unsigned long mm = mask; mm; mm >>= 1) nodes += (int) (mm & 1);
        if (nodes > 1) {
            constexpr long kMplInterleave = 3;   // MPOL_INTERLEAVE
            if (syscall(SYS_mbind, base, total, kMplInterleave, &mask, (unsigned long) 64, 0u) != 0)
                std::fprintf(stderr, "strata-m3: mbind interleave failed (%s); the pack keeps the default placement\n", std::strerror(errno));
            else
                std::fprintf(stderr, "strata-m3: expert pack interleaved over %d NUMA nodes\n", nodes);
        }
    }
#endif
    struct Item { const FileRef* f; size_t len; uint8_t* dst; };
    std::vector<Item> items;
    items.reserve((size_t) m.n_layers * m.n_expert * 3);
    uint8_t* at = (uint8_t*) base;
    for (Layer& y : m.layers) {
        if (y.dense) continue;
        y.pack = at;
        for (int e = 0; e < m.n_expert; ++e) {
            items.push_back({&y.file_g, y.gsz, at});
            items.push_back({&y.file_u, y.gsz, at + y.gsz});
            items.push_back({&y.file_d, y.dsz, at + 2 * y.gsz});
            at += 2 * y.gsz + y.dsz;
        }
        at = (uint8_t*) base + up((size_t) (at - (uint8_t*) base));
    }
    // each item's file offset: tensor start + expert * its length (gate / up / down differ in length)
    std::vector<uint64_t> offs(items.size());
    {
        size_t i = 0;
        for (Layer& y : m.layers) {
            if (y.dense) continue;
            for (int e = 0; e < m.n_expert; ++e, i += 3) {
                offs[i] = y.file_g.off + (uint64_t) e * y.gsz;
                offs[i + 1] = y.file_u.off + (uint64_t) e * y.gsz;
                offs[i + 2] = y.file_d.off + (uint64_t) e * y.dsz;
            }
        }
    }
    std::unordered_map<std::string, int> fds;
    for (const Item& it : items)
        if (!fds.count(it.f->path)) {
            const int fd = ::open(it.f->path.c_str(), O_RDONLY);
            if (fd < 0) {
                err = "cannot open " + it.f->path;
                for (auto& kv : fds) ::close(kv.second);
                munmap(base, total);
                for (Layer& y : m.layers) y.pack = nullptr;
                return false;
            }
            fds[it.f->path] = fd;
        }
    size_t read_total = 0;
    for (const Item& it : items) read_total += it.len;
    std::atomic<size_t> next{0}, done_bytes{0};
    std::atomic<bool> failed{false};
    const auto t0 = Clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&]() {
            for (size_t i; (i = next.fetch_add(1)) < items.size() && !failed.load();) {
                const Item& it = items[i];
                const int fd = fds.at(it.f->path);
                size_t got = 0;
                while (got < it.len) {
                    const ssize_t r = ::pread(fd, it.dst + got, std::min<size_t>(it.len - got, (size_t) 16 << 20),
                                              (off_t) (offs[i] + got));
                    if (r <= 0) {
                        failed = true;
                        break;
                    }
                    got += (size_t) r;
                }
                if (got == it.len) {
                    ::posix_fadvise(fd, (off_t) offs[i], (off_t) it.len, POSIX_FADV_DONTNEED);
                    done_bytes += it.len;
                }
            }
        });
    size_t shown = 0;
    while (next.load() < items.size() && !failed.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        const size_t d = done_bytes.load();
        if (d * 10 / read_total > shown) {
            shown = d * 10 / read_total;
            std::fprintf(stderr, "strata-m3: preload %.0f %% (%.1f GiB, %.2f GB/s)\n", 100.0 * (double) d / (double) read_total,
                         (double) d / (1 << 30), (double) d / 1e9 / (ms_since(t0) / 1000.0));
        }
    }
    for (auto& t : pool) t.join();
    for (auto& kv : fds) ::close(kv.second);
    if (failed.load() || done_bytes.load() != read_total) {
        err = "a read of the expert pack failed";
        munmap(base, total);
        for (Layer& y : m.layers) y.pack = nullptr;
        return false;
    }
    m.pack_base = (uint8_t*) base;
    m.pack_bytes = total;
    m.pack_read_s = ms_since(t0) / 1000.0;
    std::fprintf(stderr, "strata-m3: preloaded %.1f GiB of experts in %.1f s (%.2f GB/s, %d threads)\n",
                 (double) total / (1 << 30), m.pack_read_s, (double) total / 1e9 / m.pack_read_s, threads);
    if (pin) {
        const auto t1 = Clock::now();
        size_t refused = 0;
        for (Layer& y : m.layers) {   // one registration per layer: a blob never straddles two of them
            if (y.dense) continue;
            const size_t n = up((size_t) m.n_expert * (2 * y.gsz + y.dsz));
            if (cudaHostRegister(y.pack, n, cudaHostRegisterPortable) == cudaSuccess) m.pack_pinned += n;
            else {
                ++refused;
                (void) cudaGetLastError();
            }
        }
        m.pack_pin_s = ms_since(t1) / 1000.0;
        std::fprintf(stderr, "strata-m3: page-locked %.1f of %.1f GiB in %.1f s%s\n", (double) m.pack_pinned / (1 << 30),
                     (double) total / (1 << 30), m.pack_pin_s,
                     refused ? " (the driver refused some layers; they stay pageable, ~5 GB/s)" : "");
    }
    return true;
}
#endif

#if defined(STRATA_M3_CPU_POOL)
#include "par.hpp"
#endif

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    cublasHandle_t blas = nullptr;
    cudaStream_t st2 = nullptr;          // the MSA indexer's projections run beside the q/k/v matvecs (decode)
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    bool two_streams = true;
    int ctx = 8192, tmax = 8192;
    int pos = 0;
    bool trace = false;
    // activations [tmax][...]
    float *x, *h, *ao, *qraw, *qb, *att, *kraw, *vraw, *iqraw, *iqb, *ikraw, *g, *u, *eh, *eo, *moe, *lg, *wts, *scores,
        *logits;
    int32_t *ids, *sel, *tok, *qpos, *kpos, *e_tok;
    int32_t* tok_host = nullptr;
    float* e_w;
    void* q8 = nullptr;
    std::vector<float*> kc, vc, ikc;   // per layer: [ctx][HKV][128], [ctx][HKV][128], [ctx][128] (FP16 elements when kv_half)
    // --pf-gemm (opt-in): prompt chunks through FP16 tensor-core GEMMs on weights dequantized into a scratch, instead of
    // the q8_1 column-group matvecs (which read every weight once per 8 tokens)
    bool pf_gemm = false;
    bool pfprof = strata_env_truthy_one("STRATA_M3_PFPROF") != 0;   // prefill phase timers (syncs between phases)
    double pt_deq = 0, pt_mm = 0, pt_conv = 0, pt_attn = 0, pt_dense = 0, pt_route = 0, pt_copy = 0, pt_gemm = 0, pt_other = 0;
    Clock::time_point pt_last, pt_mark;
    std::vector<std::pair<const char*, double>> pf_marks;
    void ptick(double& acc) {
        if (!pfprof) return;
        CK(cudaStreamSynchronize(st));
        const auto n = Clock::now();
        acc += std::chrono::duration<double, std::milli>(n - pt_last).count();
        pt_last = n;
    }
    int pf_min_t = 64, pf_min_rows = 32;
    uint16_t *wscr = nullptr, *ain16 = nullptr;
    bool pf(int T, int min_t) const { return pf_gemm && T >= min_t; }
    /// out [T][n_out] = in [T][n_in] . W^T, W the GGUF matrix (n_out rows of n_in)
    void gemm_w(int type, const void* w, int n_in, int n_out, const float* in, float* out, int T, bool same_in = false) {
        if (pfprof) { CK(cudaStreamSynchronize(st)); pt_last = Clock::now(); }
        // same_in: the previous call converted this very input (nothing wrote it since): reuse its FP16 image
        if (!same_in) k_f32_to_f16<<<grid1((int64_t) T * n_in), kThreads, 0, st>>>(in, reinterpret_cast<__half*>(ain16), (int64_t) T * n_in);
        ptick(pt_conv);
        const bool iq = type == 16 || type == 17 || type == 18 || type == 19 || type == 20 || type == 21 || type == 22 ||
                        type == 23 || type == 29;
        if (iq) K::iq_dequant_f16_staged(type, w, (int64_t) n_in * n_out, wscr, st);
        else K::dequant_f16(type, w, 0, n_out, n_in, wscr, st);
        ptick(pt_deq);
        const float one = 1.f, zero = 0.f;
        if (cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, wscr, CUDA_R_16F, n_in, ain16, CUDA_R_16F, n_in,
                         &zero, out, CUDA_R_32F, n_out, CUBLAS_COMPUTE_32F, kGemmAlgo) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-m3: cublasGemmEx failed\n");
            std::exit(1);
        }
        ptick(pt_mm);
    }
    void gemm_w(const Mat& w, const float* in, float* out, int T, bool same_in = false) { gemm_w(w.type, w.d, w.n_in, w.n_out, in, out, T, same_in); }
    float* feat = nullptr;             // [rows][layers][N] target hidden states for the DSpark head (null: off)
    std::vector<int> feat_layers;      // layer ids whose INPUT is captured (n_layers: after the last layer)
    bool kv_half = false;              // --kv-f16: K, V and the indexer keys stored as FP16 (a lossy lever: half the cache memory)
    // the routed-expert slot cache
    uint8_t* slots = nullptr;
    // VRAM expert slots in two size classes: most layers' experts are 18.1 MB, three layers' are 21-24.5 MB.  One size for
    // all wasted 22 % of the slot memory, so a small slot fits the common experts and a few large slots serve the others.
    size_t slot_bytes = 0, slot_bytes_big = 0;   // strides of the two classes
    int n_slots = 0, n_small = 0;
    // STRATA_M3_GPU1=<device index>: a second GPU holds extra expert slots (indices n_slots .. n_slots + n_dev1 - 1, small class only).
    // Decode windows run the experts resident there on that GPU (q8_1 image of h by peer copy, results by peer copy into the job's
    // rows); the prefill, the T == 1 step and the large-class layers never use them.
    int dev1 = -1, n_dev1 = 0, dev0 = 0;
    uint8_t* slots1 = nullptr;
    cudaStream_t cst1 = nullptr, sst1[4] = {};
    cudaEvent_t ev_p = nullptr, sev1[4] = {};
    void *q8h1 = nullptr, *jq81 = nullptr, *jqf1 = nullptr;
    float *jgu1 = nullptr, *jeo1 = nullptr;
    long long gpu1_experts = 0;
    // STRATA_M3_GPU1_CHECK=1 (debug, slow): every second-GPU job is also run on GPU 0 from the host copy of the expert, and the output rows compared bit for bit
    bool gpu1_check = strata_env_truthy_one("STRATA_M3_GPU1_CHECK") != 0;
    uint8_t* chk_slot = nullptr; void *chk_q8 = nullptr, *chk_qf = nullptr; float *chk_gu = nullptr, *chk_eo = nullptr;
    long long chk_jobs = 0, chk_bad = 0;
    bool p2p_direct = strata_env_truthy_default("STRATA_M3_GPU1_DIRECT", 0) != 0;   // kernels read the peer's memory over PCIe instead of copy-engine copies (needs peer access; measured slower: 60.9 against 63.4 tok/s in max-mode)
    std::vector<cudaEvent_t> g1ev;   // STRATA_M3_PROF>=3: start / end events of the second GPU's first stream per window layer
    long long g1_layers = 0, g1_jobs = 0;
    int gpu1_admit = strata_env_int_or("STRATA_M3_GPU1_ADMIT", 2);   // at most this many admissions per layer onto second-GPU slots (0: only the warm-up fills them; 2 measured best, docs/MINIMAX_M31_SIM.md)
    bool on_dev1(int s) const { return s >= n_slots; }
    /// Has the admission copy into slot s finished?  A second-GPU slot's event is queried from that GPU's context.
    bool slot_copy_done(int s) {
        cudaError_t r;
        if (on_dev1(s)) { (void) cudaSetDevice(dev1); r = cudaEventQuery(slot_ev[(size_t) s]); (void) cudaSetDevice(dev0); }
        else r = cudaEventQuery(slot_ev[(size_t) s]);
        return r == cudaSuccess;
    }
    size_t blob_of(const Layer& y) const { return 2 * y.gsz + y.dsz; }
    bool big_layer(const Layer& y) const { return blob_of(y) + 4096 > slot_bytes; }
    uint8_t* slot_ptr(int s) const {
        if (s >= n_slots) return slots1 + (size_t) (s - n_slots) * slot_bytes;
        return s < n_small ? slots + (size_t) s * slot_bytes
                           : slots + (size_t) n_small * slot_bytes + (size_t) (s - n_small) * slot_bytes_big;
    }
    /// The least recently used slot of the class that fits this layer's experts.
    /// `any`: the second GPU's slots too (decode-window admission and the warm-up only).
    int victim(const Layer& y, bool any = false) const {
        const int lo = big_layer(y) ? n_small : 0, hi = big_layer(y) ? n_slots : n_small;
        int best = lo + (int) (std::min_element(used.begin() + lo, used.begin() + hi) - (used.begin() + lo));
        if (any && n_dev1 > 0 && !big_layer(y)) {
            const int b1 = n_slots + (int) (std::min_element(used.begin() + n_slots, used.begin() + n_slots + n_dev1) - (used.begin() + n_slots));
            if (used[(size_t) b1] < used[(size_t) best]) best = b1;
        }
        return best;
    }
    std::vector<int> slot_of;            // [layer * n_expert + e] -> slot or -1
    std::vector<int> owner;              // [slot] -> key or -1
    std::vector<unsigned long long> used;
    unsigned long long clock = 0;
    long long hits = 0, misses = 0;
    double copy_bytes = 0, copy_ms = 0;
    // host staging of a layer's routing
    std::vector<int32_t> h_ids, h_tok;
    std::vector<float> h_wts, h_w;
    int32_t* p_ids = nullptr;    // pinned staging of the router's ids / weights (a copy into pageable memory blocks)
    float* p_wts = nullptr;
    int32_t* d_arg = nullptr;    // argmax of the logits, on the device and pinned on the host
    int32_t* p_arg = nullptr;   // kMaxWin argmax results (pinned)
    int arg_rows[kMaxWin] = {};
    // lossy decode levers (T <= kMaxWin only; the prompt's prefill stays exact): keep at most route_topk of the router's
    // experts per token, drop those below route_min_w times the token's largest weight, optionally rescale the rest
    int route_topk = 4;
    // residency-aware lever (REPL keys missk= / missw=): only experts that are NOT resident in VRAM cost CPU time, so only
    // those are dropped: per token keep at most miss_keep of the missed experts (heaviest first) and none lighter than
    // miss_minw times the token's largest weight; resident experts are always kept.  -1 = off.
    int sub_mode = 0;   // 1: a dropped missed expert is replaced by the best resident next-ranked expert, with that expert's own weight; 2: with the dropped expert's weight
    int32_t *d_xids = nullptr, *p_xids = nullptr;
    float *d_xwts = nullptr, *p_xwts = nullptr;
    std::vector<int32_t> h_xids;
    std::vector<float> h_xwts;
    int miss_keep = -1;
    float miss_minw = 0.f;
    bool expert_resident(int layer, int e) const {
        const int s = slot_of[(size_t) layer * m->n_expert + e];
        return s >= 0 && (slot_pend.empty() || !slot_pend[(size_t) s]);
    }
    // per-layer overrides of the two levers (REPL keys lay= / layfile=): -1 = the global value
    std::vector<int> lay_topk;
    std::vector<float> lay_minw;
    bool has_lay() const { return !lay_topk.empty(); }
    void set_lay(int l, float minw, int topk) {
        if (lay_topk.empty()) { lay_topk.assign((size_t) m->n_layers, -1); lay_minw.assign((size_t) m->n_layers, -1.f); }
        if (l >= 0 && l < m->n_layers) { lay_topk[(size_t) l] = topk; lay_minw[(size_t) l] = minw; }
    }
    float route_min_w = 0.0f;
    bool route_renorm = false;
    long long route_dropped = 0, route_total = 0;
    int top_k_model() const { return m->top_k; }
    int last_argmax = -1;
    // STRATA_M3_ROUTE_TRACE=file: the router's choices of every decode token, "step layer e0 e1 e2 e3" per line
    FILE* route_trace = nullptr;
    long long route_step = 0;
    std::vector<uint32_t> pf_count;   // tokens routed to each (layer, expert) during the prefill
    long long pf_tokens = 0;
    double ph_sync = 0;   // host time waiting for the GPU chain up to the router (per run)
    double ph_fwd = 0, ph_emb = 0, ph_attn_enq = 0, ph_head = 0;   // decode (T == 1) wall time of forward() and its parts
    // STRATA_M3_PROF=2: GPU-timeline durations between marks (CUDA events), resolved after each layer's router sync
    bool prof_gpu = false, prof_all = false, prof_draft = false;   // prof_all (STRATA_M3_PROF=3): also the layers of verify windows
    long long gcalls = 0;
    double lw[10] = {};   // STRATA_M3_PROF>=3: host ms per stage of the window layer: filter+shared / jobs+classify / cpu quant+start / gpu enqueue / admit / gu join / quant-h+down / accumulate+add
    long long lw_n = 0;
    std::vector<cudaEvent_t> gev;
    std::vector<const char*> gname;
    int gn = 0;
    std::vector<std::pair<const char*, double>> gacc;
    void gmark(const char* name) {
        if (!prof_gpu) return;
        if (gn >= (int) gev.size()) {
            gev.push_back(nullptr);
            CK(cudaEventCreate(&gev.back()));
            gname.push_back(nullptr);
        }
        CK(cudaEventRecord(gev[(size_t) gn], st));
        gname[(size_t) gn++] = name;
    }
    void gflush() {   // call when the stream is known to be idle up to the last mark
        for (int i = 1; i < gn; ++i) {
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, gev[(size_t) i - 1], gev[(size_t) i]));
            bool found = false;
            for (auto& a : gacc)
                if (a.first == gname[(size_t) i]) { a.second += ms; found = true; break; }
            if (!found) gacc.emplace_back(gname[(size_t) i], ms);
        }
        gn = 0;
        ++gcalls;
    }
#if defined(STRATA_M3_CPU_POOL)
    // ---- the CPU lane (decode, one token): experts missing from VRAM run on the CPU from the preloaded pack
    bool cpu_lane = false;
    int admit_per_layer = 2;
    bool admit_freq = true;                // admit a missed expert only if it was used at least as often as the slot's owner
    std::vector<uint16_t> use_count;       // decode-time uses per (layer, expert)
    std::unique_ptr<Par> par;
    std::vector<strata::kernels::cpu::NativeFmt> nfmt;   // per layer
    cudaStream_t cst = nullptr;        // background copies into VRAM slots
    cudaEvent_t ev_st = nullptr;
    std::vector<cudaEvent_t> slot_ev;
    std::vector<char> slot_pend;
    float *h_host = nullptr, *cpu_out_host = nullptr, *cpu_out_dev = nullptr;
    // STRATA_M3_EXP_STREAMS=S (default 4, 0 = off): the resident experts of a window layer run on S side streams, each with its own scratch;
    // the same arithmetic, the adds into `moe` in the same order (k_scatter_jobs)
    int exp_streams = 0;
    cudaStream_t sst[4] = {};
    cudaEvent_t sev[4] = {}, ev_efork = nullptr;
    void* jq8 = nullptr; void* jqf = nullptr; float* jgu = nullptr; float* jeo = nullptr;
    bool in_window_lane = false;   // expert() is being called by the decode-window lane (second-GPU slots allowed)
    bool q8h_ready = false;   // ffn has already quantized h into q8h (and recorded ev_efork) for this layer
    float* gu2 = nullptr;              // [kMaxWin][2 FF]: one expert's gate and up from a single matvec
    bool fuse_exp = strata_env_truthy_one("STRATA_M3_NO_FUSE_EXPERT") == 0 && K::native_swiglu_oai_quantize_exact();   // 5 kernels per resident expert instead of 8 (bitwise the same)
    void* q8h = nullptr;               // q8_1 image of the normed hidden, kept apart from q8 (the intermediates)
    std::vector<float> ffb, outb;      // [top_k][FF], [top_k][N]
    std::vector<uint8_t> actb, hqb;
    long long cpu_experts = 0, admitted = 0, gpu_experts = 0;
    double cpu_wall_ms = 0, ph_pre = 0, ph_adm = 0, ph_act = 0, ph_gu = 0, ph_hq = 0, ph_dn = 0, ph_acc = 0;

    void init_cpu_lane(int threads, int admit, bool admit_policy_freq) {
        namespace C = strata::kernels::cpu;
        if (!C::native_experts_available()) {
            std::fprintf(stderr, "strata-m3: --cpu-experts: this build has no ggml-cpu path\n");
            std::exit(1);
        }
        nfmt.resize(m->layers.size());
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& y = m->layers[l];
            if (y.dense) continue;
            std::string err;
            C::NativeFmt& f = nfmt[l];
            if (!C::native_fmt(y.gu_type, y.d_type, m->N, m->FF, f, err)) {
                std::fprintf(stderr, "strata-m3: --cpu-experts: layer %zu: %s\n", l, err.c_str());
                std::exit(1);
            }
            f.swiglu_alpha = m->alpha;
            f.swiglu_limit = m->limit;
            f.iq256_rows = strata_env_truthy_default("STRATA_M3_IQ256", 1) != 0;   // the AVX2 multi-token row kernel: +2-5 % end to end with verify windows (STRATA_M3_IQ256=0 for ggml's dots)
            if (f.up_off != y.gsz || f.down_off != 2 * y.gsz || f.bytes != 2 * y.gsz + y.dsz || f.gu_row * (size_t) m->FF != y.gsz) {
                std::fprintf(stderr, "strata-m3: --cpu-experts: layer %zu: the CPU expert layout does not match the pack\n", l);
                std::exit(1);
            }
        }
        std::vector<int> pin;
        if (strata_env_truthy_default("STRATA_M3_PIN", 1)) {   // one worker per physical core (the first thread of each sibling group); STRATA_M3_PIN=0 leaves them unpinned
            pin = physical_cpus();
            if ((int) pin.size() < threads) pin.clear();
        }
        par = std::make_unique<Par>(std::max(0, threads - 1), pin);
        if (!pin.empty()) std::fprintf(stderr, "strata-m3: CPU lane threads pinned to %d physical cores\n", threads);
        admit_per_layer = std::max(0, admit);
        admit_freq = admit_policy_freq;
        use_count.assign((size_t) m->n_layers * m->n_expert, 0);
        pf_count.assign((size_t) m->n_layers * m->n_expert, 0);
        CK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_st, cudaEventDisableTiming));
        slot_ev.resize((size_t) n_slots);
        for (auto& e : slot_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        slot_pend.assign((size_t) n_slots, 0);
        CK(cudaHostAlloc((void**) &h_host, (size_t) kMaxWin * m->N * 4, 0));
        // The CPU lane's result goes back through mapped pinned memory that the combine kernel reads over PCIe, not through a
        // cudaMemcpyAsync: on this GPU every H2D copy shares one copy engine in issue order, so a 24 KB result queued behind a
        // 17 MB admission copy waited up to 1.4 ms per layer (and the wait varied from run to run).
        CK(cudaHostAlloc((void**) &cpu_out_host, (size_t) kMaxWin * m->N * 4, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &cpu_out_dev, cpu_out_host, 0));
        CK(cudaMalloc(&q8h, K::native_q8_1_bytes(m->N * kMaxWin, 1)));
        CK(cudaMalloc(&gu2, (size_t) kMaxWin * 2 * m->FF * 4));
        {
            const char* v = std::getenv("STRATA_M3_EXP_STREAMS");
            exp_streams = v && v[0] ? std::min(4, std::max(0, std::atoi(v))) : 4;   // measured best of 0/2/3/4 on a V100 (docs/MINIMAX_M31_SIM.md)
            if (exp_streams > 0) {
                for (int i = 0; i < exp_streams; ++i) {
                    CK(cudaStreamCreateWithFlags(&sst[i], cudaStreamNonBlocking));
                    CK(cudaEventCreateWithFlags(&sev[i], cudaEventDisableTiming));
                }
                CK(cudaEventCreateWithFlags(&ev_efork, cudaEventDisableTiming));
                CK(cudaMalloc(&jq8, (size_t) kJobCap * K::native_q8_1_bytes(m->N * kMaxWin, 1)));
                CK(cudaMalloc(&jqf, (size_t) kJobCap * K::native_q8_1_bytes(m->FF * kMaxWin, 1)));
                CK(cudaMalloc((void**) &jgu, (size_t) kJobCap * kMaxWin * 2 * m->FF * 4));
                CK(cudaMalloc((void**) &jeo, (size_t) kJobCap * kMaxWin * m->N * 4));
                std::fprintf(stderr, "strata-m3: resident experts of a window layer on %d side streams\n", exp_streams);
            }
        }
        const size_t entries = (size_t) m->top_k * kMaxWin;   // (token, expert) pairs of a window
        ffb.assign(entries * m->FF, 0.f);
        outb.assign(entries * m->N, 0.f);
        actb.assign(kMaxWin * C::kNativeActBytes, 0);
        hqb.assign(entries * C::kNativeHBytes, 0);
        init_gpu1();
        cpu_lane = true;
        std::fprintf(stderr, "strata-m3: CPU lane on: %d threads, admit up to %d missed expert(s) per layer into VRAM (%s)\n",
                     par->size(), admit_per_layer, admit_freq ? "frequency-gated" : "every miss, LRU");
    }

    /// STRATA_M3_GPU1: the second GPU's expert slots, streams, events and per-job scratch.
    void init_gpu1() {
        const char* v = std::getenv("STRATA_M3_GPU1");
        if (!v || !v[0]) return;
        const int d = std::atoi(v);
        int nd = 0;
        CK(cudaGetDeviceCount(&nd));
        CK(cudaGetDevice(&dev0));
        if (d <= 0 || d >= nd || d == dev0 || exp_streams <= 0 || !fuse_exp || slot_pend.empty()) {
            std::fprintf(stderr, "strata-m3: STRATA_M3_GPU1=%s ignored (needs a second device, the CPU lane, STRATA_M3_EXP_STREAMS > 0 and the fused experts)\n", v);
            return;
        }
        int can01 = 0, can10 = 0;
        CK(cudaDeviceCanAccessPeer(&can01, dev0, d));
        CK(cudaDeviceCanAccessPeer(&can10, d, dev0));
        if (can01) (void) cudaDeviceEnablePeerAccess(d, 0);
        CK(cudaSetDevice(d));
        if (can10) (void) cudaDeviceEnablePeerAccess(dev0, 0);
        (void) cudaGetLastError();
        const int N = m->N, FF = m->FF;
        const size_t qbN = K::native_q8_1_bytes(N * kMaxWin, 1), qbF = K::native_q8_1_bytes(FF * kMaxWin, 1);
        const size_t scratch = qbN + (size_t) kJobCap * (qbN + qbF) + (size_t) kJobCap * kMaxWin * (2 * (size_t) FF + N) * 4;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = (size_t) 768 << 20;
        const size_t budget = fr > reserve + scratch ? fr - reserve - scratch : 0;
        const int total = m->n_layers * m->n_expert;
        n_dev1 = (int) std::min<size_t>((size_t) total, budget / slot_bytes);
        if (n_dev1 <= 0) { CK(cudaSetDevice(dev0)); n_dev1 = 0; return; }
        CK(cudaMalloc(&slots1, (size_t) n_dev1 * slot_bytes));
        CK(cudaMemset(slots1, 0, (size_t) n_dev1 * slot_bytes));
        CK(cudaStreamCreateWithFlags(&cst1, cudaStreamNonBlocking));
        for (int i = 0; i < exp_streams; ++i) {
            CK(cudaStreamCreateWithFlags(&sst1[i], cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&sev1[i], cudaEventDisableTiming));
        }
        CK(cudaMalloc(&q8h1, qbN));
        CK(cudaMalloc(&jq81, (size_t) kJobCap * qbN));
        CK(cudaMalloc(&jqf1, (size_t) kJobCap * qbF));
        CK(cudaMalloc((void**) &jgu1, (size_t) kJobCap * kMaxWin * 2 * FF * 4));
        CK(cudaMalloc((void**) &jeo1, (size_t) kJobCap * kMaxWin * N * 4));
        slot_ev.resize((size_t) (n_slots + n_dev1), nullptr);
        for (int i = 0; i < n_dev1; ++i) CK(cudaEventCreateWithFlags(&slot_ev[(size_t) (n_slots + i)], cudaEventDisableTiming));
        owner.resize((size_t) (n_slots + n_dev1), -1);
        used.resize((size_t) (n_slots + n_dev1), 0);
        slot_pend.resize((size_t) (n_slots + n_dev1), 0);
        CK(cudaSetDevice(dev0));
        CK(cudaEventCreateWithFlags(&ev_p, cudaEventDisableTiming));   // recorded on the main stream, waited on by the second GPU's
        if (gpu1_check) {
            CK(cudaMalloc(&chk_slot, slot_bytes));
            CK(cudaMalloc(&chk_q8, qbN));
            CK(cudaMalloc(&chk_qf, qbF));
            CK(cudaMalloc((void**) &chk_gu, (size_t) kMaxWin * 2 * FF * 4));
            CK(cudaMalloc((void**) &chk_eo, (size_t) kMaxWin * N * 4));
        }
        dev1 = d;
        if (!(can01 && can10)) p2p_direct = false;
        std::fprintf(stderr, "strata-m3: second GPU %d: %d expert slots of %.1f MiB (%.2f GiB), peer access %d/%d\n", d, n_dev1,
                     (double) slot_bytes / (1 << 20), (double) n_dev1 * (double) slot_bytes / (1 << 30), can01, can10);
    }

    /// After the prefill: fill the small-class slots with the experts the prompt used most (per-layer routing counts
    /// from the prefill), and seed the admission gate with those counts (scaled to the decode horizon).  The prompt's
    /// own routing predicts the decode's: the cache then starts with the hot set instead of the last layers' experts.
    void warm_cache(int horizon, double prior) {
        const int E = m->n_expert;
        std::vector<std::pair<uint32_t, int>> cand;
        for (int l = 0; l < m->n_layers; ++l) {
            const Layer& y = m->layers[(size_t) l];
            if (y.dense || big_layer(y) || !y.pack) continue;
            for (int e = 0; e < E; ++e)
                if (pf_count[(size_t) l * E + e]) cand.emplace_back(pf_count[(size_t) l * E + e], l * E + e);
        }
        const size_t want = std::min<size_t>((size_t) (n_small + n_dev1), cand.size());
        std::partial_sort(cand.begin(), cand.begin() + (long) want, cand.end(),
                          [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        std::vector<char> target((size_t) m->n_layers * E, 0);
        for (size_t i = 0; i < want; ++i) target[(size_t) cand[i].second] = 1;
        // victims: small-class slots whose owner is not a target (empty first, then least recently used)
        std::vector<int> victims, victims1;
        for (int s = 0; s < n_small; ++s)
            if (owner[(size_t) s] < 0 || !target[(size_t) owner[(size_t) s]]) victims.push_back(s);
        for (int s = n_slots; s < n_slots + n_dev1; ++s)
            if (owner[(size_t) s] < 0 || !target[(size_t) owner[(size_t) s]]) victims1.push_back(s);
        std::sort(victims.begin(), victims.end(), [&](int a, int b) { return used[(size_t) a] < used[(size_t) b]; });
        std::sort(victims1.begin(), victims1.end(), [&](int a, int b) { return used[(size_t) a] < used[(size_t) b]; });
        victims.insert(victims.begin(), victims1.begin(), victims1.end());   // the coldest targets (placed first) go to the second GPU
        const auto t0 = Clock::now();
        size_t next_victim = 0, copied = 0;
        CK(cudaStreamSynchronize(st));
        CK(cudaStreamSynchronize(cst));   // admission copies of an earlier run may still be in flight
        if (dev1 >= 0) { CK(cudaSetDevice(dev1)); CK(cudaStreamSynchronize(cst1)); CK(cudaSetDevice(dev0)); }
        for (size_t i = want; i-- > 0;) {   // coldest target first, so the hottest end up most recently used
            const int key = cand[i].second;
            if (slot_of[(size_t) key] >= 0) {
                used[(size_t) slot_of[(size_t) key]] = ++clock;
                continue;
            }
            if (next_victim >= victims.size()) continue;
            const int s = victims[next_victim++];
            const int l = key / E, e = key % E;
            const Layer& y = m->layers[(size_t) l];
            if (owner[(size_t) s] >= 0) slot_of[(size_t) owner[(size_t) s]] = -1;
            owner[(size_t) s] = key;
            slot_of[(size_t) key] = s;
            used[(size_t) s] = ++clock;
            slot_pend[(size_t) s] = 0;
            if (on_dev1(s)) {
                CK(cudaSetDevice(dev1));
                CK(cudaMemcpyAsync(slot_ptr(s), y.pack + (size_t) e * (2 * y.gsz + y.dsz), 2 * y.gsz + y.dsz, cudaMemcpyHostToDevice, cst1));
                CK(cudaSetDevice(dev0));
            } else {
                CK(cudaMemcpyAsync(slot_ptr(s), y.pack + (size_t) e * (2 * y.gsz + y.dsz), 2 * y.gsz + y.dsz, cudaMemcpyHostToDevice, st));
            }
            ++copied;
        }
        CK(cudaStreamSynchronize(st));
        if (dev1 >= 0) { CK(cudaSetDevice(dev1)); CK(cudaStreamSynchronize(cst1)); CK(cudaSetDevice(dev0)); }
        // the gate's prior: the prompt's frequency of each expert times the decode horizon, times `prior`
        if (pf_tokens > 0)
            for (size_t k = 0; k < pf_count.size(); ++k)
                use_count[k] = (uint16_t) std::min<double>(65535.0, std::round(prior * horizon * (double) pf_count[k] / (double) pf_tokens));
        std::fprintf(stderr, "strata-m3: cache warmed from the prompt's routing: %zu experts targeted, %zu copied in %.2f s\n",
                     want, copied, ms_since(t0) / 1000.0);
    }

    /// The routed experts of one decode token: resident ones on the GPU, the rest on the CPU, overlapped.
    /// h (device) is the normed hidden of the layer, h_host its copy; the router's ids / weights are on the host.
    void moe_cpu_lane(const Layer& L, int l) {
        namespace C = strata::kernels::cpu;
        const int N = m->N, Kp = m->top_k, FF = m->FF;
        struct Pick { int e; float w; bool inflight = false; };
        std::vector<Pick> picks;
        for (int k = 0; k < Kp; ++k)
            if (h_ids[(size_t) k] >= 0) picks.push_back({h_ids[(size_t) k], h_wts[(size_t) k]});
        std::sort(picks.begin(), picks.end(), [](const Pick& a, const Pick& b) { return a.e < b.e; });
        const unsigned long long layer_clock0 = clock;
        std::vector<Pick> miss;
        const auto t_pre0 = Clock::now();
        K::native_quantize_q8_1(h, q8h, N, 1, st);
        for (const Pick& p : picks) {
            const int key = l * m->n_expert + p.e;
            if (use_count[(size_t) key] < 65535) ++use_count[(size_t) key];
            const int sl = slot_of[(size_t) key];
            if (sl >= 0 && slot_pend[(size_t) sl]) {
                // an admission copy still running on the copy stream: waiting for it would stall the compute stream
                // behind every copy queued before it, so the CPU takes this expert now and the slot serves later tokens
                if (slot_copy_done(sl)) slot_pend[(size_t) sl] = 0;
                else {
                    (void) cudaGetLastError();
                    Pick q = p;
                    q.inflight = true;
                    miss.push_back(q);
                    continue;
                }
            }
            if (sl < 0) {
                miss.push_back(p);
                continue;
            }
            if (on_dev1(sl)) {   // on the second GPU: this step runs it on the CPU
                Pick q = p;
                q.inflight = true;
                miss.push_back(q);
                continue;
            }
            const uint8_t* w = expert(l, p.e);   // a hit: touches the LRU clock
            mmv(L.gu_type, w, N, FF, g, 1, q8h);
            mmv(L.gu_type, w + L.gsz, N, FF, u, 1, q8h);
            swiglu(g, u, g, FF);
            quant(g, FF, 1);
            mmv(L.d_type, w + 2 * L.gsz, FF, N, eo, 1);
            k_axpy<<<grid1(N), kThreads, 0, st>>>(moe, eo, p.w, N);
            ++gpu_experts;
        }
        ph_pre += ms_since(t_pre0);
        const auto t_adm0 = Clock::now();
        // admission: copy up to admit_per_layer missed experts into the least recently used slots, on the copy
        // stream, after the GPU work queued so far (a victim may still be read by it).  They serve later tokens.
        if (!miss.empty() && admit_per_layer > 0) {
            CK(cudaEventRecord(ev_st, st));
            int done = 0;
            for (const Pick& p : miss) {
                if (done >= admit_per_layer) break;
                if (p.inflight) continue;   // already on its way into a slot
                const int s = victim(L);
                if (used[(size_t) s] > layer_clock0) break;   // every slot of the class was used by this layer
                const int key = l * m->n_expert + p.e;
                // a rarely used expert does not push out one that is used more: the cache keeps the hot set, and the
                // PCIe traffic of admissions drops (the routing trace replay: 52.6 % hits at 29 admissions per step,
                // against 47.4 % at 51 for admitting every miss)
                if (admit_freq && owner[(size_t) s] >= 0 && use_count[(size_t) key] < use_count[(size_t) owner[(size_t) s]]) continue;
                if (owner[(size_t) s] >= 0) slot_of[(size_t) owner[(size_t) s]] = -1;
                owner[(size_t) s] = key;
                slot_of[(size_t) key] = s;
                used[(size_t) s] = ++clock;
                uint8_t* d = slot_ptr(s);
                CK(cudaStreamWaitEvent(cst, ev_st, 0));
                CK(cudaMemcpyAsync(d, L.pack + (size_t) p.e * (2 * L.gsz + L.dsz), 2 * L.gsz + L.dsz, cudaMemcpyHostToDevice, cst));
                CK(cudaEventRecord(slot_ev[(size_t) s], cst));
                slot_pend[(size_t) s] = 1;
                copy_bytes += (double) (2 * L.gsz + L.dsz);
                ++admitted;
                ++done;
            }
        }
        ph_adm += ms_since(t_adm0);
        // the CPU experts, while the GPU runs the resident ones and the shared expert
        const auto t0 = Clock::now();
        if (!miss.empty()) {
            const C::NativeFmt& f = nfmt[(size_t) l];
            auto tp = Clock::now();
            auto lap = [&](double& acc) { acc += ms_since(tp); tp = Clock::now(); };
            C::native_quant_act(f, h_host, actb.data());
            lap(ph_act);
            const void* actp[1] = {actb.data()};
            const int nm = (int) miss.size();
            constexpr int kRowsGu = 32, kRowsDown = 96;
            const int gu_chunks = (FF + kRowsGu - 1) / kRowsGu, dn_chunks = (N + kRowsDown - 1) / kRowsDown;
            par->run(nm * gu_chunks, [&](int i) {
                const int j = i / gu_chunks, c = i % gu_chunks;
                float* ffp[1] = {ffb.data() + (size_t) j * FF};
                C::native_gu_rows(f, L.pack + (size_t) miss[(size_t) j].e * (2 * L.gsz + L.dsz), actp, 1, ffp, c * kRowsGu,
                                  std::min(FF, (c + 1) * kRowsGu));
            });
            lap(ph_gu);
            for (int j = 0; j < nm; ++j) C::native_quant_h(f, ffb.data() + (size_t) j * FF, hqb.data() + (size_t) j * C::kNativeHBytes);
            lap(ph_hq);
            par->run(nm * dn_chunks, [&](int i) {
                const int j = i / dn_chunks, c = i % dn_chunks;
                const void* hq[1] = {hqb.data() + (size_t) j * C::kNativeHBytes};
                float* op[1] = {outb.data() + (size_t) j * N};
                C::native_down_rows(f, L.pack + (size_t) miss[(size_t) j].e * (2 * L.gsz + L.dsz), hq, 1, op, c * kRowsDown,
                                    std::min(N, (c + 1) * kRowsDown));
            });
            lap(ph_dn);
            for (int r = 0; r < N; ++r) {
                float a = 0.f;
                for (int j = 0; j < nm; ++j) a += miss[(size_t) j].w * outb[(size_t) j * N + r];
                cpu_out_host[r] = a;
            }
            k_add<<<grid1(N), kThreads, 0, st>>>(moe, cpu_out_dev, N);   // reads the mapped host buffer
            lap(ph_acc);
            cpu_experts += nm;
        }
        cpu_wall_ms += ms_since(t0);
        k_add<<<grid1(N), kThreads, 0, st>>>(x, moe, N);
    }

    /// The routed experts of a window of T tokens (2 <= T <= kMaxWin): the same split as the single-token lane, per
    /// distinct expert of the union.  Resident experts run on the GPU over the tokens routed to them (gather, three matvecs,
    /// scatter-add); the others on the CPU with the multi-token row kernels, each token's activation quantized once.
    void moe_cpu_lane_multi(const Layer& L, int l, int T) {
        namespace C = strata::kernels::cpu;
        const int N = m->N, Kp = m->top_k, FF = m->FF, E = m->n_expert;
        struct Job { int e; int nt = 0; int tok[kMaxWin]; float w[kMaxWin]; bool inflight = false; };
        std::vector<Job> jobs;
        {
            std::vector<int> at((size_t) E, -1);
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < Kp; ++k) {
                    const int e = h_ids[(size_t) t * Kp + k];
                    if (e < 0) continue;
                    if (at[(size_t) e] < 0) { at[(size_t) e] = (int) jobs.size(); jobs.emplace_back(); jobs.back().e = e; }
                    Job& jb = jobs[(size_t) at[(size_t) e]];
                    jb.tok[jb.nt] = t;
                    jb.w[jb.nt++] = h_wts[(size_t) t * Kp + k];
                    const int key = l * E + e;
                    if (use_count[(size_t) key] < 65535) ++use_count[(size_t) key];
                }
            std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.e < b.e; });
        }
        in_window_lane = true;
        const unsigned long long layer_clock0 = clock;
        std::vector<Job> miss, gpu_jobs;
        const auto t_pre0 = Clock::now();
        auto lwt = Clock::now();
        auto lwl = [&](int k) { const auto n = Clock::now(); lw[k] += std::chrono::duration<double, std::milli>(n - lwt).count(); lwt = n; };
        if (!q8h_ready) K::native_quantize_q8_1(h, q8h, N * T, 1, st);   // the T rows' q8_1 images, back to back
        for (const Job& jb : jobs) {   // which experts run where (no GPU work is queued yet)
            const int key = l * E + jb.e;
            const int sl = slot_of[(size_t) key];
            if (sl >= 0 && slot_pend[(size_t) sl]) {
                if (slot_copy_done(sl)) slot_pend[(size_t) sl] = 0;
                else {
                    (void) cudaGetLastError();
                    Job q = jb;
                    q.inflight = true;
                    miss.push_back(q);
                    continue;
                }
            }
            if (sl < 0) { miss.push_back(jb); continue; }
            if (on_dev1(sl) && !(dev1 >= 0 && q8h_ready)) { Job q = jb; q.inflight = true; miss.push_back(q); continue; }
            gpu_jobs.push_back(jb);
        }
        lwl(1);
        // The CPU lane's gate/up phase starts on the workers first: queueing the resident experts' kernels (about 8 launches
        // per expert) takes the host thread as long as that phase, and used to run before it.  The CPU path and the sums are
        // unchanged: the same chunks, the same order of the adds into `moe`.
        const C::NativeFmt& f = nfmt[(size_t) l];
        const int nj = (int) miss.size();
        std::vector<int> ent0((size_t) nj + 1, 0);
        constexpr int kRowsGu = 32, kRowsDown = 96;
        const int gu_chunks = (FF + kRowsGu - 1) / kRowsGu, dn_chunks = (N + kRowsDown - 1) / kRowsDown;
        std::function<void(int)> gu_fn;
        const auto t0 = Clock::now();
        if (!miss.empty()) {
            for (int t = 0; t < T; ++t) C::native_quant_act(f, h_host + (size_t) t * N, actb.data() + (size_t) t * C::kNativeActBytes);
            // entries: one per (job, token slot), in job order
            for (int j = 0; j < nj; ++j) ent0[(size_t) j + 1] = ent0[(size_t) j] + miss[(size_t) j].nt;
            gu_fn = [&](int i) {
                const int j = i / gu_chunks, c = i % gu_chunks;
                const Job& jb = miss[(size_t) j];
                const void* actp[kMaxWin];
                float* ffp[kMaxWin];
                for (int k = 0; k < jb.nt; ++k) {
                    actp[k] = actb.data() + (size_t) jb.tok[k] * C::kNativeActBytes;
                    ffp[k] = ffb.data() + (size_t) (ent0[(size_t) j] + k) * FF;
                }
                C::native_gu_rows(f, L.pack + (size_t) jb.e * (2 * L.gsz + L.dsz), actp, jb.nt, ffp, c * kRowsGu, std::min(FF, (c + 1) * kRowsGu));
            };
            par->start(nj * gu_chunks, gu_fn);
        }
        lwl(2);
        bool any_dev1 = false;
        for (const Job& jb : gpu_jobs) any_dev1 |= on_dev1(slot_of[(size_t) (l * E + jb.e)]);
        if (q8h_ready && (gpu_jobs.size() >= 2 || any_dev1) && gpu_jobs.size() <= (size_t) kJobCap) {
            const size_t qbN = K::native_q8_1_bytes(N * kMaxWin, 1), qbF = K::native_q8_1_bytes(FF * kMaxWin, 1);
            bool woke[4] = {}, woke1[4] = {}, used1[4] = {};
            int n0 = 0, n1 = 0;
            auto mmv_on = [&](cudaStream_t sj, int type, const void* w, int n_in, int n_out, float* out, int nt, const void* qq) {
                K::native_mmvq(type, w, qq, out, n_in, n_out, nt, sj);
            };
            JobList jls[kJobCap / kJobMax] = {};
            for (size_t jx = 0; jx < gpu_jobs.size(); ++jx) {
                JobList& jl = jls[jx / kJobMax];
                const size_t jc = jx % kJobMax;   // slot in this launch's list
                const Job& jb = gpu_jobs[jx];
                const bool d1 = on_dev1(slot_of[(size_t) (l * E + jb.e)]);
                const uint8_t* w = expert(l, jb.e);
                RowIdx ix{};
                ix.n = jb.nt;
                for (int i = 0; i < jb.nt; ++i) { ix.t[i] = jb.tok[i]; ix.w[i] = jb.w[i]; }
                if (d1) {   // on the second GPU: its own streams, scratch and copy of the image; the output rows come back by peer copy
                    const int si1 = n1++ % exp_streams;
                    CK(cudaSetDevice(dev1));
                    cudaStream_t sj = sst1[si1];
                    if (!woke1[si1]) {
                        CK(cudaStreamWaitEvent(sj, ev_p, 0));
                        woke1[si1] = true;
                        if (prof_all && si1 == 0 && g1ev.size() < 8192) {
                            cudaEvent_t ea; CK(cudaEventCreate(&ea)); CK(cudaEventRecord(ea, sj)); g1ev.push_back(ea);
                        }
                    }
                    uint8_t* q8j = static_cast<uint8_t*>(jq81) + jx * qbN;
                    uint8_t* qfj = static_cast<uint8_t*>(jqf1) + jx * qbF;
                    float* guj = jgu1 + jx * (size_t) kMaxWin * 2 * FF;
                    float* eoj = jeo1 + jx * (size_t) kMaxWin * N;
                    k_gather_q8<<<(unsigned) jb.nt, 256, 0, sj>>>(static_cast<const uint8_t*>(p2p_direct ? q8h : q8h1), q8j, (int) K::native_q8_1_bytes(N, 1), ix);
                    mmv_on(sj, L.gu_type, w, N, 2 * FF, guj, jb.nt, q8j);
                    K::native_swiglu_oai_quantize_q8_1(guj, guj + FF, 2 * (long) FF, qfj, FF, jb.nt, m->alpha, m->limit, sj);
                    mmv_on(sj, L.d_type, w + 2 * L.gsz, FF, N, eoj, jb.nt, qfj);
                    if (!p2p_direct) CK(cudaMemcpyPeerAsync(jeo + jx * (size_t) kMaxWin * N, dev0, eoj, dev1, (size_t) jb.nt * N * 4, sj));
                    used1[si1] = true;
                    CK(cudaSetDevice(dev0));
                    jl.nt[jc] = jb.nt;
                    jl.eo[jc] = p2p_direct ? eoj : jeo + jx * (size_t) kMaxWin * N;   // direct: the accumulate kernel reads the second GPU's rows
                    for (int i = 0; i < jb.nt; ++i) { jl.tok[jc][i] = jb.tok[i]; jl.w[jc][i] = jb.w[i]; }
                    gpu_experts += jb.nt;
                    gpu1_experts += jb.nt;
                    ++g1_jobs;
                    continue;
                }
                const int si = n0++ % exp_streams;
                cudaStream_t sj = sst[si];
                if (!woke[si]) { CK(cudaStreamWaitEvent(sj, ev_efork, 0)); woke[si] = true; }
                uint8_t* q8j = static_cast<uint8_t*>(jq8) + jx * qbN;
                uint8_t* qfj = static_cast<uint8_t*>(jqf) + jx * qbF;
                float* guj = jgu + jx * (size_t) kMaxWin * 2 * FF;
                float* eoj = jeo + jx * (size_t) kMaxWin * N;
                k_gather_q8<<<(unsigned) jb.nt, 256, 0, sj>>>(static_cast<const uint8_t*>(q8h), q8j, (int) K::native_q8_1_bytes(N, 1), ix);
                mmv_on(sj, L.gu_type, w, N, 2 * FF, guj, jb.nt, q8j);
                K::native_swiglu_oai_quantize_q8_1(guj, guj + FF, 2 * (long) FF, qfj, FF, jb.nt, m->alpha, m->limit, sj);
                mmv_on(sj, L.d_type, w + 2 * L.gsz, FF, N, eoj, jb.nt, qfj);
                jl.nt[jc] = jb.nt;
                jl.eo[jc] = eoj;
                for (int i = 0; i < jb.nt; ++i) { jl.tok[jc][i] = jb.tok[i]; jl.w[jc][i] = jb.w[i]; }
                gpu_experts += jb.nt;
            }
            for (int si = 0; si < exp_streams; ++si) {
                if (woke[si]) { CK(cudaEventRecord(sev[si], sst[si])); CK(cudaStreamWaitEvent(st, sev[si], 0)); }
                if (used1[si]) {
                    CK(cudaSetDevice(dev1));
                    if (prof_all && si == 0 && g1ev.size() % 2 == 1) {
                        cudaEvent_t eb; CK(cudaEventCreate(&eb)); CK(cudaEventRecord(eb, sst1[0])); g1ev.push_back(eb);
                    }
                    CK(cudaEventRecord(sev1[si], sst1[si]));
                    CK(cudaSetDevice(dev0));
                    CK(cudaStreamWaitEvent(st, sev1[si], 0));
                }
            }
            if (gpu1_check && any_dev1) {
                CK(cudaStreamSynchronize(st));
                std::vector<float> ra((size_t) kMaxWin * N), rb((size_t) kMaxWin * N);
                for (size_t jx = 0; jx < gpu_jobs.size(); ++jx) {
                    const Job& jb = gpu_jobs[jx];
                    if (!on_dev1(slot_of[(size_t) (l * E + jb.e)])) continue;
                    RowIdx ix{};
                    ix.n = jb.nt;
                    for (int i = 0; i < jb.nt; ++i) { ix.t[i] = jb.tok[i]; ix.w[i] = jb.w[i]; }
                    CK(cudaMemcpyAsync(chk_slot, L.pack + (size_t) jb.e * (2 * L.gsz + L.dsz), 2 * L.gsz + L.dsz, cudaMemcpyHostToDevice, st));
                    k_gather_q8<<<(unsigned) jb.nt, 256, 0, st>>>(static_cast<const uint8_t*>(q8h), static_cast<uint8_t*>(chk_q8), (int) K::native_q8_1_bytes(N, 1), ix);
                    mmv_on(st, L.gu_type, chk_slot, N, 2 * FF, chk_gu, jb.nt, chk_q8);
                    K::native_swiglu_oai_quantize_q8_1(chk_gu, chk_gu + FF, 2 * (long) FF, chk_qf, FF, jb.nt, m->alpha, m->limit, st);
                    mmv_on(st, L.d_type, chk_slot + 2 * L.gsz, FF, N, chk_eo, jb.nt, chk_qf);
                    CK(cudaStreamSynchronize(st));
                    CK(cudaMemcpy(ra.data(), chk_eo, (size_t) jb.nt * N * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(rb.data(), jeo + jx * (size_t) kMaxWin * N, (size_t) jb.nt * N * 4, cudaMemcpyDeviceToHost));
                    ++chk_jobs;
                    if (std::memcmp(ra.data(), rb.data(), (size_t) jb.nt * N * 4) != 0) ++chk_bad;
                }
            }
            for (size_t c = 0; c * kJobMax < gpu_jobs.size(); ++c) {   // the adds into moe in job order, kJobMax jobs a launch
                jls[c].n = (int) std::min<size_t>(kJobMax, gpu_jobs.size() - c * kJobMax);
                k_scatter_jobs<<<(unsigned) T, 256, 0, st>>>(moe, N, jls[c]);
            }
        } else
        for (const Job& jb : gpu_jobs) {
            const uint8_t* w = expert(l, jb.e);
            RowIdx ix{};
            ix.n = jb.nt;
            for (int i = 0; i < jb.nt; ++i) { ix.t[i] = jb.tok[i]; ix.w[i] = jb.w[i]; }
            if (fuse_exp) {
                // the gate and up matrices are adjacent in the blob: one matvec of 2 FF rows (every row's arithmetic is its own), the
                // q8_1 image of the token rows is copied from the one made at the start of the layer, and the activation is fused with
                // its quantization: 5 kernels where the plain path below queues 8, the same bytes
                k_gather_q8<<<(unsigned) jb.nt, 256, 0, st>>>(static_cast<const uint8_t*>(q8h), static_cast<uint8_t*>(q8),
                                                             (int) K::native_q8_1_bytes(N, 1), ix);
                mmv(L.gu_type, w, N, 2 * FF, gu2, jb.nt);
                K::native_swiglu_oai_quantize_q8_1(gu2, gu2 + FF, 2 * (long) FF, q8, FF, jb.nt, m->alpha, m->limit, st);
            } else {
                k_gather_rows_i<<<(unsigned) jb.nt, kThreads, 0, st>>>(h, eh, N, ix);
                quant(eh, N, jb.nt);
                mmv(L.gu_type, w, N, FF, g, jb.nt);
                mmv(L.gu_type, w + L.gsz, N, FF, u, jb.nt);
                swiglu(g, u, g, (int64_t) jb.nt * FF);
                quant(g, FF, jb.nt);
            }
            mmv(L.d_type, w + 2 * L.gsz, FF, N, eo, jb.nt);
            k_scatter_add_i<<<(unsigned) jb.nt, kThreads, 0, st>>>(moe, eo, N, ix);
            gpu_experts += jb.nt;
        }
        ph_pre += ms_since(t_pre0);
        lwl(3);
        gmark("ffn: [post] resident experts queued");
        const auto t_adm0 = Clock::now();
        if (!miss.empty() && admit_per_layer > 0) {
            CK(cudaEventRecord(ev_st, st));
            int done = 0, done1 = 0;
            for (const Job& jb : miss) {
                if (done >= admit_per_layer) break;
                if (jb.inflight) continue;
                const int s = victim(L, dev1 >= 0 && done1 < gpu1_admit);
                if (used[(size_t) s] > layer_clock0) break;
                const int key = l * E + jb.e;
                if (admit_freq && owner[(size_t) s] >= 0 && use_count[(size_t) key] < use_count[(size_t) owner[(size_t) s]]) continue;
                if (owner[(size_t) s] >= 0) slot_of[(size_t) owner[(size_t) s]] = -1;
                owner[(size_t) s] = key;
                slot_of[(size_t) key] = s;
                used[(size_t) s] = ++clock;
                if (on_dev1(s)) ++done1;
                if (on_dev1(s)) {   // the second GPU's copy stream, after the work queued so far (it may still read this slot)
                    CK(cudaSetDevice(dev1));
                    CK(cudaStreamWaitEvent(cst1, ev_st, 0));
                    CK(cudaMemcpyAsync(slot_ptr(s), L.pack + (size_t) jb.e * (2 * L.gsz + L.dsz), 2 * L.gsz + L.dsz, cudaMemcpyHostToDevice, cst1));
                    CK(cudaEventRecord(slot_ev[(size_t) s], cst1));
                    CK(cudaSetDevice(dev0));
                } else {
                CK(cudaStreamWaitEvent(cst, ev_st, 0));
                CK(cudaMemcpyAsync(slot_ptr(s), L.pack + (size_t) jb.e * (2 * L.gsz + L.dsz), 2 * L.gsz + L.dsz, cudaMemcpyHostToDevice, cst));
                CK(cudaEventRecord(slot_ev[(size_t) s], cst));
                }
                slot_pend[(size_t) s] = 1;
                copy_bytes += (double) (2 * L.gsz + L.dsz);
                ++admitted;
                ++done;
            }
        }
        ph_adm += ms_since(t_adm0);
        lwl(4);
        if (!miss.empty()) {
            par->finish();   // the gate/up phase
            lwl(5);
            for (int e2 = 0; e2 < ent0[(size_t) nj]; ++e2)
                C::native_quant_h(f, ffb.data() + (size_t) e2 * FF, hqb.data() + (size_t) e2 * C::kNativeHBytes);
            par->run(nj * dn_chunks, [&](int i) {
                const int j = i / dn_chunks, c = i % dn_chunks;
                const Job& jb = miss[(size_t) j];
                const void* hq[kMaxWin];
                float* op[kMaxWin];
                for (int k = 0; k < jb.nt; ++k) {
                    hq[k] = hqb.data() + (size_t) (ent0[(size_t) j] + k) * C::kNativeHBytes;
                    op[k] = outb.data() + (size_t) (ent0[(size_t) j] + k) * N;
                }
                C::native_down_rows(f, L.pack + (size_t) jb.e * (2 * L.gsz + L.dsz), hq, jb.nt, op, c * kRowsDown, std::min(N, (c + 1) * kRowsDown));
            });
            lwl(6);
            std::memset(cpu_out_host, 0, (size_t) T * N * sizeof(float));
            for (int j = 0; j < nj; ++j)
                for (int k = 0; k < miss[(size_t) j].nt; ++k) {
                    const float w = miss[(size_t) j].w[k];
                    const float* src = outb.data() + (size_t) (ent0[(size_t) j] + k) * N;
                    float* dst = cpu_out_host + (size_t) miss[(size_t) j].tok[k] * N;
                    for (int r = 0; r < N; ++r) dst[r] += w * src[r];
                }
            gmark("ffn: [post] CPU result ready (stream before add)");
            k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(moe, cpu_out_dev, (int64_t) T * N);
            for (const Job& jb : miss) cpu_experts += jb.nt;
            lwl(7);
        }
        cpu_wall_ms += ms_since(t0);
        k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, moe, (int64_t) T * N);
        in_window_lane = false;
    }
#endif

    void init(const Model& mm, int ctx_, int chunk, double expert_vram_gib) {
        m = &mm;
        ctx = ctx_;
        tmax = chunk;
        trace = strata_env_truthy_one("STRATA_M3_TRACE") != 0;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-m3: cublasCreate failed\n");
            std::exit(1);
        }
        cublasSetStream(blas, st);
        if (pf_gemm) {
            const size_t wmax = std::max<size_t>((size_t) mm.N * std::max(mm.DFF, mm.H * kHeadDim), (size_t) mm.N * mm.FF * 3);
            CK(cudaMalloc(&wscr, wmax * 2));
            CK(cudaMalloc(&ain16, (size_t) tmax * std::max(mm.DFF, mm.H * kHeadDim) * 2));
        }
        CK(cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
        two_streams = strata_env_truthy_default("STRATA_M3_TWO_STREAMS", 1) != 0;
        const size_t T = (size_t) tmax, N = (size_t) mm.N, QD = (size_t) mm.H * kHeadDim;
        const size_t KD = (size_t) mm.HKV * kHeadDim, ID = (size_t) mm.idx_heads * kHeadDim;
        const size_t WF = (size_t) std::max(mm.DFF, std::max(mm.FF, mm.FS));
        const size_t nblk = (size_t) (ctx + mm.idx_block - 1) / mm.idx_block;
        x = dalloc(T * N);
        h = dalloc(T * N);
        ao = dalloc(T * N);
        eh = dalloc(T * N);
        eo = dalloc(T * N);
        moe = dalloc(T * N);
        qraw = dalloc(T * QD);
        qb = dalloc(T * QD);
        att = dalloc(T * QD);
        kraw = dalloc(T * KD);
        vraw = dalloc(T * KD);
        iqraw = dalloc(T * ID);
        iqb = dalloc(T * ID);
        ikraw = dalloc(T * kHeadDim);
        g = dalloc(T * WF);
        u = dalloc(T * WF);
        lg = dalloc(T * (size_t) mm.n_expert);
        wts = dalloc(T * (size_t) mm.top_k);
        scores = dalloc(T * (size_t) mm.idx_heads * nblk);
        logits = dalloc((size_t) kMaxWin * (size_t) mm.n_vocab);   // one row per token of a verify window
        ids = dalloc_t<int32_t>(T * (size_t) mm.top_k);
        CK(cudaMalloc(&d_xids, kMaxWin * 4 * 4));
        CK(cudaMalloc(&d_xwts, kMaxWin * 4 * 4));
        CK(cudaHostAlloc((void**) &p_xids, kMaxWin * 4 * 4, 0));
        CK(cudaHostAlloc((void**) &p_xwts, kMaxWin * 4 * 4, 0));
        h_xids.assign(kMaxWin * 4, -1);
        h_xwts.assign(kMaxWin * 4, 0.f);
        CK(cudaHostAlloc((void**) &p_ids, T * (size_t) mm.top_k * 4, 0));
        CK(cudaHostAlloc((void**) &p_wts, T * (size_t) mm.top_k * 4, 0));
        CK(cudaHostAlloc((void**) &p_arg, kMaxWin * 4, 0));
        CK(cudaMalloc((void**) &d_arg, kMaxWin * 4));
        sel = dalloc_t<int32_t>(T * (size_t) mm.idx_heads * (size_t) mm.idx_top_k);
        // the step's token ids: mapped pinned memory too (no copy-engine transfer queued behind the admission copies)
        CK(cudaHostAlloc((void**) &tok_host, T * 4, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &tok, tok_host, 0));
        qpos = dalloc_t<int32_t>(T);
        kpos = dalloc_t<int32_t>((size_t) ctx);
        k_iota<<<grid1(ctx), kThreads, 0, st>>>(kpos, ctx, 0);
        e_tok = dalloc_t<int32_t>(T * (size_t) mm.top_k);
        e_w = dalloc(T * (size_t) mm.top_k);
        const size_t max_in = std::max(QD, WF);
        CK(cudaMalloc(&q8, K::native_q8_1_bytes((int) (max_in * T), 1)));
        for (int l = 0; l < mm.n_layers; ++l) {
            if (kv_half) {
                // padded by one 128-key block and zeroed: the tensor-core attention reads whole key tiles
                kc.push_back(reinterpret_cast<float*>(dalloc_t<uint16_t>((size_t) (ctx + 128) * KD)));
                vc.push_back(reinterpret_cast<float*>(dalloc_t<uint16_t>((size_t) (ctx + 128) * KD)));
                ikc.push_back(reinterpret_cast<float*>(dalloc_t<uint16_t>((size_t) (ctx + 128) * kHeadDim)));
                CK(cudaMemset(ikc.back(), 0, (size_t) (ctx + 128) * kHeadDim * 2));
                CK(cudaMemset(kc.back(), 0, (size_t) (ctx + 128) * KD * 2));
                CK(cudaMemset(vc.back(), 0, (size_t) (ctx + 128) * KD * 2));
            } else {
                kc.push_back(dalloc((size_t) ctx * KD));
                vc.push_back(dalloc((size_t) ctx * KD));
                ikc.push_back(dalloc((size_t) ctx * kHeadDim));
            }
        }
        // the slot cache from the VRAM left (or --expert-vram GiB)
        size_t min_blob = SIZE_MAX, max_blob = 0;
        int n_big_layers = 0;
        for (const Layer& y : mm.layers)
            if (!y.dense) {
                min_blob = std::min(min_blob, 2 * y.gsz + y.dsz);
                max_blob = std::max(max_blob, 2 * y.gsz + y.dsz);
            }
        const bool one_size = strata_env_truthy_one("STRATA_M3_ONE_SLOT_SIZE") != 0;   // the single-size layout, for A/B
        slot_bytes_big = (max_blob + 4096 + 255) / 256 * 256;   // a zeroed tail after each slot's down rows
        slot_bytes = one_size ? slot_bytes_big : (min_blob + 4096 + 255) / 256 * 256;
        for (const Layer& y : mm.layers)
            if (!y.dense && big_layer(y)) ++n_big_layers;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        static const size_t reserve_mib = (size_t) std::max(256, strata_env_int_or("STRATA_M3_RESERVE_MIB", 512));
        const size_t reserve = reserve_mib << 20;   // left free for cuBLAS workspace, driver and later small allocations
        size_t budget = fr > reserve ? fr - reserve : 0;
        if (expert_vram_gib >= 0) budget = std::min(budget, (size_t) (expert_vram_gib * (double) (1ull << 30)));
        const int total = mm.n_layers * mm.n_expert;
        const int n_big = n_big_layers ? std::min(8, mm.n_expert) : 0;   // the large class: a few slots, LRU like the rest (24 -> 8 frees ~370 MB for small slots)
        const size_t big_bytes = (size_t) n_big * slot_bytes_big;
        n_small = (int) std::min<size_t>((size_t) total, budget > big_bytes ? (budget - big_bytes) / slot_bytes : 0);
        n_slots = n_small + n_big;
        if (n_small < mm.n_expert) {
            std::fprintf(stderr, "strata-m3: %d expert slots do not hold one layer's %d experts (%.2f GiB free)\n",
                         n_small, mm.n_expert, (double) fr / (1 << 30));
            std::exit(1);
        }
        const size_t slot_total = (size_t) n_small * slot_bytes + big_bytes;
        CK(cudaMalloc(&slots, slot_total));
        CK(cudaMemset(slots, 0, slot_total));
        slot_of.assign((size_t) total, -1);
        owner.assign((size_t) n_slots, -1);
        used.assign((size_t) n_slots, 0);
        std::fprintf(stderr, "strata-m3: context %d, chunk %d, %d expert slots of %.1f MiB + %d of %.1f MiB (%.2f GiB), "
                     "%.2f GiB free of %.2f\n", ctx, tmax, n_small, (double) slot_bytes / (1 << 20), n_big,
                     (double) slot_bytes_big / (1 << 20), (double) slot_total / (1 << 30), (double) fr / (1 << 30),
                     (double) tot / (1 << 30));
    }

    /// The slot holding expert e of layer l, copied from the mapped file when it is not resident.
    /// --pf-prefetch: while layer l-1 computes, copy every expert of layer l that is not resident on the copy stream
    /// (a prompt chunk uses nearly all of them); expert() then waits only for the one it needs.
    bool pf_pre = false;
    void prefetch_layer(int lp) {
#if defined(STRATA_M3_CPU_POOL)
        if (lp >= m->n_layers || !cst || slot_pend.empty()) return;
        const Layer& y = m->layers[(size_t) lp];
        if (y.dense || !y.pack) return;
        const int cls = big_layer(y) ? n_slots - n_small : n_small;
        if (cls < 2 * m->n_expert) return;   // a slot class too small for two layers in flight
        const size_t eb = 2 * y.gsz + y.dsz;
        for (int e = 0; e < m->n_expert; ++e) {
            const int key = lp * m->n_expert + e;
            int s = slot_of[(size_t) key];
            if (s >= 0) { used[(size_t) s] = ++clock; continue; }
            s = victim(y);
            if (slot_pend[(size_t) s]) { CK(cudaStreamWaitEvent(cst, slot_ev[(size_t) s], 0)); slot_pend[(size_t) s] = 0; }
            if (owner[(size_t) s] >= 0) slot_of[(size_t) owner[(size_t) s]] = -1;
            owner[(size_t) s] = key;
            slot_of[(size_t) key] = s;
            used[(size_t) s] = ++clock;
            CK(cudaMemcpyAsync(slot_ptr(s), y.pack + (size_t) e * eb, eb, cudaMemcpyHostToDevice, cst));
            CK(cudaEventRecord(slot_ev[(size_t) s], cst));
            slot_pend[(size_t) s] = 1;
            copy_bytes += (double) eb;
            ++misses;
        }
#else
        (void) lp;
#endif
    }

    const uint8_t* expert(int l, int e) {
        const Layer& y = m->layers[(size_t) l];
        const int key = l * m->n_expert + e;
        int s = slot_of[(size_t) key];
        ++clock;
        if (s >= n_slots && !in_window_lane) {   // a second-GPU slot cannot serve this caller: free it, take the miss path
            owner[(size_t) s] = -1;
            slot_of[(size_t) key] = -1;
            s = -1;
        }
        if (s >= 0) {
            ++hits;
            used[(size_t) s] = clock;
#if defined(STRATA_M3_CPU_POOL)
            if (!slot_pend.empty() && slot_pend[(size_t) s]) {   // an admitted copy still in flight on the copy stream
                CK(cudaStreamWaitEvent(st, slot_ev[(size_t) s], 0));
                slot_pend[(size_t) s] = 0;
            }
#endif
            return slot_ptr(s);
        }
        ++misses;
        s = victim(y);
#if defined(STRATA_M3_CPU_POOL)
        if (!slot_pend.empty() && slot_pend[(size_t) s]) {   // do not overwrite a slot whose admission copy is in flight
            CK(cudaStreamWaitEvent(st, slot_ev[(size_t) s], 0));
            slot_pend[(size_t) s] = 0;
        }
#endif
        if (owner[(size_t) s] >= 0) slot_of[(size_t) owner[(size_t) s]] = -1;
        owner[(size_t) s] = key;
        slot_of[(size_t) key] = s;
        used[(size_t) s] = clock;
        uint8_t* d = slot_ptr(s);
        const auto t0 = Clock::now();
        if (y.pack) {   // one copy from the preloaded pack
            CK(cudaMemcpyAsync(d, y.pack + (size_t) e * (2 * y.gsz + y.dsz), 2 * y.gsz + y.dsz, cudaMemcpyHostToDevice, st));
        } else {
            CK(cudaMemcpyAsync(d, y.src_g + (size_t) e * y.gsz, y.gsz, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(d + y.gsz, y.src_u + (size_t) e * y.gsz, y.gsz, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(d + 2 * y.gsz, y.src_d + (size_t) e * y.dsz, y.dsz, cudaMemcpyHostToDevice, st));
        }
        copy_ms += ms_since(t0);
        copy_bytes += (double) (2 * y.gsz + y.dsz);
        return d;
    }

    void quant(const float* xin, int n, int T) { K::native_quantize_q8_1(xin, q8, n * T, 1, st); }
    /// y [T][n_out] = W x from q8's image of x (T columns of w.n_in), 8 columns per launch.
    void mmv(int type, const void* w, int n_in, int n_out, float* out, int T, const void* qq = nullptr) {
        const size_t qb8 = K::native_q8_1_bytes(n_in, 1);
        if (!qq) qq = q8;
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(type, w, (const char*) qq + (size_t) c0 * qb8, out + (size_t) c0 * n_out, n_in, n_out, nc,
                           st);
        }
    }
    void mmv(const Mat& w, float* out, int T) { mmv(w.type, w.d, w.n_in, w.n_out, out, T); }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: the warp-per-row kernel for a few rows, cuBLAS beyond.
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= 8) {
            static const bool old_gemv = strata_env_truthy_one("STRATA_M3_OLD_GEMV") != 0;   // A/B against the warp-per-row kernel
            if (!old_gemv && n_in % 4 == 0) k_gemv_f32_blk<<<(unsigned) n_out, kThreads, 0, st>>>(W, xin, out, n_in, n_out, T);
            else k_gemv_f32<<<dim3((unsigned) ((n_out + 7) / 8), (unsigned) T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-m3: cublasSgemm failed\n");
            std::exit(1);
        }
    }
    void swiglu(const float* gate, const float* up, float* out, int64_t n) {
        k_swiglu_oai<<<grid1(n), kThreads, 0, st>>>(gate, up, out, n, m->alpha, m->limit);
    }

    /// k_head_prep into a K / indexer-key cache of the engine's element type
    void cache_prep(cudaStream_t s, const float* src, const float* w, float* cache, int nh, int T, float ts, int p0) {
        if (kv_half)
            k_head_prep<__half><<<dim3((unsigned) nh, (unsigned) T), kHeadDim, 0, s>>>(src, w, reinterpret_cast<__half*>(cache), nh,
                                                                                       m->n_rot, ts, m->eps, p0, 1);
        else
            k_head_prep<float><<<dim3((unsigned) nh, (unsigned) T), kHeadDim, 0, s>>>(src, w, cache, nh, m->n_rot, ts, m->eps, p0, 1);
    }

    void attention(const Layer& L, size_t li, int T, int p0) {
        const int N = m->N, H = m->H, HKV = m->HKV, IH = m->idx_heads;
        const float ts = powf(m->rope_base, -2.0f / (float) m->n_rot);
        auto pt = Clock::now();
        auto pmark = [&](const char* name) {
            gmark(name);
            if (g_prof.on) { g_prof.add(name, ms_since(pt)); pt = Clock::now(); }
            if (pfprof && T > kMaxWin) {   // prefill: device time per phase (syncs)
                CK(cudaStreamSynchronize(st));
                const auto n = Clock::now();
                for (auto& a : pf_marks) if (a.first == name) { a.second += std::chrono::duration<double, std::milli>(n - pt_mark).count(); pt_mark = n; return; }
                pf_marks.emplace_back(name, std::chrono::duration<double, std::milli>(n - pt_mark).count());
                pt_mark = n;
            }
        };
        if (pfprof && T > kMaxWin) { CK(cudaStreamSynchronize(st)); pt_mark = Clock::now(); }
        const bool fork = two_streams && T <= 8;
        if (fork) {   // the indexer's projections only need h: queue them on the second stream now
            CK(cudaEventRecord(ev_fork, st));
            CK(cudaStreamWaitEvent(st2, ev_fork, 0));
            k_gemv_f32_blk<<<(unsigned) (IH * kHeadDim), kThreads, 0, st2>>>(L.iq_w, h, iqraw, N, IH * kHeadDim, T);
            k_gemv_f32_blk<<<(unsigned) kHeadDim, kThreads, 0, st2>>>(L.ik_w, h, ikraw, N, kHeadDim, T);
            k_head_prep<float><<<dim3((unsigned) IH, (unsigned) T), kHeadDim, 0, st2>>>(iqraw, L.iq_norm, iqb, IH, m->n_rot, ts,
                                                                                 m->eps, p0, 0);
            cache_prep(st2, ikraw, L.ik_norm, ikc[li], 1, T, ts, p0);
            CK(cudaEventRecord(ev_join, st2));
        }
        if (pf(T, pf_min_t)) {
            gemm_w(L.wq, h, qraw, T);
            gemm_w(L.wk, h, kraw, T, true);
            gemm_w(L.wv, h, vraw, T, true);
        } else {
        quant(h, N, T);
        mmv(L.wq, qraw, T);
        mmv(L.wk, kraw, T);
        mmv(L.wv, vraw, T);
        }
        pmark("attn: quant + q/k/v matvecs");
        k_head_prep<float><<<dim3((unsigned) H, (unsigned) T), kHeadDim, 0, st>>>(qraw, L.q_norm, qb, H, m->n_rot, ts, m->eps,
                                                                           p0, 0);
        cache_prep(st, kraw, L.k_norm, kc[li], HKV, T, ts, p0);
        if (kv_half)
            k_f32_to_f16<<<grid1((int64_t) T * HKV * kHeadDim), kThreads, 0, st>>>(vraw, reinterpret_cast<__half*>(vc[li]) + (size_t) p0 * HKV * kHeadDim, (int64_t) T * HKV * kHeadDim);
        else
        CK(cudaMemcpyAsync(vc[li] + (size_t) p0 * HKV * kHeadDim, vraw, (size_t) T * HKV * kHeadDim * 4,
                           cudaMemcpyDeviceToDevice, st));
        pmark("attn: q/k norm + rope, kv write");
        // the MSA indexer: q (IH heads) and k (one head) from the same normed input, normed and roped like q / k
        if (fork) {
            CK(cudaStreamWaitEvent(st, ev_join, 0));
        } else {
            f32mm(L.iq_w, h, iqraw, N, IH * kHeadDim, T);
            f32mm(L.ik_w, h, ikraw, N, kHeadDim, T);
            k_head_prep<float><<<dim3((unsigned) IH, (unsigned) T), kHeadDim, 0, st>>>(iqraw, L.iq_norm, iqb, IH, m->n_rot, ts,
                                                                                m->eps, p0, 0);
            cache_prep(st, ikraw, L.ik_norm, ikc[li], 1, T, ts, p0);
        }
        pmark("attn: indexer projections + norm/rope");
        const int n_pos = p0 + T;
        const int nblk = (n_pos + m->idx_block - 1) / m->idx_block;
        const int topk = std::min(m->idx_top_k, nblk);
        if (kv_half)
            K::minimax_msa_block_scores_h(iqb, reinterpret_cast<const uint16_t*>(ikc[li]), qpos, kpos, T, IH, kHeadDim, n_pos,
                                          m->idx_block, nblk, scores, (size_t) nblk, st);
        else
        K::minimax_msa_block_scores(iqb, ikc[li], qpos, kpos, T, IH, kHeadDim, n_pos, m->idx_block, nblk, scores,
                                    (size_t) nblk, st);
        pmark("attn: msa block scores");
        K::minimax_msa_select_blocks(scores, qpos, T, IH, nblk, m->idx_block, topk, std::min(m->idx_local, topk),
                                     (size_t) nblk, sel, (size_t) topk, st);
        pmark("attn: msa select");
        if (kv_half)
            K::minimax_msa_attention_h(qb, reinterpret_cast<const uint16_t*>(kc[li]), reinterpret_cast<const uint16_t*>(vc[li]),
                                       qpos, kpos, sel, T, H, HKV, kHeadDim, n_pos, m->idx_block, topk, (size_t) topk, att, st);
        else
        K::minimax_msa_attention_f32(qb, kc[li], vc[li], qpos, kpos, sel, T, H, HKV, kHeadDim, n_pos, m->idx_block,
                                     topk, (size_t) topk, att, st);
        pmark("attn: msa sparse attention");
        if (pf(T, pf_min_t)) {
            gemm_w(L.wo, att, ao, T);
        } else {
        quant(att, H * kHeadDim, T);
        mmv(L.wo, ao, T);
        }
        k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, ao, (int64_t) T * N);
        pmark("attn: o-proj + residual");
    }

    /// Lossy decode lever: per token keep at most route_topk experts and drop those lighter than route_min_w times the
    /// heaviest; route_renorm rescales the kept weights to the original sum.  Dropped entries get id -1 (skipped by every lane).
    void apply_route_filter(int T, int Kp, int layer) {
        const int tk = has_lay() && lay_topk[(size_t) layer] >= 0 ? lay_topk[(size_t) layer] : route_topk;
        const float mw = has_lay() && lay_minw[(size_t) layer] >= 0.f ? lay_minw[(size_t) layer] : route_min_w;
        for (int t = 0; t < T; ++t) {
            int32_t* id = h_ids.data() + (size_t) t * Kp;
            float* wt = h_wts.data() + (size_t) t * Kp;
            int order[16];
            for (int k = 0; k < Kp; ++k) order[k] = k;
            std::sort(order, order + Kp, [&](int a, int b) { return wt[a] != wt[b] ? wt[a] > wt[b] : id[a] < id[b]; });
            const float wmax = wt[order[0]];
            float before = 0.f, after = 0.f;
            for (int k = 0; k < Kp; ++k) before += wt[k];
            int missed_kept = 0;
            for (int r = 0; r < Kp; ++r) {
                const int k = order[r];
                bool keep = r < std::max(1, tk) && wt[k] >= mw * wmax;
                if (keep && miss_keep >= 0 && r > 0 && !expert_resident(layer, id[k])) {   // the heaviest expert is never dropped
                    if (missed_kept >= miss_keep || wt[k] < miss_minw * wmax) {
                        keep = false;
                        if (sub_mode > 0) {   // a resident stand-in: the best next-ranked expert that is on the GPU and not yet used
                            for (int j = 0; j < 4; ++j) {
                                const int cand = h_xids[(size_t) t * 4 + j];
                                if (cand < 0 || !expert_resident(layer, cand)) continue;
                                bool used_already = false;
                                for (int q = 0; q < Kp; ++q) used_already |= id[q] == cand;
                                if (used_already) continue;
                                const float w_old = wt[k];
                                id[k] = cand;
                                wt[k] = sub_mode == 1 ? h_xwts[(size_t) t * 4 + j] : w_old;
                                h_xids[(size_t) t * 4 + j] = -1;
                                after += wt[k];
                                ++route_total;
                                goto substituted;
                            }
                        }
                    } else ++missed_kept;
                } else if (keep && miss_keep >= 0 && r == 0 && !expert_resident(layer, id[k])) ++missed_kept;
                if (keep) after += wt[k];
                else { id[k] = -1; wt[k] = 0.0f; ++route_dropped; }
                ++route_total;
                substituted:;
            }
            if (route_renorm && after > 0.0f)
                for (int k = 0; k < Kp; ++k) wt[k] *= before / after;
        }
    }

    void ffn(const Layer& L, int l, int T) {
        const int N = m->N, E = m->n_expert, Kp = m->top_k, FF = m->FF;
        gmark("ffn norm");
        const bool pfT = pf(T, pf_min_t);
        if (!pfT) quant(h, N, T);
        if (L.dense && pfT) {
            if (T > kMaxWin) { ptick(pt_other); }
            gemm_w(L.fg, h, g, T);
            gemm_w(L.fu, h, u, T, true);
            swiglu(g, u, g, (int64_t) T * m->DFF);
            gemm_w(L.fd, g, ao, T);
            k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, ao, (int64_t) T * N);
            return;
        }
        if (L.dense) {
            mmv(L.fg, g, T);
            mmv(L.fu, u, T);
            swiglu(g, u, g, (int64_t) T * m->DFF);
            quant(g, m->DFF, T);
            mmv(L.fd, ao, T);
            k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, ao, (int64_t) T * N);
            return;
        }
        // the shared expert first (its q8 image of h is the one just made)
        auto pt = Clock::now();
        auto pmark = [&](const char* name) { gmark(name); if (g_prof.on) { g_prof.add(name, ms_since(pt)); pt = Clock::now(); } };
        auto shared_expert = [&]() {
            if (pfT) {
                gemm_w(L.sg, h, g, T);
                gemm_w(L.su, h, u, T, true);
                swiglu(g, u, g, (int64_t) T * m->FS);
                gemm_w(L.sd, g, moe, T);
                return;
            }
            if (fuse_exp && T <= 8 && L.sg.type == L.su.type && L.sg.n_out == L.su.n_out &&
                K::native_mmvq_pair(L.sg.type, L.sg.d, L.su.d, q8, g, u, L.sg.n_in, L.sg.n_out, T, st)) {
                // gate and up in one launch, the activation fused with its quantization (bitwise the plain path)
                K::native_swiglu_oai_quantize_q8_1(g, u, m->FS, q8, m->FS, T, m->alpha, m->limit, st);
            } else {
                mmv(L.sg, g, T);
                mmv(L.su, u, T);
                swiglu(g, u, g, (int64_t) T * m->FS);
                quant(g, m->FS, T);
            }
            mmv(L.sd, moe, T);
        };
        // CPU lane (one token): the router goes first, so the host can start the CPU experts as soon as it knows them
        // while the GPU runs the shared expert and the resident experts; the sums into `moe` keep their order
        bool router_first = false;
#if defined(STRATA_M3_CPU_POOL)
        router_first = cpu_lane && T <= kMaxWin;
#endif
        if (!router_first) {
            shared_expert();
            pmark("ffn: shared expert");
        }
        // the router
        f32mm(L.router, h, lg, N, E, T);
        const bool want_x = sub_mode > 0 && miss_keep >= 0 && T <= kMaxWin;
        k_route<<<(unsigned) T, kThreads, 0, st>>>(lg, L.bias, ids, wts, E, Kp, m->w_scale, m->w_norm ? 1 : 0, want_x ? d_xids : nullptr,
                                                   want_x ? d_xwts : nullptr, want_x ? 4 : 0);
        h_ids.resize((size_t) T * Kp);
        h_wts.resize((size_t) T * Kp);
        CK(cudaMemcpyAsync(p_ids, ids, h_ids.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(p_wts, wts, h_wts.size() * 4, cudaMemcpyDeviceToHost, st));
        if (want_x) {
            CK(cudaMemcpyAsync(p_xids, d_xids, (size_t) T * 4 * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(p_xwts, d_xwts, (size_t) T * 4 * 4, cudaMemcpyDeviceToHost, st));
        }
#if defined(STRATA_M3_CPU_POOL)
        if (cpu_lane && T <= kMaxWin) CK(cudaMemcpyAsync(h_host, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
#endif
        pmark("ffn: router + route + d2h enqueue");
        const auto t_sync0 = Clock::now();
        CK(cudaStreamSynchronize(st));
        ph_sync += ms_since(t_sync0);
        std::memcpy(h_ids.data(), p_ids, h_ids.size() * 4);
        std::memcpy(h_wts.data(), p_wts, h_wts.size() * 4);
        if (want_x) {
            std::memcpy(h_xids.data(), p_xids, (size_t) T * 4 * 4);
            std::memcpy(h_xwts.data(), p_xwts, (size_t) T * 4 * 4);
        }
        if (T <= kMaxWin && (route_topk < Kp || route_min_w > 0.0f || has_lay() || miss_keep >= 0)) apply_route_filter(T, Kp, l);
        if (route_trace && T == 1) {
            std::fprintf(route_trace, "%lld %d", route_step, l);
            for (int32_t e : h_ids) std::fprintf(route_trace, " %d", e);
            std::fprintf(route_trace, "\n");
        }
        if (prof_gpu) { if (T == 1 || prof_all) { gflush(); gmark("ffn: post-router base"); } else gn = 0; }
        const auto lw_t0 = Clock::now();
#if defined(STRATA_M3_CPU_POOL)
        if (cpu_lane && T <= kMaxWin) {
            for (int32_t e : h_ids)
                if (e < -1 || e >= E) {
                    std::fprintf(stderr, "strata-m3: router returned expert %d\n", e);
                    std::exit(1);
                }
            q8h_ready = false;
            if (exp_streams > 0 && fuse_exp && T >= 2) {   // the experts' common input image, then the fork: they run beside the shared expert
                K::native_quantize_q8_1(h, q8h, N * T, 1, st);
                CK(cudaEventRecord(ev_efork, st));
                if (dev1 >= 0) {   // the second GPU's copy of the image, while the shared expert runs
                    if (!p2p_direct) CK(cudaMemcpyPeerAsync(q8h1, dev1, q8h, dev0, K::native_q8_1_bytes(N * T, 1), st));
                    CK(cudaEventRecord(ev_p, st));
                }
                q8h_ready = true;
            }
            shared_expert();   // queued behind the router's results, running while the CPU lane works
            lw[0] += ms_since(lw_t0);
            gmark("ffn: [post] shared expert queued");
            if (T == 1) moe_cpu_lane(L, l);
            else moe_cpu_lane_multi(L, l, T);
            ++lw_n;
            return;
        }
#endif
        if (T > kMaxWin) ptick(pt_dense);   // norm, shared expert, router and its host round trip up to here
        // the entries grouped per expert (ascending id; tokens in order within one)
        std::vector<int> start((size_t) E + 1, 0);
        for (int32_t e : h_ids) {
            if (e < -1 || e >= E) {
                std::fprintf(stderr, "strata-m3: router returned expert %d\n", e);
                std::exit(1);
            }
            if (e >= 0) ++start[(size_t) e + 1];
        }
        for (int e = 0; e < E; ++e) start[(size_t) e + 1] += start[(size_t) e];
        if (T > 1 && !pf_count.empty()) {
            for (int e = 0; e < E; ++e) pf_count[(size_t) l * E + e] += (uint32_t) (start[(size_t) e + 1] - start[(size_t) e]);
            if (l == 3) pf_tokens += T;   // counted once per token (the first MoE layer)
        }
        std::vector<int> cur(start.begin(), start.end() - 1);
        h_tok.resize(h_ids.size());
        h_w.resize(h_ids.size());
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < Kp; ++k) {
                const int e = h_ids[(size_t) t * Kp + k];
                if (e < 0) continue;
                const int at = cur[(size_t) e]++;
                h_tok[(size_t) at] = t;
                h_w[(size_t) at] = h_wts[(size_t) t * Kp + k];
            }
        CK(cudaMemcpyAsync(e_tok, h_tok.data(), (size_t) start[(size_t) E] * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(e_w, h_w.data(), (size_t) start[(size_t) E] * 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));   // the host vectors are reused by the next layer
        for (int e = 0; e < E; ++e) {
            const int n = start[(size_t) e + 1] - start[(size_t) e];
            if (n == 0) continue;
            if (T > kMaxWin) ptick(pt_gemm);
            const uint8_t* w = expert(l, e);
            if (T > kMaxWin) ptick(pt_copy);
            const int32_t* et = e_tok + start[(size_t) e];
            k_gather_rows<<<(unsigned) n, kThreads, 0, st>>>(h, et, eh, N);
            if (pf(n, pf_min_rows)) {
                gemm_w(L.gu_type, w, N, FF, eh, g, n);
                gemm_w(L.gu_type, w + L.gsz, N, FF, eh, u, n, true);
                swiglu(g, u, g, (int64_t) n * FF);
                gemm_w(L.d_type, w + 2 * L.gsz, FF, N, g, eo, n);
            } else {
            quant(eh, N, n);
            mmv(L.gu_type, w, N, FF, g, n);
            mmv(L.gu_type, w + L.gsz, N, FF, u, n);
            swiglu(g, u, g, (int64_t) n * FF);
            quant(g, FF, n);
            mmv(L.d_type, w + 2 * L.gsz, FF, N, eo, n);
            }
            k_scatter_add<<<(unsigned) n, kThreads, 0, st>>>(moe, eo, et, e_w + start[(size_t) e], N);
        }
        k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, moe, (int64_t) T * N);
    }

    void trace_row(const char* what, int l, int T) {
        std::vector<float> r((size_t) m->N);
        CK(cudaMemcpyAsync(r.data(), x + (size_t) (T - 1) * m->N, r.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        double s = 0, s2 = 0;
        for (float v : r) {
            s += v;
            s2 += (double) v * v;
        }
        std::fprintf(stderr, "M3_TRACE %s layer %d pos %d: sum %.6g rms %.6g x[0..3] %.5g %.5g %.5g %.5g\n", what, l,
                     pos + T - 1, s, std::sqrt(s2 / (double) m->N), r[0], r[1], r[2], r[3]);
    }

    /// T tokens at positions pos.. through every layer; the logits of the last one into `out`.
    /// `all_rows` (T <= kMaxWin): the head runs on every row of the window and arg_rows[i] is the argmax after row i.
    void forward(const int32_t* htok, int T, std::vector<float>& out, bool need_logits = true, bool all_rows = false) {
        const int N = m->N;
        if (pos + T > ctx) {
            std::fprintf(stderr, "strata-m3: the context (%d) is full\n", ctx);
            std::exit(1);
        }
        const auto tf0 = Clock::now();
        std::memcpy(tok_host, htok, (size_t) T * 4);   // the previous step ended with a stream sync: nothing reads it now
        k_iota<<<grid1(T), kThreads, 0, st>>>(qpos, T, pos);
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        if (T == 1) ph_emb += ms_since(tf0);
        if (trace) trace_row("embd", -1, T);
        auto capture = [&](int at) {
            for (size_t k = 0; feat && k < feat_layers.size(); ++k)
                if (feat_layers[k] == at)
                    CK(cudaMemcpy2DAsync(feat + k * (size_t) N, feat_layers.size() * (size_t) N * 4, x, (size_t) N * 4,
                                         (size_t) N * 4, (size_t) T, cudaMemcpyDeviceToDevice, st));
        };
        for (int l = 0; l < m->n_layers; ++l) {
            const Layer& L = m->layers[(size_t) l];
            if (pf_pre && pf_gemm && T >= 512) {
                if (l == 0) for (int lf = 0; lf < m->n_layers; ++lf) if (!m->layers[(size_t) lf].dense) { prefetch_layer(lf); break; }
                prefetch_layer(l + 1);
            }
            capture(l);
            gmark("layer start");
            k_rms<<<(unsigned) T, kThreads, 0, st>>>(x, L.attn_norm, h, N, m->eps);
            if (pfprof && T > kMaxWin) { CK(cudaStreamSynchronize(st)); pt_last = Clock::now(); }
            const auto ta0 = Clock::now();
            attention(L, (size_t) l, T, pos);
            if (T > kMaxWin) ptick(pt_attn);
            if (T == 1) ph_attn_enq += ms_since(ta0);
            if (trace) trace_row("attn", l, T);
            k_rms<<<(unsigned) T, kThreads, 0, st>>>(x, L.ffn_norm, h, N, m->eps);
            ffn(L, l, T);
            gmark("ffn: shared + resident experts + CPU wait + combine");
            if (T > kMaxWin) ptick(pt_gemm);
            if (trace) trace_row("out", l, T);
        }
        capture(m->n_layers);
        const auto th0 = Clock::now();
        const int R = (all_rows && T <= kMaxWin) ? T : 1;   // rows the head runs on: the window's, or the last one
        k_rms<<<(unsigned) R, kThreads, 0, st>>>(x + (size_t) (T - R) * N, m->out_norm, h, N, m->eps);
        quant(h, N, R);
        mmv(m->head, logits, R);
        k_argmax_rows<<<(unsigned) R, 1024, 0, st>>>(logits, m->n_vocab, d_arg);
        CK(cudaMemcpyAsync(p_arg, d_arg, (size_t) R * 4, cudaMemcpyDeviceToHost, st));
        if (need_logits) {
            out.resize((size_t) R * m->n_vocab);
            CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < R; ++i) arg_rows[i] = p_arg[i];
        last_argmax = p_arg[R - 1];
        if (T == 1) {
            ph_head += ms_since(th0);
            ph_fwd += ms_since(tf0);
        }
        pos += T;
    }
};


// ------------------------------------------------------------------------------------------------ the DSpark draft head
/// Draft attention: R block rows (positions n .. n+R-1) over the cached keys [max(0, pos-window+1), n) and all R block rows.
__global__ void k_dattn(const float* q, const float* kc, const float* vc, const float* kb, const float* vb, int n_ctx, int R,
                        int H, int HKV, int window, float* out) {
    extern __shared__ float sc[];
    __shared__ float red[128];
    const int r = blockIdx.x, h = blockIdx.y, kh = h / (H / HKV), w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j0 = max(0, n_ctx + r - window + 1), nc = n_ctx - j0, nk = nc + R;
    const float* qv = q + ((size_t) r * H + h) * kHeadDim;
    const float4 qq = reinterpret_cast<const float4*>(qv)[lane];
    const float scale = rsqrtf((float) kHeadDim);
    for (int s = w; s < nk; s += 4) {
        const float* kp = s < nc ? kc + ((size_t) (j0 + s) * HKV + kh) * kHeadDim : kb + ((size_t) (s - nc) * HKV + kh) * kHeadDim;
        const float4 kk = reinterpret_cast<const float4*>(kp)[lane];
        float d = qq.x * kk.x + qq.y * kk.y + qq.z * kk.z + qq.w * kk.w;
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        if (lane == 0) sc[s] = d * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = threadIdx.x; s < nk; s += 128) mx = fmaxf(mx, sc[s]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int s = threadIdx.x; s < nk; s += 128) {
        const float e = __expf(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __syncthreads();
    float acc = 0.f;
    for (int s = 0; s < nk; ++s) {
        const float* vp = s < nc ? vc + ((size_t) (j0 + s) * HKV + kh) * kHeadDim : vb + ((size_t) (s - nc) * HKV + kh) * kHeadDim;
        acc += sc[s] * vp[threadIdx.x];
    }
    out[((size_t) r * H + h) * kHeadDim + threadIdx.x] = acc / sum;
}

/// col[v] += w2[v] . w1[*prev] (both [V][256] Q8_0: 8 blocks of 34 bytes per row)
__global__ void k_markov(const uint8_t* w1, const uint8_t* w2, const int32_t* prev, float* col, int V) {
    __shared__ float e[256];
    const int p = *prev;
    {
        const uint8_t* b = w1 + (size_t) p * 272 + (threadIdx.x >> 5) * 34;
        e[threadIdx.x] = __half2float(*reinterpret_cast<const __half*>(b)) * (float) reinterpret_cast<const int8_t*>(b + 2)[threadIdx.x & 31];
    }
    __syncthreads();
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    // the 272-byte row (8 Q8_0 blocks of 34 bytes) as 17 aligned 16-byte loads instead of 256 byte loads (a warp's byte load
    // touches 32 different lines, 51 M such wavefronts per row were the whole cost); the same bytes, products and summation order
    uint32_t wd[68];
    {
        const uint4* rv = reinterpret_cast<const uint4*>(w2 + (size_t) v * 272);
#pragma unroll
        for (int i = 0; i < 17; ++i) { const uint4 t = rv[i]; wd[4 * i] = t.x; wd[4 * i + 1] = t.y; wd[4 * i + 2] = t.z; wd[4 * i + 3] = t.w; }
    }
    auto byte_at = [&](int o) -> uint32_t { return (wd[o >> 2] >> (8 * (o & 3))) & 0xffu; };
    float sum = 0.f;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const uint16_t hb = (uint16_t) (byte_at(b * 34) | (byte_at(b * 34 + 1) << 8));
        const float d = __half2float(*reinterpret_cast<const __half*>(&hb));
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) a += (float) (int8_t) (uint8_t) byte_at(b * 34 + 2 + i) * e[b * 32 + i];
        sum += d * a;
    }
    col[v] += sum;
}

/// DSpark confidence of draft position i: sigmoid(conf_w . [hidden_i ; markov_w1[prev_i]] + b), hidden_i the draft's final-normed row
__global__ void k_conf(const float* hrow, const uint8_t* w1, const int32_t* prev, const float* cw, float b, int N, float* out) {
    __shared__ float e[256], red[256];
    const int p = *prev;
    {
        const uint8_t* bl = w1 + (size_t) p * 272 + (threadIdx.x >> 5) * 34;
        e[threadIdx.x] = __half2float(*reinterpret_cast<const __half*>(bl)) * (float) reinterpret_cast<const int8_t*>(bl + 2)[threadIdx.x & 31];
    }
    __syncthreads();
    float acc = 0.f;
    for (int i = threadIdx.x; i < N; i += 256) acc += hrow[i] * cw[i];
    acc += e[threadIdx.x] * cw[N + threadIdx.x];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = 1.f / (1.f + __expf(-(red[0] + b)));
}

struct Dspark {
    struct DL {
        const float *an = nullptr, *fn = nullptr, *qn = nullptr, *kn = nullptr;
        Mat q, k, v, o, g, u, d;
    };
    std::shared_ptr<strata::GgufModel> file;
    std::vector<DL> ly;
    Mat fc, mk1, mk2;
    float* conf_w = nullptr, *d_conf = nullptr;
    float conf_b = 0.f;
    float pmin_default = 0.f;
    float pmin = 0.f;   // drafts whose confidence is below this are not verified (REPL key pmin=; 0 = all)
    long long conf_cut = 0;
    const float *out_norm = nullptr, *enc_norm = nullptr;
    std::vector<int> target_layers;
    int L = 0, N = 0, H = 0, HKV = 0, FF = 0, block = 7, window = 4096, mask_id = 0, ctx = 0;
    float alpha = 1.702f, limit = 7.f, ts = 0.f;
    std::vector<float*> kc, vc;
    float *g = nullptr, *gn = nullptr, *kraw = nullptr, *vraw = nullptr, *qraw = nullptr, *qb = nullptr, *kbl = nullptr,
          *xb = nullptr, *hb = nullptr, *att = nullptr, *ao = nullptr, *gt = nullptr, *up = nullptr, *ffo = nullptr;
    void* dq8 = nullptr;
    int32_t *dtok = nullptr, *dprev = nullptr;
    long long inj_rows = 0, drafts = 0;
    double ms_inject = 0, ms_draft = 0;
    double ms_dl = 0, ms_dh = 0, ms_dt = 0;   // STRATA_M3_PROF=4: GPU ms of the draft pass: layers / lm_head / markov+argmax+conf
    cudaEvent_t dev[4] = {};

    bool load(const std::string& path, int ctx_, const Model& m, std::string& err) {
        ctx = ctx_;
        file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
        const strata::GgufModel& gm = *file;
        const strata::GgufFile& meta = gm.meta();
        auto gi = [&](const char* k) { return (int) meta_int(meta, std::string("dflash.") + k); };
        L = gi("block_count"); N = gi("embedding_length"); FF = gi("feed_forward_length");
        H = gi("attention.head_count"); HKV = gi("attention.head_count_kv"); block = gi("block_size");
        window = gi("attention.sliding_window");
        if (N != m.N || L < 1 || H * kHeadDim != 8192 || HKV * kHeadDim != 512 || gi("rope.dimension_count") != m.n_rot) {
            err = "the draft head does not match this target (hidden size / heads / rope)";
            return false;
        }
        mask_id = (int) meta_int(meta, "tokenizer.ggml.mask_token_id");
        if (const strata::MetaValue* v = meta.get("dflash.target_layers"))
            for (const strata::MetaValue& it : v->items) target_layers.push_back((int) it.u);
        if (target_layers.empty()) { err = "no dflash.target_layers"; return false; }
        ts = powf(m.rope_base, -2.0f / (float) m.n_rot);
        auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
            const strata::TensorInfo* t = gm.find(name, &s);
            if (!t) throw std::runtime_error("missing " + name);
            return t;
        };
        auto f32 = [&](const std::string& name, uint64_t n) -> const float* {
            size_t s = 0;
            const strata::TensorInfo* t = find(name, s);
            if (t->type != 0 || (uint64_t) t->elements() != n) throw std::runtime_error(name + ": expected F32");
            float* d = nullptr;
            CK(cudaMalloc(&d, n * 4));
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), n * 4, cudaMemcpyHostToDevice));
            return d;
        };
        auto mat = [&](const std::string& name, int n_in, int n_out) -> Mat {
            size_t s = 0;
            const strata::TensorInfo* t = find(name, s);
            Mat w;
            w.type = (int) t->type; w.n_in = (int) t->shape[0]; w.n_out = (int) t->shape[1];
            if (w.n_in != n_in || w.n_out != n_out || !K::native_mmvq_supported(w.type)) throw std::runtime_error(name + ": unexpected shape / type");
            const uint64_t nb = strata::tensor_payload_bytes(*t);
            uint8_t* d = nullptr;
            CK(cudaMalloc(&d, nb + 256));
            CK(cudaMemset(d + nb, 0, 256));
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            w.d = d;
            return w;
        };
        const int F = (int) target_layers.size();
        fc = mat("fc.weight", F * N, N);
        enc_norm = f32("enc.output_norm.weight", N);
        out_norm = f32("output_norm.weight", N);
        {   // Markov tables: raw Q8_0 rows of 256
            for (int w = 0; w < 2; ++w) {
                size_t s = 0;
                const strata::TensorInfo* t = find(w ? "markov_w2.weight" : "markov_w1.weight", s);
                if (t->type != 8 || t->shape[0] != 256 || (int) t->shape[1] != m.n_vocab) { err = "unexpected markov table"; return false; }
                const uint64_t nb = strata::tensor_payload_bytes(*t);
                uint8_t* d = nullptr;
                CK(cudaMalloc(&d, nb));
                CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
                (w ? mk2 : mk1).d = d;
            }
        }
        {   // the confidence head (optional)
            size_t sw = 0, sb = 0;
            const strata::TensorInfo* tw = gm.find("conf_proj.weight", &sw);
            const strata::TensorInfo* tb = gm.find("conf_proj.bias", &sb);
            if (tw && tb && tw->type == 0 && (int) tw->elements() == N + 256 && tb->type == 0) {
                CK(cudaMalloc(&conf_w, (size_t) (N + 256) * 4));
                CK(cudaMemcpy(conf_w, gm.shard(sw).tensor_data(*tw), (size_t) (N + 256) * 4, cudaMemcpyHostToDevice));
                std::memcpy(&conf_b, gm.shard(sb).tensor_data(*tb), 4);
                CK(cudaMalloc(&d_conf, 8 * 4));
            }
        }
        ly.resize((size_t) L);
        for (int l = 0; l < L; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            DL& y = ly[(size_t) l];
            y.an = f32(b + "attn_norm.weight", N); y.fn = f32(b + "ffn_norm.weight", N);
            y.qn = f32(b + "attn_q_norm.weight", kHeadDim); y.kn = f32(b + "attn_k_norm.weight", kHeadDim);
            y.q = mat(b + "attn_q.weight", N, H * kHeadDim); y.k = mat(b + "attn_k.weight", N, HKV * kHeadDim);
            y.v = mat(b + "attn_v.weight", N, HKV * kHeadDim); y.o = mat(b + "attn_output.weight", H * kHeadDim, N);
            y.g = mat(b + "ffn_gate.weight", N, FF); y.u = mat(b + "ffn_up.weight", N, FF); y.d = mat(b + "ffn_down.weight", FF, N);
            kc.push_back(dalloc((size_t) ctx * HKV * kHeadDim));
            vc.push_back(dalloc((size_t) ctx * HKV * kHeadDim));
        }
        const size_t R = 8;
        g = dalloc(R * N); gn = dalloc(R * N); kraw = dalloc(R * HKV * kHeadDim); vraw = dalloc(R * HKV * kHeadDim);
        qraw = dalloc(R * H * kHeadDim); qb = dalloc(R * H * kHeadDim); kbl = dalloc(R * HKV * kHeadDim);
        xb = dalloc(R * N); hb = dalloc(R * N); att = dalloc(R * H * kHeadDim); ao = dalloc(R * N);
        gt = dalloc(R * FF); up = dalloc(R * FF); ffo = dalloc(R * N);
        CK(cudaMalloc(&dq8, K::native_q8_1_bytes((int) (F * N * R), 1)));
        dtok = dalloc_t<int32_t>(R);
        dprev = dalloc_t<int32_t>(R + 1);
        return true;
    }

    void qmm(Engine& E, const Mat& w, const float* in, float* out, int T) {
        K::native_quantize_q8_1(in, dq8, w.n_in * T, 1, E.st);
        E.mmv(w.type, w.d, w.n_in, w.n_out, out, T, dq8);
    }

    /// Fuse the captured target features of T committed rows (E.feat rows 0..T-1) and write their K/V at pos0 ..
    void inject(Engine& E, int T, int pos0) {
        const auto t0 = Clock::now();
        for (int g0 = 0; g0 < T; g0 += 8) {
            const int nc = std::min(8, T - g0);
            qmm(E, fc, E.feat + (size_t) g0 * fc.n_in, g, nc);
            k_rms<<<(unsigned) nc, kThreads, 0, E.st>>>(g, enc_norm, gn, N, E.m->eps);
            K::native_quantize_q8_1(gn, dq8, N * nc, 1, E.st);
            for (int l = 0; l < L; ++l) {
                const DL& y = ly[(size_t) l];
                E.mmv(y.k.type, y.k.d, y.k.n_in, y.k.n_out, kraw, nc, dq8);
                E.mmv(y.v.type, y.v.d, y.v.n_in, y.v.n_out, vraw, nc, dq8);
                k_head_prep<float><<<dim3((unsigned) HKV, (unsigned) nc), kHeadDim, 0, E.st>>>(kraw, y.kn, kc[(size_t) l], HKV, E.m->n_rot, ts, E.m->eps, pos0 + g0, 1);
                CK(cudaMemcpyAsync(vc[(size_t) l] + (size_t) (pos0 + g0) * HKV * kHeadDim, vraw, (size_t) nc * HKV * kHeadDim * 4, cudaMemcpyDeviceToDevice, E.st));
            }
        }
        inj_rows += T;
        ms_inject += ms_since(t0);
    }

    /// Draft D tokens after `cur` (the next token to feed, at position n): block [cur, mask * (D-1)], greedy Markov chain.
    std::vector<int32_t> draft(Engine& E, int32_t cur, int n, int D) {
        const auto t0 = Clock::now();
        const int R = std::min(D, std::min(block, kMaxWin - 1)), V = E.m->n_vocab;
        std::vector<int32_t> blk((size_t) R, mask_id);
        blk[0] = cur;
        CK(cudaMemcpyAsync(dtok, blk.data(), (size_t) R * 4, cudaMemcpyHostToDevice, E.st));
        CK(cudaMemcpyAsync(dprev, &cur, 4, cudaMemcpyHostToDevice, E.st));
        const bool dprof = E.prof_draft;
        if (dprof) { if (!dev[0]) for (auto& e : dev) CK(cudaEventCreate(&e)); CK(cudaEventRecord(dev[0], E.st)); }
        K::iq_embed_rows(E.m->embd_type, E.m->embd, E.m->embd_row, dtok, R, N, xb, E.st);
        for (int l = 0; l < L; ++l) {
            const DL& y = ly[(size_t) l];
            k_rms<<<(unsigned) R, kThreads, 0, E.st>>>(xb, y.an, hb, N, E.m->eps);
            K::native_quantize_q8_1(hb, dq8, N * R, 1, E.st);
            E.mmv(y.q.type, y.q.d, y.q.n_in, y.q.n_out, qraw, R, dq8);
            E.mmv(y.k.type, y.k.d, y.k.n_in, y.k.n_out, kraw, R, dq8);
            E.mmv(y.v.type, y.v.d, y.v.n_in, y.v.n_out, vraw, R, dq8);
            k_head_prep<float><<<dim3((unsigned) H, (unsigned) R), kHeadDim, 0, E.st>>>(qraw, y.qn, qb, H, E.m->n_rot, ts, E.m->eps, n, 0);
            k_head_prep<float><<<dim3((unsigned) HKV, (unsigned) R), kHeadDim, 0, E.st>>>(kraw, y.kn, kbl, HKV, E.m->n_rot, ts, E.m->eps, n, 0);
            k_dattn<<<dim3((unsigned) R, (unsigned) H), 128, (size_t) (n + R + 8) * 4, E.st>>>(qb, kc[(size_t) l], vc[(size_t) l], kbl, vraw, n, R, H, HKV, window, att);
            qmm(E, y.o, att, ao, R);
            k_add<<<grid1((int64_t) R * N), kThreads, 0, E.st>>>(xb, ao, (int64_t) R * N);
            k_rms<<<(unsigned) R, kThreads, 0, E.st>>>(xb, y.fn, hb, N, E.m->eps);
            K::native_quantize_q8_1(hb, dq8, N * R, 1, E.st);
            E.mmv(y.g.type, y.g.d, y.g.n_in, y.g.n_out, gt, R, dq8);
            E.mmv(y.u.type, y.u.d, y.u.n_in, y.u.n_out, up, R, dq8);
            k_swiglu_oai<<<grid1((int64_t) R * FF), kThreads, 0, E.st>>>(gt, up, gt, (int64_t) R * FF, alpha, limit);
            qmm(E, y.d, gt, ffo, R);
            k_add<<<grid1((int64_t) R * N), kThreads, 0, E.st>>>(xb, ffo, (int64_t) R * N);
        }
        if (dprof) CK(cudaEventRecord(dev[1], E.st));
        k_rms<<<(unsigned) R, kThreads, 0, E.st>>>(xb, out_norm, hb, N, E.m->eps);
        E.quant(hb, N, R);
        E.mmv(E.m->head, E.logits, R);
        if (dprof) CK(cudaEventRecord(dev[2], E.st));
        for (int i = 0; i < R; ++i) {
            k_markov<<<(unsigned) ((V + 255) / 256), 256, 0, E.st>>>(mk1.d, mk2.d, dprev + i, E.logits + (size_t) i * V, V);
            k_argmax1<<<1, 1024, 0, E.st>>>(E.logits + (size_t) i * V, V, dprev + i + 1);
        }
        const bool use_conf = pmin > 0.f && conf_w;
        if (use_conf)
            for (int i = 0; i < R; ++i)
                k_conf<<<1, 256, 0, E.st>>>(hb + (size_t) i * N, mk1.d, dprev + i, conf_w, conf_b, N, d_conf + i);
        std::vector<int32_t> out((size_t) R);
        float conf[8] = {};
        CK(cudaMemcpyAsync(out.data(), dprev + 1, (size_t) R * 4, cudaMemcpyDeviceToHost, E.st));
        if (use_conf) CK(cudaMemcpyAsync(conf, d_conf, (size_t) R * 4, cudaMemcpyDeviceToHost, E.st));
        if (dprof) CK(cudaEventRecord(dev[3], E.st));
        CK(cudaStreamSynchronize(E.st));
        if (dprof) {
            float a = 0, b = 0, c = 0;
            CK(cudaEventElapsedTime(&a, dev[0], dev[1])); CK(cudaEventElapsedTime(&b, dev[1], dev[2])); CK(cudaEventElapsedTime(&c, dev[2], dev[3]));
            ms_dl += a; ms_dh += b; ms_dt += c;
        }
        if (use_conf) {
            int keep = 0;
            while (keep < R && conf[keep] >= pmin) ++keep;
            conf_cut += R - keep;
            out.resize((size_t) keep);
        }
        drafts += R;
        ms_draft += ms_since(t0);
        return out;
    }
};

int argmax(const std::vector<float>& v) {
    return (int) (std::max_element(v.begin(), v.end()) - v.begin());
}

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> r;
    std::string cur;
    for (char c : s + ",") {
        if (c >= '0' && c <= '9') cur += c;
        else if (!cur.empty()) {
            r.push_back((int32_t) std::atol(cur.c_str()));
            cur.clear();
        }
    }
    return r;
}

std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::string s;
    char buf[65536];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) s.append(buf, n);
    std::fclose(f);
    return s;
}

}  // namespace

int strata::minimax::run(int argc, char** argv) {
    // Dense multi-column matvec (verify windows of 2-8 rows): one row per thread block and the transposed warp reduction give
    // the same results as the library defaults and decode 4-6 % faster in the engine (docs/MINIMAX_M31_SIM.md); the user's
    // environment still wins (STRATA_NO_MMVQ_ROWS2=0 / STRATA_TSUM=0 restore the library defaults).
    setenv("STRATA_NO_MMVQ_ROWS2", "1", 0);
    setenv("STRATA_TSUM", "1", 0);
    std::string model, tokens, dump_logits, force_file;
    int ctx = 8192, chunk = 0, n_gen = 32, prefault_threads = 32;
    double expert_vram = -1.0;
    bool interleave_pack = true, ignore_eos = false, preload_experts = strata_env_truthy_one("STRATA_M3_PRELOAD") != 0, pin_pack = true;
    int preload_threads = 16, cpu_threads = 35, admit = 2, runs = 1;
    bool admit_freq = true, repl = false, route_renorm_opt = false, kv_f16_opt = strata_env_truthy_one("STRATA_M3_KV_F16") != 0;
    std::string spec_mode = "none", dspark_path;
    bool spec_given = false, draft_given = false;
    bool pf_gemm_opt = strata_env_truthy_one("STRATA_M3_PF_GEMM") != 0, pf_pre_opt = false;
    int spec_draft = 2, route_topk_opt = 4;
    float route_min_w_opt = 0.0f;
    int miss_keep_opt = -1;
    bool cpu_experts = strata_env_truthy_one("STRATA_M3_CPU_EXPERTS") != 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-m3: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") tokens = read_file(next());
        else if (a == "--max-new" || a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") chunk = std::atoi(next().c_str());
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--force") force_file = next();   // teacher forcing: feed these ids as the decode tokens, still report argmax
        else if (a == "--prefault-threads") prefault_threads = std::atoi(next().c_str());
        else if (a == "--ignore-eos") ignore_eos = true;
        else if (a == "--preload-experts") preload_experts = true;
        else if (a == "--preload-threads") preload_threads = std::max(1, std::atoi(next().c_str()));
        else if (a == "--no-pin") pin_pack = false;
        else if (a == "--no-interleave") interleave_pack = false;
        else if (a == "--cpu-experts") cpu_experts = true;
        else if (a == "--repl") repl = true;
        else if (a == "--kv-f16") kv_f16_opt = true;
        else if (a == "--dspark") dspark_path = next();
        else if (a == "--pf-gemm") pf_gemm_opt = true;
        else if (a == "--pf-prefetch") pf_pre_opt = true;
        else if (a == "--spec") { spec_mode = next(); spec_given = true; }
        else if (a == "--draft") { spec_draft = std::max(1, std::atoi(next().c_str())); draft_given = true; }
        else if (a == "--route-topk") route_topk_opt = std::max(1, std::atoi(next().c_str()));
        else if (a == "--route-min-weight") route_min_w_opt = (float) std::atof(next().c_str());
        else if (a == "--route-renorm") route_renorm_opt = true;
        else if (a == "--route-miss-keep") miss_keep_opt = std::atoi(next().c_str());
        else if (a == "--speed") {   // lossy decode presets, measured on the C++ benchmark (docs/MINIMAX_M31_SIM.md)
            const std::string v = next();
            if (v == "exact") { route_topk_opt = 4; miss_keep_opt = -1; }
            else if (v == "fast") route_topk_opt = 3;            // slight loss: ~30 tok/s
            else if (v == "turbo") route_topk_opt = 2;           // approximate: ~40 tok/s
            else if (v == "max") { route_topk_opt = 2; miss_keep_opt = 0; }   // top-2, and no expert that misses VRAM: ~50 tok/s
            else { std::fprintf(stderr, "strata-m3: --speed exact|fast|turbo|max\n"); return 2; }
        }
        else if (a == "--runs") runs = std::max(1, std::atoi(next().c_str()));
        else if (a == "--cpu-threads") cpu_threads = std::max(1, std::atoi(next().c_str()));
        else if (a == "--admit") admit = std::max(0, std::atoi(next().c_str()));
        else if (a == "--admit-policy") admit_freq = std::string(next()) != "lru";
        else if (a == "--gpu") (void) next();   // applied by strata before the switch
        else if (a == "--greedy") {}
        else {
            std::fprintf(stderr, "usage: strata-m3 --model GGUF (--tokens IDS | --tokens-file F) [--max-new 32] "
                                 "[--max-context 8192] [--chunk N] [--expert-vram GiB] [--dump-logits F] "
                                 "[--ignore-eos] [--preload-experts [--preload-threads 16] [--no-pin]] [--cpu-experts [--cpu-threads 35] [--admit 2] [--admit-policy freq|lru]]\n");
            return 2;
        }
    }
    if (model.empty() || (tokens.empty() && !repl)) {
        std::fprintf(stderr, "strata-m3: --model and --tokens are required (--repl takes requests on stdin instead)\n");
        return 2;
    }
    const std::vector<int32_t> prompt = repl && tokens.empty() ? std::vector<int32_t>{0} : parse_ids(tokens);
    if (prompt.empty() || ctx < 256 || n_gen < 0 || (!repl && (int) prompt.size() + n_gen > ctx)) {
        std::fprintf(stderr, "strata-m3: the prompt (%zu tokens) and --max-new %d do not fit --max-context %d\n",
                     prompt.size(), n_gen, ctx);
        return 2;
    }
    // The K/V and indexer caches are sized for this request (prompt + new tokens + a block of slack), not for the
    // --max-context capacity: at 8192 positions they hold 2.25 GB of F32 that expert slots can use instead.
    // --max-context-alloc (or STRATA_M3_KV_FULL=1) reserves the whole --max-context as before.
    if (!strata_env_truthy_one("STRATA_M3_KV_FULL") && !repl) {
        const int need = (int) ((prompt.size() + (size_t) n_gen + 255) / 128 + 1) * 128;
        if (need < ctx) {
            std::fprintf(stderr, "strata-m3: K/V caches sized for %d positions (this request), not --max-context %d\n", need, ctx);
            ctx = std::max(256, need);
        }
    }
    if (chunk <= 0) chunk = std::min(ctx, 8192);
    // 57 stream syncs per decoded token (the router's choices go to the host every layer): spin instead of the
    // driver's blocking wake-up, which adds tens of microseconds to each.  STRATA_M3_SYNC_BLOCKING=1 keeps the default.
    if (!strata_env_truthy_one("STRATA_M3_SYNC_BLOCKING")) (void) cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    Model m;
    std::string err;
    try {
        if (!load(model, m, err)) {
            std::fprintf(stderr, "strata-m3: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-m3: %s\n", e.what());
        return 1;
    }
    for (int32_t t : prompt)
        if (t < 0 || t >= m.n_vocab) {
            std::fprintf(stderr, "strata-m3: token %d is outside the vocabulary (%d)\n", t, m.n_vocab);
            return 2;
        }
    if (strata_env_truthy_one("STRATA_M3_PREFAULT")) prefault(m, std::max(1, prefault_threads));
#if !defined(_WIN32)
    if (preload_experts) {
        std::string perr;
        if (!preload(m, preload_threads, pin_pack, interleave_pack, perr))
            std::fprintf(stderr, "strata-m3: --preload-experts: %s; reading the mapped file instead\n", perr.c_str());
    }
#else
    if (preload_experts) std::fprintf(stderr, "strata-m3: --preload-experts is not available on Windows\n");
#endif
    if (m.pack_base && strata_env_truthy_one("STRATA_M3_PRELOAD_CHECK")) {
        // byte-compare random experts of the pack with the mapped file (a few hundred MB of page faults, once)
        std::mt19937_64 rng(12345);
        int bad = 0, n = 0;
        for (int k = 0; k < 48; ++k) {
            const Layer* y = nullptr;
            while (!y || y->dense) y = &m.layers[rng() % m.layers.size()];
            const int e = (int) (rng() % (uint64_t) m.n_expert);
            const uint8_t* b = y->pack + (size_t) e * (2 * y->gsz + y->dsz);
            bad += std::memcmp(b, y->src_g + (size_t) e * y->gsz, y->gsz) != 0;
            bad += std::memcmp(b + y->gsz, y->src_u + (size_t) e * y->gsz, y->gsz) != 0;
            bad += std::memcmp(b + 2 * y->gsz, y->src_d + (size_t) e * y->dsz, y->dsz) != 0;
            ++n;
        }
        std::printf("M3_PACK_CHECK %d experts compared, %d mismatching parts\n", n, bad);
        if (bad) return 1;
    }
    std::unique_ptr<Dspark> dsp;
    if (!dspark_path.empty()) {
        dsp = std::make_unique<Dspark>();
        std::string derr;
        if (!dsp->load(dspark_path, ctx, m, derr)) {
            std::fprintf(stderr, "strata-m3: DSpark head: %s\n", derr.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata-m3: DSpark draft head loaded (%d layers, block %d, window %d)\n", dsp->L, dsp->block, dsp->window);
    }
    Dspark* ds = dsp.get();
    if (ds) {   // with the head loaded, speculation is the default (measured best: 6 drafts, confidence floor 0.7)
        if (!spec_given) spec_mode = "dspark";
        if (!draft_given) spec_draft = std::min(6, kMaxWin - 1);
        ds->pmin_default = 0.7f;
        ds->pmin = ds->pmin_default;
    }
    Engine E;
    E.pf_gemm = pf_gemm_opt;
    E.pf_pre = pf_pre_opt;
    E.kv_half = kv_f16_opt;
    if (kv_f16_opt) std::fprintf(stderr, "strata-m3: K/V and indexer-key caches in FP16\n");
    E.init(m, ctx, chunk, expert_vram);
    if (ds) {
        E.feat_layers = ds->target_layers;
        E.feat = dalloc((size_t) std::max(chunk, kMaxWin) * ds->target_layers.size() * (size_t) m.N);
    }
#if defined(STRATA_M3_CPU_POOL)
    if (cpu_experts) {
        if (!m.pack_base) {
            std::fprintf(stderr, "strata-m3: --cpu-experts needs --preload-experts (the CPU reads the pack)\n");
            return 2;
        }
        E.init_cpu_lane(cpu_threads, admit, admit_freq);
    }
#else
    if (cpu_experts) {
        std::fprintf(stderr, "strata-m3: this build has no CPU expert path (STRATA_NATIVE_EXPERTS=OFF)\n");
        return 2;
    }
#endif
    if (const char* rt = std::getenv("STRATA_M3_ROUTE_TRACE")) E.route_trace = std::fopen(rt, "w");
    FILE* dump = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    if (!dump_logits.empty() && !dump) {
        std::fprintf(stderr, "strata-m3: cannot write %s\n", dump_logits.c_str());
        return 1;
    }
    const int32_t eos = (int32_t) (m.file->meta().get("tokenizer.ggml.eos_token_id")
                                       ? m.file->meta().get("tokenizer.ggml.eos_token_id")->u : 200020);
    std::vector<float> lg;

    // ---- one request: prefill, cache warm-up, then decode (plain, teacher-forced or speculative with an n-gram drafter)
    struct RunSpec {
        std::vector<int32_t> prompt;
        int n_gen = 32;
        std::vector<int32_t> forced;
        FILE* dump = nullptr;
        std::string spec = "none";     // none | ngram
        int draft = 2;                 // drafted tokens per step (window = 1 + draft)
        int ngram_max = 3, ngram_min = 1;
        int chunk_req = 0;             // prefill chunk for this request (0: the engine's)
        int pre = -1;                  // -1: as started; 0/1: the expert prefetch off / on
        int pf = -1;                   // -1: as started; 0/1: the FP16 GEMM prefill off / on
    };
    auto propose = [&](const std::vector<int32_t>& hist, int32_t cur, const RunSpec& rs) {
        // drafted tokens: what followed the most recent earlier occurrence of the longest suffix (ending in `cur`) of the text
        std::vector<int32_t> text = hist;
        text.push_back(cur);
        const int L = (int) text.size();
        for (int n = rs.ngram_max; n >= rs.ngram_min; --n) {
            if (L < n + 1) continue;
            for (int s0 = L - n - 1; s0 >= 0; --s0) {
                bool same = true;
                for (int k = 0; k < n && same; ++k) same = text[(size_t) s0 + k] == text[(size_t) (L - n + k)];
                if (!same) continue;
                std::vector<int32_t> d;
                for (int k = s0 + n; k < L && (int) d.size() < rs.draft; ++k) d.push_back(text[(size_t) k]);
                if (!d.empty()) return d;
            }
        }
        return std::vector<int32_t>();
    };
    auto do_run = [&](int run, int runs_total, const RunSpec& rs) {
    const std::vector<int32_t>& prompt = rs.prompt;
    const int n_gen = rs.n_gen;
    // the cache (slots, LRU) carries over between runs; the counters and the position restart
    E.pos = 0;
    E.hits = E.misses = 0;
    E.copy_bytes = E.copy_ms = 0;
    E.ph_sync = 0;
    E.ph_fwd = E.ph_emb = E.ph_attn_enq = E.ph_head = 0;
    E.route_dropped = E.route_total = 0;
    if (ds) { ds->ms_draft = ds->ms_inject = 0; ds->ms_dl = ds->ms_dh = ds->ms_dt = 0; ds->conf_cut = 0; }
    E.pt_attn = E.pt_dense = E.pt_route = E.pt_copy = E.pt_gemm = E.pt_other = E.pt_deq = E.pt_mm = E.pt_conv = 0;
    E.pf_marks.clear();
#if defined(STRATA_M3_CPU_POOL)
    E.cpu_experts = E.gpu_experts = E.admitted = 0;
    E.cpu_wall_ms = E.ph_pre = E.ph_adm = E.ph_act = E.ph_gu = E.ph_hq = E.ph_dn = E.ph_acc = 0;
    if (E.cpu_lane) {   // a new request: its own routing statistics
        std::fill(E.use_count.begin(), E.use_count.end(), 0);
        std::fill(E.pf_count.begin(), E.pf_count.end(), 0);
        E.pf_tokens = 0;
    }
#endif
    if (runs_total > 1) std::printf("M3_RUN %d of %d\n", run + 1, runs_total);
    const int route_topk_saved = E.route_topk;
    const auto t0 = Clock::now();
    E.route_topk = E.top_k_model();   // the prompt's prefill stays exact
    const int miss_keep_saved = E.miss_keep;
    E.miss_keep = -1;
    const std::vector<int> lay_topk_saved = E.lay_topk;
    const std::vector<float> lay_minw_saved = E.lay_minw;
    E.lay_topk.clear();
    E.lay_minw.clear();
    const int chunk_use = rs.chunk_req > 0 ? std::min(rs.chunk_req, chunk) : chunk;
    if (rs.pf >= 0 && (E.wscr || rs.pf == 0)) E.pf_gemm = rs.pf != 0;
    if (rs.pre >= 0) E.pf_pre = rs.pre != 0;
    for (size_t at = 0; at < prompt.size(); at += (size_t) chunk_use) {
        const int T = (int) std::min(prompt.size() - at, (size_t) chunk_use);
        const int p0 = E.pos;
        E.forward(prompt.data() + at, T, lg);
        if (ds) ds->inject(E, T, p0);
    }
    E.route_topk = route_topk_saved;
    E.lay_topk = lay_topk_saved;
    E.lay_minw = lay_minw_saved;
    E.miss_keep = miss_keep_saved;
    const double pp_ms = ms_since(t0);
    const long long pp_miss = E.misses;
    const double pp_copy = E.copy_bytes;
    // from here on the counters describe the decode only
    {
        const char* pv = std::getenv("STRATA_M3_PROF");
        g_prof.on = pv && std::atoi(pv) >= 1;
        E.prof_gpu = pv && std::atoi(pv) >= 2;
        E.prof_all = pv && std::atoi(pv) >= 3;
        E.prof_draft = pv && std::atoi(pv) >= 4;
        E.gcalls = 0;
        E.gacc.clear();
        E.gn = 0;
    }
    g_prof.acc.clear();
    E.ph_sync = 0;
    E.ph_fwd = E.ph_emb = E.ph_attn_enq = E.ph_head = 0;
#if defined(STRATA_M3_CPU_POOL)
    E.cpu_experts = E.gpu_experts = E.admitted = 0;
    E.cpu_wall_ms = E.ph_pre = E.ph_adm = E.ph_act = E.ph_gu = E.ph_hq = E.ph_dn = E.ph_acc = 0;
#endif
    const long long dec_hits0 = E.hits, dec_miss0 = E.misses;
#if defined(STRATA_M3_CPU_POOL)
    if (E.cpu_lane && strata_env_truthy_default("STRATA_M3_WARM", 1)) {
        // STRATA_M3_PRIOR_LIST=a,b,c: the gate prior strength of run 0, 1, 2 ... (cycled); default 0.2
        double prior = 0.2;
        if (const char* pl = std::getenv("STRATA_M3_PRIOR_LIST")) {
            std::vector<double> vals;
            for (const char* q = pl; *q;) { vals.push_back(std::atof(q)); while (*q && *q != ',') ++q; if (*q == ',') ++q; }
            if (!vals.empty()) prior = vals[(size_t) run % vals.size()];
        }
        std::fprintf(stderr, "strata-m3: admission gate prior strength %.2f\n", prior);
        E.warm_cache(n_gen, prior);
        std::fill(E.pf_count.begin(), E.pf_count.end(), 0);   // the next run's prefill counts afresh
        E.pf_tokens = 0;
    }
#endif
#if !defined(__HIP_PLATFORM_AMD__)
    const bool nsys_range = strata_env_truthy_one("STRATA_M3_NSYS") != 0;   // nsys --capture-range=cudaProfilerApi: the decode only
    if (nsys_range) cudaProfilerStart();
#endif
    std::vector<int32_t> out;
    std::vector<std::pair<float, int>> top;
    for (int i = 0; i < m.n_vocab; ++i) top.emplace_back(lg[(size_t) i], i);
    std::partial_sort(top.begin(), top.begin() + 5, top.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::printf("M3_FIRST_TOP5");
    for (int i = 0; i < 5; ++i) std::printf(" %d:%.4f", top[(size_t) i].second, top[(size_t) i].first);
    std::printf("\n");
    const auto t1 = Clock::now();
    int force_n = 0, force_agree = 0;
    long long spec_steps = 0, spec_drafted = 0, spec_accepted = 0;
    const bool speculate = (rs.spec == "ngram" || rs.spec == "dspark") && rs.forced.empty() && (rs.spec != "dspark" || ds) && rs.draft > 0 && rs.draft < kMaxWin;
    std::vector<int32_t> hist = prompt;   // the text so far (prompt + emitted), for the drafter
    int32_t cur = argmax(lg);             // the next token to emit and feed: the prompt's own prediction first
    if (rs.spec == "window" && !rs.forced.empty()) {
        // test mode: the forced tokens go through the multi-token window path (1 + draft rows per step) instead of one at a
        // time; the dump holds the same per-step logits as the single-token run, for a numerical comparison of the two paths
        const int W0 = 1 + std::max(1, std::min(rs.draft, kMaxWin - 1));
        int written = 0;
        if (rs.dump && run == 0) { std::fwrite(lg.data(), 4, lg.size(), rs.dump); ++written; }
        ++force_n;
        force_agree += cur == rs.forced[0];
        for (int i = 0; i < n_gen && (size_t) i < rs.forced.size();) {
            const int W = std::min<int>(W0, std::min<int>(n_gen - i, (int) rs.forced.size() - i));
            E.route_step = i;
            E.forward(rs.forced.data() + i, W, lg, rs.dump != nullptr, true);
            for (int r = 0; r < W; ++r) {
                if ((size_t) (i + r + 1) < rs.forced.size() && i + r + 1 < n_gen) {
                    ++force_n;
                    force_agree += E.arg_rows[r] == rs.forced[(size_t) (i + r + 1)];
                }
                out.push_back(rs.forced[(size_t) (i + r)]);
            }
            if (rs.dump && run == 0)
                for (int r = 0; r < W && written < n_gen; ++r, ++written)
                    std::fwrite(lg.data() + (size_t) r * m.n_vocab, 4, (size_t) m.n_vocab, rs.dump);
            i += W;
        }
    } else if (!speculate) {
        for (int i = 0; i < n_gen; ++i) {
            if (rs.dump && run == 0) std::fwrite(lg.data(), 4, lg.size(), rs.dump);
            E.route_step = i;
            const int32_t pred = i == 0 ? cur : E.last_argmax;
            if (!rs.forced.empty() && (size_t) i < rs.forced.size()) {
                ++force_n;
                force_agree += pred == rs.forced[(size_t) i];
            }
            const int32_t next = (!rs.forced.empty() && (size_t) i < rs.forced.size()) ? rs.forced[(size_t) i] : pred;
            out.push_back(next);
            if ((next == eos && !ignore_eos) || i + 1 == n_gen) break;
            E.forward(&next, 1, lg, rs.dump != nullptr);   // the logits cross to the host only when they are dumped
        }
    } else {
        // speculative decoding, greedy verification: the window is [cur, d1..dD]; the model's prediction after each row
        // is compared with the next drafted token; the accepted prefix and one more token are kept.  Rows past the
        // accepted prefix wrote K/V and indexer entries at positions that every later query masks and later steps overwrite.
        while ((int) out.size() < n_gen) {
            E.route_step = (long long) spec_steps;
            std::vector<int32_t> d = rs.spec == "dspark" ? ds->draft(E, cur, E.pos, rs.draft) : propose(hist, cur, rs);
            const int D = (int) d.size();
            int32_t win[kMaxWin];
            win[0] = cur;
            for (int k = 0; k < D; ++k) win[k + 1] = d[(size_t) k];
            const int pos0 = E.pos;
            E.forward(win, 1 + D, lg, false, true);
            ++spec_steps;
            spec_drafted += D;
            out.push_back(cur);
            hist.push_back(cur);
            if (cur == eos && !ignore_eos) { E.pos = pos0 + 1; break; }
            int a = 0;
            while (a < D && (int) out.size() < n_gen && E.arg_rows[a] == d[(size_t) a]) {
                out.push_back(d[(size_t) a]);
                hist.push_back(d[(size_t) a]);
                ++a;
                if (d[(size_t) a - 1] == eos && !ignore_eos) break;
            }
            spec_accepted += a;
            E.pos = pos0 + 1 + a;           // the rows past the accepted prefix are dropped
            if (ds) ds->inject(E, a + 1, pos0);   // the accepted rows' target features become draft context
            cur = E.arg_rows[a];
            if (!out.empty() && out.back() == eos && !ignore_eos) break;
        }
    }
    const double tg_ms = ms_since(t1);
#if !defined(__HIP_PLATFORM_AMD__)
    if (nsys_range) cudaProfilerStop();
#endif
    if (force_n) std::printf("M3_FORCE teacher-forced %d steps, the engine's own argmax agreed at %d\n", force_n, force_agree);
    if (speculate)
        std::printf("M3_SPEC %s drafter, up to %d drafts per step: %lld steps, %lld drafted, %lld accepted (%.1f %%), "
                    "%.2f tokens per step\n", rs.spec.c_str(), rs.draft, spec_steps, spec_drafted, spec_accepted,
                    100.0 * (double) spec_accepted / (double) std::max<long long>(1, spec_drafted),
                    (double) out.size() / (double) std::max<long long>(1, spec_steps));
    if (E.pfprof)
        for (auto& a : E.pf_marks) std::printf("M3_PFPROF_ATTN %-48s %9.0f ms\n", a.first, a.second);
    if (E.pfprof)
        std::printf("M3_PFPROF_GEMM (inside the above, serialized) f32->f16 %.0f | dequant %.0f | cublas %.0f ms\n", E.pt_conv, E.pt_deq, E.pt_mm);
    if (E.pfprof)
        std::printf("M3_PFPROF prefill ms: attention %.0f | dense ffn + shared + router %.0f | expert copy waits %.0f | expert gemm %.0f | other %.0f\n",
                    E.pt_attn, E.pt_dense, E.pt_copy, E.pt_gemm, E.pt_other);
    if (ds && speculate)
        std::printf("M3_DSPARK draft pass %.1f ms (%.2f ms per call), inject %.1f ms over the run (host enqueue + waits), %lld drafts cut by confidence\n", ds->ms_draft,
                    ds->ms_draft / (double) std::max<long long>(1, spec_steps), ds->ms_inject, ds->conf_cut);
        if (E.prof_all && E.lw_n)
            std::printf("M3_LANE_HOST ms per window layer (%lld layers): filter+shared %.3f | jobs+classify %.3f | cpu quant+start %.3f | gpu enqueue %.3f | admit %.3f | gu join %.3f | quant-h+down %.3f | accumulate+add %.3f\n", E.lw_n, E.lw[0] / (double) E.lw_n, E.lw[1] / (double) E.lw_n, E.lw[2] / (double) E.lw_n, E.lw[3] / (double) E.lw_n, E.lw[4] / (double) E.lw_n, E.lw[5] / (double) E.lw_n, E.lw[6] / (double) E.lw_n, E.lw[7] / (double) E.lw_n);
        if (E.prof_draft)
            std::printf("M3_DSPARK_GPU per call: layers %.2f ms, norm+lm_head %.2f ms, markov+argmax+conf %.2f ms\n", ds->ms_dl / (double) std::max<long long>(1, spec_steps),
                        ds->ms_dh / (double) std::max<long long>(1, spec_steps), ds->ms_dt / (double) std::max<long long>(1, spec_steps));
    if (E.route_total)
        std::printf("M3_ROUTE lever: top-%d, min weight %.2f x max, renorm %d: dropped %lld of %lld routed experts (%.1f %%)\n",
                    E.route_topk, E.route_min_w, (int) E.route_renorm, E.route_dropped, E.route_total,
                    100.0 * (double) E.route_dropped / (double) E.route_total);
    std::printf("M3_TOKENS");
    for (int32_t t : out) std::printf(" %d", t);
    std::printf("\n");
    const int steps = std::max<int>(1, (int) out.size() - 1);
    if (m.pack_base)
        std::printf("M3_PACK %.1f GiB preloaded in %.1f s (%.2f GB/s), %.1f GiB page-locked in %.1f s\n",
                    (double) m.pack_bytes / (1 << 30), m.pack_read_s, (double) m.pack_bytes / 1e9 / m.pack_read_s,
                    (double) m.pack_pinned / (1 << 30), m.pack_pin_s);
#if defined(STRATA_M3_CPU_POOL)
    if (E.cpu_lane && E.dev1 >= 0 && E.prof_all && E.g1ev.size() >= 2) {
        double tot = 0; int n = 0;
        for (size_t i = 0; i + 1 < E.g1ev.size(); i += 2) { float ms = 0; if (cudaEventElapsedTime(&ms, E.g1ev[i], E.g1ev[i + 1]) == cudaSuccess) { tot += ms; ++n; } }
        std::printf("M3_GPU1_CHAIN first second-GPU stream per window layer: %.3f ms over %d layers (%lld jobs in all)\n", n ? tot / n : 0.0, n, E.g1_jobs);
    }
    if (E.cpu_lane && E.dev1 >= 0 && E.gpu1_check)
        std::printf("M3_GPU1_CHECK %lld second-GPU jobs recomputed on GPU 0, %lld differ bitwise\n", E.chk_jobs, E.chk_bad);
    if (E.cpu_lane && E.dev1 >= 0)
        std::printf("M3_GPU1 %d slots on the second GPU, %lld of the GPU expert uses ran there\n", E.n_dev1, E.gpu1_experts);
    if (E.cpu_lane)
        std::printf("M3_CPU experts on the CPU %lld, on the GPU %lld, admitted to VRAM %lld, CPU lane wall %.1f ms\n"
                    "M3_PHASES ms over the run: wait-for-GPU-chain %.1f | gpu-enqueue %.1f | admit %.1f | cpu: quant %.1f "
                    "gate/up %.1f quant-h %.1f down %.1f combine+h2d %.1f\n"
                    "M3_STEP ms over the decode (%d steps): forward wall %.1f = embed enqueue %.1f + attention enqueue %.1f + "
                    "head (norm, lm_head, argmax, sync) %.1f + the rest %.1f (layers' norms, ffn, CPU lane waits)\n",
                    E.cpu_experts, E.gpu_experts, E.admitted, E.cpu_wall_ms, E.ph_sync, E.ph_pre, E.ph_adm, E.ph_act,
                    E.ph_gu, E.ph_hq, E.ph_dn, E.ph_acc, steps, E.ph_fwd, E.ph_emb, E.ph_attn_enq, E.ph_head,
                    E.ph_fwd - E.ph_emb - E.ph_attn_enq - E.ph_head);
#endif
    if (g_prof.on) {
        std::printf("M3_PROF host enqueue ms over the decode (%d steps):", steps);
        for (auto& a : g_prof.acc) std::printf("\n  %-48s %8.1f ms  (%.3f ms/step)", a.first, a.second, a.second / steps);
        std::printf("\n");
    }
    if (E.prof_gpu) {
        std::printf("M3_PROF_GPU device-timeline ms between marks, summed over the decode steps' layers up to the router:");
        double sum = 0;
        for (auto& a : E.gacc) {
            std::printf("\n  %-48s %8.1f ms  (%.3f ms/step)", a.first, a.second, a.second / steps);
            sum += a.second;
        }
        std::printf("\n  %-48s %8.1f ms  (%.3f ms/step)\n", "total", sum, sum / steps);
        if (E.prof_all && E.gcalls)
            std::printf("  per layer call (%lld calls): %.3f ms; per 57-layer step: %.1f ms\n", E.gcalls, sum / (double) E.gcalls, sum / (double) E.gcalls * 57.0);
    }
    {
        long long hit_n = E.hits - dec_hits0, all_n = E.hits - dec_hits0 + E.misses - dec_miss0;
#if defined(STRATA_M3_CPU_POOL)
        if (E.cpu_lane) { hit_n = E.gpu_experts; all_n = E.gpu_experts + E.cpu_experts; }   // CPU-computed = not resident
#endif
        std::printf("M3_DECODE_CACHE resident %lld of %lld expert uses (hit rate %.1f %%)\n", hit_n, all_n,
                    100.0 * (double) hit_n / (double) std::max<long long>(1, all_n));
    }
    std::printf("M3_STATS prompt %zu tokens %.1f ms (%.2f tok/s, %lld expert copies %.2f GiB); decode %zu tokens "
                "%.1f ms (%.3f tok/s); expert hits %lld misses %lld, copied %.2f GiB in %.1f ms host time\n",
                prompt.size(), pp_ms, (double) prompt.size() * 1000.0 / pp_ms, pp_miss, pp_copy / (1 << 30),
                out.size(), tg_ms, out.size() > 1 ? (double) steps * 1000.0 / tg_ms : 0.0, E.hits, E.misses,
                E.copy_bytes / (1 << 30), E.copy_ms);
    std::fflush(stdout);
    };

    RunSpec base;
    base.prompt = prompt;
    base.n_gen = n_gen;
    base.forced = force_file.empty() ? std::vector<int32_t>() : parse_ids(read_file(force_file));
    base.dump = dump;
    base.spec = spec_mode;
    base.draft = spec_draft;
    E.route_topk = route_topk_opt;
    E.route_min_w = route_min_w_opt;
    E.route_renorm = route_renorm_opt;
    E.miss_keep = miss_keep_opt;
    if (!repl) {
        for (int run = 0; run < runs; ++run) do_run(run, runs, base);
    } else {
        // interactive mode: the pack stays loaded; each line "run key=value ..." is one request, "quit" ends
        // keys: prompt=FILE n=N spec=none|ngram draft=D topk=K minw=W renorm=0|1 force=FILE dump=FILE admit=N
        std::fprintf(stderr, "strata-m3: READY (commands on stdin: run key=value ... | quit)\n");
        std::printf("M3_READY\n");
        std::fflush(stdout);
        std::string line;
        int serial = 0;
        while (std::getline(std::cin, line)) {
            if (line == "quit") break;
            if (line.rfind("run", 0) != 0) continue;
            RunSpec rs = base;
            rs.chunk_req = 0;
            rs.pf = -1;
            rs.pre = -1;
            rs.dump = nullptr;
            rs.forced.clear();
            E.route_topk = route_topk_opt;
            E.route_min_w = route_min_w_opt;
            E.route_renorm = route_renorm_opt;
            E.lay_topk.clear();
            E.lay_minw.clear();
            E.miss_keep = miss_keep_opt;
            E.miss_minw = 0.f;
            E.sub_mode = 0;
            if (ds) ds->pmin = ds->pmin_default;
            std::string dump_path;
            bool ok = true;
            size_t at = 3;
            while (at < line.size()) {
                while (at < line.size() && line[at] == ' ') ++at;
                size_t end = line.find(' ', at);
                if (end == std::string::npos) end = line.size();
                const std::string kv = line.substr(at, end - at);
                at = end;
                const size_t eq = kv.find('=');
                if (eq == std::string::npos) continue;
                const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
                if (k == "prompt") rs.prompt = parse_ids(read_file(v));
                else if (k == "n") rs.n_gen = std::atoi(v.c_str());
                else if (k == "spec") rs.spec = v;
                else if (k == "draft") rs.draft = std::atoi(v.c_str());
                else if (k == "lay") {   // L:minw:topk,L:minw:topk,...
                    size_t q = 0;
                    while (q < v.size()) {
                        size_t e2 = v.find(',', q);
                        if (e2 == std::string::npos) e2 = v.size();
                        int ll = 0, tkk = -1;
                        float mww = -1.f;
                        if (std::sscanf(v.substr(q, e2 - q).c_str(), "%d:%f:%d", &ll, &mww, &tkk) == 3) E.set_lay(ll, mww, tkk);
                        q = e2 + 1;
                    }
                } else if (k == "layfile") {   // lines "layer minw topk"
                    const std::string txt = read_file(v);
                    size_t q = 0;
                    while (q < txt.size()) {
                        size_t e2 = txt.find('\n', q);
                        if (e2 == std::string::npos) e2 = txt.size();
                        int ll = 0, tkk = -1;
                        float mww = -1.f;
                        if (std::sscanf(txt.substr(q, e2 - q).c_str(), "%d %f %d", &ll, &mww, &tkk) == 3) E.set_lay(ll, mww, tkk);
                        q = e2 + 1;
                    }
                }
                else if (k == "missk") E.miss_keep = std::atoi(v.c_str());
                else if (k == "missw") E.miss_minw = (float) std::atof(v.c_str());
                else if (k == "sub") E.sub_mode = std::atoi(v.c_str());
                else if (k == "pmin") { if (ds) ds->pmin = (float) std::atof(v.c_str()); }
                else if (k == "topk") E.route_topk = std::atoi(v.c_str());
                else if (k == "minw") E.route_min_w = (float) std::atof(v.c_str());
                else if (k == "renorm") E.route_renorm = std::atoi(v.c_str()) != 0;
                else if (k == "chunk") rs.chunk_req = std::atoi(v.c_str());
                else if (k == "pf") rs.pf = std::atoi(v.c_str());
                else if (k == "pre") rs.pre = std::atoi(v.c_str());
                else if (k == "force") rs.forced = parse_ids(read_file(v));
                else if (k == "dump") dump_path = v;
                else if (k == "admit") E.admit_per_layer = std::atoi(v.c_str());
                else if (k == "policy") E.admit_freq = v != "lru";
                else { std::printf("M3_ERROR unknown key %s\n", k.c_str()); ok = false; }
            }
            if (rs.prompt.empty() || (int) rs.prompt.size() + rs.n_gen > ctx) {
                std::printf("M3_ERROR prompt (%zu) + n (%d) does not fit the context %d\n", rs.prompt.size(), rs.n_gen, ctx);
                ok = false;
            }
            if (!ok) { std::fflush(stdout); continue; }
            FILE* df = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "wb");
            rs.dump = df;
            std::printf("M3_REQUEST %d: %s\n", ++serial, line.c_str());
            do_run(0, 1, rs);
            if (df) std::fclose(df);
            std::printf("M3_REQUEST_DONE %d\n", serial);
            std::fflush(stdout);
        }
    }
    if (dump) std::fclose(dump);
    if (E.route_trace) { std::fclose(E.route_trace); E.route_trace = nullptr; }
    return 0;
}
