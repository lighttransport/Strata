// src/kernels/cuda/muse.cu - Muse Spark Open's block kernels (strata/kernels/muse.hpp).  Naive first: correctness
// against strata/muse/reference.hpp through the whole block, speed later.
#include "strata/kernels/muse.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace strata::kernels {
namespace {

constexpr int kThreads = 256;

// sum over the block; every thread gets the result.  `red` holds kThreads / 32 floats.
__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = (blockDim.x + 31) / 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0;
    for (int w = 0; w < warps; ++w) s += red[w];
    return s;
}

__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = (blockDim.x + 31) / 32;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float m = -INFINITY;
    for (int w = 0; w < warps; ++w) m = fmaxf(m, red[w]);
    return m;
}

__device__ inline float sigmoidf(float v) { return 1.f / (1.f + expf(-v)); }

__global__ void rms_norm_k(const float* x, const float* pre, const float* post, float* y, int cols, float eps) {
    __shared__ float red[kThreads / 32];
    const float* xr = x + (size_t) blockIdx.x * cols;
    float* yr = y + (size_t) blockIdx.x * cols;
    float ss = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = xr[i] * (pre ? pre[i] : 1.f);
        ss += v * v;
    }
    const float s = rsqrtf(block_sum(ss, red) / (float) cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = xr[i] * (pre ? pre[i] : 1.f);
        yr[i] = v * s * (post ? post[i] : 1.f);
    }
}

__global__ void rope_k(float* x, int head_dim, const int32_t* pos, float base) {
    const int i = threadIdx.x;
    if (i >= head_dim / 2) return;
    float* h = x + (size_t) blockIdx.x * head_dim;
    const float theta = (float) pos[0] * powf(base, -2.f * (float) i / (float) head_dim);
    const float c = cosf(theta), s = sinf(theta);
    const float a = h[2 * i], b = h[2 * i + 1];
    h[2 * i] = a * c - b * s;
    h[2 * i + 1] = a * s + b * c;
}

__global__ void kv_append_k(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int width, const int32_t* pos,
                            int ring) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int64_t slot = ring ? pos[0] % ring : pos[0];
    kc[slot * width + i] = f16_from_f32(k[i]);
    vc[slot * width + i] = f16_from_f32(v[i]);
}

// One block per query head, 256 threads.  Keys in chunks of 256: each thread scores one key, the block keeps an
// online softmax (running max and sum), and threads 0..63 accumulate one output dimension each.
__global__ void attention_k(const float* q, const uint16_t* kc, const uint16_t* vc, float* out, int q_heads,
                            int kv_heads, const int32_t* pos, int ring, float scale) {
    constexpr int HD = 64;
    __shared__ float qs[HD];
    __shared__ float p[kThreads];
    __shared__ float red[kThreads / 32];
    const int h = blockIdx.x, kvh = h / (q_heads / kv_heads), width = kv_heads * HD;
    if (threadIdx.x < HD) qs[threadIdx.x] = q[(size_t) h * HD + threadIdx.x];
    __syncthreads();
    const int n = ring ? min(pos[0] + 1, ring) : pos[0] + 1;
    float m = -INFINITY, l = 0, acc = 0;
    for (int c0 = 0; c0 < n; c0 += kThreads) {
        const int j = c0 + (int) threadIdx.x;
        float s = -INFINITY;
        if (j < n) {
            const uint16_t* kr = kc + (size_t) j * width + kvh * HD;
            float d = 0;
            for (int i = 0; i < HD; ++i) d += qs[i] * f32_from_f16(kr[i]);
            s = d * scale;
        }
        const float m_new = fmaxf(m, block_max(s, red));
        const float e = j < n ? expf(s - m_new) : 0.f;
        const float factor = expf(m - m_new);   // m = -inf on the first chunk: factor 0
        l = l * factor + block_sum(e, red);
        p[threadIdx.x] = e;
        __syncthreads();
        if (threadIdx.x < HD) {
            acc *= factor;
            const int count = min(kThreads, n - c0);
            for (int t = 0; t < count; ++t) acc += p[t] * f32_from_f16(vc[(size_t) (c0 + t) * width + kvh * HD + threadIdx.x]);
        }
        m = m_new;
        __syncthreads();
    }
    if (threadIdx.x < HD) out[(size_t) h * HD + threadIdx.x] = acc / l;
}

// Split-K: block (h, split) runs the same online softmax over its share of the n cached keys, about n / splits
// rounded up to whole 256-key chunks and computed from the device position, so the launch shape is fixed and a
// captured graph replays for every token.  It writes (max, sum, acc[64]) to part[h][split]; a split with no keys
// writes (-inf, 0, 0).
__global__ void attention_split_k(const float* q, const uint16_t* kc, const uint16_t* vc, float* part, int q_heads,
                                  int kv_heads, const int32_t* pos, int ring, float scale) {
    constexpr int HD = 64;
    __shared__ float qs[HD];
    __shared__ float p[kThreads];
    __shared__ float red[kThreads / 32];
    const int h = blockIdx.x, split = blockIdx.y, splits = gridDim.y, kvh = h / (q_heads / kv_heads),
              width = kv_heads * HD;
    if (threadIdx.x < HD) qs[threadIdx.x] = q[(size_t) h * HD + threadIdx.x];
    __syncthreads();
    const int n = ring ? min(pos[0] + 1, ring) : pos[0] + 1;
    const int chunks = (n + kThreads - 1) / kThreads, per = (chunks + splits - 1) / splits;
    const int j0 = min(n, split * per * kThreads), j1 = min(n, (split + 1) * per * kThreads);
    float m = -INFINITY, l = 0, acc = 0;
    for (int c0 = j0; c0 < j1; c0 += kThreads) {
        const int j = c0 + (int) threadIdx.x;
        float s = -INFINITY;
        if (j < j1) {
            const uint16_t* kr = kc + (size_t) j * width + kvh * HD;
            float d = 0;
            for (int i = 0; i < HD; ++i) d += qs[i] * f32_from_f16(kr[i]);
            s = d * scale;
        }
        const float m_new = fmaxf(m, block_max(s, red));
        const float e = j < j1 ? expf(s - m_new) : 0.f;
        const float factor = expf(m - m_new);
        l = l * factor + block_sum(e, red);
        p[threadIdx.x] = e;
        __syncthreads();
        if (threadIdx.x < HD) {
            acc *= factor;
            const int count = min(kThreads, j1 - c0);
            for (int t = 0; t < count; ++t) acc += p[t] * f32_from_f16(vc[(size_t) (c0 + t) * width + kvh * HD + threadIdx.x]);
        }
        m = m_new;
        __syncthreads();
    }
    float* o = part + ((size_t) h * splits + split) * (HD + 2);
    if (threadIdx.x < HD) o[2 + threadIdx.x] = acc;
    if (threadIdx.x == 0) { o[0] = m; o[1] = l; }
}

// out[h] = sum_s acc_s e^(m_s - M) / sum_s l_s e^(m_s - M), M the largest m_s; one block of 64 threads per head.
__global__ void attention_combine_k(const float* part, float* out, int splits) {
    constexpr int HD = 64;
    const int h = blockIdx.x, i = threadIdx.x;
    const float* ph = part + (size_t) h * splits * (HD + 2);
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, ph[(size_t) s * (HD + 2)]);
    float l = 0, acc = 0;
    for (int s = 0; s < splits; ++s) {
        const float* o = ph + (size_t) s * (HD + 2);
        if (o[1] == 0.f) continue;   // no keys
        const float f = expf(o[0] - M);
        l += o[1] * f;
        acc += o[2 + i] * f;
    }
    out[(size_t) h * HD + i] = acc / l;
}

__global__ void head_norm_gate_k(float* a, const float* gate, int head_dim, float eps) {
    __shared__ float red[kThreads / 32];
    float* ah = a + (size_t) blockIdx.x * head_dim;
    const float* gh = gate + (size_t) blockIdx.x * head_dim;
    float ss = 0;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) ss += ah[i] * ah[i];
    const float s = rsqrtf(block_sum(ss, red) / (float) head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) ah[i] = ah[i] * s * sigmoidf(gh[i]);
}

__global__ void residual_k(float* x, const float* br, const float* alpha, const float* beta, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = alpha[i] * x[i] + beta[i] * br[i];
}

__global__ void gemv_f32_k(const float* w, const float* x, float* y, int n_in) {
    __shared__ float red[kThreads / 32];
    const float* row = w + (size_t) blockIdx.x * n_in;
    float s = 0;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) s += row[i] * x[i];
    s = block_sum(s, red);
    if (threadIdx.x == 0) y[blockIdx.x] = s;
}

__global__ void logit_cap_k(float* y, int n, float scale, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = y[i] * scale;
    y[i] = cap > 0 ? cap * tanhf(v / cap) : v;
}

inline cudaStream_t st(void* s) { return (cudaStream_t) s; }
inline int blocks(int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

void muse_rms_norm(const float* x, const float* pre, const float* post, float* y, int rows, int cols, float eps,
                   void* stream) {
    rms_norm_k<<<rows, kThreads, 0, st(stream)>>>(x, pre, post, y, cols, eps);
}

void muse_rope(float* x, int rows, int head_dim, const int32_t* pos, float base, void* stream) {
    rope_k<<<rows, head_dim / 2, 0, st(stream)>>>(x, head_dim, pos, base);
}

void muse_kv_append(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, int width,
                    const int32_t* pos, int ring, void* stream) {
    kv_append_k<<<blocks(width), kThreads, 0, st(stream)>>>(k, v, k_cache, v_cache, width, pos, ring);
}

void muse_attention(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int q_heads,
                    int kv_heads, int head_dim, const int32_t* pos, int ring, float scale, void* stream, int splits,
                    float* scratch) {
    (void) head_dim;   // 64 (the kernel's constant; the caller checks the model's)
    if (splits <= 1) {
        attention_k<<<q_heads, kThreads, 0, st(stream)>>>(q, k_cache, v_cache, out, q_heads, kv_heads, pos, ring, scale);
        return;
    }
    attention_split_k<<<dim3(q_heads, splits), kThreads, 0, st(stream)>>>(q, k_cache, v_cache, scratch, q_heads,
                                                                         kv_heads, pos, ring, scale);
    attention_combine_k<<<q_heads, 64, 0, st(stream)>>>(scratch, out, splits);
}

size_t muse_attention_scratch_floats(int q_heads, int head_dim, int splits) {
    return splits <= 1 ? 0 : (size_t) q_heads * splits * (head_dim + 2);
}

void muse_head_norm_gate(float* a, const float* gate, int heads, int head_dim, float eps, void* stream) {
    head_norm_gate_k<<<heads, 64, 0, st(stream)>>>(a, gate, head_dim, eps);
}

void muse_residual(float* x, const float* branch, const float* alpha, const float* beta, int n, void* stream) {
    residual_k<<<blocks(n), kThreads, 0, st(stream)>>>(x, branch, alpha, beta, n);
}

void muse_gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, void* stream) {
    gemv_f32_k<<<n_out, kThreads, 0, st(stream)>>>(w, x, y, n_in);
}

void muse_logit_cap(float* logits, int n, float scale, float softcap, void* stream) {
    logit_cap_k<<<blocks(n), kThreads, 0, st(stream)>>>(logits, n, scale, softcap);
}

}  // namespace strata::kernels
