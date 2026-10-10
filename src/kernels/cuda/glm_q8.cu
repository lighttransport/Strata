#include "strata/kernels/glm_q8.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int kBlock = 34;
void ok(cudaError_t e, const char *what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("GLM Q8_0 ") + what + ": " + cudaGetErrorString(e));
}
__device__ float scale(const uint8_t *block) { return __half2float(*reinterpret_cast<const __half *>(block)); }

// One warp per row; lane l takes element l of every 32-value block, so a block is one coalesced load.
template <int NT>
__global__ void heads_gemv(const uint8_t *w, const float *x, float *y, int in, int out, int rows, int ld_x, int ld_y,
                           int tokens) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int blocks = in / 32;
    const uint8_t *r = w + (size_t)row * blocks * kBlock;
    const float *xs = x + (size_t)(row / out) * in + lane;
    float acc[NT] = {};
    for (int b = 0; b < blocks; ++b) {
        const float value = scale(r + b * kBlock) * (float)(int8_t)r[b * kBlock + 2 + lane];
#pragma unroll
        for (int t = 0; t < NT; ++t)
            if (t < tokens) acc[t] = fmaf(value, xs[(size_t)t * ld_x + b * 32], acc[t]);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
        for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
        if (!lane && t < tokens) y[(size_t)t * ld_y + row] = v;
    }
}
// One token (STRATA_GLM_HEADS_BLOCKS=1): a thread per Q8_0 block reads it with sixteen 2-byte loads, and the
// `blocks` consecutive threads of a row (a power of two up to 32) reduce with shuffles. The one-warp-per-row
// kernel above reads one byte per lane per block; its rows here are only 8 or 16 blocks long.
__global__ void heads_gemv_blocks(const uint8_t *w, const float *x, float *y, int in, int out, int rows) {
    const int blocks = in / 32;
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = id / blocks;
    const int b = (int)(id % blocks);
    float acc = 0.f;
    if (row < (size_t)rows) {
        const auto *h = reinterpret_cast<const unsigned short *>(w + (row * blocks + b) * kBlock);
        const float *xs = x + (row / out) * in + b * 32;
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned short v = h[1 + i];
            sum = fmaf((float)(int8_t)(v & 0xff), xs[2 * i], sum);
            sum = fmaf((float)(int8_t)(v >> 8), xs[2 * i + 1], sum);
        }
        acc = __half2float(__ushort_as_half(h[0])) * sum;
    }
    for (int delta = blocks / 2; delta; delta >>= 1) acc += __shfl_down_sync(0xffffffff, acc, delta, blocks);
    if (row < (size_t)rows && b == 0) y[row] = acc;
}
// Grid (rows, 32 slices); each 256-thread block reduces its slice of a row for every token.
template <int NT>
__global__ void rows_partial(const uint8_t *w, const float *x, float *partial, int in, int rows, int tokens) {
    const int row = blockIdx.x, slice = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int blocks = in / 32, per = blocks / 32, first = slice * per;
    const uint8_t *r = w + (size_t)row * blocks * kBlock;
    float acc[NT] = {};
    for (int b = first + warp; b < first + per; b += 8) {
        const float value = scale(r + b * kBlock) * (float)(int8_t)r[b * kBlock + 2 + lane];
#pragma unroll
        for (int t = 0; t < NT; ++t)
            if (t < tokens) acc[t] = fmaf(value, x[(size_t)t * in + b * 32 + lane], acc[t]);
    }
    __shared__ float sums[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
        for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
        if (!lane) sums[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < tokens) {
        float v = 0;
        for (int i = 0; i < 8; ++i) v += sums[threadIdx.x][i];
        partial[((size_t)threadIdx.x * rows + row) * 32 + slice] = v;
    }
}
__global__ void rows_reduce(const float *partial, float *y) {
    const int lane = threadIdx.x;
    float v = partial[(size_t)blockIdx.x * 32 + lane];
    for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
    if (!lane) y[blockIdx.x] = v;
}
// One 256-thread block per row: every thread keeps several float4 loads in flight.
template <int NT>
__global__ void f32_rows(const float4 *w, const float4 *x, float *y, int rows, int in4, int tokens) {
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const float4 *r = w + (size_t)row * in4;
    float acc[NT] = {};
    for (int i = threadIdx.x; i < in4; i += 256) {
        const float4 a = r[i];
#pragma unroll
        for (int t = 0; t < NT; ++t)
            if (t < tokens) {
                const float4 b = x[(size_t)t * in4 + i];
                acc[t] = fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, fmaf(a.w, b.w, acc[t]))));
            }
    }
    __shared__ float sums[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
        for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
        if (!lane) sums[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < tokens) {
        float v = 0;
        for (int i = 0; i < 8; ++i) v += sums[threadIdx.x][i];
        y[(size_t)threadIdx.x * rows + row] = v;
    }
}
__global__ void dequant(const uint8_t *w, float *y, long long values) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= values) return;
    const uint8_t *b = w + (i / 32) * kBlock;
    y[i] = scale(b) * (float)(int8_t)b[2 + i % 32];
}
// Fused decode mHC read for one token (STRATA_GLM_HC_FUSED=1). The RMS norm without weights is a scalar per token,
// so the Q8 projection runs on the raw residual streams and the scale is applied to its 24 outputs: blocks
// (rows, 32 slices) as rows_partial, plus one extra row of blocks that sums the squares of its input slice.
__device__ float hc_sig(float x) { return 1.f / (1.f + expf(-x)); }
__global__ void hc_partial_sumsq(const uint8_t *w, const float *r, float *partial, int in, int rows) {
    const int row = blockIdx.x, slice = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int blocks = in / 32, per = blocks / 32, first = slice * per;
    float acc = 0.f;
    if (row < rows) {
        const uint8_t *q = w + (size_t)row * blocks * kBlock;
        for (int b = first + warp; b < first + per; b += 8)
            acc = fmaf(scale(q + b * kBlock) * (float)(int8_t)q[b * kBlock + 2 + lane], r[b * 32 + lane], acc);
    } else {
        for (int b = first + warp; b < first + per; b += 8) {
            const float v = r[b * 32 + lane];
            acc = fmaf(v, v, acc);
        }
    }
    __shared__ float sums[8];
    for (int delta = 16; delta; delta >>= 1) acc += __shfl_down_sync(0xffffffff, acc, delta);
    if (!lane) sums[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.f;
        for (int i = 0; i < 8; ++i) v += sums[i];
        partial[row * 32 + slice] = v;
    }
}
// Every block reduces the 24 projections and the sum of squares, applies the norm scale, evaluates the 4x4
// Sinkhorn coefficients (mhc_coeff_warp's arithmetic) and mixes its slice of the residual streams; block 0 also
// stores the coefficients for hc_write.
__global__ void hc_read_fused(const float *partial, const float *r, const float *base, const float *scl, float *c_out,
                              float *x, int n, int iters, float eps, float rms_eps) {
    __shared__ float p[25], coef[24];
    const int lane = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        for (int k = 0; k < 25; ++k) {
            float v = partial[k * 32 + lane];
            for (int delta = 16; delta; delta >>= 1) v += __shfl_down_sync(0xffffffff, v, delta);
            if (!lane) p[k] = v;
        }
        __syncwarp();
        const float inv = rsqrtf(p[24] / (4.f * n) + rms_eps);
        const int row = lane / 4, col = lane % 4;
        if (lane < 4) {
            coef[lane] = hc_sig(p[lane] * inv * scl[0] + base[lane]) + eps;
            coef[4 + lane] = 2 * hc_sig(p[4 + lane] * inv * scl[1] + base[4 + lane]);
        }
        const float projected = lane < 16 ? p[8 + lane] * inv * scl[2] + base[8 + lane] : 0.f;
        float mx = -INFINITY;
        for (int j = 0; j < 4; ++j) mx = fmaxf(mx, __shfl_sync(0xffffffff, projected, row * 4 + j));
        float value = expf(projected - mx), sum = 0;
        for (int j = 0; j < 4; ++j) sum += __shfl_sync(0xffffffff, value, row * 4 + j);
        value = value / sum + eps;
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
                sum = eps;
                for (int j = 0; j < 4; ++j) sum += __shfl_sync(0xffffffff, value, row * 4 + j);
                value /= sum;
            }
            sum = eps;
            for (int i = 0; i < 4; ++i) sum += __shfl_sync(0xffffffff, value, i * 4 + col);
            value /= sum;
        }
        if (lane < 16) coef[8 + lane] = value;
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x < 24) c_out[threadIdx.x] = coef[threadIdx.x];
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < n) {
        float s = 0;
        for (int i = 0; i < 4; ++i) s += coef[i] * r[i * n + d];
        x[d] = s;
    }
}
template <int NT>
void launch_heads(const void *w, const float *x, float *y, int rows, int in, int out, int ld_x, int ld_y,
                  int tokens, cudaStream_t s) {
    heads_gemv<NT><<<(rows + 7) / 8, 256, 0, s>>>((const uint8_t *)w, x, y, in, out, rows, ld_x, ld_y, tokens);
}
template <int NT>
void launch_rows(const void *w, const float *x, float *scratch, int rows, int in, int tokens, cudaStream_t s) {
    rows_partial<NT><<<dim3(rows, 32), 256, 0, s>>>((const uint8_t *)w, x, scratch, in, rows, tokens);
}
} // namespace

void glm_q8_heads_gemv(const void *w, const float *x, float *y, int heads, int in, int out, int tokens, int ld_x,
                       int ld_y, void *stream) {
    if (heads < 1 || out < 1 || in < 32 || in % 32 || tokens < 1 || tokens > 8)
        throw std::invalid_argument("GLM Q8_0 heads GEMV: invalid geometry");
    auto s = (cudaStream_t)stream;
    const int rows = heads * out;
    static const bool by_blocks = [] { const char *v = std::getenv("STRATA_GLM_HEADS_BLOCKS"); return v && std::string(v) == "1"; }();
    const int blocks = in / 32;
    if (by_blocks && tokens == 1 && blocks <= 32 && (blocks & (blocks - 1)) == 0) {
        const size_t threads = (size_t)rows * blocks;
        heads_gemv_blocks<<<(unsigned)((threads + 255) / 256), 256, 0, s>>>((const uint8_t *)w, x, y, in, out, rows);
        ok(cudaGetLastError(), "heads GEMV");
        return;
    }
    switch (tokens) {
    case 1: launch_heads<1>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    case 2: launch_heads<2>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    case 3: launch_heads<3>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    case 4: launch_heads<4>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    default: launch_heads<8>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    }
    ok(cudaGetLastError(), "heads GEMV");
}

void glm_hc_read_fused(const void *w, const float *r, const float *base, const float *scale_w, float *c, float *x,
                       float *scratch, int n, int iters, float eps, float rms_eps, void *stream) {
    const int in = 4 * n;
    if (n < 256 || in % 1024) throw std::invalid_argument("GLM fused mHC read: invalid geometry");
    auto s = (cudaStream_t)stream;
    hc_partial_sumsq<<<dim3(25, 32), 256, 0, s>>>((const uint8_t *)w, r, scratch, in, 24);
    ok(cudaGetLastError(), "fused mHC partial");
    hc_read_fused<<<(n + 255) / 256, 256, 0, s>>>(scratch, r, base, scale_w, c, x, n, iters, eps, rms_eps);
    ok(cudaGetLastError(), "fused mHC read");
}

void glm_q8_rows_gemv(const void *w, const float *x, float *y, float *scratch, int rows, int in, int tokens,
                      void *stream) {
    if (rows < 1 || in < 1024 || in % 1024 || tokens < 1 || tokens > 8)
        throw std::invalid_argument("GLM Q8_0 rows GEMV: invalid geometry");
    auto s = (cudaStream_t)stream;
    switch (tokens) {
    case 1: launch_rows<1>(w, x, scratch, rows, in, tokens, s); break;
    case 2: launch_rows<2>(w, x, scratch, rows, in, tokens, s); break;
    case 3: launch_rows<3>(w, x, scratch, rows, in, tokens, s); break;
    case 4: launch_rows<4>(w, x, scratch, rows, in, tokens, s); break;
    default: launch_rows<8>(w, x, scratch, rows, in, tokens, s); break;
    }
    ok(cudaGetLastError(), "rows GEMV");
    rows_reduce<<<rows * tokens, 32, 0, s>>>(scratch, y);
    ok(cudaGetLastError(), "rows reduce");
}

void glm_f32_rows_gemv(const float *w, const float *x, float *y, int rows, int in, int tokens, void *stream) {
    const auto aligned = [](const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    if (rows < 1 || in < 4 || in % 4 || tokens < 1 || tokens > 8 || !aligned(w) || !aligned(x))
        throw std::invalid_argument("GLM FP32 rows GEMV: invalid geometry or alignment");
    auto s = (cudaStream_t)stream;
    const unsigned blocks = rows;
    const auto *w4 = (const float4 *)w;
    const auto *x4 = (const float4 *)x;
    switch (tokens) {
    case 1: f32_rows<1><<<blocks, 256, 0, s>>>(w4, x4, y, rows, in / 4, tokens); break;
    case 2: f32_rows<2><<<blocks, 256, 0, s>>>(w4, x4, y, rows, in / 4, tokens); break;
    case 3: f32_rows<3><<<blocks, 256, 0, s>>>(w4, x4, y, rows, in / 4, tokens); break;
    case 4: f32_rows<4><<<blocks, 256, 0, s>>>(w4, x4, y, rows, in / 4, tokens); break;
    default: f32_rows<8><<<blocks, 256, 0, s>>>(w4, x4, y, rows, in / 4, tokens); break;
    }
    ok(cudaGetLastError(), "FP32 rows GEMV");
}

void glm_q8_dequant(const void *w, float *y, long long values, void *stream) {
    if (values < 0 || values % 32) throw std::invalid_argument("GLM Q8_0 dequant: invalid count");
    if (!values) return;
    dequant<<<(unsigned)((values + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint8_t *)w, y, values);
    ok(cudaGetLastError(), "dequant");
}

} // namespace strata::kernels
