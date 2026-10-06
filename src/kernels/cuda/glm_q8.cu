#include "strata/kernels/glm_q8.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
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
    switch (tokens) {
    case 1: launch_heads<1>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    case 2: launch_heads<2>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    case 3: launch_heads<3>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    case 4: launch_heads<4>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    default: launch_heads<8>(w, x, y, rows, in, out, ld_x, ld_y, tokens, s); break;
    }
    ok(cudaGetLastError(), "heads GEMV");
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
