// src/kernels/cuda/hy4f.cu - Hy4-Flash decode kernels (see include/strata/kernels/hy4f.hpp).
// Reductions go through shared memory with a fixed tree, so they do not depend on the wave width (wave32 on RDNA,
// warp 32 on NVIDIA).
#include "strata/kernels/hy4f.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("hy4f ") + what + ": " + cudaGetErrorString(e));
}

constexpr int kThreads = 256;

// sum of v over the block (kThreads), every thread gets the result
__device__ float block_sum(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void hc_coeff_kernel(const float* m, const float* scale, const float* base, float* out, int hc, int with_post,
                                float magnitude, float eps) {
    const int i = threadIdx.x;
    if (i < hc) out[i] = sigmoidf(m[i] * scale[0] + base[i]) + eps;
    else if (with_post && i < 2 * hc) out[i] = sigmoidf(m[i] * scale[1] + base[i]) * magnitude + eps;
}

__global__ void hc_post_kernel(const float* x, const float* post, float* h, int n, int hc) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const float v = x[j];
    for (int s = 0; s < hc; ++s) h[size_t(s) * n + j] = h[size_t(s) * n + j] + v * post[s];
}

__global__ void hc_pre_kernel(const float* h, const float* w, float* y, int n, int hc) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = h[j] * w[0];
    for (int s = 1; s < hc; ++s) acc += h[size_t(s) * n + j] * w[s];
    y[j] = acc;
}

// one block per (row group, head): 8 rows per block, 32 threads per row
__global__ void q8_0_heads_kernel(const uint8_t* w, const float* x, int x_stride, float* y, int y_stride, int rows, int in) {
    const int head = blockIdx.y;
    const int lane = threadIdx.x & 31, sub = threadIdx.x >> 5;
    const int r = blockIdx.x * 8 + sub;
    if (r >= rows) return;
    const int nb = in / 32;
    const uint8_t* row = w + (size_t(head) * rows + r) * size_t(nb) * 34;
    const float* xv = x + size_t(head) * x_stride;
    float acc = 0;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + size_t(b) * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        acc += d * float(int8_t(blk[2 + lane])) * xv[b * 32 + lane];
    }
    __shared__ float sh[8][32];
    sh[sub][lane] = acc;
    __syncthreads();
    if (lane == 0) {
        float s = 0;
        for (int i = 0; i < 32; ++i) s += sh[sub][i];
        y[size_t(head) * y_stride + r] = s;
    }
}

__global__ void mla_pack_q_kernel(const float* qa, const float* qpe, int qpe_stride, float* q) {
    const int h = blockIdx.x;
    for (int j = threadIdx.x; j < 640; j += blockDim.x)
        q[size_t(h) * 640 + j] = j < 512 ? qa[size_t(h) * 512 + j] : j < 576 ? qpe[size_t(h) * qpe_stride + (j - 512)] : 0.0f;
}

__global__ void mla_pack_k_kernel(const float* kv, const float* kpe, float* k) {
    for (int j = threadIdx.x; j < 640; j += blockDim.x) k[j] = j < 512 ? kv[j] : j < 576 ? kpe[j - 512] : 0.0f;
}

__global__ void mul_sigmoid_kernel(const float* x, const float* g, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * sigmoidf(g[i]);
}

// one block per stream s
__global__ void engram_kernel(float* h, const float* kv, const float* g, const float* gamma, const float* hc_scale, int n, int hc,
                              float eps, int any) {
    __shared__ float sh[kThreads];
    const int s = blockIdx.x;
    float* hs = h + size_t(s) * n;
    const float* key = kv + size_t(s) * n;
    const float* val = kv + size_t(hc) * n;
    float sx = 0, sk = 0, sv = 0;
    for (int j = threadIdx.x; j < n; j += kThreads) { sx += hs[j] * hs[j]; sk += key[j] * key[j]; sv += val[j] * val[j]; }
    const float rx = 1.0f / sqrtf(block_sum(sx, sh) / n + eps);
    const float rk = 1.0f / sqrtf(block_sum(sk, sh) / n + eps);
    const float rv = 1.0f / sqrtf(block_sum(sv, sh) / n + eps);
    float dot = 0;
    for (int j = threadIdx.x; j < n; j += kThreads) dot += (hs[j] * rx) * (key[j] * rk) * g[size_t(s) * n + j];
    const float d = block_sum(dot, sh) / sqrtf(float(n));
    const float mag = sqrtf(fmaxf(fabsf(d), 1e-6f));
    const float gate = sigmoidf(d < 0 ? -mag : mag);
    if (!any) return;
    for (int j = threadIdx.x; j < n; j += kThreads) hs[j] += (val[j] * rv * gamma[j]) * (gate + hc_scale[size_t(s) * n + j]);
}

__global__ void kpool_kernel(const float* vg, const float* ape, float* out, int kpool, int dim) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    float mx = -INFINITY;
    for (int m = 0; m < kpool; ++m) mx = fmaxf(mx, vg[size_t(m) * 2 * dim + dim + c] + ape[size_t(m) * dim + c]);
    float den = 0, acc = 0;
    for (int m = 0; m < kpool; ++m) {
        const float e = expf(vg[size_t(m) * 2 * dim + dim + c] + ape[size_t(m) * dim + c] - mx);
        den += e;
        acc += e * vg[size_t(m) * 2 * dim + c];
    }
    out[c] = acc / den;
}

}  // namespace

void hy4_hc_coeff(const float* mixes, const float* scale, const float* base, float* out, int hc, bool with_post, float magnitude,
                  float eps, void* stream) {
    hc_coeff_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(mixes, scale, base, out, hc, with_post, magnitude, eps);
    check("hc_coeff");
}
void hy4_hc_post(const float* x, const float* post, float* h, int n, int hc, void* stream) {
    hc_post_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, post, h, n, hc);
    check("hc_post");
}
void hy4_hc_pre(const float* h, const float* w, float* y, int n, int hc, void* stream) {
    hc_pre_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(h, w, y, n, hc);
    check("hc_pre");
}
void hy4_q8_0_heads(const void* w, const float* x, int x_stride, float* y, int y_stride, int rows, int in, int heads, void* stream) {
    if (in % 32) throw std::invalid_argument("hy4_q8_0_heads: in % 32");
    const dim3 g(unsigned((rows + 7) / 8), unsigned(heads));
    q8_0_heads_kernel<<<g, 256, 0, (cudaStream_t) stream>>>((const uint8_t*) w, x, x_stride, y, y_stride, rows, in);
    check("q8_0_heads");
}
void hy4_mla_pack_q(const float* qa, const float* qpe, int qpe_stride, float* q, int heads, void* stream) {
    mla_pack_q_kernel<<<heads, 128, 0, (cudaStream_t) stream>>>(qa, qpe, qpe_stride, q);
    check("mla_pack_q");
}
void hy4_mla_pack_k(const float* kv, const float* kpe, float* k, void* stream) {
    mla_pack_k_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(kv, kpe, k);
    check("mla_pack_k");
}
void hy4_mul_sigmoid(const float* x, const float* g, float* y, int n, void* stream) {
    mul_sigmoid_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, g, y, n);
    check("mul_sigmoid");
}
void hy4_engram(float* h, const float* kv, const float* g, const float* gamma, const float* hc_scale, int n, int hc, float eps,
                bool any, void* stream) {
    engram_kernel<<<hc, kThreads, 0, (cudaStream_t) stream>>>(h, kv, g, gamma, hc_scale, n, hc, eps, any);
    check("engram");
}

void hy4_kpool(const float* vg, const float* ape, float* out, int kpool, int dim, void* stream) {
    kpool_kernel<<<(dim + 127) / 128, 128, 0, (cudaStream_t) stream>>>(vg, ape, out, kpool, dim);
    check("kpool");
}

}  // namespace strata::kernels
