// src/kernels/cuda/hy4f.cu - Hy4-Flash decode kernels (see include/strata/kernels/hy4f.hpp).
// Reductions go through shared memory with a fixed tree, so they do not depend on the wave width (wave32 on RDNA,
// warp 32 on NVIDIA).
#include "strata/kernels/hy4f.hpp"
#include "strata/kernels/q8_1_finite.hpp"

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


// ---- fused kernels
struct Q81 { half2 ds; int8_t qs[32]; };   // ggml block_q8_1 (36 bytes)

// value i of a q8_1 row set: the 32 lanes of a warp hold block i / 32 (iq_kernels.cu q8_1_store, the same bits)
__device__ __forceinline__ void q81_store(const float xi, Q81* y, const long long i) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);
    y[i / 32].qs[i % 32] = q8_1_quant(xi, d, amax);
    if (i % 32 == 0) y[i / 32].ds = q8_1_ds(d, sum);
}

__device__ __forceinline__ float ue4m3_half(uint8_t x) {   // ggml_cuda_ue4m3_to_fp32 (iq_kernels.cu ue4m3_to_f32_half)
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t e = (x >> 3) & 0xF, m = x & 0x7;
    return e == 0 ? (float) m * 0.0009765625f : __uint_as_float(((e + 119u) << 23) | (m << 20));
}
__constant__ float kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

constexpr int kBig = 256;   // one workgroup; 1024 waits for a free CU on a shared GPU (4x slower wall time measured)

__device__ float big_sum(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int s = kBig / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__global__ void __launch_bounds__(kBig) hc_mix_kernel(const float* h, const uint8_t* fn, int rows, const float* scale, const float* base,
                                                       float* coeff, float mag, float hc_eps, float rms_eps, const float* nw, float* x,
                                                       float* y, Q81* yq, int n, int hc) {
    __shared__ float sh[kBig];
    __shared__ float cf[16];
    const int t = threadIdx.x, N = n * hc;
    float ss = 0;
    for (int i = t; i < N; i += kBig) ss += h[i] * h[i];
    const float r = 1.0f / sqrtf(big_sum(ss, sh) / N + rms_eps);
    const int nb = N / 64;
    const size_t row_bytes = size_t(nb) * 36;
    for (int j = 0; j < rows; ++j) {
        float acc = 0;
        for (int b = t; b < nb; b += kBig) {
            const uint8_t* blk = fn + size_t(j) * row_bytes + size_t(b) * 36;
            const float* hv = h + size_t(b) * 64;
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const float d = ue4m3_half(blk[sb]);
                float part = 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint8_t q = blk[4 + sb * 8 + k];
                    part += kFp4[q & 0xF] * hv[sb * 16 + k] + kFp4[q >> 4] * hv[sb * 16 + 8 + k];
                }
                acc += d * part;
            }
        }
        const float m = big_sum(acc, sh) * r;
        if (t == 0) {
            if (j < hc) cf[j] = 1.0f / (1.0f + expf(-(m * scale[0] + base[j]))) + hc_eps;
            else cf[j] = 1.0f / (1.0f + expf(-(m * scale[1] + base[j]))) * mag + hc_eps;
        }
    }
    __syncthreads();
    if (t < rows) coeff[t] = cf[t];
    float s2 = 0;
    for (int i = t; i < n; i += kBig) {
        float v = h[i] * cf[0];
        for (int s = 1; s < hc; ++s) v += h[size_t(s) * n + i] * cf[s];
        x[i] = v;
        s2 += v * v;
    }
    const float r2 = 1.0f / sqrtf(big_sum(s2, sh) / n + rms_eps);
    for (int i = t; i < n; i += kBig) {   // n % 32 == 0 and kBig % 32 == 0: whole warps per q8_1 block
        const float v = x[i] * r2 * nw[i];
        y[i] = v;
        q81_store(v, yq, i);
    }
}

__global__ void __launch_bounds__(kBig) norm_hc_post_kernel(const float* x, const float* add, const float* w, const float* post, float* h,
                                                             int n, int hc, float eps) {
    __shared__ float sh[kBig];
    const int t = threadIdx.x;
    float ss = 0;
    for (int i = t; i < n; i += kBig) {
        const float v = add ? x[i] + add[i] : x[i];
        ss += v * v;
    }
    const float r = 1.0f / sqrtf(big_sum(ss, sh) / n + eps);
    for (int i = t; i < n; i += kBig) {
        const float v = (add ? x[i] + add[i] : x[i]) * r * w[i];
        for (int s = 0; s < hc; ++s) h[size_t(s) * n + i] = h[size_t(s) * n + i] + v * post[s];
    }
}

__device__ __forceinline__ void rot(float x0, float x1, float c, float s, float* o) {
    o[0] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    o[1] = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
}

// blocks 0..heads-1: a head's rope tail; block heads: the cache row (128 threads)
__global__ void mla_finish_kernel(const float* q, int heads, int q_stride, int nope, const float* kvpe, const float* kv_w, const float* cs,
                                  float* q640, float* krow, float eps) {
    const int t = threadIdx.x;
    if (blockIdx.x < heads) {
        float* out = q640 + size_t(blockIdx.x) * 640;
        const float* pe = q + size_t(blockIdx.x) * q_stride + nope;
        if (t < 32) rot(pe[2 * t], pe[2 * t + 1], cs[2 * t], cs[2 * t + 1], out + 512 + 2 * t);
        else if (t < 96) out[576 + (t - 32)] = 0.0f;
        return;
    }
    __shared__ float sh[128];
    float ss = 0;
    for (int i = t; i < 512; i += 128) ss += kvpe[i] * kvpe[i];
    sh[t] = ss;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float r = 1.0f / sqrtf(sh[0] / 512 + eps);
    for (int i = t; i < 512; i += 128) krow[i] = kvpe[i] * r * kv_w[i];
    if (t < 32) rot(kvpe[512 + 2 * t], kvpe[512 + 2 * t + 1], cs[2 * t], cs[2 * t + 1], krow + 512 + 2 * t);
    else if (t < 96) krow[576 + (t - 32)] = 0.0f;
}

__global__ void mul_sigmoid_q8_kernel(const float* x, const float* g, Q81* y, int n) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
#endif
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q81_store(x[i] * (1.0f / (1.0f + expf(-g[i]))), y, i);
}


__global__ void route_kernel(const float* logits, const float* bias, const unsigned long long* slots, int E, int K, float scale, int norm,
                             uint8_t* table, int* host) {
    __shared__ float sel[1024], prob[1024];
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    __shared__ int ids[16];
    const int t = threadIdx.x;
    for (int e = t; e < 1024; e += blockDim.x) {
        const float p = e < E ? 1.0f / (1.0f + expf(-logits[e])) : 0.0f;
        prob[e] = p;
        sel[e] = e < E ? p + bias[e] : -INFINITY;
    }
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        // argmax over sel (ties: the lowest index), a fixed tree
        for (int e = t; e < 1024; e += blockDim.x) { bv[e] = sel[e]; bi[e] = e; }
        __syncthreads();
        for (int s = 512; s > 0; s >>= 1) {
            for (int e = t; e < s; e += blockDim.x) {
                const float a = bv[e], b = bv[e + s];
                if (b > a || (b == a && bi[e + s] < bi[e])) { bv[e] = b; bi[e] = bi[e + s]; }
            }
            __syncthreads();
        }
        if (t == 0) { ids[k] = bi[0]; sel[bi[0]] = -INFINITY; }
        __syncthreads();
    }
    if (t != 0) return;
    float w[16], sum = 0;
    for (int k = 0; k < K; ++k) { w[k] = prob[ids[k]]; sum += w[k]; }
    if (norm) {
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int k = 0; k < K; ++k) w[k] /= sum;
    }
    auto* ptr = reinterpret_cast<unsigned long long*>(table);
    auto* start = reinterpret_cast<int32_t*>(table + size_t(K) * 8);
    int32_t* ng = start + K + 1;
    int32_t* dst = ng + 1;
    int32_t* tok = dst + K;
    float* tw = reinterpret_cast<float*>(tok + K);
    int g = 0;
    for (int k = 0; k < K; ++k) {
        w[k] *= scale;
        const unsigned long long a = slots ? slots[ids[k]] : 0ull;
        host[k] = a ? -1 : ids[k];
        host[K + k] = __float_as_int(w[k]);
        host[2 * K + k] = ids[k];
        if (a) { ptr[g] = a; start[g] = g; dst[g] = g; tok[g] = 0; tw[g] = w[k]; ++g; }
    }
    start[g] = g;
    *ng = g;
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

void hy4_hc_mix(const float* h, const void* fn, int rows, const float* scale, const float* base, float* coeff, float magnitude,
                float hc_eps, float rms_eps, const float* norm_w, float* x, float* y, void* y_q8, int n, int hc, void* stream) {
    if (n % 32 || (n * hc) % 64 || rows > 16) throw std::invalid_argument("hy4_hc_mix: shape");
    hc_mix_kernel<<<1, kBig, 0, (cudaStream_t) stream>>>(h, (const uint8_t*) fn, rows, scale, base, coeff, magnitude, hc_eps, rms_eps,
                                                          norm_w, x, y, (Q81*) y_q8, n, hc);
    check("hc_mix");
}
void hy4_norm_hc_post(const float* x, const float* add, const float* w, const float* post, float* h, int n, int hc, float eps,
                      void* stream) {
    norm_hc_post_kernel<<<1, kBig, 0, (cudaStream_t) stream>>>(x, add, w, post, h, n, hc, eps);
    check("norm_hc_post");
}
void hy4_mla_finish(const float* q, int heads, int q_stride, int nope, const float* kvpe, const float* kv_w, const float* cs, float* q640,
                    float* krow, float eps, void* stream) {
    mla_finish_kernel<<<heads + 1, 128, 0, (cudaStream_t) stream>>>(q, heads, q_stride, nope, kvpe, kv_w, cs, q640, krow, eps);
    check("mla_finish");
}
void hy4_mul_sigmoid_q8(const float* x, const float* g, void* q8, int n, void* stream) {
    if (n % 32) throw std::invalid_argument("hy4_mul_sigmoid_q8: n % 32");
    mul_sigmoid_q8_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, g, (Q81*) q8, n);
    check("mul_sigmoid_q8");
}

size_t hy4_route_table_bytes(int K) { return size_t(K) * 8 + size_t(K + 1) * 4 + 4 + size_t(K) * 12; }
void hy4_route(const float* logits, const float* bias, const unsigned long long* slots, int E, int K, float scale, bool norm,
               void* table, int* host, void* stream) {
    if (E > 1024 || K > 16 || K < 1) throw std::invalid_argument("hy4_route: shape");
    route_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(logits, bias, slots, E, K, scale, norm, (uint8_t*) table, host);
    check("route");
}

}  // namespace strata::kernels
