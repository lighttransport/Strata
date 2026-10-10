/*
MIT License

Copyright (c) 2025 Light Transport Entertainment, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Experimental RDNA4 BF16 expert GEMM. Adapted from the 128x128/32-K experiment in
// syoyo/gemm rdna4/vlm/bench_vlm_gemm.c (local reference: ~/work/gemm/main).
// X[B,T,K] and W[B,N,K] are row-major BF16; output is row-major FP32.
#include "wmma_gemm.h"
#include "strata/kernels/gfx_arch.hpp"
#include <hip/hip_runtime.h>
#include <cstring>
#include <atomic>
#include <stdexcept>

namespace {
#if defined(STRATA_WMMA_GFX12)
using B8 = unsigned short __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ B8 load8(const uint16_t* x, int row, int rows, int K, int kp) {
    B8 v{};
    if (row >= rows) return v;
    if (K % 8 == 0 && kp + 8 <= K) return *(const B8*)(x + (size_t)row*K + kp);
#pragma unroll
    for (int i=0;i<8;++i) if(kp+i<K) v[i]=x[(size_t)row*K+kp+i];
    return v;
}
__global__ void bf16_experts(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                              float* __restrict__ y, int T, int N, int K) {
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int wm = wave % 2, wn = wave / 2, half = lane / 16, idx = lane % 16;
    const int m0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
    x += (size_t)blockIdx.z * T * K;
    w += (size_t)blockIdx.z * N * K;
    y += (size_t)blockIdx.z * T * N;
    // Vector-packed LDS from the reference's tuned variants: contiguous 8-value
    // fragments per row, transposed at vector granularity to avoid strided LDS reads.
    __shared__ B8 sx[4 * 128], sw[4 * 128];
    F8 acc[4][2] = {};
    const int row = tid / 2, col = (tid % 2) * 16;
    B8 pending_x[2], pending_w[2];
#pragma unroll
    for(int v=0;v<2;++v){
        pending_x[v]=load8(x,m0+row,T,K,col+v*8);
        pending_w[v]=load8(w,n0+row,N,K,col+v*8);
    }
    for (int k = 0; k < K; k += 32) {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            sx[(col / 8 + v) * 128 + row] = pending_x[v];
            sw[(col / 8 + v) * 128 + row] = pending_w[v];
        }
        __syncthreads();
        // One K panel prefetched into registers while the current LDS panel is consumed.
        if(k+32<K){
#pragma unroll
            for(int v=0;v<2;++v){
                pending_x[v]=load8(x,m0+row,T,K,k+32+col+v*8);
                pending_w[v]=load8(w,n0+row,N,K,k+32+col+v*8);
            }
        }
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            B8 a[4], b[2];
#pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = sx[(kk / 8 + half) * 128 + wm * 64 + r * 16 + idx];
#pragma unroll
            for (int c = 0; c < 2; ++c) b[c] = sw[(kk / 8 + half) * 128 + wn * 32 + c * 16 + idx];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 2; ++c)
                    {
#if defined(__gfx1200__) || defined(__gfx1201__)
                        acc[r][c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a[r], b[c], acc[r][c]);
#else
                        __builtin_trap();
#endif
                    }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int col = n0 + wn * 32 + c * 16 + idx;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = m0 + wm * 64 + r * 16 + half * 8 + i;
                if (row < T && col < N) y[(size_t)row * N + col] = acc[r][c][i];
            }
        }
}
#endif
}

bool strata_wmma_gfx12_bf16_batched(const uint16_t* X, const uint16_t* W, float* Y,
                                   int T, int N, int K, int batches, void* stream) {
    if (T < 1 || N < 1 || K < 1 || T > 128 * 65535 || N > 128 * 65535 ||
        K > 2147483616 || batches < 1 || batches > 65535) return false;
    int device;
    static std::atomic<int> support[64];
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 64) return false;
    int available = support[device].load(std::memory_order_acquire);
    if (!available) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, device) != hipSuccess) return false;
        available = (strata::kernels::gfx_arch_is(p.gcnArchName, "gfx1200") ||
                     strata::kernels::gfx_arch_is(p.gcnArchName, "gfx1201")) ? 1 : -1;
        support[device].store(available, std::memory_order_release);
    }
    if (available < 0) return false;
#if defined(STRATA_WMMA_GFX12)
    bf16_experts<<<dim3((N + 127) / 128, (T + 127) / 128, batches), 256, 0, (hipStream_t)stream>>>(X, W, Y, T, N, K);
#else
    return false;
#endif
    const auto error = hipGetLastError();
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
    return true;
}
