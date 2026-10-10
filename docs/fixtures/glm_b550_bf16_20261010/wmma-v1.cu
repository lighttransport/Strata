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
#include <hip/hip_runtime.h>
#include <cstring>
#include <atomic>
#include <stdexcept>

namespace {
#if defined(STRATA_WMMA_GFX12)
using B8 = unsigned short __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));
__global__ void bf16_experts(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                              float* __restrict__ y, int T, int N, int K) {
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int wm = wave % 2, wn = wave / 2, half = lane / 16, idx = lane % 16;
    const int m0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
    x += (size_t)blockIdx.z * T * K;
    w += (size_t)blockIdx.z * N * K;
    y += (size_t)blockIdx.z * T * N;
    __shared__ uint16_t sx[128 * 32], sw[128 * 32];
    F8 acc[4][2] = {};
    for (int k = 0; k < K; k += 32) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int e = tid * 16 + i, row = e / 32, col = e % 32;
            sx[e] = m0 + row < T && k + col < K ? x[(size_t)(m0 + row) * K + k + col] : 0;
            sw[e] = n0 + row < N && k + col < K ? w[(size_t)(n0 + row) * K + k + col] : 0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            B8 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int kp = kk + half * 8 + i;
#pragma unroll
                for (int r = 0; r < 4; ++r) a[r][i] = sx[(wm * 64 + r * 16 + idx) * 32 + kp];
#pragma unroll
                for (int c = 0; c < 2; ++c) b[c][i] = sw[(wn * 32 + c * 16 + idx) * 32 + kp];
            }
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
    if (T < 1 || N < 1 || K < 1 || batches < 1 || batches > 65535) return false;
    int device;
    static std::atomic<int> support[64];
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 64) return false;
    int available = support[device].load(std::memory_order_acquire);
    if (!available) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, device) != hipSuccess) return false;
        available = (!std::strncmp(p.gcnArchName, "gfx1200", 7) || !std::strncmp(p.gcnArchName, "gfx1201", 7)) ? 1 : -1;
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
