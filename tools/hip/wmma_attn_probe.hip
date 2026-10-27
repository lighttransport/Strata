// WMMA issue-rate probe for the DS4.1 attention shapes on gfx12 (RDNA4), no engine dependency:
//   clang++ -x hip --offload-arch=gfx1201 -O3 -o wmma_attn_probe tools/hip/wmma_attn_probe.hip
//   LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib ./wmma_attn_probe 8192
// One block = one token, 16 heads (m), 64-row tiles (n), 512 dims (k); 8 waves own a 64-dim k slice (QK partials,
// LDS reduce) and a 64-dim output slice (PV). Measures the MMA issue rate only (the inputs stay in L1): 26-45 TF/s on
// an RX 9070 XT, against ~10 TFLOPS for the CUDA-core attn_tiled_kernel. See docs/DS41F_PLAN.md 8k.
// gfx12 WMMA probe at DS4.1 attention shapes: per block one token, 16 heads (m), 64-row tiles (n), 512 dims (k).
// QK: S[16][64] = Q[16][512] . K[64][512]^T ; PV: O[16][512] = P[16][64] . V[64][512].
// Waves 0..7 each own a 64-dim k-slice for QK (partial scores, LDS reduce) and a 64-dim n-slice for PV.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(e)); std::exit(1); } } while (0)

using f8 = float __attribute__((ext_vector_type(8)));
using h8 = _Float16 __attribute__((ext_vector_type(8)));
#define MMA(a, b, c) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)

// grid: blocks = tokens (the real kernel's grid.x = tokens, grid.y = head groups; one group here)
__global__ void __launch_bounds__(256) attn_wmma_probe(const _Float16* Q, const _Float16* K, const _Float16* VT,
                                                       const _Float16* P, float* O, int reps) {
    const int tid = threadIdx.x, wid = tid >> 5, lid = tid & 31, hf = lid >> 4, id = lid & 15;
    extern __shared__ float lds[];
    float* part = lds;              // [8][4][16][16] partial scores
    float* sc = part + 8 * 4 * 256; // [16][64] reduced scores
    for (int r = 0; r < reps; ++r) {
        // (the +r addresses keep the compiler from hoisting the loop-invariant WMMAs)
        const _Float16* Qr = Q + (r & 3);
        const _Float16* Kr = K + (r & 3);
        const _Float16* Vr = VT + (r & 3);
        const _Float16* Pr = P + (r & 3);
        // ---- QK: this wave's 64-dim k slice (4 k-steps), 4 n-tiles of 16 rows
        f8 s[4];
        for (int c = 0; c < 4; ++c) for (int i = 0; i < 8; ++i) s[c][i] = 0.f;
        for (int ks = 0; ks < 4; ++ks) {   // k steps of 16 within this wave's slice
            h8 q;
            for (int i = 0; i < 8; ++i) q[i] = Qr[id * 512 + wid * 64 + ks * 16 + hf * 8 + i];
            for (int c = 0; c < 4; ++c) {
                h8 k;
                for (int i = 0; i < 8; ++i) k[i] = Kr[(c * 16 + id) * 512 + wid * 64 + ks * 16 + hf * 8 + i];
                MMA(q, k, s[c]);
            }
        }
        for (int c = 0; c < 4; ++c)
            for (int i = 0; i < 8; ++i) part[((wid * 4 + c) * 16 + (hf * 8 + i)) * 16 + id] = s[c][i];
        __syncthreads();
        for (int e = tid; e < 16 * 64; e += 256) {   // [16 heads][64 rows] = sum over the 8 waves
            float v = 0;
            for (int w = 0; w < 8; ++w) v += part[((w * 4 + (e % 64) / 16) * 16 + (e / 64)) * 16 + (e % 16)];
            sc[e] = v;
        }
        __syncthreads();
        // ---- PV: this wave's 64-dim output slice: 4 n-tiles, 4 k-steps of 16 rows
        f8 o[4];
        for (int c = 0; c < 4; ++c) for (int i = 0; i < 8; ++i) o[c][i] = 0.f;
        for (int ks = 0; ks < 4; ++ks) {
            h8 p;
            for (int i = 0; i < 8; ++i) p[i] = Pr[id * 64 + ks * 16 + hf * 8 + i];   // A = P[head][row]
            for (int c = 0; c < 4; ++c) {
                h8 v;
                for (int i = 0; i < 8; ++i) v[i] = Vr[(wid * 64 + c * 16 + id) * 64 + ks * 16 + hf * 8 + i];
                MMA(p, v, o[c]);
            }
        }
        if (true)
            for (int c = 0; c < 4; ++c)
                for (int i = 0; i < 8; ++i) O[((hf * 8 + i) * 512) + wid * 64 + c * 16 + id] = o[c][i];
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int reps = 20;
    _Float16 *Q, *K, *VT, *P;
    float* O;
    CK(hipMalloc(&Q, 16 * 512 * 2));
    CK(hipMalloc(&K, 64 * 512 * 2));
    CK(hipMalloc(&VT, 512 * 64 * 2));
    CK(hipMalloc(&P, 16 * 64 * 2));
    CK(hipMalloc(&O, 16 * 512 * 4));
    CK(hipMemset(Q, 0, 16 * 512 * 2));
    CK(hipMemset(K, 0, 64 * 512 * 2));
    CK(hipMemset(VT, 0, 512 * 64 * 2));
    CK(hipMemset(P, 0, 16 * 64 * 2));
    const size_t shmem = (8 * 4 * 256 + 16 * 64) * sizeof(float);
    CK(hipFuncSetAttribute((const void*)attn_wmma_probe, hipFuncAttributeMaxDynamicSharedMemorySize, int(shmem)));
    attn_wmma_probe<<<blocks, 256, shmem>>>(Q, K, VT, P, O, 1);
    CK(hipDeviceSynchronize());
    for (int it = 0; it < 3; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        attn_wmma_probe<<<blocks, 256, shmem>>>(Q, K, VT, P, O, reps);
        CK(hipDeviceSynchronize());
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double flop = double(blocks) * reps * (2.0 * 16 * 64 * 512 * 2);   // QK + PV
        std::printf("blocks=%d reps=%d  %.3f s  %.1f TF/s\n", blocks, reps, el, flop / el / 1e12);
    }
    return 0;
}
