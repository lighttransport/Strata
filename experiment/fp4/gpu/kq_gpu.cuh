// Q2_K / Q3_K expert kernels for CUDA (dp4a, no tensor cores): the Strata q23 pack (Q2_K gate/up + Q3_K down)
// and ggml-org's Q2_K GGUF (Q2_K gate/up + MXFP4 down). Weights are native ggml blocks as stored in a
// GGUF / Strata pack (84-byte Q2_K, 110-byte Q3_K). Activations: int8 per 32 in natural order with a float
// scale per 32 and the int sum per 16 (for the Q2_K mins).
#pragma once
#include "common.cuh"

namespace kqg {

constexpr int Q2B = 84, Q3B = 110;

__device__ __forceinline__ uint32_t ld32a2(const uint8_t* p) {  // 2-byte aligned 32-bit load (Q3_K blocks)
    const uint16_t* h = (const uint16_t*)p;
    return uint32_t(__ldg(h)) | (uint32_t(__ldg(h + 1)) << 16);
}

// One lane computes half a 256-block (128 weights) of one row: n = which half.
__device__ __forceinline__ float q2k_half(const uint8_t* blk, int n, const int8_t* a, const float* ad, const int* as16) {
    uint32_t q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = __ldg((const uint32_t*)(blk + 16 + 32 * n) + i);
    const uint32_t s0 = __ldg((const uint32_t*)(blk + 8 * n)), s1 = __ldg((const uint32_t*)(blk + 8 * n + 4));
    const float2 dm = __half22float2(*(const __half2*)(blk + 80));
    float acc = 0.f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int e0 = 128 * n + 32 * g;  // element offset in the block
        const int4 x0 = *(const int4*)(a + e0), x1 = *(const int4*)(a + e0 + 16);  // a may be in shared memory
        const int sh = 2 * g;
        int d0 = __dp4a(int((q[0] >> sh) & 0x03030303), x0.x, 0);
        d0 = __dp4a(int((q[1] >> sh) & 0x03030303), x0.y, d0);
        d0 = __dp4a(int((q[2] >> sh) & 0x03030303), x0.z, d0);
        d0 = __dp4a(int((q[3] >> sh) & 0x03030303), x0.w, d0);
        int d1 = __dp4a(int((q[4] >> sh) & 0x03030303), x1.x, 0);
        d1 = __dp4a(int((q[5] >> sh) & 0x03030303), x1.y, d1);
        d1 = __dp4a(int((q[6] >> sh) & 0x03030303), x1.z, d1);
        d1 = __dp4a(int((q[7] >> sh) & 0x03030303), x1.w, d1);
        const uint32_t sw = (g < 2 ? s0 : s1) >> (16 * (g & 1));
        const int sc0 = sw & 0xff, sc1 = (sw >> 8) & 0xff;
        const float da = ad[e0 / 32];
        const int m = (sc0 >> 4) * as16[e0 / 16] + (sc1 >> 4) * as16[e0 / 16 + 1];
        acc += da * (dm.x * float((sc0 & 15) * d0 + (sc1 & 15) * d1) - dm.y * float(m));
    }
    return acc;
}

__device__ __forceinline__ float q3k_half(const uint8_t* blk, int n, const int8_t* a, const float* ad) {
    uint32_t q[8], hm[8], aux[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) { q[i] = ld32a2(blk + 32 + 32 * n + 4 * i); hm[i] = ld32a2(blk + 4 * i); }
    aux[0] = ld32a2(blk + 96); aux[1] = ld32a2(blk + 100); aux[2] = ld32a2(blk + 104);
    const float d = __half2float(*(const __half*)(blk + 108));
    {
        const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    }
    float acc = 0.f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int e0 = 128 * n + 32 * g;
        const int4 x0 = *(const int4*)(a + e0), x1 = *(const int4*)(a + e0 + 16);  // a may be in shared memory
        const int sh = 2 * g, hb = 4 * n + g;
        int w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = __vsub4(((q[i] >> sh) & 0x03030303) | (((hm[i] >> hb) & 0x01010101) << 2), 0x04040404);
        int d0 = __dp4a(w[0], x0.x, 0);
        d0 = __dp4a(w[1], x0.y, d0);
        d0 = __dp4a(w[2], x0.z, d0);
        d0 = __dp4a(w[3], x0.w, d0);
        int d1 = __dp4a(w[4], x1.x, 0);
        d1 = __dp4a(w[5], x1.y, d1);
        d1 = __dp4a(w[6], x1.z, d1);
        d1 = __dp4a(w[7], x1.w, d1);
        const int is = 8 * n + 2 * g;  // scale index (bytes of aux, each minus 32)
        const int sc0 = int((aux[is >> 2] >> (8 * (is & 3))) & 0xff) - 32;
        const int sc1 = int((aux[(is + 1) >> 2] >> (8 * ((is + 1) & 3))) & 0xff) - 32;
        acc += ad[e0 / 32] * d * float(sc0 * d0 + sc1 * d1);
    }
    return acc;
}

template <bool Q3>
__device__ __forceinline__ float half_dot(const uint8_t* row, int sub, const int8_t* a, const float* ad, const int* as16) {
    const int blk = sub >> 1, n = sub & 1;
    const int8_t* ab = a + blk * 256;
    const float* adb = ad + blk * 8;
    if constexpr (Q3) return q3k_half(row + blk * Q3B, n, ab, adb);
    else return q2k_half(row + blk * Q2B, n, ab, adb, as16 + blk * 16);
}

// GEMV over the 8 experts of a set: out[k][rows]. LPR = 2 * K/256 lanes per row (32 for K=4096, 16 for 2048);
// each lane group handles R rows, all dots issued before any reduction so their loads overlap.
template <bool Q3, int R>
__global__ void __launch_bounds__(256) k_kq_gemv(const uint8_t* __restrict__ w, size_t w_es, const int* __restrict__ ids,
                                                 int rows, int K, const int8_t* __restrict__ aq, const float* __restrict__ ad,
                                                 const int* __restrict__ as16, int a_per_expert, float* __restrict__ out) {
    const int k = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int lpr = 2 * K / 256, rpw = 32 / lpr;
    const size_t rb = size_t(K / 256) * (Q3 ? Q3B : Q2B);
    const int row0 = ((blockIdx.x * 8 + warp) * rpw + lane / lpr) * R;
    const int sub = lane % lpr;
    const int a = a_per_expert ? k : 0;
    const uint8_t* base = w + size_t(ids[k]) * w_es + size_t(row0) * rb;
    float v[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        v[r] = half_dot<Q3>(base + size_t(r) * rb, sub, aq + size_t(a) * K, ad + size_t(a) * (K / 32), as16 + size_t(a) * (K / 16));
#pragma unroll
    for (int r = 0; r < R; ++r) {
        for (int o = lpr / 2; o > 0; o >>= 1) v[r] += __shfl_xor_sync(0xffffffffu, v[r], o);
        if (sub == 0) out[size_t(k) * rows + row0 + r] = v[r];
    }
}

__device__ __forceinline__ int logical_pos(int phys) { return (phys & 1) ? 16 + (phys >> 1) : (phys >> 1); }

// Fused MoE part 1 for Q2_K gate/up: x -> q8 (smem) -> 32 gate + 32 up rows -> SwiGLU -> q8(h).
// H_LOGICAL: write h in the MXFP4 kernels' logical order (ggml-org mix, MXFP4 down), else natural + sums (Q3_K down).
template <bool H_LOGICAL>
__global__ void __launch_bounds__(256) k_moe_gu_q2k(const uint8_t* __restrict__ w, size_t w_es, const int* __restrict__ ids,
                                                    const float* __restrict__ x, int8_t* __restrict__ hq, float* __restrict__ hd,
                                                    int* __restrict__ hs16) {
    constexpr int H = 4096, I = 2048;
    __shared__ __align__(16) int8_t sx[H];
    __shared__ float sd[H / 32];
    __shared__ int ss[H / 16];
    __shared__ float sh[32];
    const int k = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < H / 32; b += 8) {
        const float v = x[b * 32 + lane];
        float m = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const int q = __float2int_rn(v * (m > 0 ? 127.f / m : 0.f));
        sx[b * 32 + lane] = (int8_t)q;
        int s = q;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);  // sums within each 16-lane half
        if ((lane & 15) == 0) ss[b * 2 + (lane >> 4)] = s;
        if (lane == 0) sd[b] = m / 127.f;
    }
    __syncthreads();
    const uint8_t* base = w + size_t(ids[k]) * w_es;
    constexpr size_t rb = size_t(H / 256) * Q2B;
    const int r0 = blockIdx.x * 32 + warp * 4;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[r] = half_dot<false>(base + size_t((r < 4 ? 0 : I) + r0 + (r & 3)) * rb, lane, sx, sd, ss);
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    if (lane < 4) sh[warp * 4 + lane] = acc[lane] / (1.f + __expf(-acc[lane])) * acc[4 + lane];
    __syncthreads();
    if (warp == 0) {
        const float v = sh[lane];
        float m = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const int q = __float2int_rn(v * (m > 0 ? 127.f / m : 0.f));
        hq[size_t(k) * I + blockIdx.x * 32 + (H_LOGICAL ? logical_pos(lane) : lane)] = (int8_t)q;
        if (lane == 0) hd[k * (I / 32) + blockIdx.x] = m / 127.f;
        if constexpr (!H_LOGICAL) {
            int s = q;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if ((lane & 15) == 0) hs16[k * (I / 16) + blockIdx.x * 2 + (lane >> 4)] = s;
        }
    }
}

// Fused MoE part 2 for Q3_K down: each half-warp owns one output row (16 lanes = 8 blocks x 2 halves of K=2048),
// sums the 8 experts weighted by the router.
__global__ void __launch_bounds__(256) k_moe_down_q3k(const uint8_t* __restrict__ w, size_t w_es, const int* __restrict__ ids,
                                                      const float* __restrict__ route, const int8_t* __restrict__ hq,
                                                      const float* __restrict__ hd, float* __restrict__ y) {
    constexpr int H = 4096, I = 2048;
    constexpr size_t rb = size_t(I / 256) * Q3B;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = ((blockIdx.x * 8 + warp) * 2 + (lane >> 4)) * 2, sub = lane & 15;
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll 4
    for (int k = 0; k < 8; ++k) {
        const uint8_t* rw = w + size_t(ids[k]) * w_es + size_t(row) * rb;
        const float rk = __ldg(route + k);
        acc0 += rk * half_dot<true>(rw, sub, hq + size_t(k) * I, hd + k * (I / 32), nullptr);
        acc1 += rk * half_dot<true>(rw + rb, sub, hq + size_t(k) * I, hd + k * (I / 32), nullptr);
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) { acc0 += __shfl_xor_sync(0xffffffffu, acc0, o); acc1 += __shfl_xor_sync(0xffffffffu, acc1, o); }
    if (sub == 0) { y[row] = acc0; y[row + 1] = acc1; }
    (void)H;
}

}  // namespace kqg
