// Device helpers shared by bench_gpu.cu and mma_native.cu.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

// 16-entry byte tables for the 4-bit E2M1 codes, as 4 little-endian u32 each.
// E2M1 x2 as int8: {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}
#define LUT_I8 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu
// fp16 high byte of the E2M1 value (low byte is always 0): 0,.5,1,1.5,2,3,4,6 and negatives
#define LUT_F16HI 0x3E3C3800u, 0x46444240u, 0xBEBCB880u, 0xC6C4C2C0u

// q4 holds 8 codes. Returns {bytes of codes 0,2,4,6 ; bytes of codes 1,3,5,7} looked up in the table
// (llama.cpp get_int_from_table_16, PRMT based). With the checkpoint's packing, codes 0,2,4,6 are the low
// nibbles = even elements, codes 1,3,5,7 the odd elements.
__device__ __forceinline__ int2 lut16(uint32_t q4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    uint32_t tmp[2];
    const uint32_t hsel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t0, t1, q4 >> sh);
        const uint32_t hi = __byte_perm(t2, t3, q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, hsel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ float e8m0f(uint32_t e) { return __int_as_float(e << 23); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ uint4 ld_stream(const void* p) {  // weights: read once, do not pollute L1
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}
__device__ __forceinline__ uint32_t ld_stream32(const void* p) {
    uint32_t r;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}

// ---- grouped GEMM description (weights = A rows, tokens = B columns) -----------------------------
// out[out_slot0 + i][row] = sum_k W_e[row][k] * act[act_tok0 + i][k] for i < ntok.
struct Tile { int expert, act_tok0, ntok, out_slot0; };

struct GemmArgs {
    const uint8_t* wq;   // expert pool, HF layout
    const uint8_t* we;
    size_t wq_es, we_es;  // bytes per expert
    int rows, K;
    const Tile* tiles;
    int ntiles;
    const void* act;     // format depends on kernel
    const void* act_s;   // scales, format depends on kernel
    const void* act2;    // second part (MSD)
    const void* act2_s;
    float* out;
};

// Implemented in mma_native.cu. kind: 0 I8, 1 F8FOLD, 2 F16FOLD, 3 MXF8 (kind::mxf8f6f4),
// 4 MXF4 (kind::mxf4), 5 MSD (two kind::mxf4 passes). container_shift: e2m1 position in its byte (kind 3).
void launch_gemm(int kind, int bn, int container_shift, const GemmArgs& a, cudaStream_t s);
void set_f8_tables(const uint32_t* tab64);  // [16][4] e4m3 tables for E2M1 * 2^-d
