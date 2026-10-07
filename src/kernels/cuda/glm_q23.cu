// Canonical Q2_K/Q3_K routed experts on the GPU (see canon_expert.hpp). A row is eight threads, one per float
// lane of the CPU kernel (q23_avx2.cpp): each thread forms its lane's exact integer sums per 256-value superblock
// with __dp4a and folds them into its accumulator in superblock order, then the eight lanes reduce in the CPU
// kernel's order. Not built with --use_fast_math; every float operation is an explicit rounded one.
#include "strata/kernels/glm_q23.hpp"
#include "strata/kernels/canon_expert.hpp"
#include "strata/kernels/quantize_act.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kQ8kWords = 73;   // block_q8_K: float d, int8 qs[256], int16 bsums[16] = 292 bytes
constexpr int kMaxTok = 4;      // tokens of one group handled per pass
constexpr int kMaxBlocks = 16;  // 4096 columns

size_t round_up(size_t n) { return (n + 255) & ~size_t(255); }

__device__ __forceinline__ float half_bits(uint32_t h) {
    __half_raw raw;
    raw.x = (unsigned short)h;
    return __half2float(__half(raw));
}
// Q3_K blocks are 110 bytes, so odd blocks of a row sit at 2 mod 4: read words as two aligned halves.
__device__ __forceinline__ uint32_t load_halves(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t)h[0] | ((uint32_t)h[1] << 16);
}

// A lane's words of one superblock, fetched before any of them is used: a thread issues the loads of two
// superblocks (and of the gate and up rows) back to back, so their memory latencies overlap.
struct Raw { uint32_t q0, q1, hm, s0, s1, s2, s3, dw; };
template <int TY>
__device__ __forceinline__ Raw fetch(const uint8_t* blk, int j) {
    Raw r;
    if (TY == 10) {   // block_q2_K: scales[16] (scale low nibble, min high), qs[64], half d, half dmin
        const uint32_t* w = reinterpret_cast<const uint32_t*>(blk);
        r.s0 = w[0]; r.s1 = w[1]; r.s2 = w[2]; r.s3 = w[3];
        r.q0 = w[4 + j]; r.q1 = w[12 + j];
        r.dw = w[20];
        r.hm = 0;
    } else {          // block_q3_K: hmask[32], qs[64], scales[12] (6 bits each), half d
        r.hm = load_halves(blk + 4 * j);
        r.q0 = load_halves(blk + 32 + 4 * j);
        r.q1 = load_halves(blk + 64 + 4 * j);
        r.s0 = load_halves(blk + 96); r.s1 = load_halves(blk + 100); r.s2 = load_halves(blk + 104);
        r.s3 = 0;
        r.dw = reinterpret_cast<const uint16_t*>(blk + 108)[0];
    }
    return r;
}
// One superblock of one weight row for lane j and NT tokens; y[t] is token t's Q8_K block as words.
template <int TY, int NT>
__device__ __forceinline__ void superblock(const Raw& r, int j, const int* const* y, float* acc) {
    const int sel = j >> 2;
    const uint32_t q0 = r.q0, q1 = r.q1, hm = r.hm;
    uint64_t lo, hi;
    float d, dm = 0;
    int m0 = 0, m1 = 0;
    if (TY == 10) {
        lo = (uint64_t)r.s0 | ((uint64_t)r.s1 << 32);
        hi = (uint64_t)r.s2 | ((uint64_t)r.s3 << 32);
        d = half_bits(r.dw & 0xffff);
        dm = half_bits(r.dw >> 16);
        const uint64_t mins = j < 4 ? lo : hi;
        m0 = (int)((mins >> (16 * (j & 3) + 4)) & 15);
        m1 = (int)((mins >> (16 * (j & 3) + 12)) & 15);
    } else {
        const uint32_t a0 = r.s0, a1 = r.s1, t = r.s2;
        const uint32_t m = 0x0f0f0f0f, u = 0x03030303;
        const uint32_t b0 = (a0 & m) | ((t & u) << 4), b1 = (a1 & m) | (((t >> 2) & u) << 4);
        const uint32_t b2 = ((a0 >> 4) & m) | (((t >> 4) & u) << 4), b3 = ((a1 >> 4) & m) | (((t >> 6) & u) << 4);
        lo = (uint64_t)b0 | ((uint64_t)b1 << 32);
        hi = (uint64_t)b2 | ((uint64_t)b3 << 32);
        d = half_bits(r.dw);
    }
    int scale[8], code[8];
#pragma unroll
    for (int H = 0; H < 8; ++H) {
        // Chunk H covers values [32H, 32H + 32); this lane's four are bytes 4j.. of the chunk, with scale 2H + sel.
        const uint64_t word = H < 4 ? lo : hi;
        const int shift = 16 * (H & 3) + 8 * sel;
        uint32_t c = ((H < 4 ? q0 : q1) >> (2 * (H & 3))) & 0x03030303u;
        if (TY == 10) scale[H] = (int)((word >> shift) & 15);
        else {
            scale[H] = (int)((word >> shift) & 63) - 32;
            c |= ((hm >> H) & 0x01010101u) << 2;
            c = ((c | 0x80808080u) - 0x04040404u) ^ 0x80808080u;   // code - 4 in each byte
        }
        code[H] = (int)c;
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int* yb = y[t];
        const float yd = __int_as_float(yb[0]);
        int sum = 0;
#pragma unroll
        for (int H = 0; H < 8; ++H) sum += scale[H] * __dp4a(code[H], yb[1 + 8 * H + j], 0);
        if (TY == 10) {
            const int bs = yb[65 + j];
            const int offset = m0 * (int)(short)(bs & 0xffff) + m1 * (bs >> 16);
            acc[t] = __fmaf_rn(__fmul_rn(-dm, yd), (float)offset, acc[t]);
        }
        acc[t] = __fmaf_rn(__fmul_rn(d, yd), (float)sum, acc[t]);
    }
}
// ((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7)) over the row's eight lanes; every lane ends with the row's value.
__device__ __forceinline__ float reduce_lanes(float a) {
    a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 4));
    a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 1));
    return __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 2));
}

// Canonical Q8_K quantization (canon_expert.hpp), one warp per 256-value block: the bytes quantize_q8_K writes,
// without its serial loops. Lane i holds values i, i + 32, ...; a tie in magnitude keeps the lowest index.
__global__ void __launch_bounds__(256) quantize_kernel(const float* __restrict__ x, int* __restrict__ blocks,
                                                       long long n_blocks) {
    const long long b = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* xb = x + b * 256;
    float v[8], amax = 0.f;
    int at = 0x7fffffff;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = xb[lane + 32 * k];
        const float a = fabsf(v[k]);
        if (a > amax) { amax = a; at = lane + 32 * k; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, amax, o);
        const int other_at = __shfl_xor_sync(0xffffffffu, at, o);
        if (other > amax || (other == amax && other_at < at)) { amax = other; at = other_at; }
    }
    int* out = blocks + b * kQ8kWords;
    if (amax == 0.f) {
        for (int i = lane; i < kQ8kWords; i += 32) out[i] = 0;
        return;
    }
    const float iscale = __fdiv_rn(-127.f, xb[at]);
    int q[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        // nearest(fl(iscale * x)) through the 1.5 * 2^23 bias, as ggml's nearest_int; only the upper clamp.
        const float biased = __fadd_rn(__fmul_rn(iscale, v[k]), 12582912.f);
        const int r = (__float_as_int(biased) & 0x007fffff) - 0x00400000;
        q[k] = r < 127 ? r : 127;
    }
    // qs[4w .. 4w+3] is word 1 + w: value index lane + 32k sits in word 1 + 8k + lane / 4, byte lane % 4.
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        unsigned word = (unsigned)(q[k] & 0xff) << (8 * (lane & 3));
        word |= __shfl_xor_sync(0xffffffffu, word, 1);
        word |= __shfl_xor_sync(0xffffffffu, word, 2);
        if ((lane & 3) == 0) out[1 + 8 * k + (lane >> 2)] = (int)word;
        // bsums[2k] sums lanes 0..15 of this k (values 32k .. 32k+15), bsums[2k+1] lanes 16..31.
        int sum = q[k];
        sum += __shfl_xor_sync(0xffffffffu, sum, 8);
        sum += __shfl_xor_sync(0xffffffffu, sum, 4);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        const int high = __shfl_sync(0xffffffffu, sum, 16);
        if (lane == 0) out[65 + k] = (sum & 0xffff) | (high << 16);
    }
    if (lane == 0) out[0] = __float_as_int(__fdiv_rn(1.f, iscale));
}
void quantize(const float* x, uint8_t* blocks, int64_t n, cudaStream_t s) {
    const long long n_blocks = n / 256;
    quantize_kernel<<<(unsigned)((n_blocks + 7) / 8), 256, 0, s>>>(x, reinterpret_cast<int*>(blocks), n_blocks);
}

// h[entry][row] = swiglu(gate_row . x, up_row . x). A block is 32 rows of one group.
template <int TG>
__global__ void __launch_bounds__(256) gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                 const int32_t* __restrict__ grp_start,
                                                 const int32_t* __restrict__ n_groups,
                                                 const int32_t* __restrict__ ent_tok, const int* __restrict__ xq,
                                                 NativeExpertLayout L, float* __restrict__ h) {
    __shared__ int sx[kMaxTok * kMaxBlocks * kQ8kWords];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, j = tid & 7, row = blockIdx.x * 32 + (tid >> 3);
    const int nb = (int)(L.n_embd / 256), words = nb * kQ8kWords;
    const size_t block_bytes = L.gu_row / nb;
    const uint8_t* gate = reinterpret_cast<const uint8_t*>(grp_ptr[g]) + (size_t)row * L.gu_row;
    const uint8_t* up = gate + L.up_off;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int c0 = e0; c0 < e1; c0 += kMaxTok) {
        const int cn = min(kMaxTok, e1 - c0);
        __syncthreads();
        for (int t = 0; t < cn; ++t) {
            const int* src = xq + (size_t)ent_tok[c0 + t] * words;
            for (int i = tid; i < words; i += 256) sx[t * words + i] = src[i];
        }
        __syncthreads();
        auto pass = [&](auto width) {
            constexpr int NT = decltype(width)::value;
            float ag[NT], au[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) { ag[t] = 0.f; au[t] = 0.f; }
            for (int b = 0; b < nb; b += 2) {
                const bool pair = b + 1 < nb;
                const Raw g0 = fetch<TG>(gate + b * block_bytes, j), u0 = fetch<TG>(up + b * block_bytes, j);
                const Raw g1 = pair ? fetch<TG>(gate + (b + 1) * block_bytes, j) : g0;
                const Raw u1 = pair ? fetch<TG>(up + (b + 1) * block_bytes, j) : u0;
                const int* y[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) y[t] = sx + (t * nb + b) * kQ8kWords;
                superblock<TG, NT>(g0, j, y, ag);
                superblock<TG, NT>(u0, j, y, au);
                if (!pair) break;
#pragma unroll
                for (int t = 0; t < NT; ++t) y[t] += kQ8kWords;
                superblock<TG, NT>(g1, j, y, ag);
                superblock<TG, NT>(u1, j, y, au);
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float gv = reduce_lanes(ag[t]), uv = reduce_lanes(au[t]);
                if (j == 0) h[(size_t)(c0 + t) * L.n_ff + row] = canon::swiglu(gv, uv, L.swiglu_limit, L.gate_skip);
            }
        };
        switch (cn) {
            case 1: pass(std::integral_constant<int, 1>{}); break;
            case 2: pass(std::integral_constant<int, 2>{}); break;
            case 3: pass(std::integral_constant<int, 3>{}); break;
            default: pass(std::integral_constant<int, 4>{}); break;
        }
    }
}

// out[ent_dst[entry]][row] = down_row . h[entry].
template <int TD>
__global__ void __launch_bounds__(256) down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                   const int32_t* __restrict__ grp_start,
                                                   const int32_t* __restrict__ n_groups,
                                                   const int32_t* __restrict__ ent_dst, const int* __restrict__ hq,
                                                   NativeExpertLayout L, float* __restrict__ out) {
    __shared__ int sx[kMaxTok * kMaxBlocks * kQ8kWords];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, j = tid & 7, row = blockIdx.x * 32 + (tid >> 3);
    const int nb = (int)(L.n_ff / 256), words = nb * kQ8kWords;
    const size_t block_bytes = L.d_row / nb;
    const uint8_t* down = reinterpret_cast<const uint8_t*>(grp_ptr[g]) + L.down_off + (size_t)row * L.d_row;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int c0 = e0; c0 < e1; c0 += kMaxTok) {
        const int cn = min(kMaxTok, e1 - c0);
        __syncthreads();
        for (int t = 0; t < cn; ++t) {
            const int* src = hq + (size_t)(c0 + t) * words;
            for (int i = tid; i < words; i += 256) sx[t * words + i] = src[i];
        }
        __syncthreads();
        auto pass = [&](auto width) {
            constexpr int NT = decltype(width)::value;
            float acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[t] = 0.f;
            for (int b = 0; b < nb; b += 4) {
                // Four superblocks' loads at once (the tail repeats the last one and is not applied).
                Raw r[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) r[i] = fetch<TD>(down + min(b + i, nb - 1) * block_bytes, j);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (b + i >= nb) break;
                    const int* y[NT];
#pragma unroll
                    for (int t = 0; t < NT; ++t) y[t] = sx + (t * nb + b + i) * kQ8kWords;
                    superblock<TD, NT>(r[i], j, y, acc);
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float value = reduce_lanes(acc[t]);
                if (j == 0) out[(size_t)ent_dst[c0 + t] * L.n_embd + row] = value;
            }
        };
        switch (cn) {
            case 1: pass(std::integral_constant<int, 1>{}); break;
            case 2: pass(std::integral_constant<int, 2>{}); break;
            case 3: pass(std::integral_constant<int, 3>{}); break;
            default: pass(std::integral_constant<int, 4>{}); break;
        }
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

bool glm_q23_supported(const NativeExpertLayout& L) noexcept {
    const auto type = [](int t) { return t == 10 || t == 11; };
    return type(L.gu_type) && type(L.d_type) && L.n_embd > 0 && L.n_ff > 0 && L.n_embd % 256 == 0 &&
           L.n_ff % 256 == 0 && L.n_embd / 256 <= kMaxBlocks && L.n_ff / 256 <= kMaxBlocks;
}

size_t glm_q23_scratch_bytes(const NativeExpertLayout& L, int64_t tokens, int64_t cap_entries) {
    return round_up((size_t)tokens * (size_t)(L.n_embd / 256) * kQ8kWords * 4) +
           round_up((size_t)cap_entries * (size_t)L.n_ff * sizeof(float)) +
           round_up((size_t)cap_entries * (size_t)(L.n_ff / 256) * kQ8kWords * 4);
}

void glm_q23_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                            const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                            int64_t cap_groups, int64_t cap_entries, const float* x, int64_t tokens, void* scratch,
                            float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0 || tokens <= 0) return;
    if (!glm_q23_supported(L)) {
        std::fprintf(stderr, "glm_q23_expert_grouped: unsupported layout %d/%d %lld x %lld\n", L.gu_type, L.d_type,
                     (long long)L.n_embd, (long long)L.n_ff);
        std::exit(1);
    }
    const auto s = (cudaStream_t)stream;
    auto* xq = static_cast<uint8_t*>(scratch);
    auto* h = reinterpret_cast<float*>(xq + round_up((size_t)tokens * (size_t)(L.n_embd / 256) * kQ8kWords * 4));
    auto* hq = reinterpret_cast<uint8_t*>(h) + round_up((size_t)cap_entries * (size_t)L.n_ff * sizeof(float));
    quantize(x, xq, tokens * L.n_embd, s);
    const dim3 gu_grid((unsigned)(L.n_ff / 32), (unsigned)cap_groups), down_grid((unsigned)(L.n_embd / 32), (unsigned)cap_groups);
    if (L.gu_type == 10)
        gu_kernel<10><<<gu_grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, (const int*)xq, L, h);
    else gu_kernel<11><<<gu_grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, (const int*)xq, L, h);
    check("glm_q23_expert_grouped/gu");
    quantize(h, hq, cap_entries * L.n_ff, s);
    if (L.d_type == 10)
        down_kernel<10><<<down_grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, (const int*)hq, L, out);
    else down_kernel<11><<<down_grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, (const int*)hq, L, out);
    check("glm_q23_expert_grouped/down");
}

}  // namespace strata::kernels
