// native_tc_mmvq.cu - see include/strata/kernels/native_tc_mmvq.hpp.  Weights-once quantized matvec for 1..64 columns on tensor cores (CUDA only).
#include "strata/kernels/native_tc_mmvq.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace strata::kernels {
namespace {

using namespace nvcuda;
constexpr int kNW = 4;      // warps a block: each takes every 4th 256-column super-block of the rows
constexpr int kLDA = 40;    // shared-memory row stride (halves) of the decoded 32 x 32 tile: 32 values + 8 padding
enum { FQ8 = 8, FQ4 = 12, FQ5 = 13, FQ6 = 14 };

template <int F> struct Blk;
template <> struct Blk<FQ4> { static constexpr int BYTES = 144, WORDS = 36; };
template <> struct Blk<FQ5> { static constexpr int BYTES = 176, WORDS = 44; };
template <> struct Blk<FQ6> { static constexpr int BYTES = 210, WORDS = 53; };
template <> struct Blk<FQ8> { static constexpr int BYTES = 272, WORDS = 68; };   // eight Q8_0 blocks (34 bytes each) = one 256-value super-block, 16-byte aligned

/// The super-block at `p` as 32-bit words.  Q4_K / Q5_K blocks are 16-byte aligned; a Q6_K block is only 2-byte aligned, so it is read as aligned words
/// and shifted (the read runs at most 4 bytes past the block: the weights carry 256 bytes of padding).
template <int F>
__device__ __forceinline__ void load_block(const uint8_t* p, uint32_t* w) {
    if constexpr (F == FQ6) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
        const unsigned sh = unsigned(a & 3) * 8u;
#pragma unroll
        for (int i = 0; i < Blk<F>::WORDS; ++i) w[i] = __funnelshift_r(wp[i], wp[i + 1], sh);
    } else {
        const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
        for (int i = 0; i < Blk<F>::WORDS / 4; ++i) {
            const uint4 t = q[i];
            w[4 * i] = t.x; w[4 * i + 1] = t.y; w[4 * i + 2] = t.z; w[4 * i + 3] = t.w;
        }
    }
}

/// Four byte-values (0..255 in a word) -> four halves value * s + nm, with the byte offset `off` removed first (1024 + v is exact in fp16).
__device__ __forceinline__ void store4(__half* dst, uint32_t v, __half2 s, __half2 nm, __half2 off) {
    const uint32_t a = __byte_perm(v, 0x64646464u, 0x5140), b = __byte_perm(v, 0x64646464u, 0x5342);
    const __half2 wa = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&a), off), s, nm);
    const __half2 wb = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&b), off), s, nm);
    *reinterpret_cast<__half2*>(dst) = wa;
    *reinterpret_cast<__half2*>(dst + 2) = wb;
}

__device__ __forceinline__ int sc_byte(const uint32_t* w, int i) { return int((w[1 + (i >> 2)] >> (8 * (i & 3))) & 0xffu); }

/// ggml's get_scale_min_k4 on the 12 scale bytes at words 1..3.
__device__ __forceinline__ void sc_min(const uint32_t* w, int j, int& d, int& m) {
    if (j < 4) { d = sc_byte(w, j) & 63; m = sc_byte(w, j + 4) & 63; }
    else { d = (sc_byte(w, j + 4) & 0xF) | ((sc_byte(w, j - 4) >> 6) << 4); m = (sc_byte(w, j + 4) >> 4) | ((sc_byte(w, j) >> 6) << 4); }
}

/// The 32 values of sub-block `sb` (0..7) of one super-block as halves, into out[0..31].
template <int F>
__device__ __forceinline__ void dec32(const uint32_t* w, int sb, __half* out) {
    const __half2 off1024 = __float2half2_rn(1024.f);
    if constexpr (F == FQ4 || F == FQ5) {
        const __half2 dm = *reinterpret_cast<const __half2*>(&w[0]);
        const float d = __low2float(dm), dmin = __high2float(dm);
        int sc, mn;
        sc_min(w, sb, sc, mn);
        const __half2 s = __float2half2_rn(d * (float) sc), nm = __float2half2_rn(-(dmin * (float) mn));
        const int j = sb >> 1, hi = sb & 1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t v;
            if constexpr (F == FQ5) {
                const uint32_t q = w[12 + 8 * j + i], h = w[4 + i];
                v = hi ? (((q >> 4) & 0x0F0F0F0Fu) | (((h >> (2 * j + 1)) & 0x01010101u) << 4)) : ((q & 0x0F0F0F0Fu) | (((h >> (2 * j)) & 0x01010101u) << 4));
            } else {
                const uint32_t q = w[4 + 8 * j + i];
                v = hi ? ((q >> 4) & 0x0F0F0F0Fu) : (q & 0x0F0F0F0Fu);
            }
            store4(out + 4 * i, v, s, nm, off1024);
        }
    } else if constexpr (F == FQ8) {   // sub-block sb = Q8_0 block sb: half d, then 32 int8, at byte 34 * sb of the 272
        const int o = 34 * sb;
        auto byte_at = [&](int off) -> uint32_t { return (w[off >> 2] >> (8 * (off & 3))) & 0xffu; };
        const float d = __half2float(__ushort_as_half((unsigned short) (byte_at(o) | (byte_at(o + 1) << 8))));
        const __half2 s = __float2half2_rn(d), zero = __float2half2_rn(0.f), off = __float2half2_rn(1152.f);   // 1024 + 128: the int8 as offset binary
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t v = (byte_at(o + 2 + 4 * i) | (byte_at(o + 3 + 4 * i) << 8) | (byte_at(o + 4 + 4 * i) << 16) | (byte_at(o + 5 + 4 * i) << 24)) ^ 0x80808080u;
            store4(out + 4 * i, v, s, zero, off);
        }
    } else {   // Q6_K: ql words 0..31, qh words 32..47, int8 scales words 48..51, d in the low half of word 52
        const float d = __half2float(__ushort_as_half((unsigned short) (w[52] & 0xffffu)));
        const int hh = sb >> 2, qq = sb & 3;
        const int sc0 = (int) (int8_t) ((w[48 + ((hh * 8 + 2 * qq) >> 2)] >> (8 * ((hh * 8 + 2 * qq) & 3))) & 0xffu);
        const int sc1 = (int) (int8_t) ((w[48 + ((hh * 8 + 2 * qq + 1) >> 2)] >> (8 * ((hh * 8 + 2 * qq + 1) & 3))) & 0xffu);
        const __half2 s0 = __float2half2_rn(d * (float) sc0), s1 = __float2half2_rn(d * (float) sc1);
        const __half2 zero = __float2half2_rn(0.f), off = __float2half2_rn(1056.f);   // 1024 + 32: value - 32
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t ql = w[hh * 16 + (qq & 1) * 8 + i], qh = w[32 + hh * 8 + i];
            const uint32_t nib = (qq >> 1) ? ((ql >> 4) & 0x0F0F0F0Fu) : (ql & 0x0F0F0F0Fu);
            const uint32_t v = nib | (((qh >> (2 * qq)) & 0x03030303u) << 4);
            // all four values of a word use the scale of their 16-value half: words 0..3 the first, 4..7 the second
            const __half2 s = i < 4 ? s0 : s1;
            const uint32_t a = __byte_perm(v, 0x64646464u, 0x5140), b = __byte_perm(v, 0x64646464u, 0x5342);
            const __half2 wa = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&a), off), s);
            const __half2 wb = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&b), off), s);
            (void) zero;
            *reinterpret_cast<__half2*>(out + 4 * i) = wa;
            *reinterpret_cast<__half2*>(out + 4 * i + 2) = wb;
        }
    }
}

/// grid = n_out / 32 blocks of 4 warps; warp w takes super-blocks w, w + 4, ...; one lane = one of the block's 32 rows when decoding; NG groups of 8 columns.
template <int F, int NG>
__global__ void __launch_bounds__(kNW * 32) tc_kernel(const uint8_t* __restrict__ wts, const __half* __restrict__ x16, float* __restrict__ y, int K, int n_out,
                                                       int ncols) {
    __shared__ __align__(128) __half As[kNW][32 * kLDA];
    __shared__ __align__(128) float Cs[kNW][NG][32 * 8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 32, nb = K / 256;
    const size_t row_bytes = (size_t) nb * Blk<F>::BYTES;
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> acc[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) wmma::fill_fragment(acc[g], 0.f);
    __half* A = As[warp];
    for (int kb = warp; kb < nb; kb += kNW) {
        uint32_t w[Blk<F>::WORDS + 1];
        load_block<F>(wts + (size_t) (row0 + lane) * row_bytes + (size_t) kb * Blk<F>::BYTES, w);
#pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            dec32<F>(w, sb, A + lane * kLDA);
            __syncwarp();
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                wmma::fragment<wmma::matrix_a, 32, 8, 16, __half, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, A + 16 * kk, kLDA);
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    wmma::fragment<wmma::matrix_b, 32, 8, 16, __half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fb, x16 + (size_t) (g * 8) * K + (size_t) kb * 256 + sb * 32 + 16 * kk, K);
                    wmma::mma_sync(acc[g], fa, fb, acc[g]);
                }
            }
            __syncwarp();
        }
    }
#pragma unroll
    for (int g = 0; g < NG; ++g) wmma::store_matrix_sync(Cs[warp][g], acc[g], 8, wmma::mem_row_major);
    __syncthreads();
    for (int t = threadIdx.x; t < 32 * ncols; t += kNW * 32) {
        const int r = t & 31, c = t >> 5;
        float sum = 0.f;
#pragma unroll
        for (int ww = 0; ww < kNW; ++ww) sum += Cs[ww][c >> 3][r * 8 + (c & 7)];
        y[(size_t) c * n_out + row0 + r] = sum;
    }
}

template <int F>
void launch_type(const void* w, const void* x16, float* y, int K, int n_out, int ncols, cudaStream_t s) {
    const dim3 grid((unsigned) (n_out / 32)), blk(kNW * 32);
    const uint8_t* wp = static_cast<const uint8_t*>(w);
    const __half* xp = static_cast<const __half*>(x16);
    switch ((ncols + 7) / 8) {
#define STRATA_TC_CASE(G) case G: tc_kernel<F, G><<<grid, blk, 0, s>>>(wp, xp, y, K, n_out, ncols); break;
        STRATA_TC_CASE(1) STRATA_TC_CASE(2) STRATA_TC_CASE(3) STRATA_TC_CASE(4) STRATA_TC_CASE(5) STRATA_TC_CASE(6) STRATA_TC_CASE(7) STRATA_TC_CASE(8)
#undef STRATA_TC_CASE
        default: break;
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) dst[i] = __float2half(src[i]);
}

}  // namespace

bool native_tc_mmvq_supported(int ggml_type, int n_in, int n_out, int ncols) noexcept {
    return (ggml_type == FQ4 || ggml_type == FQ5 || ggml_type == FQ6 || ggml_type == FQ8) && n_in > 0 && n_in % 256 == 0 && n_out > 0 && n_out % 32 == 0 && ncols >= 1 &&
           ncols <= kTcMaxCols;
}

void native_tc_mmvq(int ggml_type, const void* weights, const void* x16, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!native_tc_mmvq_supported(ggml_type, n_in, n_out, ncols)) throw std::invalid_argument("native_tc_mmvq: unsupported type or shape");
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (ggml_type) {
        case FQ4: launch_type<FQ4>(weights, x16, y, n_in, n_out, ncols, s); break;
        case FQ5: launch_type<FQ5>(weights, x16, y, n_in, n_out, ncols, s); break;
        case FQ8: launch_type<FQ8>(weights, x16, y, n_in, n_out, ncols, s); break;
        default: launch_type<FQ6>(weights, x16, y, n_in, n_out, ncols, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native_tc_mmvq launch: ") + cudaGetErrorString(e));
}

void native_f32_to_f16(const float* src, void* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t blocks = std::min<int64_t>((n + 255) / 256, 4096);
    f32_to_f16_kernel<<<(unsigned) blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(src, static_cast<__half*>(dst), n);
}

}  // namespace strata::kernels

#else  // HIP: no WMMA path

namespace strata::kernels {
bool native_tc_mmvq_supported(int, int, int, int) noexcept { return false; }
void native_tc_mmvq(int, const void*, const void*, float*, int, int, int, void*) { throw std::runtime_error("native_tc_mmvq: CUDA only"); }
void native_f32_to_f16(const float*, void*, int64_t, void*) { throw std::runtime_error("native_f32_to_f16: CUDA only"); }
}  // namespace strata::kernels

#endif
