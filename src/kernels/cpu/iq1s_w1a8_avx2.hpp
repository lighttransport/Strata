// src/kernels/cpu/iq1s_w1a8_avx2.hpp - IQ1_S x int8 activations without the codebook arithmetic ("W1A8").
//
// An IQ1_S value is d * (2s+1) * (g + delta): g in {-1, 0, +1} from the 2048-entry iq1s_grid (8 values per entry,
// 11-bit index = qs byte | 3 bits of qh), s the 3-bit block scale, delta = +-1/8 per 32-value block (qh bit 15).
// The engine's kernel (muse worktree Fmt32<19>) rebuilds 8g +- 1 per byte with three vpaddb, splits it into a
// magnitude and a sign, and applies the sign to the activation per token: about 14 cycles of decode per 32 weights
// on Zen 1, which made 15 cores compute-bound at 18 GB/s (92 GMAC/s at one token).
//
// Here the grid is stored once as u8 bytes g + 1 in {0, 1, 2} (16 KiB, L1-resident), so one vpmaddubsw(u, x) is
// sum((g+1) x) and the exact integer dot of a 32-block is
//     sum(w x) * 8 / (d (2s+1))  =  8 * (sum(u x) - B) + sign * B,     B = sum(x) over the block (int, per token),
// accumulated as int32 and scaled once per 256-block: no per-token sign handling, no vpaddb chain, no float until
// the block scale. Bit-exact with the dequantized dot in integer arithmetic; the float accumulation order differs
// from ggml's and from Fmt32<19>'s (rel error ~1e-7).
//
// Activations: the caller quantizes a 256-wide block to int8 with one float scale (q8 as the engine's Q8_K) and
// the per-32 sums B (eight int32); `Act` below. Weights: plain IQ1_S rows as in a GGUF/sidecar (50 bytes per 256).
//
// Only the AVX2 form is here (the bench); the AVX-VNNI/AVX-512 forms replace vpmaddubsw+vpmaddwd by vpdpbusd.
#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu::iq1s_w1a8 {

constexpr int kBlock = 256;         // weights per IQ1_S block
constexpr int kBlockBytes = 50;     // fp16 d, qs[32], qh[8] (u16)

struct Act {                        // one 256-wide activation block
    float d;                        // dequantized value = d * q
    int8_t q[kBlock];
    int32_t bsum[8];                // sum of q over each 32-value sub-block
};

/// Quantize 256 floats to one Act (symmetric, round to nearest, scale = max/127), as a decode engine does per token.
inline void quantize_act(const float* x, Act& a) {
    float amax = 0.f;
    for (int i = 0; i < kBlock; ++i) amax = x[i] > amax ? x[i] : (-x[i] > amax ? -x[i] : amax);
    const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
    a.d = d;
    for (int b = 0; b < 8; ++b) {
        int32_t s = 0;
        for (int i = 0; i < 32; ++i) { const int v = (int) lrintf(x[32 * b + i] * id); a.q[32 * b + i] = (int8_t) v; s += v; }
        a.bsum[b] = s;
    }
}

/// The grid as u8 bytes g + 1, built once from ggml's iq1s_grid (uint64 entries whose bytes are int8 -1/0/1).
struct Table {
    alignas(32) uint64_t u[2048];
    explicit Table(const uint64_t* iq1s_grid) {
        for (int i = 0; i < 2048; ++i) {
            uint64_t v = iq1s_grid[i], o = 0;
            for (int b = 0; b < 8; ++b) { const int8_t g = (int8_t) ((v >> (8 * b)) & 0xff); o |= uint64_t((uint8_t) (g + 1)) << (8 * b); }
            u[i] = o;
        }
    }
};

inline float fp16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t bits;
    if (e == 0) { if (m == 0) bits = s << 31; else { int ee = -1; uint32_t mm = m; do { ++ee; mm <<= 1; } while (!(mm & 1024)); bits = (s << 31) | ((127 - 15 - ee) << 23) | ((mm & 1023) << 13); } }
    else if (e == 31) bits = (s << 31) | 0x7f800000u | (m << 13);
    else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

inline int32_t hsum8(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
}

/// Four 8-byte grid entries -> one 256-bit vector: two movq+movhps pairs and one vinserti128 (5 uops) instead of the
/// mov/pinsrq/inserti128 chain _mm256_set_epi64x compiles to on Zen 1.
inline __m256i load4x64(const uint64_t* a, const uint64_t* b, const uint64_t* c, const uint64_t* d) {
    const __m128d lo = _mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a))), reinterpret_cast<const double*>(b));
    const __m128d hi = _mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(c))), reinterpret_cast<const double*>(d));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_castpd_si128(lo)), _mm_castpd_si128(hi), 1);
}

/// Dot of one IQ1_S row (nblocks x 256 weights) with NT activation rows; res[t] = sum_i w_i x_t,i.
template <int NT>
inline void row_dot(const Table& tab, const uint8_t* row, int nblocks, const Act* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * kBlockBytes;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        const uint8_t* qs = blk + 2;
        const uint16_t* qh = reinterpret_cast<const uint16_t*>(blk + 34);
        __m256i acci[NT];
        int32_t corr[NT];
        for (int t = 0; t < NT; ++t) { acci[t] = _mm256_setzero_si256(); corr[t] = 0; }
        for (int ib = 0; ib < 8; ++ib) {
            const uint32_t h = qh[ib];
            uint32_t q; std::memcpy(&q, qs + 4 * ib, 4);
            const __m256i u = load4x64(&tab.u[(q & 0xff) | ((h & 7) << 8)], &tab.u[((q >> 8) & 0xff) | ((h & 0x38) << 5)],
                                       &tab.u[((q >> 16) & 0xff) | ((h & 0x1c0) << 2)], &tab.u[(q >> 24) | ((h & 0xe00) >> 1)]);
            const int sc = 2 * (int) ((h >> 12) & 7) + 1;          // 1..15
            const int sign = (h & 0x8000) ? -1 : 1;
            const __m256i sc8 = _mm256_set1_epi16((short) (8 * sc));  // the 8 x sum(u x) term, scaled by (2s+1)
            const int csc = sc * (sign - 8);                           // the (sign - 8) B term, per token below
            for (int t = 0; t < NT; ++t) {
                const __m256i xv = _mm256_loadu_si256((const __m256i*) (y[t][i].q + 32 * ib));
                const __m256i p16 = _mm256_maddubs_epi16(u, xv);        // sum over pairs of (g+1) x, |.| <= 508
                acci[t] = _mm256_add_epi32(acci[t], _mm256_madd_epi16(p16, sc8));
                corr[t] += csc * y[t][i].bsum[ib];
            }
        }
        const float dw = fp16_to_f32(dh) * 0.125f;
        for (int t = 0; t < NT; ++t) {
            const int32_t dot8 = hsum8(acci[t]) + corr[t];
            accf[t] = _mm256_add_ps(accf[t], _mm256_set1_ps(dw * y[t][i].d * (float) dot8));
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm256_cvtss_f32(accf[t]);
}

// ---- the engine's current decode (muse worktree Fmt32<19>) for an A/B in the same harness ------------------------
// magnitude |8g +- 1| as the maddubs operand, the signed value for vpsignb, scale (2s+1), K = 1/8.
inline void decode_baseline(const uint64_t* grid, const uint8_t* blk, int ib, __m256i& g, __m256i& sgn, int& sc) {
    const uint8_t* q = blk + 2 + 4 * ib;
    uint16_t hh; std::memcpy(&hh, blk + 34 + 2 * ib, 2); const uint32_t h = hh;
    const __m256i v = _mm256_set_epi64x((long long) grid[q[3] | ((h >> 1) & 0x700)], (long long) grid[q[2] | ((h << 2) & 0x700)],
                                        (long long) grid[q[1] | ((h << 5) & 0x700)], (long long) grid[q[0] | ((h << 8) & 0x700)]);
    const __m256i v2 = _mm256_add_epi8(v, v), v4 = _mm256_add_epi8(v2, v2);
    const __m256i w = _mm256_add_epi8(_mm256_add_epi8(v4, v4), _mm256_set1_epi8((h & 0x8000) ? -1 : 1));
    g = _mm256_abs_epi8(w);
    sgn = w;
    sc = 2 * (int) ((h >> 12) & 7) + 1;
}

template <int NT>
inline void row_dot_baseline(const uint64_t* grid, const uint8_t* row, int nblocks, const Act* const* y, float* res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * kBlockBytes;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {
            __m256i g, sgn; int sc;
            decode_baseline(grid, blk, ib, g, sgn, sc);
            const __m256i scv = _mm256_set1_epi16((short) sc);
            for (int t = 0; t < NT; ++t) {
                const __m256i xv = _mm256_loadu_si256((const __m256i*) (y[t][i].q + 32 * ib));
                const __m256i ys = _mm256_sign_epi8(xv, sgn);
                acci[t] = _mm256_add_epi32(acci[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), scv));
            }
        }
        const float dw = fp16_to_f32(dh) * 0.125f;
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(accf[t]), _mm256_extractf128_ps(accf[t], 1));
        s = _mm_add_ps(s, _mm_movehdup_ps(s));
        s = _mm_add_ss(s, _mm_movehl_ps(s, s));
        res[t] = _mm_cvtss_f32(s);
    }
}

} // namespace strata::kernels::cpu::iq1s_w1a8
