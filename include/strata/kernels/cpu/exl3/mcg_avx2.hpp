#pragma once
// EXL3 K2 (MCG codebook) tiles on AVX2+FMA without the 256 KiB lookup table (Zen1/Zen2: gathers are slow).
// EXL3 format equations adapted from Turboderp (2025), MIT: third_party/exl3/LICENSE.
//
// A tile is 16x16 weights in a 512-bit cyclic trellis; weight (r, c) is codebook(state(offsets[16r + c])) with a
// 16-bit window per state. For one row half (8 columns) the eight windows sit 64 bits apart, so they are one
// immediate-shift pair on the even/odd word vectors (as in mul1_probe.hpp); both halves of a row pack into one
// vector of 16-bit states. The MCG codebook is computed in registers: p = (state * 0xcbac1fed & 0x8fff8fff) ^
// 0x3b603b60 as 16-bit products; its two 16-bit halves are normal FP16 with exponents 12..15 (the mask clears
// exponent bits 14..12, the xor sets 13 and 12), so F16C converts them, their FP32 sum is exact, and F16C rounds it
// to FP16 (nearest even) as the reference half_round does. Every weight equals codebook(state, 0) bit for bit; only
// the accumulation (FMA, order) differs from the scalar reference.
#include "strata/kernels/cpu/exl3/math.hpp"
namespace strata::cpu::exl3 {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define STRATA_EXL3_MCG_TARGET __attribute__((target("avx2,fma,f16c"), always_inline))

// The eight 16-bit windows of row R, half H, in 32-bit lanes.
template <unsigned R, unsigned H>
STRATA_EXL3_MCG_TARGET inline __m256i mcg_states(__m256i even, __m256i odd, __m256i previous) {
    constexpr unsigned base = exl3_offsets[R * 16 + H * 8], end = (base + 257) * 2;
    constexpr unsigned ai = ((end - 16) / 32) % 16, bi = ((end - 1) / 32) % 16, shift = 31 - ((end - 1) & 31);
    static_assert(ai == 15 || ai == 0 || ai == 1, "row halves read words 2c-1, 2c or 2c+1");
    const __m256i a = ai == 15 ? previous : (ai == 0 ? even : odd), b = bi == 0 ? even : odd;
    return _mm256_and_si256(_mm256_or_si256(_mm256_slli_epi32(a, 32 - shift), _mm256_srli_epi32(b, shift)), _mm256_set1_epi32(65535));
}

struct TileWords {
    __m256i even, odd, previous;
};
STRATA_EXL3_MCG_TARGET inline TileWords tile_words(const uint8_t* p) {
    // word(p, i) is the little-endian u32 at 4i: the 16 words as two strided vectors.
    const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    const __m256i idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i a = _mm256_permutevar8x32_epi32(w0, idx), b = _mm256_permutevar8x32_epi32(w1, idx);
    TileWords t;
    t.even = _mm256_permute2x128_si256(a, b, 0x20);
    t.odd = _mm256_permute2x128_si256(a, b, 0x31);
    t.previous = _mm256_permutevar8x32_epi32(t.odd, _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6));
    return t;
}

// Row R's 16 weights in two vectors, columns permuted: w0 = columns 0-3, 8-11; w1 = columns 4-7, 12-15
// (the in-lane order of vpackusdw). The MCG product in 16-bit lanes: for a state s < 2^16,
// s * 0xcbac1fed mod 2^32 has low half lo16(s * 0x1fed) and high half hi16(s * 0x1fed) + lo16(s * 0xcbac).
template <unsigned R>
STRATA_EXL3_MCG_TARGET inline void mcg_row(const TileWords& t, __m256& w0, __m256& w1) {
    const __m256i s = _mm256_packus_epi32(mcg_states<R, 0>(t.even, t.odd, t.previous), mcg_states<R, 1>(t.even, t.odd, t.previous));
    const __m256i k1 = _mm256_set1_epi16(short(0x1fed)), k2 = _mm256_set1_epi16(short(0xcbac));
    const __m256i mask = _mm256_set1_epi16(short(0x8fff)), flip = _mm256_set1_epi16(short(0x3b60));
    const __m256i lo = _mm256_xor_si256(_mm256_and_si256(_mm256_mullo_epi16(s, k1), mask), flip);
    const __m256i hi = _mm256_xor_si256(_mm256_and_si256(_mm256_add_epi16(_mm256_mulhi_epu16(s, k1), _mm256_mullo_epi16(s, k2)), mask), flip);
    // Both halves are normal FP16 (exponents 12..15): their FP32 sum is exact; round it to FP16 and back.
    const __m256 v0 = _mm256_add_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(lo)), _mm256_cvtph_ps(_mm256_castsi256_si128(hi)));
    const __m256 v1 = _mm256_add_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(lo, 1)), _mm256_cvtph_ps(_mm256_extracti128_si256(hi, 1)));
    w0 = _mm256_cvtph_ps(_mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    w1 = _mm256_cvtph_ps(_mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Accumulators in the permuted column order -> 16 outputs in natural order.
STRATA_EXL3_MCG_TARGET inline void mcg_store(float* y, __m256 a0, __m256 a1) {
    _mm256_storeu_ps(y, _mm256_permute2f128_ps(a0, a1, 0x20));
    _mm256_storeu_ps(y + 8, _mm256_permute2f128_ps(a0, a1, 0x31));
}

// The 256 tile weights (row-major r * 16 + c), for tests.
template <unsigned R = 0>
STRATA_EXL3_MCG_TARGET inline void mcg_tile_rows(const TileWords& t, float* out) {
    __m256 w0, w1;
    mcg_row<R>(t, w0, w1);
    mcg_store(out + R * 16, w0, w1);
    if constexpr (R < 15) mcg_tile_rows<R + 1>(t, out);
}
__attribute__((target("avx2,fma,f16c"))) inline void mcg_tile(const uint8_t* p, float* out) { mcg_tile_rows(tile_words(p), out); }

// acc[2k], acc[2k+1] (token k, permuted columns) += x_k[r] * W[r, :] over the tile's 16 rows.
template <int NT, unsigned R = 0>
STRATA_EXL3_MCG_TARGET inline void mcg_tile_fma(const TileWords& t, const float* x, size_t x_stride, __m256* acc) {
    __m256 w0, w1;
    mcg_row<R>(t, w0, w1);
    for (int k = 0; k < NT; ++k) {
        const __m256 xv = _mm256_broadcast_ss(x + size_t(k) * x_stride + R);
        acc[2 * k] = _mm256_fmadd_ps(xv, w0, acc[2 * k]);
        acc[2 * k + 1] = _mm256_fmadd_ps(xv, w1, acc[2 * k + 1]);
    }
    if constexpr (R < 15) mcg_tile_fma<NT, R + 1>(t, x, x_stride, acc);
}

// y_k[ot*16 .. +16] for output tiles [ot0, ot1) of packed rows ([ot][it] tiles), NT tokens of input h (stride in).
template <int NT>
__attribute__((target("avx2,fma,f16c"))) inline void mcg_rows(const uint8_t* packed, size_t in, size_t ot0, size_t ot1, const float* h, size_t out,
                                             float* y) {
    const size_t tiles = in / 16;
    for (size_t ot = ot0; ot < ot1; ++ot) {
        __m256 acc[2 * NT];
        for (int k = 0; k < 2 * NT; ++k) acc[k] = _mm256_setzero_ps();
        const uint8_t* p = packed + (ot - ot0) * tiles * 64;
        for (size_t it = 0; it < tiles; ++it) {
            _mm_prefetch(reinterpret_cast<const char*>(p + (it + 32) * 64), _MM_HINT_T0);
            mcg_tile_fma<NT>(tile_words(p + it * 64), h + it * 16, in, acc);
        }
        for (int k = 0; k < NT; ++k) mcg_store(y + size_t(k) * out + ot * 16, acc[2 * k], acc[2 * k + 1]);
    }
}

// Any token count: groups of up to 4 tokens per pass over the tiles (8 accumulators stay in registers).
inline void mcg_rows_tokens(const uint8_t* packed, size_t in, size_t ot0, size_t ot1, const float* h, size_t ntok, size_t out, float* y) {
    for (size_t k0 = 0; k0 < ntok; k0 += 4) {
        const size_t n = std::min<size_t>(4, ntok - k0);
        const float* hk = h + k0 * in;
        float* yk = y + k0 * out;
        switch (n) {
        case 1: mcg_rows<1>(packed, in, ot0, ot1, hk, out, yk); break;
        case 2: mcg_rows<2>(packed, in, ot0, ot1, hk, out, yk); break;
        case 3: mcg_rows<3>(packed, in, ot0, ot1, hk, out, yk); break;
        default: mcg_rows<4>(packed, in, ot0, ot1, hk, out, yk); break;
        }
    }
}
inline bool mcg_avx2_supported() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}
#else
inline bool mcg_avx2_supported() { return false; }
#endif
}  // namespace strata::cpu::exl3
