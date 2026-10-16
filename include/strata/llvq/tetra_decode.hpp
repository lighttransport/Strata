// Decoding llvq Tetra matrices (strata-glm-llvq-encode's records): word -> 24 lattice values, and GEMV / dequant on
// the CPU (scalar reference and an AVX2+FMA kernel for Zen1/Zen2: no gathers, no pdep, byte shuffles).
//
// A matrix W (rows x cols) is stored rotated: W' = W Q^T with llvq's Rotation Q, so y = W x = W' (Q x). Row r,
// block b holds a word w with gain g: W'[r, 24b + order[t]] = rs[r] * c[g] / sqrt(sum y^2) * y[t] for the word's
// point y in trio order; the cols % 24 tail columns are BF16. Kernels take the activation already rotated and,
// per block, permuted into trio order (prepare_activation), so a word is a contiguous 24-wide dot.
#pragma once
#include "strata/llvq/quantize.hpp"
#include "strata/llvq/tetra.hpp"
#include <cstring>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace strata::llvq {

struct DecodeTables {
    uint8_t pre[STATES][2], suf[STATES][2];
    uint8_t br_c[STATES][BRANCHES], br_s[STATES][BRANCHES];
    uint32_t rows[ROWS];
    int8_t vals[2][16];  // per parity p: VALUES[p][0..7], VALUES[p + 2][0..7] (index rho + 8 * pattern bit)
    uint32_t order[DIM];
    float rsqrt16[385];  // 1 / sqrt(16 m) for the shell m = sum y^2 / 16 (|y| <= 16 over 24 values: m <= 384)

    static DecodeTables build(const Tetra& t) {
        DecodeTables d{};
        for (int s = 0; s < STATES; ++s) {
            d.pre[s][0] = t.prefixes[s][0]; d.pre[s][1] = t.prefixes[s][1];
            d.suf[s][0] = t.suffixes[s][0]; d.suf[s][1] = t.suffixes[s][1];
            for (int b = 0; b < BRANCHES; ++b) { d.br_c[s][b] = t.branches[s][b].first; d.br_s[s][b] = t.branches[s][b].second; }
        }
        for (int i = 0; i < ROWS; ++i) d.rows[i] = t.rows[i];
        for (int p = 0; p < 2; ++p)
            for (int r = 0; r < 8; ++r) { d.vals[p][r] = int8_t(VALUES[p][r]); d.vals[p][8 + r] = int8_t(VALUES[p + 2][r]); }
        for (int j = 0; j < DIM; ++j) d.order[j] = t.order[j];
        d.rsqrt16[0] = 0.0f;
        for (int m = 1; m <= 384; ++m) d.rsqrt16[m] = 1.0f / std::sqrt(float(16 * m));
        return d;
    }

    // The three (pattern byte, packed row) pairs and the parity of a word.
    struct Parts {
        uint32_t p;
        uint8_t c[3];
        uint32_t row[3];
    };
    Parts parts(uint64_t w) const {
        const uint32_t p = uint32_t(w & 1), r = uint32_t(w >> 1) & 1, s8 = uint32_t(w >> 2) & 63, b1 = uint32_t(w >> 8) & 1,
                       i1 = uint32_t(w >> 9) & 2047, b2 = uint32_t(w >> 20) & 15, i2 = uint32_t(w >> 24) & 2047,
                       b3 = uint32_t(w >> 35) & 1, i3 = uint32_t(w >> 36) & 2047;
        const uint32_t s16 = br_s[s8][b2];
        const uint32_t delta = i2 >= uint32_t(N0_MIXED);
        Parts q;
        q.p = p;
        q.c[0] = pre[s8][b1];
        q.c[1] = br_c[s8][b2];
        q.c[2] = suf[s16][b3];
        q.row[0] = rows[CLASS_ROWS * r + i1];
        q.row[1] = delta ? rows[CLASS_ROWS + i2 - N0_MIXED] : rows[i2];
        q.row[2] = rows[CLASS_ROWS * ((p ^ r ^ delta) & 1) + i3];
        return q;
    }
    // 24 values in trio order.
    void decode_trio(uint64_t w, int8_t y[DIM]) const {
        Parts q = parts(w);
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < SECTION; ++j)
                y[SECTION * k + j] = vals[q.p][((q.row[k] >> (4 * j)) & 15) + 8 * ((q.c[k] >> j) & 1)];
    }
};

// One matrix record of strata-glm-llvq-encode (see tools/glm_llvq_encode.cpp, write_matrix).
struct TetraMatrix {
    uint32_t rows = 0, cols = 0, nb = 0, tail = 0, stride = 0;  // stride: bytes per row of words
    double centroids[2] = {1, 1};
    const float* scales = nullptr;
    const uint8_t* words = nullptr;
    const uint16_t* tail_bf16 = nullptr;

    // Parses one record at p; returns the pointer past it.
    const uint8_t* parse(const uint8_t* p) {
        std::memcpy(&rows, p, 4);
        std::memcpy(&cols, p + 4, 4);
        std::memcpy(centroids, p + 8, 16);
        nb = cols / 24;
        tail = cols - 24 * nb;
        stride = (nb * 6 + 7) / 8 * 8;
        scales = reinterpret_cast<const float*>(p + 24);
        words = p + 24 + 4 * size_t(rows);
        tail_bf16 = reinterpret_cast<const uint16_t*>(words + size_t(stride) * rows);
        return reinterpret_cast<const uint8_t*>(tail_bf16 + size_t(rows) * tail);
    }
    uint64_t word(uint32_t r, uint32_t b) const {
        uint64_t w = 0;
        std::memcpy(&w, words + size_t(r) * stride + 6 * size_t(b), 6);
        return w;
    }
};

inline float bf16_to_f32(uint16_t h) {
    uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Rotated, trio-permuted activation for one token: xp[24b + t] = (Q x)[24b + order[t]], tail unpermuted.
inline void prepare_activation(const DecodeTables& T, const Rotation& rot, const float* x, size_t cols, float* xp) {
    std::vector<double> v(x, x + cols);
    rot.apply(v.data());
    const size_t nb = cols / 24;
    for (size_t b = 0; b < nb; ++b)
        for (int t = 0; t < DIM; ++t) xp[24 * b + t] = float(v[24 * b + T.order[t]]);
    for (size_t c = 24 * nb; c < cols; ++c) xp[c] = float(v[c]);
}

// Row r of W' in natural (rotated) column order, f32 -- the reference for the kernels.
inline void dequant_row(const DecodeTables& T, const TetraMatrix& m, uint32_t r, float* out) {
    const float rs = m.scales[r];
    for (uint32_t b = 0; b < m.nb; ++b) {
        int8_t y[DIM];
        uint64_t w = m.word(r, b);
        T.decode_trio(w, y);
        int32_t ss = 0;
        for (int8_t v : y) ss += int32_t(v) * v;
        const float s = ss ? float(m.centroids[(w >> 47) & 1]) * rs / std::sqrt(float(ss)) : 0.0f;
        for (int t = 0; t < DIM; ++t) out[24 * b + T.order[t]] = float(y[t]) * s;
    }
    for (uint32_t c = 0; c < m.tail; ++c) out[24 * m.nb + c] = bf16_to_f32(m.tail_bf16[size_t(r) * m.tail + c]);
}

// y[r * ntok + k] = sum_c W'[r, c] xp_k[c] for rows [r0, r1), ntok tokens (xp: ntok x cols, prepared).
inline void gemv_scalar(const DecodeTables& T, const TetraMatrix& m, const float* xp, int ntok, uint32_t r0, uint32_t r1, float* y) {
    for (uint32_t r = r0; r < r1; ++r) {
        float acc[8] = {};
        for (uint32_t b = 0; b < m.nb; ++b) {
            int8_t v[DIM];
            uint64_t w = m.word(r, b);
            T.decode_trio(w, v);
            int32_t ss = 0;
            for (int8_t q : v) ss += int32_t(q) * q;
            if (!ss) continue;
            const float s = float(m.centroids[(w >> 47) & 1]) / std::sqrt(float(ss));
            for (int k = 0; k < ntok; ++k) {
                float d = 0;
                for (int t = 0; t < DIM; ++t) d += float(v[t]) * xp[size_t(k) * m.cols + 24 * b + t];
                acc[k] += s * d;
            }
        }
        for (int k = 0; k < ntok; ++k) {
            float tail = 0;
            for (uint32_t c = 0; c < m.tail; ++c)
                tail += bf16_to_f32(m.tail_bf16[size_t(r) * m.tail + c]) * xp[size_t(k) * m.cols + 24 * m.nb + c];
            y[size_t(r) * ntok + k] = acc[k] * m.scales[r] + tail;
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
// AVX2+FMA (Zen1/Zen2): per word, the three sections' nibbles unpacked to bytes, the pattern bits added as +8, one
// pshufb into the parity's 16-value table, int8 -> f32, FMAs against up to 8 tokens.
inline void gemv_avx2(const DecodeTables& T, const TetraMatrix& m, const float* xp, int ntok, uint32_t r0, uint32_t r1, float* y) {
    const __m128i lut[2] = {_mm_loadu_si128(reinterpret_cast<const __m128i*>(T.vals[0])),
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(T.vals[1]))};
    const __m128i bitsel = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, char(128), 1, 2, 4, 8, 16, 32, 64, char(128));
    const __m128i nib = _mm_set1_epi8(0x0f), eight = _mm_set1_epi8(8);
    const float c0 = float(m.centroids[0]), c1 = float(m.centroids[1]);
    for (uint32_t r = r0; r < r1; ++r) {
        __m256 acc[8];
        for (int k = 0; k < 8; ++k) acc[k] = _mm256_setzero_ps();
        const uint8_t* wrow = m.words + size_t(r) * m.stride;
        for (uint32_t b = 0; b < m.nb; ++b) {
            uint64_t w = 0;
            std::memcpy(&w, wrow + 6 * size_t(b), 6);
            const DecodeTables::Parts q = T.parts(w);
            // Sections 0,1 in one 16-byte vector, section 2 in another.
            __m128i rows01 = _mm_set_epi32(0, 0, int(q.row[1]), int(q.row[0]));
            __m128i lo = _mm_and_si128(rows01, nib), hi = _mm_and_si128(_mm_srli_epi16(rows01, 4), nib);
            __m128i rho01 = _mm_unpacklo_epi8(lo, hi);  // n0..n7 of row0, n0..n7 of row1
            __m128i rows2 = _mm_cvtsi32_si128(int(q.row[2]));
            __m128i lo2 = _mm_and_si128(rows2, nib), hi2 = _mm_and_si128(_mm_srli_epi16(rows2, 4), nib);
            __m128i rho2 = _mm_unpacklo_epi8(lo2, hi2);
            __m128i c01 = _mm_unpacklo_epi64(_mm_set1_epi8(char(q.c[0])), _mm_set1_epi8(char(q.c[1])));
            __m128i c2 = _mm_set1_epi8(char(q.c[2]));
            __m128i add01 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(c01, bitsel), bitsel), eight);
            __m128i add2 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(c2, bitsel), bitsel), eight);
            __m128i v01 = _mm_shuffle_epi8(lut[q.p], _mm_add_epi8(rho01, add01));
            __m128i v2 = _mm_shuffle_epi8(lut[q.p], _mm_add_epi8(rho2, add2));
            // sum of squares (int16 pairs; values |y| <= 16, so 24 squares fit)
            __m128i sq01 = _mm_maddubs_epi16(_mm_abs_epi8(v01), _mm_abs_epi8(v01));
            __m128i sq2 = _mm_maddubs_epi16(_mm_abs_epi8(v2), _mm_abs_epi8(v2));
            __m128i sq = _mm_add_epi16(sq01, _mm_unpacklo_epi64(sq2, _mm_setzero_si128()));
            sq = _mm_madd_epi16(sq, _mm_set1_epi16(1));
            sq = _mm_add_epi32(sq, _mm_shuffle_epi32(sq, 0x4e));
            sq = _mm_add_epi32(sq, _mm_shuffle_epi32(sq, 0xb1));
            const int ss = _mm_cvtsi128_si32(sq);
            if (!ss) continue;
            const float sc = (w >> 47 & 1 ? c1 : c0) * T.rsqrt16[ss >> 4];
            const __m256 y0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v01));
            const __m256 y1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v01, 8)));
            const __m256 y2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v2));
            if (ntok == 1) {  // dot first, then one scaled accumulate
                const float* xk = xp + 24 * size_t(b);
                __m256 d = _mm256_mul_ps(y0, _mm256_loadu_ps(xk));
                d = _mm256_fmadd_ps(y1, _mm256_loadu_ps(xk + 8), d);
                d = _mm256_fmadd_ps(y2, _mm256_loadu_ps(xk + 16), d);
                acc[0] = _mm256_fmadd_ps(_mm256_set1_ps(sc), d, acc[0]);
                continue;
            }
            const __m256 s = _mm256_set1_ps(sc);
            const __m256 f0 = _mm256_mul_ps(s, y0), f1 = _mm256_mul_ps(s, y1), f2 = _mm256_mul_ps(s, y2);
            for (int k = 0; k < ntok; ++k) {
                const float* xk = xp + size_t(k) * m.cols + 24 * size_t(b);
                acc[k] = _mm256_fmadd_ps(f0, _mm256_loadu_ps(xk), acc[k]);
                acc[k] = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xk + 8), acc[k]);
                acc[k] = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xk + 16), acc[k]);
            }
        }
        for (int k = 0; k < ntok; ++k) {
            __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[k]), _mm256_extractf128_ps(acc[k], 1));
            h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            h = _mm_add_ss(h, _mm_movehdup_ps(h));
            float tail = 0;
            for (uint32_t c = 0; c < m.tail; ++c)
                tail += bf16_to_f32(m.tail_bf16[size_t(r) * m.tail + c]) * xp[size_t(k) * m.cols + 24 * m.nb + c];
            y[size_t(r) * ntok + k] = _mm_cvtss_f32(h) * m.scales[r] + tail;
        }
    }
}
#endif

}  // namespace strata::llvq
