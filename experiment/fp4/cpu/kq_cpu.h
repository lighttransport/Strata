// Q2_K / Q3_K expert kernels on AVX2 (ports of ggml_vec_dot_q2_K_q8_K / ggml_vec_dot_q3_K_q8_K, llama.cpp MIT),
// used for the Strata q23 pack mix (Q2_K gate/up, Q3_K down) and the ggml-org Q2_K GGUF mix (Q2_K gate/up,
// MXFP4 down). Included by bench_cpu.cpp after its own helpers.
#pragma once
#include "../common/kquant.h"

#include <immintrin.h>

namespace kq {
using namespace fp4;

inline __m256i scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,  6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,  14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
inline float hsum(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

inline float dot_q2k(const BlockQ2K* x, const BlockQ8K* y, int nb) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * _cvtsh_ss(x[i].d);
        const float dmin = -y[i].d * _cvtsh_ss(x[i].dmin);
        const uint8_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0), h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 2; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p0 = _mm256_maddubs_epi16(_mm256_and_si256(q2bits, m3), q8_0);
            __m256i p1 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3), q8_1);
            __m256i p2 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3), q8_2);
            __m256i p3 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3), q8_3);
            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffle_q3k(3)), p3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
        }
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum(acc);
}

inline float dot_q3k(const BlockQ3K* x, const BlockQ8K* y, int nb) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const __m256i m3 = _mm256_set1_epi8(3), mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    uint32_t aux[3];
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * _cvtsh_ss(x[i].d);
        const uint8_t* q3 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        std::memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                                          ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                                          (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                                          (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0), h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {_mm256_set_m128i(l_scales, l_scales), _mm256_set_m128i(h_scales, h_scales)};
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;
        for (int j = 0; j < 2; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;
            __m256i ql[4], qh[4];
            for (int k = 0; k < 4; ++k, ++bit) {
                ql[k] = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2 * k), m3);
                qh[k] = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            }
            for (int k = 0; k < 4; ++k) {
                const __m256i q8k = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
                __m256i p16 = _mm256_sub_epi16(_mm256_maddubs_epi16(ql[k], q8k), _mm256_maddubs_epi16(qh[k], q8k));
                p16 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], scale_shuffle_q3k(k)), p16);
                sumi = _mm256_add_epi32(sumi, p16);
            }
        }
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum(acc);
}

// Placed (first-touched) K-quant matrix.
struct PMat { const uint8_t* data = nullptr; uint8_t* own = nullptr; int rows = 0, cols = 0; KType type = KType::Q2K; size_t rb = 0; };

inline PMat place(const KMat& src, int chunk) {
    PMat m;
    m.rows = src.rows; m.cols = src.cols; m.type = src.type; m.rb = row_bytes(src.type, src.cols);
    void* p = nullptr;
    if (posix_memalign(&p, 64, (src.bytes() + 63) & ~size_t(63))) abort();
    m.own = (uint8_t*)p; m.data = m.own;
#pragma omp parallel for schedule(static)
    for (int c = 0; c < src.rows / chunk; ++c) std::memcpy(m.own + size_t(c) * chunk * m.rb, src.data.data() + size_t(c) * chunk * m.rb, chunk * m.rb);
    return m;
}
inline void unplace(PMat& m) { free(m.own); m.own = nullptr; m.data = nullptr; }

inline float dot_row(const PMat& m, int r, const BlockQ8K* y) {
    const uint8_t* p = m.data + size_t(r) * m.rb;
    return m.type == KType::Q2K ? dot_q2k((const BlockQ2K*)p, y, m.cols / QKK) : dot_q3k((const BlockQ3K*)p, y, m.cols / QKK);
}

// out[k][rows] for the 8 matrices of a set; act per expert if act_per_expert.
inline void gemv_set(const PMat* mats, const BlockQ8K* act, size_t act_stride_blocks, float* out, int chunk) {
    const int rows = mats[0].rows;
#pragma omp parallel for schedule(static)
    for (int c = 0; c < rows / chunk; ++c)
        for (int k = 0; k < 8; ++k)
            for (int r = c * chunk; r < (c + 1) * chunk; ++r) out[size_t(k) * rows + r] = dot_row(mats[k], r, act + k * act_stride_blocks);
}

}  // namespace kq
