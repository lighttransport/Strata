// Clamped IQ gate/up rows on AVX512F/BW/DQ/VL CPUs, including Skylake and
// Cascade Lake. This file is built without the wider unit's VNNI/VBMI floor.
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_even_signs.hpp"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
namespace strata::kernels::cpu {
namespace {
#define STRATA_IQ512_PACK_SIGNS 1
#include "iq_avx512_rows.inl"
#undef STRATA_IQ512_PACK_SIGNS

template <int TY, int NT>
void gu_clamped_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                     float* const* ff, int r0, int r1, float limit, const uint8_t* separate_up) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*)act[t];
    const uint8_t* up = separate_up ? separate_up : blob + up_off;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t)r * gu_row, n / QK_K, y, g);
        row_dot<TY, NT>(up + (size_t)r * gu_row, n / QK_K, y, u);
        for (int t = 0; t < NT; ++t) {
            const float gate = limit > 0 ? std::fmin(g[t], limit) : g[t];
            const float value = limit > 0 ? std::fmax(-limit, std::fmin(u[t], limit)) : u[t];
            ff[t][r] = (gate / (1.f + std::exp(-gate))) * value;
        }
    }
}

template <int TY>
void gu_clamped_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                        const void* const* act, float* const* ff, int r0, int r1, float limit,
                        const uint8_t* separate_up) {
#define STRATA_CLAMPED_WIDTH(N) case N: gu_clamped_rows<TY, N>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break
    switch (nt) {
        STRATA_CLAMPED_WIDTH(1);
        STRATA_CLAMPED_WIDTH(2);
        STRATA_CLAMPED_WIDTH(3);
        STRATA_CLAMPED_WIDTH(4);
        STRATA_CLAMPED_WIDTH(5);
        STRATA_CLAMPED_WIDTH(6);
        STRATA_CLAMPED_WIDTH(7);
        STRATA_CLAMPED_WIDTH(8);
    }
#undef STRATA_CLAMPED_WIDTH
}

}  // namespace

void iq512_gu_rows_clamped(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                           const void* const* act, int nt, float* const* ff, int r0, int r1,
                           float limit, const uint8_t* separate_up) {
#define STRATA_CLAMPED_TYPE(T) case T: gu_clamped_rows_nt<T>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break
    switch (type) {
        STRATA_CLAMPED_TYPE(16);
        STRATA_CLAMPED_TYPE(17);
        STRATA_CLAMPED_TYPE(18);
        STRATA_CLAMPED_TYPE(21);
        STRATA_CLAMPED_TYPE(22);
    }
#undef STRATA_CLAMPED_TYPE
}

}  // namespace strata::kernels::cpu
