// src/kernels/cpu/nvfp4_avx2.cpp - see include/strata/kernels/cpu/nvfp4_avx2.hpp.
//
// Per 64-value block, once: its E2M1 values in element order as two 32-byte registers (ggml's reordering: sub-block
// s keeps value j in the low nibble of qs[8 s + j] and value j + 8 in the high one), their magnitudes, and the four
// sub-block scales. Per token, as ggml_vec_dot_nvfp4_q8_0's AVX2 path: vpsignb moves the sign onto the activation,
// vpmaddubsw + vpmaddwd give 8 int32 lanes (four per 16-value sub-block), one FMA per 32 values applies
// ue4m3(d[s]) * d_q8, and hsum_float_8's order sums the lanes.
//
// Needs AVX2, FMA and F16C - what cpu_avx2_ok() checks before anything here is called.
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/host/glm_env.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#if defined(STRATA_NVFP4_GU_HALF_PAIR_NT1) || defined(STRATA_NVFP4_GU_HALF_PAIR_NT2)
#include <type_traits>
#endif

namespace strata::kernels::cpu {
#if defined(STRATA_HAVE_NVFP4_CLANG_NT4)
void strata_clang_nvfp4_256_gu_rows(const uint8_t*,size_t,size_t,int,const void* const*,int,float* const*,int,int,float,float);
void strata_clang_nvfp4_256_gu_rows_clamped(const uint8_t*,size_t,size_t,int,const void* const*,int,float* const*,int,int,float,float,float,const uint8_t*);
#endif
namespace {
#if defined(STRATA_HAVE_NVFP4_CLANG_NT4)
bool clang_nt4_enabled() {
    static const bool enabled=strata_env_is_one("STRATA_NVFP4_CLANG_NT4");
    return enabled;
}
#endif

// A compile-time table, not a constructor: this file is compiled for a wide ISA, and a namespace-scope object with a
// constructor runs at program start on EVERY CPU - the compiler vectorises the loop with AVX/FMA, and a CPU without them
// (Westmere, Sandy Bridge, ...) died with an illegal instruction before main (#2324).  Powers of two: exact.
constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}
struct Ue4m3Table {   // ggml_ue4m3_to_fp32 (ggml-impl.h): halved for the doubled kvalues_fp4, 0x7F read as 0
    float v[256];
    constexpr Ue4m3Table() : v() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) { v[x] = 0.0f; continue; }
            const int e = (x >> 3) & 0xF, m = x & 0x7;
            const float raw = e == 0 ? (float) m * pow2i(-9) : (1.0f + (float) m / 8.0f) * pow2i(e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
constexpr Ue4m3Table kUe4m3;

// software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off), as the AVX-512 rows
// (read on first use, behind the CPU check: a static initialiser here would run on every CPU, see above)
inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NVFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// sub-blocks 2 h and 2 h + 1 (16 bytes of qs) as their 32 signed doubled E2M1 values, in element order
template <bool PackedK32 = false>
inline __m256i values32(const uint8_t* qs, const __m128i kv) {
    const __m128i q = _mm_loadu_si128((const __m128i*) qs);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(q, m4));
    const __m128i hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
    if constexpr (PackedK32) return _mm256_set_m128i(hi, lo);
    else return _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
}

inline float hsum8(const __m256 x) {   // ggml's hsum_float_8
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

constexpr int kMaxBlocks = 64;   // rows of up to 4096 values (n_embd 2560)
// tokens per pass: the pool's MAXT, so a window is one pass. Spilled accumulators cost less than another pass over
// the weights: one expert's gate+up, 8 tokens, 0.415 ms in passes of 4 + 4, 0.352 in one of 8 (5 6 7 alike)
constexpr int kMaxW = 8;

struct Acts {   // per token and block, the two Q8_0 blocks' scales; built once per call, reused by every row
    int nt = 0, nb = 0;
    const block_q8_0* y[kMaxW] = {};
    float dy[kMaxW * kMaxBlocks * 2];
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ib = 0; ib < 2 * nb; ++ib) dy[(size_t) t * 2 * nb + ib] = h2f(y[t][ib].d);
        }
    }
};

// Experimental Zen 1-friendly layout; AVX2/FMA/F16C dispatch requirements stay unchanged.
inline bool narrow_dot_enabled() {
    static const bool enabled = strata_env_is_one("STRATA_NVFP4_128") != 0;
    return enabled;
}

// Preserve the original low/high accumulator lane sequences and final reduction.
// A is Acts for signed dots and OffsetActs for biased dots; the latter's caller
// rejects raw -128 before selecting this specialization.
template <bool Bias, typename A>
inline void row_dot_128(const uint8_t* row, const A& a, float* res) {
    const __m128i kv = Bias ? _mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0)
                           : _mm_loadu_si128((const __m128i*) kvalues_fp4);
    const __m128i ones = _mm_set1_epi16(1), mask = _mm_set1_epi8(15);
    __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (ahead > 0) _mm_prefetch((const char*) (x + ib) + ahead, _MM_HINT_T0);
        for (int pair = 0; pair < 2; ++pair) {
            const __m128i packed = _mm_loadu_si128((const __m128i*) (x[ib].qs + pair*16));
            const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(packed, mask));
            const __m128i hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
            const __m128i v0 = _mm_unpacklo_epi64(lo, hi), v1 = _mm_unpackhi_epi64(lo, hi);
            const block_q8_0& y = a.y[0][2*ib+pair];
            const __m128i q0 = _mm_loadu_si128((const __m128i*) y.qs);
            const __m128i q1 = _mm_loadu_si128((const __m128i*) (y.qs+16));
            __m128i dot0, dot1;
            if constexpr (Bias) {
                dot0 = _mm_sub_epi16(_mm_maddubs_epi16(v0, q0), _mm256_castsi256_si128(a.correction[2*ib+pair]));
                dot1 = _mm_sub_epi16(_mm_maddubs_epi16(v1, q1), _mm256_extracti128_si256(a.correction[2*ib+pair], 1));
            } else {
                dot0 = _mm_maddubs_epi16(_mm_sign_epi8(v0, v0), _mm_sign_epi8(q0, v0));
                dot1 = _mm_maddubs_epi16(_mm_sign_epi8(v1, v1), _mm_sign_epi8(q1, v1));
            }
            dot0 = _mm_madd_epi16(dot0, ones);
            dot1 = _mm_madd_epi16(dot1, ones);
            const float dy = a.dy[2*ib+pair];
            const __m128 s0 = _mm_set1_ps(kUe4m3.v[x[ib].d[2*pair]] * dy);
            const __m128 s1 = _mm_set1_ps(kUe4m3.v[x[ib].d[2*pair+1]] * dy);
            acc0 = _mm_fmadd_ps(s0, _mm_cvtepi32_ps(dot0), acc0);
            acc1 = _mm_fmadd_ps(s1, _mm_cvtepi32_ps(dot1), acc1);
        }
    }
    __m128 sum = _mm_add_ps(acc1, acc0);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    res[0] = _mm_cvtss_f32(sum);
}

template <int NT, bool PackedK32 = false, bool Nta = false>
inline void row_dot(const uint8_t* row, const Acts& a, float* res) {
    const __m128i kv = _mm_loadu_si128((const __m128i*) kvalues_fp4);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int prefetch_ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0) _mm_prefetch((const char*) (x + ib) + prefetch_ahead, Nta ? _MM_HINT_NTA : _MM_HINT_T0);
        const __m256i v01 = values32<PackedK32>(x[ib].qs, kv), v23 = values32<PackedK32>(x[ib].qs + 16, kv);
        const __m256i a01 = _mm256_sign_epi8(v01, v01), a23 = _mm256_sign_epi8(v23, v23);
        // lanes 0-3 ue4m3(d[0]), 4-7 d[1] (and d[2], d[3]): times a broadcast d_q8, each lane is ggml's scalar
        // ue4m3(d[s]) * d_q8, rounded the same
        const __m256 u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x[ib].d[0]]));
        const __m256 u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x[ib].d[2]]));
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
            const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
            const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(q01, v01)), ones);
            const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(q23, v23)), ones);
            const float* dy = a.dy + (size_t) t * 2 * a.nb + 2 * ib;
            const __m256 s01 = _mm256_mul_ps(u01, _mm256_broadcast_ss(dy));
            const __m256 s23 = _mm256_mul_ps(u23, _mm256_broadcast_ss(dy + 1));
            acc[t] = _mm256_fmadd_ps(s01, _mm256_cvtepi32_ps(p1), acc[t]);
            acc[t] = _mm256_fmadd_ps(s23, _mm256_cvtepi32_ps(p2), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(acc[t]);
}

// Separate exact-two-token policy; does not affect slices of wider calls.
inline bool nt2_nta_enabled() {
    static const bool enabled = strata_env_is_one("STRATA_NVFP4_NT2_NTA") != 0;
    return enabled;
}
struct NtaActs : Acts { using Acts::Acts; };
template <int NT, bool PackedK32 = false>
inline void row_dot(const uint8_t* row, const NtaActs& a, float* res) {
    static_assert(NT == 2 && !PackedK32);
    row_dot<NT, false, true>(row, static_cast<const Acts&>(a), res);
}

// Experimental one-token path. Keep correction storage out of the default Acts.
inline bool offset_dot_enabled() {
    static const bool enabled = strata_env_is_one("STRATA_NVFP4_OFFSET_DOT") != 0;
    return enabled;
}
// The explicit K32 layout has its own opt-in; legacy offset selection is unchanged.
inline bool k32_offset_dot_enabled() {
    static const bool enabled = strata_env_is_one("STRATA_NVFP4_K32_OFFSET") != 0;
    return enabled;
}
struct OffsetActs : Acts {
    bool safe = true;
    __m256i correction[kMaxBlocks * 2];
    OffsetActs(const void* const* act, int nb) : Acts(act, 1, nb) {
        for (int ib = 0; ib < 2 * nb; ++ib) {
            const __m256i q = _mm256_loadu_si256((const __m256i*) y[0][ib].qs);
            correction[ib] = _mm256_maddubs_epi16(_mm256_set1_epi8(12), q);
            // vpsignb(-128, negative) wraps. Preserve that legacy raw-input behavior.
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(q, _mm256_set1_epi8(-128)))) safe = false;
        }
    }
};

template <int NT, bool PackedK32 = false, bool Nta = false>
inline void row_dot(const uint8_t* row, const OffsetActs& a, float* res) {
    static_assert(NT == 1);
    if (!a.safe) { row_dot<1, PackedK32>(row, static_cast<const Acts&>(a), res); return; }
    // Signed doubled E2M1 + 12. The largest pair sum is 2*24*128=6144,
    // and subtraction recovers the signed dot without int16 saturation.
    const __m128i kv = _mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (ahead > 0) _mm_prefetch((const char*) (x + ib) + ahead, Nta ? _MM_HINT_NTA : _MM_HINT_T0);
        const __m256i v01 = values32<PackedK32>(x[ib].qs, kv), v23 = values32<PackedK32>(x[ib].qs + 16, kv);
        const __m256 u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[1]]), _mm_set1_ps(kUe4m3.v[x[ib].d[0]]));
        const __m256 u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[x[ib].d[3]]), _mm_set1_ps(kUe4m3.v[x[ib].d[2]]));
        const block_q8_0* yb = a.y[0] + 2 * ib;
        const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
        const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
        const __m256i p1 = _mm256_madd_epi16(_mm256_sub_epi16(_mm256_maddubs_epi16(v01, q01), a.correction[2*ib]), ones);
        const __m256i p2 = _mm256_madd_epi16(_mm256_sub_epi16(_mm256_maddubs_epi16(v23, q23), a.correction[2*ib+1]), ones);
        const __m256 s01 = _mm256_mul_ps(u01, _mm256_broadcast_ss(a.dy + 2*ib));
        const __m256 s23 = _mm256_mul_ps(u23, _mm256_broadcast_ss(a.dy + 2*ib + 1));
        acc = _mm256_fmadd_ps(s01, _mm256_cvtepi32_ps(p1), acc);
        acc = _mm256_fmadd_ps(s23, _mm256_cvtepi32_ps(p2), acc);
    }
    res[0] = hsum8(acc);
}

// Independent opt-ins for measured worker GU and ordinary down-row paths.
inline bool gu_nta_enabled() {
    static const bool enabled = strata_env_is_one("STRATA_NVFP4_GU_NTA") != 0;
    return enabled;
}
inline bool down_nta_enabled() {
    static const bool enabled = strata_env_is_one("STRATA_NVFP4_DOWN_NTA") != 0;
    return enabled;
}
struct NtaOffsetActs : OffsetActs { using OffsetActs::OffsetActs; };
template <int NT, bool PackedK32 = false>
inline void row_dot(const uint8_t* row, const NtaOffsetActs& a, float* res) {
    static_assert(!PackedK32);
    row_dot<NT, false, true>(row, static_cast<const OffsetActs&>(a), res);
}

// Distinct types select the narrow loop at compile time, outside each row loop.
struct NarrowActs : Acts { using Acts::Acts; };
struct NarrowOffsetActs : OffsetActs { using OffsetActs::OffsetActs; };
template <int NT>
inline void row_dot(const uint8_t* row, const NarrowActs& a, float* res) {
    static_assert(NT == 1);
    row_dot_128<false>(row, a, res);
}
template <int NT>
inline void row_dot(const uint8_t* row, const NarrowOffsetActs& a, float* res) {
    static_assert(NT == 1);
    if (a.safe) row_dot_128<true>(row, a, res);
    else row_dot_128<false>(row, a, res);
}

#if defined(STRATA_NVFP4_GU_HALF_PAIR_NT1)
// Reuse one activation half between gate and up without changing either sum's
// FMA order. Packed K32, narrow/offset activations and multi-token rows retain
// their existing implementations.
inline void row_gu_half_pair(const uint8_t* gate, const uint8_t* up,
                             const Acts& a, float* g, float* u) {
    const auto* xg = reinterpret_cast<const block_nvfp4*>(gate);
    const auto* xu = reinterpret_cast<const block_nvfp4*>(up);
    const auto kv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_fp4));
    const auto ones = _mm256_set1_epi16(1);
    __m256 ag = _mm256_setzero_ps(), au = _mm256_setzero_ps();
    const int ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (ahead > 0) {
            _mm_prefetch(reinterpret_cast<const char*>(xg + ib) + ahead, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(xu + ib) + ahead, _MM_HINT_T0);
        }
        for (int half = 0; half < 2; ++half) {
            const auto q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.y[0][2 * ib + half].qs));
            const auto dy = _mm256_broadcast_ss(a.dy + 2 * ib + half);
            auto step = [&](const block_nvfp4& x, __m256 acc) {
                const auto v = values32<false>(x.qs + 16 * half, kv);
                const auto p = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_sign_epi8(v, v), _mm256_sign_epi8(q, v)), ones);
                const auto scale = _mm256_set_m128(
                    _mm_set1_ps(kUe4m3.v[x.d[2 * half + 1]]), _mm_set1_ps(kUe4m3.v[x.d[2 * half]]));
                return _mm256_fmadd_ps(_mm256_mul_ps(scale, dy), _mm256_cvtepi32_ps(p), acc);
            };
            ag = step(xg[ib], ag);
            au = step(xu[ib], au);
        }
    }
    g[0] = hsum8(ag);
    u[0] = hsum8(au);
}
#endif

#if defined(STRATA_NVFP4_GU_HALF_PAIR_NT2)
// Share unpacked gate/up halves and activation loads across two tokens.
// Each output retains its original FMA order; other widths/layouts are unchanged.
template<bool Nta = false>
inline void row_gu_half_pair2(const uint8_t* gate, const uint8_t* up,
                             const Acts& a, float* g, float* u) {
    const auto* xg = reinterpret_cast<const block_nvfp4*>(gate);
    const auto* xu = reinterpret_cast<const block_nvfp4*>(up);
    const auto kv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_fp4));
    const auto ones = _mm256_set1_epi16(1);
    __m256 ag[2] = {_mm256_setzero_ps(),_mm256_setzero_ps()};
    __m256 au[2] = {_mm256_setzero_ps(),_mm256_setzero_ps()};
    const int ahead = prefetch_distance();
    for (int ib=0;ib<a.nb;++ib) {
        if(ahead>0) {
            _mm_prefetch(reinterpret_cast<const char*>(xg+ib)+ahead,Nta ? _MM_HINT_NTA : _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(xu+ib)+ahead,Nta ? _MM_HINT_NTA : _MM_HINT_T0);
        }
        for(int half=0;half<2;++half) {
            const auto vg=values32<false>(xg[ib].qs+16*half,kv);
            const auto vu=values32<false>(xu[ib].qs+16*half,kv);
            const auto mg=_mm256_sign_epi8(vg,vg),mu=_mm256_sign_epi8(vu,vu);
            const auto sg=_mm256_set_m128(_mm_set1_ps(kUe4m3.v[xg[ib].d[2*half+1]]),_mm_set1_ps(kUe4m3.v[xg[ib].d[2*half]]));
            const auto su=_mm256_set_m128(_mm_set1_ps(kUe4m3.v[xu[ib].d[2*half+1]]),_mm_set1_ps(kUe4m3.v[xu[ib].d[2*half]]));
            for(int t=0;t<2;++t) {
                const auto q=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.y[t][2*ib+half].qs));
                const auto dy=_mm256_broadcast_ss(a.dy+t*2*a.nb+2*ib+half);
                const auto pg=_mm256_madd_epi16(_mm256_maddubs_epi16(mg,_mm256_sign_epi8(q,vg)),ones);
                const auto pu=_mm256_madd_epi16(_mm256_maddubs_epi16(mu,_mm256_sign_epi8(q,vu)),ones);
                ag[t]=_mm256_fmadd_ps(_mm256_mul_ps(sg,dy),_mm256_cvtepi32_ps(pg),ag[t]);
                au[t]=_mm256_fmadd_ps(_mm256_mul_ps(su,dy),_mm256_cvtepi32_ps(pu),au[t]);
            }
        }
    }
    for(int t=0;t<2;++t){g[t]=hsum8(ag[t]);u[t]=hsum8(au[t]);}
}

#endif

template <int NT, bool Clamp = false, bool PackedK32 = false, typename A>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const A& a, float* const* ff, int r0, int r1,
             float sg, float su, float limit = 0, const uint8_t* separate_up = nullptr) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
#if defined(STRATA_NVFP4_GU_HALF_PAIR_NT1)
        if constexpr (NT == 1 && !PackedK32 && std::is_same_v<A, Acts>) {
            row_gu_half_pair(blob + size_t(r) * gu_row,
                             (separate_up ? separate_up : blob + up_off) + size_t(r) * gu_row, a, g, u);
        } else
#endif
#if defined(STRATA_NVFP4_GU_HALF_PAIR_NT2)
        if constexpr (NT == 2 && !PackedK32 && (std::is_same_v<A, Acts> || std::is_same_v<A, NtaActs>)) {
            row_gu_half_pair2<std::is_same_v<A, NtaActs>>(blob + size_t(r) * gu_row,
                             (separate_up ? separate_up : blob + up_off) + size_t(r) * gu_row, a, g, u);
        } else
#endif
        if constexpr (PackedK32) {
            row_dot<NT, true>(blob + size_t(r) * gu_row, a, g);
            row_dot<NT, true>((separate_up ? separate_up : blob + up_off) + size_t(r) * gu_row, a, u);
        } else {
            row_dot<NT>(blob + (size_t) r * gu_row, a, g);
            row_dot<NT>((separate_up ? separate_up : blob + up_off) + (size_t) r * gu_row, a, u);
        }
        for (int t = 0; t < NT; ++t) {   // native_gu_rows' ggml path, operation for operation
            float gs = g[t] * sg, us = u[t] * su;
            if constexpr (Clamp) {
                gs = std::fmin(gs, limit);
                us = std::fmax(-limit, std::fmin(us, limit));
            }
            ff[t][r] = (gs / (1.f + std::exp(-gs))) * us;
        }
    }
}

template <int NT, typename A>
void dot_rows(const uint8_t* w, size_t row_bytes, const A& a, float* const* out, int r0, int r1, float scale) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(w + (size_t) r * row_bytes, a, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t] * scale;
    }
}

template <int NT, typename A>
void k32_dot_rows(const uint8_t* w, size_t row_bytes, const A& a, float* const* out, int r0, int r1, float scale) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT, true>(w + size_t(r) * row_bytes, a, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t] * scale;
    }
}

// Keep the additional narrow activation storage/code out of the default entry
// points. One call per row range; this does not add a call per individual row.
#if defined(_MSC_VER)
#define STRATA_NVFP4_NOINLINE __declspec(noinline)
#else
#define STRATA_NVFP4_NOINLINE __attribute__((noinline))
#endif
STRATA_NVFP4_NOINLINE void narrow_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                                         const void* const* act, float* const* ff, int r0, int r1,
                                         float sg, float su, float limit, const uint8_t* separate_up) {
    auto run = [&](const auto& a) {
        if (limit > 0) gu_rows<1, true>(blob, gu_row, up_off, a, ff, r0, r1, sg, su, limit, separate_up);
        else gu_rows<1>(blob, gu_row, up_off, a, ff, r0, r1, sg, su, 0, separate_up);
    };
    if (offset_dot_enabled()) { const NarrowOffsetActs a(act, n / QK_NVFP4); run(a); }
    else { const NarrowActs a(act, 1, n / QK_NVFP4); run(a); }
}
STRATA_NVFP4_NOINLINE void narrow_dot_rows(const uint8_t* w, size_t row_bytes, int n,
                                          const void* const* act, float* const* out, int r0, int r1, float scale) {
    if (offset_dot_enabled()) {
        const NarrowOffsetActs a(act, n / QK_NVFP4);
        dot_rows<1>(w, row_bytes, a, out, r0, r1, scale);
    } else {
        const NarrowActs a(act, 1, n / QK_NVFP4);
        dot_rows<1>(w, row_bytes, a, out, r0, r1, scale);
    }
}
#undef STRATA_NVFP4_NOINLINE

// windows wider than kMaxW (not from the pool) run in balanced slices of at most kMaxW
template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + kMaxW - 1) / kMaxW;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

bool validate_k32_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                       float* const* out, int r0, int r1) {
    if (n <= 0 || !nvfp4_256_fits(n) || nt < 1 || nt > 32 || r0 < 0 || r1 < r0 ||
        row_bytes < size_t(n / QK_NVFP4) * sizeof(block_nvfp4) || size_t(r1) > SIZE_MAX / row_bytes)
        throw std::invalid_argument("native K32 rows: invalid geometry");
    if (r0 == r1) return false;
    if (!w || !act || !out) throw std::invalid_argument("native K32 rows: null buffer");
    for (int t = 0; t < nt; ++t)
        if (!act[t] || !out[t]) throw std::invalid_argument("native K32 rows: null token buffer");
    return true;
}

}  // namespace

bool nvfp4_256_fits(int n) { return n % QK_NVFP4 == 0 && n / QK_NVFP4 <= kMaxBlocks; }

void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up) {
    if(nt==2 && nt2_nta_enabled()) {
        const NtaActs a(act, 2, n / QK_NVFP4);
        gu_rows<2>(blob,gu_row,up_off,a,ff,r0,r1,s_gate,s_up);
        return;
    }
#if defined(STRATA_HAVE_NVFP4_CLANG_NT4)
    if(nt==4 && clang_nt4_enabled()) {
        strata_clang_nvfp4_256_gu_rows(blob,gu_row,up_off,n,act,nt,ff,r0,r1,s_gate,s_up);
        return;
    }
#endif
    if (nt == 1 && narrow_dot_enabled()) {
        narrow_gu_rows(blob, gu_row, up_off, n, act, ff, r0, r1, s_gate, s_up, 0, nullptr);
        return;
    }
    if (nt == 1 && offset_dot_enabled()) {
        if (gu_nta_enabled()) {
            const NtaOffsetActs a(act, n / QK_NVFP4);
            gu_rows<1>(blob, gu_row, up_off, a, ff, r0, r1, s_gate, s_up);
        } else {
            const OffsetActs a(act, n / QK_NVFP4);
            gu_rows<1>(blob, gu_row, up_off, a, ff, r0, r1, s_gate, s_up);
        }
        return;
    }
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: gu_rows<1>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 7: gu_rows<7>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            default: gu_rows<8>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
        }
    });
}

void nvfp4_256_gu_rows_clamped(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                              const void* const* act, int nt, float* const* ff,
                              int r0, int r1, float s_gate, float s_up, float limit, const uint8_t* separate_up) {
    if(nt==2 && nt2_nta_enabled()) {
        const NtaActs a(act, 2, n / QK_NVFP4);
        if(limit>0) gu_rows<2,true>(blob,gu_row,up_off,a,ff,r0,r1,s_gate,s_up,limit,separate_up);
        else gu_rows<2>(blob,gu_row,up_off,a,ff,r0,r1,s_gate,s_up,0,separate_up);
        return;
    }
#if defined(STRATA_HAVE_NVFP4_CLANG_NT4)
    if(nt==4 && clang_nt4_enabled()) {
        strata_clang_nvfp4_256_gu_rows_clamped(blob,gu_row,up_off,n,act,nt,ff,r0,r1,s_gate,s_up,limit,separate_up);
        return;
    }
#endif
    if (nt == 1 && narrow_dot_enabled()) {
        narrow_gu_rows(blob, gu_row, up_off, n, act, ff, r0, r1, s_gate, s_up, limit, separate_up);
        return;
    }
    if (nt == 1 && offset_dot_enabled()) {
        auto run = [&](const auto& a) {
            if (limit > 0) gu_rows<1, true>(blob, gu_row, up_off, a, ff, r0, r1, s_gate, s_up, limit, separate_up);
            else gu_rows<1>(blob, gu_row, up_off, a, ff, r0, r1, s_gate, s_up, 0, separate_up);
        };
        if (gu_nta_enabled()) { const NtaOffsetActs a(act, n / QK_NVFP4); run(a); }
        else { const OffsetActs a(act, n / QK_NVFP4); run(a); }
        return;
    }
    if (!(limit > 0) && !separate_up) {
        nvfp4_256_gu_rows(blob, gu_row, up_off, n, act, nt, ff, r0, r1, s_gate, s_up);
        return;
    }
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
#define CLAMP_CASE(N) case N: { \
    if(limit>0) { gu_rows<N, true>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, limit, separate_up); } \
    else { gu_rows<N>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up, 0, separate_up); } \
    break; }
        switch (w) { CLAMP_CASE(1); CLAMP_CASE(2); CLAMP_CASE(3); CLAMP_CASE(4);
                     CLAMP_CASE(5); CLAMP_CASE(6); CLAMP_CASE(7); CLAMP_CASE(8); }
#undef CLAMP_CASE
    });
}

void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale) {
    if(nt==2 && nt2_nta_enabled()) {
        const NtaActs a(act, 2, n / QK_NVFP4);
        dot_rows<2>(w,row_bytes,a,out,r0,r1,scale);
        return;
    }
    if (nt == 1 && narrow_dot_enabled()) {
        narrow_dot_rows(w, row_bytes, n, act, out, r0, r1, scale);
        return;
    }
    if (nt == 1 && offset_dot_enabled()) {
        if (down_nta_enabled()) {
            const NtaOffsetActs a(act, n / QK_NVFP4);
            dot_rows<1>(w, row_bytes, a, out, r0, r1, scale);
        } else {
            const OffsetActs a(act, n / QK_NVFP4);
            dot_rows<1>(w, row_bytes, a, out, r0, r1, scale);
        }
        return;
    }
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
        switch (k) {
            case 1: dot_rows<1>(w, row_bytes, a, o, r0, r1, scale); break;
            case 2: dot_rows<2>(w, row_bytes, a, o, r0, r1, scale); break;
            case 3: dot_rows<3>(w, row_bytes, a, o, r0, r1, scale); break;
            case 4: dot_rows<4>(w, row_bytes, a, o, r0, r1, scale); break;
            case 5: dot_rows<5>(w, row_bytes, a, o, r0, r1, scale); break;
            case 6: dot_rows<6>(w, row_bytes, a, o, r0, r1, scale); break;
            case 7: dot_rows<7>(w, row_bytes, a, o, r0, r1, scale); break;
            default: dot_rows<8>(w, row_bytes, a, o, r0, r1, scale); break;
        }
    });
}

void nvfp4_256_k32_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                         float* const* out, int r0, int r1, float scale) {
    if (!validate_k32_rows(w, row_bytes, n, act, nt, out, r0, r1)) return;
    // Explicit K32 offset specialization; never decode these bytes as old layout.
    if (nt == 1 && k32_offset_dot_enabled()) {
        const OffsetActs a(act, n / QK_NVFP4);
        k32_dot_rows<1>(w, row_bytes, a, out, r0, r1, scale);
        return;
    }
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
        switch (k) {
            case 1: k32_dot_rows<1>(w, row_bytes, a, o, r0, r1, scale); break;
            case 2: k32_dot_rows<2>(w, row_bytes, a, o, r0, r1, scale); break;
            case 3: k32_dot_rows<3>(w, row_bytes, a, o, r0, r1, scale); break;
            case 4: k32_dot_rows<4>(w, row_bytes, a, o, r0, r1, scale); break;
            case 5: k32_dot_rows<5>(w, row_bytes, a, o, r0, r1, scale); break;
            case 6: k32_dot_rows<6>(w, row_bytes, a, o, r0, r1, scale); break;
            case 7: k32_dot_rows<7>(w, row_bytes, a, o, r0, r1, scale); break;
            default: k32_dot_rows<8>(w, row_bytes, a, o, r0, r1, scale); break;
        }
    });
}

void nvfp4_256_k32_gu_rows_clamped(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                                  const void* const* act, int nt, float* const* ff, int r0, int r1,
                                  float s_gate, float s_up, float limit, const uint8_t* separate_up) {
    if (!validate_k32_rows(blob, gu_row, n, act, nt, ff, r0, r1)) return;
    if (!separate_up && up_off > SIZE_MAX - size_t(r1) * gu_row)
        throw std::invalid_argument("native K32 gate/up: offset overflow");
    if (nt == 1 && k32_offset_dot_enabled()) {
        const OffsetActs a(act, n / QK_NVFP4);
        if (limit > 0) gu_rows<1, true, true>(blob, gu_row, up_off, a, ff, r0, r1, s_gate, s_up, limit, separate_up);
        else gu_rows<1, false, true>(blob, gu_row, up_off, a, ff, r0, r1, s_gate, s_up, 0, separate_up);
        return;
    }
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = ff + t0;
        auto run = [&]<int N>() {
            if (limit > 0) gu_rows<N, true, true>(blob, gu_row, up_off, a, o, r0, r1, s_gate, s_up, limit, separate_up);
            else gu_rows<N, false, true>(blob, gu_row, up_off, a, o, r0, r1, s_gate, s_up, 0, separate_up);
        };
        switch (k) {
            case 1: run.template operator()<1>(); break;
            case 2: run.template operator()<2>(); break;
            case 3: run.template operator()<3>(); break;
            case 4: run.template operator()<4>(); break;
            case 5: run.template operator()<5>(); break;
            case 6: run.template operator()<6>(); break;
            case 7: run.template operator()<7>(); break;
            default: run.template operator()<8>(); break;
        }
    });
}

}  // namespace strata::kernels::cpu
