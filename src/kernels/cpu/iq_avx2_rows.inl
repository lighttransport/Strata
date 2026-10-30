// src/kernels/cpu/iq_avx2_rows.inl - the row kernels of iq_avx2.cpp.  Not a header: iq_avx2.cpp includes it twice,
//
//   STRATA_ROWS_VNNI 0   the AVX2 forms (namespace plain);
//   STRATA_ROWS_VNNI 1   the same loops with AVX-VNNI (namespace vnni, built with STRATA_AVXVNNI): vpdpwssd for each
//                        madd + add, vpdpbusd for maddubs + madd(ones).  Integer sums, so the same bits.
//
// Every function here is STRATA_ROWS_FN: nothing in the plain copy, GCC/Clang's target("avxvnni") in the VNNI copy, so
// the AVX-VNNI intrinsics compile in these functions only and the plain copy stays AVX2 (-mavxvnni on the whole file
// would let the compiler use the extension anywhere).  MSVC takes the intrinsics under /arch:AVX2.  The VNNI copy is
// called only where cpu_avxvnni_ok().

// acc + madd(p16, sc).  vpdpwssd is the madd and the add in one uop; both wrap the same way.
STRATA_ROWS_FN inline __m256i madd_add(__m256i acc, __m256i p16, __m256i sc) {
#if STRATA_ROWS_VNNI
    return _mm256_dpwssd_avx_epi32(acc, p16, sc);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p16, sc));
#endif
}

// Four u8 x s8 products per int32 lane.  vpdpbusd does not saturate where maddubs would, but maddubs cannot here: the
// u8 side is an IQ4_NL |w| <= 127, so a pair is at most 2 * 127 * 128.
STRATA_ROWS_FN inline __m256i dot4u(__m256i u8, __m256i s8) {
#if STRATA_ROWS_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u8, s8);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1));
#endif
}

template <int TY, int NT> STRATA_ROWS_FN
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const int pf = prefetch_distance();
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt32<TY>::bytes;
        rows_ahead(blk, pf);
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                Fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        const float dx = h2f(u16(blk)) * Fmt32<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// ---- IQ2_XS (17) with ggml's vectorized sign decode (arch/x86/quants.c, ggml_vec_dot_iq2_xs_q8_K):
// the 7-bit sign indices never touch ksigns_iq2xs - the 8th sign bit is reconstructed by parity (two
// shifts, a xor and ggml's bit_helper pshufb) and the per-half sign vectors come out of shuffles of one
// 32-byte load.  The 8 scale bytes become all 16 half-scales with ggml's unpack trick.  The grid lookups
// stay scalar loads into set_epi64x (ggml does the same).  Every token still only pays a load, a sign,
// a maddubs, a madd and an add per half, into alternating accumulators.
template <int NT> STRATA_ROWS_FN
inline void row_dot_iq2xs(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    static const uint8_t bit_sel[32] = {
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80,
        1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80, 1, 2, 4, 8, 16, 32, 64, (uint8_t) 0x80 };
    static const char shuf_even[32] = {   // sign bits of u16 0,1,2,3 (even bytes of the sign register)
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6 };
    static const char shuf_odd[32] = {    // sign bits of u16 4,5,6,7
        8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10,
        12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14 };
    static const uint8_t bit_help[32] = {  // 0x80 when the 4-bit index has odd popcount (parity bit 7)
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00,
        0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80,
        (uint8_t) 0x80, 0x00, 0x00, (uint8_t) 0x80, 0x00, (uint8_t) 0x80, (uint8_t) 0x80, 0x00 };
    static const uint8_t k_sc_shuffle[128] = {  // half h -> scale bytes 2h (lanes 0-7), 2h+1 (lanes 8-15)
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15 };
    const __m256i bsel = _mm256_loadu_si256((const __m256i*) bit_sel);
    const __m256i shev = _mm256_loadu_si256((const __m256i*) shuf_even);
    const __m256i shod = _mm256_loadu_si256((const __m256i*) shuf_odd);
    const __m256i bhelp = _mm256_loadu_si256((const __m256i*) bit_help);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
    const __m256i one8 = _mm256_set1_epi8(1);
    const int pf = prefetch_distance();

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 74;
        rows_ahead(blk, pf);
        // the 8 scale bytes -> 16 half-scales of 2*s+1, interleaved [a0, b0, a1, b1, ...] (ggml's unpack)
        __m128i st = _mm_set1_epi64x((long long) u64(blk + 66));
        st = _mm_unpacklo_epi8(_mm_and_si128(st, m4), _mm_and_si128(_mm_srli_epi16(st, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(st, 1), m1);
        __m256i acc[NT][2];
        for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_si256();
        for (int jj = 0; jj < 2; ++jj) {   // 128 values (16 u16) per step
            const __m256i q2 = _mm256_loadu_si256((const __m256i*) (blk + 2 + 32 * jj));
            const __m256i p7 = _mm256_srli_epi16(q2, 9);              // 7 sign bits per u16, even bytes
            const __m256i p3 = _mm256_srli_epi16(q2, 13);             // sign index bits 4-6
            const __m256i fsb = _mm256_or_si256(p7,
                _mm256_shuffle_epi8(bhelp, _mm256_xor_si256(p7, p3)));  // + the parity sign bit 7
            const __m256i s01 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(fsb)); // u16 0-7
            const __m256i s23 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(fsb, 1)); // u16 8-15
            alignas(32) uint16_t gi[16];
            _mm256_store_si256((__m256i*) gi, _mm256_and_si256(q2, m511));
            for (int h = 0; h < 4; ++h) {   // the four 32-value halves of this group
                const int H = 4 * jj + h;   // global half index -> scales byte pair
                const __m256i g = _mm256_set_epi64x((long long) iq2xs_grid[gi[4 * h + 3]],
                                                   (long long) iq2xs_grid[gi[4 * h + 2]],
                                                   (long long) iq2xs_grid[gi[4 * h + 1]],
                                                   (long long) iq2xs_grid[gi[4 * h]]);
                const __m256i sb = _mm256_shuffle_epi8(h < 2 ? s01 : s23, (h & 1) ? shod : shev);
                const __m256i sgn = _mm256_or_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(sb, bsel), bsel), one8);
                const __m256i sc = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i*) k_sc_shuffle + H)));
                const int off = 128 * jj + 32 * h;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acc[t][h & 1] = madd_add(acc[t][h & 1], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        const float dx = h2f(u16(blk)) * 0.125f;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(acc[t][0], acc[t][1])), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// ---- IQ1_S (19), STRATA_IQ1S_FAST=1 (opt-in): the grid values g in {-1, 0, 1} multiply the activation by a sign
// (vpsignb gives g * y, zero where g is), one maddubs against ones and one madd by 8 (2s+1); the delta term
// +-(2s+1) * sum(y) of a 32-value half comes from the activation's bsums as a scalar.  The block's integer total is
// Fmt32<19>'s; it reaches the float accumulator through other lanes, so the float additions round differently.
template <int NT> STRATA_ROWS_FN
inline void row_dot_iq1s(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const int pf = prefetch_distance();
    const __m256i ones = _mm256_set1_epi8(1);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 50;
        rows_ahead(blk, pf);
        __m256i acci[NT];
        int dl[NT];
        for (int t = 0; t < NT; ++t) { acci[t] = _mm256_setzero_si256(); dl[t] = 0; }
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* q = blk + 2 + 4 * ib;
            const uint32_t h = u16(blk + 34 + 2 * ib);
            const __m256i v = _mm256_set_epi64x((long long) iq1s_grid[q[3] | ((h >> 1) & 0x700)],
                                                (long long) iq1s_grid[q[2] | ((h << 2) & 0x700)],
                                                (long long) iq1s_grid[q[1] | ((h << 5) & 0x700)],
                                                (long long) iq1s_grid[q[0] | ((h << 8) & 0x700)]);
            const int sc = 2 * (int) ((h >> 12) & 7) + 1;
            const __m256i sc8 = _mm256_set1_epi16((short) (8 * sc));
            const int ds = (h & 0x8000) ? -sc : sc;
            for (int t = 0; t < NT; ++t) {
                const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 32 * ib));
                acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(ones, _mm256_sign_epi8(yv, v)), sc8);
                dl[t] += ds * (y[t][i].bsums[2 * ib] + y[t][i].bsums[2 * ib + 1]);
            }
        }
        const float dx = h2f(u16(blk)) * 0.125f;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(acci[t], _mm256_setr_epi32(dl[t], 0, 0, 0, 0, 0, 0, 0))), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}


// STRATA_IQ1S_VIDX=1: IQ1_S with the block's 32 grid indices built in two registers (the qs bytes widened to 16 bits, or-ed with the
// qh fields spread by mullo/mulhi shifts: k = 0..2 shift left by 8, 5, 2, k = 3 right by 1) and stored once; a half then reads its
// four indices as 16-bit loads from the iq1s_grid8 table. The delta sign and the scale come from 16-entry tables on the top nibble of qh.
// Fmt32<119>'s values, accumulated in its order: bit-identical results (iq1s_w1a8_bench checks it).
template <int NT> STRATA_ROWS_FN
inline void row_dot_iq1s_vidx(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    const int pf = prefetch_distance();
    const __m256i c0 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7);
    const __m256i c1 = _mm256_setr_epi8(8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15);
    const __m256i mul = _mm256_setr_epi16(256, 32, 4, 0, 256, 32, 4, 0, 256, 32, 4, 0, 256, 32, 4, 0);
    const __m256i mulh = _mm256_setr_epi16(0, 0, 0, 32768, 0, 0, 0, 32768, 0, 0, 0, 32768, 0, 0, 0, 32768);
    const __m256i m700 = _mm256_set1_epi16(0x700);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * 50;
        rows_ahead(blk, pf);
        alignas(32) uint16_t idx[32];
        {
            const __m256i qs = _mm256_loadu_si256((const __m256i*) (blk + 2));
            const __m256i hb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) (blk + 34)));
            const __m256i h0 = _mm256_shuffle_epi8(hb, c0), h1 = _mm256_shuffle_epi8(hb, c1);
            const __m256i hi0 = _mm256_and_si256(_mm256_or_si256(_mm256_mullo_epi16(h0, mul), _mm256_mulhi_epu16(h0, mulh)), m700);
            const __m256i hi1 = _mm256_and_si256(_mm256_or_si256(_mm256_mullo_epi16(h1, mul), _mm256_mulhi_epu16(h1, mulh)), m700);
            _mm256_store_si256((__m256i*) idx, _mm256_or_si256(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(qs)), hi0));
            _mm256_store_si256((__m256i*) (idx + 16), _mm256_or_si256(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(qs, 1)), hi1));
        }
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {
            const uint16_t* ix = idx + 4 * ib;
            const __m128i lo = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64((const __m128i*) &iq1s_grid8[ix[0]])), (const double*) &iq1s_grid8[ix[1]]));
            const __m128i hi = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64((const __m128i*) &iq1s_grid8[ix[2]])), (const double*) &iq1s_grid8[ix[3]]));
            const __m256i v = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
            const int key = u16(blk + 34 + 2 * ib) >> 12;
            const __m256i w = _mm256_add_epi8(v, _mm256_load_si256((const __m256i*) iq1s_delta_tab[key]));
            const __m256i g = _mm256_abs_epi8(w);
            const __m256i sc = _mm256_load_si256((const __m256i*) iq1s_scale_tab[key]);
            for (int t = 0; t < NT; ++t) {
                const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 32 * ib));
                acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, _mm256_sign_epi8(yv, w)), sc);
            }
        }
        const float dx = h2f(u16(blk)) * 0.125f;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

template <int TY, int NT> STRATA_ROWS_FN
inline void row_dot_any(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    if constexpr (TY == 17) row_dot_iq2xs<NT>(row, nblocks, y, res);
    else if constexpr (TY == 19) { if (iq1s_fast()) row_dot_iq1s<NT>(row, nblocks, y, res); else row_dot<TY, NT>(row, nblocks, y, res); }
    else if constexpr (TY == 219) row_dot_iq1s_vidx<NT>(row, nblocks, y, res);
    else                    row_dot<TY, NT>(row, nblocks, y, res);
}

// IQ1_S gate/up share activations; each accumulator keeps the original
// block and lane order so the resulting dot products can remain bit-identical.
template <int TY, int NT> STRATA_ROWS_FN
inline void row_dot_iq1s_pair(const uint8_t* gate, const uint8_t* up, int nblocks,
                              const block_q8_K* const* y, float* rg, float* ru) {
    const int pf = prefetch_distance();
    __m256 ag[NT], au[NT];
    for (int t = 0; t < NT; ++t) ag[t] = au[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* bg = gate + size_t(i) * 50;
        const uint8_t* bu = up + size_t(i) * 50;
        rows_ahead(bg, pf); rows_ahead(bu, pf);
        __m256i ig[NT], iu[NT];
        for (int t = 0; t < NT; ++t) ig[t] = iu[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) for (int half = 0; half < 2; ++half) {
            __m256i gg, gs, gc, ug, us, uc;
            Fmt32<TY>::decode(bg, j, half, gg, gs, gc);   // TY 19: iq1s_grid; 119: the pre-multiplied iq1s_grid8 (STRATA_IQ1S_GRID8)
            Fmt32<TY>::decode(bu, j, half, ug, us, uc);
            for (int t = 0; t < NT; ++t) {
                const __m256i v = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64*j + 32*half));
                ig[t] = madd_add(ig[t], _mm256_maddubs_epi16(gg, _mm256_sign_epi8(v, gs)), gc);
                iu[t] = madd_add(iu[t], _mm256_maddubs_epi16(ug, _mm256_sign_epi8(v, us)), uc);
            }
        }
        const float dg = h2f(u16(bg)) * 0.125f, du = h2f(u16(bu)) * 0.125f;
        for (int t = 0; t < NT; ++t) {
            ag[t] = _mm256_fmadd_ps(_mm256_set1_ps(dg * y[t][i].d), _mm256_cvtepi32_ps(ig[t]), ag[t]);
            au[t] = _mm256_fmadd_ps(_mm256_set1_ps(du * y[t][i].d), _mm256_cvtepi32_ps(iu[t]), au[t]);
        }
    }
    for (int t = 0; t < NT; ++t) { rg[t] = hsum8(ag[t]); ru[t] = hsum8(au[t]); }
}

template <int TY, int NT> STRATA_ROWS_FN
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1, float limit, const uint8_t* separate_up) {
    // limit > 0: GLM's SwiGLU clamp (gate capped above, up clamped both ways) before SiLU. separate_up: the up rows
    // from their own tensor (a GGUF with separate gate and up expert tensors) instead of blob + up_off.
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    const uint8_t* up = separate_up ? separate_up : blob + up_off;
    float g[NT], u[NT];
    // The paired loop wins for one token on the measured Zen 1 core, but
    // slows wider groups. Keep their original dispatch.
    const bool pair_iq1s = (TY == 19 || TY == 119) && NT == 1 && iq1s_pair() && !iq1s_fast();
    for (int r = r0; r < r1; ++r) {
        if constexpr (TY == 19 || TY == 119) {
            if (pair_iq1s) row_dot_iq1s_pair<TY, NT>(blob + size_t(r) * gu_row, up + size_t(r) * gu_row, nb, y, g, u);
            else { row_dot_any<TY, NT>(blob + size_t(r) * gu_row, nb, y, g); row_dot_any<TY, NT>(up + size_t(r) * gu_row, nb, y, u); }
        } else {
            row_dot_any<TY, NT>(blob + size_t(r) * gu_row, nb, y, g);
            row_dot_any<TY, NT>(up + size_t(r) * gu_row, nb, y, u);
        }
        for (int t = 0; t < NT; ++t) {
            if (limit > 0) { g[t] = std::fmin(g[t], limit); u[t] = std::fmax(-limit, std::fmin(u[t], limit)); }
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

template <int TY, int NT> STRATA_ROWS_FN
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY> STRATA_ROWS_FN
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1, float limit, const uint8_t* separate_up) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        default: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
    }
}

template <int TY> STRATA_ROWS_FN
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// The decode of each format for a variant: the gathered one (kIq256Gather) for IQ3_XXS, IQ3_S and IQ2_S.
template <bool G> STRATA_ROWS_FN
void gu_type(int type, int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
             float* const* ff, int r0, int r1, float limit, const uint8_t* separate_up) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 19:
            if (iq1s_vidx_on()) { prepare_iq1s_grid8(); gu_rows_nt<219>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); }
            else if (iq1s_grid8_on()) { prepare_iq1s_grid8(); gu_rows_nt<119>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); }
            else gu_rows_nt<19>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up);
            break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 18: gu_rows_nt<G ? 118 : 18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 21: gu_rows_nt<G ? 121 : 21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 22: gu_rows_nt<G ? 122 : 22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        case 23: gu_rows_nt<23>(nt, blob, gu_row, up_off, n, act, ff, r0, r1, limit, separate_up); break;
        default: break;
    }
}

template <bool G> STRATA_ROWS_FN
void dot_type(int type, int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out,
              int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 19:
            if (iq1s_vidx_on()) { prepare_iq1s_grid8(); dot_rows_nt<219>(nt, w, row_bytes, n, act, out, r0, r1); }
            else if (iq1s_grid8_on()) { prepare_iq1s_grid8(); dot_rows_nt<119>(nt, w, row_bytes, n, act, out, r0, r1); }
            else dot_rows_nt<19>(nt, w, row_bytes, n, act, out, r0, r1);
            break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<G ? 118 : 18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<G ? 121 : 21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<G ? 122 : 22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 23: dot_rows_nt<23>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

// ---- IQ4_NL (type 20): 32-value blocks of 18 bytes (f16 d + 16 nibble bytes) against Q8_0 activations.
// ggml-cpu's own AVX-2 dot (ggml_vec_dot_iq4_nl_q8_0) is single-token: every token redoes the nibble ->
// kvalues_iq4nl pshufb decode and the |w| half of the signed x signed product.  Here both are computed once
// per block; every token then costs a sign, a maddubs, a madd and an fmadd.  The arithmetic is ggml's - only
// the order of the float additions differs.
template <int NT> STRATA_ROWS_FN
void iq4nl_rows(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, float* const* out,
                int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const int nb = n / QK4_NL;
    const int pf = prefetch_distance();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            rows_ahead(blk, pf);
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);                      // values 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);    // values 16..31
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);   // |w|: the unsigned operand of maddubs
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) b.qs);
                const __m256i p = dot4u(aq, _mm256_sign_epi8(q8, q4));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8(accf[t]);
    }
}

STRATA_ROWS_FN void iq4nl_nt(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y, int nt,
                             float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: iq4nl_rows<1>(w, row_bytes, n, y, out, r0, r1); break;
        case 2: iq4nl_rows<2>(w, row_bytes, n, y, out, r0, r1); break;
        case 3: iq4nl_rows<3>(w, row_bytes, n, y, out, r0, r1); break;
        case 4: iq4nl_rows<4>(w, row_bytes, n, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl_nt(w, row_bytes, n, y + t0, k, out + t0, r0, r1);
        }
    }
}

#undef STRATA_ROWS_VNNI
#undef STRATA_ROWS_FN
