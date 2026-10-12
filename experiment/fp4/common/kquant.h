// ggml K-quant blocks used by Strata expert packs: Q2_K (gate/up) and Q3_K (down), plus q8_K activations.
// Layouts, reference quantizers and dequantizers follow llama.cpp ggml-quants.c (MIT) exactly, so a block made
// here is byte-identical in meaning to one in a GGUF / Strata pack (no imatrix: the uncalibrated path).
#pragma once
#include "mxfp4.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace fp4 {

constexpr int QKK = 256;
constexpr int Q2K_BYTES = 84, Q3K_BYTES = 110;

struct BlockQ2K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };
struct BlockQ3K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct BlockQ8K { float d; int8_t qs[QKK]; int16_t bsums[16]; };
static_assert(sizeof(BlockQ2K) == Q2K_BYTES, "q2_K");
static_assert(sizeof(BlockQ3K) == Q3K_BYTES, "q3_K");

inline float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 1023, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { int sh = 0; while (!(m & 1024)) { m <<= 1; ++sh; } m &= 1023; bits = s | ((113 - sh) << 23) | (m << 13); }
    } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}
inline uint16_t f2h(float f) {  // round to nearest even
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u; int e = int((x >> 23) & 255) - 127 + 15; uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 255) == 255) return uint16_t(s | 0x7c00u | (m ? 0x200u : 0));
    if (e >= 31) return uint16_t(s | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return uint16_t(s);
        m |= 0x800000u; const int sh = 14 - e; uint32_t r = m >> sh; const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return uint16_t(s | r);
    }
    uint32_t r = (uint32_t(e) << 10) | (m >> 13); const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return uint16_t(s | r);
}

namespace kq_detail {
inline int nearest_int(float fval) {
    float val = fval + 12582912.f; int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
inline float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, uint8_t* L, float* the_min,
                              uint8_t* Laux, float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0], max = x[0], sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i]; sum_w += w; sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) { for (int i = 0; i < n; ++i) L[i] = 0; *the_min = -min; return 0.f; }
    float iscale = nmax / (max - min), scale = 1 / iscale, best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * (x[i] - min));
        L[i] = std::max(0, std::min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) { *the_min = -min; return scale; }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = std::max(0, std::min(nmax, nearest_int(iscale * (x[i] - min))));
            Laux[i] = l; float w = weights[i];
            sum_l += w * l; sum_l2 += w * l * l; sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D, this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? std::fabs(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error; scale = this_scale; min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}
inline float make_q3_quants(int n, int nmax, const float* x, int8_t* L, bool do_rmse) {
    float max = 0, amax = 0;
    for (int i = 0; i < n; ++i) { float ax = std::fabs(x[i]); if (ax > amax) { amax = ax; max = x[i]; } }
    if (amax < 1e-15f) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0, suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = std::max(-nmax, std::min(nmax - 1, nearest_int(iscale * x[i])));
            L[i] = l; float w = x[i] * x[i]; sumlx += w * x[i] * l; suml2 += w * l * l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i] * x[i], slx = sumlx - w * x[i] * L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w * L[i] * L[i];
                    int new_l = std::max(-nmax, std::min(nmax - 1, nearest_int(x[i] * sl2 / slx)));
                    if (new_l != L[i]) {
                        slx += w * x[i] * new_l; sl2 += w * new_l * new_l;
                        if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) { L[i] = new_l; sumlx = slx; suml2 = sl2; ++n_changed; }
                    }
                }
            }
            if (!n_changed) break;
        }
        for (int i = 0; i < n; ++i) L[i] += nmax;
        return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }
    for (int i = 0; i < n; ++i) L[i] = std::max(-nmax, std::min(nmax - 1, nearest_int(iscale * x[i]))) + nmax;
    return 1 / iscale;
}
}  // namespace kq_detail

inline void quantize_row_q2k(const float* x, BlockQ2K* y, int k) {
    using namespace kq_detail;
    uint8_t L[QKK], Laux[16];
    float weights[16], mins[16], scales[16];
    for (int i = 0; i < k / QKK; i++, x += QKK) {
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < 16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = std::fabs(x[16 * j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            max_scale = std::max(max_scale, scales[j]);
            max_min = std::max(max_min, mins[j]);
        }
        if (max_scale > 0) {
            float iscale = 15.f / max_scale;
            for (int j = 0; j < 16; ++j) y[i].scales[j] = nearest_int(iscale * scales[j]);
            y[i].d = f2h(max_scale / 15.f);
        } else { for (int j = 0; j < 16; ++j) y[i].scales[j] = 0; y[i].d = f2h(0.f); }
        if (max_min > 0) {
            float iscale = 15.f / max_min;
            for (int j = 0; j < 16; ++j) y[i].scales[j] |= (nearest_int(iscale * mins[j]) << 4);
            y[i].dmin = f2h(max_min / 15.f);
        } else y[i].dmin = f2h(0.f);
        for (int j = 0; j < 16; ++j) {
            const float d = h2f(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = h2f(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) L[16 * j + ii] = std::max(0, std::min(3, nearest_int((x[16 * j + ii] + dm) / d)));
        }
        for (int j = 0; j < QKK; j += 128)
            for (int l = 0; l < 32; ++l)
                y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
    }
}

inline void quantize_row_q3k(const float* x, BlockQ3K* y, int k) {
    using namespace kq_detail;
    int8_t L[QKK];
    float scales[16];
    for (int i = 0; i < k / QKK; i++, x += QKK) {
        float max_scale = 0, amax = 0;
        for (int j = 0; j < 16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16 * j, L + 16 * j, true);
            float scale = std::fabs(scales[j]);
            if (scale > amax) { amax = scale; max_scale = scales[j]; }
        }
        std::memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f / max_scale;
            for (int j = 0; j < 16; ++j) {
                int8_t l = int8_t(std::max(-32, std::min(31, nearest_int(iscale * scales[j]))) + 32);
                if (j < 8) y[i].scales[j] = l & 0xF; else y[i].scales[j - 8] |= ((l & 0xF) << 4);
                l >>= 4;
                y[i].scales[j % 4 + 8] |= (l << (2 * (j / 4)));
            }
            y[i].d = f2h(1 / iscale);
        } else y[i].d = f2h(0.f);
        for (int j = 0; j < 16; ++j) {
            int sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
            sc = (sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
            float d = h2f(y[i].d) * sc;
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) L[16 * j + ii] = int8_t(std::max(-4, std::min(3, nearest_int(x[16 * j + ii] / d))) + 4);
        }
        std::memset(y[i].hmask, 0, 32);
        int m = 0; uint8_t hm = 1;
        for (int j = 0; j < QKK; ++j) {
            if (L[j] > 3) { y[i].hmask[m] |= hm; L[j] -= 4; }
            if (++m == 32) { m = 0; hm <<= 1; }
        }
        for (int j = 0; j < QKK; j += 128)
            for (int l = 0; l < 32; ++l)
                y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
    }
}

inline void dequant_row_q2k(const BlockQ2K* x, float* y, int k) {
    for (int i = 0; i < k / QKK; i++) {
        const float d = h2f(x[i].d), min = h2f(x[i].dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        for (int n = 0; n < QKK; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l] >> shift) & 3) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

inline void q3k_scales(const uint8_t* s12, int8_t* out16) {  // 16 signed scales (already minus 32)
    uint32_t aux[4];
    std::memcpy(aux, s12, 12);
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(out16, aux, 16);
    for (int i = 0; i < 16; ++i) out16[i] -= 32;
}

inline void dequant_row_q3k(const BlockQ3K* x, float* y, int k) {
    for (int i = 0; i < k / QKK; i++) {
        const float d_all = h2f(x[i].d);
        int8_t sc[16];
        q3k_scales(x[i].scales, sc);
        const uint8_t* q = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;
        int is = 0;
        for (int n = 0; n < QKK; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * sc[is++];
                for (int l = 0; l < 16; ++l) *y++ = dl * (int((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * sc[is++];
                for (int l = 0; l < 16; ++l) *y++ = dl * (int((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

inline void quantize_row_q8k(const float* x, BlockQ8K* y, int k) {
    for (int i = 0; i < k / QKK; i++, x += QKK) {
        float max = 0, amax = 0;
        for (int j = 0; j < QKK; ++j) { float ax = std::fabs(x[j]); if (ax > amax) { amax = ax; max = x[j]; } }
        if (!amax) { y[i].d = 0; std::memset(y[i].qs, 0, QKK); std::memset(y[i].bsums, 0, 32); continue; }
        const float iscale = -127.f / max;
        for (int j = 0; j < QKK; ++j) y[i].qs[j] = int8_t(std::min(127, kq_detail::nearest_int(iscale * x[j])));
        for (int j = 0; j < 16; ++j) { int s = 0; for (int ii = 0; ii < 16; ++ii) s += y[i].qs[j * 16 + ii]; y[i].bsums[j] = int16_t(s); }
        y[i].d = 1 / iscale;
    }
}

// ---- K-quant matrices and requantized experts ------------------------------------------------------
enum class KType { Q2K, Q3K, MXFP4 };
inline size_t row_bytes(KType t, int cols) {
    return t == KType::Q2K ? size_t(cols) / QKK * Q2K_BYTES : t == KType::Q3K ? size_t(cols) / QKK * Q3K_BYTES : size_t(cols) / 2 + cols / 32;
}

struct KMat {  // K-quant rows, contiguous (MXFP4 mats keep the HF two-array layout in q/e instead)
    KType type = KType::Q2K;
    int rows = 0, cols = 0;
    std::vector<uint8_t> data;
    const uint8_t* row(int r) const { return data.data() + size_t(r) * row_bytes(type, cols); }
    void dequant_row(int r, float* out) const {
        if (type == KType::Q2K) dequant_row_q2k((const BlockQ2K*)row(r), out, cols);
        else dequant_row_q3k((const BlockQ3K*)row(r), out, cols);
    }
    double ref_dot(int r, const float* x) const {
        std::vector<float> w(cols);
        dequant_row(r, w.data());
        double s = 0;
        for (int c = 0; c < cols; ++c) s += double(w[c]) * x[c];
        return s;
    }
    size_t bytes() const { return data.size(); }
};

// Requantize an MXFP4 matrix (exact f32 dequant first), as strata-mimo-q2-pack / ggml_quantize_chunk do.
inline KMat requant(const MxView& W, KType t, double* nmse = nullptr) {
    KMat m;
    m.type = t; m.rows = W.rows; m.cols = W.cols;
    m.data.resize(size_t(W.rows) * row_bytes(t, W.cols));
    std::vector<float> row(W.cols), back(W.cols);
    double err = 0, norm = 0;
    for (int r = 0; r < W.rows; ++r) {
        for (int c = 0; c < W.cols; ++c) row[c] = W.at(r, c);
        uint8_t* dst = m.data.data() + size_t(r) * row_bytes(t, W.cols);
        if (t == KType::Q2K) quantize_row_q2k(row.data(), (BlockQ2K*)dst, W.cols);
        else quantize_row_q3k(row.data(), (BlockQ3K*)dst, W.cols);
        if (nmse) {
            m.dequant_row(r, back.data());
            for (int c = 0; c < W.cols; ++c) { double d = double(back[c]) - row[c]; err += d * d; norm += double(row[c]) * row[c]; }
        }
    }
    if (nmse) *nmse = norm > 0 ? err / norm : 0;
    return m;
}

}  // namespace fp4
