// MXFP4 (OCP MX, E2M1 elements, E8M0 scale per 32) helpers shared by the CPU and GPU benchmarks.
//
// Weight layout used everywhere is the MiMo-V2.6 checkpoint layout, unchanged ("HF layout"):
//   q[row][col/2]  : element 2i in the low nibble, 2i+1 in the high nibble
//   e[row][col/32] : E8M0 biased exponent, value = 2^(e-127)
// No kernel repacks weights. Instead, activations are permuted once per token into the
// "logical" order per 32-block: [x0,x2,...,x30 | x1,x3,...,x31] (evens then odds). Splitting a
// packed byte run into low nibbles and high nibbles then lines up with the activation directly,
// on AVX2 (pshufb) and on CUDA (dp4a / int8 mma fragments).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

namespace fp4 {

constexpr int QK = 32;  // MX block size

// E2M1 code -> value * 2 (exact int8): {0,.5,1,1.5,2,3,4,6} * 2, sign in bit 3.
static const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static const float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

inline float e8m0_to_float(uint8_t e) {  // 2^(e-127); e==0 -> 2^-127 (denormal), 255 = NaN (unused)
    return std::ldexp(1.0f, int(e) - 127);
}
// Scale to apply to kE2M1x2 values: 2^(e-128).
inline float e8m0_half(uint8_t e) { return std::ldexp(1.0f, int(e) - 128); }

// Logical position (0..31) -> physical element within a 32-block.
inline int logical_to_phys(int l) { return l < 16 ? 2 * l : 2 * (l - 16) + 1; }

// Non-owning view of an MXFP4 matrix in HF layout.
struct MxView {
    int rows = 0, cols = 0;
    const uint8_t* q = nullptr;  // rows * cols/2
    const uint8_t* e = nullptr;  // rows * cols/32
    size_t qbytes() const { return size_t(rows) * cols / 2; }
    size_t ebytes() const { return size_t(rows) * cols / QK; }
    size_t bytes() const { return qbytes() + ebytes(); }
    float at(int r, int c) const {
        uint8_t b = q[size_t(r) * cols / 2 + c / 2];
        int code = (c & 1) ? (b >> 4) : (b & 15);
        return kE2M1[code] * e8m0_to_float(e[size_t(r) * (cols / QK) + c / QK]);
    }
};

// ---- fp64 reference --------------------------------------------------------------------------
inline void ref_gemv(const MxView& W, const float* x, double* y) {
    std::vector<double> rowv(W.cols);
    for (int r = 0; r < W.rows; ++r) {
        double s = 0;
        for (int c = 0; c < W.cols; ++c) s += double(W.at(r, c)) * x[c];
        y[r] = s;
    }
}
inline double ref_dot_row(const MxView& W, int r, const float* x) {
    double s = 0;
    for (int c = 0; c < W.cols; ++c) s += double(W.at(r, c)) * x[c];
    return s;
}

// ---- activation formats (host side) ------------------------------------------------------------
// q8 in logical order; d = amax/127; sum = sum of int8 values (for offset tricks).
struct Q8Block { float d; int32_t sum; int8_t qs[QK]; };

inline void quantize_q8_logical(const float* x, int n, Q8Block* out) {
    for (int b = 0; b < n / QK; ++b) {
        const float* xb = x + b * QK;
        float amax = 0;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(xb[i]));
        float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
        int32_t s = 0;
        for (int l = 0; l < QK; ++l) {
            int v = (int)std::nearbyint(xb[logical_to_phys(l)] * id);
            v = std::max(-127, std::min(127, v));
            out[b].qs[l] = (int8_t)v;
            s += v;
        }
        out[b].d = d;
        out[b].sum = s;
    }
}

inline void permute_logical(const float* x, int n, float* out) {
    for (int b = 0; b < n / QK; ++b)
        for (int l = 0; l < QK; ++l) out[b * QK + l] = x[b * QK + logical_to_phys(l)];
}

// E4M3 (fn, no inf, max 448) encode/decode.
inline float e4m3_to_float(uint8_t v) {
    int s = v >> 7, ex = (v >> 3) & 15, m = v & 7;
    float f;
    if (ex == 15 && m == 7) f = NAN;
    else if (ex == 0) f = std::ldexp(float(m), -9);
    else f = std::ldexp(1.f + m / 8.f, ex - 7);
    return s ? -f : f;
}
inline uint8_t float_to_e4m3(float x) {  // RNE by table search, saturating
    static float pos[127];
    static bool init = false;
    if (!init) { for (int i = 0; i < 127; ++i) pos[i] = e4m3_to_float((uint8_t)i); init = true; }
    uint8_t sign = std::signbit(x) ? 0x80 : 0;
    float a = std::fabs(x);
    if (!(a < 448.f)) return sign | 126;
    int lo = int(std::upper_bound(pos, pos + 127, a) - pos) - 1;  // pos[lo] <= a
    if (lo >= 126) return sign | 126;
    float d0 = a - pos[lo], d1 = pos[lo + 1] - a;
    int pick = (d0 < d1) ? lo : (d1 < d0 ? lo + 1 : ((lo & 1) ? lo + 1 : lo));
    return sign | (uint8_t)pick;
}

// E2M1 encode with RNE (input already divided by the block scale), saturating at 6.
inline uint8_t float_to_e2m1(float x) {
    uint8_t sign = std::signbit(x) ? 8 : 0;
    float a = std::fabs(x);
    static const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    if (a >= 6.f) return sign | 7;
    int lo = 0;
    while (lo < 7 && mag[lo + 1] <= a) ++lo;
    if (lo == 7) return sign | 7;
    float d0 = a - mag[lo], d1 = mag[lo + 1] - a;
    int pick = (d0 < d1) ? lo : (d1 < d0 ? lo + 1 : ((lo & 1) ? lo + 1 : lo));
    return sign | (uint8_t)pick;
}

// OCP MX scale choice: shared exponent = floor(log2(amax)) - emax_elem (2 for E2M1, 8 for E4M3).
inline uint8_t mx_scale_exp(float amax, int emax_elem) {
    if (amax == 0.f) return 1;
    int ex = (int)std::floor(std::log2(amax)) - emax_elem;
    return (uint8_t)std::max(1, std::min(254, ex + 127));
}

// Activation -> MXFP4 (natural order, packed like the weights). Returns dequantized copy if asked.
inline void quantize_mxfp4_act(const float* x, int n, uint8_t* q, uint8_t* e, float* deq = nullptr) {
    for (int b = 0; b < n / QK; ++b) {
        float amax = 0;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[b * QK + i]));
        uint8_t se = mx_scale_exp(amax, 2);
        float s = e8m0_to_float(se), is = 1.f / s;
        e[b] = se;
        for (int i = 0; i < QK; i += 2) {
            uint8_t c0 = float_to_e2m1(x[b * QK + i] * is), c1 = float_to_e2m1(x[b * QK + i + 1] * is);
            q[(b * QK + i) / 2] = c0 | (c1 << 4);
            if (deq) { deq[b * QK + i] = kE2M1[c0] * s; deq[b * QK + i + 1] = kE2M1[c1] * s; }
        }
    }
}

// Activation -> E4M3 with an E8M0 scale per 32 (MXFP8), in logical order.
inline void quantize_mxfp8_act_logical(const float* x, int n, uint8_t* q, uint8_t* e) {
    for (int b = 0; b < n / QK; ++b) {
        float amax = 0;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[b * QK + i]));
        uint8_t se = mx_scale_exp(amax, 8);
        float is = 1.f / e8m0_to_float(se);
        e[b] = se;
        for (int l = 0; l < QK; ++l) q[b * QK + l] = float_to_e4m3(x[b * QK + logical_to_phys(l)] * is);
    }
}

// Activation -> E4M3 with a float scale per 32 (amax -> 448), logical order.
inline void quantize_fp8_act_logical(const float* x, int n, uint8_t* q, float* d) {
    for (int b = 0; b < n / QK; ++b) {
        float amax = 0;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[b * QK + i]));
        float s = amax > 0 ? amax / 448.f : 1.f;
        d[b] = s;
        for (int l = 0; l < QK; ++l) q[b * QK + l] = float_to_e4m3(x[b * QK + logical_to_phys(l)] / s);
    }
}

// ---- error metrics ------------------------------------------------------------------------------
struct ErrStats { double rel_l2 = 0, max_abs = 0, ref_rms = 0; bool finite = true; };
template <class T>
inline ErrStats compare(const T* got, const double* ref, size_t n) {
    ErrStats s;
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        double g = double(got[i]);
        if (!std::isfinite(g)) s.finite = false;
        double d = g - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
        s.max_abs = std::max(s.max_abs, std::fabs(d));
    }
    s.rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    s.ref_rms = std::sqrt(den / double(n ? n : 1));
    return s;
}

// FNV-1a over output floats (rounded to 1e-3 relative is overkill; raw bits are fine for a run).
inline uint64_t hash_floats(const float* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n * 4; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

}  // namespace fp4
