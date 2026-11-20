// include/strata/cpu_runner/vexpf.h - expf over SVE-512 vectors with the bits of the system's scalar expf, for
// A64FX nodes whose libm is glibc 2.27+ (sysdeps/ieee754/flt-32/e_expf.c; the node's 2.28 libm disassembled: one
// double multiply by 32 / ln 2, frintn / fcvtns, r = z - kd, the 32-entry table __exp2f_data.tab, three fmadd and a
// fmul in double, one rounding to float). The table and constants below are that libm's bytes. Inputs outside its fast
// path (top 12 bits of |x| above 0x42a: |x| >= ~80, inf, NaN) take the scalar expf. Callers verify vexpf_ok() (a
// self-check against std::exp) before using it.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
#include <arm_sve.h>
#endif

namespace strata::cpu_runner {

inline constexpr uint64_t kExp2fTab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull,
    0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, 0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull,
    0x3feedea64c123422ull, 0x3feece086061892dull, 0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull,
    0x3feea47eb03a5585ull, 0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull, 0x3feee89f995ad3adull,
    0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full,
    0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull};
inline constexpr uint64_t kExpfInvLn2N = 0x40471547652b82feull;                    // 32 / ln 2
inline constexpr uint64_t kExpfC[3] = {0x3ebc6af84b912394ull, 0x3f2ebfce50fac4f3ull, 0x3f962e42ff0c52d6ull};   // poly_scaled

inline double expf_bits_d(uint64_t u) { double d; std::memcpy(&d, &u, 8); return d; }

#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
// 8 doubles (each a float input) -> the expf results as doubles before the final rounding
inline svfloat64_t vexpf_core(svfloat64_t xd) {
    const svbool_t p = svptrue_b64();
    const svfloat64_t z = svmul_n_f64_x(p, xd, expf_bits_d(kExpfInvLn2N));
    const svfloat64_t kd = svrintn_f64_x(p, z);
    const svint64_t ki = svcvt_s64_f64_x(p, kd);   // (kd is an integer: fcvtns(z)'s value)
    const svfloat64_t r = svsub_f64_x(p, z, kd);
    const svuint64_t t = svadd_u64_x(p, svld1_gather_u64index_u64(p, kExp2fTab, svreinterpret_u64_s64(svand_n_s64_x(p, ki, 31))),
                                     svlsl_n_u64_x(p, svreinterpret_u64_s64(ki), 47));
    const svfloat64_t z1 = svmad_n_f64_x(p, svdup_f64(expf_bits_d(kExpfC[0])), r, expf_bits_d(kExpfC[1]));   // C0 r + C1
    const svfloat64_t y0 = svmad_n_f64_x(p, svdup_f64(expf_bits_d(kExpfC[2])), r, 1.0);                     // C2 r + 1
    const svfloat64_t y = svmla_f64_x(p, y0, z1, svmul_f64_x(p, r, r));                                     // z1 r^2 + y0
    return svmul_f64_x(p, y, svreinterpret_f64_u64(t));
}
// y[0..16) = expf(x[0..16))
inline void vexpf16(const float *x, float *y) {
    const svbool_t p32 = svptrue_b32();
    const svfloat32_t v = svld1_f32(p32, x);
    const svuint32_t top = svand_n_u32_x(p32, svlsr_n_u32_x(p32, svreinterpret_u32_f32(v), 20), 0x7ff);
    if (svptest_any(p32, svcmpgt_n_u32(p32, top, 0x42a))) {
        for (int i = 0; i < 16; ++i) y[i] = std::exp(x[i]);
        return;
    }
    const svbool_t p64 = svptrue_b64();
    const svfloat32_t lo = svcvt_f32_f64_x(p64, vexpf_core(svcvt_f64_f32_x(p64, svzip1_f32(v, v))));
    const svfloat32_t hi = svcvt_f32_f64_x(p64, vexpf_core(svcvt_f64_f32_x(p64, svzip2_f32(v, v))));
    svst1_f32(p32, y, svuzp1_f32(lo, hi));
}
#endif

// y[i] = expf(x[i]) for i < n (y may be x)
inline void vexpf(const float *x, float *y, int n) {
    int i = 0;
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    for (; i + 16 <= n; i += 16) vexpf16(x + i, y + i);
#endif
    for (; i < n; ++i) y[i] = std::exp(x[i]);
}

// a self-check against std::exp over a sweep of the fast path's inputs (both signs, every exponent): false when the
// libm differs (then callers keep std::exp)
inline bool vexpf_ok() {
    static const bool ok = [] {
        float a[16], b[16];
        uint32_t s = 2463534242u;
        for (int it = 0; it < 4096; ++it) {
            for (int k = 0; k < 16; ++k) {
                s ^= s << 13, s ^= s >> 17, s ^= s << 5;
                const uint32_t u = (s & 0x807fffffu) | ((uint32_t(it + k) % 0x86u) << 23);   // exponents 0..133
                std::memcpy(&a[k], &u, 4);
            }
            vexpf(a, b, 16);
            for (int k = 0; k < 16; ++k) {
                const float r = std::exp(a[k]);
                if (std::memcmp(&r, &b[k], 4)) return false;
            }
        }
        return true;
    }();
    return ok;
}

}  // namespace strata::cpu_runner
