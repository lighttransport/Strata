#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __HIPCC__
#define FP4_HD __host__ __device__
#else
#define FP4_HD
#endif

FP4_HD inline int fp4_x2(unsigned q) {
    unsigned m = q & 7, e = m >> 1;
    int x = e ? int((2 + (m & 1)) << (e - 1)) : int(m);
    return q & 8 ? -x : x;
}

FP4_HD inline unsigned fp4_to_fp8(unsigned q) {
    return unsigned((0x4c4844403c383000ull >> ((q & 7) * 8)) & 255) | ((q & 8) << 4);
}

FP4_HD inline float fp4_float(unsigned q) {
    unsigned m = q & 7;
    uint32_t bits = m < 2 ? m * 0x3f000000u : ((126u + (m >> 1)) << 23) | ((m & 1) << 22);
    bits |= (q & 8) << 28;
    float x;
    __builtin_memcpy(&x, &bits, sizeof(x));
    return x;
}

FP4_HD inline float scale_value(unsigned q) {
    unsigned e = q >> 3, m = q & 7;
    if (!e) return m * (1.0f / 512.0f);
    uint32_t bits = ((e + 120) << 23) | (m << 20);
    float x;
    __builtin_memcpy(&x, &bits, sizeof(x));
    return x;
}

FP4_HD inline unsigned nibble(const uint8_t* p, std::size_t index) {
    return (p[index / 2] >> ((index & 1) * 4)) & 15;
}

FP4_HD inline uint16_t to_bf16(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(bits));
    return uint16_t(bits >> 16);
}

#undef FP4_HD
