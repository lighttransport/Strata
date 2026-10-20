#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Light Transport Entertainment, Inc.

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __HIPCC__
#define E0_HD __host__ __device__
#else
#define E0_HD
#endif

// Numeric contract from gemm's cuda/int4/int4_numeric.h; see LICENSE.
E0_HD inline int e0_value(unsigned q) {
    return q & 8 ? -int(q & 7) : int(q & 7);
}

E0_HD inline float scale_value(unsigned q) {
    unsigned e = (q >> 3) & 15, m = q & 7;
    return e ? ldexpf(1.0f + m * 0.125f, int(e) - 7) : m * (1.0f / 512.0f);
}

// Each nibble's addition stays within that nibble, including negative zero.
E0_HD inline uint32_t recode(uint32_t p) {
    uint32_t s = (p >> 3) & 0x11111111u;
    uint32_t t = ((p & 0x77777777u) ^ (s * 7u)) + s;
    return (t & 0x77777777u) | ((t ^ (s << 3)) & 0x88888888u);
}

E0_HD inline unsigned nibble(const uint8_t* p, std::size_t index) {
    return (p[index / 2] >> ((index & 1) * 4)) & 15;
}

#undef E0_HD
