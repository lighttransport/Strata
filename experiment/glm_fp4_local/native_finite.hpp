#pragma once
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::experiment {
// Classify without FP arithmetic: rejecting a signaling NaN need not raise an
// FP exception. Integer reduction allows vectorization without fast-math.
inline bool native_all_finite(const float* values, size_t count) {
    static_assert(sizeof(float)==sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
    uint32_t nonfinite = 0;
    for (size_t i=0;i<count;++i)
        nonfinite |= uint32_t((std::bit_cast<uint32_t>(values[i]) & 0x7f800000u)==0x7f800000u);
    return nonfinite == 0;
}
}
