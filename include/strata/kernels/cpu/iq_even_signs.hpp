// IQ2/IQ3's seven sign bits plus an eighth bit that makes parity even.
#pragma once
#include <cstdint>

namespace strata::kernels::cpu::detail {

// Each byte holds a seven-bit index; its high bit must be zero.
constexpr uint64_t iq_even_signs_packed(uint64_t indices) noexcept {
    uint64_t parity = indices ^ (indices >> 4);
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    return indices | ((parity & 0x0101010101010101ull) << 7);
}

// Each word holds four packed seven-bit indices and four scale bits.
constexpr uint64_t iq_even_signs_words(uint32_t first, uint32_t second) noexcept {
    const uint64_t words = uint64_t(first) | (uint64_t(second) << 32);
    const uint64_t indices = (words & 0x0000007f0000007full) |
                            ((words << 1) & 0x00007f0000007f00ull) |
                            ((words << 2) & 0x007f0000007f0000ull) |
                            ((words << 3) & 0x7f0000007f000000ull);
    return iq_even_signs_packed(indices);
}

}  // namespace strata::kernels::cpu::detail
