#include <cstdint>
#include <string_view>
#include <limits>

bool parse_u64(std::string_view text, std::uint64_t& out) noexcept {
    if (text.empty()) return false;
    std::uint64_t acc = 0;
    constexpr std::uint64_t max_v = std::numeric_limits<std::uint64_t>::max();
    for (char ch : text) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (c < '0' || c > '9') return false;
        std::uint64_t d = static_cast<std::uint64_t>(c - '0');
        if (acc > (max_v - d) / 10) return false;
        acc = acc * 10 + d;
    }
    out = acc;
    return true;
}
