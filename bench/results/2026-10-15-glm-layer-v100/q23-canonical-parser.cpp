#include <cstdint>
#include <string_view>
#include <limits>

bool parse_u64(std::string_view text, std::uint64_t& out) noexcept {
    constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    if (text.empty()) return false;
    std::uint64_t acc = 0;
    bool any = false;
    for (char ch : text) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (c < '0' || c > '9') return false;
        std::uint64_t d = static_cast<std::uint64_t>(c - '0');
        // overflow check before multiply and add
        if (acc > (max - 9) / 10) return false; // conservative: acc*10+d > max
        if (acc > (max - d) / 10) return false;
        acc = acc * 10 + d;
        any = true;
    }
    if (!any) return false;
    out = acc;
    return true;
}
