#include <cstdint>
#include <string_view>
#include <limits>

bool parse_u64(std::string_view text, std::uint64_t& out) noexcept
{
    constexpr std::uint64_t MAX = std::numeric_limits<std::uint64_t>::max();
    constexpr std::uint64_t HI = MAX / 10;          // 1844674407370955161
    constexpr std::uint64_t LO_DIGIT = MAX % 10;    // 5

    if (text.empty())
        return false;

    std::uint64_t value = 0;
    for (char c : text)
    {
        const unsigned char b = static_cast<unsigned char>(c);
        if (b < '0' || b > '9')
            return false;   // rejects signs, whitespace, NUL, non-ASCII
        const std::uint64_t d = static_cast<std::uint64_t>(b - '0');

        // Overflow check before multiply/add:
        if (value > HI || (value == HI && d > LO_DIGIT))
            return false;

        value = value * 10 + d;
    }

    out = value;
    return true;
}
