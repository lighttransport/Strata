#include <cstdint>
#include <string_view>
#include <limits>

bool parse_u64(std::string_view text, std::uint64_t& out) noexcept
{
    constexpr std::uint64_t MAX = std::numeric_limits<std::uint64_t>::max();
    constexpr std::uint64_t MAX_OVER_10 = MAX / 10;
    constexpr std::uint64_t MAX_LAST_DIGIT = MAX % 10;

    if (text.empty())
        return false;

    std::uint64_t value = 0;
    for (char ch : text)
    {
        const unsigned char byte = static_cast<unsigned char>(ch);
        if (byte < '0' || byte > '9')
            return false; // rejects signs, whitespace, NUL, non-ASCII
        const std::uint64_t digit = static_cast<std::uint64_t>(byte - '0');

        // Overflow check before multiply/add
        if (value > MAX_OVER_10 || (value == MAX_OVER_10 && digit > MAX_LAST_DIGIT))
            return false;

        value = value * 10 + digit;
    }

    out = value;
    return true;
}
