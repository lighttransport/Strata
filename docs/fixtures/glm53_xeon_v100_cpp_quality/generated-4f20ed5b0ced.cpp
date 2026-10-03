#include <cstdint>
#include <string_view>
#include <limits>

bool parse_u64(std::string_view text, std::uint64_t& out) noexcept
{
    constexpr std::uint64_t MAX = std::numeric_limits<std::uint64_t>::max();
    constexpr std::uint64_t MAX_OVER_10 = MAX / 10;
    constexpr std::uint64_t MAX_REM = MAX - MAX_OVER_10 * 10; // 5

    if (text.empty())
        return false;

    std::uint64_t value = 0;
    bool any = false;

    for (char ch : text)
    {
        // Unsigned-safe byte handling: cast to unsigned char first.
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c < static_cast<unsigned char>('0') || c > static_cast<unsigned char>('9'))
            return false; // reject signs, whitespace, NUL, non-ASCII, etc.

        const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');

        // Check overflow before multiplication or addition.
        if (value > MAX_OVER_10 || (value == MAX_OVER_10 && digit > MAX_REM))
            return false;

        value = value * 10 + digit;
        any = true;
    }

    if (!any)
        return false;

    out = value;
    return true;
}
