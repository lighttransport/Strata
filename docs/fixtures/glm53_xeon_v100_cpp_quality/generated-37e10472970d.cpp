#include <cstdint>
#include <string_view>

// Parses an unsigned 64-bit decimal integer. Returns false and leaves `out`
// unchanged on any failure (empty input, non-digit bytes, or overflow).
bool parse_u64(std::string_view text, std::uint64_t& out) noexcept
{
    const std::size_t n = text.size();
    if (n == 0) {
        return false;
    }

    std::uint64_t value = 0;
    // Overflow threshold: 18446744073709551615 / 10 = 1844674407370955161,
    // remainder 5. Above this quotient, any digit > 5 overflows on *10.
    constexpr std::uint64_t kMaxOver10 = 1844674407370955161ULL;
    constexpr std::uint64_t kMaxOver10Rem = 5ULL;

    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < '0' || c > '9') {
            return false; // sign, space, NUL, non-ASCII, or other byte
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
        // Check overflow BEFORE multiply/add.
        if (value > kMaxOver10 ||
            (value == kMaxOver10 && digit > kMaxOver10Rem)) {
            return false;
        }
        value = value * 10ULL + digit;
    }

    out = value;
    return true;
}
