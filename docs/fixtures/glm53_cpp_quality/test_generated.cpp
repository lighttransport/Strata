#ifndef GENERATED_SOURCE
#define GENERATED_SOURCE "generated.cpp"
#endif
#include GENERATED_SOURCE
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
static std::size_t cases = 0;
static void check(std::string_view input) {
    std::uint64_t expected = 0;
    bool valid = !input.empty();
    for (unsigned char c : input) valid &= c >= '0' && c <= '9';
    if (valid) {
        auto result = std::from_chars(input.data(), input.data()+input.size(), expected);
        valid = result.ec == std::errc{} && result.ptr == input.data()+input.size();
    }
    constexpr std::uint64_t sentinel = 0x123456789abcdef0ULL;
    std::uint64_t actual = sentinel;
    bool ok = parse_u64(input, actual);
    ++cases;
    if (ok != valid || actual != (valid ? expected : sentinel)) {
        std::cerr << "Failure at case " << cases << " length " << input.size() << '\n';
        std::exit(1);
    }
}
int main() {
    for (auto s : {"", "0", "000", "1", "42", "18446744073709551614",
                   "18446744073709551615", "18446744073709551616", "99999999999999999999",
                   "+1", "-1", " 1", "1 ", "1\n", "1.0", "0x10", "1e3"}) check(s);
    check(std::string("12\0" "34", 5));
    check(std::string(10000, '0') + "18446744073709551615");
    check(std::string(10000, '9'));
    for (unsigned n=0; n<256; ++n) {
        check(std::string(1, static_cast<char>(n)));
        check("123" + std::string(1, static_cast<char>(n)) + "456");
    }
    std::mt19937_64 rng(20261001);
    for (int i=0; i<100000; ++i) {
        std::uint64_t n = rng();
        check(std::to_string(n));
        check(std::string(i%32, '0') + std::to_string(n));
        std::string digits;
        for (unsigned j=0, length=rng()%40; j<length; ++j) digits += char('0'+rng()%10);
        check(digits);
        std::string bytes;
        for (unsigned j=0, length=rng()%40; j<length; ++j) bytes += static_cast<char>(rng()%256);
        check(bytes);
    }
    std::cout << "PASS " << cases << " cases\n";
}
