#include <charconv>
#include <cstdint>
#include <string_view>
#include <string>
#include <limits>
#include <vector>
#include <iostream>
#include <random>
#include <type_traits>
bool parse_u64(std::string_view s, std::uint64_t& out) noexcept {
    if (s.empty()) return false;          // nonempty only
    std::uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        std::uint64_t d = static_cast<std::uint64_t>(c - '0');
        if (v > (std::uint64_t{0} - d) / 10) return false; // would exceed UINT64_MAX
        v = v * 10 + d;
    }
    out = v;                               // touch out only on success
    return true;
}

static_assert(std::is_same_v<decltype(&parse_u64), bool(*)(std::string_view, std::uint64_t&) noexcept>);
bool reference(std::string_view s, std::uint64_t& out) {
    if (s.empty()) return false;
    for (unsigned char c : s) if (c < '0' || c > '9') return false;
    std::uint64_t value = 0;
    auto result = std::from_chars(s.data(), s.data()+s.size(), value);
    if (result.ec != std::errc{} || result.ptr != s.data()+s.size()) return false;
    out = value; return true;
}
int main() {
    std::vector<std::string> cases={"", "0", "00", "00042", "1", "42", "18446744073709551615", "18446744073709551616", "9999999999999999999999999", "+1", "-1", " 1", "1 ", "1\n", "1a", std::string("1\0", 2), std::string(1, char(255)), "\xd9\xa1", std::string(10000, '0'), std::string(10000, '9')};
    std::mt19937 random(20261009);
    for (int i=0;i<5000;++i) {
        std::string s;
        int n=random()%50;
        for(int j=0;j<n;++j) s += char('0'+random()%10);
        if(i%3==0 && !s.empty()) s[random()%s.size()]=char(random()%256);
        cases.push_back(s);
    }
    int failures=0;
    for(size_t i=0;i<cases.size();++i) {
        std::uint64_t expected=0xdeadbeef12345678ULL, actual=expected;
        bool e=reference(cases[i],expected), a=parse_u64(cases[i],actual);
        if(e!=a || expected!=actual) { ++failures; if(failures<10) std::cout<<"FAIL case="<<i<<" expected_ok="<<e<<" actual_ok="<<a<<" expected="<<expected<<" actual="<<actual<<"\n"; }
    }
    std::cout<<"PARSER_QUALITY cases="<<cases.size()<<" failures="<<failures<<"\n";
    return failures ? 1 : 0;
}
