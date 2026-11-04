#pragma once

#include "strata/spec/suffix_drafter.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace strata::dsv41 {
// A request/rollback-safe adapter for the existing GLM suffix drafter. Proposals are never committed here.
class LookupDrafter {
public:
    explicit LookupDrafter(int min_match = 6) : lookup_(min_match, 32, kCapacity) {}

    std::vector<int> propose(const std::vector<int32_t>& consumed, int seed, int max_drafts) {
        if (max_drafts <= 0 || consumed.size() >= kCapacity) return {};
        max_drafts = std::min(max_drafts, 15);
        const auto& old = lookup_.history();
        if (old.size() > consumed.size() || !std::equal(old.begin(), old.end(), consumed.begin())) lookup_.reset();
        const size_t from = lookup_.size();
        if (from < consumed.size()) lookup_.append(consumed.data() + from, consumed.size() - from);
        std::array<int32_t, 15> out{};
        const int32_t cur = seed;
        const int n = lookup_.propose_after(&cur, 1, max_drafts, out.data());
        if (n < 0 || n > max_drafts) return {};
        std::vector<int> result;
        result.reserve((size_t) n);
        for (int i = 0; i < n; ++i) result.push_back(out[(size_t) i]);
        return result;
    }

private:
    static constexpr size_t kCapacity = 1u << 19;
    strata::spec::SuffixDrafter lookup_;
};
}  // namespace strata::dsv41
