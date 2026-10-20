#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace strata::spec {
// Does appending token repeat a complete n-gram already in the emitted history?
// The caller bounds history and invokes this only for an otherwise soft accept.
inline bool repeats_ngram(std::span<const int> history, int token, std::size_t n) {
    if (n < 2 || history.size() < n) return false;
    const auto suffix = history.end() - (n - 1);
    for (std::size_t i = 0; i + n <= history.size(); ++i)
        if (history[i + n - 1] == token &&
            std::equal(suffix, history.end(), history.begin() + i)) return true;
    return false;
}
// Experimental, approximate greedy verification. A zero margin is exact.
// Target logits score the actual proposed prefix, including earlier soft accepts.
struct BoundedAcceptPolicy {
    float margin = 0;
    int top_k = 4;
    int max_soft_per_window = 1;
};
struct BoundedAcceptResult {
    bool accepted = false;
    bool soft = false;
    float regret = 0;
};
inline BoundedAcceptResult bounded_accept(std::span<const float> logits, int draft, int target,
                                         const BoundedAcceptPolicy &policy, int soft_used,
                                         std::span<const int> stops) {
    if (draft < 0 || target < 0 || static_cast<std::size_t>(draft) >= logits.size() ||
        static_cast<std::size_t>(target) >= logits.size()) return {};
    if (draft == target) return {true, false, 0};
    if (!(policy.margin > 0) || !std::isfinite(policy.margin) || policy.top_k < 2 ||
        soft_used >= policy.max_soft_per_window) return {};
    // Soft verification must neither invent nor suppress a stop token.
    for (int stop : stops) if (draft == stop || target == stop) return {};
    const float gap = logits[target] - logits[draft];
    if (!std::isfinite(gap) || gap < 0 || gap > policy.margin) return {};
    int rank = 1;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        // Negative infinity masks unused vocabulary slots. It contributes no
        // probability or rank; NaNs and positive infinity remain invalid.
        if (std::isnan(logits[i]) || logits[i] == INFINITY) return {};
        // Deterministic rank: lower token ID wins equal-logit ties.
        if (logits[i] > logits[draft] || (logits[i] == logits[draft] && i < static_cast<std::size_t>(draft)))
            if (++rank > policy.top_k) return {};
    }
    return {true, true, gap};
}
} // namespace strata::spec
