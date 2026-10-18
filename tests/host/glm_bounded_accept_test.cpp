#include "strata/spec/bounded_accept.hpp"
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>

int main() {
    try {
        const auto require = [](bool ok) { if (!ok) throw std::runtime_error("bounded acceptance contract failed"); };
        const std::array<float, 5> logits{4.f, 3.f, 3.f, 2.f, 0.f};
        const std::array<int, 1> stops{4};
        strata::spec::BoundedAcceptPolicy policy{1.f, 2, 1};
        auto accept = [&](int draft, int target = 0, int used = 0) {
            return strata::spec::bounded_accept(logits, draft, target, policy, used, stops);
        };
        require(accept(0).accepted && !accept(0).soft);
        require(accept(1).accepted && accept(1).soft && accept(1).regret == 1.f);
        require(!accept(2).accepted); // Equal-logit rank resolves to the lower token ID.
        require(!accept(3).accepted && !accept(-1).accepted && !accept(5).accepted);
        require(!accept(1, 0, 1).accepted);
        require(!accept(1, 4).accepted && !accept(4).accepted);
        policy.margin = 0;
        require(!accept(1).accepted && accept(0).accepted);
        policy.margin = std::numeric_limits<float>::infinity();
        require(!accept(1).accepted);
        policy = {4.f, 5, 1};
        auto invalid = logits; invalid[3] = std::numeric_limits<float>::quiet_NaN();
        require(!strata::spec::bounded_accept(invalid, 1, 0, policy, 0, stops).accepted);
        invalid[3] = std::numeric_limits<float>::infinity();
        require(!strata::spec::bounded_accept(invalid, 1, 0, policy, 0, stops).accepted);
        auto masked = logits; masked[3] = -std::numeric_limits<float>::infinity();
        require(strata::spec::bounded_accept(masked, 1, 0, policy, 0, stops).soft);
        require(!strata::spec::bounded_accept(masked, 3, 0, policy, 0, stops).accepted);
        policy.margin=0;
        require(!strata::spec::bounded_accept(masked, 1, 0, policy, 0, stops).accepted);
        require(strata::spec::bounded_accept(masked, 0, 0, policy, 0, stops).accepted);
        const std::array<int, 5> history{1, 2, 3, 1, 2};
        require(strata::spec::repeats_ngram(history, 3, 3));
        require(!strata::spec::repeats_ngram(history, 4, 3));
        require(!strata::spec::repeats_ngram(history, 3, 1));
        require(!strata::spec::repeats_ngram(history, 3, 6));
        require(!strata::spec::repeats_ngram({}, 3, 3));
        std::puts("bounded acceptance and repeat guard PASS");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
