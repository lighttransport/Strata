#include "strata/spec/bounded_accept.hpp"
#include <cassert>
#include <limits>
#include <vector>
using strata::spec::bounded_accept;
using strata::spec::BoundedAcceptPolicy;
int main() {
    std::vector<float> logits{3.f, 2.75f, 2.5f, 2.f};
    const std::vector<int> stops{3};
    BoundedAcceptPolicy p;
    assert(bounded_accept(logits, 0, 0, p, 0, stops).accepted);
    assert(!bounded_accept(logits, 1, 0, p, 0, stops).accepted);
    p.margin = .5f; p.top_k = 2;
    auto r = bounded_accept(logits, 1, 0, p, 0, stops);
    assert(r.accepted && r.soft && r.regret == .25f);
    assert(!bounded_accept(logits, 2, 0, p, 0, stops).accepted); // rank
    assert(!bounded_accept(logits, 1, 0, p, 1, stops).accepted); // window budget
    p.top_k = 4;
    assert(bounded_accept(logits, 2, 0, p, 0, stops).accepted); // inclusive margin
    p.margin = .49f;
    assert(!bounded_accept(logits, 2, 0, p, 0, stops).accepted);
    p.margin = 10;
    assert(!bounded_accept(logits, 3, 0, p, 0, stops).accepted); // draft stop
    logits[3] = 3.1f;
    assert(!bounded_accept(logits, 0, 3, p, 0, stops).accepted); // target stop
    assert(!bounded_accept(logits, -1, 0, p, 0, stops).accepted);
    assert(!bounded_accept(logits, 4, 0, p, 0, stops).accepted);
    logits = {3.f, 3.f, 3.f, 2.f}; p.top_k = 2;
    assert(bounded_accept(logits, 1, 0, p, 0, stops).accepted);
    assert(!bounded_accept(logits, 2, 0, p, 0, stops).accepted);
    logits[3] = std::numeric_limits<float>::quiet_NaN();
    assert(!bounded_accept(logits, 1, 0, p, 0, stops).accepted);
    logits[3] = -std::numeric_limits<float>::infinity();
    assert(!bounded_accept(logits, 1, 0, p, 0, stops).accepted);
    // Disabled path preserves exact-token decisions regardless of other logits.
    p.margin = 0;
    assert(bounded_accept(logits, 0, 0, p, 0, stops).accepted);
}
