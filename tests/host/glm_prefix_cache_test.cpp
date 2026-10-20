// tests/host/glm_prefix_cache_test.cpp - the serve prefix cache's choice of a resume point (glm_prefix_cache.hpp).
#include "../../src/program/glm_prefix_cache.hpp"

#include <cstdio>
#include <vector>

using strata::glm::PrefixCache;

static int fails = 0;
#define EXPECT(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++fails; } } while (0)

int main() {
    // an agent loop: each prompt extends the previous one
    PrefixCache<int> c(2);
    std::vector<int> p1 = {1, 2, 3, 4, 5};
    EXPECT(c.choose(p1) == nullptr);                          // nothing saved yet
    c.held = p1;
    auto *e = c.slot_for(5, false); e->store = 51; c.commit(e);
    c.held.insert(c.held.end(), {9, 9});                      // generated tokens follow the prompt
    std::vector<int> p2 = {1, 2, 3, 4, 5, 6, 7, 8};
    auto *r = c.choose(p2);
    EXPECT(r && r->pos == 5 && r->store == 51);

    // the whole prompt held: resume one below it (the last prompt token is read again for the logits)
    std::vector<int> same = {1, 2, 3, 4, 5};
    EXPECT(c.choose(same) == nullptr);                        // the only entry is at 5 = n, above n - 1
    // a different conversation: no common prefix, nothing to resume
    EXPECT(c.choose({7, 7, 7, 7}) == nullptr);

    // restoring below an entry invalidates it, and held
    auto *pin = c.slot_for(2, true); pin->store = 21; c.commit(pin);
    c.truncate(2);
    EXPECT(c.held.size() == 2);
    r = c.choose({1, 2, 3, 4, 5, 6});
    EXPECT(r && r->pos == 2 && r->store == 21);

    // LRU eviction keeps the pinned entry
    c.held = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto *a = c.slot_for(6, false); c.commit(a);
    EXPECT(a != pin);
    auto *b = c.slot_for(8, false); c.commit(b);              // both slots used: evicts the unpinned one (6)
    EXPECT(b == a && b->pos == 8 && pin->live && pin->pos == 2);
    r = c.choose({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    EXPECT(r && r->pos == 8);

    // saving the same position again reuses its entry
    auto *again = c.slot_for(8, false); c.commit(again);
    EXPECT(again == b);
    c.clear();
    EXPECT(c.choose({1, 2, 3}) == nullptr && c.held.empty());

    // no entries: never resumes
    PrefixCache<int> none(0);
    none.held = {1, 2, 3};
    EXPECT(none.slot_for(2, false) == nullptr && none.choose({1, 2, 3, 4}) == nullptr);

    std::printf("glm_prefix_cache_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
