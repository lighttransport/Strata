#include "strata/dsv41/lookup_drafter.hpp"
#include <cstdio>
#include <vector>

int main() {
    strata::dsv41::LookupDrafter d(3);
    const std::vector<int32_t> history{1,2,3,4,5,6,99,1,2};
    if (d.propose(history,3,2) != std::vector<int>({4,5})) return 1;
    // A restored/shorter request cannot reuse stale continuation tokens.
    if (!d.propose(std::vector<int32_t>{1,2},3,2).empty()) return 2;
    // Same length, changed prefix: rebuild the lookup index.
    const std::vector<int32_t> other{7,8,3,10,11,12,99,7,8};
    if (d.propose(other,3,2) != std::vector<int>({10,11})) return 3;
    if (!d.propose(other,3,0).empty()) return 4;
    std::puts("lookup history, rollback, replacement and proposal bounds: OK");
    return 0;
}
