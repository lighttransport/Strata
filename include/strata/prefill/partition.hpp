#pragma once

namespace strata::prefill {
// Spread whole 16-expert groups across two devices without changing the order
// within any group. Nine primary groups reproduces the original even/odd split.
constexpr bool owns_expert_group(int group, int participant, int participants, int primary_groups = 9, int total_groups = 18) {
    if (participants == 1) return participant == 0;
    const bool primary = ((group + 1) * primary_groups + total_groups - 1) / total_groups >
                         (group * primary_groups + total_groups - 1) / total_groups;
    return participant == (primary ? 0 : 1);
}
}  // namespace strata::prefill
