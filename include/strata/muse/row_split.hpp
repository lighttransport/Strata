// include/strata/muse/row_split.hpp - one node's share of every routed expert in Muse Spark's two-node row split
// (strata_muse_row_split in strata/host/muse_names.h): ff rows [f0, f1) of gate and up, and the same columns of down
// (whole quantization blocks of each down row). Run through an ExpertLane built for ff width f1 - f0, the share gives
// each expert's partial output; the partials of the two nodes add up to the expert's output (in another summation
// order than one node's full rows, so not bit for bit), which ExpertLane::combine then normalizes and weights.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::muse {

struct ExpertShare {
    int f0 = 0, f1 = 0;                     // the ff rows of this share
    std::vector<uint8_t> gate, up, down;    // [experts][f1 - f0][gu row], same, [experts][latent][down row of f1 - f0]
};

/// gate/up: [experts][ff][gu_row] as in the GGUF; down: [experts][latent][row of ff values in d_type].
/// f0, f1: multiples of d_type's block (256 for IQ1_S), 0 <= f0 < f1 <= ff.
ExpertShare slice_experts(const uint8_t* gate, const uint8_t* up, const uint8_t* down, int experts, int latent, int ff,
                          int gu_type, int d_type, int f0, int f1);

}  // namespace strata::muse
