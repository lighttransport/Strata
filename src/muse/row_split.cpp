// src/muse/row_split.cpp - strata/muse/row_split.hpp
#include "strata/muse/row_split.hpp"

#include "ggml.h"

#include <cstring>
#include <stdexcept>

namespace strata::muse {

ExpertShare slice_experts(const uint8_t* gate, const uint8_t* up, const uint8_t* down, int experts, int latent, int ff,
                          int gu_type, int d_type, int f0, int f1) {
    const int64_t blk = ggml_blck_size((ggml_type) d_type);
    if (f0 < 0 || f1 > ff || f0 >= f1 || f0 % blk || f1 % blk) throw std::invalid_argument("muse row split: share is not whole blocks");
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, latent);
    const size_t d_full = ggml_row_size((ggml_type) d_type, ff), d_off = ggml_row_size((ggml_type) d_type, f0),
                 d_part = ggml_row_size((ggml_type) d_type, f1 - f0);
    ExpertShare s;
    s.f0 = f0;
    s.f1 = f1;
    const size_t rows = (size_t) (f1 - f0);
    s.gate.resize((size_t) experts * rows * gu_row);
    s.up.resize(s.gate.size());
    s.down.resize((size_t) experts * latent * d_part);
    for (int e = 0; e < experts; ++e) {
        const size_t src = ((size_t) e * ff + f0) * gu_row, dst = (size_t) e * rows * gu_row;
        std::memcpy(s.gate.data() + dst, gate + src, rows * gu_row);
        std::memcpy(s.up.data() + dst, up + src, rows * gu_row);
        for (int r = 0; r < latent; ++r)
            std::memcpy(s.down.data() + ((size_t) e * latent + r) * d_part, down + ((size_t) e * latent + r) * d_full + d_off, d_part);
    }
    return s;
}

}  // namespace strata::muse
