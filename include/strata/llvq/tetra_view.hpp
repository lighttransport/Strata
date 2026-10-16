// Fills core::TetraTablesView (and the base-5 row-position table) from the host Tetra tables, built the way
// llvq's Encoder::with_rule builds them (middle-byte sets by first appearance, rank table, fallbacks by
// (norm, point) minimum).
#pragma once
#include "strata/llvq/tetra.hpp"
#include "strata/llvq/tetra_encode_core.hpp"
#include <cstring>

namespace strata::llvq {

inline core::Closed closed_of(const Bound& b) { return {b.cost, core::lex(pack(b.cut))}; }

inline void make_view(const Tetra& t, core::TetraTablesView& v, std::vector<int16_t>& pos5) {
    std::memset(&v, 0, sizeof v);
    for (int j = 0; j < DIM; ++j) v.order[j] = t.order[j];
    for (int s = 0; s < STATES; ++s) {
        v.prefixes[s][0] = t.prefixes[s][0]; v.prefixes[s][1] = t.prefixes[s][1];
        v.suffixes[s][0] = t.suffixes[s][0]; v.suffixes[s][1] = t.suffixes[s][1];
        for (int b = 0; b < BRANCHES; ++b) { v.br_byte[s][b] = t.branches[s][b].first; v.br_s16[s][b] = t.branches[s][b].second; }
    }
    int nsets = 0;
    std::array<std::array<uint8_t, BRANCHES>, core::MSETS> msets{};
    for (int s8 = 0; s8 < STATES; ++s8) {
        std::array<uint8_t, BRANCHES> m{};
        for (int b = 0; b < BRANCHES; ++b) m[b] = t.branches[s8][b].first;
        std::sort(m.begin(), m.end());
        int mi = -1;
        for (int i = 0; i < nsets; ++i) if (msets[i] == m) mi = i;
        if (mi < 0) {
            if (nsets == core::MSETS) throw std::runtime_error("llvq view: more than 8 middle-byte sets");
            msets[nsets] = m;
            mi = nsets++;
        }
        for (int b2 = 0; b2 < BRANCHES; ++b2)
            v.branch_ix[s8][b2] = uint16_t(mi * BRANCHES + int(std::find(m.begin(), m.end(), t.branches[s8][b2].first) - m.begin()));
    }
    if (nsets != core::MSETS) throw std::runtime_error("llvq view: not 8 middle-byte sets");
    for (int i = 0; i < core::MSETS; ++i) std::memcpy(v.msets[i], msets[i].data(), BRANCHES);
    for (int o = 0; o < 4; ++o)
        for (int i = 0; i < 2 * core::K_REACH + 1; ++i) {
            auto r = rank_of(uint32_t(o), o + 4 * (i - core::K_REACH));
            v.ranks[o][i] = r ? uint8_t(*r) : core::RANK_NONE;
        }
    v.class_b[0] = closed_of(CLASS_BOUNDS[0]);
    v.class_b[1] = closed_of(CLASS_BOUNDS[1]);
    v.mixed = closed_of(MIXED_BOUND);
    int n = 0;
    for (int kind = 0; kind < 2; ++kind)
        for (uint32_t p = 0; p < 2; ++p)
            for (int r = 0; r < 2; ++r)
                for (int state = 0; state < STATES; ++state) {
                    const auto& patterns = kind == 0 ? t.prefixes[state] : t.suffixes[state];
                    bool have = false;
                    int64_t bn = 0;
                    std::array<int32_t, SECTION> by{};
                    uint8_t bc = 0;
                    uint32_t brow = 0;
                    for (uint8_t c : patterns)
                        for (int i = 0; i < CLASS_ROWS; ++i) {
                            uint32_t row = t.rows[CLASS_ROWS * r + i];
                            auto y = section_point(p, c, row);
                            int64_t nn = 0;
                            for (int32_t val : y) nn += int64_t(val) * val;
                            if (!have || nn < bn || (nn == bn && y < by)) { have = true; bn = nn; by = y; bc = c; brow = row; }
                        }
                    auto& f = v.fallback[n++];
                    f.byte = bc;
                    f.key = brow;
                    for (int j = 0; j < SECTION; ++j) f.y[j] = by[j];
                }
    pos5.assign(390625, -1);
    for (int i = 0; i < ROWS; ++i) pos5[core::base5(t.rows[i])] = int16_t(i);
}

}  // namespace strata::llvq
