// Tetra: the 48-bit Leech-lattice word of llvq (github.com/pjmalandrino/llvq @ ddecad31, MIT OR Apache-2.0), ported to
// C++ bit for bit: the Golay code, its 64-state trellis over the trio order, the 4,096-row rank table, the word
// fields, decode (word -> lattice point) and the inverse map (point -> word). Every table is re-derived at
// construction and checked against llvq's pinned constants; `fingerprint()` equals llvq-artifact's
// `tetra_fingerprint()` (0x9c3060086c7a13e6) when the port is exact.
//
// Sources: llvq-core/src/golay.rs, llvq-search/src/tetra/{mod,trellis,rank,word}.rs, llvq-artifact/src/codebook.rs.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::llvq {

constexpr int DIM = 24;
constexpr int SECTION = 8;
constexpr int STATES = 64;
constexpr int BRANCHES = 16;
constexpr int CLASS_ROWS = 2048;
constexpr int ROWS = 2 * CLASS_ROWS;
constexpr int N0_MIXED = 1240;
constexpr uint32_t MAX_RANK = 4;
constexpr uint32_t WORD_BITS = 48;
constexpr uint32_t LABEL_BITS = 47;
constexpr uint64_t LABEL_MASK = (uint64_t(1) << LABEL_BITS) - 1;
constexpr std::array<uint32_t, 3> TRIO = {0x0000149f, 0x000f6840, 0x00f08320};
constexpr std::array<uint32_t, 12> LINEAR_COLUMNS = {0x2d002e, 0x3a005a, 0x740033, 0x03061e, 0x050963, 0x090578,
                                                     0x0000ff, 0x2d1d00, 0x3a2b00, 0x744700, 0x638e00, 0xff0000};

// The residue progressions val(o, rho) for o = p + 2c (rank.rs `progressions`).
constexpr std::array<std::array<int32_t, 8>, 4> VALUES = {{
    {0, 4, -4, 8, -8, 12, -12, 16},
    {1, -3, 5, -7, 9, -11, 13, -15},
    {2, -2, 6, -6, 10, -10, 14, -14},
    {-1, 3, -5, 7, -9, 11, -13, 15},
}};
inline int32_t val(uint32_t o, uint32_t rho) { return VALUES[o][rho]; }
inline std::optional<uint32_t> rank_of(uint32_t o, int32_t y) {
    for (uint32_t r = 0; r < 8; ++r)
        if (VALUES[o][r] == y) return r;
    return std::nullopt;
}
using Rho = std::array<uint32_t, SECTION>;
inline uint32_t rank_class(const Rho& rho) {
    uint32_t a = 0;
    for (uint32_t r : rho) {
        int32_t q = val(0, r) / 4;  // Rust `/` truncates; rem_euclid(2) of q
        a ^= uint32_t(((q % 2) + 2) % 2);
    }
    return a;
}
inline uint32_t cost(const Rho& rho) {
    uint32_t c = 0;
    for (uint32_t r : rho) c += (2 * r + 1) * (2 * r + 1);
    return c;
}
inline uint32_t pack(const Rho& rho) {
    uint32_t a = 0;
    for (int j = 0; j < SECTION; ++j) a |= (rho[j] & 15u) << (4 * j);
    return a;
}
inline Rho unpack(uint32_t row) {
    Rho r{};
    for (int j = 0; j < SECTION; ++j) r[j] = (row >> (4 * j)) & 15u;
    return r;
}
struct Bound {
    uint32_t cost;
    Rho cut;
    bool contains(const Rho& rho) const {
        uint32_t c = llvq::cost(rho);
        return c < cost || (c == cost && rho <= cut);  // lexicographic, rho_0 first
    }
    bool operator==(const Bound&) const = default;
};
inline const std::array<Bound, 2> CLASS_BOUNDS = {Bound{88, {1, 1, 0, 3, 0, 0, 1, 1}}, Bound{96, {0, 0, 2, 2, 2, 0, 1, 1}}};
inline const Bound MIXED_BOUND = {72, {3, 0, 0, 0, 1, 0, 1, 0}};

// Word fields, LSB first (word.rs).
struct Fields {
    uint32_t p = 0, r = 0, s8 = 0, b1 = 0, i1 = 0, b2 = 0, i2 = 0, b3 = 0, i3 = 0, gain = 0;
    static Fields split(uint64_t w) {
        auto g = [&](int lo, int width) { return uint32_t((w >> lo) & ((uint64_t(1) << width) - 1)); };
        return {g(0, 1), g(1, 1), g(2, 6), g(8, 1), g(9, 11), g(20, 4), g(24, 11), g(35, 1), g(36, 11), g(47, 1)};
    }
    uint64_t join() const {
        return uint64_t(p) | uint64_t(r) << 1 | uint64_t(s8) << 2 | uint64_t(b1) << 8 | uint64_t(i1) << 9 |
               uint64_t(b2) << 20 | uint64_t(i2) << 24 | uint64_t(b3) << 35 | uint64_t(i3) << 36 | uint64_t(gain) << 47;
    }
};

inline uint64_t clmul(uint64_t a, uint64_t b) {
    uint64_t acc = 0;
    while (b) {
        if (b & 1) acc ^= a;
        a <<= 1;
        b >>= 1;
    }
    return acc;
}

// The 4,096 Golay codewords sorted by (weight, value) (golay.rs).
inline std::vector<uint32_t> golay_codewords() {
    std::vector<uint32_t> flat(4096);
    for (uint32_t m = 0; m < 4096; ++m) {
        uint32_t c = uint32_t(clmul(m, 0xC75));
        flat[m] = c | (uint32_t(std::popcount(c) & 1) << 23);
    }
    std::sort(flat.begin(), flat.end(), [](uint32_t a, uint32_t b) {
        int pa = std::popcount(a), pb = std::popcount(b);
        return pa != pb ? pa < pb : a < b;
    });
    return flat;
}

class Tetra {
public:
    std::array<uint32_t, DIM> order{};                                 // order[j]: natural coordinate at trio slot j
    std::array<std::array<uint8_t, 2>, STATES> prefixes{}, suffixes{};
    std::array<std::array<std::pair<uint8_t, uint8_t>, BRANCHES>, STATES> branches{};  // (middle byte, s16), by byte
    std::array<uint8_t, 256> prefix_of{};                               // 2*s8 + b1, 0xff = none
    std::array<uint32_t, ROWS> rows{};
    std::vector<std::pair<uint32_t, uint16_t>> index;                   // (packed row, position), sorted

    Tetra() {
        build_trellis();
        build_rows();
    }

    // Word -> point in natural order (mod.rs decode / decode_trio_order).
    std::array<int32_t, DIM> decode(uint64_t word) const {
        Fields f = Fields::split(word);
        uint8_t c1 = prefixes[f.s8][f.b1];
        auto [c2, s16] = branches[f.s8][f.b2];
        uint8_t c3 = suffixes[s16][f.b3];
        uint32_t row1 = rows[CLASS_ROWS * f.r + f.i1];
        uint32_t row2, delta;
        if (f.i2 < uint32_t(N0_MIXED)) { row2 = rows[f.i2]; delta = 0; }
        else { row2 = rows[CLASS_ROWS + f.i2 - N0_MIXED]; delta = 1; }
        uint32_t r3 = (f.p ^ f.r ^ delta) & 1;
        uint32_t row3 = rows[CLASS_ROWS * r3 + f.i3];
        std::array<int32_t, DIM> natural{};
        const uint8_t cs[3] = {c1, c2, c3};
        const uint32_t rs[3] = {row1, row2, row3};
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < SECTION; ++j)
                natural[order[SECTION * k + j]] = val(f.p + 2 * ((cs[k] >> j) & 1), (rs[k] >> (4 * j)) & 15);
        return natural;
    }

    // Point (natural order) -> word with gain 0, or nullopt when the point is not a Tetra point (mod.rs encode).
    std::optional<uint64_t> encode_point(const std::array<int32_t, DIM>& point) const {
        auto rem2 = [](int32_t v) { return ((v % 2) + 2) % 2; };
        int32_t p = rem2(point[0]);
        for (int32_t v : point)
            if (rem2(v) != p) return std::nullopt;
        std::array<int32_t, DIM> y{};
        for (int j = 0; j < DIM; ++j) y[j] = point[order[j]];
        uint32_t c = 0;
        for (int j = 0; j < DIM; ++j) c |= uint32_t(rem2((y[j] - p) >> 1)) << j;
        uint8_t c1 = uint8_t(c), c2 = uint8_t(c >> 8), c3 = uint8_t(c >> 16);
        if (prefix_of[c1] == 0xff) return std::nullopt;
        uint32_t s8 = prefix_of[c1] / 2, b1 = prefix_of[c1] % 2;
        int b2 = -1;
        uint32_t s16 = 0;
        for (int b = 0; b < BRANCHES; ++b)
            if (branches[s8][b].first == c2) { b2 = b; s16 = branches[s8][b].second; }
        if (b2 < 0) return std::nullopt;
        int b3 = suffixes[s16][0] == c3 ? 0 : suffixes[s16][1] == c3 ? 1 : -1;
        if (b3 < 0) return std::nullopt;
        auto ranks = [&](int k, uint8_t byte) -> std::optional<Rho> {
            Rho rho{};
            for (int j = 0; j < SECTION; ++j) {
                auto r = rank_of(uint32_t(p) + 2 * ((byte >> j) & 1), y[SECTION * k + j]);
                if (!r) return std::nullopt;
                rho[j] = *r;
            }
            return rho;
        };
        auto r1 = ranks(0, c1), r2 = ranks(1, c2), r3v = ranks(2, c3);
        if (!r1 || !r2 || !r3v) return std::nullopt;
        auto ci1 = class_index(*r1), mi = mixed_index(*r2), ci3 = class_index(*r3v);
        if (!ci1 || !mi || !ci3) return std::nullopt;
        if (ci3->first != ((uint32_t(p) ^ ci1->first ^ mi->first) & 1)) return std::nullopt;
        Fields f{uint32_t(p), ci1->first, s8, b1, ci1->second, uint32_t(b2), mi->second, uint32_t(b3), ci3->second, 0};
        return f.join();
    }

    std::optional<size_t> position(const Rho& rho) const {
        for (uint32_t r : rho)
            if (r > MAX_RANK) return std::nullopt;
        uint32_t key = pack(rho);
        auto it = std::lower_bound(index.begin(), index.end(), std::make_pair(key, uint16_t(0)),
                                   [](auto& a, auto& b) { return a.first < b.first; });
        if (it == index.end() || it->first != key) return std::nullopt;
        return it->second;
    }
    std::optional<std::pair<uint32_t, uint32_t>> class_index(const Rho& rho) const {
        auto p = position(rho);
        if (!p) return std::nullopt;
        return std::make_pair(uint32_t(*p / CLASS_ROWS), uint32_t(*p % CLASS_ROWS));
    }
    std::optional<std::pair<uint32_t, uint32_t>> mixed_index(const Rho& rho) const {
        auto ci = class_index(rho);
        if (!ci) return std::nullopt;
        uint32_t n = ci->first == 0 ? N0_MIXED : CLASS_ROWS - N0_MIXED;
        if (ci->second >= n) return std::nullopt;
        return std::make_pair(ci->first, ci->second + ci->first * N0_MIXED);
    }

    // llvq-artifact compute_tetra: FNV-1a 64 over the map's definition and 66 decodes.
    uint64_t fingerprint() const {
        uint64_t h = 0xcbf29ce484222325ull;
        auto byte = [&](uint8_t b) { h ^= b; h *= 0x00000100000001b3ull; };
        auto u64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) byte(uint8_t(v >> (8 * i))); };
        auto i32 = [&](int32_t v) { for (int i = 0; i < 4; ++i) byte(uint8_t(uint32_t(v) >> (8 * i))); };
        for (char ch : std::string("llvq codebook fingerprint tetra v1")) byte(uint8_t(ch));
        u64(DIM); u64(WORD_BITS); u64(LABEL_BITS); u64(N0_MIXED);
        for (uint32_t o : TRIO) u64(o);
        u64(rows.size());
        for (uint32_t r : rows) u64(r);
        u64(LINEAR_COLUMNS.size());
        for (uint32_t c : LINEAR_COLUMNS) u64(c);
        auto probe = [&](uint64_t w) { u64(w); for (int32_t v : decode(w)) i32(v); };
        auto mix = [](uint64_t z) {
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            return z ^ (z >> 31);
        };
        probe(0);
        probe(LABEL_MASK);
        for (uint64_t k = 0; k < 64; ++k) probe(mix(1 + k) & LABEL_MASK);
        return h;
    }
    static constexpr uint64_t PUBLISHED_FINGERPRINT = 0x9c3060086c7a13e6ull;

private:
    static uint32_t pivot(uint32_t b) { return uint32_t(1) << (31 - std::countl_zero(b)); }
    static uint32_t reduce(uint32_t w, const std::vector<uint32_t>& basis) {
        for (uint32_t b : basis)
            if (w & pivot(b)) w ^= b;
        return w;
    }
    static std::vector<uint32_t> echelon(const std::vector<uint32_t>& vectors) {
        std::vector<uint32_t> basis;
        for (uint32_t v : vectors) {
            v = reduce(v, basis);
            if (!v) continue;
            for (uint32_t& b : basis)
                if (b & pivot(v)) b ^= v;
            basis.push_back(v);
        }
        return basis;
    }
    static void need(bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("llvq tetra: ") + what);
    }

    void build_trellis() {
        auto g = golay_codewords();
        need((TRIO[0] & TRIO[1] | TRIO[0] & TRIO[2] | TRIO[1] & TRIO[2]) == 0 && (TRIO[0] | TRIO[1] | TRIO[2]) == 0xffffff,
             "trio is not a partition");
        int slot = 0;
        for (uint32_t octad : TRIO)
            for (uint32_t i = 0; i < 24; ++i)
                if (octad >> i & 1) order[slot++] = i;
        std::vector<uint32_t> words(g.size());
        for (size_t n = 0; n < g.size(); ++n) {
            uint32_t acc = 0;
            for (int j = 0; j < 24; ++j) acc |= (g[n] >> order[j] & 1) << j;
            words[n] = acc;
        }
        auto states = [&](uint32_t cut) {
            uint32_t low = (uint32_t(1) << cut) - 1;
            std::vector<uint32_t> side;
            for (uint32_t w : words)
                if ((w & low) == 0 || (w & ~low) == 0) side.push_back(w);
            auto basis = echelon(side);
            need(basis.size() == 6, "cut basis is not 6-dimensional");
            std::vector<uint32_t> reps(words.size());
            for (size_t n = 0; n < words.size(); ++n) reps[n] = reduce(words[n], basis);
            auto distinct = reps;
            std::sort(distinct.begin(), distinct.end());
            distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
            need(distinct.size() == STATES, "a cut does not carry 64 states");
            std::vector<uint8_t> out(words.size());
            for (size_t n = 0; n < words.size(); ++n)
                out[n] = uint8_t(std::lower_bound(distinct.begin(), distinct.end(), reps[n]) - distinct.begin());
            return out;
        };
        auto s8 = states(8), s16 = states(16);
        std::vector<std::vector<uint8_t>> pre(STATES), suf(STATES);
        std::vector<std::vector<std::pair<uint8_t, uint8_t>>> edges(STATES);
        for (size_t n = 0; n < words.size(); ++n) {
            uint32_t w = words[n];
            pre[s8[n]].push_back(uint8_t(w));
            suf[s16[n]].push_back(uint8_t(w >> 16));
            edges[s8[n]].push_back({uint8_t(w >> 8), s16[n]});
        }
        prefix_of.fill(0xff);
        for (int s = 0; s < STATES; ++s) {
            for (auto* v : {&pre[s], &suf[s]}) {
                std::sort(v->begin(), v->end());
                v->erase(std::unique(v->begin(), v->end()), v->end());
                need(v->size() == 2 && ((*v)[0] ^ (*v)[1]) == 0xff, "prefix/suffix pair");
            }
            prefixes[s] = {pre[s][0], pre[s][1]};
            suffixes[s] = {suf[s][0], suf[s][1]};
            for (int b1 = 0; b1 < 2; ++b1) {
                need(prefix_of[prefixes[s][b1]] == 0xff, "prefix byte opens two states");
                prefix_of[prefixes[s][b1]] = uint8_t(2 * s + b1);
            }
            auto& e = edges[s];
            std::sort(e.begin(), e.end());
            e.erase(std::unique(e.begin(), e.end()), e.end());
            need(e.size() == BRANCHES, "a state does not have 16 branches");
            for (int b = 0; b < BRANCHES; ++b) branches[s][b] = e[b];
        }
        // The pinned linear form of the trellis map.
        auto path = [&](uint32_t x) {
            uint32_t s = x & 63, b1 = x >> 6 & 1, b2 = x >> 7 & 15, b3 = x >> 11 & 1;
            auto [c2, t] = branches[s][b2];
            return uint32_t(prefixes[s][b1]) | uint32_t(c2) << 8 | uint32_t(suffixes[t][b3]) << 16;
        };
        for (int i = 0; i < 12; ++i) need(path(uint32_t(1) << i) == LINEAR_COLUMNS[i], "linear columns differ from llvq");
    }

    void build_rows() {
        std::vector<Rho> all;
        Rho rho{};
        auto under_cap = [&](auto&& self, int j, uint32_t spent) -> void {
            if (j == SECTION) { all.push_back(rho); return; }
            uint32_t open = uint32_t(SECTION - 1 - j);
            for (uint32_t r = 0; spent + (2 * r + 1) * (2 * r + 1) + open < 128; ++r) {
                rho[j] = r;
                self(self, j + 1, spent + (2 * r + 1) * (2 * r + 1));
            }
        };
        under_cap(under_cap, 0, 0);
        std::sort(all.begin(), all.end(), [](const Rho& a, const Rho& b) {
            uint32_t ca = cost(a), cb = cost(b);
            return ca != cb ? ca < cb : a < b;
        });
        std::array<std::vector<Rho>, 2> classes;
        for (const Rho& r : all) {
            auto& c = classes[rank_class(r)];
            if (c.size() < size_t(CLASS_ROWS)) c.push_back(r);
        }
        for (int c = 0; c < 2; ++c) {
            need(classes[c].size() == size_t(CLASS_ROWS), "class row count");
            Bound derived{cost(classes[c].back()), classes[c].back()};
            need(derived == CLASS_BOUNDS[c], "class bound differs from llvq");
        }
        int n0 = 0;
        for (int i = 0; i < CLASS_ROWS; ++i) n0 += rank_class(all[i]) == 0;
        need(n0 == N0_MIXED, "mixed split differs from llvq");
        need((Bound{cost(all[CLASS_ROWS - 1]), all[CLASS_ROWS - 1]}) == MIXED_BOUND, "mixed bound differs from llvq");
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < CLASS_ROWS; ++i) rows[CLASS_ROWS * c + i] = pack(classes[c][i]);
        index.resize(ROWS);
        for (int i = 0; i < ROWS; ++i) index[i] = {rows[i], uint16_t(i)};
        std::sort(index.begin(), index.end());
    }
};

// One section's point from its parity, pattern byte and packed row (encoder.rs section_point).
inline std::array<int32_t, SECTION> section_point(uint32_t p, uint8_t c, uint32_t row) {
    std::array<int32_t, SECTION> y{};
    for (int j = 0; j < SECTION; ++j) y[j] = val(p + 2 * ((c >> j) & 1), (row >> (4 * j)) & 15);
    return y;
}

}  // namespace strata::llvq
