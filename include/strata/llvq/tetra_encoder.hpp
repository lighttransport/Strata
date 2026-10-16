// The production Tetra encoder of llvq (llvq-search/src/tetra/encoder.rs @ ddecad31), ported for bit-identical
// words: f64 throughout, the same operation order, the same strict comparisons and loop orders. Compile without
// floating-point contraction (-ffp-contract=off) and without -ffast-math. One Encoder (tables) is shared by all
// threads; each thread owns a Scratch.
//
// Speed changes that keep the result identical: the row index is a direct base-5 table instead of a binary search
// (`position`); everything else is the reference algorithm.
#pragma once
#include "strata/llvq/tetra.hpp"
#include <cmath>
#include <limits>

namespace strata::llvq {

namespace enc_detail {
constexpr double SHRINKS[8] = {1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0};
constexpr int END_SHRINKS = 7, MID_SHRINKS = 8, WARM_JOINS = 2, MSETS = 8;
constexpr int END_ENTRIES = 2 * STATES, MID_ENTRIES = MSETS * BRANCHES;
constexpr int K_REACH = 8;
constexpr uint8_t RANK_NONE = 15;
constexpr double INF = std::numeric_limits<double>::infinity();

inline uint32_t lex(uint32_t key) {
    uint32_t swapped = ((key & 0x0f0f0f0fu) << 4) | ((key >> 4) & 0x0f0f0f0fu);
    return __builtin_bswap32(swapped);
}
struct Closed {
    uint32_t cost = 0, cut_lex = 0;
    static Closed of(const Bound& b) { return {b.cost, lex(pack(b.cut))}; }
    bool holds(uint32_t c, uint32_t key) const { return c < cost || (c == cost && lex(key) <= cut_lex); }
};
struct Coord {
    int32_t k0 = 0, dir = 0;
    double regret = 0;
    uint8_t rank[5] = {};
    uint16_t w[5] = {};
    double d[5] = {};
};
struct Tables {
    Coord t[8][SECTION][2];
    void build(const double* target, uint32_t p, const uint8_t (&ranks)[4][2 * K_REACH + 1]) {
        for (int layer = 0; layer < 8; ++layer) {
            double shrink = SHRINKS[layer];
            for (int j = 0; j < SECTION; ++j)
                for (int cbit = 0; cbit < 2; ++cbit) {
                    Coord& e = t[layer][j][cbit];
                    uint32_t o = p + 2 * uint32_t(cbit);
                    double z = (target[j] * shrink - double(o)) / 4.0;
                    z = std::min(std::max(z, -1e6), 1e6);
                    int32_t k0 = int32_t(std::round(z));
                    e.k0 = k0;
                    e.dir = z > double(k0) ? 1 : -1;
                    e.regret = std::fabs(z - double(k0));
                    for (int m = 0; m < 5; ++m) {
                        int32_t k = k0 + (m - 2);
                        uint8_t r = (k >= -K_REACH && k <= K_REACH) ? ranks[o][k + K_REACH] : RANK_NONE;
                        e.rank[m] = r;
                        e.w[m] = uint16_t((2 * uint32_t(r) + 1) * (2 * uint32_t(r) + 1));
                        double diff = double(int32_t(o) + 4 * k) - target[j];
                        e.d[m] = diff * diff;
                    }
                }
        }
    }
};
struct Entry {
    double value = INF;
    bool exact = true;
    uint8_t byte = 0;
    uint32_t key = 0;
};
inline constexpr Entry INFEASIBLE{};
struct Fallback {
    uint8_t byte;
    uint32_t key;
    std::array<int32_t, SECTION> y;
};
struct Path {
    double cost = INF;
    int i1 = 0, i2 = 0, delta = 0, i3 = 0;
};
struct Winner {
    double cost;
    uint32_t p, r;
    int s8, s16;
    uint32_t delta;
    uint8_t byte[3];
    uint32_t key[3];
};
inline double dist_to(const std::array<int32_t, SECTION>& y, const double* target) {
    double s = 0.0;
    for (int j = 0; j < SECTION; ++j) {
        double d = double(y[j]) - target[j];
        s += d * d;
    }
    return s;
}
}  // namespace enc_detail

struct TetraCode {
    uint64_t word = 0;
    std::array<int32_t, DIM> point{};
    double t = -std::numeric_limits<double>::infinity();
};

struct Scratch {
    enc_detail::Tables tabs[3];
    enc_detail::Entry e1[enc_detail::END_ENTRIES], e3[enc_detail::END_ENTRIES];
    enc_detail::Entry e2[enc_detail::MID_ENTRIES][2];
    double b1[enc_detail::END_ENTRIES], b3[enc_detail::END_ENTRIES], b2[enc_detail::MID_ENTRIES][2];
};

class Encoder {
public:
    static constexpr double ALPHA = 0.3218, RATIO = 1.14;

    explicit Encoder(const Tetra& t) : order(t.order), prefixes(t.prefixes), suffixes(t.suffixes), branches(t.branches) {
        using namespace enc_detail;
        int nsets = 0;
        for (int s8 = 0; s8 < STATES; ++s8) {
            std::array<uint8_t, BRANCHES> m{};
            for (int b = 0; b < BRANCHES; ++b) m[b] = branches[s8][b].first;
            std::sort(m.begin(), m.end());
            int mi = -1;
            for (int i = 0; i < nsets; ++i)
                if (msets[i] == m) mi = i;
            if (mi < 0) {
                if (nsets == MSETS) throw std::runtime_error("llvq encoder: more than 8 middle-byte sets");
                msets[nsets] = m;
                mi = nsets++;
            }
            for (int b2 = 0; b2 < BRANCHES; ++b2) {
                int k = int(std::find(m.begin(), m.end(), branches[s8][b2].first) - m.begin());
                branch_ix[s8][b2] = uint16_t(mi * BRANCHES + k);
            }
        }
        if (nsets != MSETS) throw std::runtime_error("llvq encoder: not 8 middle-byte sets");
        for (int o = 0; o < 4; ++o)
            for (int i = 0; i < 2 * K_REACH + 1; ++i) {
                auto r = rank_of(uint32_t(o), o + 4 * (i - K_REACH));
                ranks[o][i] = r ? uint8_t(*r) : RANK_NONE;
            }
        // Direct base-5 position table over ranks 0..4 (5^8 keys), equal to the binary search of the reference.
        pos5.assign(390625, -1);
        for (int i = 0; i < ROWS; ++i) pos5[base5(t.rows[i])] = int16_t(i);
        class_bound[0] = Closed::of(CLASS_BOUNDS[0]);
        class_bound[1] = Closed::of(CLASS_BOUNDS[1]);
        mixed = Closed::of(MIXED_BOUND);
        fallback.reserve(2 * 2 * 2 * STATES);
        for (int kind = 0; kind < 2; ++kind)
            for (uint32_t p = 0; p < 2; ++p)
                for (int r = 0; r < 2; ++r)
                    for (int state = 0; state < STATES; ++state) {
                        const auto& patterns = kind == 0 ? prefixes[state] : suffixes[state];
                        bool have = false;
                        int64_t bn = 0;
                        std::array<int32_t, SECTION> by{};
                        uint8_t bc = 0;
                        uint32_t brow = 0;
                        for (uint8_t c : patterns)
                            for (int i = 0; i < CLASS_ROWS; ++i) {
                                uint32_t row = t.rows[CLASS_ROWS * r + i];
                                auto y = section_point(p, c, row);
                                int64_t n = 0;
                                for (int32_t v : y) n += int64_t(v) * v;
                                if (!have || n < bn || (n == bn && y < by)) { have = true; bn = n; by = y; bc = c; brow = row; }
                            }
                        fallback.push_back({bc, brow, by});
                    }
    }

    // The production call (Encoder::encode): two adaptive scales; the larger t wins, the lower scale keeps ties.
    TetraCode encode(const double* x, Scratch& sc) const {
        auto [m, s0] = lower_scale(x);
        if (m == 0.0) return TetraCode{};
        if (std::isnormal(s0)) return two_scales(x, s0, sc);
        double xs[DIM];
        for (int j = 0; j < DIM; ++j) xs[j] = x[j] / m;
        TetraCode code = two_scales(xs, lower_scale(xs).second, sc);
        code.t *= m;
        return code;
    }

    static std::pair<double, double> lower_scale(const double* x) {
        double m = 0.0;
        for (int j = 0; j < DIM; ++j) m = std::max(m, std::fabs(x[j]));
        if (m == 0.0) return {0.0, 0.0};
        double s = 0.0;
        for (int j = 0; j < DIM; ++j) {
            double v = x[j] / m;
            s += v * v;
        }
        double unit_norm = std::sqrt(s);
        return {m, ALPHA * m * unit_norm / std::sqrt(double(DIM))};
    }

    TetraCode encode_at_scale(const double* x, double s, Scratch& sc) const {
        using namespace enc_detail;
        double targets[3][SECTION];
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < SECTION; ++j) targets[k][j] = x[order[SECTION * k + j]] / s;
        bool have = false;
        Winner best{};
        for (uint32_t p = 0; p < 2; ++p) {
            double u0 = have ? best.cost : INF;
            Winner w{};
            if (solve_parity(p, targets, sc, u0, w))
                if (!have || w.cost < best.cost) { best = w; have = true; }
        }
        if (!have) throw std::runtime_error("llvq encoder: no feasible path");
        return code_of(x, best);
    }

    static double t_of(const double* x, const std::array<int32_t, DIM>& y) {
        double dot = 0.0, nn = 0.0;
        for (int j = 0; j < DIM; ++j) dot += x[j] * double(y[j]);
        for (int j = 0; j < DIM; ++j) nn += double(y[j]) * double(y[j]);
        return nn > 0.0 ? dot / std::sqrt(nn) : -std::numeric_limits<double>::infinity();
    }

private:
    std::array<uint32_t, DIM> order;
    std::array<std::array<uint8_t, 2>, STATES> prefixes, suffixes;
    std::array<std::array<std::pair<uint8_t, uint8_t>, BRANCHES>, STATES> branches;
    std::array<std::array<uint8_t, BRANCHES>, enc_detail::MSETS> msets{};
    uint16_t branch_ix[STATES][BRANCHES]{};
    enc_detail::Closed class_bound[2], mixed;
    uint8_t ranks[4][2 * enc_detail::K_REACH + 1]{};
    std::vector<enc_detail::Fallback> fallback;
    std::vector<int16_t> pos5;

    static uint32_t base5(uint32_t key) {
        uint32_t v = 0, mul = 1;
        for (int j = 0; j < SECTION; ++j, mul *= 5) v += ((key >> (4 * j)) & 15u) * mul;
        return v;
    }
    size_t position(uint32_t key) const {
        int16_t p = pos5[base5(key)];
        if (p < 0) throw std::runtime_error("llvq encoder: member key is not a row");
        return size_t(p);
    }
    const enc_detail::Fallback& fb(int kind, uint32_t p, int r, int state) const {
        return fallback[((kind * 2 + int(p)) * 2 + r) * STATES + state];
    }

    // Both parities' shrink-1.0 bases of one pattern (base_pair). Output: (dist, member, key) per parity.
    struct Base { double d; bool m; uint32_t key; };
    static void base_pair(const enc_detail::Tables& tab, uint8_t pattern, const enc_detail::Closed* b0,
                          const enc_detail::Closed* b1, Base out[2]) {
        using namespace enc_detail;
        const Coord* e[SECTION];
        for (int j = 0; j < SECTION; ++j) e[j] = &tab.t[0][j][(pattern >> j) & 1];
        int j0 = 0;
        for (int j = 1; j < SECTION; ++j)
            if (e[j]->regret > e[j0]->regret) j0 = j;
        uint32_t c = 0, key = 0;
        double dist = 0.0;
        int32_t ksum = 0;
        for (int j = 0; j < SECTION; ++j) {
            c += e[j]->w[2];
            key |= uint32_t(e[j]->rank[2]) << (4 * j);
            dist += e[j]->d[2];
            ksum += e[j]->k0;
        }
        int par0 = ((ksum % 2) + 2) % 2;
        int f0 = 2 + e[j0]->dir;
        uint32_t cost1 = c - e[j0]->w[2] + e[j0]->w[f0];
        uint32_t key1 = (key & ~(0xfu << (4 * j0))) | (uint32_t(e[j0]->rank[f0]) << (4 * j0));
        double dist1 = dist - e[j0]->d[2] + e[j0]->d[f0];
        const Closed* bounds[2] = {b0, b1};
        out[par0] = {dist, bounds[par0]->holds(c, key), key};
        out[par0 ^ 1] = {dist1, bounds[par0 ^ 1]->holds(cost1, key1), key1};
    }

    // The candidate rule for one pattern and one k-parity over the first n_shrinks layers (solve).
    static bool solve(const enc_detail::Tables& tab, uint8_t pattern, uint32_t want, const enc_detail::Closed& bound,
                      int n_shrinks, double& out_d, uint32_t& out_key) {
        using namespace enc_detail;
        bool have = false;
        double best_d = INF;
        uint32_t best_key = 0;
        int cbits[SECTION];
        for (int j = 0; j < SECTION; ++j) cbits[j] = (pattern >> j) & 1;
        for (int si = 0; si < n_shrinks; ++si) {
            const Coord* e[SECTION];
            for (int j = 0; j < SECTION; ++j) e[j] = &tab.t[si][j][cbits[j]];
            int j0 = 0, j1 = 1;
            if (e[1]->regret > e[0]->regret) { j0 = 1; j1 = 0; }
            for (int j = 2; j < SECTION; ++j) {
                if (e[j]->regret > e[j0]->regret) { j1 = j0; j0 = j; }
                else if (e[j]->regret > e[j1]->regret) j1 = j;
            }
            int f[SECTION];
            for (int j = 0; j < SECTION; ++j) f[j] = 2;
            int32_t ksum = 0;
            for (int j = 0; j < SECTION; ++j) ksum += e[j]->k0;
            if (((ksum % 2) + 2) % 2 != int32_t(want)) f[j0] = 2 + e[j0]->dir;
            uint32_t c = 0, key = 0;
            double dist = 0.0;
            for (int j = 0; j < SECTION; ++j) {
                c += e[j]->w[f[j]];
                key |= uint32_t(e[j]->rank[f[j]]) << (4 * j);
                dist += e[j]->d[f[j]];
            }
            if (bound.holds(c, key) && dist < best_d) {
                best_d = dist;
                best_key = key;
                have = true;
                if (si == 0) break;
            }
            for (int j = 0; j < SECTION; ++j) {
                int j2 = j == j0 ? j1 : j0;
                int s2 = f[j2] == 2 ? e[j2]->dir : 2 - f[j2];
                int f2 = f[j2] + s2;
                for (int step : {-1, 1}) {
                    int fj = f[j] + step;
                    double d2 = dist - e[j]->d[f[j]] - e[j2]->d[f[j2]] + e[j]->d[fj] + e[j2]->d[f2];
                    if (d2 >= best_d) continue;
                    uint32_t c2 = c - e[j]->w[f[j]] - e[j2]->w[f[j2]] + e[j]->w[fj] + e[j2]->w[f2];
                    uint32_t mask = ~((0xfu << (4 * j)) | (0xfu << (4 * j2)));
                    uint32_t k2 = (key & mask) | (uint32_t(e[j]->rank[fj]) << (4 * j)) | (uint32_t(e[j2]->rank[f2]) << (4 * j2));
                    if (bound.holds(c2, k2)) { best_d = d2; best_key = k2; have = true; }
                }
            }
        }
        if (have) { out_d = best_d; out_key = best_key; }
        return have;
    }

    enc_detail::Entry end_resolve(const enc_detail::Tables& tab, const double* target, int kind, uint32_t p, int r,
                                  int state) const {
        using namespace enc_detail;
        const auto& patterns = kind == 0 ? prefixes[state] : suffixes[state];
        const Fallback& f = fb(kind, p, r, state);
        Entry e{dist_to(f.y, target), true, f.byte, f.key};
        for (uint8_t c : patterns) {
            double d;
            uint32_t key;
            if (solve(tab, c, uint32_t(r), class_bound[r], END_SHRINKS, d, key))
                if (d < e.value) e = Entry{d, true, c, key};
        }
        return e;
    }
    enc_detail::Entry mid_resolve(const enc_detail::Tables& tab, uint8_t byte, uint32_t delta) const {
        using namespace enc_detail;
        double d;
        uint32_t key;
        if (solve(tab, byte, delta, mixed, MID_SHRINKS, d, key)) return Entry{d, true, byte, key};
        return INFEASIBLE;
    }

    enc_detail::Path join(uint32_t p, const Scratch& sc) const {
        using namespace enc_detail;
        Path bp;
        for (int r = 0; r < 2; ++r)
            for (int s8 = 0; s8 < STATES; ++s8) {
                int i1 = r * STATES + s8;
                double a = sc.e1[i1].value;
                if (a >= bp.cost) continue;
                for (int b2 = 0; b2 < BRANCHES; ++b2) {
                    int s16 = branches[s8][b2].second;
                    int i2 = branch_ix[s8][b2];
                    for (int delta = 0; delta < 2; ++delta) {
                        int r_out = (int(p) ^ r ^ delta) & 1;
                        int i3 = r_out * STATES + s16;
                        double c = a + sc.e2[i2][delta].value + sc.e3[i3].value;
                        if (c < bp.cost) bp = Path{c, i1, i2, delta, i3};
                    }
                }
            }
        return bp;
    }

    bool solve_parity(uint32_t p, const double (&targets)[3][SECTION], Scratch& sc, double u0, enc_detail::Winner& out) const {
        using namespace enc_detail;
        for (int k = 0; k < 3; ++k) sc.tabs[k].build(targets[k], p, ranks);
        for (int kind = 0; kind < 2; ++kind)
            for (int st = 0; st < STATES; ++st) {
                const auto& patterns = kind == 0 ? prefixes[st] : suffixes[st];
                const Tables& tab = sc.tabs[2 * kind];
                const double* target = targets[2 * kind];
                Entry ent[2];
                for (int r = 0; r < 2; ++r) {
                    const Fallback& f = fb(kind, p, r, st);
                    ent[r] = Entry{dist_to(f.y, target), true, f.byte, f.key};
                }
                double lb[2] = {INF, INF};
                bool unresolved[2] = {false, false};
                for (uint8_t c : patterns) {
                    Base pair[2];
                    base_pair(tab, c, &class_bound[0], &class_bound[1], pair);
                    for (int r = 0; r < 2; ++r) {
                        if (pair[r].m) {
                            if (pair[r].d < ent[r].value) ent[r] = Entry{pair[r].d, true, c, pair[r].key};
                        } else {
                            lb[r] = std::min(lb[r], pair[r].d);
                            unresolved[r] = true;
                        }
                    }
                }
                Entry* outv = kind == 0 ? sc.e1 : sc.e3;
                for (int r = 0; r < 2; ++r) {
                    Entry e = ent[r];
                    if (unresolved[r] && lb[r] < ent[r].value) { e.value = lb[r]; e.exact = false; }
                    outv[r * STATES + st] = e;
                }
            }
        for (int m = 0; m < MSETS; ++m)
            for (int k = 0; k < BRANCHES; ++k) {
                uint8_t b = msets[m][k];
                Base pair[2];
                base_pair(sc.tabs[1], b, &mixed, &mixed, pair);
                for (int d = 0; d < 2; ++d) sc.e2[m * BRANCHES + k][d] = Entry{pair[d].d, pair[d].m, b, pair[d].key};
            }
        double u = u0;
        for (int w = 0; w < WARM_JOINS; ++w) {
            Path path = join(p, sc);
            bool all = true;
            if (!sc.e1[path.i1].exact) {
                sc.e1[path.i1] = end_resolve(sc.tabs[0], targets[0], 0, p, path.i1 / STATES, path.i1 % STATES);
                all = false;
            }
            if (!sc.e2[path.i2][path.delta].exact) {
                sc.e2[path.i2][path.delta] = mid_resolve(sc.tabs[1], msets[path.i2 / BRANCHES][path.i2 % BRANCHES], uint32_t(path.delta));
                all = false;
            }
            if (!sc.e3[path.i3].exact) {
                sc.e3[path.i3] = end_resolve(sc.tabs[2], targets[2], 1, p, path.i3 / STATES, path.i3 % STATES);
                all = false;
            }
            if (all) { u = std::min(u, path.cost); break; }
        }
        for (int i = 0; i < END_ENTRIES; ++i) { sc.b1[i] = INF; sc.b3[i] = INF; }
        for (int i = 0; i < MID_ENTRIES; ++i) { sc.b2[i][0] = INF; sc.b2[i][1] = INF; }
        for (int r = 0; r < 2; ++r)
            for (int s8 = 0; s8 < STATES; ++s8) {
                int i1 = r * STATES + s8;
                const Entry a = sc.e1[i1];
                for (int b2 = 0; b2 < BRANCHES; ++b2) {
                    int s16 = branches[s8][b2].second;
                    int i2 = branch_ix[s8][b2];
                    for (int delta = 0; delta < 2; ++delta) {
                        int r_out = (int(p) ^ r ^ delta) & 1;
                        int i3 = r_out * STATES + s16;
                        const Entry& v2 = sc.e2[i2][delta];
                        const Entry& v3 = sc.e3[i3];
                        double c = a.value + v2.value + v3.value;
                        sc.b1[i1] = std::min(sc.b1[i1], c);
                        sc.b2[i2][delta] = std::min(sc.b2[i2][delta], c);
                        sc.b3[i3] = std::min(sc.b3[i3], c);
                        if (a.exact && v2.exact && v3.exact && c < u) u = c;
                    }
                }
            }
        for (int i1 = 0; i1 < END_ENTRIES; ++i1)
            if (!sc.e1[i1].exact && sc.b1[i1] < u) sc.e1[i1] = end_resolve(sc.tabs[0], targets[0], 0, p, i1 / STATES, i1 % STATES);
        for (int i2 = 0; i2 < MID_ENTRIES; ++i2)
            for (int d = 0; d < 2; ++d)
                if (!sc.e2[i2][d].exact && sc.b2[i2][d] < u)
                    sc.e2[i2][d] = mid_resolve(sc.tabs[1], msets[i2 / BRANCHES][i2 % BRANCHES], uint32_t(d));
        for (int i3 = 0; i3 < END_ENTRIES; ++i3)
            if (!sc.e3[i3].exact && sc.b3[i3] < u) sc.e3[i3] = end_resolve(sc.tabs[2], targets[2], 1, p, i3 / STATES, i3 % STATES);
        for (auto& e : sc.e1) if (!e.exact) e = INFEASIBLE;
        for (auto& e : sc.e3) if (!e.exact) e = INFEASIBLE;
        for (auto& pr : sc.e2) for (auto& e : pr) if (!e.exact) e = INFEASIBLE;
        Path path = join(p, sc);
        if (!std::isfinite(path.cost)) return false;
        int r = path.i1 / STATES, s8 = path.i1 % STATES;
        const Entry mid = sc.e2[path.i2][path.delta];
        int b2 = -1;
        for (int b = 0; b < BRANCHES; ++b)
            if (branches[s8][b].first == mid.byte) { b2 = b; break; }
        if (b2 < 0) throw std::runtime_error("llvq encoder: middle byte is not a branch");
        const Entry e1 = sc.e1[path.i1], e3 = sc.e3[path.i3];
        out = Winner{path.cost, p, uint32_t(r), s8, int(branches[s8][b2].second), uint32_t(path.delta),
                     {e1.byte, mid.byte, e3.byte}, {e1.key, mid.key, e3.key}};
        return true;
    }

    TetraCode code_of(const double* x, const enc_detail::Winner& w) const {
        auto find2 = [](const std::array<uint8_t, 2>& a, uint8_t b) { return a[0] == b ? 0u : 1u; };
        uint32_t b1 = find2(prefixes[w.s8], w.byte[0]);
        uint32_t b2 = 0;
        for (int b = 0; b < BRANCHES; ++b)
            if (branches[w.s8][b].first == w.byte[1]) { b2 = uint32_t(b); break; }
        uint32_t b3 = find2(suffixes[w.s16], w.byte[2]);
        size_t pos1 = position(w.key[0]), pos2 = position(w.key[1]), pos3 = position(w.key[2]);
        uint32_t i2 = uint32_t(pos2 % CLASS_ROWS + w.delta * N0_MIXED);
        Fields f{w.p, w.r, uint32_t(w.s8), b1, uint32_t(pos1 % CLASS_ROWS), b2, i2, b3, uint32_t(pos3 % CLASS_ROWS), 0};
        TetraCode code;
        code.word = f.join();
        for (int k = 0; k < 3; ++k) {
            auto y = section_point(w.p, w.byte[k], w.key[k]);
            for (int j = 0; j < SECTION; ++j) code.point[order[SECTION * k + j]] = y[j];
        }
        code.t = t_of(x, code.point);
        return code;
    }

    TetraCode two_scales(const double* x, double s0, Scratch& sc) const {
        TetraCode a = encode_at_scale(x, s0, sc);
        TetraCode b = encode_at_scale(x, s0 * RATIO, sc);
        return b.t > a.t ? b : a;
    }
};

}  // namespace strata::llvq
