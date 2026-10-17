// Single-source Tetra encoder core for host and device (CUDA/HIP), the algorithm of llvq's Encoder::encode
// (tetra_encoder.hpp, itself a bit-exact port of llvq-search/src/tetra/encoder.rs) restructured for GPUs:
//   - no per-pass coordinate tables: a coordinate's (k0, dir, regret, rank, weight, distance) is recomputed where it
//     is read, with the same arithmetic, so the values (and the words) are identical;
//   - small read-only tables live in one struct (`TetraTablesView`) that a kernel can stage into shared memory.
// With Real = double and no floating-point contraction the words equal llvq's bit for bit (checked by
// tests/llvq/tetra_test.cpp); Real = float is a faster, format-compatible fallback (words may differ).
#pragma once
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLVQ_HD __host__ __device__ inline
#else
#define LLVQ_HD inline
#endif

namespace strata::llvq::core {

constexpr int DIM = 24, SECTION = 8, STATES = 64, BRANCHES = 16, MSETS = 8, CLASS_ROWS = 2048, N0_MIXED = 1240;
constexpr int END_ENTRIES = 2 * STATES, MID_ENTRIES = MSETS * BRANCHES, K_REACH = 8;
constexpr uint8_t RANK_NONE = 15;

struct Closed {
    uint32_t cost, cut_lex;
};
struct Fallback {
    uint8_t byte;
    uint32_t key;
    int32_t y[SECTION];
};
// Everything the encoder reads besides the block. Built on the host from strata::llvq::Encoder's tables.
struct TetraTablesView {
    uint32_t order[DIM];
    uint8_t prefixes[STATES][2], suffixes[STATES][2];
    uint8_t br_byte[STATES][BRANCHES], br_s16[STATES][BRANCHES];
    uint16_t branch_ix[STATES][BRANCHES];
    uint8_t msets[MSETS][BRANCHES];
    uint8_t ranks[4][2 * K_REACH + 1];
    Closed class_b[2], mixed;
    Fallback fallback[2 * 2 * 2 * STATES];
};

LLVQ_HD uint32_t lex(uint32_t key) {
    uint32_t s = ((key & 0x0f0f0f0fu) << 4) | ((key >> 4) & 0x0f0f0f0fu);
    return (s >> 24) | ((s >> 8) & 0xff00u) | ((s << 8) & 0xff0000u) | (s << 24);
}
LLVQ_HD bool holds(const Closed& b, uint32_t c, uint32_t key) { return c < b.cost || (c == b.cost && lex(key) <= b.cut_lex); }

template <class Real>
LLVQ_HD Real r_round(Real z);
template <>
LLVQ_HD double r_round<double>(double z) { return ::round(z); }
template <>
LLVQ_HD float r_round<float>(float z) { return ::roundf(z); }
template <class Real>
LLVQ_HD Real r_abs(Real v) { return v < Real(0) ? -v : v; }

template <class Real>
struct Big {
    static constexpr Real value = Real(1e300);
};
template <>
struct Big<float> {
    static constexpr float value = 3e38f;
};

// One coordinate at one shrink and offset (Tables::build's per-entry arithmetic).
template <class Real>
struct Coord {
    int32_t k0, dir;
    Real regret;
    uint8_t rank[5];
    uint16_t w[5];
    Real d[5];
};
LLVQ_HD double shrink_of(int layer) {
    switch (layer) {
    case 0: return 1.0;
    case 1: return 0.85;
    case 2: return 0.7;
    case 3: return 0.55;
    case 4: return 0.4;
    case 5: return 0.25;
    case 6: return 0.1;
    default: return 0.0;
    }
}

template <class Real>
LLVQ_HD void coord(const TetraTablesView& T, Real target, int layer, uint32_t o, Coord<Real>& e) {
    const Real shrink = Real(shrink_of(layer));
    Real z = (target * shrink - Real(o)) / Real(4);
    z = z < Real(-1e6) ? Real(-1e6) : (z > Real(1e6) ? Real(1e6) : z);
    int32_t k0 = int32_t(r_round<Real>(z));
    e.k0 = k0;
    e.dir = z > Real(k0) ? 1 : -1;
    e.regret = r_abs(z - Real(k0));
    for (int m = 0; m < 5; ++m) {
        int32_t k = k0 + (m - 2);
        uint8_t r = (k >= -K_REACH && k <= K_REACH) ? T.ranks[o][k + K_REACH] : RANK_NONE;
        e.rank[m] = r;
        e.w[m] = uint16_t((2 * uint32_t(r) + 1) * (2 * uint32_t(r) + 1));
        Real diff = Real(int32_t(o) + 4 * k) - target;
        e.d[m] = diff * diff;
    }
}

template <class Real>
struct Entry {
    Real value;
    bool exact;
    uint8_t byte;
    uint32_t key;
};
template <class Real>
struct Scratch {
    Entry<Real> e1[END_ENTRIES], e3[END_ENTRIES], e2[MID_ENTRIES][2];
    Real b1[END_ENTRIES], b3[END_ENTRIES], b2[MID_ENTRIES][2];
};
template <class Real>
struct Path {
    Real cost;
    int i1, i2, delta, i3;
};

template <class Real>
LLVQ_HD Real infinity() {
    return Real(1) / Real(0);
}

template <class Real>
LLVQ_HD Real dist_to(const int32_t* y, const Real* target) {
    Real s = Real(0);
    for (int j = 0; j < SECTION; ++j) {
        Real d = Real(y[j]) - target[j];
        s += d * d;
    }
    return s;
}

// base_pair: both parities' shrink-1.0 bases of one pattern. dist/member/key per parity.
template <class Real>
LLVQ_HD void base_pair(const TetraTablesView& T, const Real* target, uint32_t p, uint8_t pattern, const Closed& b0,
                       const Closed& b1, Real d_out[2], bool m_out[2], uint32_t key_out[2]) {
    Coord<Real> e[SECTION];
    for (int j = 0; j < SECTION; ++j) coord<Real>(T, target[j], 0, p + 2 * ((pattern >> j) & 1), e[j]);
    int j0 = 0;
    for (int j = 1; j < SECTION; ++j)
        if (e[j].regret > e[j0].regret) j0 = j;
    uint32_t c = 0, key = 0;
    Real dist = Real(0);
    int32_t ksum = 0;
    for (int j = 0; j < SECTION; ++j) {
        c += e[j].w[2];
        key |= uint32_t(e[j].rank[2]) << (4 * j);
        dist += e[j].d[2];
        ksum += e[j].k0;
    }
    int par0 = ((ksum % 2) + 2) % 2;
    int f0 = 2 + e[j0].dir;
    uint32_t cost1 = c - e[j0].w[2] + e[j0].w[f0];
    uint32_t key1 = (key & ~(0xfu << (4 * j0))) | (uint32_t(e[j0].rank[f0]) << (4 * j0));
    Real dist1 = dist - e[j0].d[2] + e[j0].d[f0];
    const Closed* bounds[2] = {&b0, &b1};
    d_out[par0] = dist; m_out[par0] = holds(*bounds[par0], c, key); key_out[par0] = key;
    d_out[par0 ^ 1] = dist1; m_out[par0 ^ 1] = holds(*bounds[par0 ^ 1], cost1, key1); key_out[par0 ^ 1] = key1;
}

// solve: the candidate rule for one pattern and one k-parity over n_shrinks layers.
template <class Real>
LLVQ_HD bool solve(const TetraTablesView& T, const Real* target, uint32_t p, uint8_t pattern, uint32_t want,
                   const Closed& bound, int n_shrinks, Real& out_d, uint32_t& out_key) {
    bool have = false;
    Real best_d = infinity<Real>();
    uint32_t best_key = 0;
    for (int si = 0; si < n_shrinks; ++si) {
        Coord<Real> e[SECTION];
        for (int j = 0; j < SECTION; ++j) coord<Real>(T, target[j], si, p + 2 * ((pattern >> j) & 1), e[j]);
        int j0 = 0, j1 = 1;
        if (e[1].regret > e[0].regret) { j0 = 1; j1 = 0; }
        for (int j = 2; j < SECTION; ++j) {
            if (e[j].regret > e[j0].regret) { j1 = j0; j0 = j; }
            else if (e[j].regret > e[j1].regret) j1 = j;
        }
        int f[SECTION];
        int32_t ksum = 0;
        for (int j = 0; j < SECTION; ++j) { f[j] = 2; ksum += e[j].k0; }
        if (((ksum % 2) + 2) % 2 != int32_t(want)) f[j0] = 2 + e[j0].dir;
        uint32_t c = 0, key = 0;
        Real dist = Real(0);
        for (int j = 0; j < SECTION; ++j) {
            c += e[j].w[f[j]];
            key |= uint32_t(e[j].rank[f[j]]) << (4 * j);
            dist += e[j].d[f[j]];
        }
        if (holds(bound, c, key) && dist < best_d) {
            best_d = dist;
            best_key = key;
            have = true;
            if (si == 0) break;
        }
        for (int j = 0; j < SECTION; ++j) {
            int j2 = j == j0 ? j1 : j0;
            int s2 = f[j2] == 2 ? e[j2].dir : 2 - f[j2];
            int f2 = f[j2] + s2;
            for (int step = -1; step <= 1; step += 2) {
                int fj = f[j] + step;
                Real d2 = dist - e[j].d[f[j]] - e[j2].d[f[j2]] + e[j].d[fj] + e[j2].d[f2];
                if (d2 >= best_d) continue;
                uint32_t c2 = c - e[j].w[f[j]] - e[j2].w[f[j2]] + e[j].w[fj] + e[j2].w[f2];
                uint32_t mask = ~((0xfu << (4 * j)) | (0xfu << (4 * j2)));
                uint32_t k2 = (key & mask) | (uint32_t(e[j].rank[fj]) << (4 * j)) | (uint32_t(e[j2].rank[f2]) << (4 * j2));
                if (holds(bound, c2, k2)) { best_d = d2; best_key = k2; have = true; }
            }
        }
    }
    if (have) { out_d = best_d; out_key = best_key; }
    return have;
}

template <class Real>
LLVQ_HD Entry<Real> end_resolve(const TetraTablesView& T, const Real* target, int kind, uint32_t p, int r, int state) {
    const uint8_t* patterns = kind == 0 ? T.prefixes[state] : T.suffixes[state];
    const Fallback& f = T.fallback[((kind * 2 + int(p)) * 2 + r) * STATES + state];
    Entry<Real> e{dist_to<Real>(f.y, target), true, f.byte, f.key};
    for (int i = 0; i < 2; ++i) {
        Real d;
        uint32_t key;
        if (solve<Real>(T, target, p, patterns[i], uint32_t(r), T.class_b[r], 7, d, key))
            if (d < e.value) e = Entry<Real>{d, true, patterns[i], key};
    }
    return e;
}
template <class Real>
LLVQ_HD Entry<Real> mid_resolve(const TetraTablesView& T, const Real* target, uint32_t p, uint8_t byte, uint32_t delta) {
    Real d;
    uint32_t key;
    if (solve<Real>(T, target, p, byte, delta, T.mixed, 8, d, key)) return Entry<Real>{d, true, byte, key};
    return Entry<Real>{infinity<Real>(), true, 0, 0};
}

template <class Real>
LLVQ_HD Path<Real> join(const TetraTablesView& T, uint32_t p, const Scratch<Real>& sc) {
    Path<Real> bp{infinity<Real>(), 0, 0, 0, 0};
    for (int r = 0; r < 2; ++r)
        for (int s8 = 0; s8 < STATES; ++s8) {
            int i1 = r * STATES + s8;
            Real a = sc.e1[i1].value;
            if (a >= bp.cost) continue;
            for (int b2 = 0; b2 < BRANCHES; ++b2) {
                int s16 = T.br_s16[s8][b2];
                int i2 = T.branch_ix[s8][b2];
                for (int delta = 0; delta < 2; ++delta) {
                    int i3 = ((int(p) ^ r ^ delta) & 1) * STATES + s16;
                    Real c = a + sc.e2[i2][delta].value + sc.e3[i3].value;
                    if (c < bp.cost) bp = Path<Real>{c, i1, i2, delta, i3};
                }
            }
        }
    return bp;
}

template <class Real>
struct Winner {
    Real cost;
    uint32_t p, r;
    int s8, s16;
    uint32_t delta;
    uint8_t byte[3];
    uint32_t key[3];
};

template <class Real>
LLVQ_HD bool solve_parity(const TetraTablesView& T, uint32_t p, const Real (&targets)[3][SECTION], Scratch<Real>& sc,
                          Real u0, Winner<Real>& out) {
    const Real INF = infinity<Real>();
    for (int kind = 0; kind < 2; ++kind)
        for (int st = 0; st < STATES; ++st) {
            const uint8_t* patterns = kind == 0 ? T.prefixes[st] : T.suffixes[st];
            const Real* target = targets[2 * kind];
            Entry<Real> ent[2];
            for (int r = 0; r < 2; ++r) {
                const Fallback& f = T.fallback[((kind * 2 + int(p)) * 2 + r) * STATES + st];
                ent[r] = Entry<Real>{dist_to<Real>(f.y, target), true, f.byte, f.key};
            }
            Real lb[2] = {INF, INF};
            bool unresolved[2] = {false, false};
            for (int i = 0; i < 2; ++i) {
                uint8_t c = patterns[i];
                Real d[2];
                bool m[2];
                uint32_t key[2];
                base_pair<Real>(T, target, p, c, T.class_b[0], T.class_b[1], d, m, key);
                for (int r = 0; r < 2; ++r) {
                    if (m[r]) {
                        if (d[r] < ent[r].value) ent[r] = Entry<Real>{d[r], true, c, key[r]};
                    } else {
                        lb[r] = lb[r] < d[r] ? lb[r] : d[r];
                        unresolved[r] = true;
                    }
                }
            }
            Entry<Real>* outv = kind == 0 ? sc.e1 : sc.e3;
            for (int r = 0; r < 2; ++r) {
                Entry<Real> e = ent[r];
                if (unresolved[r] && lb[r] < ent[r].value) { e.value = lb[r]; e.exact = false; }
                outv[r * STATES + st] = e;
            }
        }
    for (int m = 0; m < MSETS; ++m)
        for (int k = 0; k < BRANCHES; ++k) {
            uint8_t b = T.msets[m][k];
            Real d[2];
            bool mm[2];
            uint32_t key[2];
            base_pair<Real>(T, targets[1], p, b, T.mixed, T.mixed, d, mm, key);
            for (int dd = 0; dd < 2; ++dd) sc.e2[m * BRANCHES + k][dd] = Entry<Real>{d[dd], mm[dd], b, key[dd]};
        }
    Real u = u0;
    for (int w = 0; w < 2; ++w) {
        Path<Real> path = join<Real>(T, p, sc);
        bool all = true;
        if (!sc.e1[path.i1].exact) {
            sc.e1[path.i1] = end_resolve<Real>(T, targets[0], 0, p, path.i1 / STATES, path.i1 % STATES);
            all = false;
        }
        if (!sc.e2[path.i2][path.delta].exact) {
            sc.e2[path.i2][path.delta] = mid_resolve<Real>(T, targets[1], p, T.msets[path.i2 / BRANCHES][path.i2 % BRANCHES], uint32_t(path.delta));
            all = false;
        }
        if (!sc.e3[path.i3].exact) {
            sc.e3[path.i3] = end_resolve<Real>(T, targets[2], 1, p, path.i3 / STATES, path.i3 % STATES);
            all = false;
        }
        if (all) { u = u < path.cost ? u : path.cost; break; }
    }
    for (int i = 0; i < END_ENTRIES; ++i) { sc.b1[i] = INF; sc.b3[i] = INF; }
    for (int i = 0; i < MID_ENTRIES; ++i) { sc.b2[i][0] = INF; sc.b2[i][1] = INF; }
    for (int r = 0; r < 2; ++r)
        for (int s8 = 0; s8 < STATES; ++s8) {
            int i1 = r * STATES + s8;
            const Real av = sc.e1[i1].value;
            const bool ax = sc.e1[i1].exact;
            Real best1 = sc.b1[i1];
            for (int b2 = 0; b2 < BRANCHES; ++b2) {
                int s16 = T.br_s16[s8][b2];
                int i2 = T.branch_ix[s8][b2];
                for (int delta = 0; delta < 2; ++delta) {
                    int i3 = ((int(p) ^ r ^ delta) & 1) * STATES + s16;
                    const Entry<Real>& v2 = sc.e2[i2][delta];
                    const Entry<Real>& v3 = sc.e3[i3];
                    Real c = av + v2.value + v3.value;
                    best1 = best1 < c ? best1 : c;
                    sc.b2[i2][delta] = sc.b2[i2][delta] < c ? sc.b2[i2][delta] : c;
                    sc.b3[i3] = sc.b3[i3] < c ? sc.b3[i3] : c;
                    if (ax && v2.exact && v3.exact && c < u) u = c;
                }
            }
            sc.b1[i1] = best1;
        }
    for (int i1 = 0; i1 < END_ENTRIES; ++i1)
        if (!sc.e1[i1].exact && sc.b1[i1] < u) sc.e1[i1] = end_resolve<Real>(T, targets[0], 0, p, i1 / STATES, i1 % STATES);
    for (int i2 = 0; i2 < MID_ENTRIES; ++i2)
        for (int d = 0; d < 2; ++d)
            if (!sc.e2[i2][d].exact && sc.b2[i2][d] < u)
                sc.e2[i2][d] = mid_resolve<Real>(T, targets[1], p, T.msets[i2 / BRANCHES][i2 % BRANCHES], uint32_t(d));
    for (int i3 = 0; i3 < END_ENTRIES; ++i3)
        if (!sc.e3[i3].exact && sc.b3[i3] < u) sc.e3[i3] = end_resolve<Real>(T, targets[2], 1, p, i3 / STATES, i3 % STATES);
    for (int i = 0; i < END_ENTRIES; ++i) {
        if (!sc.e1[i].exact) sc.e1[i] = Entry<Real>{INF, true, 0, 0};
        if (!sc.e3[i].exact) sc.e3[i] = Entry<Real>{INF, true, 0, 0};
    }
    for (int i = 0; i < MID_ENTRIES; ++i)
        for (int d = 0; d < 2; ++d)
            if (!sc.e2[i][d].exact) sc.e2[i][d] = Entry<Real>{INF, true, 0, 0};
    Path<Real> path = join<Real>(T, p, sc);
    if (!(path.cost < INF)) return false;
    int r = path.i1 / STATES, s8 = path.i1 % STATES;
    const Entry<Real> mid = sc.e2[path.i2][path.delta];
    int b2 = 0;
    for (int b = 0; b < BRANCHES; ++b)
        if (T.br_byte[s8][b] == mid.byte) { b2 = b; break; }
    out = Winner<Real>{path.cost, p, uint32_t(r), s8, int(T.br_s16[s8][b2]), uint32_t(path.delta),
                       {sc.e1[path.i1].byte, mid.byte, sc.e3[path.i3].byte},
                       {sc.e1[path.i1].key, mid.key, sc.e3[path.i3].key}};
    return true;
}

// The section point of (p, byte, packed row) and the base-5 position index for the word.
LLVQ_HD int32_t val(uint32_t o, uint32_t rho) {
    const int32_t V[4][8] = {{0, 4, -4, 8, -8, 12, -12, 16}, {1, -3, 5, -7, 9, -11, 13, -15},
                             {2, -2, 6, -6, 10, -10, 14, -14}, {-1, 3, -5, 7, -9, 11, -13, 15}};
    return V[o][rho];
}
LLVQ_HD uint32_t base5(uint32_t key) {
    uint32_t v = 0, mul = 1;
    for (int j = 0; j < SECTION; ++j, mul *= 5) v += ((key >> (4 * j)) & 15u) * mul;
    return v;
}

struct Code {
    uint64_t word;
    double t;
};

// code_of + t_of. pos5: 5^8 table of row positions (-1 = not a row).
template <class Real>
LLVQ_HD Code code_of(const TetraTablesView& T, const int16_t* pos5, const Real* x, const Winner<Real>& w) {
    uint32_t b1 = T.prefixes[w.s8][0] == w.byte[0] ? 0u : 1u;
    uint32_t b2 = 0;
    for (int b = 0; b < BRANCHES; ++b)
        if (T.br_byte[w.s8][b] == w.byte[1]) { b2 = uint32_t(b); break; }
    uint32_t b3 = T.suffixes[w.s16][0] == w.byte[2] ? 0u : 1u;
    uint32_t pos1 = uint32_t(pos5[base5(w.key[0])]), pos2 = uint32_t(pos5[base5(w.key[1])]), pos3 = uint32_t(pos5[base5(w.key[2])]);
    uint32_t i2 = pos2 % CLASS_ROWS + w.delta * N0_MIXED;
    uint64_t word = uint64_t(w.p) | uint64_t(w.r) << 1 | uint64_t(w.s8) << 2 | uint64_t(b1) << 8 |
                    uint64_t(pos1 % CLASS_ROWS) << 9 | uint64_t(b2) << 20 | uint64_t(i2) << 24 | uint64_t(b3) << 35 |
                    uint64_t(pos3 % CLASS_ROWS) << 36;
    int32_t point[DIM];
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < SECTION; ++j)
            point[T.order[SECTION * k + j]] = val(w.p + 2 * ((w.byte[k] >> j) & 1), (w.key[k] >> (4 * j)) & 15u);
    Real dot = Real(0), nn = Real(0);
    for (int j = 0; j < DIM; ++j) dot += x[j] * Real(point[j]);
    for (int j = 0; j < DIM; ++j) nn += Real(point[j]) * Real(point[j]);
    Real t = nn > Real(0) ? dot / sqrt(nn) : -infinity<Real>();
    return Code{word, double(t)};
}

template <class Real>
LLVQ_HD Code encode_at_scale(const TetraTablesView& T, const int16_t* pos5, const Real* x, Real s, Scratch<Real>& sc) {
    Real targets[3][SECTION];
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < SECTION; ++j) targets[k][j] = x[T.order[SECTION * k + j]] / s;
    bool have = false;
    Winner<Real> best{};
    for (uint32_t p = 0; p < 2; ++p) {
        Real u0 = have ? best.cost : infinity<Real>();
        Winner<Real> w;
        if (solve_parity<Real>(T, p, targets, sc, u0, w))
            if (!have || w.cost < best.cost) { best = w; have = true; }
    }
    if (!have) return Code{0, -infinity<double>()};
    return code_of<Real>(T, pos5, x, best);
}

template <class Real>
LLVQ_HD Code two_scales(const TetraTablesView& T, const int16_t* pos5, const Real* x, Real s0, Scratch<Real>& sc) {
    Code a = encode_at_scale<Real>(T, pos5, x, s0, sc);
    Code b = encode_at_scale<Real>(T, pos5, x, s0 * Real(1.14), sc);
    return b.t > a.t ? b : a;
}

// Encoder::encode.
template <class Real>
LLVQ_HD Code encode(const TetraTablesView& T, const int16_t* pos5, const Real* x, Scratch<Real>& sc) {
    Real m = Real(0);
    for (int j = 0; j < DIM; ++j) { Real a = r_abs(x[j]); m = m > a ? m : a; }
    if (m == Real(0)) return Code{0, -infinity<double>()};
    Real s = Real(0);
    for (int j = 0; j < DIM; ++j) { Real v = x[j] / m; s += v * v; }
    Real unit_norm = sqrt(s);
    Real s0 = Real(0.3218) * m * unit_norm / sqrt(Real(DIM));
    bool normal = s0 == s0 && s0 < infinity<Real>() && s0 >= (sizeof(Real) == 8 ? Real(2.2250738585072014e-308) : Real(1.17549435e-38f));
    if (normal) return two_scales<Real>(T, pos5, x, s0, sc);
    Real xs[DIM];
    for (int j = 0; j < DIM; ++j) xs[j] = x[j] / m;
    Real m2 = Real(0);
    for (int j = 0; j < DIM; ++j) { Real a = r_abs(xs[j]); m2 = m2 > a ? m2 : a; }
    Real s2 = Real(0);
    for (int j = 0; j < DIM; ++j) { Real v = xs[j] / m2; s2 += v * v; }
    Real s0b = Real(0.3218) * m2 * sqrt(s2) / sqrt(Real(DIM));
    Code c = two_scales<Real>(T, pos5, xs, s0b, sc);
    c.t *= double(m);
    return c;
}

}  // namespace strata::llvq::core
