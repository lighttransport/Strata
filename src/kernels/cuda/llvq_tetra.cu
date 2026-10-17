// llvq Tetra encoder on the GPU (CUDA and HIP from one source). One 32-lane warp encodes one 24-weight block:
// the per-pass work of llvq's Encoder (encoder.rs) is spread over the lanes -- the coordinate table, the 512 base
// entries, the 4,096-path joins, the bound pass, and the resolves of the entries that pass (compacted into a list,
// one resolve per lane). Each step returns the same values as the sequential reference: joins reduce by (cost, path
// index) so the first minimum wins as before, and min() over non-NaN values does not depend on order.
// With f64 and no contraction (nvcc -fmad=false / hipcc -ffp-contract=off) the words and t are bit-identical to
// llvq's; f32 is a faster, format-compatible fallback.
#include "strata/kernels/llvq_tetra.hpp"
#include "strata/llvq/tetra_encode_core.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {
using namespace strata::llvq::core;

constexpr int LANES = 32;
#ifndef LLVQ_TETRA_WARPS
#if defined(__HIP_PLATFORM_AMD__)
#define LLVQ_TETRA_WARPS 4  // 4 x 12 KiB + tables in 64 KiB LDS (gfx1201: 112 K blocks/s vs 97 K at 2)
#else
#define LLVQ_TETRA_WARPS 3  // within CUDA's 48 KiB static shared memory
#endif
#endif
constexpr int WARPS = LLVQ_TETRA_WARPS;  // warps (blocks in flight) per CTA
constexpr int ENTRIES = 512;  // e1 [0,128), e3 [128,256), e2 [256,512) as 256 + 2*i2 + delta
constexpr int E3 = 128, E2 = 256;

struct WarpTables {
    uint32_t order[DIM];
    uint8_t prefixes[STATES][2], suffixes[STATES][2];
    uint8_t br_byte[STATES][BRANCHES], br_s16[STATES][BRANCHES];
    uint16_t branch_ix[STATES][BRANCHES];
    uint8_t msets[MSETS][BRANCHES];
    uint8_t ranks[4][2 * K_REACH + 1];
    Closed class_b[2], mixed;
    uint16_t inv2_start[MID_ENTRIES + 1], inv2[STATES * BRANCHES];  // (s8, b2) with branch_ix == i2
    uint16_t inv3_start[STATES + 1], inv3[STATES * BRANCHES];       // (s8, b2) with br_s16 == s16
    uint32_t pad;
};
static_assert(sizeof(WarpTables) % 4 == 0, "tables copy as 32-bit words");

template <class Real>
struct WarpState {
    Real z[3 * 8 * 8 * 2];  // [section][layer][j][cbit]
    Real tgt[DIM];          // section-major targets x[order[t]] / s
    Real x[DIM];
    Real val[ENTRIES];
    uint32_t key[ENTRIES];
    uint16_t flag[ENTRIES];  // byte | exact << 8
    uint16_t list[ENTRIES];
    Real sd[36];  // per-(pattern, layer) results of the cooperative warm resolves
    uint32_t skey[36];
    uint8_t sflag[36];  // have | base0 << 1
};

template <class S>
__device__ __forceinline__ void put(S& st, int idx, bool exact, uint8_t byte, uint32_t key) {
    st.key[idx] = key;
    st.flag[idx] = uint16_t(byte | uint16_t(exact) << 8);
}
template <class S>
__device__ __forceinline__ bool exact_at(const S& st, int idx) { return st.flag[idx] >> 8; }
template <class S>
__device__ __forceinline__ uint8_t byte_at(const S& st, int idx) { return uint8_t(st.flag[idx]); }

template <class Real>
struct CZ {
    int32_t k0, dir;
    Real regret, tgt;
    uint32_t o;
    __device__ void load(Real z, uint32_t o_, Real t) {
        k0 = int32_t(r_round<Real>(z));
        dir = z > Real(k0) ? 1 : -1;
        regret = r_abs(z - Real(k0));
        o = o_;
        tgt = t;
    }
    __device__ uint32_t rank(const WarpTables& T, int m) const {
        int32_t k = k0 + (m - 2);
        return (k >= -K_REACH && k <= K_REACH) ? T.ranks[o][k + K_REACH] : RANK_NONE;
    }
    __device__ uint32_t w(const WarpTables& T, int m) const {
        uint32_t r = rank(T, m);
        return (2 * r + 1) * (2 * r + 1);
    }
    __device__ Real d(int m) const {
        int32_t k = k0 + (m - 2);
        Real diff = Real(int32_t(o) + 4 * k) - tgt;
        return diff * diff;
    }
};

__device__ __forceinline__ int zidx(int k, int layer, int j, int cb) { return k * 128 + layer * 16 + j * 2 + cb; }

template <class Real>
__device__ void base_pair_w(const WarpTables& T, const WarpState<Real>& S, int k, uint32_t p, uint8_t pattern,
                            const Closed& b0, const Closed& b1, Real d_out[2], bool m_out[2], uint32_t key_out[2]) {
    CZ<Real> e[SECTION];
    for (int j = 0; j < SECTION; ++j) {
        int cb = (pattern >> j) & 1;
        e[j].load(S.z[zidx(k, 0, j, cb)], p + 2 * cb, S.tgt[SECTION * k + j]);
    }
    int j0 = 0;
    for (int j = 1; j < SECTION; ++j)
        if (e[j].regret > e[j0].regret) j0 = j;
    uint32_t c = 0, key = 0;
    Real dist = Real(0);
    int32_t ksum = 0;
    for (int j = 0; j < SECTION; ++j) {
        c += e[j].w(T, 2);
        key |= e[j].rank(T, 2) << (4 * j);
        dist += e[j].d(2);
        ksum += e[j].k0;
    }
    int par0 = ((ksum % 2) + 2) % 2;
    int f0 = 2 + e[j0].dir;
    uint32_t cost1 = c - e[j0].w(T, 2) + e[j0].w(T, f0);
    uint32_t key1 = (key & ~(0xfu << (4 * j0))) | (e[j0].rank(T, f0) << (4 * j0));
    Real dist1 = dist - e[j0].d(2) + e[j0].d(f0);
    const Closed& bp0 = par0 == 0 ? b0 : b1;
    const Closed& bp1 = par0 == 0 ? b1 : b0;
    d_out[par0] = dist; m_out[par0] = holds(bp0, c, key); key_out[par0] = key;
    d_out[par0 ^ 1] = dist1; m_out[par0 ^ 1] = holds(bp1, cost1, key1); key_out[par0 ^ 1] = key1;
}

template <class Real>
__device__ bool solve_w(const WarpTables& T, const WarpState<Real>& S, int k, uint32_t p, uint8_t pattern,
                        uint32_t want, const Closed& bound, int n_shrinks, Real& out_d, uint32_t& out_key) {
    bool have = false;
    Real best_d = infinity<Real>();
    uint32_t best_key = 0;
    for (int si = 0; si < n_shrinks; ++si) {
        CZ<Real> e[SECTION];
        for (int j = 0; j < SECTION; ++j) {
            int cb = (pattern >> j) & 1;
            e[j].load(S.z[zidx(k, si, j, cb)], p + 2 * cb, S.tgt[SECTION * k + j]);
        }
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
            c += e[j].w(T, f[j]);
            key |= e[j].rank(T, f[j]) << (4 * j);
            dist += e[j].d(f[j]);
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
            Real dj = e[j].d(f[j]), dj2 = e[j2].d(f[j2]), df2 = e[j2].d(f2);
            uint32_t wj = e[j].w(T, f[j]), wj2 = e[j2].w(T, f[j2]), wf2 = e[j2].w(T, f2);
            for (int step = -1; step <= 1; step += 2) {
                int fj = f[j] + step;
                Real d2 = dist - dj - dj2 + e[j].d(fj) + df2;
                if (d2 >= best_d) continue;
                uint32_t c2 = c - wj - wj2 + e[j].w(T, fj) + wf2;
                uint32_t mask = ~((0xfu << (4 * j)) | (0xfu << (4 * j2)));
                uint32_t k2 = (key & mask) | (e[j].rank(T, fj) << (4 * j)) | (e[j2].rank(T, f2) << (4 * j2));
                if (holds(bound, c2, k2)) { best_d = d2; best_key = k2; have = true; }
            }
        }
    }
    if (have) { out_d = best_d; out_key = best_key; }
    return have;
}

// One shrink layer of solve(): this layer's first minimum among member candidates. base0: layer 0's base was a
// member, which ends solve() with that candidate.
template <class Real>
struct LayerResult {
    Real d;
    uint32_t key;
    bool have, base0;
};
template <class Real>
__device__ LayerResult<Real> solve_layer(const WarpTables& T, const WarpState<Real>& S, int k, uint32_t p,
                                         uint8_t pattern, uint32_t want, const Closed& bound, int si) {
    LayerResult<Real> res{infinity<Real>(), 0, false, false};
    CZ<Real> e[SECTION];
    for (int j = 0; j < SECTION; ++j) {
        int cb = (pattern >> j) & 1;
        e[j].load(S.z[zidx(k, si, j, cb)], p + 2 * cb, S.tgt[SECTION * k + j]);
    }
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
        c += e[j].w(T, f[j]);
        key |= e[j].rank(T, f[j]) << (4 * j);
        dist += e[j].d(f[j]);
    }
    if (holds(bound, c, key) && dist < res.d) {
        res = LayerResult<Real>{dist, key, true, si == 0};
        if (si == 0) return res;
    }
    for (int j = 0; j < SECTION; ++j) {
        int j2 = j == j0 ? j1 : j0;
        int s2 = f[j2] == 2 ? e[j2].dir : 2 - f[j2];
        int f2 = f[j2] + s2;
        Real dj = e[j].d(f[j]), dj2 = e[j2].d(f[j2]), df2 = e[j2].d(f2);
        uint32_t wj = e[j].w(T, f[j]), wj2 = e[j2].w(T, f[j2]), wf2 = e[j2].w(T, f2);
        for (int step = -1; step <= 1; step += 2) {
            int fj = f[j] + step;
            Real d2 = dist - dj - dj2 + e[j].d(fj) + df2;
            if (d2 >= res.d) continue;
            uint32_t c2 = c - wj - wj2 + e[j].w(T, fj) + wf2;
            uint32_t mask = ~((0xfu << (4 * j)) | (0xfu << (4 * j2)));
            uint32_t k2 = (key & mask) | (e[j].rank(T, fj) << (4 * j)) | (e[j2].rank(T, f2) << (4 * j2));
            if (holds(bound, c2, k2)) { res.d = d2; res.key = k2; res.have = true; }
        }
    }
    return res;
}

// solve() from per-layer results in slots [off, off + n): layer 0's member base, else the first minimum.
template <class Real>
__device__ bool combine_layers(const WarpState<Real>& S, int off, int n, Real& out_d, uint32_t& out_key) {
    if (S.sflag[off] & 2) { out_d = S.sd[off]; out_key = S.skey[off]; return true; }
    bool have = false;
    Real best = infinity<Real>();
    for (int i = 0; i < n; ++i)
        if ((S.sflag[off + i] & 1) && S.sd[off + i] < best) { best = S.sd[off + i]; out_key = S.skey[off + i]; have = true; }
    if (have) out_d = best;
    return have;
}

template <class Real>
__device__ Real dist_fb(const Fallback& f, const Real* target) {
    Real s = Real(0);
    for (int j = 0; j < SECTION; ++j) {
        Real d = Real(f.y[j]) - target[j];
        s += d * d;
    }
    return s;
}

// Resolve one entry of the 512 (end_resolve / mid_resolve) and store it.
template <class Real>
__device__ void resolve_w(const WarpTables& T, const Fallback* fbg, WarpState<Real>& S, uint32_t p, int idx) {
    if (idx < E2) {
        int kind = idx >= E3, rem = idx - kind * E3, r = rem / STATES, st = rem % STATES, k = 2 * kind;
        const uint8_t* patterns = kind == 0 ? T.prefixes[st] : T.suffixes[st];
        const Fallback& f = fbg[((kind * 2 + int(p)) * 2 + r) * STATES + st];
        Real value = dist_fb<Real>(f, S.tgt + SECTION * k);
        uint8_t byte = f.byte;
        uint32_t key = f.key;
        for (int i = 0; i < 2; ++i) {
            Real d;
            uint32_t kk;
            if (solve_w<Real>(T, S, k, p, patterns[i], uint32_t(r), T.class_b[r], 7, d, kk))
                if (d < value) { value = d; byte = patterns[i]; key = kk; }
        }
        S.val[idx] = value;
        put(S, idx, true, byte, key);
    } else {
        int t = idx - E2, i2 = t >> 1, delta = t & 1;
        uint8_t byte = T.msets[i2 / BRANCHES][i2 % BRANCHES];
        Real d;
        uint32_t key;
        if (solve_w<Real>(T, S, 1, p, byte, uint32_t(delta), T.mixed, 8, d, key)) {
            S.val[idx] = d;
            put(S, idx, true, byte, key);
        } else {
            S.val[idx] = infinity<Real>();
            put(S, idx, true, 0, 0);
        }
    }
}

template <class Real>
struct WPath {
    Real cost;
    int i1, i2, delta, i3;
};

// join: the first (lowest path index) minimum of e1 + e2 + e3 over the 4,096 (r, s8, b2, delta) paths.
template <class Real>
__device__ WPath<Real> join_w(const WarpTables& T, const WarpState<Real>& S, uint32_t p, int lane) {
    Real best = infinity<Real>();
    int bidx = INT_MAX;
    for (int q = 0; q < 4; ++q) {
        int i1 = lane + LANES * q, r = i1 / STATES, s8 = i1 % STATES;
        Real a = S.val[i1];
        if (!(a < best)) continue;  // every path through i1 costs >= a: none can beat this lane's best
        for (int b2 = 0; b2 < BRANCHES; ++b2) {
            int s16 = T.br_s16[s8][b2], i2 = T.branch_ix[s8][b2];
            for (int delta = 0; delta < 2; ++delta) {
                int i3 = ((int(p) ^ r ^ delta) & 1) * STATES + s16;
                Real c = a + S.val[E2 + 2 * i2 + delta] + S.val[E3 + i3];
                if (c < best) { best = c; bidx = (i1 * BRANCHES + b2) * 2 + delta; }
            }
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        Real ob = __shfl_xor_sync(0xffffffffu, best, off);
        int oi = __shfl_xor_sync(0xffffffffu, bidx, off);
        if (ob < best || (ob == best && oi < bidx)) { best = ob; bidx = oi; }
    }
    WPath<Real> path{best, 0, 0, 0, 0};
    if (bidx != INT_MAX) {
        int i1 = bidx / (BRANCHES * 2), b2 = (bidx / 2) % BRANCHES, delta = bidx & 1;
        int r = i1 / STATES, s8 = i1 % STATES;
        path.i1 = i1;
        path.i2 = T.branch_ix[s8][b2];
        path.delta = delta;
        path.i3 = ((int(p) ^ r ^ delta) & 1) * STATES + T.br_s16[s8][b2];
    }
    return path;
}

template <class Real>
__device__ Real warp_min(Real v) {
    for (int off = 16; off > 0; off >>= 1) {
        Real o = __shfl_xor_sync(0xffffffffu, v, off);
        v = o < v ? o : v;
    }
    return v;
}

// solve_parity for one block on one warp; every lane returns the same result.
template <class Real>
__device__ bool pass_w(const WarpTables& T, const Fallback* fbg, WarpState<Real>& S, uint32_t p, Real u0, int lane,
                       Winner<Real>& out) {
    const Real INF = infinity<Real>();
    __syncwarp();
    for (int i = lane; i < 384; i += LANES) {
        int k = i >> 7, layer = (i >> 4) & 7, j = (i >> 1) & 7, cb = i & 1;
        uint32_t o = p + 2 * uint32_t(cb);
        Real z = (S.tgt[SECTION * k + j] * Real(shrink_of(layer)) - Real(o)) / Real(4);
        z = z < Real(-1e6) ? Real(-1e6) : (z > Real(1e6) ? Real(1e6) : z);
        S.z[i] = z;
    }
    __syncwarp();
    for (int task = lane; task < 2 * STATES; task += LANES) {
        int kind = task / STATES, st = task % STATES, k = 2 * kind;
        const uint8_t* patterns = kind == 0 ? T.prefixes[st] : T.suffixes[st];
        Real value[2];
        uint8_t byte[2];
        uint32_t key[2];
        for (int r = 0; r < 2; ++r) {
            const Fallback& f = fbg[((kind * 2 + int(p)) * 2 + r) * STATES + st];
            value[r] = dist_fb<Real>(f, S.tgt + SECTION * k);
            byte[r] = f.byte;
            key[r] = f.key;
        }
        Real lb[2] = {INF, INF};
        bool unresolved[2] = {false, false};
        for (int i = 0; i < 2; ++i) {
            uint8_t c = patterns[i];
            Real d[2];
            bool m[2];
            uint32_t kk[2];
            base_pair_w<Real>(T, S, k, p, c, T.class_b[0], T.class_b[1], d, m, kk);
            for (int r = 0; r < 2; ++r) {
                if (m[r]) {
                    if (d[r] < value[r]) { value[r] = d[r]; byte[r] = c; key[r] = kk[r]; }
                } else {
                    lb[r] = lb[r] < d[r] ? lb[r] : d[r];
                    unresolved[r] = true;
                }
            }
        }
        for (int r = 0; r < 2; ++r) {
            int idx = kind * E3 + r * STATES + st;
            bool inexact = unresolved[r] && lb[r] < value[r];
            S.val[idx] = inexact ? lb[r] : value[r];
            put(S, idx, !inexact, byte[r], key[r]);
        }
    }
    for (int task = lane; task < MID_ENTRIES; task += LANES) {
        uint8_t b = T.msets[task / BRANCHES][task % BRANCHES];
        Real d[2];
        bool m[2];
        uint32_t kk[2];
        base_pair_w<Real>(T, S, 1, p, b, T.mixed, T.mixed, d, m, kk);
        for (int dd = 0; dd < 2; ++dd) {
            S.val[E2 + 2 * task + dd] = d[dd];
            put(S, E2 + 2 * task + dd, m[dd], b, kk[dd]);
        }
    }
    __syncwarp();
    Real u = u0;
    for (int w = 0; w < 2; ++w) {
        WPath<Real> path = join_w<Real>(T, S, p, lane);
        int ix[3] = {path.i1, E2 + 2 * path.i2 + path.delta, E3 + path.i3};
        bool x1 = exact_at(S, ix[0]), x2 = exact_at(S, ix[1]), x3 = exact_at(S, ix[2]);
        __syncwarp();
        // The up to three resolves, one lane per (pattern, layer): ends 2 x 7 layers, the middle 8.
        const int n0 = x1 ? 0 : 14, n1 = x2 ? 0 : 8, n2 = x3 ? 0 : 14;
        for (int t = lane; t < n0 + n1 + n2; t += LANES) {
            int job = t < n0 ? 0 : t < n0 + n1 ? 1 : 2;
            int lt = t - (job == 0 ? 0 : job == 1 ? n0 : n0 + n1);
            LayerResult<Real> res;
            if (job == 1) {
                int tt = ix[1] - E2, i2 = tt >> 1;
                res = solve_layer<Real>(T, S, 1, p, T.msets[i2 / BRANCHES][i2 % BRANCHES], uint32_t(tt & 1), T.mixed, lt);
            } else {
                int kind = job == 2, rem = ix[job] - kind * E3, r = rem / STATES, st = rem % STATES;
                const uint8_t* patterns = kind == 0 ? T.prefixes[st] : T.suffixes[st];
                res = solve_layer<Real>(T, S, 2 * kind, p, patterns[lt / 7], uint32_t(r), T.class_b[r], lt % 7);
            }
            S.sd[t] = res.d;
            S.skey[t] = res.key;
            S.sflag[t] = uint8_t(res.have | res.base0 << 1);
        }
        __syncwarp();
        if (lane < 3 && !(lane == 0 ? x1 : lane == 1 ? x2 : x3)) {
            const int idx = ix[lane];
            if (lane == 1) {
                int i2 = (idx - E2) >> 1;
                Real d;
                uint32_t key;
                if (combine_layers<Real>(S, n0, 8, d, key)) {
                    S.val[idx] = d;
                    put(S, idx, true, T.msets[i2 / BRANCHES][i2 % BRANCHES], key);
                } else {
                    S.val[idx] = infinity<Real>();
                    put(S, idx, true, 0, 0);
                }
            } else {
                int kind = lane == 2, off = kind ? n0 + n1 : 0, rem = idx - kind * E3, r = rem / STATES, st = rem % STATES;
                const uint8_t* patterns = kind == 0 ? T.prefixes[st] : T.suffixes[st];
                const Fallback& f = fbg[((kind * 2 + int(p)) * 2 + r) * STATES + st];
                Real value = dist_fb<Real>(f, S.tgt + SECTION * 2 * kind);
                uint8_t byte = f.byte;
                uint32_t key = f.key;
                for (int i = 0; i < 2; ++i) {
                    Real d;
                    uint32_t kk;
                    if (combine_layers<Real>(S, off + 7 * i, 7, d, kk))
                        if (d < value) { value = d; byte = patterns[i]; key = kk; }
                }
                S.val[idx] = value;
                put(S, idx, true, byte, key);
            }
        }
        __syncwarp();
        if (x1 && x2 && x3) { u = path.cost < u ? path.cost : u; break; }
    }
    // Bound pass: the cheapest path through each entry, and the cheapest all-exact path.
    Real ul = u;
    Real bnd1[4], bnd3[4], bnd2[8];
    for (int q = 0; q < 4; ++q) {
        int i1 = lane + LANES * q, r = i1 / STATES, s8 = i1 % STATES;
        Real a = S.val[i1];
        bool ax = exact_at(S, i1);
        Real best1 = INF;
        for (int b2 = 0; b2 < BRANCHES; ++b2) {
            int s16 = T.br_s16[s8][b2], i2 = T.branch_ix[s8][b2];
            for (int delta = 0; delta < 2; ++delta) {
                int e2 = E2 + 2 * i2 + delta, e3 = E3 + ((int(p) ^ r ^ delta) & 1) * STATES + s16;
                Real c = a + S.val[e2] + S.val[e3];
                best1 = c < best1 ? c : best1;
                if (ax && exact_at(S, e2) && exact_at(S, e3) && c < ul) ul = c;
            }
        }
        bnd1[q] = best1;
    }
    for (int q = 0; q < 8; ++q) {
        int t = lane + LANES * q, i2 = t >> 1, delta = t & 1;
        Real v2 = S.val[E2 + t], best = INF;
        for (int e = T.inv2_start[i2]; e < T.inv2_start[i2 + 1]; ++e) {
            int s8 = T.inv2[e] >> 4, b2 = T.inv2[e] & 15, s16 = T.br_s16[s8][b2];
            for (int r = 0; r < 2; ++r) {
                Real c = S.val[r * STATES + s8] + v2 + S.val[E3 + ((int(p) ^ r ^ delta) & 1) * STATES + s16];
                best = c < best ? c : best;
            }
        }
        bnd2[q] = best;
    }
    for (int q = 0; q < 4; ++q) {
        int i3 = lane + LANES * q, r_out = i3 / STATES, s16 = i3 % STATES;
        Real v3 = S.val[E3 + i3], best = INF;
        for (int e = T.inv3_start[s16]; e < T.inv3_start[s16 + 1]; ++e) {
            int s8 = T.inv3[e] >> 4, b2 = T.inv3[e] & 15, i2 = T.branch_ix[s8][b2];
            for (int r = 0; r < 2; ++r) {
                int delta = (int(p) ^ r ^ r_out) & 1;
                Real c = S.val[r * STATES + s8] + S.val[E2 + 2 * i2 + delta] + v3;
                best = c < best ? c : best;
            }
        }
        bnd3[q] = best;
    }
    u = warp_min<Real>(ul);
    __syncwarp();
    int n = 0;
    for (int q = 0; q < ENTRIES / LANES; ++q) {
        int idx = q * LANES + lane;
        Real b = q < 4 ? bnd1[q] : q < 8 ? bnd3[q - 4] : bnd2[q - 8];
        bool need = !exact_at(S, idx) && b < u;
        unsigned mask = __ballot_sync(0xffffffffu, need);
        if (need) S.list[n + __popc(mask & ((1u << lane) - 1u))] = uint16_t(idx);
        n += __popc(mask);
    }
    __syncwarp();
    for (int q = lane; q < n; q += LANES) resolve_w<Real>(T, fbg, S, p, S.list[q]);
    __syncwarp();
    for (int idx = lane; idx < ENTRIES; idx += LANES)
        if (!exact_at(S, idx)) { S.val[idx] = INF; put(S, idx, true, 0, 0); }
    __syncwarp();
    WPath<Real> path = join_w<Real>(T, S, p, lane);
    if (!(path.cost < INF)) return false;
    int r = path.i1 / STATES, s8 = path.i1 % STATES;
    const int x2i = E2 + 2 * path.i2 + path.delta, x3i = E3 + path.i3;
    uint8_t mid_byte = byte_at(S, x2i);
    int b2 = 0;
    for (int b = 0; b < BRANCHES; ++b)
        if (T.br_byte[s8][b] == mid_byte) { b2 = b; break; }
    out.cost = path.cost;
    out.p = p;
    out.r = uint32_t(r);
    out.s8 = s8;
    out.s16 = int(T.br_s16[s8][b2]);
    out.delta = uint32_t(path.delta);
    out.byte[0] = byte_at(S, path.i1); out.byte[1] = mid_byte; out.byte[2] = byte_at(S, x3i);
    out.key[0] = S.key[path.i1]; out.key[1] = S.key[x2i]; out.key[2] = S.key[x3i];
    return true;
}

template <class Real>
__device__ Code code_w(const WarpTables& T, const int16_t* pos5, const Real* x, const Winner<Real>& w) {
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
__device__ Code at_scale_w(const WarpTables& T, const Fallback* fbg, const int16_t* pos5, WarpState<Real>& S, Real s,
                           int lane) {
    __syncwarp();
    if (lane < DIM) S.tgt[lane] = S.x[T.order[lane]] / s;
    __syncwarp();
    bool have = false;
    Winner<Real> best{};
    for (uint32_t p = 0; p < 2; ++p) {
        Real u0 = have ? best.cost : infinity<Real>();
        Winner<Real> w;
        if (pass_w<Real>(T, fbg, S, p, u0, lane, w))
            if (!have || w.cost < best.cost) { best = w; have = true; }
    }
    if (!have) return Code{0, -infinity<double>()};
    return code_w<Real>(T, pos5, S.x, best);
}

template <class Real>
__device__ Code two_scales_w(const WarpTables& T, const Fallback* fbg, const int16_t* pos5, WarpState<Real>& S, Real s0,
                             int lane) {
    Code a = at_scale_w<Real>(T, fbg, pos5, S, s0, lane);
    Code b = at_scale_w<Real>(T, fbg, pos5, S, s0 * Real(1.14), lane);
    return b.t > a.t ? b : a;
}

template <class Real>
__global__ void __launch_bounds__(LANES* WARPS) encode_warp(const WarpTables* __restrict__ tg, const Fallback* __restrict__ fbg,
                                                            const int16_t* __restrict__ pos5, const Real* __restrict__ x,
                                                            int64_t n, uint64_t* __restrict__ words, double* __restrict__ t) {
    __shared__ WarpTables T;
    __shared__ WarpState<Real> SS[WARPS];
    const uint32_t* src = reinterpret_cast<const uint32_t*>(tg);
    uint32_t* dst = reinterpret_cast<uint32_t*>(&T);
    for (int i = threadIdx.x; i < int(sizeof(WarpTables) / 4); i += blockDim.x) dst[i] = src[i];
    __syncthreads();
    const int lane = threadIdx.x % LANES, warp = threadIdx.x / LANES;
    WarpState<Real>& S = SS[warp];
    for (int64_t i = int64_t(blockIdx.x) * WARPS + warp; i < n; i += int64_t(gridDim.x) * WARPS) {
        __syncwarp();
        if (lane < DIM) S.x[lane] = x[i * DIM + lane];
        __syncwarp();
        Real m = Real(0);
        for (int j = 0; j < DIM; ++j) { Real a = r_abs(S.x[j]); m = m > a ? m : a; }
        Code c{0, -infinity<double>()};
        if (m != Real(0)) {
            Real s = Real(0);
            for (int j = 0; j < DIM; ++j) { Real v = S.x[j] / m; s += v * v; }
            Real s0 = Real(0.3218) * m * sqrt(s) / sqrt(Real(DIM));
            const Real tiny = sizeof(Real) == 8 ? Real(2.2250738585072014e-308) : Real(1.17549435e-38f);
            if (s0 == s0 && s0 < infinity<Real>() && s0 >= tiny) {
                c = two_scales_w<Real>(T, fbg, pos5, S, s0, lane);
            } else {
                Real xs[DIM];
                for (int j = 0; j < DIM; ++j) xs[j] = S.x[j] / m;
                Real m2 = Real(0);
                for (int j = 0; j < DIM; ++j) { Real a = r_abs(xs[j]); m2 = m2 > a ? m2 : a; }
                Real s2 = Real(0);
                for (int j = 0; j < DIM; ++j) { Real v = xs[j] / m2; s2 += v * v; }
                Real s0b = Real(0.3218) * m2 * sqrt(s2) / sqrt(Real(DIM));
                __syncwarp();
                if (lane < DIM) S.x[lane] = xs[lane];
                c = two_scales_w<Real>(T, fbg, pos5, S, s0b, lane);
                c.t *= double(m);
            }
        }
        if (lane == 0) {
            words[i] = c.word;
            if (t) t[i] = c.t;
        }
    }
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("llvq tetra: ") + what + ": " + cudaGetErrorString(e));
}
}  // namespace

struct LlvqTetraGpu {
    WarpTables* tables = nullptr;
    Fallback* fallback = nullptr;
    int16_t* pos5 = nullptr;
    int ctas = 0;
};

LlvqTetraGpu* llvq_tetra_gpu_create(const void* host_view, const int16_t* host_pos5) {
    const auto& v = *static_cast<const TetraTablesView*>(host_view);
    WarpTables w;
    std::memset(&w, 0, sizeof w);
    std::memcpy(w.order, v.order, sizeof w.order);
    std::memcpy(w.prefixes, v.prefixes, sizeof w.prefixes);
    std::memcpy(w.suffixes, v.suffixes, sizeof w.suffixes);
    std::memcpy(w.br_byte, v.br_byte, sizeof w.br_byte);
    std::memcpy(w.br_s16, v.br_s16, sizeof w.br_s16);
    std::memcpy(w.branch_ix, v.branch_ix, sizeof w.branch_ix);
    std::memcpy(w.msets, v.msets, sizeof w.msets);
    std::memcpy(w.ranks, v.ranks, sizeof w.ranks);
    w.class_b[0] = v.class_b[0];
    w.class_b[1] = v.class_b[1];
    w.mixed = v.mixed;
    std::vector<std::vector<uint16_t>> by2(MID_ENTRIES), by3(STATES);
    for (int s8 = 0; s8 < STATES; ++s8)
        for (int b2 = 0; b2 < BRANCHES; ++b2) {
            by2[v.branch_ix[s8][b2]].push_back(uint16_t(s8 * BRANCHES + b2));
            by3[v.br_s16[s8][b2]].push_back(uint16_t(s8 * BRANCHES + b2));
        }
    int n2 = 0, n3 = 0;
    for (int i = 0; i < MID_ENTRIES; ++i) {
        w.inv2_start[i] = uint16_t(n2);
        for (uint16_t e : by2[i]) w.inv2[n2++] = e;
    }
    w.inv2_start[MID_ENTRIES] = uint16_t(n2);
    for (int i = 0; i < STATES; ++i) {
        w.inv3_start[i] = uint16_t(n3);
        for (uint16_t e : by3[i]) w.inv3[n3++] = e;
    }
    w.inv3_start[STATES] = uint16_t(n3);
    auto* g = new LlvqTetraGpu;
    check(cudaMalloc((void**)&g->tables, sizeof w), "alloc tables");
    check(cudaMemcpy(g->tables, &w, sizeof w, cudaMemcpyHostToDevice), "copy tables");
    check(cudaMalloc((void**)&g->fallback, sizeof v.fallback), "alloc fallback");
    check(cudaMemcpy(g->fallback, v.fallback, sizeof v.fallback, cudaMemcpyHostToDevice), "copy fallback");
    check(cudaMalloc((void**)&g->pos5, 390625 * sizeof(int16_t)), "alloc pos5");
    check(cudaMemcpy(g->pos5, host_pos5, 390625 * sizeof(int16_t), cudaMemcpyHostToDevice), "copy pos5");
    int dev = 0, sms = 0;
    check(cudaGetDevice(&dev), "device");
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "sm count");
    g->ctas = sms * 16;
    return g;
}

void llvq_tetra_gpu_destroy(LlvqTetraGpu* g) {
    if (!g) return;
    cudaFree(g->tables);
    cudaFree(g->fallback);
    cudaFree(g->pos5);
    delete g;
}

void llvq_tetra_gpu_set_ctas(LlvqTetraGpu* g, int ctas) {
    if (ctas > 0) g->ctas = ctas;
}

void llvq_tetra_encode_f64(LlvqTetraGpu* g, const double* x, int64_t n, uint64_t* words, double* t, void* stream) {
    if (n <= 0) return;
    int64_t ctas = std::min<int64_t>(g->ctas, (n + WARPS - 1) / WARPS);
    encode_warp<double><<<unsigned(ctas), LANES * WARPS, 0, (cudaStream_t)stream>>>(g->tables, g->fallback, g->pos5, x, n, words, t);
    check(cudaGetLastError(), "encode f64 launch");
}
void llvq_tetra_encode_f32(LlvqTetraGpu* g, const float* x, int64_t n, uint64_t* words, double* t, void* stream) {
    if (n <= 0) return;
    int64_t ctas = std::min<int64_t>(g->ctas, (n + WARPS - 1) / WARPS);
    encode_warp<float><<<unsigned(ctas), LANES * WARPS, 0, (cudaStream_t)stream>>>(g->tables, g->fallback, g->pos5, x, n, words, t);
    check(cudaGetLastError(), "encode f32 launch");
}

// ---- decode ---------------------------------------------------------------------------------------------------
namespace {
struct DecodeTablesDev {
    uint32_t rows[2 * CLASS_ROWS];
    uint8_t pre[STATES][2], suf[STATES][2];
    uint8_t br_c[STATES][BRANCHES], br_s[STATES][BRANCHES];
    int8_t vals[2][16];
    uint32_t order[DIM];
};
static_assert(sizeof(DecodeTablesDev) % 4 == 0, "decode tables copy as 32-bit words");

__device__ __forceinline__ void stage_tables(const DecodeTablesDev* g, DecodeTablesDev& s) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(g);
    uint32_t* dst = reinterpret_cast<uint32_t*>(&s);
    for (int i = threadIdx.x; i < int(sizeof(DecodeTablesDev) / 4); i += blockDim.x) dst[i] = src[i];
    __syncthreads();
}

// Word -> 24 values in trio order; returns sum of squares.
__device__ __forceinline__ int decode_word(const DecodeTablesDev& T, const uint32_t* __restrict__ rows, uint64_t w, int8_t* y) {
    const uint32_t p = uint32_t(w & 1), r = uint32_t(w >> 1) & 1, s8 = uint32_t(w >> 2) & 63, b1 = uint32_t(w >> 8) & 1,
                   i1 = uint32_t(w >> 9) & 2047, b2 = uint32_t(w >> 20) & 15, i2 = uint32_t(w >> 24) & 2047,
                   b3 = uint32_t(w >> 35) & 1, i3 = uint32_t(w >> 36) & 2047;
    const uint32_t s16 = T.br_s[s8][b2], delta = i2 >= uint32_t(N0_MIXED);
    const uint32_t c[3] = {T.pre[s8][b1], T.br_c[s8][b2], T.suf[s16][b3]};
    const uint32_t row[3] = {rows[CLASS_ROWS * r + i1], delta ? rows[CLASS_ROWS + i2 - N0_MIXED] : rows[i2],
                             rows[CLASS_ROWS * ((p ^ r ^ delta) & 1) + i3]};
    int ss = 0;
#pragma unroll
    for (int k = 0; k < 3; ++k)
#pragma unroll
        for (int j = 0; j < SECTION; ++j) {
            int8_t v = T.vals[p][((row[k] >> (4 * j)) & 15) + 8 * ((c[k] >> j) & 1)];
            y[SECTION * k + j] = v;
            ss += int(v) * v;
        }
    return ss;
}

__device__ __forceinline__ uint64_t load_word(const uint8_t* row, uint32_t b) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(row + 6 * size_t(b));
    return uint64_t(h[0]) | uint64_t(h[1]) << 16 | uint64_t(h[2]) << 32;
}
__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }

constexpr int GEMV_WARPS = 8;

template <int NT>
__global__ void __launch_bounds__(LANES* GEMV_WARPS) gemv_kernel(const DecodeTablesDev* __restrict__ tg, LlvqTetraMatrixDev m,
                                                                  const float* __restrict__ xp, float* __restrict__ y) {
    // The small tables in shared memory. One token: the 16 KB row table stays in global memory (cache resident;
    // 19.7 vs 21.3 us on a 2048 x 4096 matrix, RX 9070 XT); more tokens: staged too (51 vs 58 us at 8 tokens).
    __shared__ DecodeTablesDev T;
    {
        constexpr int off0 = int(offsetof(DecodeTablesDev, pre) / 4), words = int(sizeof(DecodeTablesDev) / 4);
        constexpr int off = NT == 1 ? off0 : 0;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(tg);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&T);
        for (int i = off + threadIdx.x; i < words; i += blockDim.x) dst[i] = src[i];
        __syncthreads();
    }
    const int lane = threadIdx.x % LANES, warp = threadIdx.x / LANES;
    const uint32_t nb = m.cols / 24, tail = m.cols - 24 * nb;
    for (uint32_t r = blockIdx.x * GEMV_WARPS + warp; r < m.rows; r += gridDim.x * GEMV_WARPS) {
        const uint8_t* wrow = m.words + size_t(r) * m.stride;
        float acc[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) acc[k] = 0.f;
        for (uint32_t b = lane; b < nb; b += LANES) {
            const uint64_t w = load_word(wrow, b);
            int8_t v[DIM];
            const int ss = decode_word(T, NT == 1 ? tg->rows : T.rows, w, v);
            if (!ss) continue;
            const float s = ((w >> 47) & 1 ? m.c1 : m.c0) * rsqrtf(float(ss));
#pragma unroll
            for (int k = 0; k < NT; ++k) {
                const float4* x4 = reinterpret_cast<const float4*>(xp + size_t(k) * m.cols + 24 * size_t(b));
                float d = 0.f;
#pragma unroll
                for (int q = 0; q < 6; ++q) {
                    float4 xv = x4[q];
                    d += float(v[4 * q]) * xv.x + float(v[4 * q + 1]) * xv.y + float(v[4 * q + 2]) * xv.z + float(v[4 * q + 3]) * xv.w;
                }
                acc[k] += s * d;
            }
        }
        float tl[NT];
#pragma unroll
        for (int k = 0; k < NT; ++k) tl[k] = 0.f;
        for (uint32_t c = lane; c < tail; c += LANES) {
            const float wv = bf16f(m.tail_bf16[size_t(r) * tail + c]);
#pragma unroll
            for (int k = 0; k < NT; ++k) tl[k] += wv * xp[size_t(k) * m.cols + 24 * nb + c];
        }
        const float rs = m.scales[r];
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            float v = acc[k] * rs + tl[k];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0) y[size_t(r) * NT + k] = v;
        }
    }
}

__global__ void dequant_kernel(const DecodeTablesDev* __restrict__ tg, LlvqTetraMatrixDev m, __half* __restrict__ out) {
    __shared__ DecodeTablesDev T;
    stage_tables(tg, T);
    const uint32_t nb = m.cols / 24, tail = m.cols - 24 * nb;
    const size_t total = size_t(m.rows) * (nb + (tail ? 1 : 0));
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
        const uint32_t r = uint32_t(i / (nb + (tail ? 1 : 0))), b = uint32_t(i % (nb + (tail ? 1 : 0)));
        __half* orow = out + size_t(r) * m.cols;
        if (b == nb) {
            for (uint32_t c = 0; c < tail; ++c) orow[24 * nb + c] = __float2half(bf16f(m.tail_bf16[size_t(r) * tail + c]));
            continue;
        }
        const uint64_t w = load_word(m.words + size_t(r) * m.stride, b);
        int8_t v[DIM];
        const int ss = decode_word(T, T.rows, w, v);
        const float s = ss ? ((w >> 47) & 1 ? m.c1 : m.c0) * m.scales[r] * rsqrtf(float(ss)) : 0.f;
#pragma unroll
        for (int t = 0; t < DIM; ++t) orow[24 * b + T.order[t]] = __float2half(float(v[t]) * s);
    }
}

// One CTA per token: signs, in-place WHT in shared memory, 1/sqrt(n), trio permutation.
__global__ void prepare_kernel(const DecodeTablesDev* __restrict__ tg, const float* __restrict__ signs, const float* __restrict__ x,
                               int cols, float* __restrict__ xp) {
    extern __shared__ float v[];
    __shared__ uint32_t order[DIM];
    if (threadIdx.x < DIM) order[threadIdx.x] = tg->order[threadIdx.x];
    const float* xt = x + size_t(blockIdx.x) * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) v[i] = xt[i] * signs[i];
    __syncthreads();
    for (int len = 1; len < cols; len <<= 1) {
        for (int t = threadIdx.x; t < cols / 2; t += blockDim.x) {
            int i = (t / len) * (2 * len) + (t % len);
            float a = v[i], b = v[i + len];
            v[i] = a + b;
            v[i + len] = a - b;
        }
        __syncthreads();
    }
    const float s = rsqrtf(float(cols));
    const int nb = cols / 24;
    float* o = xp + size_t(blockIdx.x) * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const int b = i / 24, t = i % 24;
        o[i] = (b < nb ? v[24 * b + order[t]] : v[i]) * s;
    }
}
}  // namespace

struct LlvqTetraDecodeGpu {
    DecodeTablesDev* tables = nullptr;
    unsigned max_ctas = 256;  // GEMV grid cap: each CTA stages ~19 KB of tables
};

LlvqTetraDecodeGpu* llvq_tetra_decode_create(const uint8_t* pre, const uint8_t* suf, const uint8_t* br_c, const uint8_t* br_s,
                                             const uint32_t* rows, const int8_t* vals, const uint32_t* order) {
    DecodeTablesDev h;
    std::memcpy(h.rows, rows, sizeof h.rows);
    std::memcpy(h.pre, pre, sizeof h.pre);
    std::memcpy(h.suf, suf, sizeof h.suf);
    std::memcpy(h.br_c, br_c, sizeof h.br_c);
    std::memcpy(h.br_s, br_s, sizeof h.br_s);
    std::memcpy(h.vals, vals, sizeof h.vals);
    std::memcpy(h.order, order, sizeof h.order);
    auto* d = new LlvqTetraDecodeGpu;
    check(cudaMalloc((void**)&d->tables, sizeof h), "alloc decode tables");
    check(cudaMemcpy(d->tables, &h, sizeof h, cudaMemcpyHostToDevice), "copy decode tables");
    int dev = 0, sms = 0;
    check(cudaGetDevice(&dev), "device");
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "sm count");
    d->max_ctas = unsigned(std::max(1, 4 * sms));
    return d;
}
void llvq_tetra_decode_destroy(LlvqTetraDecodeGpu* d) {
    if (!d) return;
    cudaFree(d->tables);
    delete d;
}

void llvq_tetra_gemv(const LlvqTetraDecodeGpu* d, const LlvqTetraMatrixDev& m, const float* xp, int ntok, float* y, void* stream) {
    const unsigned grid = std::max(1u, std::min(d->max_ctas, (m.rows + GEMV_WARPS - 1) / GEMV_WARPS));
    auto s = (cudaStream_t)stream;
    switch (ntok) {
    case 1: gemv_kernel<1><<<grid, LANES * GEMV_WARPS, 0, s>>>(d->tables, m, xp, y); break;
    case 2: gemv_kernel<2><<<grid, LANES * GEMV_WARPS, 0, s>>>(d->tables, m, xp, y); break;
    case 4: gemv_kernel<4><<<grid, LANES * GEMV_WARPS, 0, s>>>(d->tables, m, xp, y); break;
    case 8: gemv_kernel<8><<<grid, LANES * GEMV_WARPS, 0, s>>>(d->tables, m, xp, y); break;
    default: throw std::runtime_error("llvq tetra gemv: ntok must be 1, 2, 4 or 8");
    }
    check(cudaGetLastError(), "gemv launch");
}
void llvq_tetra_dequant_f16(const LlvqTetraDecodeGpu* d, const LlvqTetraMatrixDev& m, void* out_f16, void* stream) {
    dequant_kernel<<<1024, 256, 0, (cudaStream_t)stream>>>(d->tables, m, static_cast<__half*>(out_f16));
    check(cudaGetLastError(), "dequant launch");
}
void llvq_tetra_prepare(const LlvqTetraDecodeGpu* d, const float* signs, const float* x, int ntok, int cols, float* xp, void* stream) {
    if (cols <= 0 || (cols & (cols - 1))) throw std::runtime_error("llvq tetra prepare: width must be a power of two");
    prepare_kernel<<<ntok, 512, size_t(cols) * 4, (cudaStream_t)stream>>>(d->tables, signs, x, cols, xp);
    check(cudaGetLastError(), "prepare launch");
}

}  // namespace strata::kernels
