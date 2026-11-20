// k31pro_runner: Kimi K3.1 Pro (kimi_linear: 69 KDA + 24 gated DSA layers, latent MoE 896 x top-16 + 2 shared, block
// attention residuals) decode on CPU ranks, one rank per node. Development runner: dummy weights at the real shapes
// for any subset of the 93 layers (--layers), so per-layer time and memory can be measured before the real loader.
//
// Layout over W ranks (rank R):
//   KDA       heads split (96 / W per rank); q/k/v/g/f_b/b rows, conv and state local; o_proj columns local (partial sum)
//   DSA       q_a, kv_a, indexer projections replicated; q_b / kv_b / g rows and o_proj columns by head;
//             latent cache (F16, 576 per token) and indexer keys (F16, 128) sharded by token blocks of 64:
//             token t on rank (t / 64) % W. Indexer: local scores, local top-k, global top-2048 over all ranks;
//             attention: per-rank partial softmax over its selected tokens, merged by max / sum all-reduces.
//   MoE       router replicated; latent down / up rows split; expert e on rank e % W; shared experts' inter split
//   dense 0   inter split
//   lm_head   vocab rows split, global argmax
// Matrix rows go through ds41_rows (BF16 x F32, MXFP4 x Q8_0). Collectives are MPI (-DK31_MPI); without it, W = 1.
// The K3 experiments (gemm/k3, k3_full_runner.c) are the reference for the layer order and every kernel but the
// indexer; see tools/a64fx_probe/k31_kernel_test.c.
#include "strata/kernels/cpu/ds41_sve.h"
#include "strata/kernels/cpu/k31_sve.h"
#include "k31_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(K31_MPI)
#include <mpi.h>
#endif

namespace {

using Vec = std::vector<float>;
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- comm
#if defined(K31_UTOFU_AR)
extern "C" {
int cpu_utofu_init(int rank, int world, int max_count);
void cpu_utofu_allreduce(float *x, int n);
const char *cpu_utofu_mode(void);
}
#endif
// K31_AR=utofu (build with -DK31_UTOFU_AR): float sums through comm_utofu.c's deterministic rsag all-reduce
// (rank-order sums, same bits on every rank); max, gathers and the load's checks stay MPI.
#if defined(K31_UTOFU)
extern "C" {
int k31ar_init(int R, int max_count);
int k31ar_allreduce(float *x, int n);
}
#endif
struct Comm {
    bool utofu_ar = false;
    bool hier_ar = false;   // K31_AR=hier: k31ar_allreduce (2-level, R x W/R grid; K31AR_R, default 8)
    bool sim = false;   // --sim-world: one process plays one rank of a W-rank layout; collectives are no-ops
    int rank = 0, world = 1;
    double t = 0;   // seconds in collectives
    void init(int *argc, char ***argv) {
#if defined(K31_MPI)
        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
#else
        (void)argc; (void)argv;
#endif
    }
    void fini() {
#if defined(K31_MPI)
        MPI_Finalize();
#endif
    }
    void sum(float *x, int n) {
#if defined(K31_MPI)
        const double t0 = now();
#if defined(K31_UTOFU)
        if (hier_ar && world > 1 && !sim) {
            if (k31ar_allreduce(x, n)) throw std::runtime_error("k31: k31ar_allreduce failed");
            t += now() - t0; return;
        }
#endif
#if defined(K31_UTOFU_AR)
        if (utofu_ar && world > 1 && !sim) { cpu_utofu_allreduce(x, n); t += now() - t0; return; }
#endif
        if (world > 1 && !sim) MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        t += now() - t0;
#else
        (void)x; (void)n;
#endif
    }
    void init_utofu_ar() {
#if defined(K31_UTOFU)
        if (const char *h = std::getenv("K31_AR"); h && std::strcmp(h, "hier") == 0 && world > 1 && !sim) {
            const char *er = std::getenv("K31AR_R");
            const int R = er ? std::atoi(er) : 8;
            const int rc = k31ar_init(R, 65536);
            if (rc) throw std::runtime_error("k31: k31ar_init failed rc=" + std::to_string(rc));
            hier_ar = true;
            if (rank == 0) std::printf("RUNNER_K31_AR hier %d x %d\n", R, world / R);
            return;
        }
#endif
#if defined(K31_UTOFU_AR)
        const char *e = std::getenv("K31_AR");
        if (!e || std::strcmp(e, "utofu") != 0 || world < 2) return;
        const int rc = cpu_utofu_init(rank, world, 65536);   // largest decode sum: 96 heads x 512 (DSA merge)
        if (rc) throw std::runtime_error("k31: cpu_utofu_init failed rc=" + std::to_string(rc));
        utofu_ar = true;
        if (rank == 0) std::printf("RUNNER_K31_AR %s\n", cpu_utofu_mode());
#endif
    }
    void max(float *x, int n) {
#if defined(K31_MPI)
        const double t0 = now();
        if (world > 1 && !sim) MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        t += now() - t0;
#else
        (void)x; (void)n;
#endif
    }
    void dsum(double *x, int n) {
#if defined(K31_MPI)
        if (world > 1 && !sim) MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
        (void)x; (void)n;
#endif
    }
    // every rank's n floats, rank order
    void gather(const float *mine, int n, float *all) {
#if defined(K31_MPI)
        const double t0 = now();
        if (world > 1 && !sim) { MPI_Allgather(mine, n, MPI_FLOAT, all, n, MPI_FLOAT, MPI_COMM_WORLD); t += now() - t0; return; }
#endif
        for (int r = 0; r < (sim ? world : 1); ++r) std::memcpy(all + size_t(r) * n, mine, size_t(n) * sizeof(float));
    }
    void gatheri(const int *mine, int n, int *all) {
#if defined(K31_MPI)
        const double t0 = now();
        if (world > 1 && !sim) { MPI_Allgather(mine, n, MPI_INT, all, n, MPI_INT, MPI_COMM_WORLD); t += now() - t0; return; }
#endif
        for (int r = 0; r < (sim ? world : 1); ++r) std::memcpy(all + size_t(r) * n, mine, size_t(n) * sizeof(int));
    }
    // global (max value, lowest index on ties)
    void argmax(float &v, int &i) {
#if defined(K31_MPI)
        if (world > 1 && !sim) {
            struct { float v; int i; } in{v, i}, out{};
            MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
            v = out.v; i = out.i;
        }
#else
        (void)v; (void)i;
#endif
    }
    void barrier() {
#if defined(K31_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
};
Comm comm;

// ---------------------------------------------------------------- dummy weights
// Realistic values (no subnormals, no memset): BF16 +-[2^-7, 2^-6) * pattern, scaled per tensor; MXFP4 random nibbles with
// E8M0 scale 2^-6. Same seed, tensor tag and global index -> same value on any rank count.
inline uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}
uint64_t g_seed = 1;
inline uint64_t tag_of(const std::string &s) {
    uint64_t h = 1469598103934665603ull;
    for (char c : s) { h ^= uint8_t(c); h *= 1099511628211ull; }
    return h ^ mix(g_seed);
}

struct Mat {
    int type = DS41_T_BF16, rows = 0, cols = 0;
    std::vector<uint8_t> d;
    const uint8_t *p = nullptr;   // real weights: a view into the arena
    bool tr = false;              // BF16 stored transposed [cols][rows] (k31_bf16_tmatvec)
    bool planar = false;          // MXFP4 in the planar layout (k31_mxfp4p_rows), rows of stride bytes
    size_t stride = 0;
    size_t row_bytes() const { return ds41_row_bytes(type, cols); }
    const uint8_t *data() const { return p ? p : d.data(); }
    const uint8_t *row(int r) const { return data() + size_t(r) * row_bytes(); }
    size_t bytes() const { return p ? row_bytes() * size_t(rows) : d.size(); }
};
size_t g_weight_bytes = 0;

bool g_tcols = true;   // column slices (o_proj, down_proj) stored transposed (--no-tcols: row layout)
void transpose_bf16(Mat &m);

// ---------------------------------------------------------------- real weights: plan / bind
// build_layer() runs three ways. W_DUMMY: generated values (the development path, unchanged). W_PLAN: every weight
// call records a piece (checkpoint tensor, slice, arena offset) for rank g_plan_rank and returns an empty shape.
// W_BIND: the same calls, in the same order, return views of this rank's arena (vectors copied out as F32).
enum WMode { W_DUMMY, W_PLAN, W_BIND };
WMode g_wmode = W_DUMMY;
int g_plan_rank = -1;
std::vector<k31ld::Piece> *g_pieces = nullptr;
size_t g_cursor = 0;
uint64_t g_off = 0;
uint8_t *g_arena = nullptr;
bool g_raw_a = false;
const std::vector<char> *g_plan_keep = nullptr;   // plan only pieces of these tensor ids (nullptr: all)   // --test-raw-a: A_log used as stored (dummy-dump parity tests)

std::string real_name(const std::string &n) {
    const std::string M = "language_model.model.";
    if (n == "lm_head") return "language_model.lm_head.weight";
    if (n.compare(0, 7, "layers.") != 0) return M + n + ".weight";
    const size_t d = n.find('.', 7);
    const std::string pre = M + n.substr(0, d + 1), r = n.substr(d + 1);
    auto starts = [&](const char *s) { return r.compare(0, std::strlen(s), s) == 0; };
    if (starts("experts.")) return pre + "block_sparse_moe." + r;
    if (r == "gate") return pre + "block_sparse_moe.gate.weight";
    if (r == "e_score_correction_bias") return pre + "block_sparse_moe.gate.e_score_correction_bias";
    if (starts("routed_expert_") || starts("shared_experts.")) return pre + "block_sparse_moe." + r + ".weight";
    if (starts("mlp.") || r == "input_layernorm" || r == "post_attention_layernorm" || starts("self_attention_res_") ||
        starts("mlp_res_"))
        return pre + r + ".weight";
    if (r == "A") return pre + "self_attn.A_log";
    if (r == "dt_bias") return pre + "self_attn.dt_bias";
    if (starts("indexer.k_norm.")) return pre + "self_attn." + r;
    return pre + "self_attn." + r + ".weight";
}

// --dump DIR (dummy weights, one rank): every generated tensor written as a checkpoint (BF16 / F32 / compressed-tensors
// MXFP4), for loader parity tests on a laptop
struct DumpT { std::string name, dtype; std::vector<int64_t> shape; std::vector<uint8_t> d; int layer = -1; };
bool g_dump = false;
std::vector<DumpT> g_dumped;
int dump_layer(const std::string &n) { return n.compare(0, 7, "layers.") == 0 ? std::atoi(n.c_str() + 7) : -1; }
void dump_add(const std::string &n, const char *dt, std::vector<int64_t> shape, const uint8_t *p, size_t bytes) {
    DumpT t; t.name = real_name(n); t.dtype = dt; t.shape = std::move(shape); t.d.assign(p, p + bytes); t.layer = dump_layer(n);
    g_dumped.push_back(std::move(t));
}
void dump_mx(const std::string &n, int rows, int cols, const uint8_t *blocks) {
    const int nb = cols / 32;
    std::vector<uint8_t> pk(size_t(rows) * cols / 2), sc(size_t(rows) * nb);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = blocks + (size_t(r) * nb + b) * 17;
            sc[size_t(r) * nb + b] = blk[0];
            uint8_t q[32];
            for (int j = 0; j < 16; ++j) { q[j] = blk[1 + j] & 15; q[j + 16] = blk[1 + j] >> 4; }
            for (int i = 0; i < 16; ++i) pk[size_t(r) * cols / 2 + size_t(b) * 16 + i] = uint8_t(q[2 * i] | q[2 * i + 1] << 4);
        }
    dump_add(n + ".weight_packed", "U8", {rows, cols / 2}, pk.data(), pk.size());
    g_dumped.back().name = real_name(n) + ".weight_packed";
    dump_add(n + ".weight_scale", "U8", {rows, nb}, sc.data(), sc.size());
    g_dumped.back().name = real_name(n) + ".weight_scale";
}
void dump_write(const std::string &dir) {
    std::map<int, std::vector<const DumpT *>> files;
    for (const DumpT &t : g_dumped) files[t.layer].push_back(&t);
    std::string index = "{\"metadata\":{},\"weight_map\":{";
    bool first_i = true;
    for (auto &f : files) {
        const std::string fn = "model-" + std::to_string(f.first + 1) + ".safetensors";
        std::string h = "{";
        uint64_t off = 0;
        for (const DumpT *t : f.second) {
            if (h.size() > 1) h += ",";
            h += "\"" + t->name + "\":{\"dtype\":\"" + t->dtype + "\",\"shape\":[";
            for (size_t i = 0; i < t->shape.size(); ++i) h += (i ? "," : "") + std::to_string(t->shape[i]);
            h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t->d.size()) + "]}";
            off += t->d.size();
            index += std::string(first_i ? "" : ",") + "\"" + t->name + "\":\"" + fn + "\"";
            first_i = false;
        }
        h += "}";
        while (h.size() % 8) h += " ";
        FILE *o = std::fopen((dir + "/" + fn).c_str(), "wb");
        if (!o) throw std::runtime_error("k31: cannot write " + dir + "/" + fn);
        const uint64_t n = h.size();
        std::fwrite(&n, 8, 1, o); std::fwrite(h.data(), 1, h.size(), o);
        for (const DumpT *t : f.second) std::fwrite(t->d.data(), 1, t->d.size(), o);
        std::fclose(o);
    }
    index += "}}";
    FILE *o = std::fopen((dir + "/model.safetensors.index.json").c_str(), "wb");
    if (!o) throw std::runtime_error("k31: cannot write the index");
    std::fwrite(index.data(), 1, index.size(), o);
    std::fclose(o);
}

const k31ld::Piece &wpiece(k31ld::Piece p) {
    if (g_wmode == W_PLAN) {
        p.bytes = k31ld::piece_bytes(p);
        p.off = g_off;
        p.dst = g_plan_rank;
        g_off = (g_off + p.bytes + 255) & ~uint64_t(255);
        if (g_plan_keep && (p.tid >= g_plan_keep->size() || !(*g_plan_keep)[p.tid])) { static k31ld::Piece dummy; dummy = p; return dummy; }
        g_pieces->push_back(p);
        return g_pieces->back();
    }
    if (g_cursor >= g_pieces->size()) throw std::runtime_error("k31: bind ran past the plan at " + p.tensor());
    const k31ld::Piece &q = (*g_pieces)[g_cursor++];
    if (q.tid != p.tid || q.kind != p.kind || q.r0 != p.r0 || q.rn != p.rn || q.c0 != p.c0 || q.cn != p.cn)
        throw std::runtime_error("k31: bind does not match the plan at " + p.tensor());
    g_weight_bytes += q.bytes;
    return q;
}
Mat real_mat(k31ld::Kind kind, const std::string &name, int type, int rows, int cols, int64_t r0, int64_t c0, int64_t cn,
             int64_t gcols) {
    k31ld::Piece p;
    p.kind = kind; p.tid = k31ld::name_id(real_name(name)); p.r0 = int32_t(r0); p.rn = rows; p.c0 = int32_t(c0); p.cn = int32_t(cn); p.cols = int32_t(gcols);
    const k31ld::Piece &q = wpiece(p);
    Mat m; m.type = type; m.rows = rows; m.cols = cols;
    if (type == DS41_T_MXFP4 && k31ld::mx_planar(cn)) { m.planar = true; m.stride = k31ld::mx_row_bytes(cn); }
    if (g_wmode == W_BIND) {
        m.p = g_arena + q.off;
        if (kind == k31ld::COLS && g_tcols) transpose_bf16(m);
    }
    return m;
}

// rows [g0, g0 + rows) of a global [*, cols] BF16 tensor; scale ~ 1 / sqrt(cols) keeps activations O(1)
Mat bf16_mat(const std::string &name, int g0, int rows, int cols) {
    if (g_wmode != W_DUMMY) return real_mat(k31ld::ROWS, name, DS41_T_BF16, rows, cols, g0, 0, 0, cols);
    Mat m; m.type = DS41_T_BF16; m.rows = rows; m.cols = cols;
    m.d.resize(size_t(rows) * cols * 2);
    const uint64_t t = tag_of(name);
    const float s = 2.0f / std::sqrt(float(cols));
    uint16_t *w = reinterpret_cast<uint16_t *>(m.d.data());
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const uint64_t h = mix(t + (uint64_t(g0 + r) << 24) + uint64_t(c));
            const float v = (1.0f + float(h & 0xffff) / 65536.0f) * s * ((h >> 16) & 1 ? -1.0f : 1.0f);
            w[size_t(r) * cols + c] = k31_f32_to_bf16(v);
        }
    g_weight_bytes += m.bytes();
    if (g_dump) dump_add(name, "BF16", {rows, cols}, m.d.data(), m.d.size());
    return m;
}
// columns [c0, c0 + n) of a global [rows, gcols] BF16 tensor (an o_proj's local heads)
Mat bf16_cols(const std::string &name, int rows, int gcols, int c0, int n) {
    if (g_wmode != W_DUMMY) return real_mat(k31ld::COLS, name, DS41_T_BF16, rows, n, 0, c0, n, gcols);
    Mat m; m.type = DS41_T_BF16; m.rows = rows; m.cols = n;
    m.d.resize(size_t(rows) * n * 2);
    const uint64_t t = tag_of(name);
    const float s = 2.0f / std::sqrt(float(gcols));
    uint16_t *w = reinterpret_cast<uint16_t *>(m.d.data());
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < n; ++c) {
            const uint64_t h = mix(t + (uint64_t(r) << 24) + uint64_t(c0 + c));
            const float v = (1.0f + float(h & 0xffff) / 65536.0f) * s * ((h >> 16) & 1 ? -1.0f : 1.0f);
            w[size_t(r) * n + c] = k31_f32_to_bf16(v);
        }
    g_weight_bytes += m.bytes();
    if (g_dump) dump_add(name, "BF16", {rows, n}, m.d.data(), m.d.size());
    if (comm.sim && g_tcols) transpose_bf16(m);   // sim timing: the real-weight layout
    return m;
}
// MXFP4 in the GGUF block layout ds41_rows reads (E8M0 + 16 bytes per 32 weights)
int g_cols_hint = 0;   // a slice's global column count (expert-TP), for the dummy scale
Mat mxfp4_mat(const std::string &name, int rows, int cols, int g_r0 = 0, int g_b0 = 0) {
    if (g_wmode != W_DUMMY)
        return real_mat(k31ld::MX, name, DS41_T_MXFP4, rows, cols, g_r0, g_b0, cols / 32, g_cols_hint > 0 ? g_cols_hint : cols);
    Mat m; m.type = DS41_T_MXFP4; m.rows = rows; m.cols = cols;
    m.d.resize(size_t(rows) * (cols / 32) * 17);
    const uint64_t t = tag_of(name);
    const int nb = cols / 32;
    // scale so that a row's dot with O(1) inputs is O(1): e2m1 values up to 6, 2^-e with e ~ log2(6 sqrt(cols))
    const int e = int(std::lround(std::log2(6.0 * std::sqrt(double(g_cols_hint > 0 ? g_cols_hint : cols)))));
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t *blk = m.d.data() + (size_t(r) * nb + b) * 17;
            blk[0] = uint8_t(127 - e);
            uint64_t h = mix(t + (uint64_t(g_r0 + r) << 20) + uint64_t(g_b0 + b));
            for (int j = 0; j < 16; ++j) { if (j == 8) h = mix(h); blk[1 + j] = uint8_t(h >> (8 * (j & 7))); }
        }
    g_weight_bytes += m.bytes();
    if (g_dump) dump_mx(name, rows, cols, m.d.data());
    if (k31ld::mx_planar(nb)) {   // the real-weight layout (same values)
        const uint64_t rb = k31ld::mx_row_bytes(nb);
        std::vector<uint8_t> t(size_t(rb * rows), 0);
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = m.d.data() + (size_t(r) * nb + b) * 17;
                t[size_t(rb * r) + size_t(nb) * 16 + size_t(b)] = blk[0];
                std::memcpy(t.data() + rb * r + size_t(b) * 16, blk + 1, 16);
            }
        m.d.swap(t); m.planar = true; m.stride = rb;
    }
    return m;
}
// e0: the first element of this rank's slice (real weights only; the dummy values keep their old indexing)
Vec f32_vec(const std::string &name, int n, float base, float spread, int64_t e0 = 0) {
    if (g_wmode != W_DUMMY) {
        k31ld::Piece p;
        p.kind = k31ld::VEC; p.tid = k31ld::name_id(real_name(name)); p.c0 = int32_t(e0); p.cn = n;
        const k31ld::Piece &q = wpiece(p);
        Vec v(static_cast<size_t>(n), 0.f);
        if (g_wmode == W_BIND) {
            std::memcpy(v.data(), g_arena + q.off, size_t(n) * 4);
            if (!g_raw_a && name.size() > 2 && name.compare(name.size() - 2, 2, ".A") == 0)
                for (float &a : v) a = -std::exp(a);   // A_log -> A = -exp(A_log), the form k31_kda_log_decay takes
        }
        return v;
    }
    Vec v(static_cast<size_t>(n));
    const uint64_t t = tag_of(name);
    for (int i = 0; i < n; ++i) v[size_t(i)] = base + spread * (float(mix(t + uint64_t(i)) & 0xffff) / 65536.0f - 0.5f);
    g_weight_bytes += v.size() * 4;
    if (g_dump) dump_add(name, "F32", {n}, reinterpret_cast<const uint8_t *>(v.data()), v.size() * 4);
    return v;
}

// ---------------------------------------------------------------- matvec
int g_threads = 1;
bool g_k31mx = true;
int64_t g_par_min = 65536;
bool g_dynamic = false;   // --dynamic: the old schedule(dynamic, 1) over small row chunks (default: static, ~2 tasks per thread)   // --par-min N: matvecs with fewer weights than this run on the calling thread   // MXFP4 through k31_mxfp4_rows (--ds41-mx: ds41_rows)
// a Q8_0 activation laid out for k31_mxfp4_rows: int8 values [n] and block scales [n / 32]
struct MxAct { std::vector<int8_t> q, qp; std::vector<float> d, d4; std::vector<ds41_q80> t; };
void mx_act(const float *x, int n, MxAct &a) {
    a.t.resize(size_t(n / 32)); a.q.resize(size_t(n)); a.d.resize(size_t(n / 32));
    ds41_quant_q80(x, n, a.t.data());
    for (int b = 0; b < n / 32; ++b) { std::memcpy(a.q.data() + size_t(b) * 32, a.t[size_t(b)].q, 32); a.d[size_t(b)] = a.t[size_t(b)].d; }
    if (k31ld::g_planar && (n / 32) % 4 == 0) {
        a.qp.resize(size_t(n)); a.d4.resize(size_t(n / 8));
        k31_mxfp4p_act(a.q.data(), a.d.data(), n / 32, a.qp.data(), a.d4.data());
    }
}
inline void mx_rows(const Mat &m, int r0, int r1, const MxAct &a, float *y) {
    if (m.planar) k31_mxfp4p_rows(m.data(), m.stride, m.cols, r0, r1, a.qp.data(), a.d4.data(), y);
    else k31_mxfp4_rows(m.data(), m.cols, r0, r1, a.q.data(), a.d.data(), y);
}
std::vector<ds41_q80> g_q80;
// y[0..m.rows) = m . x (x has m.cols floats)
MxAct g_mx1;
// transpose a BF16 [rows][cols] matrix in place (into [cols][rows])
void transpose_bf16(Mat &m) {
    uint16_t *w = const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(m.data()));
    std::vector<uint16_t> t(size_t(m.rows) * m.cols);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < m.rows; ++r)
        for (int c = 0; c < m.cols; ++c) t[size_t(c) * m.rows + r] = w[size_t(r) * m.cols + c];
    std::memcpy(w, t.data(), t.size() * 2);
    m.tr = true;
}
void mv(const Mat &m, const float *x, float *y) {
    if (m.tr) {
        const int rows = m.rows, chunk = std::max(64, ((rows + g_threads - 1) / g_threads + 63) / 64 * 64);
        const uint16_t *wt = reinterpret_cast<const uint16_t *>(m.data());
#pragma omp parallel for schedule(static) if (int64_t(rows) * m.cols >= g_par_min)
        for (int r0 = 0; r0 < rows; r0 += chunk) k31_bf16_tmatvec(wt, rows, m.cols, x, y, r0, std::min(rows, r0 + chunk));
        return;
    }
    if (g_k31mx && m.type == DS41_T_MXFP4) {
        mx_act(x, m.cols, g_mx1);
        const int rows = m.rows, chunk = std::max(4, (rows + g_threads * (g_dynamic ? 4 : 2) - 1) / (g_threads * (g_dynamic ? 4 : 2)));
#pragma omp parallel for schedule(runtime) if (int64_t(rows) * m.cols >= g_par_min)
        for (int r0 = 0; r0 < rows; r0 += chunk) mx_rows(m, r0, std::min(rows, r0 + chunk), g_mx1, y);
        return;
    }
    const void *act = x;
    if (ds41_act_kind(m.type) == DS41_ACT_Q80) {
        g_q80.resize(size_t(m.cols / 32));
        ds41_quant_q80(x, m.cols, g_q80.data());
        act = g_q80.data();
    }
    const void *acts[1] = {act};
    const int rows = m.rows, chunk = std::max(8, (rows + g_threads * (g_dynamic ? 4 : 2) - 1) / (g_threads * (g_dynamic ? 4 : 2)));
#pragma omp parallel for schedule(runtime) if (int64_t(rows) * m.cols >= g_par_min)
    for (int r0 = 0; r0 < rows; r0 += chunk)
        ds41_rows(m.type, m.data(), m.cols, r0, std::min(rows, r0 + chunk), acts, 1, y, rows);
}

// several matvecs in one parallel region (one fork / join): job i computes ys[i] = ms[i] . xs[i]. Same row kernel and
// per-row sums as mv(), so the results are bit-identical to separate mv() calls.
struct MvJob { const Mat *m; const float *x; float *y; };
double g_mvm_prep = 0, g_mvm_region = 0;
long g_mvm_calls = 0;
std::vector<std::vector<ds41_q80>> g_q80m;
bool g_fuse = true;   // K31_FUSE=0: separate mv() calls (the old path)
void mv_multi(const MvJob *J, int nj) {
    bool any_tr = false;
    for (int i = 0; i < nj; ++i) any_tr |= J[i].m->tr;
    if (!g_fuse || any_tr) { for (int i = 0; i < nj; ++i) mv(*J[i].m, J[i].x, J[i].y); return; }
    if (nj > 40) throw std::runtime_error("k31: mv_multi takes at most 40 jobs");
    const double tp0 = now();
    if (int(g_q80m.size()) < nj) g_q80m.resize(size_t(nj));
    const void *act[40];
    static MxAct mxa[40];
    struct T { int j, r0, r1; };
    static std::vector<T> tasks;   // read by the whole team: not thread_local
    tasks.clear();
    int total = 0;
    int64_t work = 0;
    for (int i = 0; i < nj; ++i) { total += J[i].m->rows; work += int64_t(J[i].m->rows) * J[i].m->cols; }
    const int chunk_all = std::max(2, (total + g_threads * 3 - 1) / (g_threads * 3));
    for (int i = 0; i < nj; ++i) {
        const Mat &m = *J[i].m;
        act[i] = J[i].x;
        if (g_k31mx && m.type == DS41_T_MXFP4) {
            int prev = -1;
            for (int k = 0; k < i; ++k) if (J[k].x == J[i].x && J[k].m->type == DS41_T_MXFP4 && J[k].m->cols == m.cols) prev = k;
            if (prev >= 0) act[i] = act[prev];
            else { mx_act(J[i].x, m.cols, mxa[i]); act[i] = &mxa[i]; }
        } else if (ds41_act_kind(m.type) == DS41_ACT_Q80) {
            int prev = -1;   // reuse an earlier job's quantization of the same input
            for (int k = 0; k < i; ++k) if (J[k].x == J[i].x && J[k].m->cols == m.cols && ds41_act_kind(J[k].m->type) == DS41_ACT_Q80) prev = k;
            if (prev >= 0) act[i] = act[prev];
            else { g_q80m[size_t(i)].resize(size_t(m.cols / 32)); ds41_quant_q80(J[i].x, m.cols, g_q80m[size_t(i)].data()); act[i] = g_q80m[size_t(i)].data(); }
        }
        int chunk;
        if (g_dynamic) chunk = std::min(std::max(8, (m.rows + g_threads * 4 - 1) / (g_threads * 4)), chunk_all);
        else {   // equal work per task (rows x cols), about 2 tasks per thread over all jobs
            const int64_t per = std::max<int64_t>(1, work / (2 * g_threads));
            chunk = int(std::max<int64_t>(4, (per + m.cols - 1) / m.cols));
        }
        for (int r0 = 0; r0 < m.rows; r0 += chunk) tasks.push_back({i, r0, std::min(m.rows, r0 + chunk)});
    }
    const int nt = int(tasks.size());
    const double tp1 = now();
    g_mvm_prep += tp1 - tp0; ++g_mvm_calls;
#pragma omp parallel for schedule(runtime) if (work >= g_par_min)
    for (int t = 0; t < nt; ++t) {
        const T &k = tasks[size_t(t)];
        const Mat &m = *J[k.j].m;
        if (g_k31mx && m.type == DS41_T_MXFP4) {
            mx_rows(m, k.r0, k.r1, *static_cast<const MxAct *>(act[k.j]), J[k.j].y);
            continue;
        }
        const void *a[1] = {act[k.j]};
        ds41_rows(m.type, m.data(), m.cols, k.r0, k.r1, a, 1, J[k.j].y, m.rows);
    }
    g_mvm_region += now() - tp1;
}

void mv_multi(std::initializer_list<MvJob> jobs) { mv_multi(jobs.begin(), int(jobs.size())); }

// RMSNorm with a 16-lane float sum of squares (vectorizes; k31_rmsnorm sums in double, one lane). --serial-res: k31_rmsnorm
bool g_par_res = true;
void rmsnorm_f(float *o, const float *x, const float *w, int n, float eps) {
    if (!g_par_res) { k31_rmsnorm(o, x, w, n, eps); return; }
    float a[16] = {0};
    int i = 0;
    for (; i + 16 <= n; i += 16)
        for (int l = 0; l < 16; ++l) a[l] += x[i + l] * x[i + l];
    float s2 = 0;
    for (int l = 0; l < 16; ++l) s2 += a[l];
    for (; i < n; ++i) s2 += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(s2 / float(n) + eps);
    for (int j = 0; j < n; ++j) o[j] = x[j] * inv * w[j];
}

// ---------------------------------------------------------------- timers
std::map<std::string, double> g_tm;
struct Lap {
    double t = now();
    void operator()(const char *k) { const double t2 = now(); g_tm[k] += t2 - t; t = t2; }
};

// ---------------------------------------------------------------- layers
constexpr int H = K31_HIDDEN, HD = K31_HEAD_DIM;
constexpr float EPS = 1e-5f;
constexpr int KV_BS = 64;

struct Kda {
    int hl = 0, h0 = 0;
    Mat q, k, v, g, fa, fb, b, o;
    Vec qc, kc, vc, a, dt, onorm;   // conv weights [ch][4], A [hl], dt_bias [ch], o_norm [128]
    Vec qs, ks, vs, S;              // conv states, recurrent state [hl][128][128]
};
struct Dsa {
    int hl = 0, h0 = 0;
    Mat qa, kva, qb, kvb, g, o, iqb, iwk, iwp;
    int pr0[4] = {0, 0, 0, 0}, prn[4] = {K31_Q_LORA, K31_LAT, K31_IDX_DIM, K31_IDX_HEADS};   // q_a, kv_a, wk, weights_proj rows
    Vec qa_norm, kv_norm, ik_w, ik_b;
    std::vector<uint16_t> cache, ikeys;   // local token slots
    int64_t cap = 0;                      // local slots
};
struct Moe {
    bool dense = false;
    int lat0 = 0, latn = 0, sh0 = 0, shn = 0, out0 = 0, outn = 0, e0 = 0, en = K31_EXPERTS;   // router rows [e0, e0+en)
    Mat router, ldown, lup, sg, su, sd;   // ldown rows [lat0, lat0+latn) of [3584, H]; lup rows [out0, out0+outn) of [H, 3584]
    Vec bias, lnorm;
    std::vector<int> eid;                 // local expert ids
    std::vector<Mat> w1, w3, w2;          // gate, up [3072, 3584], down [3584, 3072] (expert-TP: rows / columns [tp_i0, +tp_in))
    int tp_i0 = 0, tp_in = 0;
};
struct Layer {
    int idx = 0;
    bool dsa = false;
    Vec in_norm, post_norm, ares_n, ares_p, mres_n, mres_p;
    Kda kda;
    Dsa a;
    Moe moe;
};

bool g_no_moe = false;
int g_sim_k = 2;
// --dsa-hp: head-parallel DSA. Every rank keeps the whole latent cache and indexer keys (it computes kv_a and the
// indexer key itself: those weights are replicated), selects the same top-k, and attends its own heads only: no query
// gather, no softmax merge; only the o_proj partial sum remains.
bool g_dsa_hp = false;
int g_ar_bench = 0;   // --ar-bench N: time N back-to-back all-reduces (7168, 4480 and 96 floats) before decoding
// router rows split over ranks (default; --router-repl: every rank the whole router); the logit slices ride in the latent
// all-reduce (disjoint slices of one vector: exact)
bool g_router_shard = true;
// --dsa-shard (with --dsa-hp): q_a, kv_a and the indexer's wk / weights_proj rows split over ranks, one all-reduce of
// the disjoint slices gives every rank the full vectors
bool g_dsa_shard = false;   // --sim-experts K: routed experts this rank computes per layer in --sim-world mode
bool g_expert_tp = false;
int g_etp_g = 0;   // --expert-tp G: groups of G ranks; expert e on group e % (W / G), its inter split over the group (0: all W)
int etp_g() { return g_etp_g > 0 ? g_etp_g : comm.world; }   // --expert-tp: every expert split over all ranks (inter rows of w1/w3, inter columns of w2)   // --no-moe: attention, residuals and caches only (memory for long-context tests)
int split0(int n, int r) { return int(int64_t(n) * r / comm.world); }

Layer build_layer(int li, int64_t ctx) {
    Layer L; L.idx = li; L.dsa = k31_is_dsa_layer(li);
    const std::string p = "layers." + std::to_string(li) + ".";
    const int W = comm.world, R = g_plan_rank >= 0 ? g_plan_rank : comm.rank;
    L.in_norm = f32_vec(p + "input_layernorm", H, 1.0f, 0.2f);
    L.post_norm = f32_vec(p + "post_attention_layernorm", H, 1.0f, 0.2f);
    L.ares_n = f32_vec(p + "self_attention_res_norm", H, 1.0f, 0.2f);
    L.ares_p = f32_vec(p + "self_attention_res_proj", H, 0.0f, 0.02f);
    L.mres_n = f32_vec(p + "mlp_res_norm", H, 1.0f, 0.2f);
    L.mres_p = f32_vec(p + "mlp_res_proj", H, 0.0f, 0.02f);
    const int h0 = split0(K31_HEADS, R), hl = split0(K31_HEADS, R + 1) - h0;
    if (!L.dsa) {
        Kda &K = L.kda; K.h0 = h0; K.hl = hl;
        const int ch = hl * HD, c0 = h0 * HD;
        K.q = bf16_mat(p + "q_proj", c0, ch, H); K.k = bf16_mat(p + "k_proj", c0, ch, H);
        K.v = bf16_mat(p + "v_proj", c0, ch, H); K.g = bf16_mat(p + "g_proj", c0, ch, H);
        K.fa = bf16_mat(p + "f_a_proj", 0, HD, H); K.fb = bf16_mat(p + "f_b_proj", c0, ch, HD);
        K.b = bf16_mat(p + "b_proj", h0, hl, H);
        K.o = bf16_cols(p + "o_proj", H, K31_HEADS * HD, c0, ch);
        K.qc = f32_vec(p + "q_conv1d", ch * K31_CONV, 0.25f, 0.2f, int64_t(c0) * K31_CONV);
        K.kc = f32_vec(p + "k_conv1d", ch * K31_CONV, 0.25f, 0.2f, int64_t(c0) * K31_CONV);
        K.vc = f32_vec(p + "v_conv1d", ch * K31_CONV, 0.25f, 0.2f, int64_t(c0) * K31_CONV);
        K.a = f32_vec(p + "A", hl, 1.0f, 1.0f, h0);
        K.dt = f32_vec(p + "dt_bias", ch, 0.0f, 1.0f, c0);
        K.onorm = f32_vec(p + "o_norm", HD, 1.0f, 0.2f);
        K.qs.assign(size_t(ch) * (K31_CONV - 1), 0.f); K.ks = K.qs; K.vs = K.qs;
        K.S.assign(size_t(hl) * HD * HD, 0.f);
    } else {
        Dsa &A = L.a; A.h0 = h0; A.hl = hl;
        if (g_dsa_shard) {
            const int full[4] = {K31_Q_LORA, K31_LAT, K31_IDX_DIM, K31_IDX_HEADS};
            for (int i = 0; i < 4; ++i) { A.pr0[i] = split0(full[i], R); A.prn[i] = split0(full[i], R + 1) - A.pr0[i]; }
        }
        A.qa = bf16_mat(p + "q_a_proj", A.pr0[0], A.prn[0], H);
        A.kva = bf16_mat(p + "kv_a_proj_with_mqa", A.pr0[1], A.prn[1], H);
        A.qb = bf16_mat(p + "q_b_proj", h0 * 192, hl * 192, K31_Q_LORA);
        A.kvb = bf16_mat(p + "kv_b_proj", h0 * 256, hl * 256, K31_KV_LORA);
        A.g = bf16_mat(p + "g_proj", h0 * HD, hl * HD, H);
        A.o = bf16_cols(p + "o_proj", H, K31_HEADS * HD, h0 * HD, hl * HD);
        A.iqb = bf16_mat(p + "indexer.wq_b", 0, K31_IDX_HEADS * K31_IDX_DIM, K31_Q_LORA);
        A.iwk = bf16_mat(p + "indexer.wk", A.pr0[2], A.prn[2], H);
        A.iwp = bf16_mat(p + "indexer.weights_proj", A.pr0[3], A.prn[3], H);
        A.qa_norm = f32_vec(p + "q_a_layernorm", K31_Q_LORA, 1.0f, 0.2f);
        A.kv_norm = f32_vec(p + "kv_a_layernorm", K31_KV_LORA, 1.0f, 0.2f);
        A.ik_w = f32_vec(p + "indexer.k_norm.weight", K31_IDX_DIM, 1.0f, 0.2f);
        A.ik_b = f32_vec(p + "indexer.k_norm.bias", K31_IDX_DIM, 0.0f, 0.1f);
        const int64_t blocks = (ctx + KV_BS - 1) / KV_BS;
        A.cap = g_dsa_hp ? blocks * KV_BS : ((blocks + W - 1) / W) * KV_BS;
        A.cache.assign(size_t(A.cap) * K31_LAT, 0);
        A.ikeys.assign(size_t(A.cap) * K31_IDX_DIM, 0);
    }
    Moe &M = L.moe;
    M.dense = li == 0;
    if (g_no_moe) return L;
    if (M.dense) {
        const int inter = 33792, s0 = split0(inter, R), sn = split0(inter, R + 1) - s0;
        M.sh0 = s0; M.shn = sn;
        M.sg = bf16_mat(p + "mlp.gate_proj", s0, sn, H); M.su = bf16_mat(p + "mlp.up_proj", s0, sn, H);
        M.sd = bf16_cols(p + "mlp.down_proj", H, inter, s0, sn);
    } else {
        if (g_router_shard) { M.e0 = split0(K31_EXPERTS, R); M.en = split0(K31_EXPERTS, R + 1) - M.e0; }
        M.router = bf16_mat(p + "gate", M.e0, M.en, H);
        M.bias = f32_vec(p + "e_score_correction_bias", K31_EXPERTS, 0.0f, 0.02f);
        M.lat0 = split0(K31_LATENT, R); M.latn = split0(K31_LATENT, R + 1) - M.lat0;
        M.out0 = split0(H, R); M.outn = split0(H, R + 1) - M.out0;
        M.ldown = bf16_mat(p + "routed_expert_down_proj", M.lat0, M.latn, H);
        M.lup = bf16_mat(p + "routed_expert_up_proj", M.out0, M.outn, K31_LATENT);
        M.lnorm = f32_vec(p + "routed_expert_norm", K31_LATENT, 1.0f, 0.2f);
        const int si = K31_SHARED * K31_EXPERT_INTER, s0 = split0(si, R), sn = split0(si, R + 1) - s0;
        M.sh0 = s0; M.shn = sn;
        M.sg = bf16_mat(p + "shared_experts.gate_proj", s0, sn, H);
        M.su = bf16_mat(p + "shared_experts.up_proj", s0, sn, H);
        M.sd = bf16_cols(p + "shared_experts.down_proj", H, si, s0, sn);
        if (g_expert_tp) {
            const int G = etp_g(), NG = W / G, grp = R / G, j = R % G;
            if (W % G) throw std::runtime_error("k31: --expert-tp G must divide the rank count");
            const int i0 = int(int64_t(K31_EXPERT_INTER) * j / G), in = int(int64_t(K31_EXPERT_INTER) * (j + 1) / G) - i0;
            if (i0 % 32 || in % 32) throw std::runtime_error("k31: expert-TP slices must be multiples of 32");
            M.tp_i0 = i0; M.tp_in = in;
            for (int e = grp; e < K31_EXPERTS; e += NG) {
                const std::string q = p + "experts." + std::to_string(e) + ".";
                M.eid.push_back(e);
                g_cols_hint = K31_LATENT;
                M.w1.push_back(mxfp4_mat(q + "w1", in, K31_LATENT, i0, 0));
                M.w3.push_back(mxfp4_mat(q + "w3", in, K31_LATENT, i0, 0));
                g_cols_hint = K31_EXPERT_INTER;
                M.w2.push_back(mxfp4_mat(q + "w2", K31_LATENT, in, 0, i0 / 32));
            }
            g_cols_hint = 0;
        } else
        for (int e = R; e < K31_EXPERTS; e += W) {
            const std::string q = p + "experts." + std::to_string(e) + ".";
            M.eid.push_back(e);
            M.w1.push_back(mxfp4_mat(q + "w1", K31_EXPERT_INTER, K31_LATENT));
            M.w3.push_back(mxfp4_mat(q + "w3", K31_EXPERT_INTER, K31_LATENT));
            M.w2.push_back(mxfp4_mat(q + "w2", K31_LATENT, K31_EXPERT_INTER));
        }
    }
    return L;
}

// ---------------------------------------------------------------- forward pieces
Vec t_ch, t_ch2, t_ch3, t_ch4, t_ch5, t_small, t_small2;

// x: normed input; out: this rank's partial attention output [H]
void kda_forward(Kda &K, const float *x, float *out) {
    Lap lap;
    const int ch = K.hl * HD;
    t_ch.resize(size_t(ch)); t_ch2.resize(size_t(ch)); t_ch3.resize(size_t(ch)); t_ch4.resize(size_t(ch)); t_ch5.resize(size_t(ch));
    static Vec q, k, v, fa, beta;
    q.assign(static_cast<size_t>(ch), 0.f); k.assign(static_cast<size_t>(ch), 0.f); v.assign(static_cast<size_t>(ch), 0.f); fa.assign(HD, 0.f); beta.assign(static_cast<size_t>(K.hl), 0.f);
    mv_multi({{&K.q, x, t_ch.data()}, {&K.k, x, t_ch2.data()}, {&K.v, x, t_ch3.data()}, {&K.g, x, t_ch4.data()},
              {&K.fa, x, fa.data()}, {&K.b, x, beta.data()}});
    lap("kda.proj");
    k31_conv_step(q.data(), t_ch.data(), K.qs.data(), K.qc.data(), ch, K31_CONV);
    k31_conv_step(k.data(), t_ch2.data(), K.ks.data(), K.kc.data(), ch, K31_CONV);
    k31_conv_step(v.data(), t_ch3.data(), K.vs.data(), K.vc.data(), ch, K31_CONV);
    {   // the three SiLUs in one parallel pass (same per-element function as k31_silu_inplace)
        float *qkv[3] = {q.data(), k.data(), v.data()};
#pragma omp parallel for schedule(static)
        for (int t = 0; t < 3 * ch; ++t) k31_silu_inplace(qkv[t / ch] + t % ch, 1);
    }
    lap("kda.conv");
    mv(K.fb, fa.data(), t_ch5.data());   // decay projection
    lap("kda.fb");
    for (int h = 0; h < K.hl; ++h) {
        k31_l2_normalize(q.data() + h * HD, HD, 1e-6f);
        k31_l2_normalize(k.data() + h * HD, HD, 1e-6f);
        beta[size_t(h)] = k31_sigmoid(beta[size_t(h)]);
    }
    k31_kda_log_decay(t_ch.data(), t_ch5.data(), K.a.data(), K.dt.data(), K.hl, HD, -5.0f);
    for (int i = 0; i < ch; ++i) t_ch[size_t(i)] = std::exp(t_ch[size_t(i)]);
    lap("kda.conv+decay");
    k31_kda_step(t_ch2.data(), q.data(), k.data(), v.data(), t_ch.data(), beta.data(), K.S.data(), K.hl, HD, HD);
    lap("kda.step");
    for (int h = 0; h < K.hl; ++h)
        k31_gated_rmsnorm(t_ch3.data() + h * HD, t_ch2.data() + h * HD, t_ch4.data() + h * HD, K.onorm.data(), HD, EPS);
    mv(K.o, t_ch3.data(), out);
    lap("kda.o");
}

int owner_of(int64_t t) { return int((t / KV_BS) % comm.world); }
int64_t slot_of(int64_t t) { return (t / KV_BS / comm.world) * KV_BS + t % KV_BS; }
int64_t token_of_slot(int64_t s, int r) { return ((s / KV_BS) * comm.world + r) * KV_BS + s % KV_BS; }

struct DsaStats { double selected = 0; int calls = 0; };
DsaStats g_dsa;

void dsa_forward(Dsa &A, const float *x, int64_t pos, float *out) {
    Lap lap;
    const int R = comm.rank, W = comm.world, HL = A.hl;
    static Vec ql, kv, qb, gate, iq, iw, ik, ikn;
    ql.assign(K31_Q_LORA, 0.f); kv.assign(K31_LAT, 0.f); qb.assign(size_t(HL) * 192, 0.f); gate.assign(size_t(HL) * HD, 0.f); iq.assign(K31_IDX_HEADS * K31_IDX_DIM, 0.f); iw.assign(K31_IDX_HEADS, 0.f); ik.assign(K31_IDX_DIM, 0.f); ikn.assign(K31_IDX_DIM, 0.f);
    mv_multi({{&A.qa, x, ql.data()}, {&A.kva, x, kv.data()}, {&A.g, x, gate.data()}, {&A.iwk, x, ik.data()},
              {&A.iwp, x, iw.data()}});
    rmsnorm_f(ql.data(), ql.data(), A.qa_norm.data(), K31_Q_LORA, EPS);
    rmsnorm_f(kv.data(), kv.data(), A.kv_norm.data(), K31_KV_LORA, EPS);   // rope part [512, 576) stays raw (no rotation)
    mv_multi({{&A.qb, ql.data(), qb.data()}, {&A.iqb, ql.data(), iq.data()}});
    k31_layernorm(ikn.data(), ik.data(), A.ik_w.data(), A.ik_b.data(), K31_IDX_DIM, 1e-6f);
    lap("dsa.proj");
    if (owner_of(pos) == R) {
        const int64_t s = slot_of(pos);
        for (int d = 0; d < K31_LAT; ++d) A.cache[size_t(s) * K31_LAT + d] = k31_f32_to_f16(kv[size_t(d)]);
        for (int d = 0; d < K31_IDX_DIM; ++d) A.ikeys[size_t(s) * K31_IDX_DIM + d] = k31_f32_to_f16(ikn[size_t(d)]);
    }
    // local tokens <= pos: local slots [0, n_local)
    const int64_t nb = pos / KV_BS + 1, my_blocks = nb > R ? (nb - 1 - R) / W + 1 : 0;
    int64_t n_local = 0;
    if (my_blocks > 0) {
        const int64_t last_tok_block = R + (my_blocks - 1) * W;   // global block index of my last block
        n_local = (my_blocks - 1) * KV_BS + (last_tok_block == pos / KV_BS ? pos % KV_BS + 1 : KV_BS);
    }
    // indexer: global top-K over all ranks' tokens (dense below K tokens)
    std::vector<int> sel;
    const int K = K31_IDX_TOPK;
    if (pos + 1 <= K) {
        sel.resize(size_t(n_local));
        for (int64_t i = 0; i < n_local; ++i) sel[size_t(i)] = int(i);
        lap("dsa.index");
    } else {
        static Vec sc;
        sc.assign(static_cast<size_t>(n_local), 0.f);
        k31_index_scores(iq.data(), iw.data(), K31_IDX_HEADS, K31_IDX_DIM, A.ikeys.data(), int(n_local),
                         1.0f / std::sqrt(float(K31_IDX_HEADS * K31_IDX_DIM)), sc.data());
        lap("dsa.index.scores");
        // local top-K by (score desc, slot asc), linear time; k31_topk_indices (heap) is the reference for the same order
        std::vector<int> loc(static_cast<size_t>(n_local));
        for (int64_t i = 0; i < n_local; ++i) loc[size_t(i)] = int(i);
        const int m = int(std::min<int64_t>(K, n_local));
        if (m < n_local)
            std::nth_element(loc.begin(), loc.begin() + m, loc.end(), [&](int a, int b) {
                return sc[size_t(a)] != sc[size_t(b)] ? sc[size_t(a)] > sc[size_t(b)] : a < b;
            });
        loc.resize(size_t(m));
        // candidates (score, global token) from every rank; pad with -inf
        static Vec cs, all_s;
        cs.assign(size_t(K), -INFINITY); all_s.assign(size_t(K) * W, 0.f);
        std::vector<int> ct(size_t(K), -1), all_t(size_t(K) * W);
        for (int i = 0; i < m; ++i) { cs[size_t(i)] = sc[size_t(loc[size_t(i)])]; ct[size_t(i)] = int(token_of_slot(loc[size_t(i)], R)); }
        comm.gather(cs.data(), K, all_s.data()); comm.gatheri(ct.data(), K, all_t.data());
        // global top-K by (score desc, token asc); keep my own
        std::vector<int> ord(all_s.size());
        for (size_t i = 0; i < ord.size(); ++i) ord[i] = int(i);
        std::nth_element(ord.begin(), ord.begin() + K, ord.end(), [&](int a, int b) {
            if (all_s[size_t(a)] != all_s[size_t(b)]) return all_s[size_t(a)] > all_s[size_t(b)];
            return all_t[size_t(a)] < all_t[size_t(b)];
        });
        for (int i = 0; i < K; ++i) {
            const int j = ord[size_t(i)], t = all_t[size_t(j)];
            if (t >= 0 && owner_of(t) == R) sel.push_back(int(slot_of(t)));
        }
        std::sort(sel.begin(), sel.end());
        lap("dsa.index.topk");
    }
    g_dsa.selected += double(sel.size()); ++g_dsa.calls;
    // absorbed queries of my heads: qa[h] = (W_UK^T q_nope, q_rope); gather all heads
    const int QK = 192;
    static Vec qa_mine, qa_all;
    qa_mine.assign(size_t(HL) * K31_LAT, 0.f); qa_all.assign(size_t(K31_HEADS) * K31_LAT, 0.f);
    // one head per task; columns split in two halves so 8 heads still give 16 tasks
#pragma omp parallel for schedule(static) collapse(2)
    for (int h = 0; h < HL; ++h)
        for (int half = 0; half < 2; ++half) {
            const float *qn = qb.data() + size_t(h) * QK;
            float *dst = qa_mine.data() + size_t(h) * K31_LAT;
            const uint16_t *wk = reinterpret_cast<const uint16_t *>(A.kvb.row(h * 256));   // 128 nope rows of head h
            const int c0 = half * (K31_KV_LORA / 2), cn = K31_KV_LORA / 2;
            for (int d = c0; d < c0 + cn; ++d) dst[d] = 0.f;
            for (int r = 0; r < HD; ++r) k31_bf16_axpy(dst + c0, wk + size_t(r) * K31_KV_LORA + c0, qn[r], cn);
            if (half == 0) for (int r = 0; r < K31_ROPE; ++r) dst[K31_KV_LORA + r] = qn[HD + r];
        }
    // ranks hold different head counts only when 96 % W != 0; gather assumes equal counts
    if (K31_HEADS % W) throw std::runtime_error("k31: heads must divide evenly over ranks");
    comm.gather(qa_mine.data(), HL * K31_LAT, qa_all.data());
    lap("dsa.absorb");
    // partial softmax over my selected tokens, all heads
    const float scale = 1.0f / std::sqrt(float(QK));
    static Vec mx, lsum, acc;
    mx.assign(K31_HEADS, -INFINITY); lsum.assign(K31_HEADS, 0.f); acc.assign(size_t(K31_HEADS) * K31_KV_LORA, 0.f);
    const int ns = int(sel.size());
#pragma omp parallel for schedule(static)
    for (int h = 0; h < K31_HEADS; ++h) {
        const float *q = qa_all.data() + size_t(h) * K31_LAT;
        std::vector<float> s(size_t(std::max(ns, 1)));
        float m = -INFINITY;
        for (int i = 0; i < ns; ++i) {
            const uint16_t *c = A.cache.data() + size_t(sel[size_t(i)]) * K31_LAT;
            float a = 0;
            for (int d = 0; d < K31_LAT; ++d) a += q[d] * k31_f16_to_f32(c[d]);
            s[size_t(i)] = a * scale; m = std::max(m, s[size_t(i)]);
        }
        mx[size_t(h)] = m;
        float l = 0;
        float *o = acc.data() + size_t(h) * K31_KV_LORA;
        for (int i = 0; i < ns; ++i) {
            const float p = std::exp(s[size_t(i)] - m);
            l += p;
            const uint16_t *c = A.cache.data() + size_t(sel[size_t(i)]) * K31_LAT;
            for (int d = 0; d < K31_KV_LORA; ++d) o[d] += p * k31_f16_to_f32(c[d]);
        }
        lsum[size_t(h)] = l;
    }
    lap("dsa.attn");
    Vec gm = mx;
    comm.max(gm.data(), K31_HEADS);
    for (int h = 0; h < K31_HEADS; ++h) {
        const float f = mx[size_t(h)] == -INFINITY ? 0.f : std::exp(mx[size_t(h)] - gm[size_t(h)]);
        lsum[size_t(h)] *= f;
        for (int d = 0; d < K31_KV_LORA; ++d) acc[size_t(h) * K31_KV_LORA + d] *= f;
    }
    comm.sum(lsum.data(), K31_HEADS);
    comm.sum(acc.data(), K31_HEADS * K31_KV_LORA);
    lap("dsa.merge");
    // my heads: v = W_UV (acc / l), gated, o_proj partial
    static Vec vg;
    vg.assign(size_t(HL) * HD, 0.f);
    for (int h = 0; h < HL; ++h) {
        const int gh = A.h0 + h;
        static Vec lat;
        lat.assign(K31_KV_LORA, 0.f);
        const float inv = lsum[size_t(gh)] > 0 ? 1.0f / lsum[size_t(gh)] : 0.f;
        for (int d = 0; d < K31_KV_LORA; ++d) lat[size_t(d)] = acc[size_t(gh) * K31_KV_LORA + d] * inv;
        // W_UV: the head's 128 v rows of kv_b, through the BF16 row kernel
        const void *acts[1] = {lat.data()};
        float vh[HD];
#pragma omp parallel for schedule(static)
        for (int r0 = 0; r0 < HD; r0 += 16)
            ds41_rows(DS41_T_BF16, A.kvb.row(h * 256 + HD), K31_KV_LORA, r0, r0 + 16, acts, 1, vh, HD);
        for (int r = 0; r < HD; ++r) vg[size_t(h) * HD + r] = vh[r] * k31_sigmoid(gate[size_t(h) * HD + r]);
    }
    mv(A.o, vg.data(), out);
    lap("dsa.o");
}

void dsa_forward_hp(Dsa &A, const float *x, int64_t pos, float *out) {
    Lap lap;
    const int HL = A.hl;
    static Vec ql, kv, qb, gate, iq, iw, ik, ikn;
    ql.assign(K31_Q_LORA, 0.f); kv.assign(K31_LAT, 0.f); qb.assign(size_t(HL) * 192, 0.f); gate.assign(size_t(HL) * HD, 0.f); iq.assign(K31_IDX_HEADS * K31_IDX_DIM, 0.f); iw.assign(K31_IDX_HEADS, 0.f); ik.assign(K31_IDX_DIM, 0.f); ikn.assign(K31_IDX_DIM, 0.f);
    if (g_dsa_shard) {
        const int off[4] = {0, K31_Q_LORA, K31_Q_LORA + K31_LAT, K31_Q_LORA + K31_LAT + K31_IDX_DIM};
        static Vec P;
        P.assign(size_t(off[3] + K31_IDX_HEADS), 0.f);
        mv_multi({{&A.qa, x, P.data() + off[0] + A.pr0[0]}, {&A.kva, x, P.data() + off[1] + A.pr0[1]}, {&A.g, x, gate.data()},
                  {&A.iwk, x, P.data() + off[2] + A.pr0[2]}, {&A.iwp, x, P.data() + off[3] + A.pr0[3]}});
        comm.sum(P.data(), int(P.size()));
        std::memcpy(ql.data(), P.data() + off[0], K31_Q_LORA * 4); std::memcpy(kv.data(), P.data() + off[1], K31_LAT * 4);
        std::memcpy(ik.data(), P.data() + off[2], K31_IDX_DIM * 4); std::memcpy(iw.data(), P.data() + off[3], K31_IDX_HEADS * 4);
    } else
        mv_multi({{&A.qa, x, ql.data()}, {&A.kva, x, kv.data()}, {&A.g, x, gate.data()}, {&A.iwk, x, ik.data()},
                  {&A.iwp, x, iw.data()}});
    rmsnorm_f(ql.data(), ql.data(), A.qa_norm.data(), K31_Q_LORA, EPS);
    rmsnorm_f(kv.data(), kv.data(), A.kv_norm.data(), K31_KV_LORA, EPS);
    if (pos + 1 > K31_IDX_TOPK) mv_multi({{&A.qb, ql.data(), qb.data()}, {&A.iqb, ql.data(), iq.data()}});
    else mv(A.qb, ql.data(), qb.data());   // dense selection: the indexer query is not needed
    k31_layernorm(ikn.data(), ik.data(), A.ik_w.data(), A.ik_b.data(), K31_IDX_DIM, 1e-6f);
    lap("dsa.proj");
    for (int d = 0; d < K31_LAT; ++d) A.cache[size_t(pos) * K31_LAT + d] = k31_f32_to_f16(kv[size_t(d)]);
    for (int d = 0; d < K31_IDX_DIM; ++d) A.ikeys[size_t(pos) * K31_IDX_DIM + d] = k31_f32_to_f16(ikn[size_t(d)]);
    const int64_t n = pos + 1;
    const int K = K31_IDX_TOPK;
    std::vector<int> sel;
    if (n <= K) {
        sel.resize(size_t(n));
        for (int64_t i = 0; i < n; ++i) sel[size_t(i)] = int(i);
        lap("dsa.index");
    } else {
        static Vec sc;
        sc.assign(static_cast<size_t>(n), 0.f);
        k31_index_scores(iq.data(), iw.data(), K31_IDX_HEADS, K31_IDX_DIM, A.ikeys.data(), int(n),
                         1.0f / std::sqrt(float(K31_IDX_HEADS * K31_IDX_DIM)), sc.data());
        lap("dsa.index.scores");
        sel.resize(size_t(n));
        for (int64_t i = 0; i < n; ++i) sel[size_t(i)] = int(i);
        std::nth_element(sel.begin(), sel.begin() + K, sel.end(), [&](int a, int b) {
            return sc[size_t(a)] != sc[size_t(b)] ? sc[size_t(a)] > sc[size_t(b)] : a < b;
        });
        sel.resize(size_t(K));
        std::sort(sel.begin(), sel.end());
        lap("dsa.index.topk");
    }
    g_dsa.selected += double(sel.size()); ++g_dsa.calls;
    const int QK = 192, ns = int(sel.size());
    const float scale = 1.0f / std::sqrt(float(QK));
    static Vec vg;
    vg.assign(size_t(HL) * HD, 0.f);
    for (int h = 0; h < HL; ++h) {
        // absorbed query: (W_UK^T q_nope, q_rope)
        static Vec q;
        q.assign(K31_LAT, 0.f);
        const float *qn = qb.data() + size_t(h) * QK;
        const uint16_t *wk = reinterpret_cast<const uint16_t *>(A.kvb.row(h * 256));
        for (int r = 0; r < HD; ++r) k31_bf16_axpy(q.data(), wk + size_t(r) * K31_KV_LORA, qn[r], K31_KV_LORA);
        for (int r = 0; r < K31_ROPE; ++r) q[size_t(K31_KV_LORA + r)] = qn[HD + r];
        // scores over the selected tokens (threads over tokens), softmax, latent sum
        static Vec sc;
        sc.assign(size_t(std::max(ns, 1)), 0.f);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ns; ++i) {
            const uint16_t *c = A.cache.data() + size_t(sel[size_t(i)]) * K31_LAT;
            float a = 0;
            for (int d = 0; d < K31_LAT; ++d) a += q[size_t(d)] * k31_f16_to_f32(c[d]);
            sc[size_t(i)] = a * scale;
        }
        float m = -INFINITY;
        for (int i = 0; i < ns; ++i) m = std::max(m, sc[size_t(i)]);
        float l = 0;
        for (int i = 0; i < ns; ++i) { sc[size_t(i)] = std::exp(sc[size_t(i)] - m); l += sc[size_t(i)]; }
        static Vec lat;
        lat.assign(K31_KV_LORA, 0.f);
        const int DC = 32;   // threads over latent column blocks: each column summed in token order
#pragma omp parallel for schedule(static)
        for (int d0 = 0; d0 < K31_KV_LORA; d0 += DC)
            for (int i = 0; i < ns; ++i) {
                const uint16_t *c = A.cache.data() + size_t(sel[size_t(i)]) * K31_LAT;
                const float p = sc[size_t(i)];
                for (int d = d0; d < d0 + DC; ++d) lat[size_t(d)] += p * k31_f16_to_f32(c[d]);
            }
        const float inv = l > 0 ? 1.0f / l : 0.f;
        for (float &v : lat) v *= inv;
        const void *acts[1] = {lat.data()};
        float vh[HD];
#pragma omp parallel for schedule(static)
        for (int r0 = 0; r0 < HD; r0 += 16)
            ds41_rows(DS41_T_BF16, A.kvb.row(h * 256 + HD), K31_KV_LORA, r0, r0 + 16, acts, 1, vh, HD);
        for (int r = 0; r < HD; ++r) vg[size_t(h) * HD + r] = vh[r] * k31_sigmoid(gate[size_t(h) * HD + r]);
    }
    lap("dsa.attn");
    mv(A.o, vg.data(), out);
    lap("dsa.o");
}

// k31_router_topk's selection and weights (score desc, lowest index on ties; weights in double) with one partial sort
void router_topk(const float *logits, const float *bias, int *idx, float *w) {
    static float prob[K31_EXPERTS], score[K31_EXPERTS];
    static int ord[K31_EXPERTS];
#pragma omp parallel for schedule(static)
    for (int e = 0; e < K31_EXPERTS; ++e) { prob[e] = k31_sigmoid(logits[e]); score[e] = prob[e] + (bias ? bias[e] : 0.0f); ord[e] = e; }
    std::partial_sort(ord, ord + K31_TOPK, ord + K31_EXPERTS,
                      [](int a, int b) { return score[a] != score[b] ? score[a] > score[b] : a < b; });
    double sum = 0.0;
    for (int k = 0; k < K31_TOPK; ++k) { idx[k] = ord[k]; sum += (w[k] = prob[ord[k]]); }
    for (int k = 0; k < K31_TOPK; ++k) w[k] = float(w[k] / sum);
}

// partial MoE output [H] of this rank (dense layer 0 or routed + shared)
void moe_forward(Moe &M, const float *x, float *out) {
    Lap lap;
    std::fill(out, out + H, 0.f);
    static Vec g, u;
    g.assign(static_cast<size_t>(M.shn), 0.f); u.assign(static_cast<size_t>(M.shn), 0.f);
    static Vec lat;
    lat.assign(K31_LATENT + K31_EXPERTS, 0.f);   // [latent | router logits]
    float *logits = lat.data() + K31_LATENT;
    if (M.dense) mv_multi({{&M.sg, x, g.data()}, {&M.su, x, u.data()}});
    else mv_multi({{&M.sg, x, g.data()}, {&M.su, x, u.data()}, {&M.router, x, logits + M.e0}, {&M.ldown, x, lat.data() + M.lat0}});
    k31_situ_fast(g.data(), g.data(), u.data(), M.shn);
    static Vec part;
    part.assign(H, 0.f);
    mv(M.sd, g.data(), part.data());
    for (int i = 0; i < H; ++i) out[i] += part[size_t(i)];
    lap(M.dense ? "dense" : "moe.shared+router+down");
    if (M.dense) return;
    if (M.en == K31_EXPERTS) comm.sum(lat.data(), K31_LATENT);   // disjoint row blocks -> the full latent
    else comm.sum(lat.data(), K31_LATENT + K31_EXPERTS);         // ... and the full router logits
    lap("moe.latsum");
    int idx[K31_TOPK]; float w[K31_TOPK];
    router_topk(logits, M.bias.data(), idx, w);
    lap("moe.topk");
    static Vec rl, eg, eu, eo;
    rl.assign(K31_LATENT, 0.f); eg.assign(K31_EXPERT_INTER, 0.f); eu.assign(K31_EXPERT_INTER, 0.f); eo.assign(K31_LATENT, 0.f);
    int used = 0;
    if (g_expert_tp) {   // this rank's inter slice of its group's selected experts; the latent all-reduce sums the slices
        const int G = etp_g(), NG = comm.world / G, grp = comm.rank / G, in = M.tp_in;
        int loc[K31_TOPK], nl = 0;
        for (int k = 0; k < K31_TOPK; ++k) if (comm.sim ? k % NG == 0 : idx[k] % NG == grp) loc[nl++] = k;
        auto slot = [&](int k) { return comm.sim ? size_t(k / NG) % M.w1.size() : size_t(idx[k] / NG); };
        if (nl) {
            static Vec g1, g3, d2;
            g1.assign(size_t(nl) * in, 0.f); g3.assign(size_t(nl) * in, 0.f); d2.assign(size_t(nl) * K31_LATENT, 0.f);
            static MvJob jb[2 * K31_TOPK];
            for (int i = 0; i < nl; ++i) {
                jb[2 * i] = {&M.w1[slot(loc[i])], lat.data(), g1.data() + size_t(i) * in};
                jb[2 * i + 1] = {&M.w3[slot(loc[i])], lat.data(), g3.data() + size_t(i) * in};
            }
            mv_multi(jb, 2 * nl);
            lap("moe.tp.w13");
            for (int i = 0; i < nl; ++i) {
                k31_situ_fast(g1.data() + size_t(i) * in, g1.data() + size_t(i) * in, g3.data() + size_t(i) * in, in);
                jb[i] = {&M.w2[slot(loc[i])], g1.data() + size_t(i) * in, d2.data() + size_t(i) * K31_LATENT};
            }
            lap("moe.tp.situ");
            mv_multi(jb, nl);
            lap("moe.tp.w2");
            for (int i = 0; i < nl; ++i) {
                for (int d = 0; d < K31_LATENT; ++d) rl[size_t(d)] += w[loc[i]] * d2[size_t(i) * K31_LATENT + d];
                ++used;
            }
        }
    } else {
        // local experts (rank e % W): all w1 / w3 in one parallel region, then all w2; summed in route order
        int loc[K31_TOPK], nl = 0;
        for (int k = 0; k < K31_TOPK; ++k) {
            if (comm.sim) { if (k < g_sim_k) loc[nl++] = k; }   // sim: the first K picks are local
            else if (idx[k] % comm.world == comm.rank) loc[nl++] = k;
        }
        auto slot = [&](int k) { return comm.sim ? size_t(k) % M.w1.size() : size_t(idx[k] / comm.world); };
        static Vec G, U, O;
        G.assign(size_t(std::max(nl, 1)) * K31_EXPERT_INTER, 0.f); U.assign(G.size(), 0.f); O.assign(size_t(std::max(nl, 1)) * K31_LATENT, 0.f);
        if (nl) {
            MvJob jb[16];
            for (int i = 0; i < nl; ++i) {
                const size_t j = slot(loc[i]);
                jb[2 * i] = {&M.w1[j], lat.data(), G.data() + size_t(i) * K31_EXPERT_INTER};
                jb[2 * i + 1] = {&M.w3[j], lat.data(), U.data() + size_t(i) * K31_EXPERT_INTER};
            }
            mv_multi(jb, 2 * nl);
            for (int i = 0; i < nl; ++i)
                k31_situ_fast(G.data() + size_t(i) * K31_EXPERT_INTER, G.data() + size_t(i) * K31_EXPERT_INTER,
                              U.data() + size_t(i) * K31_EXPERT_INTER, K31_EXPERT_INTER);
            for (int i = 0; i < nl; ++i) jb[i] = {&M.w2[slot(loc[i])], G.data() + size_t(i) * K31_EXPERT_INTER, O.data() + size_t(i) * K31_LATENT};
            mv_multi(jb, nl);
            for (int i = 0; i < nl; ++i) {
                for (int d = 0; d < K31_LATENT; ++d) rl[size_t(d)] += w[loc[i]] * O[size_t(i) * K31_LATENT + d];
                ++used;
            }
        }
    }
    (void)eg; (void)eu; (void)eo;
    g_tm["moe.experts_used"] += used;
    lap("moe.experts");
    comm.sum(rl.data(), K31_LATENT);
    lap("moe.rlsum");
    rmsnorm_f(rl.data(), rl.data(), M.lnorm.data(), K31_LATENT, EPS);
    mv(M.lup, rl.data(), part.data());
    for (int i = 0; i < M.outn; ++i) out[M.out0 + i] += part[size_t(i)];
    lap("moe.up");
}

// ---------------------------------------------------------------- model
struct Model {
    std::vector<Layer> L;
    Mat head;          // lm_head rows [v0, v0 + vn)
    Mat emb;           // real weights: embedding rows [v0, v0 + vn)
    bool real_embed = false;
    int v0 = 0, vn = 0;
    Vec final_norm, ores_n, ores_p;
    Vec hidden, blocks;   // block residuals [count][H]
    int block_count = 0;
};

uint64_t fnv(const void *p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t *b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

void embed(int token, float *x) {   // dummy embedding row
    const uint64_t t = tag_of("embed_tokens");
    for (int i = 0; i < H; ++i) x[i] = (float(mix(t + (uint64_t(token) << 24) + uint64_t(i)) & 0xffff) / 65536.0f - 0.5f) * 0.1f;
}

// k31_attn_res over the team: per-block sums in 512-element chunks (partials reduced in chunk order: deterministic),
// then the weighted combination split over elements. --serial-res: the kernel.
void attn_res(float *out, const float *prefix, const float *blocks, int bc, const float *nw, const float *pw, int n, float eps) {
    if (!g_par_res || bc <= 0) { k31_attn_res(out, prefix, blocks, bc, nw, pw, n, eps); return; }
    const int count = bc + 1, CH = 512, nch = (n + CH - 1) / CH;
    std::vector<double> ss(size_t(count) * nch), sp(size_t(count) * nch);
    auto vec = [&](int c) { return c == bc ? prefix : blocks + size_t(c) * n; };
#pragma omp parallel for schedule(static)
    for (int t = 0; t < count * nch; ++t) {
        const int c = t / nch, ch = t % nch, a = ch * CH, b = std::min(n, a + CH);
        const float *v = vec(c);
        double s2 = 0, s1 = 0;
        for (int j = a; j < b; ++j) { s2 += double(v[j]) * v[j]; s1 += double(v[j]) * nw[j] * (pw ? pw[j] : 1.0f); }
        ss[size_t(t)] = s2; sp[size_t(t)] = s1;
    }
    float sc[K31_LAYERS / K31_RES_BLOCK + 2], pr[K31_LAYERS / K31_RES_BLOCK + 2], mx = -INFINITY, z = 0;
    for (int c = 0; c < count; ++c) {
        double s2 = 0, s1 = 0;
        for (int ch = 0; ch < nch; ++ch) { s2 += ss[size_t(c) * nch + ch]; s1 += sp[size_t(c) * nch + ch]; }
        sc[c] = float(s1 / std::sqrt(s2 / n + eps));
        mx = std::max(mx, sc[c]);
    }
    for (int c = 0; c < count; ++c) z += (pr[c] = std::exp(sc[c] - mx));
    for (int c = 0; c < count; ++c) pr[c] /= z;
#pragma omp parallel for schedule(static)
    for (int ch = 0; ch < nch; ++ch)
        for (int j = ch * CH; j < std::min(n, ch * CH + CH); ++j) {
            float v = 0;
            for (int c = 0; c < count; ++c) v += vec(c)[j] * pr[c];
            out[j] = v;
        }
}

// attn_res + RMSNorm in one parallel region: res = the block-residual mix of (blocks, prefix), normed = RMSNorm(res) * nrm.
// Per-chunk partials reduced in chunk order (deterministic). bc <= 0: res = prefix.
bool g_res_serial = false;   // --res-serial: attn_res_norm on the calling thread (16-lane float sums, no barriers)
void attn_res_norm_serial(float *res, float *normed, const float *prefix, const float *blocks, int bc, const float *nw,
                          const float *pw, const float *nrm, int n, float eps) {
    const int count = std::max(bc, 0) + 1;
    auto vec = [&](int c) { return c == count - 1 ? prefix : blocks + size_t(c) * n; };
    if (bc > 0) {
        float sc[16], pr[16], mx = -INFINITY, z = 0;
        for (int c = 0; c < count; ++c) {
            const float *v = vec(c);
            float a2[16] = {0}, a1[16] = {0};
            int j = 0;
            for (; j + 16 <= n; j += 16)
                for (int l = 0; l < 16; ++l) { const float x = v[j + l]; a2[l] += x * x; a1[l] += x * nw[j + l] * (pw ? pw[j + l] : 1.0f); }
            float s2 = 0, s1 = 0;
            for (int l = 0; l < 16; ++l) { s2 += a2[l]; s1 += a1[l]; }
            for (; j < n; ++j) { s2 += v[j] * v[j]; s1 += v[j] * nw[j] * (pw ? pw[j] : 1.0f); }
            sc[c] = s1 / std::sqrt(s2 / float(n) + eps);
            mx = std::max(mx, sc[c]);
        }
        for (int c = 0; c < count; ++c) z += (pr[c] = std::exp(sc[c] - mx));
        for (int c = 0; c < count; ++c) pr[c] /= z;
        for (int j = 0; j < n; ++j) res[j] = 0;
        for (int c = 0; c < count; ++c) { const float *v = vec(c); const float p = pr[c]; for (int j = 0; j < n; ++j) res[j] += v[j] * p; }
    } else std::memcpy(res, prefix, size_t(n) * 4);
    rmsnorm_f(normed, res, nrm, n, eps);
}
// two barriers: each thread owns chunk t of n; partials per (block, chunk); every thread reduces them itself (same order)
bool g_res2 = true;   // --res5: the five-barrier version
void attn_res_norm2(float *res, float *normed, const float *prefix, const float *blocks, int bc, const float *nw, const float *pw,
                    const float *nrm, int n, float eps) {
    const int count = std::max(bc, 0) + 1, T = g_threads;
    static std::vector<double> ss, sp, sq;
    ss.assign(size_t(count) * T, 0); sp.assign(size_t(count) * T, 0); sq.assign(size_t(T), 0);
    auto vec = [&](int c) { return c == count - 1 ? prefix : blocks + size_t(c) * n; };
    const int per = ((n + T - 1) / T + 15) / 16 * 16;
#pragma omp parallel num_threads(T)
    {
#if defined(_OPENMP)
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
        const int a = std::min(n, t * per), b = std::min(n, a + per);
        float pr[16];
        if (bc > 0) {
            for (int c = 0; c < count; ++c) {
                const float *v = vec(c);
                double s2 = 0, s1 = 0;
                for (int j = a; j < b; ++j) { s2 += double(v[j]) * v[j]; s1 += double(v[j]) * nw[j] * (pw ? pw[j] : 1.0f); }
                ss[size_t(c) * T + t] = s2; sp[size_t(c) * T + t] = s1;
            }
#pragma omp barrier
            float sc[16], mx = -INFINITY, z = 0;
            for (int c = 0; c < count; ++c) {
                double s2 = 0, s1 = 0;
                for (int u = 0; u < T; ++u) { s2 += ss[size_t(c) * T + u]; s1 += sp[size_t(c) * T + u]; }
                sc[c] = float(s1 / std::sqrt(s2 / n + eps));
                mx = std::max(mx, sc[c]);
            }
            for (int c = 0; c < count; ++c) z += (pr[c] = std::exp(sc[c] - mx));
            for (int c = 0; c < count; ++c) pr[c] /= z;
        }
        float q = 0;
        for (int j = a; j < b; ++j) {
            float v;
            if (bc > 0) { v = 0; for (int c = 0; c < count; ++c) v += vec(c)[j] * pr[c]; }
            else v = prefix[j];
            res[j] = v;
            q += v * v;
        }
        sq[size_t(t)] = q;
#pragma omp barrier
        float s2 = 0;
        for (int u = 0; u < T; ++u) s2 += float(sq[size_t(u)]);
        const float inv = 1.0f / std::sqrt(s2 / float(n) + eps);
        for (int j = a; j < b; ++j) normed[j] = res[j] * inv * nrm[j];
    }
}
void attn_res_norm(float *res, float *normed, const float *prefix, const float *blocks, int bc, const float *nw, const float *pw,
                   const float *nrm, int n, float eps) {
    if (g_res_serial) { attn_res_norm_serial(res, normed, prefix, blocks, bc, nw, pw, nrm, n, eps); return; }
    if (g_res2 && g_par_res) { attn_res_norm2(res, normed, prefix, blocks, bc, nw, pw, nrm, n, eps); return; }
    if (!g_par_res) {
        attn_res(res, prefix, blocks, bc, nw, pw, n, eps);
        rmsnorm_f(normed, res, nrm, n, eps);
        return;
    }
    const int count = std::max(bc, 0) + 1, CH = 512, nch = (n + CH - 1) / CH;
    static double ss[16 * 16], sp[16 * 16], sq[16];
    static float pr[16];
    static float inv;
    auto vec = [&](int c) { return c == count - 1 ? prefix : blocks + size_t(c) * n; };
#pragma omp parallel
    {
        if (bc > 0) {
#pragma omp for schedule(static)
            for (int t = 0; t < count * nch; ++t) {
                const int c = t / nch, ch = t % nch, a = ch * CH, b = std::min(n, a + CH);
                const float *v = vec(c);
                double s2 = 0, s1 = 0;
                for (int j = a; j < b; ++j) { s2 += double(v[j]) * v[j]; s1 += double(v[j]) * nw[j] * (pw ? pw[j] : 1.0f); }
                ss[t] = s2; sp[t] = s1;
            }
#pragma omp single
            {
                float sc[16], mx = -INFINITY, z = 0;
                for (int c = 0; c < count; ++c) {
                    double s2 = 0, s1 = 0;
                    for (int ch = 0; ch < nch; ++ch) { s2 += ss[c * nch + ch]; s1 += sp[c * nch + ch]; }
                    sc[c] = float(s1 / std::sqrt(s2 / n + eps));
                    mx = std::max(mx, sc[c]);
                }
                for (int c = 0; c < count; ++c) z += (pr[c] = std::exp(sc[c] - mx));
                for (int c = 0; c < count; ++c) pr[c] /= z;
            }
        }
#pragma omp for schedule(static)
        for (int ch = 0; ch < nch; ++ch) {
            float q = 0;
            for (int j = ch * CH; j < std::min(n, ch * CH + CH); ++j) {
                float v;
                if (bc > 0) { v = 0; for (int c = 0; c < count; ++c) v += vec(c)[j] * pr[c]; }
                else v = prefix[j];
                res[j] = v;
                q += v * v;
            }
            sq[ch] = q;
        }
#pragma omp single
        {
            float s2 = 0;
            for (int ch = 0; ch < nch; ++ch) s2 += float(sq[ch]);
            inv = 1.0f / std::sqrt(s2 / float(n) + eps);
        }
#pragma omp for schedule(static)
        for (int j = 0; j < n; ++j) normed[j] = res[j] * inv * nrm[j];
    }
}

// the embedding row of a token: its owner converts the BF16 row, one all-reduce gives it to every rank
void embed_real(Model &m, int token, float *x) {
    std::fill(x, x + H, 0.f);
    if (token >= m.v0 && token < m.v0 + m.vn) {
        const uint16_t *r = reinterpret_cast<const uint16_t *>(m.emb.row(token - m.v0));
        for (int i = 0; i < H; ++i) x[i] = k31_bf16_to_f32(r[i]);
    }
    comm.sum(x, H);
}

int forward(Model &m, int token, int64_t pos, bool head) {
    Vec &h = m.hidden;
    h.resize(H);
    if (m.real_embed) embed_real(m, token, h.data()); else embed(token, h.data());
    m.blocks.assign(size_t(K31_LAYERS / K31_RES_BLOCK + 2) * H, 0.f);
    m.block_count = 0;
    static Vec tmp, tmp2, normed, attn, moe;
    tmp.assign(H, 0.f); tmp2.assign(H, 0.f); normed.assign(H, 0.f); attn.assign(H, 0.f); moe.assign(H, 0.f);
    for (Layer &L : m.L) {
        const double tl = now();
        const double c0 = comm.t;
        Lap fl;
        std::memcpy(tmp.data(), h.data(), H * sizeof(float));
        attn_res_norm(tmp2.data(), normed.data(), tmp.data(), m.blocks.data(), m.block_count, L.ares_n.data(), L.ares_p.data(),
                      L.in_norm.data(), H, EPS);
        const bool new_block = L.idx % K31_RES_BLOCK == 0;
        if (new_block) { std::memcpy(m.blocks.data() + size_t(m.block_count) * H, tmp.data(), H * sizeof(float)); ++m.block_count; }
        fl("fwd.res+norm");
        if (L.dsa) { if (g_dsa_hp) dsa_forward_hp(L.a, normed.data(), pos, attn.data()); else dsa_forward(L.a, normed.data(), pos, attn.data()); }
        else kda_forward(L.kda, normed.data(), attn.data());
        fl("fwd.attn");
        comm.sum(attn.data(), H);
        fl("fwd.attnsum");
        if (new_block) std::memcpy(tmp.data(), attn.data(), H * sizeof(float));
        else for (int i = 0; i < H; ++i) tmp[size_t(i)] = h[size_t(i)] + attn[size_t(i)];
        attn_res_norm(tmp2.data(), normed.data(), tmp.data(), m.blocks.data(), m.block_count, L.mres_n.data(), L.mres_p.data(),
                      L.post_norm.data(), H, EPS);
        fl("fwd.res+norm");
        if (g_no_moe) std::fill(moe.begin(), moe.end(), 0.f);
        else { moe_forward(L.moe, normed.data(), moe.data()); fl("fwd.moe"); comm.sum(moe.data(), H); fl("fwd.moesum"); }
        if (new_block) for (int i = 0; i < H; ++i) h[size_t(i)] = attn[size_t(i)] + moe[size_t(i)];
        else for (int i = 0; i < H; ++i) h[size_t(i)] += attn[size_t(i)] + moe[size_t(i)];
        const std::string k = L.dsa ? "layer.dsa" : (L.idx == 0 ? "layer.dense0" : "layer.kda");
        g_tm[k] += now() - tl;
        g_tm[k + ".comm"] += comm.t - c0;
        g_tm[k + ".n"] += 1;
    }
    attn_res(tmp2.data(), h.data(), m.blocks.data(), m.block_count, m.ores_n.data(), m.ores_p.data(), H, EPS);
    rmsnorm_f(h.data(), tmp2.data(), m.final_norm.data(), H, EPS);
    if (!head) return 0;
    const double th = now();
    static Vec lg;
    lg.assign(static_cast<size_t>(m.vn), 0.f);
    mv(m.head, h.data(), lg.data());
    float best = -INFINITY; int bi = 0;
    for (int i = 0; i < m.vn; ++i) if (lg[size_t(i)] > best) { best = lg[size_t(i)]; bi = m.v0 + i; }
    comm.argmax(best, bi);
    g_tm["head"] += now() - th;
    return bi;
}

// fill the DSA caches with synthetic tokens [0, n) (a long context without running its prompt)
void fill_context(Model &m, int64_t n) {
    for (Layer &L : m.L) {
        if (!L.dsa) continue;
        Dsa &A = L.a;
        const uint64_t t = tag_of("ctx" + std::to_string(L.idx));
#pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < A.cap; ++s) {
            const int64_t tok = g_dsa_hp ? s : token_of_slot(s, comm.rank);
            if (tok >= n) continue;
            for (int d = 0; d < K31_LAT; ++d)
                A.cache[size_t(s) * K31_LAT + d] = k31_f32_to_f16((float(mix(t + (uint64_t(tok) << 20) + uint64_t(d)) & 0xffff) / 65536.0f - 0.5f) * 2.0f);
            for (int d = 0; d < K31_IDX_DIM; ++d)
                A.ikeys[size_t(s) * K31_IDX_DIM + d] = k31_f32_to_f16((float(mix(t + (uint64_t(tok) << 20) + 4096 + uint64_t(d)) & 0xffff) / 65536.0f - 0.5f) * 2.0f);
        }
    }
}

std::vector<int> parse_layers(const std::string &s) {
    std::vector<int> v;
    if (s == "all") { for (int i = 0; i < K31_LAYERS; ++i) v.push_back(i); return v; }
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        const std::string t = s.substr(p, q - p);
        const size_t d = t.find('-');
        if (d == std::string::npos) v.push_back(std::atoi(t.c_str()));
        else for (int i = std::atoi(t.substr(0, d).c_str()); i <= std::atoi(t.substr(d + 1).c_str()); ++i) v.push_back(i);
        p = q + 1;
    }
    for (int i : v) if (i < 0 || i >= K31_LAYERS) throw std::runtime_error("k31: layer index out of range");
    return v;
}

// final norms, the embedding (real weights) and lm_head, in the same order for plan and bind
const int VOCAB = 163840;
void build_head(Model &m, bool head, int R) {
    m.final_norm = f32_vec("norm", H, 1.0f, 0.2f);
    m.ores_n = f32_vec("output_attn_res_norm", H, 1.0f, 0.2f);
    m.ores_p = f32_vec("output_attn_res_proj", H, 0.0f, 0.02f);
    m.v0 = split0(VOCAB, R); m.vn = split0(VOCAB, R + 1) - m.v0;
    if (g_wmode != W_DUMMY && head) { m.emb = bf16_mat("embed_tokens", m.v0, m.vn, H); m.real_embed = true; }
    if (head) m.head = bf16_mat("lm_head", m.v0, m.vn, H);
}

// rank R's pieces for the listed layers (+ head), offsets in R's arena
std::vector<k31ld::Piece> plan_for(int R, const std::vector<int> &layers, bool head, uint64_t *arena_bytes) {
    std::vector<k31ld::Piece> v;
    const WMode wm = g_wmode;
    g_wmode = W_PLAN; g_plan_rank = R; g_pieces = &v; g_off = 0;
    for (int li : layers) (void)build_layer(li, KV_BS);
    Model tmp;
    build_head(tmp, head, R);
    *arena_bytes = g_off;
    g_wmode = wm; g_plan_rank = -1; g_pieces = nullptr;
    return v;
}

uint64_t hash_par(const uint8_t *p, uint64_t n, uint64_t off) {
    if (n < (8u << 20)) return k31ld::hash_words(p, n, off);
    const uint64_t step = 1u << 20;   // 1 MiB slices, word aligned
    const int64_t ns = int64_t((n + step - 1) / step);
    uint64_t h = 0;
#pragma omp parallel for reduction(+ : h) schedule(static)
    for (int64_t i = 0; i < ns; ++i) {
        const uint64_t a = uint64_t(i) * step, len = std::min(step, n - a);
        h += k31ld::hash_words(p + a, len, off + a);
    }
    return h;
}

#if defined(K31_UTOFU)
extern "C" {
int k31x_init(void *arena, size_t arena_bytes, void *send, size_t send_bytes);
int k31x_put(int dst, size_t src_off, size_t dst_off, size_t len);
int k31x_flush(void);
void k31x_settle(void);
void k31x_stats(unsigned long long *puts, unsigned long long *bytes, int *tnis, size_t *put_max);
void k31x_fini(void);
}
#endif

// one source's delivery of packed bytes: to this rank's arena (memcpy-free: packed in place) or Put from the send
// buffer to a peer. Every delivered byte is hashed into hsent[dst].
struct Deliver {
    uint8_t *send = nullptr;
    uint64_t send_cap = 0, send_used = 0;
    std::vector<uint64_t> hsent;
    uint64_t pieces = 0, bytes_local = 0, bytes_remote = 0;
    double t_flush = 0;
    uint8_t *acquire(int dst, uint64_t off, uint64_t len) {
        if (dst == comm.rank) return g_arena + off;
        if (len > send_cap) throw std::runtime_error("k31: a packed chunk exceeds the send buffer");
        if (send_used + len > send_cap) flush();
        return send + send_used;
    }
    void commit(int dst, uint64_t off, uint8_t *p, uint64_t len) {
        if (off % 8) throw std::runtime_error("k31: unaligned piece chunk");
        hsent[size_t(dst)] += hash_par(p, len, off);
        if (dst == comm.rank) { bytes_local += len; return; }
#if defined(K31_UTOFU)
        if (k31x_put(dst, size_t(p - send), size_t(off), size_t(len))) throw std::runtime_error("k31: uTofu put failed");
        send_used = (send_used + len + 255) & ~uint64_t(255);
        bytes_remote += len;
#else
        (void)p;
        throw std::runtime_error("k31: a remote piece needs the uTofu build (-DK31_UTOFU)");
#endif
    }
    void flush() {
#if defined(K31_UTOFU)
        const double t0 = now();
        if (k31x_flush()) throw std::runtime_error("k31: uTofu flush failed");
        t_flush += now() - t0;
#endif
        send_used = 0;
    }
};

// deliver every piece of one checkpoint tensor (all its destinations), reading the source rows once in chunks
struct LoadStats { double t_read = 0, t_pack = 0; uint64_t read_bytes = 0; };
LoadStats g_ls;
std::vector<uint8_t> g_raw, g_raw2;
uint64_t g_chunk = 256u << 20;

void deliver_tensor(const k31ld::File &f, const std::string &name, const std::vector<const k31ld::Piece *> &ps, Deliver &D) {
    using namespace k31ld;
    const bool mx = ps[0]->kind == MX;
    const Tensor *t = f.find(mx ? name + ".weight_packed" : name), *ts = mx ? f.find(name + ".weight_scale") : nullptr;
    if (!t || (mx && !ts)) throw std::runtime_error("k31: tensor " + name + " not in " + f.path);
    if (ps[0]->kind == VEC) {
        const uint64_t n = t->end - t->begin;
        g_raw.resize(size_t(n));
        double t0 = now();
        pread_all(f.fd, g_raw.data(), size_t(n), t->begin, f.path);
        g_ls.t_read += now() - t0; g_ls.read_bytes += n;
        for (const Piece *p : ps) {
            if (p->c0 + p->cn > t->numel()) throw std::runtime_error("k31: vector slice out of range: " + name);
            uint8_t *dst = D.acquire(p->dst, p->off, p->bytes);
            pack_vec(*p, t->dt, dst, g_raw.data());
            D.commit(p->dst, p->off, dst, p->bytes);
            ++D.pieces;
        }
        return;
    }
    // matrix: check the shape, then row chunks over the union of the pieces' rows
    const int64_t rows = t->rows(), cols = ps[0]->cols;
    if (mx) {
        if (t->dt != DT_U8 || t->shape.size() != 2 || t->shape[1] * 2 != cols || ts->shape[1] * 32 != cols)
            throw std::runtime_error("k31: unexpected MXFP4 shape for " + name);
    } else if (t->dt != DT_BF16 || t->shape.size() != 2 || t->shape[1] != cols)
        throw std::runtime_error("k31: expected BF16 [*, " + std::to_string(cols) + "] for " + name);
    int64_t lo = rows, hi = 0;
    for (const Piece *p : ps) if (p->rn == 0 || p->bytes == 0) ++D.pieces;   // empty slices (e.g. 32 rows over 96 ranks)
    for (const Piece *p : ps) {
        const int64_t a = p->kind == COLS ? 0 : p->r0, b = p->kind == COLS ? p->rn : p->r0 + p->rn;
        if (b > rows || (p->kind == COLS && p->c0 + p->cn > cols)) throw std::runtime_error("k31: slice out of range: " + name);
        lo = std::min(lo, a); hi = std::max(hi, b);
    }
    const uint64_t srow = mx ? uint64_t(cols / 2) : uint64_t(cols) * 2, scrow = mx ? uint64_t(cols / 32) : 0;
    int64_t crows = std::max<int64_t>(8, int64_t(g_chunk / srow) & ~int64_t(7));
    for (int64_t a = lo; a < hi; a += crows) {
        const int64_t b = std::min(hi, a + crows);
        double t0 = now();
        g_raw.resize(size_t(uint64_t(b - a) * srow));
        pread_all(f.fd, g_raw.data(), g_raw.size(), t->begin + uint64_t(a) * srow, f.path);
        if (mx) {
            g_raw2.resize(size_t(uint64_t(b - a) * scrow));
            pread_all(f.fd, g_raw2.data(), g_raw2.size(), ts->begin + uint64_t(a) * scrow, f.path);
        }
        g_ls.t_read += now() - t0; g_ls.read_bytes += g_raw.size() + (mx ? g_raw2.size() : 0);
        t0 = now();
        for (const Piece *p : ps) {
            const int64_t pr0 = p->kind == COLS ? 0 : p->r0, pa = std::max(a, pr0), pb = std::min(b, pr0 + p->rn);
            if (pa >= pb || p->bytes == 0) continue;
            const uint64_t rb = piece_row_bytes(*p), off = p->off + uint64_t(pa - pr0) * rb, len = uint64_t(pb - pa) * rb;
            uint8_t *dst = D.acquire(p->dst, off, len);
            const uint8_t *src = g_raw.data() + uint64_t(pa - a) * srow;
            if (p->kind == MX) {
                const uint8_t *sc = g_raw2.data() + uint64_t(pa - a) * scrow;
                const int64_t n = pb - pa;
#pragma omp parallel for schedule(static)
                for (int64_t r0 = 0; r0 < n; r0 += 16)
                    repack_mx_rows(dst + uint64_t(r0) * rb, src + uint64_t(r0) * srow, sc + uint64_t(r0) * scrow,
                                   std::min<int64_t>(16, n - r0), p->cols, p->c0, p->cn);
            } else pack_rows(*p, dst, src, nullptr, pa - pr0, pb - pr0);
            D.commit(p->dst, off, dst, len);
            if (pb == pr0 + p->rn) ++D.pieces;
        }
        g_ls.t_pack += now() - t0;
    }
}

std::vector<std::string> list_safetensors(const std::string &dir) {
    std::vector<std::string> v;
    DIR *d = opendir(dir.c_str());
    if (!d) throw std::runtime_error("k31: cannot open " + dir);
    while (dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) v.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(v.begin(), v.end());
    return v;
}

// load this rank's arena. direct: read my pieces from the model directory myself. staged: every rank reads the
// files in stage_dir (its own share of the checkpoint) and Puts each destination's pieces (uTofu).
struct LoadResult { std::vector<k31ld::Piece> mine; uint64_t arena = 0; };
LoadResult load_real(const std::string &model_dir, const std::string &stage_dir, const std::vector<int> &layers, bool head) {
    using namespace k31ld;
    const double t0 = now();
    LoadResult res;
    res.mine = plan_for(comm.rank, layers, head, &res.arena);
    const uint64_t send_cap = stage_dir.empty() ? 0 : 2 * g_chunk;
    // arena: touched here (first touch on this rank's node), then registered for Puts
    g_arena = static_cast<uint8_t *>(std::aligned_alloc(256, size_t((res.arena + 255) & ~uint64_t(255)) + 256));
    if (!g_arena) throw std::runtime_error("k31: arena allocation failed");
    {
        const int64_t n = int64_t(res.arena / 4096 + 1);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) std::memset(g_arena + uint64_t(i) * 4096, 0, size_t(std::min<uint64_t>(4096, res.arena + 256 - uint64_t(i) * 4096)));
    }
    Deliver D;
    D.hsent.assign(size_t(comm.world), 0);
    std::vector<uint8_t> sendbuf;
    const double t_plan = now() - t0;
    double t_init = 0;
    if (stage_dir.empty()) {
        // direct: group my pieces by file through the index
        const auto idx = read_index(model_dir + "/model.safetensors.index.json");
        std::map<std::string, std::map<std::string, std::vector<const Piece *>>> by_file;
        for (const Piece &p : res.mine) {
            auto it = idx.find(p.kind == MX ? p.tensor() + ".weight_packed" : p.tensor());
            if (it == idx.end()) throw std::runtime_error("k31: " + p.tensor() + " not in the index");
            by_file[it->second][p.tensor()].push_back(&p);
        }
        for (auto &fk : by_file) {
            File f; f.open(model_dir + "/" + fk.first);
            for (auto &tk : fk.second) deliver_tensor(f, tk.first, tk.second, D);
            f.close();
        }
    } else {
#if defined(K31_UTOFU)
        // every rank's plan (deterministic), indexed by tensor
        // the tensors in my staged files; other ranks' plans keep only those pieces
        const std::vector<std::string> my_files = list_safetensors(stage_dir);
        std::vector<char> keep;
        for (const std::string &path : my_files) {
            File f; f.open(path);
            for (const Tensor &t : f.tensors) {
                std::string base = t.name;
                const std::string sp = ".weight_packed";
                if (base.size() > sp.size() && base.compare(base.size() - sp.size(), sp.size(), sp) == 0) base.resize(base.size() - sp.size());
                const uint32_t id = name_id(base);
                if (keep.size() <= id) keep.resize(size_t(id) + 1, 0);
                keep[id] = 1;
            }
            f.close();
        }
        std::vector<std::vector<Piece>> all(size_t(comm.world));
        std::vector<uint64_t> sizes(size_t(comm.world));
        for (int r = 0; r < comm.world; ++r) {
            if (r == comm.rank) continue;
            g_plan_keep = &keep;   // ids interned while planning lie past keep: never mine
            all[size_t(r)] = plan_for(r, layers, head, &sizes[size_t(r)]);
            g_plan_keep = nullptr;
        }
        std::unordered_map<std::string, std::vector<const Piece *>> by_t;
        for (const Piece &p : res.mine) if (p.tid < keep.size() && keep[p.tid]) by_t[p.tensor()].push_back(&p);
        for (auto &v : all) for (const Piece &p : v) by_t[p.tensor()].push_back(&p);
        sendbuf.resize(size_t(send_cap) + 256);
        D.send = sendbuf.data(); D.send_cap = send_cap;
        const double ti = now();
        const int rc = k31x_init(g_arena, size_t(res.arena + 256), D.send, size_t(send_cap));
        if (rc) throw std::runtime_error("k31: k31x_init failed rc=" + std::to_string(rc));
        t_init = now() - ti;
        for (const std::string &path : my_files) {
            File f; f.open(path);
            std::vector<std::pair<uint64_t, std::string>> order;   // file order
            for (const Tensor &t : f.tensors) {
                std::string base = t.name;
                const std::string sp = ".weight_packed";
                if (base.size() > sp.size() && base.compare(base.size() - sp.size(), sp.size(), sp) == 0) base.resize(base.size() - sp.size());
                if (by_t.count(base)) order.emplace_back(t.begin, base);
            }
            std::sort(order.begin(), order.end());
            for (auto &o : order) deliver_tensor(f, o.second, by_t[o.second], D);
            f.close();
        }
        D.flush();
#else
        throw std::runtime_error("k31: --stage-dir needs the uTofu build (-DK31_UTOFU)");
#endif
    }
    const double t_src = now() - t0;
    comm.barrier();
#if defined(K31_UTOFU)
    if (!stage_dir.empty()) k31x_settle();
#endif
    const double t_all = now() - t0;
    // checks: every planned piece delivered once; the hash of every destination's arena = what its sources sent
    double cnt[2] = {double(D.pieces), double(res.mine.size())};
    if (stage_dir.empty()) cnt[1] = cnt[0] == cnt[1] ? cnt[1] : -1;
    comm.dsum(cnt, 2);
    std::vector<uint64_t> hrecv_expect(size_t(comm.world), 0);
#if defined(K31_MPI)
    MPI_Allreduce(MPI_IN_PLACE, D.hsent.data(), comm.world, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    uint64_t hmine = 0;
    for (const Piece &p : res.mine) hmine += hash_par(g_arena + p.off, p.bytes, p.off);
    // a Put's local completion can precede its remote write: on a mismatch wait, drop cached lines and hash again
    int retries = 0;
    while (hmine != D.hsent[size_t(comm.rank)] && retries < 3 && !stage_dir.empty()) {
        ++retries;
        struct timespec ts = {0, 200000000};
        nanosleep(&ts, nullptr);
#if defined(K31_UTOFU)
        k31x_settle();
#endif
        hmine = 0;
        for (const Piece &p : res.mine) hmine += hash_par(g_arena + p.off, p.bytes, p.off);
    }
    if (retries) std::fprintf(stderr, "k31 rank %d: arena hash retries=%d %s\n", comm.rank, retries,
                              hmine == D.hsent[size_t(comm.rank)] ? "then ok" : "still bad");
    const int ok_hash = hmine == D.hsent[size_t(comm.rank)] ? 1 : 0;
    double bad = ok_hash ? 0 : 1;
    comm.dsum(&bad, 1);
    double gbytes[2] = {double(res.arena), double(g_ls.read_bytes)};
    double mx_arena = double(res.arena);
#if defined(K31_MPI)
    MPI_Allreduce(MPI_IN_PLACE, &mx_arena, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    comm.dsum(gbytes, 2);
    double tmax[2] = {t_src, g_ls.t_read};
#if defined(K31_MPI)
    MPI_Allreduce(MPI_IN_PLACE, tmax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (comm.rank == 0 || comm.sim)
        std::printf("RUNNER_K31_LOAD mode=%s pieces=%.0f/%.0f hash_bad_ranks=%.0f arena_GB_rank0=%.3f arena_GB_max=%.3f "
                    "arena_GB_total=%.1f read_GB_total=%.1f plan_s=%.2f init_s=%.2f src_s_max=%.1f read_s_max=%.1f "
                    "rank0_read_s=%.1f rank0_pack_s=%.1f rank0_flush_s=%.1f load_s=%.1f\n",
                    stage_dir.empty() ? "direct" : "staged-utofu", cnt[0], cnt[1], bad, res.arena / 1e9, mx_arena / 1e9,
                    gbytes[0] / 1e9, gbytes[1] / 1e9, t_plan, t_init, tmax[0], tmax[1], g_ls.t_read, g_ls.t_pack,
                    D.t_flush, t_all);
    std::fflush(stdout);
    if (bad > 0 || cnt[0] != cnt[1]) throw std::runtime_error("k31: weight load check failed (hash or piece count)");
#if defined(K31_UTOFU)
    if (!stage_dir.empty()) k31x_fini();
#endif
    return res;
}

std::vector<int> read_ids(const std::string &path) {
    std::vector<int> v;
    FILE *f = std::fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("k31: cannot open " + path);
    int x;
    while (std::fscanf(f, " %d%*[ ,\n]", &x) == 1) v.push_back(x);
    std::fclose(f);
    return v;
}

size_t rss_bytes() {
    FILE *f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long pages = 0, res = 0;
    if (std::fscanf(f, "%ld %ld", &pages, &res) != 2) res = 0;
    std::fclose(f);
    return size_t(res) * 4096;
}

}  // namespace

int main(int argc, char **argv) {
    k31ld::g_planar = false;   // planar MXFP4 off by default (--planar: on; no gain measured, +27% arena with TP slices)
    comm.init(&argc, &argv);
    std::string layers = "1,3";
    int steps = 8, first_token = 1;
    int64_t ctx = 4096, fill = 0;
    bool head = true;
    std::string model_dir, stage_dir, prompt_ids, dump_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("k31: " + a + " needs a value"); return argv[++i]; };
        if (a == "--layers") layers = next();
        else if (a == "--steps") steps = std::atoi(next());
        else if (a == "--context") ctx = std::atoll(next());
        else if (a == "--fill") fill = std::atoll(next());
        else if (a == "--seed") g_seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--token") first_token = std::atoi(next());
        else if (a == "--no-head") head = false;
        else if (a == "--no-moe") g_no_moe = true;
        else if (a == "--expert-tp") g_expert_tp = true;
        else if (a == "--etp-group") { g_expert_tp = true; g_etp_g = std::atoi(next()); }
        else if (a == "--model") model_dir = next();
        else if (a == "--stage-dir") stage_dir = next();
        else if (a == "--prompt-ids") prompt_ids = next();
        else if (a == "--chunk-mb") g_chunk = uint64_t(std::atoll(next())) << 20;
        else if (a == "--test-raw-a") g_raw_a = true;
        else if (a == "--no-fuse") g_fuse = false;
        else if (a == "--dsa-hp") g_dsa_hp = true;
        else if (a == "--ar-bench") g_ar_bench = std::atoi(next());
        else if (a == "--no-tcols") g_tcols = false;
        else if (a == "--serial-res") g_par_res = false;
        else if (a == "--res-serial") g_res_serial = true;
        else if (a == "--res5") g_res2 = false;
        else if (a == "--router-repl") g_router_shard = false;
        else if (a == "--dsa-shard") { g_dsa_shard = true; g_dsa_hp = true; }
        else if (a == "--par-min") g_par_min = std::atoll(next());
        else if (a == "--ds41-mx") { g_k31mx = false; k31ld::g_planar = false; }
        else if (a == "--no-planar") k31ld::g_planar = false;
        else if (a == "--planar") k31ld::g_planar = true;
        else if (a == "--dynamic") g_dynamic = true;
        else if (a == "--sim-world") { comm.world = std::atoi(next()); comm.sim = true; }
        else if (a == "--sim-rank") comm.rank = std::atoi(next());
        else if (a == "--sim-experts") g_sim_k = std::atoi(next());
        else if (a == "--dump") { dump_dir = next(); g_dump = true; }
        else if (a == "--help") {
            std::printf("k31pro_runner [--layers 1,3|0-11|all] [--steps N] [--context N] [--fill N] [--seed S] [--token T] [--no-head] [--no-moe] [--expert-tp]\n"
                        "  dummy-weight decode of the listed layers (real shapes); --fill N puts N synthetic tokens in the DSA caches\n"
                        "  --model DIR: real weights (safetensors checkpoint); each rank reads its own pieces\n"
                        "  --stage-dir D: real weights from the files staged in D on each node (each rank's share), redistributed by uTofu\n"
                        "  --prompt-ids FILE: token ids (real weights) prefilled one by one before the timed decode\n"
                        "  --chunk-mb N: rows read per chunk (default 256)\n");
            comm.fini(); return 0;
        } else throw std::runtime_error("k31: unknown option " + a);
    }
    std::vector<int> prompt;
    try { if (!prompt_ids.empty()) prompt = read_ids(prompt_ids); }
    catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); comm.fini(); return 2; }
    if (ctx < fill + int64_t(prompt.size()) + steps) ctx = fill + int64_t(prompt.size()) + steps;
#if defined(_OPENMP)
    g_threads = omp_get_max_threads();
    omp_set_schedule(g_dynamic ? omp_sched_dynamic : omp_sched_static, 1);
#endif
    try {
        const double tb = now();
        Model m;
        comm.init_utofu_ar();
        const bool real = !model_dir.empty() || !stage_dir.empty();
        const std::vector<int> lv = parse_layers(layers);
        LoadResult lr;
        if (real) {
            lr = load_real(model_dir, stage_dir, lv, head);
            g_wmode = W_BIND; g_pieces = &lr.mine; g_cursor = 0;
        }
        for (int li : lv) m.L.push_back(build_layer(li, ctx));
        build_head(m, head, comm.rank);
        if (real && g_cursor != lr.mine.size()) throw std::runtime_error("k31: bind used fewer pieces than planned");
        g_wmode = W_DUMMY;
        if (g_dump) {
            if (comm.world != 1) throw std::runtime_error("k31: --dump needs one rank");
            dump_write(dump_dir);
            std::printf("RUNNER_K31_DUMP tensors=%zu dir=%s\n", g_dumped.size(), dump_dir.c_str());
            g_dumped.clear(); g_dump = false;
        }
        fill_context(m, fill);
        size_t cache_bytes = 0;
        for (Layer &L : m.L) if (L.dsa) cache_bytes += (L.a.cache.size() + L.a.ikeys.size()) * 2;
        comm.barrier();
        const double build_s = now() - tb;
        if (comm.rank == 0 || comm.sim)
            std::printf("RUNNER_K31 ready ranks=%d threads=%d layers=%zu context=%" PRId64 " fill=%" PRId64
                        " weight_GB=%.3f cache_GB=%.3f rss_GB=%.3f build_s=%.2f\n",
                        comm.world, g_threads, m.L.size(), ctx, fill, g_weight_bytes / 1e9, cache_bytes / 1e9,
                        rss_bytes() / 1e9, build_s);
        std::fflush(stdout);
        if (g_ar_bench > 0) {
            for (int n : {int(H), int(K31_LATENT + K31_EXPERTS), 2272, int(K31_HEADS)}) {
                Vec v(size_t(n), 1.0f);
                {   // correctness: v[i] = rank + 0.001 i summed over ranks
                    for (int i = 0; i < n; ++i) v[size_t(i)] = float(comm.rank) + 0.001f * float(i % 1000);
                    comm.sum(v.data(), n);
                    double bad = 0;
                    const double W = comm.world;
                    for (int i = 0; i < n; ++i) {
                        const double want = W * (W - 1) / 2 + W * 0.001 * (i % 1000);
                        if (std::fabs(v[size_t(i)] - want) > 1e-4 * want + 1e-3) bad += 1;
                    }
                    const uint64_t hv = fnv(v.data(), v.size() * 4);
                    double hl[2] = {double(hv & 0xffffffff), -double(hv & 0xffffffff)};
#if defined(K31_MPI)
                    MPI_Allreduce(MPI_IN_PLACE, hl, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
                    comm.dsum(&bad, 1);
                    if (comm.rank == 0) std::printf("RUNNER_K31_ARCHECK floats=%d bad=%.0f same_bits_all_ranks=%d\n", n, bad, hl[0] == -hl[1] ? 1 : 0);
                }
                std::fill(v.begin(), v.end(), 1.0f);
                for (int i = 0; i < 20; ++i) comm.sum(v.data(), n);
                comm.barrier();
                const double t0 = now();
                for (int i = 0; i < g_ar_bench; ++i) comm.sum(v.data(), n);
                const double us = 1e6 * (now() - t0) / g_ar_bench;
                double mx = us;
#if defined(K31_MPI)
                MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
                if (comm.rank == 0) std::printf("RUNNER_K31_ARBENCH floats=%d us_rank0=%.2f us_max=%.2f\n", n, us, mx);
            }
            std::fflush(stdout);
        }
        // decode: step 0 is warm-up (not timed into the per-token figure)
        std::vector<int> gen;
        int tok = first_token;
        double t_steps = 0;
        int64_t pos0 = fill;
        if (!prompt_ids.empty()) {
            const std::vector<int> &ids = prompt;
            if (ids.empty()) throw std::runtime_error("k31: empty prompt");
            const double tp = now();
            int nt = 0;
            for (size_t i = 0; i < ids.size(); ++i) nt = forward(m, ids[i], pos0 + int64_t(i), head && i + 1 == ids.size());
            comm.barrier();
            if (comm.rank == 0 || comm.sim)
                std::printf("RUNNER_K31_PROMPT tokens=%zu s=%.2f ms_per_token=%.2f\n", ids.size(), now() - tp,
                            1e3 * (now() - tp) / double(ids.size()));
            pos0 += int64_t(ids.size());
            tok = head ? nt : tok;
            gen.push_back(nt);
            g_tm.clear(); comm.t = 0; g_dsa = DsaStats{};
        }
        std::map<std::string, double> warm;
        for (int s = 0; s < steps; ++s) {
            if (s == 1) { warm = g_tm; g_tm.clear(); comm.t = 0; g_dsa = DsaStats{}; g_mvm_prep = g_mvm_region = 0; g_mvm_calls = 0; }
            comm.barrier();
            const double t0 = now();
            const int nt = forward(m, tok, pos0 + s, head);
            comm.barrier();
            if (s >= 1) t_steps += now() - t0;
            gen.push_back(nt);
            tok = head ? nt : tok + 1;
        }
        // hidden hash over the final hidden state (rank 0's copy; all ranks hold the same) and the generated ids
        uint64_t hh = fnv(m.hidden.data(), m.hidden.size() * sizeof(float));
        hh = fnv(gen.data(), gen.size() * sizeof(int), hh);
        int finite = 1;
        for (float v : m.hidden) finite &= std::isfinite(v) ? 1 : 0;
        const int timed = std::max(1, steps - 1);
        if (comm.rank == 0 || comm.sim) {
            std::printf("RUNNER_K31_GEN steps=%d timed=%d ms_per_token=%.3f finite=%d hidden_hash=%016" PRIx64 " tokens=",
                        steps, timed, 1e3 * t_steps / timed, finite, hh);
            for (size_t i = 0; i < gen.size(); ++i) std::printf("%s%d", i ? "," : "", gen[i]);
            std::printf("\n");
            std::printf("RUNNER_K31_TIMERS");
            for (auto &kv : g_tm) {
                if (kv.first.size() > 2 && kv.first.compare(kv.first.size() - 2, 2, ".n") == 0) continue;
                if (kv.first == "moe.experts_used") std::printf(" %s=%.2f", kv.first.c_str(), kv.second / timed);
                else std::printf(" %s=%.3f", kv.first.c_str(), 1e3 * kv.second / timed);
            }
            std::printf(" comm=%.3f mvm_prep=%.3f mvm_region=%.3f mvm_calls=%.1f\n", 1e3 * comm.t / timed, 1e3 * g_mvm_prep / timed,
                        1e3 * g_mvm_region / timed, double(g_mvm_calls) / timed);
            std::printf("RUNNER_K31_LAYER");
            for (const char *k : {"layer.dense0", "layer.kda", "layer.dsa"}) {
                const double n = g_tm[std::string(k) + ".n"];
                if (n > 0) std::printf(" %s_ms=%.3f %s_comm_ms=%.3f", k + 6, 1e3 * g_tm[k] / n, k + 6, 1e3 * g_tm[std::string(k) + ".comm"] / n);
            }
            if (g_dsa.calls) std::printf(" dsa_selected_rank0=%.1f", g_dsa.selected / g_dsa.calls);
            std::printf("\n");
        }
        comm.fini();
        return finite ? 0 : 3;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "k31pro_runner rank %d: %s\n", comm.rank, e.what());
        comm.fini();
        return 2;
    }
}
