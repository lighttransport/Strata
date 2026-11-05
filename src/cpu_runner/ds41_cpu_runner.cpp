// src/cpu_runner/ds41_cpu_runner.cpp - DeepSeek-V4.1-Flash GGUF (GSQ-RCO builds) on A64FX nodes, one process per node.
//
// v0, correctness first: every routed expert and the shared expert sliced into 256-row chunks (gate/up rows and the
// matching down block); chunk c of expert e lives on rank (9e + c) % world, shared chunk c on rank c % world. The
// trunk is replicated and computed on every rank from identical inputs; one all-reduce per layer (the FFN output)
// and one for the head (each rank its vocabulary slice). Kernels: src/kernels/cpu/ds41_sve.c (ggml's integer sums).
// The forward follows src/program/ds41_decode.cpp one token at a time: the lagged hyper-connection pre-mixes, the
// sliding window + compressed rows with the indexer's top-k, attention sinks, the grouped output projection, Engram
// before layers 1 and 14, the router sqrt(softplus) + bias top-k.
//
//   ds41_cpu_runner MODEL-00001-of-0000N.gguf --eval IDS [--tokens N] [--layers N] [--context N] [--generate N]
//                   [--save-logits FILE] [--threads N]
#include "strata/cpu_runner/infra.hpp"
#include "strata/cpu_runner/row_prefetch.hpp"
#include "strata/cpu_runner/vexpf.h"
#include "strata/core/model.hpp"
#include "strata/ds41/config.hpp"
#include "strata/kernels/cpu/ds41_sve.h"
#include "strata/host/glm_env.h"
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
typedef svfloat32_t vf32 __attribute__((arm_sve_vector_bits(512)));
#endif

#include <condition_variable>
#include <climits>
#include <cstdio>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#if defined(__linux__)
#include <ucontext.h>
#endif

namespace {
using namespace strata::cpu_runner;
using strata::ds41::Config;
using strata::ds41::Rope;
using Vec = std::vector<float>;

// ------------------------------------------------------------------------------------------------ weights
struct Mat {
    int type = -1, rows = 0, cols = 0;
    size_t rb = 0;
    const uint8_t *p = nullptr;
};

// --image-dir: every resident buffer comes from anon() in a fixed order; after a full load the rank writes them back to
// back into DIR/ds41-r<r>of<W>-L<layers>.img, and a later load in the same allocation (/local lives as long as it)
// replays them in that order instead of reading the shared volume (g_replay: the fill code is skipped)
std::vector<std::pair<uint8_t *, size_t>> g_bufs;
bool g_replay = false;
int g_img_fd = -1;
size_t g_img_off = 0;
void img_read(uint8_t *dst, size_t n) {
    for (size_t d = 0; d < n;) {
        const ssize_t r = ::pread(g_img_fd, dst + d, std::min(n - d, size_t(1) << 30), off_t(g_img_off + d));
        if (r <= 0) throw std::runtime_error("ds41: image read failed");
        d += size_t(r);
    }
    g_img_off += n;
}
uint8_t *anon(size_t n) {   // fresh anonymous pages: first touched by the fill (or the image replay)
    void *p = mmap(nullptr, (n + 4095) & ~size_t(4095), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("ds41: out of memory");
    uint8_t *b = static_cast<uint8_t *>(p);
    g_bufs.push_back({b, n});
    if (g_replay) img_read(b, n);
    return b;
}
size_t g_resident = 0;
void par_copy(uint8_t *dst, const uint8_t *src, size_t n) {
#pragma omp parallel
    {
        const int t = omp_get_thread_num(), k = omp_get_num_threads();
        const size_t a = n * t / k, b = n * (t + 1) / k;
        std::memcpy(dst + a, src + a, b - a);
    }
}

// Reads go through pread in 8 MiB pieces on 16 threads: page faults on the mapped GGUF crawl through the LLIO cache of
// the shared volume (a few MB/s), large preads stream. Dense tensors are read cooperatively (each rank 1/world of the
// bytes, then an all-gather), so the shared volume serves every byte once.
std::map<std::string, int> g_fds;
int file_fd(const strata::GgufFile *f) {
    auto it = g_fds.find(f->path());
    if (it != g_fds.end()) return it->second;
    const int fd = ::open(f->path().c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("ds41: cannot open " + f->path());
    return g_fds[f->path()] = fd;
}
double g_read_s = 0;
size_t g_read_b = 0;
void read_bytes(const strata::core::ArtifactTensor &a, size_t off, size_t n, uint8_t *dst) {
    const double t0 = now();
    const int fd = file_fd(a.file);
    // STRATA_DS41_READ_THREADS / STRATA_DS41_READ_PIECE_MB: the shared volume's read pattern (default 16 x 8 MiB)
    static const int rthreads = std::getenv("STRATA_DS41_READ_THREADS") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_READ_THREADS"))) : 16;
    static const size_t rpiece = size_t(std::getenv("STRATA_DS41_READ_PIECE_MB") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_READ_PIECE_MB"))) : 8) << 20;
    const size_t base = a.file->data_start() + a.tensor->offset + off, piece = rpiece;
    std::atomic<size_t> next{0};
    std::atomic<int> fail{0};
    auto work = [&] {
        for (size_t o; (o = next.fetch_add(piece)) < n;) {
            const size_t len = std::min(piece, n - o);
            for (size_t done = 0; done < len;) {
                const ssize_t r = ::pread(fd, dst + o + done, len - done, off_t(base + o + done));
                if (r <= 0) { fail = 1; return; }
                done += size_t(r);
            }
        }
    };
    const int nt = n >= (size_t(64) << 20) ? rthreads : n >= (size_t(8) << 20) ? std::min(4, rthreads) : 1;
    std::vector<std::thread> th;
    for (int t = 1; t < nt; ++t) th.emplace_back(work);
    work();
    for (auto &t : th) t.join();
    if (fail) throw std::runtime_error("ds41: read failed: " + a.tensor->name);
    g_read_s += now() - t0, g_read_b += n;
}
// many reads at once: the requests cut into pieces of at most STRATA_DS41_READ_PIECE_MB (8) on READ_THREADS (16)
// threads. The shared volume through LLIO needs ~16 reads in flight per node (12 nodes: ~1 GB/s together at 16, ~0.2
// GB/s at 1-8; job 52873550), and a layer's expert chunks are ~2.5 MB each: one at a time they ran at the 1-thread rate
struct RReq { const strata::core::ArtifactTensor *a; size_t off, n; uint8_t *dst; };
void read_many(const std::vector<RReq> &reqs) {
    if (reqs.empty()) return;
    const double t0 = now();
    static const int rthreads = std::getenv("STRATA_DS41_READ_THREADS") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_READ_THREADS"))) : 16;
    static const size_t rpiece = size_t(std::getenv("STRATA_DS41_READ_PIECE_MB") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_READ_PIECE_MB"))) : 8) << 20;
    struct Piece { int fd; size_t at, n; uint8_t *dst; };
    std::vector<Piece> ps;
    size_t total = 0;
    for (const auto &r : reqs) {
        const int fd = file_fd(r.a->file);
        const size_t base = r.a->file->data_start() + r.a->tensor->offset + r.off;
        for (size_t o = 0; o < r.n; o += rpiece) ps.push_back({fd, base + o, std::min(rpiece, r.n - o), r.dst + o});
        total += r.n;
    }
    std::atomic<size_t> next{0};
    std::atomic<int> fail{0};
    auto work = [&] {
        for (size_t i; (i = next.fetch_add(1)) < ps.size();)
            for (size_t d = 0; d < ps[i].n;) {
                const ssize_t r = ::pread(ps[i].fd, ps[i].dst + d, ps[i].n - d, off_t(ps[i].at + d));
                if (r <= 0) { fail = 1; return; }
                d += size_t(r);
            }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < std::min<int>(rthreads, int(ps.size())); ++t) th.emplace_back(work);
    work();
    for (auto &t : th) t.join();
    if (fail) throw std::runtime_error("ds41: batched read failed");
    g_read_s += now() - t0, g_read_b += total;
}
Comm *g_comm = nullptr;
// every rank gets the bytes; each reads 1/world of them
void read_coop(const strata::core::ArtifactTensor &a, size_t n, uint8_t *dst) {
    const int W = g_comm->world, R = g_comm->rank;
    if (W == 1 || n < (size_t(1) << 20)) { read_bytes(a, 0, n, dst); return; }
    std::vector<size_t> o(size_t(W) + 1);
    for (int r = 0; r <= W; ++r) o[size_t(r)] = n * size_t(r) / size_t(W);
    read_bytes(a, o[size_t(R)], o[size_t(R) + 1] - o[size_t(R)], dst + o[size_t(R)]);
#ifdef STRATA_CPU_RUNNER_MPI
    for (int r = 0; r < W; ++r)   // (counts fit an int: the largest dense tensor is far below 2 GiB per rank)
        MPI_Bcast(dst + o[size_t(r)], int(o[size_t(r) + 1] - o[size_t(r)]), MPI_BYTE, r, MPI_COMM_WORLD);
#endif
}

struct Model {
    // the target's artifact, or (the DSpark drafter, an arch ModelArtifact does not take) a GGUF's tensors alone
    const std::map<std::string, strata::core::ArtifactTensor> &tens;
    strata::core::ModelArtifact *art = nullptr;
    explicit Model(strata::core::ModelArtifact &a) : tens(a.tensors()), art(&a) {}
    explicit Model(const std::map<std::string, strata::core::ArtifactTensor> &t) : tens(t) {}
    bool has(const std::string &n) const { return tens.count(n) != 0; }
    const strata::core::ArtifactTensor &t(const std::string &n) const {
        const auto it = tens.find(n);
        if (it == tens.end()) throw std::runtime_error("ds41: missing tensor " + n);
        return it->second;
    }
    // the whole tensor as a matrix (rows = product of dims 1..), copied resident unless `map`
    Mat mat(const std::string &n, bool map = false) const {
        const auto &a = t(n);
        Mat m;
        m.type = int(a.tensor->type);
        if (!ds41_type_ok(m.type)) throw std::runtime_error("ds41: unsupported type of " + n);
        m.cols = int(a.tensor->shape[0]);
        m.rows = 1;
        for (size_t d = 1; d < a.tensor->shape.size(); ++d) m.rows *= int(a.tensor->shape[d]);
        m.rb = ds41_row_bytes(m.type, m.cols);
        if (map) { m.p = a.data(); return m; }
        uint8_t *dst = anon(m.rb * m.rows);
        if (!g_replay) read_coop(a, m.rb * m.rows, dst);
        g_resident += m.rb * m.rows;
        m.p = dst;
        return m;
    }
    // rows [r0, r0 + n) of a tensor, copied
    Mat rows(const std::string &n, int r0, int count) const {
        Mat m = mat(n, true);
        uint8_t *dst = anon(m.rb * count);
        if (!g_replay) read_bytes(t(n), m.rb * size_t(r0), m.rb * count, dst);
        g_resident += m.rb * count;
        m.p = dst, m.rows = count;
        return m;
    }
    Vec floats(const std::string &n) const {
        const auto &a = t(n);
        const int type = int(a.tensor->type);
        const size_t e = a.tensor->elements();
        Vec v(e);
        std::vector<uint8_t> raw(ds41_row_bytes(type, int(e)));
        read_bytes(a, 0, raw.size(), raw.data());
        if (type == DS41_T_F32) std::memcpy(v.data(), raw.data(), e * 4);
        else ds41_dequant_row(type, raw.data(), int(e), v.data());
        return v;
    }
};

std::vector<int64_t> tensor_ints(const strata::core::ArtifactTensor &t) {
    const size_t n = size_t(t.tensor->elements());
    std::vector<int64_t> out(n);
    if (t.tensor->type == 26) {
        std::vector<int32_t> v(n);
        std::memcpy(v.data(), t.data(), n * 4);
        for (size_t i = 0; i < n; ++i) out[i] = v[i];
    } else if (t.tensor->type == 27) std::memcpy(out.data(), t.data(), n * 8);
    else throw std::runtime_error("ds41: " + t.tensor->name + " is not an integer tensor");
    return out;
}

// ------------------------------------------------------------------------------------------------ GEMV
struct Act { std::vector<ds41_q8k> k; std::vector<ds41_q80> z; };
// f(a, b) on every thread of the team for its static block [a, b) of n items. team_run: inside with_team (the decode
// loop) the open persistent region (~1.6 us per dispatch + the flag barrier) instead of an OpenMP fork per region
template <class F> void team_for(int n, F &&f) {
    team_run([&] {
        const int t = omp_get_thread_num(), T = omp_get_num_threads();
        const int a = int(int64_t(n) * t / T), b = int(int64_t(n) * (t + 1) / T);
        if (b > a) f(a, b);
    });
}
double g_gemv_s = 0;
// STRATA_DS41_SKEWPROF: every all-reduce's duration on this rank from the generation on (site, s). skew_report takes
// per call the minimum over the ranks (the last rank to arrive waits least: about the call's own cost), its rank and
// the mean: mean - min is the time the ranks spent waiting for the slowest one
bool g_skew_on = false;
std::vector<float> g_skew_dt;
std::vector<uint8_t> g_skew_site;
std::vector<const char *> g_skew_names;
void skew_log(const char *site, double dt) {
    size_t k = 0;
    while (k < g_skew_names.size() && std::strcmp(g_skew_names[k], site)) ++k;
    if (k == g_skew_names.size()) g_skew_names.push_back(site);
    g_skew_dt.push_back(float(dt)), g_skew_site.push_back(uint8_t(k));
}
void skew_report(Comm &comm, int ntok) {
#ifdef STRATA_CPU_RUNNER_MPI
    const int n = int(g_skew_dt.size()), W = comm.world;
    int nmin = n;
    MPI_Allreduce(&n, &nmin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    struct FI { float v; int r; };
    std::vector<FI> loc(static_cast<size_t>(nmin)), mins(static_cast<size_t>(nmin));
    for (int i = 0; i < nmin; ++i) loc[size_t(i)] = {g_skew_dt[size_t(i)], comm.rank};
    std::vector<float> sums(static_cast<size_t>(nmin));
    MPI_Reduce(loc.data(), mins.data(), nmin, MPI_FLOAT_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(g_skew_dt.data(), sums.data(), nmin, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (comm.rank) return;
    double tot_wait = 0, tot_min = 0;
    for (size_t k = 0; k < g_skew_names.size(); ++k) {
        double c = 0, mean = 0, mn = 0;
        std::vector<double> last(static_cast<size_t>(W), 0.0);
        for (int i = 0; i < nmin; ++i)
            if (g_skew_site[size_t(i)] == k) c += 1, mean += sums[size_t(i)] / W, mn += mins[size_t(i)].v, last[size_t(mins[size_t(i)].r)] += 1;
        if (!c) continue;
        tot_wait += mean - mn, tot_min += mn;
        std::printf("RUNNER_DS41_SKEW %-16s calls_per_token=%.1f mean_us=%.1f min_us=%.1f wait_ms_per_token=%.3f last_rank_pct=", g_skew_names[k], c / ntok, 1e6 * mean / c,
                    1e6 * mn / c, 1e3 * (mean - mn) / ntok);
        for (int r = 0; r < W; ++r) std::printf("%s%.0f", r ? "," : "", 100 * last[size_t(r)] / c);
        std::printf("\n");
    }
    std::printf("RUNNER_DS41_SKEW total wait_ms_per_token=%.3f min_ms_per_token=%.3f calls=%d\n", 1e3 * tot_wait / ntok, 1e3 * tot_min / ntok, nmin);
    std::fflush(stdout);
#else
    (void) comm, (void) ntok;
#endif
}
double g_xtw[64][4];   // STRATA_DS41_XPROF: per thread, the expert phase's gate/up work, its barrier wait, the down work (s)
int g_prompt_io_core = -1, g_prompt_io_threads = 0, g_prompt_io_pass = 1;
void configure_prompt_io(bool prompt_only, int prompt_pass) {
    if (!strata_env_is_one("STRATA_DS41_PREFILL_EG_ASYNC")) return;
    int valid = 1, core = -1;
    std::string error;
    try {
        if (!prompt_only || !strata_env_is_one("STRATA_DS41_PREFILL_EG_CACHE") ||
            strata_env_is_one("STRATA_DS41_COMM_COPY_TEAM") || omp_get_max_threads() != 46 ||
            omp_get_num_places() != 48 || omp_get_place_num() != 0 || omp_get_proc_bind() != omp_proc_bind_close)
            throw std::runtime_error("prompt async rows need cached prefill,46 compute,48 close places and no copy pool");
        std::set<int> places;
        int first = -1;
        for (int i = 0; i < 48; ++i) {
            int cpu = -1;
            if (omp_get_place_num_procs(i) != 1) throw std::runtime_error("prompt async rows need single CPU places");
            omp_get_place_proc_ids(i, &cpu);
            if (cpu < 0 || cpu >= CPU_SETSIZE || !places.insert(cpu).second)
                throw std::runtime_error("prompt async rows invalid places");
            if (!i) first = cpu;
            if (cpu != first + i) throw std::runtime_error("prompt async rows need contiguous places");
        }
        cpu_set_t cs; CPU_ZERO(&cs);
        if (sched_getaffinity(0, sizeof cs, &cs) || CPU_COUNT(&cs) != 1 || !CPU_ISSET(first, &cs))
            throw std::runtime_error("prompt async rows main affinity");
        const int configured = strata_env_is_set("STRATA_RUNNER_COMM_CORE") ? strata_env_int_or("STRATA_RUNNER_COMM_CORE", 0) : -1;
        const int comm_core = configured < 0 ? first + 46 : configured;
        if (comm_core != first + 46) throw std::runtime_error("prompt async rows need separate comm and IO CPUs");
        core = first + 47;
    } catch (const std::exception &e) { valid = 0; error = e.what(); }
    int all_valid = valid;
#ifdef STRATA_CPU_RUNNER_MPI
    if (g_comm && g_comm->world > 1) MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
    if (!all_valid) throw std::runtime_error(error.empty() ? "prompt async setup failed on another rank" : error);
    g_prompt_io_core = core;
    g_prompt_io_pass = prompt_pass;
    g_prompt_io_threads = std::max(1, std::min(16, strata_env_int_or("STRATA_DS41_PREFILL_EG_ASYNC_THREADS", 8)));
    std::printf("RUNNER_DS41_PROMPT_IO_CONFIG compute=46 comm_core=%d io_core=%d readers=%d\n", core-1,core,g_prompt_io_threads);
    std::fflush(stdout);
}
#ifdef STRATA_CPU_RUNNER_UTOFU
extern "C" void cpu_utofu_set_par(void (*p)(void (*fn)(void *, int, int), void *arg, int n));
// the uTofu all-reduce's copies and shard sum over the team for long vectors (batched / multi-token passes)
// (on the comm thread, which is not the team's: serially; each element's sum keeps its rank order either way)
void utofu_par(void (*fn)(void *, int, int), void *arg, int n) {
    if (omp_get_level() == 0 && g_team.open) {
        fn(arg, 0, n);
        return;
    }
    team_for(n, [&](int a, int b) { fn(arg, a, b); });
}
#endif
// per call site (tag): calls, serial activation prep, the team pass, weight bytes (RUNNER_DS41_GEMV)
struct GStat { double calls = 0, prep = 0, rows = 0, bytes = 0; };
std::map<std::string, GStat> g_gstat;
// several independent row ranges in one team pass: each job's activation prepared once (shared when two jobs read the
// same x in the same form), the rows of all jobs split over the threads by weight bytes. A row is computed whole by one
// thread with the same kernel, so the outputs are those of separate calls. nt tokens per job (a multi-token pass):
// token k reads x + k ldx and writes y[k ldy + row] (ldy 0: the matrix's rows); the kernels take knt() at a time, each
// token with its own accumulators, so every token's result is the one-token call's
// STRATA_DS41_KNT (1-8, default 4): tokens per kernel call (ds41_rows). Each token's sums keep their order at any count,
// so the bits are the same; a verify pass of 5-8 entries then reads each weight row once instead of twice
static int knt(int type, int cols) {
    // STRATA_DS41_IQ2P_TILE16=1: reuse a one-block IQ2P down row across up to 16 tokens.
    static const bool tile16 = strata_env_truthy_one("STRATA_DS41_IQ2P_TILE16");
    if (tile16 && type == DS41_T_IQ2P && cols == 256) return 16;
    static const int v = std::getenv("STRATA_DS41_KNT") ? std::clamp(std::atoi(std::getenv("STRATA_DS41_KNT")), 1, 8) : 4;
    return v;
}
struct GJob { const Mat *m; const float *x; float *y; int r0, r1; int nt = 1; size_t ldx = 0, ldy = 0; const ds41_q8k *xq = nullptr; };   // xq: (Q8_K matrices) the tokens' blocks already made, cols / 256 per token
constexpr int kGemvMaxNt = 512;   // (a pass: up to Runner::kMaxBatch entries)
// a gemv_multi call planned outside a region (gplan) and run by every thread of one (gbody): the activations' blocks
// quantized by the team (or thread 0 alone for few blocks), a barrier, then each thread its rows by weight bytes.
// deferred: the activation is written inside the same region before gbody (STRATA_DS41_MERGE), so no block is
// quantized in gplan and gbody always ends its quantization with a barrier
struct GPlan {
    const GJob *j = nullptr;
    int nj = 0, nq = 0, qjob[8] = {}, qbt[8] = {};
    const void *act[8][kGemvMaxNt];   // (only the planned jobs' tokens are set and read)
    size_t qb0[9] = {0}, off[9] = {0}, ldy[8] = {}, QB = 0, B = 0;
    bool team_quant = false, deferred = false;
    Act *a = nullptr;
    const std::function<void(int, int, int)> *post = nullptr;
    void quant(size_t qa, size_t qe) const {   // global blocks [qa, qe) over the activations to quantize
        for (int q = 0; q < nq; ++q) {
            const size_t lo = std::max(qa, qb0[q]), hi = std::min(qe, qb0[q + 1]);
            const GJob &g = j[qjob[q]];
            const int bpt = qbt[q], bs = ds41_act_kind(g.m->type) == DS41_ACT_Q8K ? 256 : 32;
            for (size_t gb = lo; gb < hi;) {   // within one token's blocks
                const size_t lb = gb - qb0[q], t = lb / size_t(bpt), b0 = lb % size_t(bpt), nb = std::min(hi - gb, size_t(bpt) - b0);
                const float *xs = g.x + t * g.ldx + size_t(bs) * b0;
                if (bs == 256) ds41_quant_q8k(xs, int(256 * nb), a[qjob[q]].k.data() + lb);
                else ds41_quant_q80(xs, int(32 * nb), a[qjob[q]].z.data() + lb);
                gb += nb;
            }
        }
    }
};
// (act: one Act array per concurrently planned call; slot 0 is gemv_multi's)
void gplan(GPlan &P, const GJob *j, int nj, const std::function<void(int, int, int)> *post, bool deferred, int slot = 0) {
    if (nj > 8) throw std::runtime_error("ds41: gemv_multi takes at most 8 jobs");
    static thread_local Act a_tl[2][8];
    P.j = j, P.nj = nj, P.post = post, P.deferred = deferred, P.a = a_tl[slot], P.nq = 0;
    Act *const a = P.a;
    for (int i = 0; i < nj; ++i) {
        const int nt = j[i].nt, kind = ds41_act_kind(j[i].m->type), cols = j[i].m->cols;
        if (nt < 1 || nt > kGemvMaxNt) throw std::runtime_error("ds41: gemv_multi token count");
        P.ldy[i] = j[i].ldy ? j[i].ldy : size_t(j[i].m->rows);
        int s = -1;
        for (int k = 0; k < i && s < 0; ++k)
            if (j[k].x == j[i].x && j[k].nt == nt && j[k].ldx == j[i].ldx && j[k].m->cols == cols && ds41_act_kind(j[k].m->type) == kind) s = k;
        if (s >= 0) { for (int t = 0; t < nt; ++t) P.act[i][t] = P.act[s][t]; }
        else if (kind == DS41_ACT_Q8K && j[i].xq) { for (int t = 0; t < nt; ++t) P.act[i][t] = j[i].xq + size_t(t) * (cols / 256); }
        else if (kind == DS41_ACT_Q8K || kind == DS41_ACT_Q80) {
            const int bpt = kind == DS41_ACT_Q8K ? cols / 256 : cols / 32;   // blocks per token
            if (kind == DS41_ACT_Q8K) a[i].k.resize(size_t(bpt) * nt);
            else a[i].z.resize(size_t(bpt) * nt);
            for (int t = 0; t < nt; ++t)
                P.act[i][t] = kind == DS41_ACT_Q8K ? static_cast<const void *>(a[i].k.data() + size_t(t) * bpt) : static_cast<const void *>(a[i].z.data() + size_t(t) * bpt);
            P.qjob[P.nq] = i, P.qbt[P.nq] = bpt, P.qb0[P.nq + 1] = P.qb0[P.nq] + size_t(bpt) * nt, ++P.nq;
        } else
            for (int t = 0; t < nt; ++t) P.act[i][t] = j[i].x + size_t(t) * j[i].ldx;
        P.off[i + 1] = P.off[i] + j[i].m->rb * size_t(j[i].r1 - j[i].r0);
    }
    // STRATA_DS41_GQ_SERIAL=N: activations of <= N blocks quantized by thread 0 before the region (no barrier in it;
    // each block's quantization is the same either way). Default 1: a single block stays serial
    static const size_t gq_serial = std::getenv("STRATA_DS41_GQ_SERIAL") ? size_t(std::max(1, std::atoi(std::getenv("STRATA_DS41_GQ_SERIAL")))) : 1;
    P.QB = P.qb0[P.nq];
    P.team_quant = P.QB > gq_serial;
    if (!P.team_quant && !deferred) P.quant(0, P.QB);
    P.B = P.off[nj];
}
void gbody(const GPlan &P, int t, int T) {
    if (P.team_quant) {
        P.quant(P.QB * size_t(t) / size_t(T), P.QB * size_t(t + 1) / size_t(T));
        team_barrier();
    } else if (P.deferred) {
        if (t == 0) P.quant(0, P.QB);
        team_barrier();
    }
    const GJob *j = P.j;
    const size_t ba = P.B * size_t(t) / size_t(T), bb = P.B * size_t(t + 1) / size_t(T);
    for (int i = 0; i < P.nj; ++i) {   // job i's rows starting in [ba, bb)
        if (bb <= P.off[i] || ba >= P.off[i + 1]) continue;
        const size_t rb = j[i].m->rb;
        const int a0 = int((std::max(ba, P.off[i]) - P.off[i] + rb - 1) / rb), b0 = int((std::min(bb, P.off[i + 1]) - P.off[i] + rb - 1) / rb);
        if (b0 > a0) {
            for (int g = 0, kn = knt(j[i].m->type, j[i].m->cols); g < j[i].nt; g += kn)
                ds41_rows(j[i].m->type, j[i].m->p, j[i].m->cols, j[i].r0 + a0, j[i].r0 + b0, &P.act[i][g], std::min(kn, j[i].nt - g),
                          j[i].y + size_t(g) * P.ldy[i], int(P.ldy[i]));
            if (P.post) (*P.post)(i, j[i].r0 + a0, j[i].r0 + b0);
        }
    }
}
void gstat(const char *tag, double prep, double rows, double bytes) {
    GStat &g = g_gstat[tag];
    g.calls += 1, g.prep += prep, g.rows += rows, g.bytes += bytes;
    g_gemv_s += prep + rows;
}
// post(i, r0, r1): run by the thread that computed job i's rows [r0, r1), right after them (no extra dispatch)
void gemv_multi(const GJob *j, int nj, const char *tag, const std::function<void(int, int, int)> *post = nullptr) {
    const double t0 = now();
    GPlan P;
    gplan(P, j, nj, post, false);
    const double t1 = now();
    team_run([&] { gbody(P, omp_get_thread_num(), omp_get_num_threads()); });
    const double t2 = now();
    gstat(tag, t1 - t0, t2 - t1, double(P.B));
}
// y[r] = row r . x for r in [r0, r1) (y indexed by the matrix's row), over the team
void gemv_rows(const Mat &m, int r0, int r1, const float *x, float *y, const char *tag = "misc") {
    const GJob j{&m, x, y, r0, r1};
    gemv_multi(&j, 1, tag);
}
void gemv(const Mat &m, const float *x, float *y, const char *tag = "misc") { gemv_rows(m, 0, m.rows, x, y, tag); }

// L2 prefetch of [p, p + n) (A64FX: 256-byte lines). STRATA_DS41_XPF: a thread's rows of the next expert chunk while it
// computes the current one - its runs are ~2-12 KB at unrelated addresses, where the hardware prefetcher starts cold
// (cmg_bw scatter probe: 6-row IQ2R runs 147 -> 171 GB/s per node, 1-row runs 76 -> 132)
inline void prefetch_l2(const uint8_t *p, size_t n) {
#if defined(__aarch64__)
    for (size_t o = 0; o < n; o += 256) __asm__ volatile("prfm pldl2keep, [%0]" ::"r"(p + o));
#else
    (void) p, (void) n;
#endif
}
// STRATA_DS41_XROT=1: chunk u's 256 gate/up rows split at floor((256 t + s_u) / T), s_u = 256 u mod T (one row of a
// boundary moves at most): the threads with the extra row change from chunk to chunk, as if the pass's chunks were one
// array split evenly. A row stays whole on one thread with the same kernel, so the bits are the default split's
static bool xrot_on() { static const bool on = strata_env_is_one("STRATA_DS41_XROT"); return on; }
static inline int xrow(int u, int t, int T) { return xrot_on() ? int((int64_t(256) * t + (int64_t(256) * u) % T) / T) : int(int64_t(256) * t / T); }
// STRATA_DS41_MERGE: bits of the region / barrier / serial changes (1 out_a in the core's region, 2 the core's max
// in the scores phase, 4 the site's first phase over the whole team, 8 xn's Q8_K in the router's region, 16 each thread
// its own h blocks, 32 the core's and the routing's buffers kept across layers + the scores prefetched before the
// top-k; 64 the core's row pointers written by the scores phase's threads, out_a / out_b's buffers zeroed in the core's
// region; 128 the scores of all heads of a row in one loop (dot_heads); 256 (batched, by CMG) the core's phases with CMG-local
// barriers; 512 dot_heads' 6 horizontal sums by unzip-add steps in FADDV's order; 1024 the core's exps and the experts'
// SwiGLU exps 16 at a time (vexpf.h: the libm expf's bits, used only when its self-check passes); 2048 the core's KV rows
// prefetched a few rows ahead in the scores phase (loads only)); each keeps every operation's inputs and order. Default 63 (all): 12 A64FX nodes, wiki decode 20.75-20.99 ms
// per token against 22.18-22.33 with 0 (three alternated runs, same allocation, same md5); 0 gives the old path
static int merge_bits() { static const int v = std::getenv("STRATA_DS41_MERGE") ? std::atoi(std::getenv("STRATA_DS41_MERGE")) : 63; return v; }
static bool xpf_on() { static const bool on = strata_env_is_one("STRATA_DS41_XPF"); return on; }

// x . y over n floats (n % 16 == 0 on the SVE path)
inline float dot(const float *x, const float *y, int n) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    const svbool_t p = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        a0 = svmla_f32_x(p, a0, svld1_f32(p, x + i), svld1_f32(p, y + i));
        a1 = svmla_f32_x(p, a1, svld1_f32(p, x + i + 16), svld1_f32(p, y + i + 16));
        a2 = svmla_f32_x(p, a2, svld1_f32(p, x + i + 32), svld1_f32(p, y + i + 32));
        a3 = svmla_f32_x(p, a3, svld1_f32(p, x + i + 48), svld1_f32(p, y + i + 48));
    }
    for (; i < n; i += 16) {
        const svbool_t q = svwhilelt_b32(i, n);
        a0 = svmla_f32_m(q, a0, svld1_f32(q, x + i), svld1_f32(q, y + i));
    }
    return svaddv_f32(p, svadd_f32_x(p, svadd_f32_x(p, a0, a1), svadd_f32_x(p, a2, a3)));
#else
    float s = 0;
    for (int i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
#endif
}
// dot(x, y r) with y[i] r formed in registers: the bits of dot() over a materialized y r
inline float dot_scaled(const float *x, const float *y, float r, int n) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    const svbool_t p = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        a0 = svmla_f32_x(p, a0, svld1_f32(p, x + i), svmul_n_f32_x(p, svld1_f32(p, y + i), r));
        a1 = svmla_f32_x(p, a1, svld1_f32(p, x + i + 16), svmul_n_f32_x(p, svld1_f32(p, y + i + 16), r));
        a2 = svmla_f32_x(p, a2, svld1_f32(p, x + i + 32), svmul_n_f32_x(p, svld1_f32(p, y + i + 32), r));
        a3 = svmla_f32_x(p, a3, svld1_f32(p, x + i + 48), svmul_n_f32_x(p, svld1_f32(p, y + i + 48), r));
    }
    for (; i < n; i += 16) {
        const svbool_t q = svwhilelt_b32(i, n);
        a0 = svmla_f32_m(q, a0, svld1_f32(q, x + i), svmul_n_f32_x(q, svld1_f32(q, y + i), r));
    }
    return svaddv_f32(p, svadd_f32_x(p, svadd_f32_x(p, a0, a1), svadd_f32_x(p, a2, a3)));
#else
    float s = 0;
    for (int i = 0; i < n; ++i) { const float v = y[i] * r; s += x[i] * v; }
    return s;
#endif
}
// o[0..n) = sum_r w[r] rows[r][0..n)
// (columns [off, off + n) of the rows: a 128-aligned block of them gives the same bits as the whole row's pass)
inline void axpy_rows(const float *const *rows, const float *w, int nr, int n, float *o, int off = 0) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    for (int i0 = 0; i0 < n; i0 += 128) {   // 8 vectors of accumulators per pass over the rows
        const int m = std::min(128, n - i0);
        vf32 a[8];
        for (int v = 0; v < 8; ++v) a[v] = svdup_f32(0);
        for (int r = 0; r < nr; ++r) {
            const float *x = rows[r] + off + i0;
            const svfloat32_t wr = svdup_f32(w[r]);
            for (int v = 0; v < 8; ++v) {
                const svbool_t q = svwhilelt_b32(16 * v, m);
                a[v] = svmla_f32_m(q, a[v], svld1_f32(q, x + 16 * v), wr);
            }
        }
        for (int v = 0; v < 8; ++v) svst1_f32(svwhilelt_b32(16 * v, m), o + i0 + 16 * v, a[v]);
    }
#else
    for (int i = 0; i < n; ++i) o[i] = 0;
    for (int r = 0; r < nr; ++r)
        for (int i = 0; i < n; ++i) o[i] += w[r] * rows[r][off + i];
#endif
}

#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
// o[h][c0, c0 + 16) = sum_r w[h nr + r] rows[r][c0, c0 + 16) for NH heads (the first nh of them), each row's block
// loaded once: per element the FMA chain of axpy_rows, in row order
template <int NH>
inline void axpy_heads(const float *const *rows, const float *w, int nr, int c0, float *o, int hd, int nh = NH) {
    const svbool_t pt = svptrue_b32();
    vf32 acc[NH];
    for (int h = 0; h < NH; ++h) acc[h] = svdup_f32(0);
    for (int r = 0; r < nr; ++r) {
        const svfloat32_t xr = svld1_f32(pt, rows[r] + c0);
        for (int h = 0; h < NH; ++h)
            if (h < nh) acc[h] = svmla_f32_x(pt, acc[h], xr, svdup_f32(w[size_t(h) * nr + r]));
    }
    for (int h = 0; h < NH; ++h)
        if (h < nh) svst1_f32(pt, o + size_t(h) * hd + c0, acc[h]);
}
#endif

#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
// out[h] = dot(q + h hd, row, hd) for NH heads (the first nh), hd a multiple of 64: each head's four accumulators and
// their order are dot()'s (the same bits); the row's 64 floats loaded once for all heads, the heads' FMA chains
// independent (A64FX overlaps few of the chains of separate dot() calls)
template <int NH>
inline void dot_heads(const float *q, const float *row, int hd, float *out, int nh = NH) {
    const svbool_t p = svptrue_b32();
    vf32 a[NH][4];
    for (int h = 0; h < NH; ++h) a[h][0] = a[h][1] = a[h][2] = a[h][3] = svdup_f32(0);
    for (int i = 0; i + 64 <= hd; i += 64) {
        const svfloat32_t r0 = svld1_f32(p, row + i), r1 = svld1_f32(p, row + i + 16), r2 = svld1_f32(p, row + i + 32), r3 = svld1_f32(p, row + i + 48);
        for (int h = 0; h < NH; ++h) {
            if (h >= nh) continue;
            const float *x = q + size_t(h) * hd + i;
            a[h][0] = svmla_f32_x(p, a[h][0], svld1_f32(p, x), r0);
            a[h][1] = svmla_f32_x(p, a[h][1], svld1_f32(p, x + 16), r1);
            a[h][2] = svmla_f32_x(p, a[h][2], svld1_f32(p, x + 32), r2);
            a[h][3] = svmla_f32_x(p, a[h][3], svld1_f32(p, x + 48), r3);
        }
    }
    if (NH == 6 && (merge_bits() & 512)) {   // (STRATA_DS41_MERGE & 512) the 6 heads' horizontal sums at once: FADDV's
        // recursive pairwise order (R(x0..15) = R(x0..7) + R(x8..15), down to x0 + x1) by unzip-add steps, each step one
        // level of the tree for every head (lanes per head halving), so each head's sum has svaddv's bits
        vf32 v[6];
        for (int h = 0; h < 6; ++h) v[h] = h < nh ? svadd_f32_x(p, svadd_f32_x(p, a[h][0], a[h][1]), svadd_f32_x(p, a[h][2], a[h][3])) : svdup_f32(0);
        auto ua = [&](svfloat32_t x, svfloat32_t y) { return svadd_f32_x(p, svuzp1_f32(x, y), svuzp2_f32(x, y)); };
        const svfloat32_t w01 = ua(v[0], v[1]), w23 = ua(v[2], v[3]), w45 = ua(v[4], v[5]);
        const svfloat32_t x03 = ua(w01, w23), x45 = ua(w45, w45);
        const svfloat32_t y = ua(x03, x45);
        float t[16];
        svst1_f32(p, t, ua(y, y));
        for (int h = 0; h < nh; ++h) out[h] = t[h];
        return;
    }
    for (int h = 0; h < NH; ++h)
        if (h < nh) out[h] = svaddv_f32(p, svadd_f32_x(p, svadd_f32_x(p, a[h][0], a[h][1]), svadd_f32_x(p, a[h][2], a[h][3])));
}
#endif

// sum of squares in fixed blocks of 1024 (a float vector sum each, the blocks added in order in double): the same bits
// for any thread count, which the replicated residual needs (every rank must compute identical values)
double sumsq_blocks(const float *x, int n) {
    const int nb = (n + 1023) / 1024;
    double part[64];
    if (nb > 64) throw std::runtime_error("ds41: sumsq_blocks too long");
    auto blocks = [&](int a, int e) {
        for (int b = a; b < e; ++b) {
            const int i0 = 1024 * b, i1 = std::min(n, i0 + 1024);
            part[b] = dot(x + i0, x + i0, i1 - i0);
        }
    };
    if (n >= 8192) team_for(nb, blocks); else blocks(0, nb);
    double s = 0;
    for (int b = 0; b < nb; ++b) s += part[b];
    return s;
}
void rms_norm(const float *x, const float *w, float *y, int n, float eps) {
    const float r = 1.0f / std::sqrt(float(sumsq_blocks(x, n) / n) + eps);
    if (n >= 8192) {
        team_for(n / 64, [&](int a, int b) {   // (n % 64 == 0 here: the 4 x hidden streams)
#pragma omp simd
            for (int i = 64 * a; i < 64 * b; ++i) y[i] = w ? x[i] * r * w[i] : x[i] * r;
        });
        for (int i = n / 64 * 64; i < n; ++i) y[i] = w ? x[i] * r * w[i] : x[i] * r;
    } else if (w) {
#pragma omp simd
        for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
    } else {
#pragma omp simd
        for (int i = 0; i < n; ++i) y[i] = x[i] * r;
    }
}
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
float swiglu(float g, float u, float limit) {
    if (limit > 0) g = std::min(g, limit), u = std::clamp(u, -limit, limit);
    return g * sigmoid(g) * u;
}
// swiglu over g[0..n) with u[0..n) into g: the exps 16 at a time (vexpf), every other operation swiglu's
void swiglu_rows(float *g, const float *u, int n, float limit) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float gg[16], uu[16], e[16];
        for (int k = 0; k < 16; ++k) {
            gg[k] = g[i + k], uu[k] = u[i + k];
            if (limit > 0) gg[k] = std::min(gg[k], limit), uu[k] = std::clamp(uu[k], -limit, limit);
            e[k] = -gg[k];
        }
        vexpf(e, e, 16);
        for (int k = 0; k < 16; ++k) g[i + k] = gg[k] * (1.0f / (1.0f + e[k])) * uu[k];
    }
    for (; i < n; ++i) g[i] = swiglu(g[i], u[i], limit);
}
static bool vexp_on() { static const bool v = (merge_bits() & 1024) && vexpf_ok(); return v; }
// indices of the k largest values, ties to the lower index, by value (descending)
// top_k's result for small k by one ascending scan with the best k kept sorted (the same strict order)
std::vector<int> top_k_small(const float *v, int n, int k) {
    k = std::min(k, n);
    std::vector<int> b;
    b.reserve(static_cast<size_t>(k) + 1);
    for (int t = 0; t < n; ++t) {   // t ascending: an equal value never beats a kept one
        if (int(b.size()) == k && !(v[t] > v[b.back()])) continue;
        size_t p = b.size();
        while (p > 0 && v[t] > v[b[p - 1]]) --p;
        b.insert(b.begin() + std::ptrdiff_t(p), t);
        if (int(b.size()) > k) b.pop_back();
    }
    return b;
}
// top_k_small's result with 16-wide skips (STRATA_DS41_MERGE & 32): once k values are kept, a vector of 16 whose lanes
// are all !(v > the k-th) is passed over (the scalar loop would insert none of them); the others go element by element
// in order, so the picks and their tie order are top_k_small's
std::vector<int> top_k_small_skip(const float *v, int n, int k) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    k = std::min(k, n);
    std::vector<int> b;
    b.reserve(static_cast<size_t>(k) + 1);
    auto one = [&](int t) {
        if (int(b.size()) == k && !(v[t] > v[b.back()])) return;
        size_t p = b.size();
        while (p > 0 && v[t] > v[b[p - 1]]) --p;
        b.insert(b.begin() + std::ptrdiff_t(p), t);
        if (int(b.size()) > k) b.pop_back();
    };
    const svbool_t pt = svptrue_b32();
    int t = 0;
    for (; t + 16 <= n; t += 16) {
        if (int(b.size()) == k && !svptest_any(pt, svcmpgt_n_f32(pt, svld1_f32(pt, v + t), v[b.back()]))) continue;
        for (int u = t; u < t + 16; ++u) one(u);
    }
    for (; t < n; ++t) one(t);
    return b;
#else
    return top_k_small(v, n, k);
#endif
}
// the rows of top_k(v, n, k) in ascending row order: the k-th key by a radix select on order-preserving keys, then a
// scan by row (equal keys at the threshold: the lowest rows, top_k's tie order)
// STRATA_DS41_TOPK_PAR=1 (opt-in): top_k_rows over many rows (n >= 65536, outside a parallel region) by the team: the
// same keys, the same per-pass histograms (summed per thread), the same threshold and the same rows (equal keys at the
// threshold still the lowest rows); the long-context indexer's selection (one thread over 262K rows: ~4 ms per layer)
static bool top_k_rows_par(const float *v, int n, int k, std::vector<int> &out) {
    static const bool on = strata_env_is_one("STRATA_DS41_TOPK_PAR");
    static const int nmin = std::getenv("STRATA_DS41_TOPK_PAR_MIN") ? std::max(1024, std::atoi(std::getenv("STRATA_DS41_TOPK_PAR_MIN"))) : 65536;
    if (!on || n < nmin || omp_in_parallel()) return false;
    std::vector<uint32_t> key(static_cast<size_t>(n));
    const int nth = omp_get_max_threads();
    std::vector<int> hist(size_t(nth) * 2048);
    uint32_t prefix = 0, mask = 0;
    int need = k;
    const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
    std::vector<int> gt(size_t(nth) + 1, 0), eq(size_t(nth) + 1, 0);
    std::vector<std::vector<int>> part(static_cast<size_t>(nth));
#pragma omp parallel num_threads(nth)
    {
        const int t = omp_get_thread_num(), nt = omp_get_num_threads(), a = int(int64_t(n) * t / nt), b = int(int64_t(n) * (t + 1) / nt);
        for (int i = a; i < b; ++i) {
            uint32_t u;
            std::memcpy(&u, v + i, 4);
            if (u == 0x80000000u) u = 0;
            key[size_t(i)] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        }
        for (int pass = 0; pass < 3; ++pass) {
            const int sh = shifts[pass], nb = 1 << widths[pass];
            int *h = hist.data() + size_t(t) * 2048;
            std::fill(h, h + nb, 0);
            for (int i = a; i < b; ++i)
                if ((key[size_t(i)] & mask) == prefix) ++h[size_t((key[size_t(i)] >> sh) & uint32_t(nb - 1))];
#pragma omp barrier
#pragma omp single
            {
                for (int d = 0; d < nb; ++d)
                    for (int q = 1; q < nt; ++q) hist[size_t(d)] += hist[size_t(q) * 2048 + d];
                int d = nb - 1;
                for (; d > 0 && hist[size_t(d)] < need; --d) need -= hist[size_t(d)];
                prefix |= uint32_t(d) << sh, mask |= uint32_t(nb - 1) << sh;
            }
        }
        int g = 0, e = 0;
        for (int i = a; i < b; ++i) g += key[size_t(i)] > prefix, e += key[size_t(i)] == prefix;
        gt[size_t(t) + 1] = g, eq[size_t(t) + 1] = e;
#pragma omp barrier
#pragma omp single
        for (int q = 0; q < nt; ++q) gt[size_t(q) + 1] += gt[size_t(q)], eq[size_t(q) + 1] += eq[size_t(q)];
        int left = std::max(0, std::min(e, need - eq[size_t(t)]));   // this chunk's share of the lowest equal rows
        auto &o = part[size_t(t)];
        for (int i = a; i < b; ++i) {
            const uint32_t kk = key[size_t(i)];
            if (kk > prefix) o.push_back(i);
            else if (kk == prefix && left > 0) o.push_back(i), --left;
        }
    }
    for (auto &o : part) out.insert(out.end(), o.begin(), o.end());
    return true;
}
void top_k_rows(const float *v, int n, int k, std::vector<int> &out) {
    out.clear();
    k = std::min(k, n);
    if (k <= 0) return;
    if (k == n) { for (int t = 0; t < n; ++t) out.push_back(t); return; }
    if (top_k_rows_par(v, n, k, out)) return;
    std::vector<uint32_t> key(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) {
        uint32_t u;
        std::memcpy(&u, v + t, 4);
        if (u == 0x80000000u) u = 0;   // -0 == +0 for the float order
        key[size_t(t)] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    // the k-th largest key, 11 / 11 / 10 bits from the top
    uint32_t prefix = 0, mask = 0;
    int need = k;
    const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
    std::vector<int> hist(2048);
    for (int pass = 0; pass < 3; ++pass) {
        const int sh = shifts[pass], nb = 1 << widths[pass];
        std::fill(hist.begin(), hist.begin() + nb, 0);
        for (int t = 0; t < n; ++t)
            if ((key[size_t(t)] & mask) == prefix) ++hist[size_t((key[size_t(t)] >> sh) & uint32_t(nb - 1))];
        int d = nb - 1;
        for (; d > 0 && hist[size_t(d)] < need; --d) need -= hist[size_t(d)];
        prefix |= uint32_t(d) << sh, mask |= uint32_t(nb - 1) << sh;
    }
    // keys above the threshold, then `need` of the equal ones (lowest rows first), all in row order
    for (int t = 0; t < n; ++t) {
        const uint32_t kk = key[size_t(t)];
        if (kk > prefix) out.push_back(t);
        else if (kk == prefix && need > 0) out.push_back(t), --need;
    }
}
std::vector<int> top_k(const float *v, int n, int k) {
    std::vector<int> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, n);
    // (a strict total order: nth_element then sorting the first k gives partial_sort's sequence, in O(n + k log k))
    const auto cmp = [&](int a, int b) { return v[a] > v[b] || (v[a] == v[b] && a < b); };
    if (k < n) std::nth_element(idx.begin(), idx.begin() + k, idx.end(), cmp);
    idx.resize(static_cast<size_t>(k));
    std::sort(idx.begin(), idx.end(), cmp);
    return idx;
}

// ggml's YaRN rotation table for one position (src/program/ds41_decode.cpp rope_table), pairs {cos, sin}
void rope_table(const Rope &r, int n_dims, int64_t pos, float sign, float *cs) {
    const float theta_scale = std::pow(r.base, -2.0f / float(n_dims));
    float corr[2] = {0, 0};
    if (r.ext_factor != 0.0f) {
        auto dim = [&](float rot) { return float(n_dims) * std::log(float(r.ctx_orig) / (rot * 2.0f * float(M_PI))) / (2.0f * std::log(r.base)); };
        corr[0] = std::max(0.0f, std::floor(dim(r.beta_fast)));
        corr[1] = std::min(float(n_dims - 1), std::ceil(dim(r.beta_slow)));
    }
    float theta = float(pos);
    for (int i0 = 0; i0 < n_dims; i0 += 2) {
        const float extrap = theta, interp = r.freq_scale * extrap;
        float t = interp, mscale = r.attn_factor;
        if (r.ext_factor != 0.0f) {
            const float y = (float(i0 / 2) - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
            const float ramp = (1 - std::min(1.0f, std::max(0.0f, y))) * r.ext_factor;
            t = interp * (1 - ramp) + extrap * ramp;
            mscale *= 1.0f + 0.1f * std::log(1.0f / r.freq_scale);
        }
        cs[i0] = std::cos(t) * mscale;
        cs[i0 + 1] = std::sin(t) * mscale * sign;
        theta *= theta_scale;
    }
}
void rope(float *v, int pairs, const float *cs) {
    for (int p = 0; p < pairs; ++p) {
        const float c = cs[2 * p], s = cs[2 * p + 1], x0 = v[2 * p], x1 = v[2 * p + 1];
        v[2 * p] = x0 * c - x1 * s;
        v[2 * p + 1] = x0 * s + x1 * c;
    }
}

// ------------------------------------------------------------------------------------------------ the model
struct Chunk { Mat gate, up, down; };
struct Layer {
    Vec attn_norm, sinks, q_a_norm, kv_a_norm, ffn_norm, bias;
    Mat q_a, q_b, kv, out_a, out_b, gate_inp;
    Vec hc_attn_base, hc_attn_scale, hc_ffn_base, hc_ffn_scale;
    Mat hc_attn_fn, hc_ffn_fn;
    // compressor (KV source layers) and indexer (index source layers)
    Mat comp_kv, comp_gate, idx_k, idx_q_b, idx_proj;
    Vec comp_norm, idx_k_norm;
    // Engram
    Mat engram_wkv, engram_table;
    Vec engram_gate;
    // this rank's expert chunks: chunk[9 e + c] (index into chunks, -1: another rank's)
    std::vector<int> chunk_of;
    // tensor parallel (split_tp): this rank's q_b rows, out_a's column blocks of its heads per output group, out_b rows
    Mat q_b_t, out_b_t, q_a_t, kv_t, engram_wkv_t, idx_q_b_t, comp_kv_t, comp_gate_t;
    Mat idx_q_b_full;   // STRATA_DS41_ES_INDEX: all heads for entry-owned score/selection
    Mat out_b_full;   // STRATA_DS41_ES_OUTB: every row of out_b (the owner's own entries' rows)
    Mat q_a_full, kv_full, comp_kv_full, comp_gate_full;   // STRATA_DS41_ES_QK: every row
    Mat gate_inp_t;   // STRATA_DS41_ROUTER_TP: the router rows of experts [ge0, ge0 + rows)
    int ge0 = 0;
    struct OutA { int g, a, b; Mat m; };
    std::vector<OutA> out_a_t;
    int engram_fd = -1;          // this rank's row shard on /local (--engram-dir), rows [engram_r0, engram_r1)
    int64_t engram_r0 = 0, engram_r1 = 0;
    const uint8_t *engram_mem = nullptr;   // the shard in RAM (engram_to_ram)
    const uint8_t *engram_part = nullptr;  // (STRATA_DS41_ENGRAM_RAM_GB) the shard's rows [engram_r0, engram_pr1) in RAM
    Mat q_b_x;                              // (STRATA_DS41_HEAD_BAL, a helper) q_b's rows of the head it helps with
    int64_t engram_pr1 = 0;
    std::vector<Chunk> chunks, shared;   // shared: this rank's shared-expert chunks (by chunk number in shared_c)
    std::vector<Chunk> shared_all;       // load_shared_all: every shared chunk (assigned per token to the least-loaded ranks)
    std::vector<int> shared_c;
};

struct Runner {
    Model &M;
    Comm &comm;
    Config c;
    int layers = 0, context = 4096, ring = 0, nchunk = 0, vocab0 = 0, vocab1 = 0;
    std::vector<Layer> L;
    Mat embd, head;
    Vec out_norm;
    // state: a sequence's KV rings (per layer: ringcap x head_dim, the window plus a pass's positions), compressed
    // caches (per KV source: rows x (head_dim + index_dim)), the ratio-2 stash and its token history; a pass's tokens
    // (Ent) carry their streams, coefficients, rope tables and per-index-source selections (ascending rows)
    static constexpr int kMaxPass = 32;
    // a pass's entries: up to kMaxBatch (a batched step, one entry per sequence; a long prefill pass); a sequence's own
    // entries in one pass up to pass_max() (its KV ring holds the window plus pass_max() - 1 rows)
    static constexpr int kMaxBatch = 512;
    // STRATA_DS41_PASS_MAX (default kMaxPass = 32, up to kMaxBatch): a sequence's entries in one pass, so the prompt's
    // passes (prefill, --prompt-pass) may be longer; the ring grows by the same rows
    static int pass_max() {
        static const int v = std::getenv("STRATA_DS41_PASS_MAX") ? std::max(kMaxPass, std::min(kMaxBatch, std::atoi(std::getenv("STRATA_DS41_PASS_MAX")))) : kMaxPass;
        return v;
    }
    struct Seq {
        std::vector<Vec> window, cache, stash_v, stash_s;
        std::vector<int> hist;
        int cmg = -1;   // the CMG its KV lives on (batched decode: its attention runs there); -1: spread over all
    };
    struct Ent {
        Seq *s = nullptr;
        int pos = 0, token = 0;
        int last = -1;   // the drafter's block: its window rows run to this position (non-causal); -1: to pos
        Vec h, h2, x, xn, y, coeff_attn, coeff_ffn, cs;
        std::vector<std::vector<int>> sel;
        std::vector<int> candidates;
        std::vector<Vec> sv, ss;   // spec_track_: the ratio-2 stash this token left (per KV source layer; empty: none)
        const float *table(int which) const { return cs.data() + 64 * which; }
    };
    bool spec_track_ = false;
    // speculative verification: the stash of the token at index m of a pass (the last accepted one), else `saved` (the
    // stash before the pass); the KV rows of rejected positions are rewritten before anything reads them
    void rewind_stash(Seq &q, const std::vector<Ent> &E, int m, const std::vector<Vec> &saved_v, const std::vector<Vec> &saved_s) const {
        for (int l = 0; l < layers; ++l) {
            if (q.stash_v[size_t(l)].empty()) continue;
            int j = m;
            while (j >= 0 && E[size_t(j)].sv[size_t(l)].empty()) --j;
            q.stash_v[size_t(l)] = j >= 0 ? E[size_t(j)].sv[size_t(l)] : saved_v[size_t(l)];
            q.stash_s[size_t(l)] = j >= 0 ? E[size_t(j)].ss[size_t(l)] : saved_s[size_t(l)];
        }
    }
    int ringcap = 0;
    Seq seq0;                 // step()'s sequence
    std::vector<Ent> ents_;   // step()'s pass
    std::map<std::string, double> timers;
    // a timer by its key's address (the keys are literals or stable pointers): the map's element looked up once, then
    // a pointer hash (std::map with a std::string built per update cost ~0.3 us each, ~1000 updates per token)
    std::unordered_map<const char *, double *> tcache_;
    double &tm(const char *k) {
        const auto it = tcache_.find(k);
        if (it != tcache_.end()) return *it->second;
        double *p = &timers[k];
        tcache_.emplace(k, p);
        return *p;
    }

    std::string engram_dir;
    double stage_s = 0;
    Runner(Model &m, Comm &cm, int nlayers, int ctx, const std::string &edir = "") : M(m), comm(cm), context(ctx), engram_dir(edir) {
        const auto &art = *M.art;
        c = Config::read(*art.metadata(), art.descriptor(), [&](const std::string &n) { return M.has(n); },
                         [&](const std::string &n) { return tensor_ints(M.t(n)); });
        layers = nlayers > 0 ? std::min(nlayers, c.layers) : c.layers;
        // lossy opt-ins (judged by the KL gate, not exact): STRATA_DS41_INDEX_TOPK = the indexer's selected rows (model:
        // 512), STRATA_DS41_EXPERT_TOPK = routed experts per token (model: 6)
        if (const char *e = std::getenv("STRATA_DS41_INDEX_TOPK")) c.index_top_k = std::max(1, std::atoi(e));
        if (const char *e = std::getenv("STRATA_DS41_EXPERT_TOPK")) c.top_k = std::clamp(std::atoi(e), 1, c.top_k);
        ring = c.window;
        nchunk = c.ff / 256;
        if (c.ff % 256 || c.hidden % 256) throw std::runtime_error("ds41: widths not whole 256-blocks");
        const int R = comm.rank, W = comm.world;
        embd = M.mat("token_embd.weight", true);
        vocab0 = int(int64_t(c.vocab) * R / W), vocab1 = int(int64_t(c.vocab) * (R + 1) / W);
        head = M.rows("output.weight", vocab0, vocab1 - vocab0);
        out_norm = M.floats("output_norm.weight");
        L.resize(static_cast<size_t>(layers));
        for (int l = 0; l < layers; ++l) load_layer(l);
        // state
        ringcap = ring + pass_max() - 1;
        init_seq(seq0);
        // (an owner's out_a shard: up to 11 x 8192 at B=128; a long pass: its owners' entries x the widest row)
        if (ag_bits() || entry_shard_env())
            comm.xc_init(std::max(131072, (pass_max() / comm.world + 2) * std::max(c.hidden, c.out_groups * c.out_rank)));   // (collective: every rank builds the target Runner)
    }
    // --dspark: the DSpark drafter's blocks (llama.cpp's dflash GGUF) with the target's geometry in `cfg` (draft_config),
    // the target's embedding slice and lm head; its seq0 holds the drafter's key rings
    Runner(Model &m, Comm &cm, const Config &cfg, int ctx, const Runner &target) : M(m), comm(cm), c(cfg), context(ctx) {
        layers = c.layers;
        ring = c.window;
        nchunk = c.ff / 256;
        if (c.ff % 256 || c.hidden % 256) throw std::runtime_error("ds41: drafter widths not whole 256-blocks");
        embd = target.embd, embd_mem = target.embd_mem, vocab0 = target.vocab0, vocab1 = target.vocab1, head = target.head;
        out_norm = M.floats("output_norm.weight");
        L.resize(static_cast<size_t>(layers));
        for (int l = 0; l < layers; ++l) load_layer(l);
        ringcap = ring + pass_max() - 1;
        init_seq(seq0);
    }
    void load_layer(int l) {
        const int R = comm.rank, W = comm.world;
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer &Y = L[size_t(l)];
        Y.attn_norm = M.floats(p + "attn_norm.weight");
        Y.sinks = M.floats(p + "attn_sinks.weight");
        Y.q_a_norm = M.floats(p + "attn_q_a_norm.weight");
        Y.kv_a_norm = M.floats(p + "attn_kv_a_norm.weight");
        Y.ffn_norm = M.floats(p + "ffn_norm.weight");
        Y.bias = M.floats(p + "exp_probs_b.bias");
        Y.q_a = M.mat(p + "attn_q_a.weight");
        Y.q_b = M.mat(p + "attn_q_b.weight");
        Y.kv = M.mat(p + "attn_kv.weight");
        Y.out_a = M.mat(p + "attn_output_a.weight");
        Y.out_b = M.mat(p + "attn_output_b.weight");
        Y.gate_inp = M.mat(p + "ffn_gate_inp.weight");
        Y.hc_attn_fn = M.mat(p + "hc_attn_fn.weight");
        Y.hc_ffn_fn = M.mat(p + "hc_ffn_fn.weight");
        Y.hc_attn_base = M.floats(p + "hc_attn_base.weight");
        Y.hc_attn_scale = M.floats(p + "hc_attn_scale.weight");
        Y.hc_ffn_base = M.floats(p + "hc_ffn_base.weight");
        Y.hc_ffn_scale = M.floats(p + "hc_ffn_scale.weight");
        if (c.kv_src[size_t(l)] == l) {
            Y.comp_kv = M.mat(p + "attn_compressor_kv.weight");
            if (c.ratio[size_t(l)] > 1) Y.comp_gate = M.mat(p + "attn_compressor_gate.weight");
            Y.comp_norm = M.floats(p + "attn_compressor_norm.weight");
            Y.idx_k = M.mat(p + "indexer.attn_k.weight");
            Y.idx_k_norm = M.floats(p + "indexer.k_norm.weight");
        }
        if (c.topk_src[size_t(l)] == l) {
            Y.idx_q_b = M.mat(p + "indexer.attn_q_b.weight");
            Y.idx_proj = M.mat(p + "indexer.proj.weight");
        }
        if (c.engram_index(l) >= 0) {
            Y.engram_wkv = M.mat(p + "engram_wkv.weight");
            Y.engram_table = M.mat(p + "engram_embd.weight", true);   // rows read on demand
            if (!engram_dir.empty()) stage_engram(l, Y);
            Vec q = M.floats(p + "engram_q.weight"), k = M.floats(p + "engram_k.weight");
            for (size_t i = 0; i < q.size(); ++i) q[i] *= k[i];
            Y.engram_gate = q;
        }
        // expert chunks: gate/up rows [256 c, +256) of expert e (read directly), down's block c of each of its rows (each
        // expert's down matrix read whole by rank e % world, its column blocks sent to their owners)
        const Mat G = M.mat(p + "ffn_gate_exps.weight", true), U = M.mat(p + "ffn_up_exps.weight", true), D = M.mat(p + "ffn_down_exps.weight", true);
        const auto &tg = M.t(p + "ffn_gate_exps.weight"), &tu = M.t(p + "ffn_up_exps.weight"), &td = M.t(p + "ffn_down_exps.weight");
        const int E = c.experts;
        if (G.cols != c.hidden || G.rows != E * c.ff || D.cols != c.ff || D.rows != E * c.hidden) throw std::runtime_error("ds41: expert geometry");
        Y.chunk_of.assign(size_t(E) * nchunk, -1);
        std::vector<std::pair<int, int>> mine;
        for (int e = 0; e < E; ++e)
            for (int ch = 0; ch < nchunk; ++ch)
                if ((9 * e + ch) % W == R) mine.push_back({e, ch});
        const size_t bb = D.rb / size_t(nchunk), dbytes = bb * size_t(c.hidden);
        Y.chunks.resize(mine.size());
        std::vector<RReq> reqs;
        for (size_t i = 0; i < mine.size(); ++i) {
            const auto [e, ch] = mine[i];
            Chunk &k = Y.chunks[i];
            k.gate = G, k.up = U, k.down = D;
            k.gate.rows = k.up.rows = 256;
            uint8_t *pg = anon(G.rb * 256), *pu = anon(U.rb * 256);
            if (!g_replay) {
                reqs.push_back({&tg, G.rb * (size_t(e) * c.ff + 256 * ch), G.rb * 256, pg});
                reqs.push_back({&tu, U.rb * (size_t(e) * c.ff + 256 * ch), U.rb * 256, pu});
            }
            k.gate.p = pg, k.up.p = pu;
            k.down.p = anon(dbytes), k.down.rows = c.hidden, k.down.cols = 256, k.down.rb = bb;
            g_resident += G.rb * 256 + U.rb * 256 + dbytes;
            Y.chunk_of[size_t(e) * nchunk + ch] = int(i);
        }
        read_many(reqs);
        if (!g_replay) {   // down: this rank reads experts e % W == R whole, sends block c to rank (9e + c) % W
            std::vector<std::vector<uint8_t>> out(static_cast<size_t>(W));
            const size_t fb = D.rb * size_t(c.hidden);
            std::vector<uint8_t> all(fb * size_t((E - R + W - 1) / W));
            {
                std::vector<RReq> dr;
                for (int e = R, k = 0; e < E; e += W, ++k) dr.push_back({&td, fb * size_t(e), fb, all.data() + fb * size_t(k)});
                read_many(dr);
            }
            for (int e = R, k = 0; e < E; e += W, ++k) {
                const uint8_t *fullp = all.data() + fb * size_t(k);
                for (int ch = 0; ch < nchunk; ++ch) {
                    auto &o = out[size_t((9 * e + ch) % W)];
                    const size_t at = o.size();
                    o.resize(at + dbytes);
                    for (int r = 0; r < c.hidden; ++r) std::memcpy(o.data() + at + bb * r, fullp + D.rb * r + bb * ch, bb);
                }
            }
            // the blocks arrive in (source rank, expert, chunk) order: the order the senders produced them
            std::vector<uint8_t> in;
            if (W == 1) in = std::move(out[0]);
#ifdef STRATA_CPU_RUNNER_MPI
            else {
                std::vector<int> sc(static_cast<size_t>(W)), rc(static_cast<size_t>(W)), sd(static_cast<size_t>(W)), rd(static_cast<size_t>(W));
                for (int r = 0; r < W; ++r) sc[size_t(r)] = int(out[size_t(r)].size() / dbytes);
                MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
                MPI_Datatype blk;
                MPI_Type_contiguous(int(dbytes), MPI_BYTE, &blk);
                MPI_Type_commit(&blk);
                std::vector<uint8_t> send;
                int so = 0, ro = 0;
                for (int r = 0; r < W; ++r) {
                    sd[size_t(r)] = so, rd[size_t(r)] = ro;
                    so += sc[size_t(r)], ro += rc[size_t(r)];
                    send.insert(send.end(), out[size_t(r)].begin(), out[size_t(r)].end());
                }
                in.resize(size_t(ro) * dbytes);
                MPI_Alltoallv(send.data(), sc.data(), sd.data(), blk, in.data(), rc.data(), rd.data(), blk, MPI_COMM_WORLD);
                MPI_Type_free(&blk);
            }
#endif
            size_t at = 0;
            for (int src = 0; src < W; ++src)
                for (int e = src; e < E; e += W)
                    for (int ch = 0; ch < nchunk; ++ch)
                        if ((9 * e + ch) % W == R) {
                            std::memcpy(const_cast<uint8_t *>(Y.chunks[size_t(Y.chunk_of[size_t(e) * nchunk + ch])].down.p), in.data() + at, dbytes);
                            at += dbytes;
                        }
            if (at != in.size()) throw std::runtime_error("ds41: down exchange size");
        }
        {   // the shared expert: every rank reads its own chunks whole (11 MB per layer)
            const Mat SG = M.mat(p + "ffn_gate_shexp.weight", true), SU = M.mat(p + "ffn_up_shexp.weight", true), SD = M.mat(p + "ffn_down_shexp.weight", true);
            const size_t sbb = SD.rb / size_t(nchunk);
            std::vector<uint8_t> full(SD.rb * size_t(c.hidden));
            bool have = false;
            for (int ch = 0; ch < nchunk; ++ch) {
                if (ch % W != R) continue;
                if (!have && !g_replay) read_bytes(M.t(p + "ffn_down_shexp.weight"), 0, full.size(), full.data()), have = true;
                Chunk k;
                k.gate = SG, k.up = SU, k.down = SD;
                k.gate.rows = k.up.rows = 256;
                uint8_t *pg = anon(SG.rb * 256), *pu = anon(SU.rb * 256), *pd = anon(sbb * c.hidden);
                if (!g_replay) {
                    read_bytes(M.t(p + "ffn_gate_shexp.weight"), SG.rb * 256 * ch, SG.rb * 256, pg);
                    read_bytes(M.t(p + "ffn_up_shexp.weight"), SU.rb * 256 * ch, SU.rb * 256, pu);
                    for (int r = 0; r < c.hidden; ++r) std::memcpy(pd + sbb * r, full.data() + SD.rb * r + sbb * ch, sbb);
                }
                k.gate.p = pg, k.up.p = pu, k.down.p = pd, k.down.rows = c.hidden, k.down.cols = 256, k.down.rb = sbb;
                Y.shared.push_back(k), Y.shared_c.push_back(ch);
            }
        }
        if (comm.rank == 0 && (l % 8 == 0 || l == layers - 1))
            std::cerr << "DS41 loaded layer " << l << " resident " << g_resident / 1e9 << " GB\n";
    }
    int row_width() const { return c.head_dim + c.index_dim; }

    // STRATA_DS41_TP (1): after the load (and the image), the trunk's large attention projections become this rank's
    // slices: heads [h0, h1) (q_b rows, the attention itself), out_a's 256-column blocks of those heads (a partial
    // sum, all-reduced), out_b rows [y0, y1) (all-reduced); the full matrices are released
    bool tp = false;
    int h0 = 0, h1 = 0, y0 = 0, y1 = 0, qa0 = 0, kv0 = 0, ih0 = 0, ih1 = 0, ck0 = 0;   // + q_a / kv / compressor rows, indexer heads
    static Mat copy_rows(const Mat &m, int r0, int n) {
        Mat o = m;
        uint8_t *d = static_cast<uint8_t *>(mmap(nullptr, (m.rb * n + 4095) & ~size_t(4095), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        std::memcpy(d, m.p + m.rb * size_t(r0), m.rb * size_t(n));
        o.p = d, o.rows = n;
        return o;
    }
    static void release(Mat &m) { munmap(const_cast<uint8_t *>(m.p), (m.rb * m.rows + 4095) & ~size_t(4095)); m.p = nullptr; }
    // STRATA_DS41_ENGRAM_RAM (on): after split_tp, each Engram shard read from /local into anonymous memory when
    // MemAvailable keeps 2 GB beyond it. /local is the LLIO node-temporary area on the SIO nodes, so a 144-byte
    // pread is a ~1 ms round trip (4.6 ms per token for two Engram layers); from RAM a row is a memcpy
    double engram_ram_s = 0;
    size_t engram_ram_b = 0;
    static size_t mem_available() {
        std::ifstream f("/proc/meminfo");
        std::string k;
        size_t v = 0;
        while (f >> k >> v) { if (k == "MemAvailable:") return v * 1024; std::string unit; f >> unit; }
        return 0;
    }
    static bool engram_ram_on() { return !(strata_env_equals("STRATA_DS41_ENGRAM_RAM", "0")); }
    // STRATA_DS41_ENGRAM_RAM_LAYERS (1): the first N Engram layers in RAM; the later ones' rows are read from /local by
    // a helper thread at the pass start (prefetch_engram), long before the pass reaches them (layer 14)
    bool engram_ram_layer(int l) const {
        static const int n = std::getenv("STRATA_DS41_ENGRAM_RAM_LAYERS") ? std::atoi(std::getenv("STRATA_DS41_ENGRAM_RAM_LAYERS")) : 1;
        return engram_ram_on() && c.engram_index(l) >= 0 && c.engram_index(l) < n;
    }
    static uint8_t *ram_buffer(size_t bytes) {
        void *m = mmap(nullptr, (bytes + 4095) & ~size_t(4095), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return m == MAP_FAILED ? nullptr : static_cast<uint8_t *>(m);
    }
    // default on (STRATA_DS41_ENGRAM_RAM=0: off; the same rows, so the same bits). Shards staged in this run were filled into RAM on the way (stage_engram); shards an
    // earlier run of the allocation staged are read back from /local here (STRATA_DS41_LOCAL_THREADS /
    // STRATA_DS41_LOCAL_PIECE_MB, default 8 x 8 MiB: ~9.5 GB/s over 12 nodes; 8+ threads x 32 MiB collapse to ~50 MB/s on
    // localtmp, jobs 52856640 / 52873550)
    // STRATA_DS41_SHARED_BALANCE (1): the shared expert on every rank (+~1 GB per rank, read after the image so its
    // layout is unchanged); per token and layer its 9 chunks go to the ranks with the fewest routed chunks (the same
    // choice on every rank: the routing is replicated), which levels the expert phase the all-reduce waits for
    double shared_load_s = 0;
    // STRATA_DS41_EMBD_SLICE (1): this rank's vocabulary slice [vocab0, vocab1) of token_embd in RAM (read after the
    // image); a pass's embeddings come from the owning ranks through one all-reduce with zeros elsewhere (exact),
    // instead of a pread from the shared volume per token (~1 ms each)
    // STRATA_DS41_REHOME (1): layout only (the same bits). Matrices one gemv call reads alone are copied so each
    // thread's share of the call (its byte split; hc_fn: its 48 half-row items) sits in pages it touched first, on its
    // own CMG (2 MB calls: 8 us local vs 26 us from one CMG); the KV pages are spread over the CMGs (rows the indexer
    // selects are data dependent)
    static bool rehome_on() { return !(strata_env_equals("STRATA_DS41_REHOME", "0")); }
    static void rehome(Mat &m, int units = 0) {   // units 0: gemv_multi's split of one job (rows by bytes)
        if (!m.p || m.rows < 1) return;
        const size_t rb = m.rb, B = rb * size_t(m.rows);
        uint8_t *d = ram_buffer(B);
        if (!d) return;
        const uint8_t *src = m.p;
        team_run([&] {
            const int t = omp_get_thread_num(), T = omp_get_num_threads();
            size_t a0, b0;
            if (units) a0 = B / size_t(units) * (size_t(units) * t / T), b0 = B / size_t(units) * (size_t(units) * (t + 1) / T);
            else {
                const size_t ba = B * size_t(t) / size_t(T), bb = B * size_t(t + 1) / size_t(T);
                a0 = (ba + rb - 1) / rb * rb, b0 = std::min(B, (bb + rb - 1) / rb * rb);
            }
            if (b0 > a0) std::memcpy(d + a0, src + a0, b0 - a0);
        });
        Mat old = m;
        m.p = d;
        release(old);
    }
    // cmg >= 0: the pages on that CMG (touched by its threads only); else over every CMG round-robin
    // several matrices one gemv_multi call reads (whole, in job order): each thread's rows of its byte split of the
    // concatenation copied into fresh pages it touches first (layout only)
    static void rehome_joint(const std::vector<Mat *> &ms) {
        size_t B = 0;
        std::vector<size_t> off(1, 0);
        for (Mat *m : ms) B += m->p ? m->rb * size_t(m->rows) : 0, off.push_back(B);
        if (!B) return;
        std::vector<uint8_t *> d(ms.size(), nullptr);
        for (size_t i = 0; i < ms.size(); ++i)
            if (ms[i]->p && !(d[i] = ram_buffer(ms[i]->rb * size_t(ms[i]->rows)))) return;
        team_run([&] {
            const int t = omp_get_thread_num(), T = omp_get_num_threads();
            const size_t ba = B * size_t(t) / size_t(T), bb = B * size_t(t + 1) / size_t(T);
            for (size_t i = 0; i < ms.size(); ++i) {
                if (!ms[i]->p || bb <= off[i] || ba >= off[i + 1]) continue;
                const size_t rb = ms[i]->rb, rows = size_t(ms[i]->rows);
                const size_t a0 = (std::max(ba, off[i]) - off[i] + rb - 1) / rb, b0 = std::min(rows, (std::min(bb, off[i + 1]) - off[i] + rb - 1) / rb);
                if (b0 > a0) std::memcpy(d[i] + rb * a0, ms[i]->p + rb * a0, rb * (b0 - a0));
            }
        });
        for (size_t i = 0; i < ms.size(); ++i)
            if (ms[i]->p) {
                Mat old = *ms[i];
                ms[i]->p = d[i];
                release(old);
            }
    }
    static void spread_pages(float *p, size_t n, int cmg = -1) {
        const uintptr_t pg = uintptr_t(sysconf(_SC_PAGESIZE));
        const uintptr_t a = (uintptr_t(p) + pg - 1) & ~(pg - 1), b = uintptr_t(p + n) & ~(pg - 1);
        if (b <= a) return;
        madvise(reinterpret_cast<void *>(a), b - a, MADV_DONTNEED);   // (zeros again on the next touch)
        const size_t np = (b - a) / pg;
        const int G = cmg_threads();
        team_run([&] {
            const int t = omp_get_thread_num(), T = omp_get_num_threads();
            if (cmg < 0) {
                for (size_t q = size_t(t); q < np; q += size_t(T)) *reinterpret_cast<volatile float *>(a + q * pg) = 0.f;
                return;
            }
            const int m0 = std::min(T - 1, cmg * G), mc = std::min(T, m0 + G) - m0;
            if (t < m0 || t >= m0 + mc) return;
            for (size_t q = size_t(t - m0); q < np; q += size_t(mc)) *reinterpret_cast<volatile float *>(a + q * pg) = 0.f;
        });
    }
    double rehome_s = 0;
    void rehome_all() {
        if (!rehome_on()) return;
        const double t0 = now();
        for (int l = 0; l < layers; ++l) {
            Layer &Y = L[size_t(l)];
            rehome(Y.gate_inp), rehome(Y.q_b_t), rehome(Y.out_b_t), rehome(Y.engram_wkv_t), rehome(Y.out_b_full);
            rehome_joint({&Y.q_a_full, &Y.kv_full, &Y.comp_kv_full, &Y.comp_gate_full});
            {   // q_a + kv (+ the compressor's value / gate rows): attention()'s job order
                std::vector<Mat *> qj = {&Y.q_a_t, &Y.kv_t};
                const int ratio = c.ratio[size_t(l)];
                if (ratio && c.kv_src[size_t(l)] == l) {
                    if (Y.comp_kv_t.p) qj.push_back(&Y.comp_kv_t);
                    if (ratio > 1 && Y.comp_gate_t.p) qj.push_back(&Y.comp_gate_t);
                }
                rehome_joint(qj);
            }
            {
                std::vector<Mat *> oj;
                for (auto &o : Y.out_a_t) oj.push_back(&o.m);
                rehome_joint(oj);
            }
            if (Y.idx_q_b_full.p) {
                rehome_joint({&Y.idx_q_b_full, &Y.idx_proj});
                rehome(Y.idx_q_b_t);
            } else if (Y.idx_q_b_t.p) rehome_joint({&Y.idx_q_b_t, &Y.idx_proj});
            if (Y.hc_attn_fn.type == DS41_T_F32 && Y.hc_attn_fn.rows == 24) rehome(Y.hc_attn_fn, 48);
            if (Y.hc_ffn_fn.type == DS41_T_F32 && Y.hc_ffn_fn.rows == 24) rehome(Y.hc_ffn_fn, 48);
        }
        rehome(head);
        rehome_s = now() - t0;
    }
    // STRATA_DS41_IQ2R (1): the routed experts' IQ2_S / IQ2_XXS / IQ2_XS matrices repacked to IQ2R after the load (no
    // grid gathers: the same integer sums, so the same bits; +24-49 % kernel rate), in that order of gain per extra
    // byte (1.29x / 1.61x / 1.43x the bytes) while MemAvailable keeps STRATA_DS41_IQ2R_KEEP_GB (3) free
    double iq2r_s = 0, iq2r_gb = 0;
    int iq2r_n = 0;
    // STRATA_DS41_Q5P=1: every resident Q5_K matrix (the TP slices, the indexer's, the shared expert's chunks, the head)
    // to Q5P in place - Q5_K's bits in the kernel's read order (same bytes, same sums); after split_tp / repack
    void convert_q5p() {
        if (!(strata_env_is_one("STRATA_DS41_Q5P"))) return;
        std::set<const uint8_t *> done;
        int n = 0;
        auto cv = [&](Mat &m) {
            if (m.type != DS41_T_Q5_K || !m.p || !done.insert(m.p).second) return;
            ds41_q5k_to_q5p(const_cast<uint8_t *>(m.p), size_t(m.rows) * size_t(m.cols / 256));
            m.type = DS41_T_Q5P, ++n;
        };
        for (auto &Y : L) {
            for (Mat *m : {&Y.q_a_t, &Y.kv_t, &Y.q_b_t, &Y.out_b_t, &Y.out_b_full, &Y.q_a_full, &Y.kv_full, &Y.comp_kv_full, &Y.comp_gate_full, &Y.comp_kv_t, &Y.comp_gate_t, &Y.idx_k, &Y.idx_q_b_t, &Y.idx_q_b_full, &Y.idx_proj, &Y.engram_wkv_t}) cv(*m);
            for (auto &o : Y.out_a_t) cv(o.m);
            for (auto *v : {&Y.chunks, &Y.shared, &Y.shared_all})
                for (auto &k : *v) cv(k.gate), cv(k.up), cv(k.down);
        }
        cv(head);
        if (comm.rank == 0) std::cerr << "DS41 q5p matrices=" << n << "\n";
    }
    size_t owner_batch_future_bytes_ = 0;  // enabled-owner batch caches allocated after repacking
    void repack_experts() {
        if (strata_env_equals("STRATA_DS41_IQ2R", "0")) return;
        const double keep = std::getenv("STRATA_DS41_IQ2R_KEEP_GB") ? std::atof(std::getenv("STRATA_DS41_IQ2R_KEEP_GB")) : 3.0;
        const double t0 = now();
        for (const int ty : {int(DS41_T_IQ2_S), int(DS41_T_IQ2_XXS), int(DS41_T_IQ2_XS)}) {
            std::vector<Mat *> ms;
            size_t extra = 0;
            for (auto &Y : L)
                for (auto &k : Y.chunks)
                    for (Mat *m : {&k.gate, &k.up, &k.down})
                        if (m->type == ty && m->p) ms.push_back(m), extra += (ds41_row_bytes(DS41_T_IQ2R, m->cols) - m->rb) * size_t(m->rows);
            if (ms.empty()) continue;
            // a type whole or not at all, per rank (the bits are the same either way)
            if (double(mem_available()) - double(extra) - double(owner_batch_future_bytes_) < keep * 1e9) continue;
            std::atomic<int> bad{0};
            team_run([&] {
                const int t = omp_get_thread_num(), T = omp_get_num_threads();
                for (size_t q = size_t(t); q < ms.size(); q += size_t(T)) {
                    Mat &m = *ms[q];
                    const size_t rr = ds41_row_bytes(DS41_T_IQ2R, m.cols);
                    uint8_t *d = ram_buffer(rr * size_t(m.rows));
                    if (!d) { bad = 1; continue; }
                    for (int r = 0; r < m.rows; ++r)
                        if (ds41_repack_iq2r(m.type, m.p + m.rb * size_t(r), m.cols, d + rr * size_t(r))) bad = 1;
                    // STRATA_DS41_IQ2P=1: IQ2R's bits in the kernel's read order (one code load, predicate-loaded signs)
                    static const bool iq2p = strata_env_is_one("STRATA_DS41_IQ2P");
                    if (iq2p) ds41_iq2r_to_iq2p(d, size_t(m.rows) * size_t(m.cols / 256));
                    Mat old = m;
                    m.p = d, m.type = iq2p ? int(DS41_T_IQ2P) : int(DS41_T_IQ2R), m.rb = rr;
                    release(old);
                }
            });
            if (bad) throw std::runtime_error("ds41: IQ2R repack failed");
            iq2r_n += int(ms.size()), iq2r_gb += double(extra) / 1e9;
        }
        iq2r_s = now() - t0;
    }
    // the expert matrices' rows on the CMG of the thread that runs them (gate / up / down: each thread rows
    // [rows t / T, rows (t + 1) / T) of every chunk); layout only
    void rehome_experts() {
        if (!rehome_on()) return;
        const double t0 = now();
        for (auto &Y : L) {
            for (auto &k : Y.chunks) rehome(k.gate, k.gate.rows), rehome(k.up, k.up.rows), rehome(k.down, k.down.rows);
            for (auto &k : Y.shared_all) rehome(k.gate, k.gate.rows), rehome(k.up, k.up.rows), rehome(k.down, k.down.rows);
            for (auto &k : Y.shared) rehome(k.gate, k.gate.rows), rehome(k.up, k.up.rows), rehome(k.down, k.down.rows);
        }
        rehome_s += now() - t0;
    }
    const uint8_t *embd_mem = nullptr;
    double embd_load_s = 0;
    void load_embd_slice() {
        if (strata_env_equals("STRATA_DS41_EMBD_SLICE", "0")) return;
        const double t0 = now();
        const size_t bytes = embd.rb * size_t(vocab1 - vocab0);
        uint8_t *m = ram_buffer(bytes);
        if (!m) return;
        read_bytes(M.t("token_embd.weight"), embd.rb * size_t(vocab0), bytes, m);
        embd_mem = m;
        embd_load_s = now() - t0;
    }
    void load_shared_all() {
        if (comm.world == 1 || (strata_env_equals("STRATA_DS41_SHARED_BALANCE", "0"))) return;
        const double t0 = now();
        for (int l = 0; l < layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            Layer &Y = L[size_t(l)];
            const Mat SG = M.mat(p + "ffn_gate_shexp.weight", true), SU = M.mat(p + "ffn_up_shexp.weight", true), SD = M.mat(p + "ffn_down_shexp.weight", true);
            std::vector<uint8_t> g(SG.rb * size_t(SG.rows)), u(SU.rb * size_t(SU.rows)), d(SD.rb * size_t(SD.rows));
            read_many({{&M.t(p + "ffn_gate_shexp.weight"), 0, g.size(), g.data()}, {&M.t(p + "ffn_up_shexp.weight"), 0, u.size(), u.data()},
                       {&M.t(p + "ffn_down_shexp.weight"), 0, d.size(), d.data()}});
            const size_t sbb = SD.rb / size_t(nchunk);
            for (int ch = 0; ch < nchunk; ++ch) {
                Chunk k;
                k.gate = SG, k.up = SU, k.down = SD;
                k.gate.rows = k.up.rows = 256;
                uint8_t *pg = ram_buffer(SG.rb * 256), *pu = ram_buffer(SU.rb * 256), *pd = ram_buffer(sbb * c.hidden);
                std::memcpy(pg, g.data() + SG.rb * 256 * ch, SG.rb * 256);
                std::memcpy(pu, u.data() + SU.rb * 256 * ch, SU.rb * 256);
                for (int r = 0; r < c.hidden; ++r) std::memcpy(pd + sbb * r, d.data() + SD.rb * r + sbb * ch, sbb);
                k.gate.p = pg, k.up.p = pu, k.down.p = pd, k.down.rows = c.hidden, k.down.cols = 256, k.down.rb = sbb;
                Y.shared_all.push_back(k);
            }
        }
        shared_load_s = now() - t0;
    }
    double bal_routed = 0, bal_total = 0, bal_mean = 0;   // sums over layer steps of the busiest rank's chunks, and the mean
    long bal_n = 0;

    void engram_to_ram() {
        if (!engram_ram_on()) { engram_part_to_ram(); return; }
        const double t0 = now();
        for (int l = 0; l < layers; ++l) {
            Layer &Y = L[size_t(l)];
            if (Y.engram_fd < 0 || Y.engram_mem || !engram_ram_layer(l)) continue;
            const size_t bytes = Y.engram_table.rb * size_t(Y.engram_r1 - Y.engram_r0);
            const double tl = now();
            std::cerr << "DS41 rank " << comm.rank << " engram_to_ram start bytes " << bytes / 1e9 << " GB avail " << mem_available() / 1e9 << " GB\n" << std::flush;
            if (mem_available() < bytes + (size_t(2) << 30)) {
                if (comm.rank == 0) std::cerr << "DS41 Engram shard stays on /local (MemAvailable " << mem_available() / 1e9 << " GB)\n";
                continue;
            }
            uint8_t *m = static_cast<uint8_t *>(mmap(nullptr, (bytes + 4095) & ~size_t(4095), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            if (m == MAP_FAILED) continue;
            std::atomic<size_t> next{0};
            std::atomic<int> fail{0};
            static const int lthreads = std::getenv("STRATA_DS41_LOCAL_THREADS") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_LOCAL_THREADS"))) : 8;
            const size_t piece = size_t(std::getenv("STRATA_DS41_LOCAL_PIECE_MB") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_LOCAL_PIECE_MB"))) : 8) << 20;
            auto work = [&] {
                for (size_t o; (o = next.fetch_add(piece)) < bytes;)
                    for (size_t d = 0, len = std::min(piece, bytes - o); d < len;) {
                        const ssize_t r = ::pread(Y.engram_fd, m + o + d, len - d, off_t(o + d));
                        if (r <= 0) { fail = 1; return; }
                        d += size_t(r);
                    }
            };
            std::vector<std::thread> th;
            for (int t = 0; t < lthreads; ++t) th.emplace_back(work);
            for (auto &t : th) t.join();
            if (fail) { munmap(m, (bytes + 4095) & ~size_t(4095)); continue; }
            Y.engram_mem = m;
            engram_ram_b += bytes;
            std::cerr << "DS41 rank " << comm.rank << " engram_to_ram done " << now() - tl << " s avail " << mem_available() / 1e9 << " GB\n" << std::flush;
        }
        engram_ram_s = now() - t0;
        engram_part_to_ram();
    }
    // STRATA_DS41_ENGRAM_RAM_GB=g: when the first Engram layer's shard stays on /local (ENGRAM_RAM=0, or it does not fit),
    // its first g GB of rows in RAM (MemAvailable keeping 2 GB beyond them); the other rows are read as before. A row
    // has the same bytes either way
    void engram_part_to_ram() {
        const double g = std::getenv("STRATA_DS41_ENGRAM_RAM_GB") ? std::atof(std::getenv("STRATA_DS41_ENGRAM_RAM_GB")) : 0.0;
        if (g <= 0) return;
        for (int l = 0; l < layers; ++l) {
            Layer &Y = L[size_t(l)];
            if (c.engram_index(l) != 0 || Y.engram_fd < 0 || Y.engram_mem) continue;
            const size_t rb = Y.engram_table.rb;
            const int64_t nr = std::min<int64_t>(Y.engram_r1 - Y.engram_r0, int64_t(g * 1e9 / double(rb)));
            const size_t bytes = rb * size_t(nr);
            if (nr <= 0 || mem_available() < bytes + (size_t(2) << 30)) continue;
            uint8_t *m = ram_buffer(bytes);
            if (!m) continue;
            for (size_t o = 0; o < bytes;) {
                const ssize_t r = ::pread(Y.engram_fd, m + o, std::min(bytes - o, size_t(64) << 20), off_t(o));
                if (r <= 0) { munmap(m, (bytes + 4095) & ~size_t(4095)); m = nullptr; break; }
                o += size_t(r);
            }
            if (!m) continue;
            Y.engram_part = m, Y.engram_pr1 = Y.engram_r0 + nr, engram_ram_b += bytes;
            if (comm.rank == 0) std::cerr << "DS41 engram layer " << l << " first " << bytes / 1e9 << " GB in RAM\n";
        }
    }

    // STRATA_DS41_ROUTER_TP=1: each rank the router logits and scores of its experts only, into zeroed [probs | selection]
    // rows, one all-reduce (the other ranks add zeros: every value bitwise the replicated router's)
    // STRATA_DS41_HEAD_BAL=1 (batched passes of >= 12 tokens): when the heads do not divide over the ranks, each rank with
    // one head more (ext) shares that last head's tokens with hp ranks of one head less (helpers): the pass's tokens in
    // hp + 1 contiguous parts, part 0 the ext rank's, part k its k-th helper's. A helper computes the head's q (its copy of
    // the q_b rows), scores, softmax and weighted sums for its part with the core's per-element operations, and sends the
    // rows to the ext rank (an all-to-all, the bits as sent) before out_a: every head's output is the same, and out_a's
    // partial sums are unchanged
    static bool head_bal_env() { static const bool v = strata_env_is_one("STRATA_DS41_HEAD_BAL"); return v; }
    bool bal_on_ = false;
    int bal_hp_ = 0;
    std::vector<int> bal_role_, bal_home_, bal_part_;   // per rank: 0 / 1 ext / 2 helper; a helper's ext rank; its part
    std::vector<std::vector<int>> bal_helpers_;          // per ext rank: its helpers by part (1..hp)
    int bal_head(int ext) const { return c.heads * (ext + 1) / comm.world - 1; }   // the shared head (global index)
    void bal_plan() {
        const int W = comm.world, Hh = c.heads;
        if (!head_bal_env() || W < 2 || Hh % W == 0) return;
        std::vector<int> ext, lit;
        const int lo = Hh / W;
        for (int r = 0; r < W; ++r) (Hh * (r + 1) / W - Hh * r / W > lo ? ext : lit).push_back(r);
        if (ext.empty() || lit.size() % ext.size()) return;
        bal_hp_ = int(lit.size() / ext.size());
        bal_role_.assign(size_t(W), 0), bal_home_.assign(size_t(W), -1), bal_part_.assign(size_t(W), 0), bal_helpers_.assign(size_t(W), {});
        for (size_t e = 0; e < ext.size(); ++e) {
            bal_role_[size_t(ext[e])] = 1;
            for (int k = 0; k < bal_hp_; ++k) {
                const int h = lit[e * size_t(bal_hp_) + size_t(k)];
                bal_role_[size_t(h)] = 2, bal_home_[size_t(h)] = ext[e], bal_part_[size_t(h)] = k + 1;
                bal_helpers_[size_t(ext[e])].push_back(h);
            }
        }
        bal_on_ = true;
        if (comm.rank == 0) std::cerr << "DS41 head balance: " << ext.size() << " ranks share a head with " << bal_hp_ << " helpers each\n";
    }
    static bool router_tp_on() { return strata_env_is_one("STRATA_DS41_ROUTER_TP"); }
    void split_tp() {
        const int W = comm.world, R = comm.rank, Hh = c.heads, hd = c.head_dim, G = c.out_groups, hpg = Hh / G;
        if (strata_env_equals("STRATA_DS41_TP", "0")) return;
        tp = true;
        h0 = Hh * R / W, h1 = Hh * (R + 1) / W, y0 = c.hidden * R / W, y1 = c.hidden * (R + 1) / W;
        bal_plan();
        for (auto &Y : L) {
            if (hd % 256 || Y.out_a.cols % 256) throw std::runtime_error("ds41: TP needs 256-aligned heads");
            Y.q_b_t = copy_rows(Y.q_b, h0 * hd, (h1 - h0) * hd);
            if (bal_on_ && bal_role_[size_t(R)] == 2) Y.q_b_x = copy_rows(Y.q_b, bal_head(bal_home_[size_t(R)]) * hd, hd);
            const size_t bb = Y.out_a.rb / size_t(Y.out_a.cols / 256), hb = bb * size_t(hd / 256);   // bytes per block, per head
            for (int g = 0; g < G; ++g) {
                const int a = std::max(h0, g * hpg), b = std::min(h1, (g + 1) * hpg);
                if (a >= b) continue;
                Mat m = Y.out_a;
                m.cols = (b - a) * hd, m.rows = c.out_rank, m.rb = hb * size_t(b - a);
                uint8_t *d = static_cast<uint8_t *>(mmap(nullptr, (m.rb * m.rows + 4095) & ~size_t(4095), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
                for (int j = 0; j < c.out_rank; ++j)
                    std::memcpy(d + m.rb * j, Y.out_a.p + Y.out_a.rb * size_t(g * c.out_rank + j) + hb * size_t(a - g * hpg), m.rb);
                m.p = d;
                Y.out_a_t.push_back({g, a, b, m});
            }
            Y.out_b_t = copy_rows(Y.out_b, y0, y1 - y0);
            if (es_outb_env()) Y.out_b_full = copy_rows(Y.out_b, 0, Y.out_b.rows);
            if (router_tp_on()) Y.ge0 = c.experts * R / W, Y.gate_inp_t = copy_rows(Y.gate_inp, Y.ge0, c.experts * (R + 1) / W - Y.ge0);
            release(Y.q_b), release(Y.out_a), release(Y.out_b);
            // q_a and kv rows (one all-reduce for both), Engram's projection rows, the indexer's heads
            qa0 = c.q_rank * R / W, kv0 = c.head_dim * R / W;
            if (es_qk_env()) Y.q_a_full = copy_rows(Y.q_a, 0, Y.q_a.rows), Y.kv_full = copy_rows(Y.kv, 0, Y.kv.rows);
            Y.q_a_t = copy_rows(Y.q_a, qa0, c.q_rank * (R + 1) / W - qa0);
            Y.kv_t = copy_rows(Y.kv, kv0, c.head_dim * (R + 1) / W - kv0);
            release(Y.q_a), release(Y.kv);
            // the compressor's rows (KV sources): computed with q_a / kv and summed in their all-reduce
            ck0 = c.head_dim * R / W;
            if (Y.comp_kv.p && es_qk_env()) Y.comp_kv_full = copy_rows(Y.comp_kv, 0, Y.comp_kv.rows);
            if (Y.comp_gate.p && es_qk_env()) Y.comp_gate_full = copy_rows(Y.comp_gate, 0, Y.comp_gate.rows);
            if (Y.comp_kv.p) Y.comp_kv_t = copy_rows(Y.comp_kv, ck0, c.head_dim * (R + 1) / W - ck0), release(Y.comp_kv);
            if (Y.comp_gate.p) Y.comp_gate_t = copy_rows(Y.comp_gate, ck0, c.head_dim * (R + 1) / W - ck0), release(Y.comp_gate);
            if (Y.engram_wkv.p) {
                const int e0 = Y.engram_wkv.rows * R / W;
                Y.engram_wkv_t = copy_rows(Y.engram_wkv, e0, Y.engram_wkv.rows * (R + 1) / W - e0);
                release(Y.engram_wkv);
            }
            ih0 = c.index_heads * R / W, ih1 = c.index_heads * (R + 1) / W;
            if (Y.idx_q_b.p) {
                Y.idx_q_b_t = copy_rows(Y.idx_q_b, ih0 * c.index_dim, (ih1 - ih0) * c.index_dim);
                if (es_index_env()) {
                    Y.idx_q_b_full = Y.idx_q_b;
                    index_full_bytes_ += Y.idx_q_b.rb * size_t(Y.idx_q_b.rows);
                    Y.idx_q_b.p = nullptr;   // retain the original full allocation instead of copying it
                } else release(Y.idx_q_b);
            }
        }
    }

    // --engram-dir: rows [N r / W, N (r + 1) / W) of the layer's table in DIR/engram-L<l>-r<r>of<W>.bin (kept when its
    // size is right: /local lives as long as the allocation)
    void stage_engram(int l, Layer &Y) {
        const int W = comm.world, R = comm.rank;
        const int64_t N = Y.engram_table.rows;
        Y.engram_r0 = N * R / W, Y.engram_r1 = N * (R + 1) / W;
        const size_t rb = Y.engram_table.rb, bytes = rb * size_t(Y.engram_r1 - Y.engram_r0);
        std::filesystem::create_directories(engram_dir);
        const std::string f = engram_dir + "/engram-L" + std::to_string(l) + "-r" + std::to_string(R) + "of" + std::to_string(W) + ".bin";
        if (!std::filesystem::exists(f) || std::filesystem::file_size(f) != bytes) {
            const double t0 = now();
            const auto &tt = M.t("blk." + std::to_string(l) + ".engram_embd.weight");
            const int fd = ::open((f + ".partial").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) throw std::runtime_error("ds41: cannot write " + f);
            const size_t piece = size_t(256) << 20;
            std::vector<uint8_t> buf(piece);
            uint8_t *ram = engram_ram_layer(l) && mem_available() > bytes + (size_t(6) << 30) ? ram_buffer(bytes) : nullptr;
            for (size_t o = 0; o < bytes; o += piece) {
                const size_t n = std::min(piece, bytes - o);
                read_bytes(tt, rb * size_t(Y.engram_r0) + o, n, buf.data());
                if (ram) std::memcpy(ram + o, buf.data(), n);
                for (size_t d = 0; d < n;) {
                    const ssize_t w = ::pwrite(fd, buf.data() + d, n - d, off_t(o + d));
                    if (w <= 0) throw std::runtime_error("ds41: write failed: " + f);
                    d += size_t(w);
                }
            }
            ::close(fd);
            std::filesystem::rename(f + ".partial", f);
            stage_s += now() - t0;
            if (ram) Y.engram_mem = ram, engram_ram_b += bytes;   // filled on the way: no re-read of /local
        }
        Y.engram_fd = ::open(f.c_str(), O_RDONLY);
        if (Y.engram_fd < 0) throw std::runtime_error("ds41: cannot open " + f);
    }

    // hyper-connection coefficients (src/kernels/cuda/ds41.cu hc_coeff_kernel's serial order)
    void hc_coeff(const float *m, const float *scale, const float *base, float *co) const {
        const float eps = c.hc_eps;
        for (int t = 0; t < 4; ++t) co[t] = sigmoid(m[t] * scale[0] + base[t]) + eps;
        for (int t = 4; t < 8; ++t) co[t] = sigmoid(m[t] * scale[1] + base[t]) * 2.0f;
        float k[16];
        for (int t = 0; t < 16; ++t) k[t] = m[8 + t] * scale[2] + base[8 + t];
        float v[16];
        for (int sc = 0; sc < 4; ++sc) {   // comb[dst + 4 src]: softmax over dst for each src (each exp once)
            float mx = -INFINITY;
            for (int i = 0; i < 4; ++i) mx = std::max(mx, k[i + 4 * sc]);
            float e[4], sum = 0;
            for (int i = 0; i < 4; ++i) { e[i] = std::exp(k[i + 4 * sc] - mx); sum += e[i]; }
            for (int d = 0; d < 4; ++d) v[d + 4 * sc] = e[d] * (1.0f / sum) + eps;
        }
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
        // Sinkhorn on the 16 entries in one vector: the column / row sums in the scalar order (0 + k[0] + ... + k[3]),
        // then a lane-wise IEEE division (the scalar loop's bits, without 640 serial divisions)
        static const uint32_t ci[4][16] = {{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}, {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7},
                                           {8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11}, {12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15}};
        static const uint32_t ri[4][16] = {{0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12}, {1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13},
                                           {2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14}, {3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15}};
        const svbool_t pt = svptrue_b32();
        const svuint32_t C0 = svld1_u32(pt, ci[0]), C1 = svld1_u32(pt, ci[1]), C2 = svld1_u32(pt, ci[2]), C3 = svld1_u32(pt, ci[3]);
        const svuint32_t R0 = svld1_u32(pt, ri[0]), R1 = svld1_u32(pt, ri[1]), R2 = svld1_u32(pt, ri[2]), R3 = svld1_u32(pt, ri[3]);
        const svfloat32_t Z = svdup_f32(0), Ev = svdup_f32(eps);
        svfloat32_t K = svld1_f32(pt, v);
        auto norm = [&](svuint32_t I0, svuint32_t I1, svuint32_t I2, svuint32_t I3) {
            svfloat32_t sm = svadd_f32_x(pt, Z, svtbl_f32(K, I0));
            sm = svadd_f32_x(pt, sm, svtbl_f32(K, I1));
            sm = svadd_f32_x(pt, sm, svtbl_f32(K, I2));
            sm = svadd_f32_x(pt, sm, svtbl_f32(K, I3));
            K = svdiv_f32_x(pt, K, svadd_f32_x(pt, sm, Ev));
        };
        norm(C0, C1, C2, C3);
        for (int i = 1; i < c.sinkhorn; ++i) { norm(R0, R1, R2, R3); norm(C0, C1, C2, C3); }
        svst1_f32(pt, co + 8, K);
#else
        std::memcpy(k, v, sizeof k);
        auto cols = [&] {
            for (int t = 0; t < 16; ++t) {
                const int d = t % 4;
                float sum = 0;
                for (int i = 0; i < 4; ++i) sum += k[d + 4 * i];
                v[t] = k[t] / (sum + eps);
            }
            std::memcpy(k, v, sizeof k);
        };
        auto rows = [&] {
            for (int t = 0; t < 16; ++t) {
                const int sc = t / 4;
                float sum = 0;
                for (int i = 0; i < 4; ++i) sum += k[i + 4 * sc];
                v[t] = k[t] / (sum + eps);
            }
            std::memcpy(k, v, sizeof k);
        };
        cols();
        for (int i = 1; i < c.sinkhorn; ++i) { rows(); cols(); }
        for (int t = 0; t < 16; ++t) co[8 + t] = k[t];
#endif
    }
    // ---- the forward over a pass: T tokens (Ent), each of a sequence (Seq) at its position. One sequence's
    // consecutive positions (a speculative verify, a prompt) or several sequences' next tokens (batched decode). Every
    // projection runs once per layer for all T tokens (gemv_multi nt = T) and each all-reduce carries all T vectors;
    // each token's operations are those of a one-token pass, so its results are too.
    void mixes(Ent &e, const Mat &fn, const Vec &scale, const Vec &base, float *co) {
        Vec flat(static_cast<size_t>(4) * c.hidden), m(24);
        rms_norm(e.h.data(), nullptr, flat.data(), 4 * c.hidden, c.rms_eps);
        if (fn.type == DS41_T_F32 && fn.rows == 24 && fn.cols % 32 == 0) {   // 24 rows x 2 column halves: 48 items, halves added in order
            const int half = fn.cols / 2;
            float part[48];
            const float *W = reinterpret_cast<const float *>(fn.p);
            team_for(48, [&](int a, int b) {
                for (int i = a; i < b; ++i) part[i] = dot(W + size_t(i / 2) * fn.cols + size_t(i % 2) * half, flat.data() + size_t(i % 2) * half, half);
            });
            for (int r = 0; r < 24; ++r) m[size_t(r)] = part[2 * r] + part[2 * r + 1];
        } else
            gemv(fn, flat.data(), m.data(), "hc");
        hc_coeff(m.data(), scale.data(), base.data(), co);
    }
    void hc_pre(const Ent &e, const float *co, float *out) const {   // out = sum_s pre[s] h[s], over the team in vectors
        const int n = c.hidden;
        const float *h0 = e.h.data(), *h1 = h0 + n, *h2p = h0 + 2 * size_t(n), *h3 = h0 + 3 * size_t(n);
        const float c0 = co[0], c1 = co[1], c2 = co[2], c3 = co[3];
        team_for((n + 63) / 64, [&](int a, int b) {
            const int j1 = std::min(n, 64 * b);
#pragma omp simd
            for (int j = 64 * a; j < j1; ++j) out[j] = ((h0[j] * c0 + h1[j] * c1) + h2p[j] * c2) + h3[j] * c3;
        });
    }
    void hc_post(Ent &e, const float *xo, const float *co) {   // h = post * xo + comb . h
        const int n = c.hidden;
        const float *s0 = e.h.data(), *s1 = s0 + n, *s2 = s0 + 2 * size_t(n), *s3 = s0 + 3 * size_t(n);
        float *o = e.h2.data();
        float cm[4][5];
        for (int d = 0; d < 4; ++d) {
            cm[d][0] = co[4 + d];
            for (int s = 0; s < 4; ++s) cm[d][1 + s] = co[8 + d + 4 * s];
        }
        team_for((n + 255) / 256, [&](int ba, int bb) {
          for (int j0 = 256 * ba; j0 < 256 * bb; j0 += 256)
            for (int d = 0; d < 4; ++d) {
                const float a = cm[d][0], b0 = cm[d][1], b1 = cm[d][2], b2 = cm[d][3], b3 = cm[d][4];
                float *od = o + size_t(d) * n;
                const int j1 = std::min(n, j0 + 256);
#pragma omp simd
                for (int j = j0; j < j1; ++j) od[j] = (((xo[j] * a + s0[j] * b0) + s1[j] * b1) + s2[j] * b2) + s3[j] * b3;
            }
        });
        std::swap(e.h, e.h2);
    }
    // one mHC site in one team region, the same operations as hc_post (when post_y), mixes, hc_pre (pre_co; null: the
    // first stream) and rms_norm in turn: [post -> h2] | h's sum-of-squares blocks + hc_pre -> x | flat = h r and x's
    // blocks | the 48 hc_fn dots and xn = x r w | thread 0: hc_coeff -> co. STRATA_DS41_FUSED_SITE=0: the separate calls
    Vec flat_;
    void site(Ent &e, const float *post_y, const float *post_co, const Mat &fn, const Vec &scale, const Vec &base, float *co, const float *pre_co,
              const Vec &norm_w) {
        static const bool fused = !(strata_env_equals("STRATA_DS41_FUSED_SITE", "0"));
        const int n = c.hidden, n4 = 4 * n, nb4 = (n4 + 1023) / 1024, nbx = (n + 1023) / 1024;
        if (!fused || fn.type != DS41_T_F32 || fn.rows != 24 || fn.cols != n4 || n % 1024 || nb4 > 64 || n >= 8192) {
            if (post_y) hc_post(e, post_y, post_co);
            mixes(e, fn, scale, base, co);
            if (pre_co) hc_pre(e, pre_co, e.x.data());
            else std::copy(e.h.begin(), e.h.begin() + n, e.x.begin());
            rms_norm(e.x.data(), norm_w.data(), e.xn.data(), n, c.rms_eps);
            return;
        }
        double ph[64], px[64];
        float part[48];
        float cm[4][5];
        if (post_y)
            for (int d = 0; d < 4; ++d) {
                cm[d][0] = post_co[4 + d];
                for (int s2 = 0; s2 < 4; ++s2) cm[d][1 + s2] = post_co[8 + d + 4 * s2];
            }
        const float *W = reinterpret_cast<const float *>(fn.p);
        const int half = fn.cols / 2;
        float *X = e.x.data(), *XN = e.xn.data();
        const float eps = c.rms_eps;
        double mk[6] = {now(), 0, 0, 0, 0, 0};
        team_run([&] {
            const int t = omp_get_thread_num(), T = omp_get_num_threads();
            auto share = [&](int items, int &a, int &b) { a = int(int64_t(items) * t / T), b = int(int64_t(items) * (t + 1) / T); };
            int a, b;
            // a thread per 1024 columns of the 4 streams: hc_post's expression (into h2), the 4 sum-of-squares blocks
            // (d, k) of sumsq_blocks' order, hc_pre's expression (or the first stream) and x's block k: no barrier
            // between them (each needs only its own columns)
            const float *hp = post_y ? e.h2.data() : e.h.data();
            if (t == 0) mk[1] = now();
            if (merge_bits() & 4) {   // the expressions by 64 columns over the team, a barrier, then the 5 nbx dots by item
                share(n / 64, a, b);
                for (int ch = a; ch < b; ++ch) {
                    const int j0 = 64 * ch, j1 = j0 + 64;
                    if (post_y) {
                        const float *s0 = e.h.data(), *s1 = s0 + n, *s2 = s0 + 2 * size_t(n), *s3 = s0 + 3 * size_t(n);
                        float *o = e.h2.data();
                        const float *xo = post_y;
                        for (int d = 0; d < 4; ++d) {
                            const float a0 = cm[d][0], b0 = cm[d][1], b1 = cm[d][2], b2 = cm[d][3], b3 = cm[d][4];
                            float *od = o + size_t(d) * n;
#pragma omp simd
                            for (int j = j0; j < j1; ++j) od[j] = (((xo[j] * a0 + s0[j] * b0) + s1[j] * b1) + s2[j] * b2) + s3[j] * b3;
                        }
                    }
                    if (pre_co) {
                        const float *h0 = hp, *h1 = h0 + n, *h2p = h0 + 2 * size_t(n), *h3 = h0 + 3 * size_t(n);
                        const float c0 = pre_co[0], c1 = pre_co[1], c2 = pre_co[2], c3 = pre_co[3];
#pragma omp simd
                        for (int j = j0; j < j1; ++j) X[j] = ((h0[j] * c0 + h1[j] * c1) + h2p[j] * c2) + h3[j] * c3;
                    } else
                        for (int j = j0; j < j1; ++j) X[j] = hp[j];
                }
                team_barrier();
                for (int it = t; it < 5 * nbx; it += T) {   // (d, k) of the streams, then x's block k
                    const int d = it / nbx, k = it % nbx, j0 = 1024 * k, j1 = std::min(n, j0 + 1024);
                    if (d < 4) ph[d * nbx + k] = dot(hp + size_t(d) * n + j0, hp + size_t(d) * n + j0, j1 - j0);
                    else px[k] = dot(X + j0, X + j0, j1 - j0);
                }
            } else
            for (int k = t; k < nbx; k += T) {
                const int j0 = 1024 * k, j1 = std::min(n, j0 + 1024);
                if (post_y) {
                    const float *s0 = e.h.data(), *s1 = s0 + n, *s2 = s0 + 2 * size_t(n), *s3 = s0 + 3 * size_t(n);
                    float *o = e.h2.data();
                    const float *xo = post_y;
                    for (int d = 0; d < 4; ++d) {
                        const float a0 = cm[d][0], b0 = cm[d][1], b1 = cm[d][2], b2 = cm[d][3], b3 = cm[d][4];
                        float *od = o + size_t(d) * n;
#pragma omp simd
                        for (int j = j0; j < j1; ++j) od[j] = (((xo[j] * a0 + s0[j] * b0) + s1[j] * b1) + s2[j] * b2) + s3[j] * b3;
                    }
                }
                for (int d = 0; d < 4; ++d) ph[d * nbx + k] = dot(hp + size_t(d) * n + j0, hp + size_t(d) * n + j0, j1 - j0);
                if (pre_co) {
                    const float *h0 = hp, *h1 = h0 + n, *h2p = h0 + 2 * size_t(n), *h3 = h0 + 3 * size_t(n);
                    const float c0 = pre_co[0], c1 = pre_co[1], c2 = pre_co[2], c3 = pre_co[3];
#pragma omp simd
                    for (int j = j0; j < j1; ++j) X[j] = ((h0[j] * c0 + h1[j] * c1) + h2p[j] * c2) + h3[j] * c3;
                } else
                    for (int j = j0; j < j1; ++j) X[j] = hp[j];
                px[k] = dot(X + j0, X + j0, j1 - j0);
            }
            team_barrier();
            if (t == 0) mk[2] = now();
            {   // the 48 hc_fn dots against h r (formed in registers: the flat vector's bits) and x's blocks
                double sum = 0;
                for (int k = 0; k < nb4; ++k) sum += ph[k];
                const float r = 1.0f / std::sqrt(float(sum / n4) + eps);
                share(48, a, b);
                for (int i = a; i < b; ++i) part[i] = dot_scaled(W + size_t(i / 2) * fn.cols + size_t(i % 2) * half, hp + size_t(i % 2) * half, r, half);
            }
            team_barrier();
            if (t == 0) mk[3] = now();
            {
                double sum = 0;
                for (int k = 0; k < nbx; ++k) sum += px[k];
                const float r = 1.0f / std::sqrt(float(sum / n) + eps);
                const float *w = norm_w.data();
                share(n / 64, a, b);
#pragma omp simd
                for (int i = 64 * a; i < 64 * b; ++i) XN[i] = X[i] * r * w[i];
            }
            if (t == 0) {
                mk[4] = now();
                float m[24];
                for (int r = 0; r < 24; ++r) m[r] = part[2 * r] + part[2 * r + 1];
                hc_coeff(m, scale.data(), base.data(), co);
                mk[5] = now();
            }
        });
        {
            const double te = now();
            tm("site.post") += (mk[1] ? mk[1] : mk[0]) - mk[0], tm("site.pre") += mk[2] - (mk[1] ? mk[1] : mk[0]);
            tm("site.dots") += mk[3] - mk[2], tm("site.xn") += mk[4] - mk[3], tm("site.coeff") += mk[5] - mk[4], tm("site.join") += te - mk[5];
        }
        if (post_y) std::swap(e.h, e.h2);
    }

    // every token's site of one kind in one team region (the fused site's operations per token): kind 0 the attention
    // site (post: the previous layer's FFN output; pre: coeff_ffn, or the first stream at layer 0), 1 the FFN site
    // (post: the attention output with coeff_attn; pre: coeff_attn). The 48 hc_fn half rows are read once for all tokens.
    // (E: T entries; the pass's own entries under STRATA_DS41_ENTRY_SHARD)
    void site_all(Ent *E, int T, bool post, int kind, bool first, const Mat &fn, const Vec &scale, const Vec &base, const Vec &norm_w) {
        const int n = c.hidden, n4 = 4 * n, nb4 = (n4 + 1023) / 1024, nbx = (n + 1023) / 1024;
        if (T < 1) return;
        static const bool fused = !(strata_env_equals("STRATA_DS41_FUSED_SITE", "0"));
        auto post_co = [&](Ent &e) { return kind == 0 ? e.coeff_ffn.data() : e.coeff_attn.data(); };
        auto out_co = [&](Ent &e) { return kind == 0 ? e.coeff_attn.data() : e.coeff_ffn.data(); };
        auto pre_co = [&](Ent &e) -> const float * { return kind == 0 ? (first ? nullptr : e.coeff_ffn.data()) : e.coeff_attn.data(); };
        if (T == 1 || !fused || fn.type != DS41_T_F32 || fn.rows != 24 || fn.cols != n4 || n % 1024 || nb4 > 64 || n >= 8192) {
            for (int i = 0; i < T; ++i) {
                Ent &e = E[i];
                site(e, post ? e.y.data() : nullptr, post_co(e), fn, scale, base, out_co(e), pre_co(e), norm_w);
            }
            return;
        }
        std::vector<double> ph(size_t(T) * nb4), px(size_t(T) * nbx);
        std::vector<float> part(size_t(T) * 48), cm(size_t(T) * 20);
        if (post)
            for (int i = 0; i < T; ++i)
                for (int d = 0; d < 4; ++d) {
                    const float *pc = post_co(E[size_t(i)]);
                    cm[size_t(i) * 20 + size_t(d) * 5] = pc[4 + d];
                    for (int s2 = 0; s2 < 4; ++s2) cm[size_t(i) * 20 + size_t(d) * 5 + 1 + size_t(s2)] = pc[8 + d + 4 * s2];
                }
        const float *W = reinterpret_cast<const float *>(fn.p);
        const int half = fn.cols / 2;
        const float eps = c.rms_eps;
        const double t0 = now();
        team_run([&] {
            const int t = omp_get_thread_num(), TT = omp_get_num_threads();
            for (int q = t; q < T * nbx; q += TT) {   // (token, 1024 columns): post, the stream blocks, pre, x's block
                const int i = q / nbx, k = q % nbx, j0 = 1024 * k, j1 = std::min(n, j0 + 1024);
                Ent &e = E[size_t(i)];
                const float *hp = post ? e.h2.data() : e.h.data();
                if (post) {
                    const float *s0 = e.h.data(), *s1 = s0 + n, *s2 = s0 + 2 * size_t(n), *s3 = s0 + 3 * size_t(n);
                    float *o = e.h2.data();
                    const float *xo = e.y.data();
                    for (int d = 0; d < 4; ++d) {
                        const float *cc = &cm[size_t(i) * 20 + size_t(d) * 5];
                        const float a0 = cc[0], b0 = cc[1], b1 = cc[2], b2 = cc[3], b3 = cc[4];
                        float *od = o + size_t(d) * n;
#pragma omp simd
                        for (int j = j0; j < j1; ++j) od[j] = (((xo[j] * a0 + s0[j] * b0) + s1[j] * b1) + s2[j] * b2) + s3[j] * b3;
                    }
                }
                for (int d = 0; d < 4; ++d) ph[size_t(i) * nb4 + size_t(d * nbx + k)] = dot(hp + size_t(d) * n + j0, hp + size_t(d) * n + j0, j1 - j0);
                float *X = e.x.data();
                if (const float *pc = pre_co(e)) {
                    const float *h0 = hp, *h1 = h0 + n, *h2p = h0 + 2 * size_t(n), *h3 = h0 + 3 * size_t(n);
                    const float c0 = pc[0], c1 = pc[1], c2 = pc[2], c3 = pc[3];
#pragma omp simd
                    for (int j = j0; j < j1; ++j) X[j] = ((h0[j] * c0 + h1[j] * c1) + h2p[j] * c2) + h3[j] * c3;
                } else
                    for (int j = j0; j < j1; ++j) X[j] = hp[j];
                px[size_t(i) * nbx + size_t(k)] = dot(X + j0, X + j0, j1 - j0);
            }
            team_barrier();
            for (int it = int(int64_t(48) * t / TT); it < int(int64_t(48) * (t + 1) / TT); ++it)   // a half row, every token
                for (int i = 0; i < T; ++i) {
                    Ent &e = E[size_t(i)];
                    const float *hp = post ? e.h2.data() : e.h.data();
                    double sum = 0;
                    for (int k = 0; k < nb4; ++k) sum += ph[size_t(i) * nb4 + size_t(k)];
                    const float r = 1.0f / std::sqrt(float(sum / n4) + eps);
                    part[size_t(i) * 48 + size_t(it)] = dot_scaled(W + size_t(it / 2) * fn.cols + size_t(it % 2) * half, hp + size_t(it % 2) * half, r, half);
                }
            team_barrier();
            const int nch = n / 64;
            for (int q = int(int64_t(T) * nch * t / TT); q < int(int64_t(T) * nch * (t + 1) / TT); ++q) {   // xn by (token, 64)
                const int i = q / nch, ch = q % nch;
                Ent &e = E[size_t(i)];
                double sum = 0;
                for (int k = 0; k < nbx; ++k) sum += px[size_t(i) * nbx + size_t(k)];
                const float r = 1.0f / std::sqrt(float(sum / n) + eps);
                const float *w = norm_w.data(), *X = e.x.data();
                float *XN = e.xn.data();
#pragma omp simd
                for (int j = 64 * ch; j < 64 * ch + 64; ++j) XN[j] = X[j] * r * w[j];
            }
            for (int i = t; i < T; i += TT) {   // each token's coefficients
                float m[24];
                for (int r = 0; r < 24; ++r) m[r] = part[size_t(i) * 48 + size_t(2 * r)] + part[size_t(i) * 48 + size_t(2 * r + 1)];
                hc_coeff(m, scale.data(), base.data(), out_co(E[size_t(i)]));
            }
        });
        tm("site.all") += now() - t0;
        if (post)
            for (int i = 0; i < T; ++i) std::swap(E[i].h, E[i].h2);
    }

    void allreduce_timed(float *v, int n, const char *site) {
        if (n <= 0) return;
        const double t0 = now();
        if (fib_cur_ >= 0) cdo([this, v, n] { comm.allreduce(v, n); });
        else comm.allreduce(v, n);
        const double dt = now() - t0;
        tm("comm") += dt, tm(site) += dt;
        if (g_skew_on) skew_log(site, dt);
    }
    // STRATA_DS41_AG (bits, default 0): an all-gather instead of the all-reduce where each rank's rows are its slice
    // and the rest zero (1 the attention's q_a + kv (+ compressor) rows, 2 out_b's rows); the same bits (Comm::
    // allgather_shards). Needs the reduce-scatter / all-gather region (Comm::xc_init)
    static int ag_bits() { static const int v = std::getenv("STRATA_DS41_AG") ? std::atoi(std::getenv("STRATA_DS41_AG")) : 0; return v; }
    // V: T rows of ldv floats; section k covers [sec[k][0], sec[k][0] + sec[k][1]) of a row, rank r's piece of it
    // [D r / W, D (r + 1) / W) (D = sec[k][1]); every rank's pieces gathered into V
    Vec agbuf_, a2a_s_, a2a_r_, oa_own_, yb_own_;
    void gather_slices(float *V, int T, size_t ldv, const int (*sec)[2], int nsec, const char *site) {
        const double t0 = now();
        const int W = comm.world, R = comm.rank;
        int off[65];
        off[0] = 0;
        for (int r = 0; r < W; ++r) {
            int L = 0;
            for (int k = 0; k < nsec; ++k) L += int(int64_t(sec[k][1]) * (r + 1) / W - int64_t(sec[k][1]) * r / W);
            off[r + 1] = off[r] + T * L;
        }
        agbuf_.resize(size_t(off[W]));
        auto walk = [&](int r, bool pack) {
            float *b = agbuf_.data() + off[r];
            for (int i = 0; i < T; ++i)
                for (int k = 0; k < nsec; ++k) {
                    const int a = int(int64_t(sec[k][1]) * r / W), e = int(int64_t(sec[k][1]) * (r + 1) / W);
                    float *v = V + size_t(i) * ldv + sec[k][0] + a;
                    if (pack) std::copy(v, v + (e - a), b);
                    else std::copy(b, b + (e - a), v);
                    b += e - a;
                }
        };
        walk(R, true);
        float *const gb = agbuf_.data();
        cdo([&, gb] { comm.allgather_shards(gb, off); });
        for (int r = 0; r < W; ++r) walk(r, false);
        const double dt = now() - t0;
        tm("comm") += dt, tm(site) += dt;
        if (g_skew_on) skew_log(site, dt);
    }
    // STRATA_DS41_ENTRY_SHARD=1 (opt-in; batched passes of >= world entries, greedy, no drafter features, no ROUTER_TP):
    // rank r owns the pass's entries [T r / W, T (r + 1) / W) and alone keeps their streams: the mHC sites, the router
    // and the expert picks, the Engram gate and the final norm run for its own entries only. Every rank still needs
    // every entry's xn (the row-split projections, the expert chunks, the head): gathered after each site (the bits as
    // computed). The attention's out_b rows go to the owners (an all-to-all; each value + 0.0f, as the all-reduce of
    // zeros and the one rank's value), the FFN output is reduce-scattered (the all-reduce's rank-order sums). Per
    // entry the same operations as without it
    // STRATA_DS41_ES_INDEX=1: retain all index-query heads and gather selected rows from entry owners.
    static bool es_index_env() {
        static const bool on = entry_shard_env() && strata_env_truthy_one("STRATA_DS41_ES_INDEX");
        return on;
    }
    uint64_t index_owned_calls_ = 0, index_verify_calls_ = 0, index_verified_values_ = 0;
    uint64_t index_owner_qk_calls_ = 0, index_verify_x_gathers_ = 0;
    size_t index_full_bytes_ = 0;
    // STRATA_DS41_ES_OUTB=1 (with ENTRY_SHARD; +out_b's size in RAM per rank): out_a's partial sums reduce-scattered to the
    // owners (the all-reduce's rank-order sums), each owner its entries' out_b rows in full (the same row dots), so no
    // out_b exchange; each value + 0.0f (the all-reduce of zeros and the one rank's value)
    static bool es_outb_env() { static const bool v = entry_shard_env() && strata_env_is_one("STRATA_DS41_ES_OUTB"); return v; }
    // STRATA_DS41_ES_QK=1 (with ENTRY_SHARD; + q_a / kv / compressor in RAM): on layers without the indexer each owner its
    // entries' q_a / kv / compressor rows in full (the same row dots), gathered with each value + 0.0f (the all-reduce of
    // the row slices' bits); the xn gather before them goes (the indexer's layers keep it: the indexer reads xn)
    static bool es_qk_env() { static const bool v = entry_shard_env() && strata_env_is_one("STRATA_DS41_ES_QK"); return v; }
    bool index_owner_qk(int l) const {
        static const bool enabled = strata_env_truthy_one("STRATA_DS41_ES_INDEX_QK");
        return enabled && L[size_t(l)].idx_q_b_full.p;
    }
    bool es_qk_layer(int l) const { return es_on_ && es_qk_env() && L[size_t(l)].q_a_full.p &&
        (!(c.ratio[size_t(l)] && c.topk_src[size_t(l)] == l) || index_owner_qk(l)); }
    // STRATA_DS41_ES_XQ=1 (with ENTRY_SHARD): on layers whose expert chunks all read xn as Q8_K (gate / up), the FFN's
    // gather carries each owner's xn as its Q8_K blocks (292 bytes per 256 values, the bits as sent) instead of floats:
    // every rank's expert kernels read the blocks ds41_quant_q8k makes of the same xn bits on the owner
    static bool es_xq_env() { static const bool v = entry_shard_env() && strata_env_is_one("STRATA_DS41_ES_XQ"); return v; }
    std::vector<int8_t> xq_layer_;   // per layer: 1 every chunk's gate / up read Q8_K, 0 not, -1 not checked
    bool es_xq_layer(int l) {
        if (!es_on_ || !es_xq_env() || c.hidden % 256) return false;
        if (xq_layer_.empty()) xq_layer_.assign(size_t(layers), -1);
        int8_t &v = xq_layer_[size_t(l)];
        if (v < 0) {
            const Layer &Y = L[size_t(l)];
            bool ok = true;
            for (const auto *cs : {&Y.chunks, &Y.shared, &Y.shared_all})
                for (const Chunk &k : *cs) ok = ok && ds41_act_kind(k.gate.type) == DS41_ACT_Q8K && ds41_act_kind(k.up.type) == DS41_ACT_Q8K;
            v = ok ? 1 : 0;
        }
        return v == 1;
    }
    static bool entry_shard_env() { static const bool v = strata_env_is_one("STRATA_DS41_ENTRY_SHARD"); return v; }
    bool es_on_ = false;
    int es_a_ = 0, es_b_ = 0, es_off_[65] = {0};
    // V: T rows of ldv floats, this rank's columns [D r / W, D (r + 1) / W) of each (r = this rank; the rest unused):
    // every owner gets its entries' full D columns from every rank into dst(i) (each value + 0.0f: the bits of the
    // all-reduce of the zero-padded slices)
    template <class F> void slices_to_owners(const float *V, size_t ldv, int D, int T, F &&dst, const char *site) {
        const double t0 = now();
        const int W = comm.world, R = comm.rank, own = es_b_ - es_a_;
        const int c0 = int(int64_t(D) * R / W), c1 = int(int64_t(D) * (R + 1) / W);
        int soff[65], roff[65];
        soff[0] = roff[0] = 0;
        for (int r = 0; r < W; ++r) {
            soff[r + 1] = soff[r] + (es_off_[r + 1] - es_off_[r]) * (c1 - c0);
            roff[r + 1] = roff[r] + own * int(int64_t(D) * (r + 1) / W - int64_t(D) * r / W);
        }
        a2a_s_.resize(size_t(soff[W])), a2a_r_.resize(size_t(roff[W]));
        for (int i = 0; i < T; ++i) std::copy(V + size_t(i) * ldv + c0, V + size_t(i) * ldv + c1, a2a_s_.begin() + size_t(i) * (c1 - c0));
        const float *const sp = a2a_s_.data();
        float *const rp = a2a_r_.data();
        cdo([&, sp, rp] { comm.alltoall(sp, soff, rp, roff); });
        for (int r = 0; r < W; ++r) {
            const int a = int(int64_t(D) * r / W), e = int(int64_t(D) * (r + 1) / W);
            const float *src = a2a_r_.data() + roff[r];
            for (int i = es_a_; i < es_b_; ++i, src += e - a) {
                float *d = dst(i);
                for (int j = a; j < e; ++j) d[j] = src[j - a] + 0.0f;
            }
        }
        const double dt = now() - t0;
        tm("comm") += dt, tm(site) += dt;
    }
    void share_xn(std::vector<Ent> &E, const char *site) {   // every owner's xn rows to every rank
        const double t0 = now();
        const int n = c.hidden, T = int(E.size()), W = comm.world;
        pX_.resize(size_t(T) * n);
        for (int i = es_a_; i < es_b_; ++i) std::copy(E[size_t(i)].xn.begin(), E[size_t(i)].xn.end(), pX_.begin() + size_t(i) * n);
        int off[65];
        for (int r = 0; r <= W; ++r) off[r] = es_off_[r] * n;
        float *const xp = pX_.data();
        cdo([&, xp] { comm.allgather_shards(xp, off, false); });
        for (int i = 0; i < T; ++i)
            if (i < es_a_ || i >= es_b_) std::copy(pX_.begin() + size_t(i) * n, pX_.begin() + size_t(i + 1) * n, E[size_t(i)].xn.begin());
        const double dt = now() - t0;
        tm("comm") += dt, tm(site) += dt;
    }
    // the pass's xn rows side by side (T x hidden): the projections read them as nt tokens
    Vec pX_;
    // STRATA_DS41_ES_XNOWN=1 (ENTRY_SHARD): only the own entries' rows (the others' are read only after a gather writes
    // them: share_xn, the FFN's gather, the projections' own-row jobs)
    const float *gather_xn(std::vector<Ent> &E) {
        static const bool own_only = strata_env_is_one("STRATA_DS41_ES_XNOWN");
        const int n = c.hidden;
        pX_.resize(E.size() * size_t(n));
        const size_t a = own_only && es_on_ ? size_t(es_a_) : 0, b = own_only && es_on_ ? size_t(es_b_) : E.size();
        for (size_t i = a; i < b; ++i) std::copy(E[i].xn.begin(), E[i].xn.end(), pX_.begin() + i * size_t(n));
        return pX_.data();
    }

    // the Engram rows of one token at layer l (ds41_decode.cpp engram_rows_for): cols rows
    void engram_rows(int l, const Ent &en, int64_t *rows) const {
        const int e = c.engram_index(l), cols = c.engram_cols(), ng = c.ngram;
        {
            const std::vector<int> &hist = en.s->hist;
            const int pos = en.pos;
            const uint64_t *mult = c.multipliers.data() + size_t(e) * ng;
            const uint64_t *primes = c.primes.data() + size_t(e) * cols, *offs = c.offsets.data() + size_t(e) * cols;
            const int64_t pad = c.engram_pad_mapped ? c.engram_pad : c.token_map[size_t(c.engram_pad)];
            auto cid = [&](int64_t t) -> int64_t { return t < 0 || t >= int64_t(c.token_map.size()) ? -1 : c.token_map[size_t(t)]; };
            std::vector<uint64_t> toks(static_cast<size_t>(ng));
            int64_t id = cid(hist[size_t(pos)]);
            bool blocked = id == -1;
            toks[0] = uint64_t(blocked ? pad : id);
            for (int s = 1; s < ng; ++s) {
                id = pos - s < 0 ? -1 : cid(hist[size_t(pos - s)]);
                blocked = blocked || id == -1;
                toks[size_t(s)] = uint64_t(blocked ? pad : id);
            }
            uint64_t rolling = toks[0] * mult[0];
            for (int o = 1; o < ng; ++o) {
                rolling ^= toks[size_t(o)] * mult[o];
                for (int hh = 0; hh < c.engram_heads; ++hh) {
                    const int col = (o - 1) * c.engram_heads + hh;
                    rows[col] = int64_t(rolling % primes[col] + offs[col]);
                }
            }
        }
    }
    // Opt-in prompt-only cache of original table bytes. Never mutates seq0 early.
    struct PromptEgCache { std::vector<int64_t> rows; std::vector<uint8_t> raw; std::vector<size_t> jobs; bool active = false; };
    std::vector<PromptEgCache> prompt_eg_cache_;
    struct PromptRowJob { int pos, layer; size_t row; };
    std::vector<PromptRowJob> prompt_row_jobs_;
    std::unique_ptr<strata::cpu_runner::RowPrefetch> prompt_row_io_;
    std::atomic<uint64_t> prompt_row_waits_{0};
    double prompt_row_start_ = 0;

    std::atomic<uint64_t> prompt_eg_hits_{0}, prompt_eg_misses_{0}, prompt_eg_checks_{0};
    size_t prompt_eg_rows_ = 0, prompt_eg_bytes_ = 0, prompt_eg_layers_ = 0;
    double prompt_eg_load_s_ = 0;
    bool prompt_eg_enabled_ = false, prompt_eg_verify_ = false;
    void prepare_prompt_eg_cache(const std::vector<int> &ids) {
        if (!strata_env_is_one("STRATA_DS41_PREFILL_EG_CACHE")) return;
        const double t0 = now();
        prompt_eg_enabled_ = true;
        prompt_eg_verify_ = strata_env_is_one("STRATA_DS41_PREFILL_EG_CACHE_VERIFY");
        constexpr size_t cap = size_t(128) << 20;  // raw bytes across cached layers; candidate IDs separately bounded
        const int cols = c.engram_cols();
        if (cols <= 0 || ids.size() > size_t(INT_MAX) || ids.size() > size_t(-1) / size_t(cols))
            throw std::runtime_error("ds41: prompt Engram row count overflow");
        Seq history;
        history.hist = ids;
        prompt_eg_cache_.resize(size_t(layers));
        std::vector<int64_t> row_ids(static_cast<size_t>(cols));
        for (int l = 0; l < layers; ++l) {
            const Layer &Y = L[size_t(l)];
            if (c.engram_index(l) < 0 || Y.engram_fd < 0 || Y.engram_mem || Y.engram_pr1 >= Y.engram_r1) continue;
            auto &cache = prompt_eg_cache_[size_t(l)];
            const size_t rb = Y.engram_table.rb, max_candidates = cap / sizeof(int64_t);
            if (!rb) throw std::runtime_error("ds41: empty Engram row");
            cache.rows.reserve(std::min(max_candidates, ids.size() * size_t(cols) / size_t(std::max(1, comm.world))));
            bool bounded = true;
            for (size_t pos = 0; pos < ids.size() && bounded; ++pos) {
                Ent e;
                e.s = &history, e.pos = int(pos), e.token = ids[pos];
                engram_rows(l, e, row_ids.data());
                for (int64_t r : row_ids) {
                    if (r < std::max(Y.engram_r0, Y.engram_pr1) || r >= Y.engram_r1) continue;
                    if (cache.rows.size() >= max_candidates) { bounded = false; break; }
                    cache.rows.push_back(r);
                }
            }
            std::sort(cache.rows.begin(), cache.rows.end());
            cache.rows.erase(std::unique(cache.rows.begin(), cache.rows.end()), cache.rows.end());
            if (!bounded || cache.rows.size() > (cap - prompt_eg_bytes_) / rb || cache.rows.size() > size_t(INT_MAX)) {
                std::vector<int64_t>().swap(cache.rows);
                continue;  // this layer keeps the existing prefetch/read path
            }
            cache.raw.resize(cache.rows.size() * rb);
            if (g_prompt_io_core < 0) team_for(int(cache.rows.size()), [&](int a, int b) {
                for (int i = a; i < b; ++i) {
                    const off_t off = off_t(size_t(cache.rows[size_t(i)] - Y.engram_r0) * rb);
                    if (::pread(Y.engram_fd, cache.raw.data() + size_t(i) * rb, rb, off) != ssize_t(rb)) {
                        std::fprintf(stderr, "DS41 prompt Engram cache read failed\n");
                        std::abort();
                    }
                }
            });
            cache.active = true;
            prompt_eg_rows_ += cache.rows.size(), prompt_eg_bytes_ += cache.raw.size(), ++prompt_eg_layers_;
        }
        if (g_prompt_io_core >= 0) {
            // Build immutable first-use jobs only after sorted storage is final.
            for (int l = 0; l < layers; ++l) {
                auto &cache = prompt_eg_cache_[size_t(l)];
                if (!cache.active) continue;
                cache.jobs.assign(cache.rows.size(), size_t(-1));
                for (size_t pos = 0; pos < ids.size(); ++pos) {
                    Ent e; e.s=&history; e.pos=int(pos); e.token=ids[pos];
                    engram_rows(l,e,row_ids.data());
                    for (int64_t r : row_ids) {
                        auto it=std::lower_bound(cache.rows.begin(),cache.rows.end(),r);
                        if (it==cache.rows.end() || *it!=r) continue;
                        size_t i=size_t(it-cache.rows.begin());
                        if (cache.jobs[i]==size_t(-1)) {
                            cache.jobs[i]=prompt_row_jobs_.size();
                            prompt_row_jobs_.push_back({int(pos),l,i});
                        }
                    }
                }
            }
            std::sort(prompt_row_jobs_.begin(),prompt_row_jobs_.end(),[](const auto &a,const auto &b){
                const int ab=a.pos/g_prompt_io_pass, bb=b.pos/g_prompt_io_pass;
                if(ab!=bb)return ab<bb;
                if(a.layer!=b.layer)return a.layer<b.layer;
                if(a.pos!=b.pos)return a.pos<b.pos;
                return a.row<b.row;
            });
            for(size_t j=0;j<prompt_row_jobs_.size();++j) {
                const auto &job=prompt_row_jobs_[j];
                prompt_eg_cache_[size_t(job.layer)].jobs[job.row]=j;
            }
        }
        if (g_prompt_io_core >= 0) {
            int complete = 1;
            for (int l = 0; l < layers; ++l) {
                const Layer &Y = L[size_t(l)];
                if (c.engram_index(l) >= 0 && Y.engram_fd >= 0 && !Y.engram_mem && Y.engram_pr1 < Y.engram_r1 &&
                    !prompt_eg_cache_[size_t(l)].active) complete = 0;
            }
            int all_complete = complete;
#ifdef STRATA_CPU_RUNNER_MPI
            if (comm.world > 1) MPI_Allreduce(&complete, &all_complete, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
            if (!all_complete) throw std::runtime_error("async rows need complete prompt Engram caching on every rank");
        }
        if (g_prompt_io_core >= 0) {
            if (prompt_row_jobs_.size() != prompt_eg_rows_) throw std::runtime_error("prompt row first-use coverage");
            prompt_row_start_ = now();
            prompt_row_io_ = std::make_unique<strata::cpu_runner::RowPrefetch>(prompt_row_jobs_.size(),g_prompt_io_threads,
                [this](size_t j) {
                    const auto &job=prompt_row_jobs_[j];
                    const Layer &Y=L[size_t(job.layer)];
                    auto &cache=prompt_eg_cache_[size_t(job.layer)];
                    const size_t rb=Y.engram_table.rb;
                    const off_t off=off_t(size_t(cache.rows[job.row]-Y.engram_r0)*rb);
                    return ::pread(Y.engram_fd,cache.raw.data()+job.row*rb,rb,off)==ssize_t(rb);
                },[](int) {
                    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(g_prompt_io_core,&cs);
                    if(sched_setaffinity(0,sizeof cs,&cs))return false;
                    CPU_ZERO(&cs);
                    return !sched_getaffinity(0,sizeof cs,&cs) && CPU_COUNT(&cs)==1 && CPU_ISSET(g_prompt_io_core,&cs);
                });
        }
        prompt_eg_load_s_ = now() - t0;
    }
    const uint8_t *prompt_eg_row(int l, int64_t r) {
        if (prompt_eg_cache_.empty() || !prompt_eg_cache_[size_t(l)].active) return nullptr;
        const Layer &Y = L[size_t(l)];
        if (r < std::max(Y.engram_r0, Y.engram_pr1) || r >= Y.engram_r1) return nullptr;
        const auto &cache = prompt_eg_cache_[size_t(l)];
        const auto it = std::lower_bound(cache.rows.begin(), cache.rows.end(), r);
        if (it == cache.rows.end() || *it != r) {
            prompt_eg_misses_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        const size_t rb = Y.engram_table.rb;
        if (prompt_row_io_) {
            prompt_row_waits_.fetch_add(1,std::memory_order_relaxed);
            if (!prompt_row_io_->wait(cache.jobs[size_t(it-cache.rows.begin())])) {
                std::fprintf(stderr,"DS41 async prompt row read/affinity failed\n");std::abort();
            }
        }
        const uint8_t *raw = cache.raw.data() + size_t(it - cache.rows.begin()) * rb;
        if (prompt_eg_verify_) {
            std::vector<uint8_t> reference(rb);
            // Independent per-use read at the original caller row, not the cache's sorted index.
            if (::pread(Y.engram_fd, reference.data(), rb, off_t(size_t(r - Y.engram_r0) * rb)) != ssize_t(rb) ||
                std::memcmp(reference.data(), raw, rb)) {
                std::fprintf(stderr, "DS41 prompt Engram cache verification failed\n");
                std::abort();
            }
            prompt_eg_checks_.fetch_add(1, std::memory_order_relaxed);
        }
        prompt_eg_hits_.fetch_add(1, std::memory_order_relaxed);
        return raw;
    }
    void finish_prompt_eg_cache() {
        if (!prompt_eg_enabled_) return;
        if (prompt_row_io_) {
            if (!prompt_row_io_->finish()) { std::fprintf(stderr,"DS41 async prompt row drain failed\n");std::abort(); }
            std::printf("RUNNER_DS41_PROMPT_IO rank=%d jobs=%zu completed=%zu waits=%llu readers=%d core=%d lifetime_s=%.6f\n",
                comm.rank,prompt_row_jobs_.size(),prompt_row_io_->completed(),
                (unsigned long long)prompt_row_waits_.load(),g_prompt_io_threads,g_prompt_io_core,now()-prompt_row_start_);
            prompt_row_io_.reset();
            std::vector<PromptRowJob>().swap(prompt_row_jobs_);
        }
        std::printf("RUNNER_DS41_PREFILL_EG_CACHE rank=%d layers=%zu rows=%zu bytes=%zu hits=%llu misses=%llu verify_calls=%llu preload_s=%.6f\n",
                    comm.rank, prompt_eg_layers_, prompt_eg_rows_, prompt_eg_bytes_,
                    (unsigned long long) prompt_eg_hits_.load(), (unsigned long long) prompt_eg_misses_.load(),
                    (unsigned long long) prompt_eg_checks_.load(), prompt_eg_load_s_);
        std::fflush(stdout);
        std::vector<PromptEgCache>().swap(prompt_eg_cache_);
        prompt_eg_enabled_ = false;
    }

    // the prefetch: per Engram layer without RAM, the pass's rows of this rank's shard read into buf by a helper
    // thread (pinned to the core after the team's); engram() waits for it
    struct EgFetch {
        std::vector<int64_t> rows;   // T x cols (all rows; those outside the shard are not read)
        std::vector<uint8_t> buf;    // T x cols x row bytes
        std::atomic<int> done{1};
        std::atomic<int> left{0};   // parts still being read (STRATA_DS41_EG_THREADS)
        bool advised = false;   // (STRATA_DS41_EG_ADVISE) its rows' reads already requested
    };
    std::vector<std::unique_ptr<EgFetch>> egf_;   // per layer (null: not prefetched)
    // STRATA_DS41_EG_THREADS (default 1): helper threads, each layer's rows split among them (row q to part q % n), so
    // several reads are in flight; the bytes read are the same
    static int eg_threads() { static const int v = std::getenv("STRATA_DS41_EG_THREADS") ? std::max(1, std::min(16, std::atoi(std::getenv("STRATA_DS41_EG_THREADS")))) : 1; return v; }
    std::vector<std::thread> eg_thread_;
    std::mutex eg_mu_;
    std::condition_variable eg_cv_;
    struct EgJob { int l; EgFetch *f; int part; };
    std::deque<EgJob> eg_q_;
    bool eg_stop_ = false;
    double eg_wait_s = 0;
    void eg_worker() {
        {
            cpu_set_t cs;
            CPU_ZERO(&cs);
            int first = 0;
            if (sched_getaffinity(0, sizeof cs, &cs) == 0)
                for (int k = 0; k < CPU_SETSIZE; ++k) if (CPU_ISSET(k, &cs)) { first = k; break; }
            CPU_ZERO(&cs);
            CPU_SET(first + omp_get_max_threads(), &cs);
            sched_setaffinity(0, sizeof cs, &cs);
        }
        static const bool advise = strata_env_is_one("STRATA_DS41_EG_ADVISE");
        const int np = eg_threads();
        for (;;) {
            EgJob job;
            std::vector<std::pair<int, EgFetch *>> all;   // (advise: the queued jobs this worker requests)
            {
                std::unique_lock<std::mutex> lk(eg_mu_);
                eg_cv_.wait(lk, [&] { return eg_stop_ || !eg_q_.empty(); });
                if (eg_stop_ && eg_q_.empty()) return;
                job = eg_q_.front();
                eg_q_.pop_front();
                if (advise && !job.f->advised) {
                    all.push_back({job.l, job.f}), job.f->advised = true;
                    for (const EgJob &j : eg_q_)
                        if (!j.f->advised) all.push_back({j.l, j.f}), j.f->advised = true;
                }
            }
            // STRATA_DS41_EG_ADVISE=1: before the first read of a batch of jobs, every queued job's rows requested from the
            // kernel at once (posix_fadvise WILLNEED: asynchronous reads), so the preads below find them read or in flight
            for (auto &[l2, f2] : all) {
                const Layer &Y2 = L[size_t(l2)];
                const size_t rb2 = Y2.engram_table.rb;
                for (const int64_t r : f2->rows)
                    if (r >= std::max(Y2.engram_r0, Y2.engram_pr1) && r < Y2.engram_r1) ::posix_fadvise(Y2.engram_fd, off_t(size_t(r - Y2.engram_r0) * rb2), off_t(rb2), POSIX_FADV_WILLNEED);
            }
            const Layer &Y = L[size_t(job.l)];
            EgFetch &f = *job.f;
            const size_t rb = Y.engram_table.rb;
            for (size_t q = size_t(job.part); q < f.rows.size(); q += size_t(np)) {
                const int64_t r = f.rows[q];
                if (r < std::max(Y.engram_r0, Y.engram_pr1) || r >= Y.engram_r1) continue;
                if (::pread(Y.engram_fd, f.buf.data() + q * rb, rb, off_t(size_t(r - Y.engram_r0) * rb)) != ssize_t(rb)) std::abort();
            }
            if (f.left.fetch_sub(1, std::memory_order_acq_rel) == 1) f.done.store(1, std::memory_order_release);
        }
    }
    void prefetch_engram(std::vector<Ent> &E) {
        const int cols = c.engram_cols(), T = int(E.size());
        for (int l = 0; l < layers; ++l) {
            const Layer &Y = L[size_t(l)];
            if (c.engram_index(l) < 0 || Y.engram_fd < 0 || Y.engram_mem) continue;
            if (!prompt_eg_cache_.empty() && prompt_eg_cache_[size_t(l)].active) continue;
            if (egf_.size() < size_t(layers)) egf_.resize(size_t(layers));
            if (!egf_[size_t(l)]) egf_[size_t(l)] = std::make_unique<EgFetch>();
            EgFetch &f = *egf_[size_t(l)];
            while (!f.done.load(std::memory_order_acquire)) {}   // (a previous pass's job: done before its engram())
            f.rows.resize(size_t(cols) * T), f.buf.resize(size_t(cols) * T * Y.engram_table.rb);
            for (int i = 0; i < T; ++i) engram_rows(l, E[size_t(i)], f.rows.data() + size_t(i) * cols);
            const int np = eg_threads();
            f.done.store(0, std::memory_order_relaxed), f.left.store(np, std::memory_order_relaxed), f.advised = false;
            while (int(eg_thread_.size()) < np) eg_thread_.emplace_back([this] { eg_worker(); });
            {
                std::lock_guard<std::mutex> lk(eg_mu_);
                for (int k = 0; k < np; ++k) eg_q_.push_back({l, &f, k});
            }
            if (np == 1) eg_cv_.notify_one();
            else eg_cv_.notify_all();
        }
    }
    ~Runner() {
        prompt_row_io_.reset();  // join readers before raw storage or model file descriptors can be freed
        if (!eg_thread_.empty()) {
            { std::lock_guard<std::mutex> lk(eg_mu_); eg_stop_ = true; }
            eg_cv_.notify_all();
            for (auto &t : eg_thread_) t.join();
        }
    }

    void engram(int l, std::vector<Ent> &E) {
        const Layer &Y = L[size_t(l)];
        const int cols = c.engram_cols(), dim = c.engram_dim, n = c.hidden, T = int(E.size());
        const size_t erow = size_t(cols) * dim;
        double te = now();
        auto lap = [&](const char *k) { const double t = now(); tm(k) += t - te; te = t; };
        EgFetch *pf = size_t(l) < egf_.size() && egf_[size_t(l)] && !Y.engram_mem ? egf_[size_t(l)].get() : nullptr;
        std::vector<int64_t> rows_own;
        const int64_t *rows;
        if (pf) {
            const double tw = now();
            while (!pf->done.load(std::memory_order_acquire)) {}
            eg_wait_s += now() - tw;
            tm("engram.wait") += now() - tw;
            rows = pf->rows.data();
        } else {
            rows_own.resize(static_cast<size_t>(cols) * T);
            for (int i = 0; i < T; ++i) engram_rows(l, E[size_t(i)], rows_own.data() + size_t(i) * cols);
            rows = rows_own.data();
        }
        lap("engram.hash");
        const int kvn = Y.engram_wkv.rows;   // (the full projection's rows; released by split_tp, its shape kept)
        Vec emb(erow * T, 0.f), kv(size_t(kvn) * T, 0.f);
        std::vector<uint8_t> raw(Y.engram_table.rb);
        if (Y.engram_fd >= 0) {   // the shards: each rank its own rows, the others zero, summed
            auto row = [&](int q, std::vector<uint8_t> &rw) {
                const int i = q / cols, j = q % cols;
                const int64_t r = rows[size_t(i) * cols + size_t(j)];
                if (r < Y.engram_r0 || r >= Y.engram_r1) return;
                const uint8_t *src;
                const uint8_t *cached = prompt_eg_row(l, r);
                if (cached) src = cached;
                else if (Y.engram_mem) src = Y.engram_mem + size_t(r - Y.engram_r0) * rw.size();
                else if (r < Y.engram_pr1) src = Y.engram_part + size_t(r - Y.engram_r0) * rw.size();
                else if (pf) src = pf->buf.data() + (size_t(i) * cols + size_t(j)) * rw.size();
                else {
                    if (::pread(Y.engram_fd, rw.data(), rw.size(), off_t(size_t(r - Y.engram_r0) * rw.size())) != ssize_t(rw.size())) std::abort();
                    src = rw.data();
                }
                ds41_dequant_row(Y.engram_table.type, src, dim, emb.data() + size_t(i) * erow + size_t(j) * dim);
            };
            static const bool emb_ag = strata_env_is_one("STRATA_DS41_EMB_AG");
            const int W = comm.world, wr = int((raw.size() + 3) / 4), nq = T * cols;
            const int64_t N = Y.engram_table.rows;
            // STRATA_DS41_EMB_AG=1: each rank's rows gathered as their table bytes (an all-gather, the bits as sent), then
            // every row dequantized on every rank + 0.0f: the all-reduce's value (the one rank's value and zeros, summed)
            std::vector<int> slot, cnt;
            int woff[65];
            bool ag = emb_ag && comm.xc && W > 1 && W <= 64;
            if (ag) {
                slot.resize(size_t(nq)), cnt.assign(size_t(W), 0);
                for (int q = 0; q < nq; ++q) {
                    const int64_t r = rows[size_t(q)];
                    int o = r < 0 || r >= N ? -1 : int(r * W / N);
                    while (o >= 0 && o + 1 < W && r >= N * (o + 1) / W) ++o;
                    while (o > 0 && r < N * o / W) --o;
                    slot[size_t(q)] = o < 0 ? -1 : (o << 24 | cnt[size_t(o)]++);
                }
                woff[0] = 0;
                for (int o = 0; o < W; ++o) {
                    ag = ag && int64_t(cnt[size_t(o)]) * wr <= comm.xc_cap() && cnt[size_t(o)] < (1 << 24);
                    woff[o + 1] = woff[o] + cnt[size_t(o)] * wr;
                }
            }
            if (ag) {
                Vec G(static_cast<size_t>(woff[W]));
                uint8_t *const gb = reinterpret_cast<uint8_t *>(G.data());
                auto put = [&](int q, std::vector<uint8_t> &rw) {   // an own row's bytes to its slot
                    const int sl = slot[size_t(q)];
                    if (sl < 0 || (sl >> 24) != comm.rank) return;
                    const int64_t r = rows[size_t(q)];
                    uint8_t *dst = gb + (size_t(woff[comm.rank]) + size_t(sl & 0xffffff) * wr) * 4;
                    const uint8_t *cached = prompt_eg_row(l, r);
                    if (cached) std::memcpy(dst, cached, rw.size());
                    else if (Y.engram_mem) std::memcpy(dst, Y.engram_mem + size_t(r - Y.engram_r0) * rw.size(), rw.size());
                    else if (r < Y.engram_pr1) std::memcpy(dst, Y.engram_part + size_t(r - Y.engram_r0) * rw.size(), rw.size());
                    else if (pf) std::memcpy(dst, pf->buf.data() + size_t(q) * rw.size(), rw.size());
                    else if (::pread(Y.engram_fd, dst, rw.size(), off_t(size_t(r - Y.engram_r0) * rw.size())) != ssize_t(rw.size())) std::abort();
                };
                if (T == 1) for (int q = 0; q < nq; ++q) put(q, raw);
                else team_for(nq, [&](int a, int b) { for (int q = a; q < b; ++q) put(q, raw); });
                lap("engram.rows");
                const double tg = now();
                float *const gp = G.data();
                cdo([&, gp] { comm.allgather_shards(gp, woff, false); });
                tm("comm") += now() - tg, tm("comm.emb") += now() - tg;
                auto deq = [&](int q) {
                    const int sl = slot[size_t(q)];
                    if (sl < 0) return;
                    float *e = emb.data() + size_t(q) * dim;
                    ds41_dequant_row(Y.engram_table.type, gb + (size_t(woff[sl >> 24]) + size_t(sl & 0xffffff) * wr) * 4, dim, e);
                    for (int k = 0; k < dim; ++k) e[k] += 0.0f;
                };
                if (T == 1) for (int q = 0; q < nq; ++q) deq(q);
                else team_for(nq, [&](int a, int b) { for (int q = a; q < b; ++q) deq(q); });
                lap("engram.comm_rows");
            } else {
                if (T == 1) for (int q = 0; q < cols; ++q) row(q, raw);
                else team_for(T * cols, [&](int a, int b) { std::vector<uint8_t> rw(raw.size()); for (int q = a; q < b; ++q) row(q, rw); });
                lap("engram.rows");
                allreduce_timed(emb.data(), int(emb.size()), "comm.emb");
                lap("engram.comm_rows");
            }
        } else {
            const auto &tt = M.t("blk." + std::to_string(l) + ".engram_embd.weight");
            for (int i = 0; i < T; ++i)
                for (int j = 0; j < cols; ++j) {
                    read_bytes(tt, Y.engram_table.rb * size_t(rows[size_t(i) * cols + size_t(j)]), raw.size(), raw.data());
                    ds41_dequant_row(Y.engram_table.type, raw.data(), dim, emb.data() + size_t(i) * erow + size_t(j) * dim);
                }
        }
        {   // this rank's rows of the projection, summed over the ranks (zeros elsewhere)
            const int e0 = int(int64_t(kvn) * comm.rank / comm.world);
            const GJob ej{&Y.engram_wkv_t, emb.data(), kv.data() + e0, 0, Y.engram_wkv_t.rows, T, erow, size_t(kvn)};
            gemv_multi(&ej, 1, "engram_wkv");
            lap("engram.proj");
            if (es_on_)   // (ENTRY_SHARD: only the owners read their entries' rows)
                slices_to_owners(kv.data(), size_t(kvn), kvn, T, [&](int i) { return kv.data() + size_t(i) * kvn; }, "comm.engram_kv");
            else allreduce_timed(kv.data(), int(kv.size()), "comm.engram_kv");
            lap("engram.comm_kv");
        }
        // the gate per (token, stream) over the team: each stream's sums in its serial order
        team_for(4 * (es_b_ - es_a_), [&](int qa, int qb) {   // (the own entries: all without ENTRY_SHARD)
          for (int q = qa; q < qb; ++q) {
            const int i = es_a_ + q / 4, s = q % 4;
            Ent &en = E[size_t(i)];
            const float *kvi = kv.data() + size_t(i) * kvn, *value = kvi + size_t(4) * n;
            {
                float *xs = en.h.data() + size_t(s) * n;
                const float *key = kvi + size_t(s) * n, *g = Y.engram_gate.data() + size_t(s) * n;
                double xx = 0, kk = 0;
                for (int j = 0; j < n; ++j) xx += double(xs[j]) * xs[j], kk += double(key[j]) * key[j];
                const float rx = 1.0f / std::sqrt(float(xx / n) + c.rms_eps), rk = 1.0f / std::sqrt(float(kk / n) + c.rms_eps);
                double dt = 0;
                for (int j = 0; j < n; ++j) dt += double(xs[j] * rx) * (key[j] * rk) * g[j];
                const float d = float(dt) * (1.0f / std::sqrt(float(n)));
                const float mag = std::sqrt(std::max(std::fabs(d), 1e-6f));
                const float gate = sigmoid(d < 0 ? -mag : mag);
                for (int j = 0; j < n; ++j) xs[j] += value[j] * gate;
            }
          }
        });
        lap("engram.gate");
    }

    // a KV source layer's compressed rows for the pass's positions, in order (ratio 2: the odd position closes its pair)
    // VS: each token's value (and gate) rows, ld apart (computed with q_a / kv)
    void compress(int l, std::vector<Ent> &E, float *VS, size_t ld) {
        const Layer &Y = L[size_t(l)];
        const int ratio = c.ratio[size_t(l)], hd = c.head_dim, T = int(E.size());
        for (int i = 0; i < T; ++i) {
            Ent &e = E[size_t(i)];
            Seq &q = *e.s;
            const int pos = e.pos;
            float *v = VS + size_t(i) * ld;
            const float *s = v + hd;
            if (ratio > 1 && pos % ratio == 0) {
                std::copy(v, v + hd, q.stash_v[size_t(l)].begin()), std::copy(s, s + hd, q.stash_s[size_t(l)].begin());
                if (spec_track_) e.sv[size_t(l)].assign(v, v + hd), e.ss[size_t(l)].assign(s, s + hd);
                continue;
            }
            if (ratio > 1)
                for (int j = 0; j < hd; ++j) {
                    const float a = q.stash_s[size_t(l)][size_t(j)], b = s[j], mx = std::max(a, b);
                    const float e0 = std::exp(a - mx), e1 = std::exp(b - mx), inv = 1.0f / (e0 + e1);
                    v[j] = q.stash_v[size_t(l)][size_t(j)] * (e0 * inv) + v[j] * (e1 * inv);
                }
            const int g = pos / ratio, first = pos - (ratio - 1);
            if (g >= int(q.cache[size_t(l)].size() / size_t(row_width()))) throw std::runtime_error("ds41: --context too small");
            float *row = q.cache[size_t(l)].data() + size_t(g) * row_width();
            rms_norm(v, Y.comp_norm.data(), row, hd, c.rms_eps);
            Vec k(static_cast<size_t>(c.index_dim));
            gemv(Y.idx_k, row, k.data(), "idx_k");
            rms_norm(k.data(), Y.idx_k_norm.data(), row + hd, c.index_dim, c.rms_eps);
            const float *t = e.table(first == pos ? 2 : 4);
            rope(row + hd + c.index_dim - c.rope_dim, c.rope_dim / 2, t);
            rope(row + hd - c.rope_dim, c.rope_dim / 2, t);
        }
    }

    // Diagnostic only: independently execute the original distributed head projections,
    // score loops and rank-order all-reduce, then compare every owner's score bitwise.
    // These runs are correctness evidence, never speed measurements.
    void index_owned_check(int l, const std::vector<Ent> &E, const float *QR, const float *X,
                           const Vec &owned_scores, int own_rows) {
        const Layer &Y = L[size_t(l)];
        const int T = int(E.size()), H = c.index_heads, D = c.index_dim, ni = ih1 - ih0;
        const int ratio = c.ratio[size_t(l)], src = c.kv_src[size_t(l)];
        const Mat &iqb = Y.idx_q_b_t;
        // The owner-QK path intentionally leaves non-owner X rows stale. Rebuild
        // a separate reference input from owners before the legacy verifier projects it.
        Vec reference_x;
        if (index_owner_qk(l) && es_qk_layer(l)) {
            reference_x.resize(size_t(T) * c.hidden);
            std::copy(X + size_t(es_a_) * c.hidden, X + size_t(es_b_) * c.hidden,
                      reference_x.data() + size_t(es_a_) * c.hidden);
            int off[65];
            for (int r = 0; r <= comm.world; ++r) off[r] = es_off_[r] * c.hidden;
            float *const xp = reference_x.data();
            cdo([&, xp] { comm.allgather_shards(xp, off, false); });
            X = reference_x.data();
            ++index_verify_x_gathers_;
        }
        Vec IQ(size_t(iqb.rows) * T), IW(size_t(Y.idx_proj.rows) * T);
        const GJob ij[2] = {{&iqb, QR, IQ.data(), 0, iqb.rows, T, size_t(c.q_rank), size_t(iqb.rows)},
                            {&Y.idx_proj, X, IW.data(), 0, Y.idx_proj.rows, T, size_t(c.hidden), size_t(Y.idx_proj.rows)}};
        gemv_multi(ij, 2, "idx_verify_q_b+proj");
        std::vector<int> offsets(static_cast<size_t>(T) + 1, 0);
        for (int i = 0; i < T; ++i) {
            offsets[size_t(i) + 1] = offsets[size_t(i)] + std::max(0, (E[size_t(i)].pos + 1) / ratio);
            for (int hh = 0; hh < ni; ++hh)
                rope(IQ.data() + size_t(i) * iqb.rows + size_t(hh) * D + D - c.rope_dim, c.rope_dim / 2, E[size_t(i)].table(2));
        }
        const int total = offsets[size_t(T)];
        Vec reference(static_cast<size_t>(std::max(total, 1)));
        const float w_scale = 1.0f / std::sqrt(float(D * H));
        team_for(total, [&](int ra, int rb) {
            int i = 0;
            for (int r = ra; r < rb; ++r) {
                while (r >= offsets[size_t(i) + 1]) ++i;
                const float *iq = IQ.data() + size_t(i) * iqb.rows, *iw = IW.data() + size_t(i) * Y.idx_proj.rows;
                const float *k = E[size_t(i)].s->cache[size_t(src)].data() + c.head_dim + size_t(r - offsets[size_t(i)]) * row_width();
                float tot = 0;
                for (int hh = 0; hh < ni; ++hh)
                    tot += std::max(dot(k, iq + size_t(hh) * D, D), 0.0f) * (iw[size_t(ih0 + hh)] * w_scale);
                reference[size_t(r)] = tot;
            }
        });
        allreduce_timed(reference.data(), total, "comm.idx_verify");
        if (offsets[size_t(es_b_)] - offsets[size_t(es_a_)] != own_rows)
            throw std::runtime_error("ds41: index owner verification range");
        float mismatch = std::memcmp(owned_scores.data(), reference.data() + offsets[size_t(es_a_)], size_t(own_rows) * sizeof(float)) ? 1.f : 0.f;
        allreduce_timed(&mismatch, 1, "comm.idx_verify_flag");
        if (mismatch != 0.f) throw std::runtime_error("ds41: index owner score bits differ at layer " + std::to_string(l));
        ++index_verify_calls_;
        index_verified_values_ += uint64_t(own_rows);
    }

    // Entry-owned indexer. Keep the old head partitions and +0, rank-order reduction
    // locally, then transmit only selected row IDs. QR/X and compressed keys are already
    // replicated; an owner's candidate lists stay on that owner through every layer.
    void index_owned(int l, std::vector<Ent> &E, const float *QR, const float *X) {
        ++index_owned_calls_;
        const Layer &Y = L[size_t(l)];
        const int H = c.index_heads, D = c.index_dim, ratio = c.ratio[size_t(l)], src = c.kv_src[size_t(l)];
        const int T = int(E.size()), W = comm.world, a = es_a_, own = es_b_ - es_a_;
        const Mat &iqb = Y.idx_q_b_full;
        if (own < 1 || iqb.rows != H * D) throw std::runtime_error("ds41: index owner projection shape");
        Vec IQ(size_t(iqb.rows) * own), IW(size_t(Y.idx_proj.rows) * own);
        const GJob ij[2] = {{&iqb, QR + size_t(a) * c.q_rank, IQ.data(), 0, iqb.rows, own, size_t(c.q_rank), size_t(iqb.rows)},
                            {&Y.idx_proj, X + size_t(a) * c.hidden, IW.data(), 0, Y.idx_proj.rows, own, size_t(c.hidden), size_t(Y.idx_proj.rows)}};
        double ti = now();
        auto lap = [&](const char *k) { const double t2 = now(); tm(k) += t2 - ti; ti = t2; };
        gemv_multi(ij, 2, "idx_owner_q_b+proj");
        lap("idx.proj");
        std::vector<int> vis(static_cast<size_t>(own)), off(static_cast<size_t>(own) + 1, 0);
        for (int i = 0; i < own; ++i) {
            const Ent &e = E[size_t(a + i)];
            vis[size_t(i)] = (e.pos + 1) / ratio;
            off[size_t(i) + 1] = off[size_t(i)] + std::max(0, vis[size_t(i)]);
            for (int hh = 0; hh < H; ++hh)
                rope(IQ.data() + size_t(i) * iqb.rows + size_t(hh) * D + D - c.rope_dim, c.rope_dim / 2, e.table(2));
        }
        const int total = off[size_t(own)];
        Vec scores(static_cast<size_t>(std::max(total, 1)));
        const float w_scale = 1.0f / std::sqrt(float(D * H));
        team_for(total, [&](int ra, int rb) {
            int i = 0;
            for (int r = ra; r < rb; ++r) {
                while (r >= off[size_t(i) + 1]) ++i;
                const float *iq = IQ.data() + size_t(i) * iqb.rows, *iw = IW.data() + size_t(i) * Y.idx_proj.rows;
                const float *k = E[size_t(a + i)].s->cache[size_t(src)].data() + c.head_dim + size_t(r - off[size_t(i)]) * row_width();
                float sum = 0;
                for (int rank = 0; rank < W; ++rank) {
                    float part = 0;
                    for (int hh = H * rank / W; hh < H * (rank + 1) / W; ++hh)
                        part += std::max(dot(k, iq + size_t(hh) * D, D), 0.0f) * (iw[size_t(hh)] * w_scale);
                    sum += part;   // all-reduce's +0 and rank order, including empty partitions
                }
                scores[size_t(r)] = sum;
            }
        });
        lap("idx.scores");
        static const bool verify = strata_env_truthy_one("STRATA_DS41_ES_INDEX_VERIFY");
        if (verify) {
            index_owned_check(l, E, QR, X, scores, total);
            lap("idx.verify");
        }
        const int bs = c.candidate_block;
        team_for(own, [&](int ia, int ib) {
            for (int i = ia; i < ib; ++i) {
                Ent &e = E[size_t(a + i)];
                auto &picked = e.sel[size_t(l)];
                picked.clear();
                const int visible = vis[size_t(i)];
                if (visible <= 0) continue;
                float *si = scores.data() + off[size_t(i)];
                if (c.candidate_source == l) {
                    const int nb = (visible + bs - 1) / bs;
                    Vec best(static_cast<size_t>(nb), -INFINITY);
                    for (int t = 0; t < visible; ++t) best[size_t(t / bs)] = std::max(best[size_t(t / bs)], si[t]);
                    best[size_t((visible - 1) / bs)] += 1e30f;
                    e.candidates = top_k(best.data(), nb, c.candidate_blocks);
                } else if (c.candidate_source >= 0 && c.candidate_source < l) {
                    std::vector<char> keep(static_cast<size_t>((visible + bs - 1) / bs), 0);
                    for (int b : e.candidates) if (b < int(keep.size())) keep[size_t(b)] = 1;
                    for (int t = 0; t < visible; ++t) if (!keep[size_t(t / bs)]) si[t] = -INFINITY;
                }
                top_k_rows(si, visible, c.index_top_k, picked);
                picked.erase(std::remove_if(picked.begin(), picked.end(), [&](int t) { return std::isinf(si[t]) && si[t] < 0; }), picked.end());
            }
        });
        lap("idx.select");
        // Integers are carried as raw bits in float slots; canon=false performs copies,
        // never floating-point arithmetic (so subnormal/NaN encodings are preserved).
        static_assert(sizeof(int) == sizeof(float));
        if (c.index_top_k < 0) throw std::runtime_error("ds41: index owner top-k");
        const size_t stride = size_t(c.index_top_k) + 1;
        if (size_t(T) * stride > size_t(0x7fffffff)) throw std::runtime_error("ds41: index owner payload size");
        int bounds[65];
        for (int r = 0; r <= W; ++r) bounds[r] = int(size_t(es_off_[r]) * stride);
        for (int r = 0; r < W; ++r)
            if (bounds[r + 1] - bounds[r] > comm.xc_cap()) throw std::runtime_error("ds41: index owner shard capacity");
        Vec packed(size_t(T) * stride, 0.f);
        for (int i = a; i < es_b_; ++i) {
            const auto &picked = E[size_t(i)].sel[size_t(l)];
            const int count = int(picked.size());
            if (count > c.index_top_k) throw std::runtime_error("ds41: index owner selection size");
            std::memcpy(packed.data() + size_t(i) * stride, &count, sizeof count);
            if (count) std::memcpy(packed.data() + size_t(i) * stride + 1, picked.data(), size_t(count) * sizeof(int));
        }
        float *const values = packed.data();
        cdo([&, values] { comm.allgather_shards(values, bounds, false); });
        for (int i = 0; i < T; ++i) {
            if (i >= a && i < es_b_) continue;
            int count;
            std::memcpy(&count, values + size_t(i) * stride, sizeof count);
            if (count < 0 || count > c.index_top_k) throw std::runtime_error("ds41: index owner received count");
            auto &picked = E[size_t(i)].sel[size_t(l)];
            picked.resize(size_t(count));
            if (count) std::memcpy(picked.data(), values + size_t(i) * stride + 1, size_t(count) * sizeof(int));
            const int visible = (E[size_t(i)].pos + 1) / ratio;
            for (int row : picked)
                if (row < 0 || row >= visible) throw std::runtime_error("ds41: index owner received row");
        }
        const double dt = now() - ti;
        tm("comm") += dt, tm("comm.idx_sel") += dt;
        lap("idx.comm");
    }

    // an index source layer's top-k of each token's visible compressed rows (ds41_decode.cpp index + host_select)
    void index(int l, std::vector<Ent> &E, const float *QR, const float *X) {
        const Layer &Y = L[size_t(l)];
        if (es_on_ && Y.idx_q_b_full.p) { index_owned(l, E, QR, X); return; }
        const int ratio = c.ratio[size_t(l)], H = c.index_heads, D = c.index_dim, src = c.kv_src[size_t(l)], T = int(E.size());
        const int ia = ih0, ni = ih1 - ih0;   // this rank's indexer heads (partial scores, all-reduced)
        const Mat &iqb = Y.idx_q_b_t;
        Vec IQ(size_t(iqb.rows) * T), IW(size_t(Y.idx_proj.rows) * T);
        const GJob ij[2] = {{&iqb, QR, IQ.data(), 0, iqb.rows, T, size_t(c.q_rank), size_t(iqb.rows)},
                            {&Y.idx_proj, X, IW.data(), 0, Y.idx_proj.rows, T, size_t(c.hidden), size_t(Y.idx_proj.rows)}};
        double ti = now();
        auto lap = [&](const char *k) { const double t2 = now(); tm(k) += t2 - ti; ti = t2; };
        gemv_multi(ij, 2, "idx_q_b+proj");
        lap("idx.proj");
        std::vector<int> vis(static_cast<size_t>(T)), off(static_cast<size_t>(T) + 1, 0);
        for (int i = 0; i < T; ++i) {
            vis[size_t(i)] = (E[size_t(i)].pos + 1) / ratio;
            off[size_t(i) + 1] = off[size_t(i)] + std::max(0, vis[size_t(i)]);
            for (int hh = 0; hh < ni; ++hh) rope(IQ.data() + size_t(i) * iqb.rows + size_t(hh) * D + D - c.rope_dim, c.rope_dim / 2, E[size_t(i)].table(2));
        }
        const int total = off[size_t(T)];
        Vec s(static_cast<size_t>(std::max(total, 1)));
        const float w_scale = 1.0f / std::sqrt(float(D * H));
        // STRATA_DS41_INDEX_ROWS=1: one prompt's shared index key stays hot while scoring
        // 16 queries. Each query retains its original head sum and dot-product order.
        static const bool index_rows_env = strata_env_truthy_one("STRATA_DS41_INDEX_ROWS");
        bool index_rows = index_rows_env && T >= 64;
        for (int i = 1; i < T && index_rows; ++i) index_rows = E[size_t(i)].s == E[0].s;
        if (index_rows) {
            team_run([&] {
                const int t = omp_get_thread_num(), nth = omp_get_num_threads();
                constexpr int tile = 16;
                for (int i0 = 0; i0 < T; i0 += tile) {
                    const int i1 = std::min(T, i0 + tile);
                    int max_rows = 0;
                    for (int i = i0; i < i1; ++i) max_rows = std::max(max_rows, vis[size_t(i)]);
                    const int ra = int(int64_t(max_rows) * t / nth), rb = int(int64_t(max_rows) * (t + 1) / nth);
                    for (int r = ra; r < rb; ++r) {
                        const float *k = E[0].s->cache[size_t(src)].data() + c.head_dim + size_t(r) * row_width();
                        for (int i = i0; i < i1; ++i) {
                            if (r >= vis[size_t(i)]) continue;
                            const float *iq = IQ.data() + size_t(i) * iqb.rows, *iw = IW.data() + size_t(i) * Y.idx_proj.rows;
                            float tot = 0;
                            for (int hh = 0; hh < ni; ++hh) tot += std::max(dot(k, iq + size_t(hh) * D, D), 0.0f) * (iw[size_t(ia + hh)] * w_scale);
                            s[size_t(off[size_t(i)]) + r] = tot;
                        }
                    }
                }
            });
        } else
        team_for(total, [&](int ra, int rb) {
            int i = 0;
            for (int r = ra; r < rb; ++r) {
                while (r >= off[size_t(i) + 1]) ++i;
                const float *iq = IQ.data() + size_t(i) * iqb.rows, *iw = IW.data() + size_t(i) * Y.idx_proj.rows;
                const float *k = E[size_t(i)].s->cache[size_t(src)].data() + c.head_dim + size_t(r - off[size_t(i)]) * row_width();
                float tot = 0;
                for (int hh = 0; hh < ni; ++hh) tot += std::max(dot(k, iq + size_t(hh) * D, D), 0.0f) * (iw[size_t(ia + hh)] * w_scale);
                s[size_t(r)] = tot;
            }
        });
        lap("idx.scores");
        allreduce_timed(s.data(), total, "comm.idx_scores");
        lap("idx.comm");
        const int bs = c.candidate_block;
        auto select = [&](int i) {
            Ent &e = E[size_t(i)];
            e.sel[size_t(l)].clear();
            const int visible = vis[size_t(i)];
            if (visible <= 0) return;
            float *si = s.data() + off[size_t(i)];
            if (c.candidate_source == l) {
                const int nb = (visible + bs - 1) / bs;
                Vec best(static_cast<size_t>(nb), -INFINITY);
                for (int t = 0; t < visible; ++t) best[size_t(t / bs)] = std::max(best[size_t(t / bs)], si[t]);
                best[size_t((visible - 1) / bs)] += 1e30f;
                e.candidates = top_k(best.data(), nb, c.candidate_blocks);
            } else if (c.candidate_source >= 0 && c.candidate_source < l) {
                std::vector<char> keep(static_cast<size_t>((visible + bs - 1) / bs), 0);
                for (int b : e.candidates) if (b < int(keep.size())) keep[size_t(b)] = 1;
                for (int t = 0; t < visible; ++t) if (!keep[size_t(t / bs)]) si[t] = -INFINITY;
            }
            // the top-k set in ascending row order (top_k + sort's result): the k-th element of top_k's strict order
            // by one nth_element, then the rows before it in that order, scanned by row; -inf rows dropped
            std::vector<int> &picked = e.sel[size_t(l)];
            top_k_rows(si, visible, c.index_top_k, picked);
            picked.erase(std::remove_if(picked.begin(), picked.end(), [&](int t) { return std::isinf(si[t]) && si[t] < 0; }), picked.end());
        };
        if (T == 1) select(0);
        else team_for(T, [&](int a, int b) { for (int i = a; i < b; ++i) select(i); });
        lap("idx.select");
    }

    void attention(int l, std::vector<Ent> &E) {
        const Layer &Y = L[size_t(l)];
        const int ratio = c.ratio[size_t(l)], hd = c.head_dim, nope = hd - c.rope_dim, pairs = c.rope_dim / 2, T = int(E.size());
        const int fwd = ratio ? 2 : 0, back = ratio ? 3 : 1, n = c.hidden, qk = c.q_rank + hd;
        const int ha = h0, nh = h1 - h0;   // this rank's heads
        // (STRATA_DS41_HEAD_BAL) this pass: role 1 shares its last head's tokens beyond part 0, role 2 computes part bpart
        // of its ext rank's head
        const bool bal = bal_on_ && T >= 12 && !(merge_bits() & 2 && T <= 4);
        const int brole = bal ? bal_role_[size_t(comm.rank)] : 0, bhp = bal_hp_;
        auto bpart_lo = [&](int k) { return int(int64_t(T) * k / (bhp + 1)); };
        const int nhc = brole == 1 ? nh - 1 : nh;   // heads the main core computes for every token
        double ta = now();
        auto lap = [&](const char *k) { const double t = now(); tm(k) += t - ta; ta = t; sub(k); };
        sub_begin();
        const float *X = gather_xn(E);
        const int src = ratio ? c.kv_src[size_t(l)] : -1, ts = ratio ? c.topk_src[size_t(l)] : -1;
        const int ncmp = src == l ? (ratio > 1 ? 2 : 1) : 0;   // the compressor's value (and gate) rows
        const int qkw = qk + ncmp * hd;
        // q_a's, kv's (and the compressor's) rows of this rank for every token, one all-reduce for all
        Vec QK(size_t(qkw) * T, 0.f);
        // STRATA_DS41_ES_QKQ=1 (with ES_QK; q_b reading Q8_K): the owner normalizes its entries' q_a rows and quantizes them
        // (each rank's own steps, on the same canonical bits), and the gather carries those Q8_K blocks for q_a
        static const bool qkq_env = strata_env_is_one("STRATA_DS41_ES_QKQ");
        const bool qkq = qkq_env && es_qk_layer(l) && c.q_rank % 256 == 0 && ds41_act_kind(Y.q_b_t.type) == DS41_ACT_Q8K;
        if (es_qk_layer(l)) {   // the own entries' rows in full, then every entry's row gathered (+ 0.0f)
            if (index_owner_qk(l) && ratio && ts == l) ++index_owner_qk_calls_;
            const int own = es_b_ - es_a_;
            const float *Xo = X + size_t(es_a_) * n;
            float *Qo = QK.data() + size_t(es_a_) * qkw;
            GJob fj[4] = {{&Y.q_a_full, Xo, Qo, 0, Y.q_a_full.rows, own, size_t(n), size_t(qkw)},
                          {&Y.kv_full, Xo, Qo + c.q_rank, 0, Y.kv_full.rows, own, size_t(n), size_t(qkw)}};
            if (ncmp >= 1) fj[2] = {&Y.comp_kv_full, Xo, Qo + qk, 0, Y.comp_kv_full.rows, own, size_t(n), size_t(qkw)};
            if (ncmp >= 2) fj[3] = {&Y.comp_gate_full, Xo, Qo + qk + hd, 0, Y.comp_gate_full.rows, own, size_t(n), size_t(qkw)};
            gemv_multi(fj, 2 + ncmp, "q_a+kv");
            const double t0 = now();
            int off[65];
            if (qkq) {   // (STRATA_DS41_ES_QKQ) per entry [q_a normed, as Q8_K blocks | kv (+ compressor) rows + 0.0f]
                const int W = comm.world, nbq = c.q_rank / 256, wq = nbq * 73 + (qkw - c.q_rank);
                xqr_.resize(size_t(nbq) * T);
                Vec &G = agbuf_;
                G.resize(size_t(T) * wq);
                team_for(own, [&](int a, int b) {
                    std::vector<float> qr(static_cast<size_t>(c.q_rank));
                    for (int i = es_a_ + a; i < es_a_ + b; ++i) {
                        float *row = QK.data() + size_t(i) * qkw, *g = G.data() + size_t(i) * wq;
                        for (int k = 0; k < qkw; ++k) row[k] += 0.0f;   // (the canonical gather's bits)
                        rms_norm(row, Y.q_a_norm.data(), qr.data(), c.q_rank, c.rms_eps);
                        for (int bq = 0; bq < nbq; ++bq) {
                            ds41_q8k &blk = xqr_[size_t(i) * nbq + size_t(bq)];
                            ds41_quant_q8k(qr.data() + 256 * size_t(bq), 256, &blk);
                            std::memcpy(g + size_t(bq) * 73, &blk, 4 * 73);
                        }
                        std::copy(row + c.q_rank, row + qkw, g + size_t(nbq) * 73);
                    }
                });
                for (int r = 0; r <= W; ++r) off[r] = es_off_[r] * wq;
                float *const gp = G.data();
                cdo([&, gp] { comm.allgather_shards(gp, off, false); });
                for (int i = 0; i < T; ++i) {
                    if (i >= es_a_ && i < es_b_) continue;
                    const float *g = G.data() + size_t(i) * wq;
                    for (int bq = 0; bq < nbq; ++bq) std::memcpy(&xqr_[size_t(i) * nbq + size_t(bq)], g + size_t(bq) * 73, 4 * 73);
                    std::copy(g + size_t(nbq) * 73, g + wq, QK.data() + size_t(i) * qkw + c.q_rank);
                }
            } else {
                for (int r = 0; r <= comm.world; ++r) off[r] = es_off_[r] * qkw;
                float *const qp = QK.data();
                cdo([&, qp] { comm.allgather_shards(qp, off, true); });
            }
            const double dt = now() - t0;
            tm("comm") += dt, tm("comm.qk") += dt;
        } else {
        GJob qj[4] = {{&Y.q_a_t, X, QK.data() + qa0, 0, Y.q_a_t.rows, T, size_t(n), size_t(qkw)},
                      {&Y.kv_t, X, QK.data() + c.q_rank + kv0, 0, Y.kv_t.rows, T, size_t(n), size_t(qkw)}};
        if (ncmp >= 1) qj[2] = {&Y.comp_kv_t, X, QK.data() + qk + ck0, 0, Y.comp_kv_t.rows, T, size_t(n), size_t(qkw)};
        if (ncmp >= 2) qj[3] = {&Y.comp_gate_t, X, QK.data() + qk + hd + ck0, 0, Y.comp_gate_t.rows, T, size_t(n), size_t(qkw)};
        gemv_multi(qj, 2 + ncmp, "q_a+kv");
        if ((ag_bits() & 1) && comm.xc) {
            const int sec[4][2] = {{0, c.q_rank}, {c.q_rank, hd}, {qk, hd}, {qk + hd, hd}};
            gather_slices(QK.data(), T, size_t(qkw), sec, 2 + ncmp, "comm.qk");
        } else allreduce_timed(QK.data(), int(QK.size()), "comm.qk");
        }
        Vec QR(size_t(c.q_rank) * T), Q(size_t(nh) * hd * T);
        auto per_token = [&](auto &&f) {   // independent per token: over the team when there are several
            if (T == 1) f(0);
            else team_for(T, [&](int a, int b) { for (int i = a; i < b; ++i) f(i); });
        };
        if (!qkq) per_token([&](int i) { rms_norm(QK.data() + size_t(i) * qkw, Y.q_a_norm.data(), QR.data() + size_t(i) * c.q_rank, c.q_rank, c.rms_eps); });
        Vec QX;   // (a helper) the helped head's q for its part's tokens
        const int bt0 = brole == 2 ? bpart_lo(bal_part_[size_t(comm.rank)]) : 0, bt1 = brole == 2 ? bpart_lo(bal_part_[size_t(comm.rank)] + 1) : 0;
        if (brole == 1) {   // the last head's rows for part 0 only
            const int t1 = bpart_lo(1);
            const GJob bj2[2] = {{&Y.q_b_t, QR.data(), Q.data(), 0, (nh - 1) * hd, T, size_t(c.q_rank), size_t(nh) * hd, qkq ? xqr_.data() : nullptr},
                                 {&Y.q_b_t, QR.data(), Q.data(), (nh - 1) * hd, nh * hd, t1, size_t(c.q_rank), size_t(nh) * hd, qkq ? xqr_.data() : nullptr}};
            gemv_multi(bj2, 2, "q_b");
        } else {
            const GJob bj{&Y.q_b_t, QR.data(), Q.data(), 0, Y.q_b_t.rows, T, size_t(c.q_rank), size_t(nh) * hd, qkq ? xqr_.data() : nullptr};
            if (brole == 2 && bt1 > bt0) {
                QX.resize(size_t(bt1 - bt0) * hd);
                const GJob bj2[2] = {bj, {&Y.q_b_x, QR.data() + size_t(bt0) * c.q_rank, QX.data(), 0, hd, bt1 - bt0, size_t(c.q_rank), size_t(hd),
                                          qkq ? xqr_.data() + size_t(bt0) * (c.q_rank / 256) : nullptr}};
                gemv_multi(bj2, 2, "q_b");
            } else gemv_multi(&bj, 1, "q_b");
        }
        per_token([&](int i) {
            Ent &e = E[size_t(i)];
            float *q = Q.data() + size_t(i) * nh * hd;
            for (int hh = 0; hh < nh; ++hh) rope(q + size_t(hh) * hd + nope, pairs, e.table(fwd));
            if (brole == 2 && i >= bt0 && i < bt1) rope(QX.data() + size_t(i - bt0) * hd + nope, pairs, e.table(fwd));
            float *wrow = e.s->window[size_t(l)].data() + size_t(e.pos % ringcap) * hd;
            rms_norm(QK.data() + size_t(i) * qkw + c.q_rank, Y.kv_a_norm.data(), wrow, hd, c.rms_eps);
            rope(wrow + nope, pairs, e.table(fwd));
        });
        lap("attn.proj_in");
        if (ratio) {
            if (src == l) compress(l, E, QK.data() + qk, size_t(qkw));
            lap("attn.compress");
            if (ts == l) index(l, E, QR.data(), X);
            lap("attn.index");
        }
        const double tcp = now();
        // STRATA_DS41_MERGE & 32: the core's buffers kept across layers (capacity reused, no zero fill: ATT and the
        // scores are written whole before they are read) and the row lists reserved
        const bool reuse = merge_bits() & 32;
        Vec ATT_own;
        if (!reuse) ATT_own.resize(size_t(nh) * hd * T);
        else if (core_att_.size() < size_t(nh) * hd * T) core_att_.resize(size_t(nh) * hd * T);
        float *const ATTp = reuse ? core_att_.data() : ATT_own.data();
        const float scale = 1.0f / std::sqrt(float(hd));
        if (hd % 16 || nope % 16 || nh > 64) throw std::runtime_error("ds41: attention blocks need 16-aligned heads");
        // every token's rows (its window oldest first, then its selected compressed rows) and score offsets
        std::vector<std::vector<const float *>> RW_own;
        if (!reuse) RW_own.resize(static_cast<size_t>(T));
        else if (core_rw_.size() < size_t(T)) core_rw_.resize(size_t(T));
        std::vector<std::vector<const float *>> &RW = reuse ? core_rw_ : RW_own;
        std::vector<int> roff(static_cast<size_t>(T) + 1, 0);
        // STRATA_DS41_MERGE & 64: only each token's row counts here; the scores phase writes the pointers of the rows it
        // scores (each thread its rows) into one flat list, read by the weighted sums after the phase's barrier
        const bool m_rw = merge_bits() & 64;
        int rfirst[kMaxBatch], rwin[kMaxBatch];
        for (int i = 0; i < T && m_rw; ++i) {
            const Ent &e = E[size_t(i)];
            const int pos = e.pos, first = std::max(0, pos - ring + 1), wrows = (e.last >= 0 ? e.last : pos) - first + 1;
            if (wrows > ringcap) throw std::runtime_error("ds41: attention rows beyond the ring");
            rfirst[i] = first, rwin[i] = wrows;
            roff[size_t(i) + 1] = roff[size_t(i)] + wrows + (ratio && (pos + 1) / ratio > 0 ? int(e.sel[size_t(ts)].size()) : 0);
        }
        for (int i = 0; i < T && !m_rw; ++i) {
            const Ent &e = E[size_t(i)];
            const int pos = e.pos, first = std::max(0, pos - ring + 1), wrows = (e.last >= 0 ? e.last : pos) - first + 1;
            if (wrows > ringcap) throw std::runtime_error("ds41: attention rows beyond the ring");
            auto &rows = RW[size_t(i)];
            if (reuse) rows.clear(), rows.reserve(size_t(wrows) + (ratio ? e.sel[size_t(ts)].size() : 0));
            for (int r = 0; r < wrows; ++r) rows.push_back(e.s->window[size_t(l)].data() + size_t((first + r) % ringcap) * hd);
            if (ratio && (pos + 1) / ratio > 0)
                for (int r : e.sel[size_t(ts)]) rows.push_back(e.s->cache[size_t(src)].data() + size_t(r) * row_width());
            roff[size_t(i) + 1] = roff[size_t(i)] + int(rows.size());
        }
        const int NR = roff[size_t(T)];
        if (m_rw && core_rwf_cap_ < size_t(NR)) core_rwf_cap_ = size_t(NR) * 2, core_rwf_.reset(new const float *[core_rwf_cap_]);
        const float **const RWF = m_rw ? core_rwf_.get() : nullptr;
        auto row_of = [&](int i, int r) -> const float * {   // (m_rw) token i's row r: its window oldest first, then its selection
            const Ent &e = E[size_t(i)];
            return r < rwin[i] ? e.s->window[size_t(l)].data() + size_t((rfirst[i] + r) % ringcap) * hd
                               : e.s->cache[size_t(src)].data() + size_t(e.sel[size_t(ts)][size_t(r - rwin[i])]) * row_width();
        };
        tm("core.p.rows") += now() - tcp;
        tm("core.n.rows_k") += NR * 1e-6, tm("core.n.heads") += nh * 1e-3 / std::max(layers, 1);   // (counts per step: thousands of rows; heads per rank)
        // MQA: every head reads the same rows. One team region for all tokens: the scores by rows (each row read once for
        // all heads) | each (token, head)'s max | the exps over the team | each (token, head)'s sum in row order and its
        // inverse | the weighted sums by (token, 16 columns), every head at once, then the scale and the rope pairs.
        // Per element the operations of a token alone.
        Vec sc_own;
        if (!reuse) sc_own.resize(static_cast<size_t>(nh) * NR);
        else if (core_sc_.size() < size_t(nh) * NR) core_sc_.resize(size_t(nh) * NR);
        float *const scp = reuse ? core_sc_.data() : sc_own.data();
        std::vector<float> inv(static_cast<size_t>(nh) * T), mxh(static_cast<size_t>(nh) * T);
        double mk[6] = {0, 0, 0, 0, 0, 0};
        mk[0] = now();
        // the split: every thread over every token, or (batched: each token's sequence homed on a CMG) each CMG's threads
        // over its own tokens, whose KV pages are local there
        const int Gc = cmg_threads(), ncmg = (omp_get_max_threads() + Gc - 1) / Gc;
        bool by_cmg = T > 1;
        for (int i = 0; i < T && by_cmg; ++i) by_cmg = E[size_t(i)].s->cmg >= 0 && E[size_t(i)].s->cmg < ncmg;
        // STRATA_DS41_PREFILL_ROWS=1: score adjacent prompt tokens by row within 4-token tiles.
        // A thread revisits nearby shared KV rows before moving to the next row; every dot keeps its own order.
        static const bool prefill_rows_env = strata_env_truthy_one("STRATA_DS41_PREFILL_ROWS");
        bool prefill_rows = prefill_rows_env && T >= 64 && !by_cmg;
        for (int i = 1; i < T && prefill_rows; ++i) prefill_rows = E[size_t(i)].s == E[0].s;
        const int ng = by_cmg ? ncmg : 1;
        std::vector<std::vector<int>> gid(static_cast<size_t>(ng)), grw(static_cast<size_t>(ng), std::vector<int>(1, 0));
        for (int i = 0; i < T; ++i) {
            const int g = by_cmg ? E[size_t(i)].s->cmg : 0;
            gid[size_t(g)].push_back(i);
            grw[size_t(g)].push_back(grw[size_t(g)].back() + roff[size_t(i) + 1] - roff[size_t(i)]);
        }
        // STRATA_DS41_MERGE (bits, default 63): 1 out_a's rows in the core's region (no all-reduce between them: one region
        // less per layer); 2 (passes of <= 4 tokens) each thread's running max per (token, head) in the scores phase, the
        // maxima taken over the threads' values after it (the max phase and its barrier go; a max is the same in any order)
        const int merge = merge_bits();
        const bool m_oa = (merge & 1) && !bal, m_mx = (merge & 2) && T <= 4, m_dh = (merge & 128) && hd % 64 == 0;
        // 256: (by CMG) the phases' barriers within each CMG's threads (the groups share no data until out_a)
        const bool m_cb = (merge & 256) && by_cmg;
        // 2048: the KV rows prefetched in the scores phase, the whole row kPfS rows ahead in the thread's list (pf1: in
        // the weighted sums too, 8 rows ahead, it cost 10 ms/step at B=128: off there)
        static const int kPfS = std::getenv("STRATA_DS41_CORE_PF") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_CORE_PF"))) : 4;   // (rows ahead)
        const bool m_pf = merge & 2048;
        auto core_bar = [&] { if (m_cb) cmg_barrier(); else team_barrier(); };
        // 8192: (by CMG, T > 4) the tokens in subgroups through every phase (sgn tokens; CMG-local barriers)
        static const int sg_env = std::getenv("STRATA_DS41_CORE_SG") ? std::max(1, std::atoi(std::getenv("STRATA_DS41_CORE_SG"))) : 4;
        const bool m_sg = (merge & 8192) && by_cmg && !m_mx;
        const int sgn = m_sg ? sg_env : T;
        auto sg_bar = [&] { if (m_sg) cmg_barrier(); else core_bar(); };
        double ph[4] = {0, 0, 0, 0};   // (thread 0) scores | max + exps | sums | weighted sums
        (void) m_dh;
        const int MT = omp_get_max_threads();
        thread_local Vec pmx;
        const size_t pms = (size_t(T) * nh + 63) / 64 * 64;   // a thread's maxima in their own 256-byte lines
        if (m_mx) pmx.resize(size_t(MT) * pms);
        const int G = c.out_groups, oar = G * c.out_rank;
        Vec OA_own, YB_own;   // (m_rw: the kept buffers, zeroed in the core's region)
        if (!m_rw) OA_own.assign(size_t(oar) * T, 0.f), YB_own.assign(size_t(n) * T, 0.f);
        else {
            if (core_oa_.size() < size_t(oar) * T) core_oa_.resize(size_t(oar) * T);
            if (core_yb_.size() < size_t(n) * T) core_yb_.resize(size_t(n) * T);
        }
        float *const OA = m_rw ? core_oa_.data() : OA_own.data(), *const YB = m_rw ? core_yb_.data() : YB_own.data();
        const size_t noa = size_t(oar) * T, nyb = size_t(n) * T;
        std::vector<GJob> oj;
        for (const auto &o : Y.out_a_t)
            oj.push_back({&o.m, ATTp + static_cast<size_t>(o.a - h0) * hd, OA + static_cast<size_t>(o.g) * c.out_rank, 0, o.m.rows, T,
                          size_t(nh) * hd, size_t(oar)});
        GPlan PO;
        if (m_oa) gplan(PO, oj.data(), int(oj.size()), nullptr, true);
        tm("core.prep") += now() - tcp;
        float *const pm_all = m_mx ? pmx.data() : nullptr;
        team_run([&] {
            const int t = omp_get_thread_num(), TT = omp_get_num_threads();
            const int g = by_cmg ? std::min(t / Gc, ng - 1) : 0, m0 = by_cmg ? g * Gc : 0, mc = by_cmg ? std::min(TT, m0 + Gc) - m0 : TT, li = t - m0;
            const std::vector<int> &ids = gid[size_t(g)], &gr = grw[size_t(g)];
            const int nk = int(ids.size());
            float *pm = m_mx ? pm_all + size_t(t) * pms : nullptr;
            if (pm)
                for (int i = 0; i < T; ++i)
                    for (int hh = 0; hh < nh; ++hh) pm[size_t(i) * nh + size_t(hh)] = Y.sinks[static_cast<size_t>(ha + hh)];
            if (m_rw) {   // out_a's and out_b's all-reduce buffers zeroed (each thread a slice)
                std::fill(OA + noa * size_t(t) / size_t(TT), OA + noa * size_t(t + 1) / size_t(TT), 0.f);
                std::fill(YB + nyb * size_t(t) / size_t(TT), YB + nyb * size_t(t + 1) / size_t(TT), 0.f);
            }
            // (STRATA_DS41_MERGE & 8192, by CMG) the group's tokens in subgroups of sgn (STRATA_DS41_CORE_SG, default 4), each
            // through all phases while its rows are in the CMG's L2; else one subgroup of all. Per element the same work. A
            // group without tokens still makes one (empty) pass: its threads meet every barrier of the phases
            double tp = t == 0 ? now() : 0;
            for (int s0 = 0; s0 < std::max(nk, 1); s0 += sgn) {
                const int s1 = std::min(nk, s0 + sgn), R0 = gr[size_t(s0)], R1 = gr[size_t(s1)];
                auto score_row = [&](int i, int r) {
                    const int nr = roff[size_t(i) + 1] - roff[size_t(i)];
                    const float *q = Q.data() + size_t(i) * nh * hd, *row = m_rw ? (RWF[roff[size_t(i)] + r] = row_of(i, r)) : RW[size_t(i)][size_t(r)];
                    float *sci = scp + size_t(nh) * roff[size_t(i)];
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                    if (m_dh) {   // (STRATA_DS41_MERGE & 128) all heads' dots of the row at once: dot()'s bits
                        float dv[8];
                        // (STRATA_DS41_DH_EXACT=1: the head count as the template's, no per-head test in the
                        // loop; each head's accumulators and order are the same)
                        static const bool dh_exact = strata_env_is_one("STRATA_DS41_DH_EXACT");
                        if (dh_exact && nhc <= 5) {
                            switch (nhc) {
                            case 1: dot_heads<1>(q, row, hd, dv); break;
                            case 2: dot_heads<2>(q, row, hd, dv); break;
                            case 3: dot_heads<3>(q, row, hd, dv); break;
                            case 4: dot_heads<4>(q, row, hd, dv); break;
                            default: dot_heads<5>(q, row, hd, dv); break;
                            }
                            for (int u = 0; u < nhc; ++u) sci[size_t(u) * nr + r] = dv[u] * scale;
                        } else
                        for (int h0b = 0; h0b < nhc; h0b += 6) {
                            const int k = std::min(6, nhc - h0b);
                            dot_heads<6>(q + size_t(h0b) * hd, row, hd, dv, k);
                            for (int u = 0; u < k; ++u) sci[size_t(h0b + u) * nr + r] = dv[u] * scale;
                        }
                    } else
#endif
                    for (int hh = 0; hh < nhc; ++hh) sci[static_cast<size_t>(hh) * nr + r] = dot(q + static_cast<size_t>(hh) * hd, row, hd) * scale;
                    if (pm)
                        for (int hh = 0; hh < nh; ++hh) pm[size_t(i) * nh + size_t(hh)] = std::max(pm[size_t(i) * nh + size_t(hh)], sci[static_cast<size_t>(hh) * nr + r]);
                };
                if (prefill_rows) {
                    constexpr int tile = 4;
                    for (int k0 = s0; k0 < s1; k0 += tile) {
                        const int k1 = std::min(s1, k0 + tile);
                        int max_rows = 0;
                        for (int k = k0; k < k1; ++k) {
                            const int i = ids[size_t(k)];
                            max_rows = std::max(max_rows, roff[size_t(i) + 1] - roff[size_t(i)]);
                        }
                        const int ra = int(int64_t(max_rows) * li / mc), rb = int(int64_t(max_rows) * (li + 1) / mc);
                        for (int r = ra; r < rb; ++r)
                            for (int k = k0; k < k1; ++k) {
                                const int i = ids[size_t(k)], nr = roff[size_t(i) + 1] - roff[size_t(i)];
                                if (r < nr) score_row(i, r);
                            }
                    }
                } else
                {   // the scores of the subgroup's rows
                    const int ga = R0 + int(int64_t(R1 - R0) * li / mc), gb = R0 + int(int64_t(R1 - R0) * (li + 1) / mc);
                    int k = s0;
                    int kp = k;
                    for (int gg = ga; gg < gb; ++gg) {
                        while (gg >= gr[size_t(k) + 1]) ++k;
                        const int i = ids[size_t(k)], r = gg - gr[size_t(k)];
                        if (m_pf && gg + kPfS < gb) {   // (2048) the row kPfS ahead, every 256-byte line
                            const int gp = gg + kPfS;
                            if (kp < k) kp = k;
                            while (gp >= gr[size_t(kp) + 1]) ++kp;
                            const int ip = ids[size_t(kp)], rp = gp - gr[size_t(kp)];
                            const float *prow = m_rw ? row_of(ip, rp) : RW[size_t(ip)][size_t(rp)];
                            for (int c = 0; c < hd; c += 64) __builtin_prefetch(prow + c, 0, 3);
                        }
                        score_row(i, r);
                    }
                }
                sg_bar();
                if (t == 0) ph[0] += now() - tp, tp = now();
                auto mx_of = [&](int i, int hh) {   // (m_mx) the max over the group's threads' running maxima
                    float mx = pm_all[size_t(m0) * pms + size_t(i) * nh + size_t(hh)];
                    for (int u = m0 + 1; u < m0 + mc; ++u) mx = std::max(mx, pm_all[size_t(u) * pms + size_t(i) * nh + size_t(hh)]);
                    return mx;
                };
                if (m_mx)
                    for (int q = s0 * nh + li; q < s1 * nh; q += mc) mxh[size_t(ids[size_t(q / nh)]) * nh + size_t(q % nh)] = mx_of(ids[size_t(q / nh)], q % nh);
                else {
                    for (int q = s0 * nh + li; q < s1 * nh; q += mc) {
                        const int i = ids[size_t(q / nh)], hh = q % nh, nr = roff[size_t(i) + 1] - roff[size_t(i)];
                        if (hh >= nhc) continue;
                        const float *sc0 = scp + size_t(nh) * roff[size_t(i)] + static_cast<size_t>(hh) * nr;
                        float mx = Y.sinks[static_cast<size_t>(ha + hh)];
                        for (int r = 0; r < nr; ++r) mx = std::max(mx, sc0[r]);
                        mxh[size_t(i) * nh + size_t(hh)] = mx;
                    }
                    sg_bar();
                }
                if (vexp_on()) {   // (STRATA_DS41_MERGE & 1024) the same, a (token, head)'s run of scores at a time
                    const int ne = nh * (R1 - R0), ka = nh * R0 + int(int64_t(ne) * li / mc), kb = nh * R0 + int(int64_t(ne) * (li + 1) / mc);
                    int k = s0;
                    for (int e2 = ka; e2 < kb;) {
                        while (e2 >= nh * gr[size_t(k) + 1]) ++k;
                        const int i = ids[size_t(k)], nr = roff[size_t(i) + 1] - roff[size_t(i)], loc = e2 - nh * gr[size_t(k)], hh = loc / nr;
                        const int end = std::min(kb, nh * gr[size_t(k)] + (hh + 1) * nr);
                        if (hh >= nhc) { e2 = end; continue; }
                        const float mx = m_mx ? mx_of(i, hh) : mxh[size_t(i) * nh + size_t(hh)];
                        float *v = scp + size_t(nh) * roff[size_t(i)] + size_t(loc);
                        for (int j = 0; j < end - e2; ++j) v[j] = v[j] - mx;
                        vexpf(v, v, end - e2);
                        e2 = end;
                    }
                } else {   // the exps of the group's scores (token k's nh x nr block)
                    const int ne = nh * (R1 - R0), ka = nh * R0 + int(int64_t(ne) * li / mc), kb = nh * R0 + int(int64_t(ne) * (li + 1) / mc);
                    int k = s0, ci = -1, ch = -1;
                    float cmx = 0;
                    for (int e2 = ka; e2 < kb; ++e2) {
                        while (e2 >= nh * gr[size_t(k) + 1]) ++k;
                        const int i = ids[size_t(k)], nr = roff[size_t(i) + 1] - roff[size_t(i)], loc = e2 - nh * gr[size_t(k)], hh = loc / nr;
                        if (hh >= nhc) continue;
                        float &v = scp[size_t(nh) * roff[size_t(i)] + size_t(loc)];
                        if (m_mx) {
                            if (i != ci || hh != ch) ci = i, ch = hh, cmx = mx_of(i, hh);
                            v = std::exp(v - cmx);
                        } else v = std::exp(v - mxh[size_t(i) * nh + size_t(hh)]);
                    }
                }
                sg_bar();
                if (t == 0) ph[1] += now() - tp, tp = now();
                for (int q = s0 * nh + li; q < s1 * nh; q += mc) {
                    const int i = ids[size_t(q / nh)], hh = q % nh, nr = roff[size_t(i) + 1] - roff[size_t(i)];
                    if (hh >= nhc) continue;
                    const float *sc0 = scp + size_t(nh) * roff[size_t(i)] + static_cast<size_t>(hh) * nr;
                    float sum = 0;
                    for (int r = 0; r < nr; ++r) sum += sc0[r];
                    inv[size_t(i) * nh + size_t(hh)] = 1.0f / (sum + std::exp(Y.sinks[static_cast<size_t>(ha + hh)] - mxh[size_t(i) * nh + size_t(hh)]));
                }
                sg_bar();
                if (t == 0) ph[2] += now() - tp, tp = now();
                const int nblk = hd / 16, items = (s1 - s0) * nblk;
                for (int it = s0 * nblk + int(int64_t(items) * li / mc); it < s0 * nblk + int(int64_t(items) * (li + 1) / mc); ++it) {
                    const int i = ids[size_t(it / nblk)], c0 = 16 * (it % nblk), nr = roff[size_t(i) + 1] - roff[size_t(i)];
                    const float *const *rows = m_rw ? RWF + roff[size_t(i)] : RW[size_t(i)].data();
                    const float *sci = scp + size_t(nh) * roff[size_t(i)];
                    float *att = ATTp + size_t(i) * nh * hd;
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                    switch (nhc) {
                    case 1: axpy_heads<1>(rows, sci, nr, c0, att, hd); break;
                    case 2: axpy_heads<2>(rows, sci, nr, c0, att, hd); break;
                    case 3: axpy_heads<3>(rows, sci, nr, c0, att, hd); break;
                    case 4: axpy_heads<4>(rows, sci, nr, c0, att, hd); break;
                    case 5: axpy_heads<5>(rows, sci, nr, c0, att, hd); break;
                    case 6: axpy_heads<6>(rows, sci, nr, c0, att, hd); break;
                    case 7: axpy_heads<7>(rows, sci, nr, c0, att, hd); break;
                    default:
                        for (int h0b = 0; h0b < nhc; h0b += 8) axpy_heads<8>(rows, sci + size_t(h0b) * nr, nr, c0, att + size_t(h0b) * hd, hd, std::min(8, nhc - h0b));
                    }
#else
                    for (int hh = 0; hh < nhc; ++hh) {
                        float *o = att + static_cast<size_t>(hh) * hd + c0;
                        for (int k = 0; k < 16; ++k) o[k] = 0;
                        for (int r = 0; r < nr; ++r)
                            for (int k = 0; k < 16; ++k) o[k] += sci[static_cast<size_t>(hh) * nr + r] * rows[r][c0 + k];
                    }
#endif
                    const float *tb = E[size_t(i)].table(back);
                    for (int hh = 0; hh < nhc; ++hh) {
                        float *o = att + static_cast<size_t>(hh) * hd + c0;
                        for (int k = 0; k < 16; ++k) o[k] *= inv[size_t(i) * nh + size_t(hh)];
                        if (c0 >= nope) rope(o, 8, tb + (c0 - nope));
                    }
                }
                if (t == 0) ph[3] += now() - tp, tp = now();
            }
            if (t == 0) mk[3] = now();
            if (m_oa) {   // out_a's rows over the whole team once ATT is complete
                team_barrier();
                if (t == 0) mk[5] = now();
                gbody(PO, t, TT);
            }
        });
        mk[4] = now();
        if (!m_oa) mk[5] = mk[4];
        tm("core.scores") += ph[0], tm("core.exp") += ph[1], tm("core.sum") += ph[2], tm("core.axpy") += ph[3] + (mk[5] - mk[3]);
        if (m_oa) gstat("out_a", 0, mk[4] - mk[5], double(PO.B));
        if (bal) {   // (STRATA_DS41_HEAD_BAL) the shared head's other parts, then the rows to the ext ranks
            const double tb0 = now();
            const int W = comm.world, me = comm.rank;
            // one head over a list of tokens, with the core's per-element operations, as one team region: the scores by
            // rows over the team | per token the sink-started max, the exps, the sum in row order, the inverse | the weighted
            // sums by (token, 16 columns) in row order, the scale, the rope pairs
            auto heads1 = [&](int t0, int nt, int hg, const float *qs, size_t ldq, float *os, size_t ldo) {
                if (nt <= 0) return;
                std::vector<int> ro(static_cast<size_t>(nt) + 1, 0);
                for (int k = 0; k < nt; ++k) ro[size_t(k) + 1] = ro[size_t(k)] + roff[size_t(t0 + k) + 1] - roff[size_t(t0 + k)];
                std::vector<float> sc(static_cast<size_t>(ro[size_t(nt)])), iv(static_cast<size_t>(nt));
                auto rows_of = [&](int i) { return m_rw ? RWF + roff[size_t(i)] : RW[size_t(i)].data(); };
                const int nblk = hd / 16, NRt = ro[size_t(nt)];
                team_run([&] {
                    const int t = omp_get_thread_num(), TT = omp_get_num_threads();
                    int k = 0;
                    for (int g = int(int64_t(NRt) * t / TT); g < int(int64_t(NRt) * (t + 1) / TT); ++g) {
                        while (g >= ro[size_t(k) + 1]) ++k;
                        const int r = g - ro[size_t(k)];
                        const float *q = qs + size_t(k) * ldq, *row = rows_of(t0 + k)[r];
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                        if (m_dh) { float dv[8]; dot_heads<6>(q, row, hd, dv, 1); sc[size_t(g)] = dv[0] * scale; } else
#endif
                        sc[size_t(g)] = dot(q, row, hd) * scale;
                    }
                    team_barrier();
                    for (int kk = t; kk < nt; kk += TT) {
                        float *v = sc.data() + ro[size_t(kk)];
                        const int nr = ro[size_t(kk) + 1] - ro[size_t(kk)];
                        float mx = Y.sinks[size_t(hg)];
                        for (int r = 0; r < nr; ++r) mx = std::max(mx, v[r]);
                        if (vexp_on()) { for (int r = 0; r < nr; ++r) v[r] = v[r] - mx; vexpf(v, v, nr); }
                        else for (int r = 0; r < nr; ++r) v[r] = std::exp(v[r] - mx);
                        float sum = 0;
                        for (int r = 0; r < nr; ++r) sum += v[r];
                        iv[size_t(kk)] = 1.0f / (sum + std::exp(Y.sinks[size_t(hg)] - mx));
                    }
                    team_barrier();
                    for (int it = int(int64_t(nt) * nblk * t / TT); it < int(int64_t(nt) * nblk * (t + 1) / TT); ++it) {
                        const int kk = it / nblk, c0 = 16 * (it % nblk), i = t0 + kk, nr = ro[size_t(kk) + 1] - ro[size_t(kk)];
                        const float *const *rows = rows_of(i);
                        const float *w = sc.data() + ro[size_t(kk)];
                        float *o = os + size_t(kk) * ldo;
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                        axpy_heads<1>(rows, w, nr, c0, o, hd);
#else
                        for (int j = 0; j < 16; ++j) o[c0 + j] = 0;
                        for (int r = 0; r < nr; ++r)
                            for (int j = 0; j < 16; ++j) o[c0 + j] += w[r] * rows[r][c0 + j];
#endif
                        for (int j = 0; j < 16; ++j) o[c0 + j] *= iv[size_t(kk)];
                        if (c0 >= nope) rope(o + c0, 8, E[size_t(i)].table(back) + (c0 - nope));
                    }
                });
            };
            Vec AX;
            if (brole == 1)   // part 0 into the last head's slot
                heads1(0, bpart_lo(1), ha + nh - 1, Q.data() + size_t(nh - 1) * hd, size_t(nh) * hd, ATTp + size_t(nh - 1) * hd, size_t(nh) * hd);
            else if (brole == 2 && bt1 > bt0) {
                AX.resize(size_t(bt1 - bt0) * hd);
                heads1(bt0, bt1 - bt0, bal_head(bal_home_[size_t(me)]), QX.data(), size_t(hd), AX.data(), size_t(hd));
            }
            int soff[65], roff2[65];
            soff[0] = roff2[0] = 0;
            for (int r = 0; r < W; ++r) {
                const int ns = brole == 2 && r == bal_home_[size_t(me)] ? (bt1 - bt0) * hd : 0;
                int nrv = 0;
                if (brole == 1 && bal_role_[size_t(r)] == 2 && bal_home_[size_t(r)] == me) nrv = (bpart_lo(bal_part_[size_t(r)] + 1) - bpart_lo(bal_part_[size_t(r)])) * hd;
                soff[r + 1] = soff[r] + ns, roff2[r + 1] = roff2[r] + nrv;
            }
            Vec RX(size_t(roff2[W]) + 1);
            AX.resize(AX.size() + 1);
            const float *const sp = AX.data();
            float *const rp = RX.data();
            cdo([&, sp, rp] { comm.alltoall(sp, soff, rp, roff2); });
            if (brole == 1)
                for (int r = 0; r < W; ++r) {
                    if (roff2[r + 1] == roff2[r]) continue;
                    const int k0 = bpart_lo(bal_part_[size_t(r)]), k1 = bpart_lo(bal_part_[size_t(r)] + 1);
                    for (int i = k0; i < k1; ++i)
                        std::copy_n(RX.data() + roff2[r] + size_t(i - k0) * hd, hd, ATTp + size_t(i) * nh * hd + size_t(nh - 1) * hd);
                }
            tm("core.bal") += now() - tb0;
        }
        lap("attn.core");
        // grouped output projection: this rank's heads' share of each group's rows (a partial sum), then out_b's rows;
        // both summed over the ranks
        if (!m_oa) gemv_multi(oj.data(), int(oj.size()), "out_a");
        if (es_on_ && Y.out_b_full.p) {
            const double t0 = now();
            const int W = comm.world, own = es_b_ - es_a_;
            int off[65];
            for (int r = 0; r <= W; ++r) off[r] = es_off_[r] * oar;
            oa_own_.resize(size_t(own) * oar), yb_own_.resize(size_t(own) * n);
            float *const op = oa_own_.data();
            // STRATA_DS41_ES_OASP=1: rank s's out_a partial is +0.0 outside the groups its heads touch ([glo[s], ghi[s]]); each
            // rank sends each owner only those groups' columns (an all-to-all, the bits as sent) and the owner sums every
            // element from +0.0f over the contributing ranks in rank order: the reduce-scatter's sum (adding a rank's +0.0
            // leaves the running sum, which is never -0.0, unchanged)
            static const bool oasp = strata_env_is_one("STRATA_DS41_ES_OASP");
            const int Rk = c.out_rank, Hh = c.heads, hpg = Hh / G;
            int glo[64], ghi[64], soff[65], roff[65];
            bool sp = oasp && W <= 64 && G * hpg == Hh;
            if (sp) {
                for (int r = 0; r < W; ++r) glo[r] = (Hh * r / W) / hpg, ghi[r] = (Hh * (r + 1) / W - 1) / hpg;
                soff[0] = roff[0] = 0;
                for (int r = 0; r < W; ++r) {
                    soff[r + 1] = soff[r] + (es_off_[r + 1] - es_off_[r]) * (ghi[comm.rank] - glo[comm.rank] + 1) * Rk;
                    roff[r + 1] = roff[r] + own * (ghi[r] - glo[r] + 1) * Rk;
                    sp = sp && soff[r + 1] - soff[r] <= comm.xc_cap() && roff[r + 1] - roff[r] <= comm.xc_cap();
                }
            }
            if (sp) {
                const int cme = (ghi[comm.rank] - glo[comm.rank] + 1) * Rk;
                Vec &S = a2a_s_, &Rv = a2a_r_;
                S.resize(size_t(soff[W])), Rv.resize(size_t(roff[W]));
                for (int i = 0; i < T; ++i) std::copy_n(OA + size_t(i) * oar + size_t(glo[comm.rank]) * Rk, cme, S.data() + size_t(i) * cme);
                const float *const sp_ = S.data();
                float *const rp_ = Rv.data();
                cdo([&, sp_, rp_] { comm.alltoall(sp_, soff, rp_, roff); });
                team_for(own * G, [&](int a, int b) {
                    for (int q = a; q < b; ++q) {
                        const int i = q / G, g = q % G;
                        float *o = op + size_t(i) * oar + size_t(g) * Rk;
                        for (int j = 0; j < Rk; ++j) o[j] = 0.0f;
                        for (int r = 0; r < W; ++r) {
                            if (g < glo[r] || g > ghi[r]) continue;
                            const int cr = (ghi[r] - glo[r] + 1) * Rk;
                            const float *src = Rv.data() + roff[r] + size_t(i) * cr + size_t(g - glo[r]) * Rk;
                            for (int j = 0; j < Rk; ++j) o[j] += src[j];
                        }
                    }
                });
            } else cdo([&, op] { comm.reduce_scatter(OA, off, op); });
            const double dt = now() - t0;
            tm("comm") += dt, tm("comm.oa") += dt;
            const GJob ob{&Y.out_b_full, oa_own_.data(), yb_own_.data(), 0, Y.out_b_full.rows, own, size_t(oar), size_t(n)};
            gemv_multi(&ob, 1, "out_b");
            for (int i = es_a_; i < es_b_; ++i) {
                const float *src = yb_own_.data() + size_t(i - es_a_) * n;
                float *yi = E[size_t(i)].y.data();
                for (int j = 0; j < n; ++j) yi[j] = src[j] + 0.0f;
            }
            lap("attn.proj_out");
            return;
        }
        allreduce_timed(OA, int(noa), "comm.oa");
        const GJob ob{&Y.out_b_t, OA, YB + y0, 0, Y.out_b_t.rows, T, size_t(oar), size_t(n)};
        gemv_multi(&ob, 1, "out_b");
        if (es_on_) {   // each owner its entries' rows from every rank (+ 0.0f: the all-reduce's bits)
            slices_to_owners(YB, size_t(n), n, T, [&](int i) { return E[size_t(i)].y.data(); }, "comm.attn_y");
        } else {
            if ((ag_bits() & 2) && comm.xc) {
                const int sec[1][2] = {{0, n}};
                gather_slices(YB, T, size_t(n), sec, 1, "comm.attn_y");
            } else allreduce_timed(YB, int(nyb), "comm.attn_y");
            for (int i = 0; i < T; ++i) std::copy(YB + size_t(i) * n, YB + size_t(i + 1) * n, E[size_t(i)].y.begin());
        }
        lap("attn.proj_out");
    }

    // the layer's routed and shared chunks of this rank in one team region. Each token routes its own experts; a chunk
    // routed by several tokens runs once for all of them (nt = its tokens). xn of every token quantized; every gate/up
    // row of the chunks split over the threads (by rows x tokens); SwiGLU and the Q8_K form of each chunk-token's 256
    // values; then each thread a block of the hidden rows over every chunk's down, each token's sum in its own job order
    // (the same sums for any thread count and any pass)
    struct FJob { const Chunk *k; float w, clamp; };
    std::vector<ds41_q8k> xq_pre_;
    std::vector<ds41_q8k> xqr_;   // (STRATA_DS41_ES_QKQ) q_a's normed Q8_K blocks per entry
    struct Use { int u, slot; float w; };
    struct Job { int id; float w, clamp; };
    struct RouteBufs {   // ffn's routing lists (Utok may hold more entries than U: only the first U.size() are used)
        std::vector<const Chunk *> U;
        std::vector<float> Uclamp;
        std::vector<std::vector<int>> Utok;
        std::vector<std::vector<Use>> jobs;
        std::vector<std::vector<Job>> tj;
        std::vector<double> bal;
        std::vector<int> uid;
        void reset(int T) {
            U.clear(), Uclamp.clear();
            if (int(jobs.size()) < T) jobs.resize(size_t(T)), tj.resize(size_t(T));
            for (int i = 0; i < T; ++i) jobs[size_t(i)].clear(), tj[size_t(i)].clear();
            bal.assign(size_t(T) * 3, 0.0);
        }
    };
    RouteBufs route_bufs_;
    Vec core_att_, core_sc_;   // (STRATA_DS41_MERGE & 32) the core's buffers across layers
    std::vector<std::vector<const float *>> core_rw_;   // (STRATA_DS41_MERGE & 8) xn's Q8_K blocks, quantized in the router's region
    // (STRATA_DS41_MERGE & 64) the core's row pointers (all tokens', flat, written in the scores phase) and out_a / out_b's
    // all-reduce buffers (zeroed in the core's region), kept across layers (grown only, never initialized whole)
    std::unique_ptr<const float *[]> core_rwf_;
    size_t core_rwf_cap_ = 0;
    Vec core_oa_, core_yb_;
    // ---- STRATA_DS41_BATCH_FIBERS: a batched step as two half-batch passes, each a fiber (ucontext) on the team's
    // thread 0. At a collective a fiber posts it to the comm thread (Comm::post_fn, in posting order) and switches to the
    // other, then waits for its ticket: one half's exchanges overlap the other half's compute. The fibers switch at the
    // same points on every rank, so every rank posts the same sequence; each entry's operations are a pass's. The pass
    // scratch that lives across a collective (below) is swapped per fiber at each switch (vector swaps)
    struct PassScratch {
        Vec pX, agbuf, a2a_s, a2a_r, oa_own, yb_own, core_att, core_sc, core_oa, core_yb, flat;
        std::vector<std::vector<const float *>> core_rw;
        std::unique_ptr<const float *[]> core_rwf;
        size_t core_rwf_cap = 0;
        RouteBufs rb;
        std::vector<ds41_q8k> xq_pre;
        bool es_on = false;
        int es_a = 0, es_b = 0, es_off[65] = {0};
        std::vector<std::unique_ptr<EgFetch>> egf;
    };
    void swap_scratch(PassScratch &q) {
        std::swap(pX_, q.pX), std::swap(agbuf_, q.agbuf), std::swap(a2a_s_, q.a2a_s), std::swap(a2a_r_, q.a2a_r);
        std::swap(oa_own_, q.oa_own), std::swap(yb_own_, q.yb_own), std::swap(core_att_, q.core_att), std::swap(core_sc_, q.core_sc);
        std::swap(core_oa_, q.core_oa), std::swap(core_yb_, q.core_yb), std::swap(flat_, q.flat), std::swap(core_rw_, q.core_rw);
        std::swap(core_rwf_, q.core_rwf), std::swap(core_rwf_cap_, q.core_rwf_cap), std::swap(route_bufs_, q.rb), std::swap(xq_pre_, q.xq_pre);
        std::swap(es_on_, q.es_on), std::swap(es_a_, q.es_a), std::swap(es_b_, q.es_b), std::swap(es_off_, q.es_off), std::swap(egf_, q.egf);
    }
#if defined(__linux__)
    struct Fib { ucontext_t ctx; std::vector<char> stack; PassScratch st; bool done = true; std::function<void()> body; std::exception_ptr err; };
    Fib fib_[2];
    ucontext_t fib_main_;
    int fib_cur_ = -1;   // the running fiber (-1: none)
    static void fib_entry(int hi, int lo) {
        Runner *r = reinterpret_cast<Runner *>((uintptr_t(uint32_t(hi)) << 32) | uintptr_t(uint32_t(lo)));
        Fib &f = r->fib_[r->fib_cur_];
        try { f.body(); } catch (...) { f.err = std::current_exception(); }
        r->fib_[r->fib_cur_].done = true;   // (uc_link: back to fib_pair)
    }
    void fib_switch(int from, int to, ucontext_t *save) {
        swap_scratch(fib_[from].st), swap_scratch(fib_[to].st);
        fib_cur_ = to;
        swapcontext(save, &fib_[to].ctx);
    }
    void fib_yield() {
        if (fib_cur_ < 0) return;
        const int me = fib_cur_, o = 1 - me;
        if (fib_[o].done) return;
        fib_switch(me, o, &fib_[me].ctx);
    }
    // a and b as two fibers until both have ended
    void fib_pair(std::function<void()> a, std::function<void()> b) {
        const uintptr_t self = reinterpret_cast<uintptr_t>(this);
        for (int k = 0; k < 2; ++k) {
            Fib &f = fib_[k];
            f.body = k ? std::move(b) : std::move(a), f.done = false, f.err = nullptr;
            if (f.stack.empty()) f.stack.resize(size_t(64) << 20);
            getcontext(&f.ctx);
            f.ctx.uc_stack.ss_sp = f.stack.data(), f.ctx.uc_stack.ss_size = f.stack.size(), f.ctx.uc_link = &fib_main_;
            makecontext(&f.ctx, reinterpret_cast<void (*)()>(&Runner::fib_entry), 2, int(uint32_t(self >> 32)), int(uint32_t(self)));
        }
        fib_cur_ = 0;
        swapcontext(&fib_main_, &fib_[0].ctx);
        while (!(fib_[0].done && fib_[1].done)) {   // the fiber that ended is fib_cur_: the other continues
            const int e = fib_cur_;
            fib_switch(e, 1 - e, &fib_main_);
        }
        fib_cur_ = -1;
        for (auto &f : fib_)
            if (f.err) std::rethrow_exception(f.err);
    }
#else   // (no ucontext: the halves in turn)
    int fib_cur_ = -1;
    void fib_yield() {}
    void fib_pair(std::function<void()> a, std::function<void()> b) { a(), b(); }
#endif
    // a collective: run now, or (in a fiber) posted to the comm thread while the other fiber runs. op captures pointers
    // and values only (the scratch members are the other fiber's while it runs)
    // (fiber stats, RUNNER_DS41_FIB: posts, posts whose other fiber had ended (no overlap), thread 0's time from the
    // switch back to the ticket's completion, and from the post to the switch back (the other fiber's run))
    double fib_posts_ = 0, fib_alone_ = 0, fib_wait_s_ = 0, fib_away_s_ = 0;
    template <class F> void cdo(F &&op) {
        if (fib_cur_ < 0) { op(); return; }
        std::function<void()> fn(std::forward<F>(op));
        const double t0 = now();
        const long t = comm.post_fn(&fn);
        fib_posts_ += 1;
#if defined(__linux__)
        if (fib_[1 - fib_cur_].done) fib_alone_ += 1;
#endif
        fib_yield();
        const double t1 = now();
        comm.ar_wait(t);
        fib_away_s_ += t1 - t0, fib_wait_s_ += now() - t1;
    }
    void ffn(int l, std::vector<Ent> &E) {
        const Layer &Y = L[size_t(l)];
        const int Ex = c.experts, K = c.top_k, H = c.hidden, T = int(E.size());
        double ta = now();
        sub_begin();
        const float *X = gather_xn(E);
        Vec LG(size_t(Ex) * T), PR(size_t(Ex) * T), SL(size_t(Ex) * T);
        bool xq_done = false;   // xq_pre_ holds this pass's xn (STRATA_DS41_MERGE & 8)
        {   // the router, the scores of each thread's experts right after their logits
            const std::function<void(int, int, int)> score = [&](int, int e0, int e1) {
                for (int i = es_a_; i < es_b_; ++i) {   // (the own entries: all without ENTRY_SHARD)
                    const float *lg = LG.data() + size_t(i) * Ex;
                    float *pr = PR.data() + size_t(i) * Ex, *sl = SL.data() + size_t(i) * Ex;
                    for (int e = e0; e < e1; ++e) pr[e] = std::sqrt(softplus(lg[e])), sl[e] = pr[e] + Y.bias[size_t(e)];
                }
            };
            if (Y.gate_inp_t.p) {
                Vec PS(size_t(2 * Ex) * T, 0.f);   // per token: probs, then selection scores
                const std::function<void(int, int, int)> score_t = [&](int, int r0, int r1) {
                    for (int i = 0; i < T; ++i) {
                        const float *lg = LG.data() + size_t(i) * Ex;
                        float *pr = PS.data() + size_t(i) * 2 * Ex, *sl = pr + Ex;
                        for (int e = Y.ge0 + r0; e < Y.ge0 + r1; ++e) pr[e] = std::sqrt(softplus(lg[e])), sl[e] = pr[e] + Y.bias[size_t(e)];
                    }
                };
                const GJob rj{&Y.gate_inp_t, X, LG.data() + Y.ge0, 0, Y.gate_inp_t.rows, T, size_t(H), size_t(Ex)};
                gemv_multi(&rj, 1, "router", &score_t);
                allreduce_timed(PS.data(), int(PS.size()), "comm.route");
                for (int i = 0; i < T; ++i) {
                    std::copy_n(PS.data() + size_t(i) * 2 * Ex, Ex, PR.data() + size_t(i) * Ex);
                    std::copy_n(PS.data() + size_t(i) * 2 * Ex + Ex, Ex, SL.data() + size_t(i) * Ex);
                }
            } else {
                const GJob rj{&Y.gate_inp, X + size_t(es_a_) * H, LG.data() + size_t(es_a_) * Ex, 0, Y.gate_inp.rows, es_b_ - es_a_, size_t(H), size_t(Ex)};
                if ((merge_bits() & 8) && H % 256 == 0 && !es_on_) {   // (ENTRY_SHARD: xn is the own rows' only here)   // the team quantizes xn (Q8_K) after its router rows: the
                    // region's end barrier covers it (the experts region then starts without its quantization barrier)
                    const double t0 = now();
                    GPlan P;
                    gplan(P, &rj, 1, &score, false);
                    const int nb = H / 256;
                    xq_pre_.resize(size_t(nb) * T);
                    const double t1 = now();
                    team_run([&] {
                        const int t = omp_get_thread_num(), TT = omp_get_num_threads();
                        gbody(P, t, TT);
                        for (int b = int(int64_t(nb) * T * t / TT); b < int(int64_t(nb) * T * (t + 1) / TT); ++b) ds41_quant_q8k(X + 256 * size_t(b), 256, &xq_pre_[size_t(b)]);
                    });
                    gstat("router", t1 - t0, now() - t1, double(P.B));
                    xq_done = true;
                } else gemv_multi(&rj, 1, "router", &score);
            }
        }
        tm("ffn.route.gate") += now() - ta;
        sub("ffn.route.gate");
        const double tu0 = now();
        // per token: its jobs in order (routed by k, each expert's chunks of this rank; then the shared chunks the
        // balance gives this rank); the distinct chunks (U) with their tokens
        // (STRATA_DS41_MERGE & 32: these lists kept across layers, cleared with their capacity)
        const bool reuse = merge_bits() & 32;
        RouteBufs rb_own;
        RouteBufs &RB = reuse ? route_bufs_ : rb_own;
        RB.reset(T);
        std::vector<const Chunk *> &U = RB.U;
        std::vector<float> &Uclamp = RB.Uclamp;
        std::vector<std::vector<int>> &Utok = RB.Utok;   // per distinct chunk: its tokens, in pass order
        std::vector<std::vector<Use>> &jobs = RB.jobs;
        // per token (over the team when there are several): its picks, weights and this rank's jobs in order, as
        // chunk ids (routed: the index in Y.chunks; shared: after them); then merged in token order
        std::vector<std::vector<Job>> &tj = RB.tj;
        std::vector<double> &bal = RB.bal;
        const int nrout = int(Y.chunks.size());
        tm("route.setup") += now() - tu0;
        double tr0 = now();
        double rt[3] = {0, 0, 0};   // (one token) after the pick, after the jobs, after the balance
        // (STRATA_DS41_ENTRY_SHARD: the owner picks and weighs its entries' experts (pick), every rank then makes every
        // entry's jobs from the gathered picks (route))
        std::vector<int> pk_own;
        std::vector<float> wt_own;
        if (es_on_) pk_own.resize(size_t(T) * K), wt_own.resize(size_t(T) * K);
        auto pick_w = [&](int i, int *pk, float *wt) {   // entry i's experts and weights
            const float *probs = PR.data() + size_t(i) * Ex;
#if defined(__aarch64__)
            if (merge_bits() & 32)   // the scores were written by other threads (other CMGs): their lines requested at once
                for (int e = 0; e < Ex; e += 64)
                    __asm__ volatile("prfm pldl1keep, [%0]\n\tprfm pldl1keep, [%1]" ::"r"(SL.data() + size_t(i) * Ex + e), "r"(probs + e));
#endif
            const auto pick = (merge_bits() & 32) ? top_k_small_skip(SL.data() + size_t(i) * Ex, Ex, K) : top_k_small(SL.data() + size_t(i) * Ex, Ex, K);
            if (T == 1) rt[0] = now();
            float sum = 0;
            for (int k = 0; k < K; ++k) sum += probs[size_t(pick[size_t(k)])];
            for (int k = 0; k < K; ++k) {
                float w = probs[size_t(pick[size_t(k)])];
                if (c.normalize && K > 1) w /= sum + 1e-20f;
                w *= c.expert_scale;
                pk[k] = pick[size_t(k)], wt[k] = w;
            }
        };
        auto route = [&](int i) {
            int pkb[64];
            float wtb[64];
            const int *pick = pkb;
            const float *wts = wtb;
            if (es_on_) pick = pk_own.data() + size_t(i) * K, wts = wt_own.data() + size_t(i) * K;
            else pick_w(i, pkb, wtb);
            auto &L2 = tj[size_t(i)];
            for (int k = 0; k < K; ++k) {
                const float w = wts[k];
                for (int ch = 0; ch < nchunk; ++ch)
                    if (const int q = Y.chunk_of[size_t(pick[size_t(k)]) * nchunk + ch]; q >= 0) L2.push_back({q, w, c.clamp[size_t(l)]});
            }
            if (T == 1) rt[1] = now();
            // STRATA_DS41_SHARED_WEIGHT (1): a shared chunk's cost in routed chunks for the balance (Q5_K shared chunks
            // run slower than the IQ2R routed ones; another weight moves them, so the sums' grouping and the bits change)
            static const float shw = std::getenv("STRATA_DS41_SHARED_WEIGHT") ? float(std::atof(std::getenv("STRATA_DS41_SHARED_WEIGHT"))) : 1.0f;
            float load[64] = {0};
            const int Wd = comm.world;
            for (int k = 0; k < K; ++k)
                for (int ch = 0; ch < nchunk; ++ch) load[(9 * pick[size_t(k)] + ch) % Wd] += 1.0f;
            bal[size_t(i) * 3] = *std::max_element(load, load + Wd);
            if (!Y.shared_all.empty()) {
                for (int ch = 0; ch < nchunk; ++ch) {
                    const int r = int(std::min_element(load, load + Wd) - load);
                    load[r] += shw;
                    if (r == comm.rank) L2.push_back({nrout + ch, 1.0f, c.shared_clamp[size_t(l)]});
                }
            } else {
                for (int ch = 0; ch < nchunk; ++ch) load[ch % Wd] += shw;
                for (int k = 0; k < int(Y.shared.size()); ++k) L2.push_back({nrout + k, 1.0f, c.shared_clamp[size_t(l)]});
            }
            bal[size_t(i) * 3 + 1] = *std::max_element(load, load + Wd);
            bal[size_t(i) * 3 + 2] = double(K * nchunk + nchunk) / Wd;
        };
        if (comm.world > 64) throw std::runtime_error("ds41: at most 64 ranks");
        if (K > 64) throw std::runtime_error("ds41: at most 64 experts per token");
        if (es_on_) {   // the own entries' picks and weights, then every entry's [xn | ids' bits | weights] gathered (the
            // bits as sent): X (pX_) and the entries' xn complete for the expert chunks
            team_for(es_b_ - es_a_, [&](int a, int b) { for (int i = es_a_ + a; i < es_a_ + b; ++i) pick_w(i, pk_own.data() + size_t(i) * K, wt_own.data() + size_t(i) * K); });
            const double tg = now();
            const int W = comm.world;
            if (es_xq_layer(l)) {   // (STRATA_DS41_ES_XQ) [xn's Q8_K blocks | ids' bits | weights]: xq_pre_ for the experts
                constexpr int qw = 73;   // a block's q, bsum and d: 292 bytes
                static_assert(offsetof(ds41_q8k, d) + sizeof(float) == 4 * qw, "ds41_q8k layout");
                const int nb = H / 256, wq = nb * qw + 2 * K;
                xq_pre_.resize(size_t(nb) * T);
                Vec &G = agbuf_;
                G.resize(size_t(T) * wq);
                team_for((es_b_ - es_a_) * nb, [&](int a, int b) {
                    for (int q = a; q < b; ++q) {
                        const int i = es_a_ + q / nb, k = q % nb;
                        ds41_q8k &blk = xq_pre_[size_t(i) * nb + size_t(k)];
                        ds41_quant_q8k(X + size_t(i) * H + 256 * size_t(k), 256, &blk);
                        std::memcpy(G.data() + size_t(i) * wq + size_t(k) * qw, &blk, 4 * qw);
                    }
                });
                for (int i = es_a_; i < es_b_; ++i) {
                    float *g = G.data() + size_t(i) * wq + size_t(nb) * qw;
                    for (int k = 0; k < K; ++k) std::memcpy(g + k, &pk_own[size_t(i) * K + k], 4), g[K + k] = wt_own[size_t(i) * K + k];
                }
                int off[65];
                for (int r = 0; r <= W; ++r) off[r] = es_off_[r] * wq;
                float *const gp = G.data();
                cdo([&, gp] { comm.allgather_shards(gp, off, false); });
                for (int i = 0; i < T; ++i) {
                    const float *g = G.data() + size_t(i) * wq;
                    if (i < es_a_ || i >= es_b_)
                        for (int k = 0; k < nb; ++k) std::memcpy(&xq_pre_[size_t(i) * nb + size_t(k)], g + size_t(k) * qw, 4 * qw);
                    for (int k = 0; k < K; ++k) std::memcpy(&pk_own[size_t(i) * K + k], g + size_t(nb) * qw + k, 4), wt_own[size_t(i) * K + k] = g[size_t(nb) * qw + K + k];
                }
                xq_done = true;
                const double dt = now() - tg;
                tm("comm") += dt, tm("comm.xn_ffn") += dt;
            } else {
                const int wd = H + 2 * K;
                Vec &G = agbuf_;
                G.resize(size_t(T) * wd);
                for (int i = es_a_; i < es_b_; ++i) {
                    float *g = G.data() + size_t(i) * wd;
                    std::copy(X + size_t(i) * H, X + size_t(i + 1) * H, g);
                    for (int k = 0; k < K; ++k) std::memcpy(g + H + k, &pk_own[size_t(i) * K + k], 4), g[H + K + k] = wt_own[size_t(i) * K + k];
                }
                int off[65];
                for (int r = 0; r <= W; ++r) off[r] = es_off_[r] * wd;
                float *const gp = G.data();
                cdo([&, gp] { comm.allgather_shards(gp, off, false); });
                for (int i = 0; i < T; ++i) {
                    const float *g = G.data() + size_t(i) * wd;
                    if (i < es_a_ || i >= es_b_) {
                        std::copy(g, g + H, pX_.begin() + size_t(i) * H);
                        std::copy(g, g + H, E[size_t(i)].xn.begin());
                    }
                    for (int k = 0; k < K; ++k) std::memcpy(&pk_own[size_t(i) * K + k], g + H + k, 4), wt_own[size_t(i) * K + k] = g[H + K + k];
                }
                const double dt = now() - tg;
                tm("comm") += dt, tm("comm.xn_ffn") += dt;
            }
        }
        if (T == 1) route(0);
        else team_for(T, [&](int a, int b) { for (int i = a; i < b; ++i) route(i); });
        tm("route.pick") += now() - tr0;
        if (T == 1) tm("route.p.topk") += rt[0] - tr0, tm("route.p.jobs") += rt[1] - rt[0], tm("route.p.bal") += now() - rt[1];
        {
            std::vector<int> &uid = RB.uid;
            uid.assign(size_t(nrout) + size_t(std::max(nchunk, int(Y.shared.size()))), -1);
            for (int i = 0; i < T; ++i) {
                for (const Job &j : tj[size_t(i)]) {
                    int &u = uid[size_t(j.id)];
                    if (u < 0) {
                        u = int(U.size());
                        U.push_back(j.id < nrout ? &Y.chunks[size_t(j.id)] : (!Y.shared_all.empty() ? &Y.shared_all[size_t(j.id - nrout)] : &Y.shared[size_t(j.id - nrout)]));
                        Uclamp.push_back(j.clamp);
                        if (Utok.size() < U.size()) Utok.emplace_back();
                        Utok[U.size() - 1].clear();
                    }
                    Utok[size_t(u)].push_back(i);
                    jobs[size_t(i)].push_back({u, int(Utok[size_t(u)].size()) - 1, j.w});
                }
                bal_routed += bal[size_t(i) * 3], bal_total += bal[size_t(i) * 3 + 1], bal_mean += bal[size_t(i) * 3 + 2], ++bal_n;
            }
        }
        const double trk = 0;
        tm("route.topk") += trk, tm("route.jobs") += now() - tr0 - trk;
        tm("ffn.route") += now() - ta, ta = now();
        sub("ffn.route.rest");
        Vec YB(size_t(H) * T, 0.f);
        const int nu = int(U.size());
        std::vector<int> ubase(size_t(nu) + 1, 0);   // a chunk-token's index: ubase[u] + slot
        for (int u = 0; u < nu; ++u) ubase[size_t(u) + 1] = ubase[size_t(u)] + int(Utok[size_t(u)].size());
        const int nuse = ubase[size_t(nu)];
        if (nuse) {
            // each chunk's activations by its type: Q8_K (the target's IQ / Q5_K chunks), Q8_0 (MXFP4: the drafter's
            // experts) or the floats (BF16: the drafter's shared expert); a form is made only when a chunk takes it
            bool need_k = false, need_z = false, need_hk = false, need_hz = false;
            std::vector<int> gkind(static_cast<size_t>(nu)), dkind(static_cast<size_t>(nu)), juse(static_cast<size_t>(nuse));
            for (int u = 0; u < nu; ++u) {
                const Chunk *k = U[size_t(u)];
                const int gk = ds41_act_kind(k->gate.type), dk = ds41_act_kind(k->down.type);
                if (gk < 0 || dk < 0 || ds41_act_kind(k->up.type) != gk) throw std::runtime_error("ds41: expert types");
                gkind[size_t(u)] = gk, dkind[size_t(u)] = dk;
                need_k |= gk == DS41_ACT_Q8K, need_z |= gk == DS41_ACT_Q80, need_hk |= dk == DS41_ACT_Q8K, need_hz |= dk == DS41_ACT_Q80;
                for (int j = ubase[size_t(u)]; j < ubase[size_t(u) + 1]; ++j) juse[size_t(j)] = u;
            }
            const int nb = H / 256, nbz = H / 32;
            // STRATA_DS41_XCMG=1: the gate / up rows by CMG. Each CMG keeps its row share of every chunk (the rows
            // rehome_experts placed on it: thread shares m0 .. m0 + mc of xrow); within the CMG the chunks are dealt to its
            // threads (greedily by token count), each computing the CMG's whole share of a chunk in one call. Each row is
            // computed whole by one thread with the same kernel and tokens: the same bits
            static const bool xcmg = strata_env_is_one("STRATA_DS41_XCMG");
            const int Gx = cmg_threads();
            std::vector<int> xown;   // per chunk: its thread within a CMG of Gx threads
            if (xcmg && nu > 0) {
                std::vector<int> ord(static_cast<size_t>(nu));
                for (int u = 0; u < nu; ++u) ord[size_t(u)] = u;
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return Utok[size_t(a)].size() > Utok[size_t(b)].size(); });
                std::vector<int64_t> load(static_cast<size_t>(Gx), 0);
                xown.assign(size_t(nu), 0);
                for (int u : ord) {
                    const int k = int(std::min_element(load.begin(), load.end()) - load.begin());
                    xown[size_t(u)] = k, load[size_t(k)] += int64_t(Utok[size_t(u)].size()) + 1;
                }
            }
            const bool pre_k = need_k && xq_done;
            // each thread quantizes every h block itself (bp4: ~7 blocks per thread cost more than the barrier: <= 3 only)
            const bool own_h = (merge_bits() & 16) && nuse <= 3 && !need_hz;
            std::vector<ds41_q8k> xq(need_k && !pre_k ? static_cast<size_t>(nb) * T : 0), hq(need_hk && !own_h ? static_cast<size_t>(nuse) : 0);
            const ds41_q8k *const xqp = pre_k ? xq_pre_.data() : xq.data();
            std::vector<ds41_q80> xz(need_z ? static_cast<size_t>(nbz) * T : 0), hz(need_hz ? static_cast<size_t>(nuse) * 8 : 0);
            double tq = now();
            double tph[3] = {0, 0, 0};
            Vec gu(static_cast<size_t>(nuse) * 512);
            static const bool xprof = strata_env_is_set("STRATA_DS41_XPROF");
            team_run([&] {
                const int t = omp_get_thread_num(), TT = omp_get_num_threads();
                if (need_k && !pre_k)
                    for (int b = int(int64_t(nb) * T * t / TT); b < int(int64_t(nb) * T * (t + 1) / TT); ++b)   // xn's Q8_K blocks over the team
                        ds41_quant_q8k(X + 256 * size_t(b), 256, &xq[size_t(b)]);
                if (need_z)
                    for (int b = int(int64_t(nbz) * T * t / TT); b < int(int64_t(nbz) * T * (t + 1) / TT); ++b)
                        ds41_quant_q80(X + 32 * size_t(b), 32, &xz[size_t(b)]);
                if ((need_k && !pre_k) || need_z) team_barrier();
                const double tw0 = xprof ? now() : 0;
                // each thread its static share of every chunk's gate and up rows (the rows rehome_experts placed on its
                // CMG; the same share for every token count)
                // (gate and up share the split, so each thread's SwiGLU values are its own rows': no barrier before them)
                const int xg = xcmg ? std::min(t / Gx, (TT + Gx - 1) / Gx - 1) : 0, xm0 = xg * Gx, xmc = xcmg ? std::min(TT, xm0 + Gx) - xm0 : 0, xli = t - xm0;
                auto mine = [&](int u) { return !xcmg || xown[size_t(u)] % xmc == xli; };
                auto rows_of = [&](int u, int &a, int &b) {
                    if (xcmg) a = xrow(u, xm0, TT), b = xrow(u, xm0 + xmc, TT);
                    else a = xrow(u, t, TT), b = xrow(u, t + 1, TT);
                };
                for (int u = 0; u < nu; ++u) {
                    if (!mine(u)) continue;
                    const int ntok = int(Utok[size_t(u)].size());
                    const void *xa[kMaxBatch];
                    for (int s2 = 0; s2 < ntok; ++s2) {
                        const size_t tk = size_t(Utok[size_t(u)][size_t(s2)]);
                        const int gk = gkind[size_t(u)];
                        xa[s2] = gk == DS41_ACT_Q8K ? static_cast<const void *>(&xqp[tk * nb])
                                 : gk == DS41_ACT_Q80 ? static_cast<const void *>(&xz[tk * nbz]) : static_cast<const void *>(X + tk * H);
                    }
                    int r0, r1;
                    rows_of(u, r0, r1);
                    if (r1 <= r0) continue;
                    if (xpf_on()) {   // the next chunk this thread computes
                        int un = u + 1;
                        while (un < nu && !mine(un)) ++un;
                        if (un < nu) {
                            const Chunk *kn = U[size_t(un)];
                            int n0, n1;
                            rows_of(un, n0, n1);
                            prefetch_l2(kn->gate.p + kn->gate.rb * size_t(n0), kn->gate.rb * size_t(n1 - n0));
                            prefetch_l2(kn->up.p + kn->up.rb * size_t(n0), kn->up.rb * size_t(n1 - n0));
                        }
                    }
                    for (int m = 0; m < 2; ++m) {
                        const Mat &M2 = m ? U[size_t(u)]->up : U[size_t(u)]->gate;
                        for (int g = 0, kn = knt(M2.type, M2.cols); g < ntok; g += kn)
                            ds41_rows(M2.type, M2.p, M2.cols, r0, r1, &xa[g], std::min(kn, ntok - g), gu.data() + size_t(ubase[size_t(u)] + g) * 512 + 256 * m, 512);
                    }
                    for (int s2 = 0; s2 < ntok; ++s2) {   // SwiGLU in place of the gate rows
                        float *g = gu.data() + size_t(ubase[size_t(u)] + s2) * 512;
                        if (vexp_on()) swiglu_rows(g + r0, g + 256 + r0, r1 - r0, Uclamp[size_t(u)]);
                        else for (int i = r0; i < r1; ++i) g[i] = swiglu(g[i], g[256 + i], Uclamp[size_t(u)]);
                    }
                }
                const double tw1 = xprof ? now() : 0;
                team_barrier();
                if (xprof && t < 64) g_xtw[t][0] += tw1 - tw0, g_xtw[t][1] += now() - tw1;
                if (t == 0) tph[0] = now();
                thread_local std::vector<ds41_q8k> hown;
                if (own_h) {   // every h block by this thread (the same blocks the team would make): no barrier
                    if (need_hk) hown.resize(size_t(nuse));
                    for (int j = 0; j < nuse; ++j)
                        if (dkind[size_t(juse[size_t(j)])] == DS41_ACT_Q8K) ds41_quant_q8k(gu.data() + size_t(j) * 512, 256, &hown[size_t(j)]);
                } else {
                    for (int j = int(int64_t(nuse) * t / TT); j < int(int64_t(nuse) * (t + 1) / TT); ++j) {
                        const int dk = dkind[size_t(juse[size_t(j)])];
                        if (dk == DS41_ACT_Q8K) ds41_quant_q8k(gu.data() + size_t(j) * 512, 256, &hq[size_t(j)]);
                        else if (dk == DS41_ACT_Q80) ds41_quant_q80(gu.data() + size_t(j) * 512, 256, &hz[size_t(j) * 8]);
                    }
                    team_barrier();
                }
                const ds41_q8k *const hqp = own_h ? hown.data() : hq.data();
                if (t == 0) tph[1] = now();
                const int r0 = int(int64_t(H) * t / TT), r1 = int(int64_t(H) * (t + 1) / TT), nr = r1 - r0;
                const double tw2 = xprof ? now() : 0;
                struct Done { double t0; bool on; int t; ~Done() { if (on && t < 64) g_xtw[t][2] += now() - t0; } } done{tw2, xprof, t};
                if (nr > 0) {
                    thread_local Vec tmp;
                    tmp.resize(static_cast<size_t>(nr) * nuse);
                    for (int u = 0; u < nu; ++u) {
                        const Mat &D = U[size_t(u)]->down;
                        const int ntok = int(Utok[size_t(u)].size());
                        if (xpf_on() && u + 1 < nu) prefetch_l2(U[size_t(u) + 1]->down.p + U[size_t(u) + 1]->down.rb * size_t(r0), U[size_t(u) + 1]->down.rb * size_t(nr));
                        const void *ha[kMaxBatch];
                        for (int s2 = 0; s2 < ntok; ++s2) {
                            const size_t j = size_t(ubase[size_t(u)] + s2);
                            const int dk = dkind[size_t(u)];
                            ha[s2] = dk == DS41_ACT_Q8K ? static_cast<const void *>(&hqp[j])
                                     : dk == DS41_ACT_Q80 ? static_cast<const void *>(&hz[j * 8]) : static_cast<const void *>(gu.data() + j * 512);
                        }
                        for (int g = 0, kn = knt(D.type, D.cols); g < ntok; g += kn)   // (y indexed by the absolute row: shifted by r0)
                            ds41_rows(D.type, D.p, D.cols, r0, r1, &ha[g], std::min(kn, ntok - g), tmp.data() + size_t(ubase[size_t(u)] + g) * nr - r0, nr);
                    }
                    for (int i = 0; i < T; ++i) {
                        float *yi = YB.data() + size_t(i) * H;
                        for (const Use &j : jobs[size_t(i)]) {
                            const float *tj = tmp.data() + size_t(ubase[size_t(j.u)] + j.slot) * nr;
                            const float w = j.w;
                            for (int r = r0; r < r1; ++r) yi[r] += w * tj[r - r0];
                        }
                    }
                }
            });
            const double tend = now();
            tm("ffn.x.gu") += tph[0] - tq;   // with xn's quantization
            tm("ffn.x.act") += tph[1] - tph[0];
            tm("ffn.x.down") += tend - tph[1];
        }
        tm("ffn.experts") += now() - ta;
        sub("ffn.experts");
        if (es_on_) {   // each owner its entries' sums (the all-reduce's rank-order sums)
            const double t0 = now();
            const int W = comm.world;
            int off[65];
            for (int r = 0; r <= W; ++r) off[r] = es_off_[r] * H;
            a2a_r_.resize(size_t(es_b_ - es_a_) * H);
            float *const rp = a2a_r_.data();
            cdo([&, rp] { comm.reduce_scatter(YB.data(), off, rp); });
            for (int i = es_a_; i < es_b_; ++i) std::copy(a2a_r_.begin() + size_t(i - es_a_) * H, a2a_r_.begin() + size_t(i - es_a_ + 1) * H, E[size_t(i)].y.begin());
            const double dt = now() - t0;
            tm("comm") += dt, tm("comm.ffn_y") += dt;
        } else {
            allreduce_timed(YB.data(), int(YB.size()), "comm.ffn_y");
            for (int i = 0; i < T; ++i) std::copy(YB.begin() + size_t(i) * H, YB.begin() + size_t(i + 1) * H, E[size_t(i)].y.begin());
        }
    }

    void xprof_print(int ntok) const {
        const int T = std::min(64, omp_get_max_threads());
        for (int k = 0; k < 3; ++k) {
            double mn = 1e30, mx = 0, sum = 0;
            int amin = 0, amax = 0;
            for (int t = 0; t < T; ++t) {
                const double v = g_xtw[t][k];
                if (v < mn) mn = v, amin = t;
                if (v > mx) mx = v, amax = t;
                sum += v;
            }
            static const char *names[3] = {"gu_work", "gu_wait", "down_work"};
            std::printf("RUNNER_DS41_XPROF %s ms_per_token min=%.3f (t%d) mean=%.3f max=%.3f (t%d)\n", names[k], 1e3 * mn / ntok, amin, 1e3 * sum / T / ntok,
                        1e3 * mx / ntok, amax);
        }
        std::fflush(stdout);
    }
    void init_seq(Seq &q, int cmg = -1) const {
        q.cmg = cmg;
        q.window.assign(static_cast<size_t>(layers), Vec(static_cast<size_t>(ringcap) * c.head_dim, 0.f));
        q.cache.assign(static_cast<size_t>(layers), Vec());
        q.stash_v.assign(static_cast<size_t>(layers), Vec());
        q.stash_s.assign(static_cast<size_t>(layers), Vec());
        for (int l = 0; l < layers; ++l)
            if (c.kv_src[size_t(l)] == l) {
                q.cache[size_t(l)].assign(static_cast<size_t>(context / c.ratio[size_t(l)] + 1) * row_width(), 0.f);
                q.stash_v[size_t(l)].assign(static_cast<size_t>(c.head_dim), 0.f);
                q.stash_s[size_t(l)].assign(static_cast<size_t>(c.head_dim), 0.f);
            }
        if (rehome_on())
            for (int l = 0; l < layers; ++l) {
                spread_pages(q.window[size_t(l)].data(), q.window[size_t(l)].size(), cmg);
                if (!q.cache[size_t(l)].empty()) spread_pages(q.cache[size_t(l)].data(), q.cache[size_t(l)].size(), cmg);
            }
        q.hist.clear();
    }
    void init_ent(Ent &e) const {
        const size_t n = size_t(c.hidden);
        if (e.h.size() != 4 * n) e.h.assign(4 * n, 0.f), e.h2 = e.h, e.x.assign(n, 0.f), e.xn = e.x, e.y = e.x;
        e.coeff_attn.assign(24, 0.f), e.coeff_ffn.assign(24, 0.f), e.cs.assign(5 * 64, 0.f);
        e.sel.assign(static_cast<size_t>(layers), {});
        e.candidates.clear();
        if (spec_track_) e.sv.assign(static_cast<size_t>(layers), Vec()), e.ss.assign(static_cast<size_t>(layers), Vec());
    }

    // ---- DSpark (--dspark draft.gguf: llama.cpp's dflash drafter, src/program/ds41_decode.cpp's semantics) ----
    // target side: each seq0 position's mean of the four streams at the start of the drafter's target layers (the
    // previous layer's FFN output posted), in a ring of kFeatRing positions: the drafter's input features
    static constexpr int kFeatRing = 256;
    std::vector<int> feat_layers_;   // empty: no drafter
    Vec feat_;
    void capture_features(int l, const std::vector<Ent> &E, bool post) {
        const int n = c.hidden, F = int(feat_layers_.size()) * n;
        for (size_t k = 0; k < feat_layers_.size(); ++k) {
            if (feat_layers_[k] != l) continue;
            for (const Ent &e : E) {
                if (e.s != &seq0) continue;
                float *dst = feat_.data() + size_t(e.pos % kFeatRing) * F + k * size_t(n);
                const float *s0 = e.h.data(), *s1 = s0 + n, *s2 = s0 + 2 * size_t(n), *s3 = s0 + 3 * size_t(n), *xo = e.y.data(), *co = e.coeff_ffn.data();
                for (int j = 0; j < n; ++j) {
                    float m = 0;
                    for (int d = 0; d < 4; ++d)
                        m += post ? (((xo[j] * co[4 + d] + s0[j] * co[8 + d]) + s1[j] * co[12 + d]) + s2[j] * co[16 + d]) + s3[j] * co[20 + d]
                                  : e.h[size_t(d) * n + size_t(j)];
                    dst[j] = 0.25f * m;
                }
            }
        }
    }
    // drafter side (a Runner over the dflash blocks): fc rows [y0, y1), the Markov tables (w1 whole, w2's vocabulary
    // slice), the confidence head, the feature norm
    bool drafter_ = false;
    Mat fc_t, w1, w2_t;
    Vec enc_norm, conf_w;
    int markov = 0, mask = 128799, anchor_ = 0;
    float pmin_ = 0;
    std::vector<int> *draft_out_ = nullptr;
    std::vector<float> draft_conf_;
    void load_dspark_extra() {
        fc_t = M.rows("fc.weight", y0, y1 - y0);
        w1 = M.mat("markov_w1.weight");
        w2_t = M.rows("markov_w2.weight", vocab0, vocab1 - vocab0);
        markov = w1.cols;
        if (w2_t.cols != markov || w1.type != DS41_T_BF16 || w1.rows != c.vocab) throw std::runtime_error("ds41: drafter Markov tables");
        enc_norm = M.floats("enc.output_norm.weight");
        if (M.has("conf_proj.weight")) conf_w = M.floats("conf_proj.weight");
        if (const char *e = std::getenv("STRATA_DS41_DSPARK_MASK")) mask = std::atoi(e);
        if (const char *e = std::getenv("STRATA_DS41_DSPARK_PMIN")) pmin_ = float(std::atof(e));
        drafter_ = true;
    }
    // the drafter's keys of committed positions [p0, p0 + n) from the target's feature ring: fc, norm, each block's
    // kv rows, kv norm, rope - into the key rings (seq0.window)
    void inject(const Runner &T, int p0, int n) {
        const int H = c.hidden, F = int(T.feat_layers_.size()) * H, hd = c.head_dim, nope = hd - c.rope_dim, pairs = c.rope_dim / 2;
        const double t0 = now();
        for (int a = 0; a < n; a += pass_max()) {
            const int m = std::min(pass_max(), n - a), q0 = p0 + a;
            Vec feat(size_t(m) * F), G(size_t(m) * H, 0.f), GN(size_t(m) * H), KV(size_t(m) * layers * hd, 0.f);
            for (int i = 0; i < m; ++i) std::copy_n(T.feat_.data() + size_t((q0 + i) % kFeatRing) * F, F, feat.data() + size_t(i) * F);
            const GJob fj{&fc_t, feat.data(), G.data() + y0, 0, fc_t.rows, m, size_t(F), size_t(H)};
            gemv_multi(&fj, 1, "dsp.fc");
            allreduce_timed(G.data(), m * H, "comm.dsp");
            for (int i = 0; i < m; ++i) rms_norm(G.data() + size_t(i) * H, enc_norm.data(), GN.data() + size_t(i) * H, H, c.rms_eps);
            GJob kj[8];
            for (int l = 0; l < layers; ++l)
                kj[l] = {&L[size_t(l)].kv_t, GN.data(), KV.data() + size_t(l) * hd + kv0, 0, L[size_t(l)].kv_t.rows, m, size_t(H), size_t(layers) * hd};
            gemv_multi(kj, layers, "dsp.kv");
            allreduce_timed(KV.data(), int(KV.size()), "comm.dsp");
            float cs[64];
            for (int i = 0; i < m; ++i) {
                rope_table(c.rope_plain, c.rope_dim, q0 + i, 1.0f, cs);
                for (int l = 0; l < layers; ++l) {
                    float *row = seq0.window[size_t(l)].data() + size_t((q0 + i) % ringcap) * hd;
                    rms_norm(KV.data() + (size_t(i) * layers + size_t(l)) * hd, L[size_t(l)].kv_a_norm.data(), row, hd, c.rms_eps);
                    rope(row + nope, pairs, cs);
                }
            }
        }
        tm("dsp.inject") += now() - t0;
    }
    // B drafts after `anchor` at pos: one non-causal pass over [anchor, MASK x (B - 1)] (slot i predicts pos + i + 1),
    // then the target's lm head plus the Markov bias, chained greedily. draft_conf_: each draft's confidence
    std::vector<Ent> dE_;
    std::vector<int> draft(int anchor, int pos, int B) {
        if (!drafter_ || B < 1 || B > kMaxPass) throw std::runtime_error("ds41: draft size");
        const double t0 = now();
        dE_.resize(size_t(B));
        for (int i = 0; i < B; ++i) dE_[size_t(i)].s = &seq0, dE_[size_t(i)].pos = pos + i, dE_[size_t(i)].token = i ? mask : anchor, dE_[size_t(i)].last = pos + B - 1;
        std::vector<int> out;
        draft_out_ = &out, anchor_ = anchor;
        int dummy[kMaxPass];
        pass(dE_, dummy);
        draft_out_ = nullptr;
        tm("dsp.draft") += now() - t0;
        return out;
    }
    void draft_head(std::vector<Ent> &E) {
        const int nv = vocab1 - vocab0, T = int(E.size()), H = c.hidden;
        const double th0 = now();
        const float *X = gather_xn(E);
        Vec part(size_t(nv) * T), row(static_cast<size_t>(markov)), bias(static_cast<size_t>(nv));
        const GJob hj{&head, X, part.data(), 0, head.rows, T, size_t(H), size_t(nv)};
        gemv_multi(&hj, 1, "dsp.head");
        tm("dsp.h.head") += now() - th0;
        draft_conf_.clear();
        int prev = anchor_;
        for (int i = 0; i < T; ++i) {
            ds41_dequant_row(DS41_T_BF16, w1.p + w1.rb * size_t(prev), markov, row.data());
            if (!conf_w.empty()) {   // sigmoid of conf_proj . [the pre-norm hidden; the Markov row]
                double z = 0;
                for (int j = 0; j < H; ++j) z += double(conf_w[size_t(j)]) * E[size_t(i)].x[size_t(j)];
                for (int j = 0; j < markov; ++j) z += double(conf_w[size_t(H + j)]) * row[size_t(j)];
                const float p = float(1.0 / (1.0 + std::exp(-z)));
                if (pmin_ > 0 && p < pmin_) break;
                draft_conf_.push_back(p);
            }
            const double tc0 = now();
            const GJob bj{&w2_t, row.data(), bias.data(), 0, nv, 1, size_t(markov), size_t(nv)};
            gemv_multi(&bj, 1, "dsp.w2");
            float *pi = part.data() + size_t(i) * nv;
            for (int v = 0; v < nv; ++v) pi[v] += bias[size_t(v)];
            const float *pp = pi;
            const double tc1 = now();
            comm.argmax(&pp, 1, nv, vocab0, &prev);
            tm("dsp.h.bias") += tc1 - tc0, tm("dsp.h.argmax") += now() - tc1;
            draft_out_->push_back(prev);
        }
    }

    // one pass over the tokens E (each: s, pos, token; consecutive positions of a sequence in order). greedy: each
    // token's argmax (comm.argmax over the vocabulary slices: the first maximum, as std::max_element over the full
    // logits); otherwise (one token) the full logits on every rank
    // STRATA_DS41_BARPROF: team regions per pass phase (the count since the last mark goes to the phase)
    std::map<std::string, double> rgn_n, rgn_s;   // per phase: regions, thread 0's time inside them
    double rgn_mark = 0, rgn_smark = 0;
    void rgn(const char *phase) {
        if (!g_barprof) return;
        rgn_n[phase] += g_regions - rgn_mark, rgn_mark = g_regions;
        rgn_s[phase] += g_region_s - rgn_smark, rgn_smark = g_region_s;
    }
    // and per site inside attention / ffn (sub_begin at their entry, sub(k) after each part): regions, thread 0's region
    // time, the team's summed barrier wait (read between regions: the workers are idle then)
    struct SubP { double n = 0, s = 0, w = 0; };
    std::map<std::string, SubP> sub_p;
    SubP sub_mark;
    static double bar_wait_sum() {
        double w = 0;
        for (const auto &b : g_barp) w += b.wait;
        return w;
    }
    void sub_begin() { if (g_barprof) sub_mark = {g_regions, g_region_s, bar_wait_sum()}; }
    void sub(const char *k) {
        if (!g_barprof) return;
        const SubP now_{g_regions, g_region_s, bar_wait_sum()};
        SubP &d = sub_p[k];
        d.n += now_.n - sub_mark.n, d.s += now_.s - sub_mark.s, d.w += now_.w - sub_mark.w;
        sub_mark = now_;
    }
    Vec pass(std::vector<Ent> &E, int *greedy) {
        if (g_barprof) rgn_mark = g_regions, rgn_smark = g_region_s;
        const int n = c.hidden, T = int(E.size());
        if (!tp) throw std::runtime_error("ds41: the pass needs the tensor-parallel split (STRATA_DS41_TP=1)");
        if (T < 1 || T > kMaxBatch || (!greedy && T != 1)) throw std::runtime_error("ds41: pass size");
        if (T > pass_max()) {   // each sequence's entries within its ring's extra rows
            std::map<const Seq *, int> per;
            for (const Ent &e : E)
                if (++per[e.s] > pass_max()) throw std::runtime_error("ds41: a sequence's entries in one pass beyond pass_max()");
        }
        {   // STRATA_DS41_ENTRY_SHARD: this pass's own entries (all of them when off)
            const int W = comm.world;
            es_on_ = entry_shard_env() && comm.xc && T >= W && greedy && !drafter_ && feat_layers_.empty() && !router_tp_on();
            for (int r = 0; r <= W; ++r) es_off_[r] = es_on_ ? int(int64_t(T) * r / W) : (r ? T : 0);
            es_a_ = es_on_ ? es_off_[comm.rank] : 0, es_b_ = es_on_ ? es_off_[comm.rank + 1] : T;
        }
        for (Ent &e : E) {
            init_ent(e);
            if (e.pos >= context) throw std::runtime_error("ds41: past --context");
            if (int(e.s->hist.size()) <= e.pos) e.s->hist.resize(static_cast<size_t>(e.pos) + 1);
            e.s->hist[size_t(e.pos)] = e.token;
        }
        prefetch_engram(E);
        double te0 = now();
        static const bool es_embd = strata_env_is_one("STRATA_DS41_ES_EMBD");
        if (embd_mem && es_on_ && es_embd && comm.xc) {
            // STRATA_DS41_ES_EMBD=1 (ENTRY_SHARD): only an entry's owner keeps its residual, so each token's row goes from
            // the rank holding it to the entry's owner alone, as table bytes (an all-to-all, the bits as sent), dequantized
            // there + 0.0f: the all-reduce's value (the one rank's row and zeros, summed)
            const int W = comm.world, me = comm.rank, wr = int((embd.rb + 3) / 4);
            auto holder = [&](int tok) {
                int h = int(int64_t(tok) * W / c.vocab);
                while (h + 1 < W && tok >= int(int64_t(c.vocab) * (h + 1) / W)) ++h;
                while (h > 0 && tok < int(int64_t(c.vocab) * h / W)) --h;
                return h;
            };
            std::vector<int> hold(static_cast<size_t>(T));
            for (int i = 0; i < T; ++i) hold[size_t(i)] = holder(E[size_t(i)].token);
            int soff[65], roff[65];
            soff[0] = roff[0] = 0;
            for (int r = 0; r < W; ++r) {
                int ns = 0, nr = 0;
                for (int i = es_off_[r]; i < es_off_[r + 1]; ++i) ns += hold[size_t(i)] == me;
                for (int i = es_a_; i < es_b_; ++i) nr += hold[size_t(i)] == r;
                soff[r + 1] = soff[r] + ns * wr, roff[r + 1] = roff[r] + nr * wr;
            }
            Vec &S = a2a_s_, &Rv = a2a_r_;
            S.resize(size_t(soff[W]) + 1), Rv.resize(size_t(roff[W]) + 1);
            {
                size_t w = 0;
                for (int i = 0; i < T; ++i)   // (entries in order: owner by owner, as soff)
                    if (hold[size_t(i)] == me)
                        std::memcpy(S.data() + w, embd_mem + embd.rb * size_t(E[size_t(i)].token - vocab0), embd.rb), w += size_t(wr);
            }
            const double t0 = now();
            const float *const sp = S.data();
            float *const rp = Rv.data();
            cdo([&, sp, rp] { comm.alltoall(sp, soff, rp, roff); });
            const double dt = now() - t0;
            tm("comm") += dt, tm("comm.embd") += dt;
            int pos[65];
            for (int r = 0; r < W; ++r) pos[r] = roff[r];
            for (int i = es_a_; i < es_b_; ++i) {
                const int h = hold[size_t(i)];
                float *x = E[size_t(i)].x.data();
                ds41_dequant_row(embd.type, reinterpret_cast<const uint8_t *>(Rv.data() + pos[h]), n, x);
                for (int j = 0; j < n; ++j) x[j] += 0.0f;
                pos[h] += wr;
            }
        } else if (embd_mem) {   // each token's row from the rank that holds it, the others zero, summed
            pX_.assign(size_t(n) * T, 0.f);
            for (int i = 0; i < T; ++i) {
                const int tok = E[size_t(i)].token;
                if (tok >= vocab0 && tok < vocab1) ds41_dequant_row(embd.type, embd_mem + embd.rb * size_t(tok - vocab0), n, pX_.data() + size_t(i) * n);
            }
            allreduce_timed(pX_.data(), n * T, "comm.embd");
            for (int i = 0; i < T; ++i) std::copy(pX_.begin() + size_t(i) * n, pX_.begin() + size_t(i + 1) * n, E[size_t(i)].x.begin());
        } else
            for (Ent &e : E) {
                std::vector<uint8_t> raw(embd.rb);
                read_bytes(M.t("token_embd.weight"), embd.rb * size_t(e.token), raw.size(), raw.data());
                ds41_dequant_row(embd.type, raw.data(), n, e.x.data());
            }
        tm("embd") += now() - te0;
        for (Ent &e : E) {
            const int pos = e.pos;
            float *cs = e.cs.data();
            rope_table(c.rope_plain, c.rope_dim, pos, 1.0f, cs);
            rope_table(c.rope_plain, c.rope_dim, pos, -1.0f, cs + 64);
            rope_table(c.rope_compressed, c.rope_dim, pos, 1.0f, cs + 128);
            rope_table(c.rope_compressed, c.rope_dim, pos, -1.0f, cs + 192);
            rope_table(c.rope_compressed, c.rope_dim, std::max(pos - 1, 0), 1.0f, cs + 256);
            for (int s = 0; s < 4; ++s) std::copy(e.x.begin(), e.x.end(), e.h.begin() + size_t(s) * n);
        }
        bool post = false;   // the previous layer's FFN output still to post (fused into the next attention site)
        for (int l = 0; l < layers; ++l) {
            const Layer &Y = L[size_t(l)];
            double t = now();
            if (!feat_layers_.empty()) capture_features(l, E, post);
            if (c.engram_index(l) >= 0) {   // Engram reads and changes the streams: post first
                if (post) for (int i = es_a_; i < es_b_; ++i) hc_post(E[size_t(i)], E[size_t(i)].y.data(), E[size_t(i)].coeff_ffn.data());
                post = false;
                tm("mhc") += now() - t, t = now();
                engram(l, E);
            }
            tm("engram") += now() - t, t = now();
            rgn("engram");
            site_all(E.data() + es_a_, es_b_ - es_a_, post, 0, l == 0, Y.hc_attn_fn, Y.hc_attn_scale, Y.hc_attn_base, Y.attn_norm);
            if (es_on_ && !es_qk_layer(l)) share_xn(E, "comm.xn_attn");
            tm("mhc") += now() - t, t = now();
            rgn("mhc");
            attention(l, E);
            tm("attn") += now() - t, t = now();
            rgn("attn");
            site_all(E.data() + es_a_, es_b_ - es_a_, true, 1, false, Y.hc_ffn_fn, Y.hc_ffn_scale, Y.hc_ffn_base, Y.ffn_norm);
            // (ENTRY_SHARD: ffn routes the own entries first, then gathers every entry's xn with the picks)
            tm("mhc") += now() - t, t = now();
            rgn("mhc");
            ffn(l, E);
            tm("ffn") += now() - t;
            rgn("ffn");
            post = true;
        }
        double t = now();
        for (int i = es_a_; i < es_b_; ++i) {
            Ent &e = E[size_t(i)];
            if (post) hc_post(e, e.y.data(), e.coeff_ffn.data());
            hc_pre(e, e.coeff_ffn.data(), e.x.data());
            rms_norm(e.x.data(), out_norm.data(), e.xn.data(), n, c.rms_eps);
        }
        if (es_on_) share_xn(E, "comm.xn_head");
        tm("mhc") += now() - t, t = now();
        rgn("mhc");
        if (drafter_) {
            draft_head(E);
            tm("head") += now() - t;
            return {};
        }
        if (greedy) {
            const int nv = vocab1 - vocab0;
            const float *X = gather_xn(E);
            Vec part(size_t(nv) * T);
            const GJob hj{&head, X, part.data(), 0, head.rows, T, size_t(n), size_t(nv)};
            gemv_multi(&hj, 1, "head");
            const float *pp[kMaxBatch];
            for (int i = 0; i < T; ++i) pp[i] = part.data() + size_t(i) * nv;
            comm.argmax(pp, T, nv, vocab0, greedy);
            tm("head") += now() - t;
            return {};
        }
        Vec logits(static_cast<size_t>(c.vocab), 0.f);
        gemv(head, E[0].xn.data(), logits.data() + vocab0, "head");
        comm.allreduce(logits.data(), c.vocab);
        tm("head") += now() - t;
        return logits;
    }
    // one token at position pos of the main sequence (seq0); greedy != nullptr: only its argmax
    Vec step(int token, int pos, int *greedy = nullptr) {
        ents_.resize(1);
        Ent &e = ents_[0];
        e.s = &seq0, e.pos = pos, e.token = token;
        return pass(ents_, greedy);
    }
};

// the DSpark drafter's Config: the target's geometry (checked against the dflash keys), its own blocks, experts, norms
// and clamps; no compressed rows, no Engram. The rope is the target's plain one (as the GPU drafter's tables)
Config draft_config(const strata::GgufFile &f, const Config &tc) {
    auto num = [&](const char *k, double fb) {
        const auto *v = f.get(std::string("dflash.") + k);
        return v ? (v->is_num() ? v->num() : v->items.empty() ? fb : v->items[0].num()) : fb;
    };
    auto must = [&](const char *k) {
        if (!f.get(std::string("dflash.") + k)) throw std::runtime_error(std::string("ds41: drafter has no dflash.") + k);
        return num(k, 0);
    };
    Config c = tc;
    c.layers = int(must("block_count"));
    if (int(must("embedding_length")) != tc.hidden || int(must("attention.head_count")) != tc.heads ||
        int(must("attention.key_length")) != tc.head_dim || int(must("attention.q_lora_rank")) != tc.q_rank ||
        int(must("attention.output_group_count")) != tc.out_groups || int(must("attention.output_lora_rank")) != tc.out_rank ||
        int(must("rope.dimension_count")) != tc.rope_dim || int(num("hyper_connection.count", 4)) != 4 || c.layers < 1 || c.layers > 8)
        throw std::runtime_error("ds41: drafter geometry differs from the target's");
    c.window = int(must("attention.sliding_window"));
    c.experts = int(must("expert_count"));
    c.top_k = int(must("expert_used_count"));
    c.ff = int(must("expert_feed_forward_length"));
    c.expert_scale = float(num("expert_weights_scale", 1.5));
    if (const auto *v = f.get("dflash.expert_weights_norm")) c.normalize = v->u != 0;
    c.rms_eps = float(must("attention.layer_norm_rms_epsilon"));
    c.hc_eps = float(num("hyper_connection.epsilon", 1e-6));
    c.sinkhorn = int(num("hyper_connection.sinkhorn_iterations", 20));
    c.ratio.assign(size_t(c.layers), 0), c.kv_src.assign(size_t(c.layers), -1), c.topk_src.assign(size_t(c.layers), -1);
    c.candidate_source = -1;
    c.engram_layers.clear();
    c.clamp.assign(size_t(c.layers), float(num("swiglu_clamp_exp", 10)));
    c.shared_clamp.assign(size_t(c.layers), float(num("swiglu_clamp_shexp", 10)));
    return c;
}

// a llama.cpp --kl-divergence-base file (tools/ds41f_kl_base.py): "_logits_", n_ctx, n_vocab, n_chunk, every chunk's
// tokens, then per chunk n_ctx/2 - 1 rows (positions n_ctx/2 .. n_ctx-2): scale, min_log_prob, nv uint16 codes
struct KlBase {
    std::ifstream f;
    int n_ctx = 0, n_vocab = 0, n_chunk = 0, nv = 0, rows = 0;
    size_t row_bytes = 0, rows_at = 0;
    explicit KlBase(const std::string &path) : f(path, std::ios::binary) {
        char magic[8];
        if (!f.read(magic, 8) || std::memcmp(magic, "_logits_", 8)) throw std::runtime_error("ds41: not a kl-divergence-base file: " + path);
        int32_t h[3];
        f.read(reinterpret_cast<char *>(h), 12);
        n_ctx = h[0], n_vocab = h[1], n_chunk = h[2];
        nv = 2 * ((n_vocab + 1) / 2), rows = n_ctx / 2 - 1, row_bytes = size_t(nv + 4) * 2;
        rows_at = 20 + size_t(n_chunk) * n_ctx * 4;
    }
    std::vector<int> tokens(int chunk) {
        std::vector<int32_t> t(static_cast<size_t>(n_ctx));
        f.seekg(std::streamoff(20 + size_t(chunk) * n_ctx * 4));
        f.read(reinterpret_cast<char *>(t.data()), std::streamsize(t.size() * 4));
        return std::vector<int>(t.begin(), t.end());
    }
    std::vector<double> log_probs(int chunk, int row) {   // row j: position n_ctx/2 + j
        std::vector<uint8_t> raw(row_bytes);
        f.seekg(std::streamoff(rows_at + (size_t(chunk) * rows + size_t(row)) * row_bytes));
        f.read(reinterpret_cast<char *>(raw.data()), std::streamsize(raw.size()));
        float scale, mn;
        std::memcpy(&scale, raw.data(), 4), std::memcpy(&mn, raw.data() + 4, 4);
        std::vector<double> lp(static_cast<size_t>(n_vocab));
        for (int v = 0; v < n_vocab; ++v) {
            uint16_t code;
            std::memcpy(&code, raw.data() + 8 + 2 * size_t(v), 2);
            lp[size_t(v)] = double(code) * scale + mn;
        }
        return lp;
    }
};

std::vector<int> read_ids(const std::string &path, size_t limit) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("ds41: cannot read " + path);
    std::vector<int> ids;
    long v;
    while (ids.size() < limit && f >> v) ids.push_back(int(v));
    return ids;
}

}  // namespace

int main(int argc, char **argv) {
    Comm comm;
    comm.init(&argc, &argv);
    g_comm = &comm;
#ifdef STRATA_CPU_RUNNER_UTOFU
    cpu_utofu_set_par(utofu_par);
#endif
    int code = 0;
    try {
        if (argc < 2) throw std::runtime_error("usage: ds41_cpu_runner MODEL.gguf --eval IDS [--tokens N] [--layers N] [--context N] [--generate N] [--save-logits F]");
        std::string eval, save, kl_base, engram_dir, image_dir, dspark, batch_state;
        int kl_chunk = 0;
        int tokens = 1 << 30, nlayers = 0, context = 4096, generate = 0, prompt_pass = 0, batch = 0, spec = 0, spec_ngram = 3, dspark_n = 5;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&] { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[++i]); };
            if (a == "--eval") eval = next();
            else if (a == "--tokens") tokens = std::stoi(next());
            else if (a == "--layers") nlayers = std::stoi(next());
            else if (a == "--context") context = std::stoi(next());
            else if (a == "--generate") generate = std::stoi(next());
            else if (a == "--prompt-pass") prompt_pass = std::stoi(next());
            else if (a == "--batch") batch = std::stoi(next());
            else if (a == "--batch-state") batch_state = next();
            else if (a == "--spec") spec = std::stoi(next());
            else if (a == "--spec-ngram") spec_ngram = std::stoi(next());
            else if (a == "--dspark") dspark = next();
            else if (a == "--dspark-n") dspark_n = std::stoi(next());
            else if (a == "--save-logits") save = next();
            else if (a == "--kl-base") kl_base = next();
            else if (a == "--kl-chunk") kl_chunk = std::stoi(next());
            else if (a == "--engram-dir") engram_dir = next();
            else if (a == "--image-dir") image_dir = next();
            else if (a == "--threads") omp_set_num_threads(std::stoi(next()));
            else throw std::runtime_error("unknown option " + a);
        }
        flush_subnormals();
#pragma omp parallel
        flush_subnormals();
#ifdef STRATA_CPU_RUNNER_UTOFU
        configure_prompt_io(prompt_pass > 0 && batch == 0 && dspark.empty(), prompt_pass);
#endif
        const double t0 = now();
        strata::core::ModelArtifact artifact{std::filesystem::path(argv[1])};
        Model model(artifact);
        std::string img;
        if (!image_dir.empty()) {   // replay only when every rank has its image (the down exchange needs them all)
            std::filesystem::create_directories(image_dir);
            img = image_dir + "/ds41-r" + std::to_string(comm.rank) + "of" + std::to_string(comm.world) + "-L" + std::to_string(nlayers) + "-v1.img";
            int have = std::filesystem::exists(img) ? 1 : 0, all = have;
#ifdef STRATA_CPU_RUNNER_MPI
            MPI_Allreduce(&have, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
            if (all) {
                g_img_fd = ::open(img.c_str(), O_RDONLY);
                g_replay = g_img_fd >= 0;
            }
        }
        Runner run(model, comm, nlayers, context, engram_dir);
        if (g_replay) {
            if (g_img_off != std::filesystem::file_size(img)) throw std::runtime_error("ds41: image size mismatch (stale image?)");
            ::close(g_img_fd);
        } else if (!img.empty()) {   // write it for the next load
            const int fd = ::open((img + ".partial").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            size_t off = 0;
            for (const auto &[b, n] : g_bufs) {
                for (size_t d = 0; d < n;) {
                    const ssize_t w = ::pwrite(fd, b + d, std::min(n - d, size_t(1) << 30), off_t(off + d));
                    if (w <= 0) throw std::runtime_error("ds41: image write failed");
                    d += size_t(w);
                }
                off += n;
            }
            ::close(fd);
            std::filesystem::rename(img + ".partial", img);
        }
        run.split_tp();
        run.engram_to_ram();
        run.load_shared_all();
        run.load_embd_slice();
        run.rehome_all();
        if (batch > 0 && Runner::es_index_env()) {
            // The batch creates B additional Seq objects; seq0 stays allocated.
            // Only enabled-owner batches change their repacking policy.
            size_t sequence_bytes = 0;
            for (const auto *vv : {&run.seq0.window, &run.seq0.cache, &run.seq0.stash_v, &run.seq0.stash_s})
                for (const Vec &v : *vv) {
                    if (v.size() > (size_t(-1) - sequence_bytes) / sizeof(float))
                        throw std::runtime_error("ds41: owner batch sequence storage overflow");
                    sequence_bytes += v.size() * sizeof(float);
                }
            if (sequence_bytes && size_t(batch) > size_t(-1) / sequence_bytes)
                throw std::runtime_error("ds41: owner batch future storage overflow");
            run.owner_batch_future_bytes_ = sequence_bytes * size_t(batch);
            if (comm.rank == 0)
                std::cerr << "DS41 owner_batch_future_bytes=" << run.owner_batch_future_bytes_ << "\n";
        }
        run.repack_experts();
        run.convert_q5p();
        run.rehome_experts();
        // --dspark: the drafter, read from its file after the target's image is done (not part of it)
        std::unique_ptr<strata::GgufFile> dgf;
        std::map<std::string, strata::core::ArtifactTensor> dtens;
        std::unique_ptr<Model> dmodel;
        std::unique_ptr<Runner> drun;
        if (!dspark.empty()) {
            const double td = now();
            dgf = std::make_unique<strata::GgufFile>(dspark);
            if (const auto *a = dgf->get("general.architecture"); !a || a->s != "dflash") throw std::runtime_error("ds41: --dspark wants a dflash GGUF");
            for (const auto &t : dgf->tensors()) dtens.emplace(t.name, strata::core::ArtifactTensor{dgf.get(), &t, strata::tensor_payload_bytes(t)});
            dmodel = std::make_unique<Model>(dtens);
            g_replay = false;
            drun = std::make_unique<Runner>(*dmodel, comm, draft_config(*dgf, run.c), context, run);
            drun->split_tp();
            drun->load_dspark_extra();
            for (int64_t l : dgf->int_array("dflash.target_layers")) {
                if (l < 0 || l >= run.layers) throw std::runtime_error("ds41: drafter target layer out of range");
                run.feat_layers_.push_back(int(l));
            }
            run.feat_.assign(size_t(Runner::kFeatRing) * run.feat_layers_.size() * size_t(run.c.hidden), 0.f);
            if (comm.rank == 0)
                std::cerr << "DS41 dspark layers=" << drun->layers << " experts=" << drun->c.experts << " top_k=" << drun->c.top_k << " markov=" << drun->markov
                          << " conf=" << !drun->conf_w.empty() << " load_s=" << now() - td << " mem_avail_GB=" << Runner::mem_available() / 1e9 << "\n";
        }
        comm.barrier();
        {   // where the resident pages live (first touch): pages per NUMA node from /proc/self/numa_maps
            std::ifstream nm("/proc/self/numa_maps");
            std::string line;
            double pg[8] = {0};
            while (std::getline(nm, line)) {
                std::istringstream ls(line);
                std::string tok;
                while (ls >> tok)
                    if (tok.size() > 2 && tok[0] == 'N' && std::isdigit(static_cast<unsigned char>(tok[1]))) {
                        const auto eq = tok.find('=');
                        const int node = std::atoi(tok.c_str() + 1);
                        if (eq != std::string::npos && node >= 0 && node < 8) pg[node] += std::atof(tok.c_str() + eq + 1);
                    }
            }
            if (comm.rank == 0) {
                std::printf("RUNNER_DS41_NUMA pages");
                for (int k = 0; k < 8; ++k) if (pg[k] > 0) std::printf(" N%d=%.0f", k, pg[k]);
                std::printf("\n");
                std::fflush(stdout);
            }
        }
        if (comm.rank == 0)
            std::cerr << "DS41 ready tp=" << run.tp << "  ranks=" << comm.world << " layers=" << run.layers << " resident_GB=" << g_resident / 1e9
                      << " load_s=" << now() - t0 << " read_GB=" << g_read_b / 1e9 << " read_s=" << g_read_s << " engram_stage_s=" << run.stage_s << " engram_ram_GB=" << run.engram_ram_b / 1e9 << " engram_ram_s=" << run.engram_ram_s
                      << " embd_slice_s=" << run.embd_load_s << " rehome_s=" << run.rehome_s << " iq2r_mats=" << run.iq2r_n << " iq2r_extra_GB=" << run.iq2r_gb << " iq2r_s=" << run.iq2r_s << " mem_avail_GB=" << Runner::mem_available() / 1e9 << "\n";
        std::unique_ptr<KlBase> kb;
        std::vector<int> ids;
        if (!kl_base.empty()) {
            kb = std::make_unique<KlBase>(kl_base);
            ids = kb->tokens(kl_chunk);
            if (ids.size() > size_t(tokens)) ids.resize(size_t(tokens));
        } else ids = read_ids(eval, size_t(tokens));
        double kl_sum = 0, kl_nll_ref = 0, kl_nll = 0;
        int kl_n = 0, kl_top = 0;
        std::ofstream lf;
        if (!save.empty() && comm.rank == 0) lf.open(save, std::ios::binary);
        double nll = 0;
        int scored = 0, top1 = 0;
        const double te = now();
        Vec logits;
        // the decode loop in one open team (team_run dispatches to it); STRATA_RUNNER_PERSISTENT=0: an OpenMP fork per region
        g_bar.init(omp_get_max_threads(), cmg_threads());
        if (strata_env_is_set("STRATA_DS41_ARBENCH")) {   // all-reduce latency at the decode sizes: team idle, then inside the open team
            auto bench = [&](const char *where) {
                for (int n : {1792, 5120, 8192, 16384, 65536, 131072, 262144}) {
                    Vec v(static_cast<size_t>(n), 1.f);
                    for (int i = 0; i < 20; ++i) comm.allreduce(v.data(), n);
                    const double t0 = now();
                    const int reps = n > 16384 ? 50 : 400;
                    for (int i = 0; i < reps; ++i) comm.allreduce(v.data(), n);
                    if (comm.rank == 0) std::printf("RUNNER_DS41_ARBENCH %s n=%d us=%.1f GBps=%.2f\n", where, n, 1e6 * (now() - t0) / reps, 4.0 * n * reps / (now() - t0) / 1e9);
                }
                std::fflush(stdout);
            };
            bench("idle");
            with_team([&] { bench("team"); });
        }
        if (strata_env_is_set("STRATA_DS41_BARBENCH"))   // the open team's costs: an empty region, a barrier inside one
            with_team([&] {
                for (int k : {0, 1, 4}) {
                    for (int i = 0; i < 2000; ++i) team_run([&] { for (int j = 0; j < k; ++j) team_barrier(); });
                    const int reps = 20000;
                    const double t0 = now();
                    for (int i = 0; i < reps; ++i) team_run([&] { for (int j = 0; j < k; ++j) team_barrier(); });
                    if (comm.rank == 0) std::printf("RUNNER_DS41_BARBENCH inner_barriers=%d us_per_region=%.3f\n", k, 1e6 * (now() - t0) / reps);
                }
                std::fflush(stdout);
            });
        if (batch > 0) {   // --batch B: B sequences (the prompts: KL chunks kl_chunk .. + B - 1, --tokens each) prefilled
            // in passes of 16, then --generate tokens for all of them in passes of B entries (one per sequence)
            if (!kb || batch > Runner::kMaxBatch) throw std::runtime_error("ds41: --batch needs --kl-base and B <= Runner::kMaxBatch");
            const int B = batch, P = int(ids.size());
            std::vector<Runner::Seq> seqs(static_cast<size_t>(B));
            std::vector<std::vector<int>> prompts(static_cast<size_t>(B)), outs(static_cast<size_t>(B));
            for (int b = 0; b < B; ++b) {
                // (STRATA_DS41_BATCH_CMG, default 1: sequence b's KV on CMG b % 4, its attention by that CMG's threads)
                static const bool home = !(strata_env_equals("STRATA_DS41_BATCH_CMG", "0"));
                const int ncmg = (omp_get_max_threads() + cmg_threads() - 1) / cmg_threads();
                run.init_seq(seqs[size_t(b)], home && B > 1 ? b % ncmg : -1);
                prompts[size_t(b)] = kb->tokens(kl_chunk + (strata_env_is_set("STRATA_DS41_BATCH_SAME") ? 0 : b));   // (SAME: one prompt for all)
                prompts[size_t(b)].resize(size_t(P));
            }
            std::vector<int> nxt(static_cast<size_t>(B)), am(16);
            if (comm.rank == 0) std::printf("RUNNER_DS41_BATCHMEM B=%d mem_available_gb=%.2f\n", B, run.mem_available() / 1e9);
            // --batch-state save:DIR | load:DIR: every rank's sequences after the prefill (KV rings, compressed caches,
            // stashes, history) and the next tokens in DIR/bstate.<rank> (a node-local directory): load skips the prefill
            const bool st_save = batch_state.rfind("save:", 0) == 0, st_load = batch_state.rfind("load:", 0) == 0;
            if (!batch_state.empty() && !st_save && !st_load) throw std::runtime_error("ds41: --batch-state save:DIR or load:DIR");
            const std::string st_file = batch_state.empty() ? "" : batch_state.substr(5) + "/bstate." + std::to_string(comm.rank);
            const int64_t st_key[6] = {B, P, run.layers, run.ringcap, kl_chunk, strata_env_is_set("STRATA_DS41_BATCH_SAME") ? 1 : 0};
            auto st_vecs = [&](Runner::Seq &q, auto &&f) {   // every state vector of a sequence, in one fixed order
                for (auto *vv : {&q.window, &q.cache, &q.stash_v, &q.stash_s})
                    for (Vec &v : *vv) f(v);
            };
            const double tp0 = now();
            if (st_load) {
                std::FILE *fp = std::fopen(st_file.c_str(), "rb");
                if (!fp) throw std::runtime_error("ds41: cannot read " + st_file);
                bool ok = true;
                int64_t key[6];
                ok &= std::fread(key, sizeof key, 1, fp) == 1 && std::equal(key, key + 6, st_key);
                for (int b = 0; b < B && ok; ++b) {
                    Runner::Seq &q = seqs[size_t(b)];
                    st_vecs(q, [&](Vec &v) { int64_t m = 0; ok = ok && std::fread(&m, 8, 1, fp) == 1 && m == int64_t(v.size()) && std::fread(v.data(), 4, v.size(), fp) == v.size(); });
                    int64_t h = 0;
                    ok = ok && std::fread(&h, 8, 1, fp) == 1 && h >= 0 && h <= run.context;
                    if (ok) q.hist.resize(size_t(h)), ok = std::fread(q.hist.data(), 4, size_t(h), fp) == size_t(h);
                }
                ok = ok && std::fread(nxt.data(), 4, size_t(B), fp) == size_t(B);
                std::fclose(fp);
                if (!ok) throw std::runtime_error("ds41: " + st_file + " does not match this run");
            } else
            with_team([&] {
                std::vector<Runner::Ent> E;
                for (int b = 0; b < B; ++b)
                    for (int p0 = 0; p0 < P; p0 += 16) {
                        const int T = std::min(16, P - p0);
                        E.resize(size_t(T));
                        for (int i = 0; i < T; ++i) E[size_t(i)].s = &seqs[size_t(b)], E[size_t(i)].pos = p0 + i, E[size_t(i)].token = prompts[size_t(b)][size_t(p0 + i)];
                        run.pass(E, am.data());
                        nxt[size_t(b)] = am[size_t(T - 1)];
                    }
            });
            if (st_save) {
                std::FILE *fp = std::fopen(st_file.c_str(), "wb");
                bool ok = fp && std::fwrite(st_key, sizeof st_key, 1, fp) == 1;
                for (int b = 0; b < B && ok; ++b) {
                    Runner::Seq &q = seqs[size_t(b)];
                    st_vecs(q, [&](Vec &v) { const int64_t m = int64_t(v.size()); ok = ok && std::fwrite(&m, 8, 1, fp) == 1 && std::fwrite(v.data(), 4, v.size(), fp) == v.size(); });
                    const int64_t h = int64_t(q.hist.size());
                    ok = ok && std::fwrite(&h, 8, 1, fp) == 1 && std::fwrite(q.hist.data(), 4, size_t(h), fp) == size_t(h);
                }
                ok = ok && std::fwrite(nxt.data(), 4, size_t(B), fp) == size_t(B);
                if (fp) ok &= std::fclose(fp) == 0;
                if (!ok) throw std::runtime_error("ds41: cannot write " + st_file);
            }
            const double tpre = now() - tp0;
            if (comm.rank == 0) std::printf("RUNNER_DS41_BATCHMEM after_prefill mem_available_gb=%.2f state=%s\n", run.mem_available() / 1e9, batch_state.empty() ? "-" : batch_state.c_str());
            const auto timers0 = run.timers;
            const auto gstat0 = g_gstat;
            std::memset(g_xtw, 0, sizeof g_xtw);   // (STRATA_DS41_XPROF: the decode steps only)
            const double tg0 = now();
            // (STRATA_DS41_BATCH_FIBERS=1, B >= 2: each step as two half-batch fibers, Runner::fib_pair)
            static const bool fibers = strata_env_is_one("STRATA_DS41_BATCH_FIBERS");
            with_team([&] {
                std::vector<Runner::Ent> E(static_cast<size_t>(B));
                const int h = B / 2;
                std::vector<Runner::Ent> Ea(static_cast<size_t>(h)), Eb(static_cast<size_t>(B - h));
                for (int g = 0; g < generate; ++g) {
                    for (int b = 0; b < B; ++b) {
                        E[size_t(b)].s = &seqs[size_t(b)], E[size_t(b)].pos = P + g, E[size_t(b)].token = nxt[size_t(b)];
                        outs[size_t(b)].push_back(nxt[size_t(b)]);
                    }
                    if (fibers && B >= 2) {
                        for (int b = 0; b < B; ++b) {
                            Runner::Ent &e = b < h ? Ea[size_t(b)] : Eb[size_t(b - h)];
                            e.s = &seqs[size_t(b)], e.pos = P + g, e.token = nxt[size_t(b)];
                        }
                        run.fib_pair([&] { run.pass(Ea, nxt.data()); }, [&] { run.pass(Eb, nxt.data() + h); });
                    } else run.pass(E, nxt.data());
                }
            });
            const double tdec = now() - tg0;
            if (strata_env_is_set("STRATA_DS41_RANKTIMERS")) {   // every rank its own timers (ms per step; its own output file)
                std::printf("RUNNER_DS41_RANKTIMERS rank=%d heads=%d", comm.rank, run.h1 - run.h0);
                for (const auto &[k, v] : run.timers) {
                    const auto it = timers0.find(k);
                    std::printf(" %s=%.3f", k.c_str(), 1e3 * (v - (it == timers0.end() ? 0.0 : it->second)) / std::max(generate, 1));
                }
                std::printf("\n");
                std::fflush(stdout);
            }
            if (comm.rank == 0) {
                std::printf("RUNNER_DS41_BATCH B=%d prompt=%d prefill_tok_per_s=%.1f gen=%d step_ms=%.3f tok_per_s=%.1f\n", B, P, double(B) * P / tpre, generate,
                            1e3 * tdec / std::max(generate, 1), double(B) * generate / tdec);
                std::printf("RUNNER_DS41_BATCHTIMERS");   // ms per pass of B tokens
                for (const auto &[k, v] : run.timers) {
                    const auto it = timers0.find(k);
                    std::printf(" %s=%.3f", k.c_str(), 1e3 * (v - (it == timers0.end() ? 0.0 : it->second)) / std::max(generate, 1));
                }
                std::printf("\n");
                for (const auto &[k, g] : g_gstat) {
                    const auto it = gstat0.find(k);
                    const GStat e0 = it == gstat0.end() ? GStat{} : it->second;
                    std::printf("RUNNER_DS41_BATCHGEMV %-14s rows_ms=%.3f prep_ms=%.3f GBps=%.1f\n", k.c_str(), 1e3 * (g.rows - e0.rows) / std::max(generate, 1),
                                1e3 * (g.prep - e0.prep) / std::max(generate, 1), g.rows > e0.rows ? (g.bytes - e0.bytes) / (g.rows - e0.rows) / 1e9 : 0.0);
                }
                if (fibers && B >= 2)
                    std::printf("RUNNER_DS41_FIB posts=%.1f alone=%.1f away_ms=%.3f wait_ms=%.3f comm_thread_busy_ms=%.3f\n", run.fib_posts_ / std::max(generate, 1),
                                run.fib_alone_ / std::max(generate, 1), 1e3 * run.fib_away_s_ / std::max(generate, 1), 1e3 * run.fib_wait_s_ / std::max(generate, 1),
                                1e3 * comm.ar_thread_busy / std::max(generate, 1));
                if (strata_env_is_set("STRATA_DS41_XPROF")) run.xprof_print(std::max(generate, 1));   // (ms per step)
                for (int b = 0; b < B; ++b) {
                    uint64_t hsh = 1469598103934665603ull;
                    for (int t : outs[size_t(b)]) hsh = (hsh ^ uint64_t(t)) * 1099511628211ull;
                    std::printf("RUNNER_DS41_BATCHSEQ %d chunk=%d hash=%016llx\n", b, kl_chunk + b, (unsigned long long) hsh);
                }
                std::printf("RUNNER_DS41_GENERATED");
                for (int t : outs[0]) std::printf(" %d", t);
                std::printf("\n");
                std::fflush(stdout);
            }
            comm.finish();
            return 0;
        }
        int pass_last = -1;
        if (prompt_pass > 0) {   // --prompt-pass T: the prompt in passes of T tokens, each position's argmax (no
            // scores): RUNNER_DS41_PASS with a hash of the argmax list (the same for every T when the pass is exact)
            run.prepare_prompt_eg_cache(ids);  // preload cost stays inside te..PASS timing
            std::vector<int> am(ids.size());
            const auto timers_pp = run.timers;
            const double gemv_pp = g_gemv_s;
            const bool prefill_fibers = strata_env_truthy_one("STRATA_DS41_PREFILL_FIBERS");
            if (prefill_fibers && strata_env_truthy_one("STRATA_DS41_ES_QKQ"))
                throw std::runtime_error("ds41: prefill fibers need separate QKQ scratch");
            uint64_t prefill_fiber_pairs = 0;
            with_team([&] {
                std::vector<Runner::Ent> E, Ea, Eb;
                for (size_t p0 = 0; p0 < ids.size(); p0 += size_t(prompt_pass)) {
                    const int T = int(std::min(ids.size() - p0, size_t(prompt_pass)));
                    // Experimental causal pipeline: the first half always starts first.
                    // Exact prompt/cache continuation gates are required before any speed claim.
                    if (prefill_fibers && comm.ar_async_ok() && T >= 64 && !(T % 2) && T <= run.pass_max()) {
                        const int h = T / 2;
                        Ea.resize(size_t(h)), Eb.resize(size_t(T - h));
                        for (int i = 0; i < h; ++i)
                            Ea[size_t(i)].s = &run.seq0, Ea[size_t(i)].pos = int(p0) + i, Ea[size_t(i)].token = ids[p0 + size_t(i)];
                        for (int i = h; i < T; ++i)
                            Eb[size_t(i - h)].s = &run.seq0, Eb[size_t(i - h)].pos = int(p0) + i, Eb[size_t(i - h)].token = ids[p0 + size_t(i)];
                        run.fib_pair([&] { run.pass(Ea, am.data() + p0); },
                                     [&] { run.pass(Eb, am.data() + p0 + size_t(h)); });
                        ++prefill_fiber_pairs;
                    } else {
                        E.resize(size_t(T));
                        for (int i = 0; i < T; ++i) E[size_t(i)].s = &run.seq0, E[size_t(i)].pos = int(p0) + i, E[size_t(i)].token = ids[p0 + size_t(i)];
                        run.pass(E, am.data() + p0);
                    }
                }
            });
            if (prefill_fibers) {
                comm.ar_shutdown();
                if (comm.rank == 0)
                    std::printf("RUNNER_DS41_PREFILL_FIBERS pairs=%llu posts=%.0f alone=%.0f wait_s=%.6f comm_busy_s=%.6f\n",
                                (unsigned long long) prefill_fiber_pairs, run.fib_posts_, run.fib_alone_, run.fib_wait_s_, comm.ar_thread_busy);
            }
            if (run.index_owned_calls_) {
                std::printf("RUNNER_DS41_INDEX_OWNED rank=%d calls=%llu verify_calls=%llu values=%llu qk_calls=%llu x_gathers=%llu weight_bytes=%zu\n",
                    comm.rank,(unsigned long long)run.index_owned_calls_,(unsigned long long)run.index_verify_calls_,
                    (unsigned long long)run.index_verified_values_,(unsigned long long)run.index_owner_qk_calls_,
                    (unsigned long long)run.index_verify_x_gathers_,run.index_full_bytes_);
                std::fflush(stdout);
            }
            run.finish_prompt_eg_cache();
            pass_last = am.back();
            uint64_t hsh = 1469598103934665603ull;
            for (size_t i = 0; i < am.size(); ++i) {
                hsh = (hsh ^ uint64_t(am[i])) * 1099511628211ull;
                if (i + 1 < ids.size()) ++scored, top1 += am[i] == ids[i + 1];
            }
            if (comm.rank == 0)
                std::printf("RUNNER_DS41_PASS T=%d tokens=%zu ms_per_token=%.3f tok_per_s=%.1f top1=%.6f hash=%016llx\n", prompt_pass, ids.size(),
                            1e3 * (now() - te) / double(ids.size()), double(ids.size()) / (now() - te), double(top1) / std::max(scored, 1),
                            (unsigned long long) hsh);
            if (comm.rank == 0) {   // ms per position by timer site (the pass-shape split: verify / batch costs per entry)
                std::printf("RUNNER_DS41_PASSTIMERS T=%d", prompt_pass);
                for (const auto &[k, v] : run.timers) {
                    const auto it = timers_pp.find(k);
                    std::printf(" %s=%.3f", k.c_str(), 1e3 * (v - (it == timers_pp.end() ? 0.0 : it->second)) / double(ids.size()));
                }
                std::printf(" gemv=%.3f\n", 1e3 * (g_gemv_s - gemv_pp) / double(ids.size()));
                if (run.index_owned_calls_)
                    std::printf("RUNNER_DS41_INDEX_OWNER calls=%llu verify_calls=%llu verify_values_local=%llu weight_GB=%.6f\n",
                                (unsigned long long) run.index_owned_calls_, (unsigned long long) run.index_verify_calls_,
                                (unsigned long long) run.index_verified_values_, double(run.index_full_bytes_) / 1e9);
                if (run.index_owner_qk_calls_)
                    std::printf("RUNNER_DS41_INDEX_OWNER_QK calls=%llu verify_x_gathers=%llu\n",
                                (unsigned long long) run.index_owner_qk_calls_,
                                (unsigned long long) run.index_verify_x_gathers_);
            }
        } else
        with_team([&] {
        for (size_t pos = 0; pos < ids.size(); ++pos) {
            logits = run.step(ids[pos], int(pos));
            const int am = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (!std::isfinite(logits[size_t(am)])) throw std::runtime_error("ds41: non-finite logits at " + std::to_string(pos));
            if (lf) lf.write(reinterpret_cast<const char *>(logits.data()), std::streamsize(logits.size() * 4));
            if (kb && comm.rank == 0 && int(pos) >= kb->n_ctx / 2 && int(pos) < kb->n_ctx - 1 && pos + 1 < ids.size()) {   // KL(ref || ours)
                const auto lp = kb->log_probs(kl_chunk, int(pos) - kb->n_ctx / 2);
                const float mx = logits[size_t(am)];
                double se = 0;
                for (float v : logits) se += std::exp(double(v - mx));
                const double lse = std::log(se) + mx;
                double kl = 0;
                int ra = 0;
                for (int v = 0; v < kb->n_vocab; ++v) {
                    const double pr = std::exp(lp[size_t(v)]);
                    kl += pr * (lp[size_t(v)] - (double(logits[size_t(v)]) - lse));
                    if (lp[size_t(v)] > lp[size_t(ra)]) ra = v;
                }
                kl_sum += kl, ++kl_n, kl_top += ra == am;
                kl_nll_ref -= lp[size_t(ids[pos + 1])], kl_nll -= double(logits[size_t(ids[pos + 1])]) - lse;
            }
            if (pos + 1 < ids.size()) {
                const float mx = logits[size_t(am)];
                double se = 0;
                for (float v : logits) se += std::exp(double(v - mx));
                nll += std::log(se) - double(logits[size_t(ids[pos + 1])] - mx);
                ++scored, top1 += am == ids[pos + 1];
            }
            if (comm.rank == 0 && (pos % 16 == 0 || pos + 1 == ids.size()))
                std::cerr << "DS41 pos " << pos << " token " << ids[pos] << " argmax " << am << " ppl " << (scored ? std::exp(nll / scored) : 0.0)
                          << (kl_n ? " kl " + std::to_string(kl_sum / kl_n) + " top1_ref " + std::to_string(double(kl_top) / kl_n) : std::string())
                          << " ms_per_token " << 1e3 * (now() - te) / double(pos + 1) << "\n";
        }
        });
        const double es = now() - te;
        std::vector<int> gen;
        int pos = int(ids.size());
        // decode-only breakdown: the timers from here on (RUNNER_DS41_GENTIMERS, rank 0, ms per generated token)
        const auto timers_eval = run.timers;
        g_skew_on = strata_env_is_set("STRATA_DS41_SKEWPROF");
        g_barprof = strata_env_is_set("STRATA_DS41_BARPROF");
        for (auto &b : g_barp) b = BarProf{};
        g_region_s = g_regions = 0;
        run.rgn_n.clear(), run.rgn_s.clear(), run.sub_p.clear();
        const auto gstat_eval = g_gstat;
        const double gemv_eval = g_gemv_s;
        const double tg = now();
        long spec_passes = 0, spec_drafted = 0, spec_accepted = 0;
        double dsp_verify_s = 0, dsp_width_s[16] = {0};
        long dsp_width[16] = {0}, dsp_hist[16] = {0}, dsp_cut = 0;
        // STRATA_DS41_DSPARK_WTIMERS=1: the target's phase timers around each verify pass, summed per width (printed at the end)
        static const bool dsp_wt_on = strata_env_truthy_one("STRATA_DS41_DSPARK_WTIMERS");
        std::map<std::string, double> dsp_wt[16];
        with_team([&] {
            int tok = pass_last >= 0 ? pass_last : int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            comm.bcast(&tok, 1);
            if (drun) {   // --dspark: B = --dspark-n drafts per step from the drafter, verified in one pass (greedy: the
                // tokens are plain decode's); the accepted positions' features then become the drafter's keys
                Runner &D = *drun;
                run.spec_track_ = true;
                const int m0 = std::min({pos, D.c.window, Runner::kFeatRing});
                D.inject(run, pos - m0, m0);
                gen.push_back(tok);
                std::vector<Runner::Ent> E;
                std::vector<int> am(static_cast<size_t>(Runner::kMaxPass));
                std::vector<Vec> sv(static_cast<size_t>(run.layers)), ss(static_cast<size_t>(run.layers));
                // STRATA_DS41_DSPARK_SCHED=1: of the drafts, verify the prefix with the most expected tokens per ms (the
                // GPU decoder's sched_prefix): expected tokens of verifying j = 1 + sum_{i<=j} prod_{t<=i} conf_t; cost =
                // the draft's ms + the measured verify ms of width j + 1 (running means, weight 0.2; unmeasured widths
                // from the line through the measured ones). The first 3 steps verify everything (widths get measured)
                // STRATA_DS41_DSPARK_SCHED=2: each width's cost the median of its last 8 verify times, and a width not
                // measured in the last 32 steps (or never) from the least-squares line through the fresh ones (a width
                // that once looked slow, e.g. in a network spike, is not kept out for good)
                static const int sched_mode = std::getenv("STRATA_DS41_DSPARK_SCHED") ? std::atoi(std::getenv("STRATA_DS41_DSPARK_SCHED")) : 0;
                static const bool sched = sched_mode == 1 || sched_mode == 2;
                double vms[16] = {0}, dms = 0;
                double vring[16][8] = {{0}};
                int vn[16] = {0};
                long vlast[16] = {0};
                auto refresh_vms = [&](int width, double ms) {   // (sched 2) the median model, written into vms
                    vring[width][vn[width] % 8] = ms, ++vn[width], vlast[width] = spec_passes;
                    for (int t = 1; t < 16; ++t) {
                        const int k = std::min(vn[t], 8);
                        if (!k || spec_passes - vlast[t] > 32) { vms[t] = 0; continue; }
                        double v[8];
                        std::copy(vring[t], vring[t] + k, v);
                        std::sort(v, v + k);
                        vms[t] = k % 2 ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
                    }
                };
                auto sched_prefix = [&](const std::vector<float> &cf, int kmax) {
                    if (kmax <= 0 || spec_passes < 3) return kmax;
                    double sx = 0, sy = 0, sxx = 0, sxy = 0;
                    int nm = 0, t0 = 0;
                    for (int t = 1; t < 16; ++t)
                        if (vms[t] > 0) { sx += t, sy += vms[t], sxx += double(t) * t, sxy += t * vms[t], ++nm; if (!t0) t0 = t; }
                    if (!nm) return kmax;
                    double slope = 0.15 * vms[t0], icpt = vms[t0] - slope * t0;
                    if (nm >= 2) slope = (nm * sxy - sx * sy) / std::max(1e-9, nm * sxx - sx * sx), icpt = (sy - slope * sx) / nm;
                    auto cost = [&](int t) { return t < 16 && vms[t] > 0 ? vms[t] : std::max(1e-3, icpt + slope * t); };
                    int best = 0;
                    double best_rate = 1.0 / (dms + cost(1)), expect = 1.0, prod = 1.0;
                    for (int j = 1; j <= kmax; ++j) {
                        prod *= double(cf[size_t(j - 1)]), expect += prod;
                        const double rate = expect / (dms + cost(j + 1));
                        if (rate > best_rate) best_rate = rate, best = j;
                    }
                    return best;
                };
                while (int(gen.size()) < generate) {
                    const int cap = std::min({dspark_n, generate - int(gen.size()), run.context - pos - 1, Runner::kMaxPass - 1});
                    const double td0 = now();
                    std::vector<int> d = cap > 0 ? D.draft(tok, pos, cap) : std::vector<int>{};
                    dms = dms > 0 ? 0.8 * dms + 0.2 * 1e3 * (now() - td0) : 1e3 * (now() - td0);
                    if (sched && D.draft_conf_.size() == d.size()) {
                        int keep = sched_prefix(D.draft_conf_, int(d.size()));
                        comm.bcast(&keep, 1);   // (from rank 0's timings: every rank must verify the same prefix)
                        dsp_cut += long(d.size()) - keep;
                        d.resize(size_t(keep));
                    }
                    const double tv = now();
                    const int T = 1 + int(d.size());
                    E.resize(size_t(T));
                    for (int i = 0; i < T; ++i) E[size_t(i)].s = &run.seq0, E[size_t(i)].pos = pos + i, E[size_t(i)].token = i ? d[size_t(i - 1)] : tok;
                    for (int l = 0; l < run.layers; ++l) sv[size_t(l)] = run.seq0.stash_v[size_t(l)], ss[size_t(l)] = run.seq0.stash_s[size_t(l)];
                    const std::map<std::string, double> wt0 = dsp_wt_on ? run.timers : std::map<std::string, double>();
                    run.pass(E, am.data());
                    if (dsp_wt_on && T < 16)
                        for (const auto &[k, v] : run.timers) { const auto it = wt0.find(k); dsp_wt[T][k] += v - (it == wt0.end() ? 0.0 : it->second); }
                    int m = 0;
                    while (m < int(d.size()) && am[size_t(m)] == d[size_t(m)]) ++m;
                    run.rewind_stash(run.seq0, E, m, sv, ss);
                    dsp_verify_s += now() - tv;
                    if (T < 16) {
                        dsp_width[T] += 1, dsp_width_s[T] += now() - tv;
                        if (sched_mode == 2) refresh_vms(T, 1e3 * (now() - tv));
                        else vms[T] = vms[T] > 0 ? 0.8 * vms[T] + 0.2 * 1e3 * (now() - tv) : 1e3 * (now() - tv);
                    }
                    D.inject(run, pos, m + 1);
                    for (int k = 0; k < m; ++k) gen.push_back(d[size_t(k)]);
                    tok = am[size_t(m)];
                    gen.push_back(tok);
                    pos += m + 1;
                    ++spec_passes, spec_drafted += long(d.size()), spec_accepted += m;
                    if (m < 16) ++dsp_hist[m];
                }
                gen.resize(size_t(generate));
                return;
            }
            if (spec > 0) {   // --spec K: n-gram (prompt-lookup) drafts of up to K tokens, verified in one pass; greedy, so the
                // tokens are plain decode's (every rank drafts the same from the same history)
                run.spec_track_ = true;
                std::vector<int> histv(ids.begin(), ids.end());
                gen.push_back(tok), histv.push_back(tok);
                std::vector<Runner::Ent> E;
                std::vector<int> am(static_cast<size_t>(Runner::kMaxPass));
                std::vector<Vec> sv(static_cast<size_t>(run.layers)), ss(static_cast<size_t>(run.layers));
                while (int(gen.size()) < generate) {
                    std::vector<int> d;   // the drafts: what followed the latest earlier occurrence of the last n tokens
                    const int H = int(histv.size()), K = std::min({spec, Runner::kMaxPass - 1, run.context - pos - 1});
                    for (int ng = std::min(spec_ngram, H - 1); ng >= 1 && d.empty(); --ng)
                        for (int st = H - ng - 1; st >= 0; --st) {
                            bool eq = true;
                            for (int k = 0; k < ng && eq; ++k) eq = histv[size_t(st + k)] == histv[size_t(H - ng + k)];
                            if (!eq) continue;
                            for (int k = st + ng; k < H && int(d.size()) < K; ++k) d.push_back(histv[size_t(k)]);
                            break;
                        }
                    const int T = 1 + int(d.size());
                    E.resize(size_t(T));
                    for (int i = 0; i < T; ++i) E[size_t(i)].s = &run.seq0, E[size_t(i)].pos = pos + i, E[size_t(i)].token = i ? d[size_t(i - 1)] : tok;
                    for (int l = 0; l < run.layers; ++l) sv[size_t(l)] = run.seq0.stash_v[size_t(l)], ss[size_t(l)] = run.seq0.stash_s[size_t(l)];
                    run.pass(E, am.data());
                    int m = 0;
                    while (m < int(d.size()) && am[size_t(m)] == d[size_t(m)]) ++m;
                    run.rewind_stash(run.seq0, E, m, sv, ss);
                    for (int k = 0; k < m; ++k) gen.push_back(d[size_t(k)]), histv.push_back(d[size_t(k)]);
                    tok = am[size_t(m)];
                    gen.push_back(tok), histv.push_back(tok);
                    pos += m + 1;
                    ++spec_passes, spec_drafted += long(d.size()), spec_accepted += m;
                }
                gen.resize(size_t(generate));
                return;
            }
            for (int g = 0; g < generate; ++g) {   // every rank gets the same argmax: no broadcast per token
                gen.push_back(tok);
                int next = 0;
                run.step(tok, pos++, &next);
                tok = next;
            }
        });
        if (comm.rank == 0 && drun) {
            const double ps = double(std::max(spec_passes, 1L));
            std::printf("RUNNER_DS41_DSPARK n=%d steps=%ld drafted=%ld accepted=%ld accept_rate=%.3f tokens_per_step=%.3f draft_ms=%.3f verify_ms=%.3f "
                        "inject_ms=%.3f cut=%ld verify_ms_by_width=", dspark_n, spec_passes, spec_drafted, spec_accepted, double(spec_accepted) / std::max(spec_drafted, 1L),
                        double(generate) / ps, 1e3 * drun->timers["dsp.draft"] / ps, 1e3 * dsp_verify_s / ps, 1e3 * drun->timers["dsp.inject"] / ps, dsp_cut);
            for (int w = 1; w < 9; ++w) std::printf("%s%.2f", w > 1 ? "," : "", dsp_width[w] ? 1e3 * dsp_width_s[w] / double(dsp_width[w]) : 0.0);
            std::printf(" accept_hist=");
            for (int k = 0; k <= dspark_n && k < 16; ++k) std::printf("%s%ld", k ? "," : "", dsp_hist[k]);
            std::printf("\n");
            if (dsp_wt_on)
                for (int w = 1; w < 9; ++w) {
                    if (!dsp_width[w]) continue;
                    std::printf("RUNNER_DS41_WTIMERS width=%d passes=%ld", w, dsp_width[w]);   // ms per pass, phases of 0.2 ms or more
                    for (const auto &[k, v] : dsp_wt[w])
                        if (1e3 * v / double(dsp_width[w]) >= 0.2) std::printf(" %s=%.3f", k.c_str(), 1e3 * v / double(dsp_width[w]));
                    std::printf("\n");
                }
            std::printf("RUNNER_DS41_DSPARKTIMERS");   // the drafter's timers, ms per step
            for (const auto &[k, v] : drun->timers) std::printf(" %s=%.3f", k.c_str(), 1e3 * v / ps);
            std::printf("\n");
        }
        if (comm.rank == 0 && spec > 0)
            std::printf("RUNNER_DS41_SPEC K=%d ngram=%d passes=%ld drafted=%ld accepted=%ld tokens_per_pass=%.3f\n", spec, spec_ngram, spec_passes, spec_drafted,
                        spec_accepted, double(generate) / std::max(spec_passes, 1L));
        if (comm.rank == 0 && strata_env_is_set("STRATA_DS41_XPROF")) run.xprof_print(int(ids.size() + gen.size()));
        if (g_skew_on && generate > 0) skew_report(comm, generate);
        if (g_barprof && generate > 0) {   // every rank (its own out file): per generated token
            const int T = std::min(64, omp_get_max_threads());
            double mn = 1e30, mx = 0, sum = 0;
            int amin = 0, amax = 0;
            for (int t = 0; t < T; ++t) {
                const double v = g_barp[t].wait;
                if (v < mn) mn = v, amin = t;
                if (v > mx) mx = v, amax = t;
                sum += v;
            }
            std::printf("RUNNER_DS41_BARPROF rank=%d gen_ms_per_token=%.3f regions_per_token=%.1f region_ms=%.3f barriers_per_token=%.1f wait_ms min=%.3f (t%d) mean=%.3f max=%.3f (t%d)\n",
                        comm.rank, 1e3 * (now() - tg) / generate, g_regions / generate, 1e3 * g_region_s / generate, g_barp[1].n / generate, 1e3 * mn / generate, amin,
                        1e3 * sum / T / generate, 1e3 * mx / generate, amax);
            if (comm.rank == 0) {   // per phase (head, embedding, spec loop and other regions outside the layers: rest)
                double inl = 0;
                std::printf("RUNNER_DS41_REGIONS per_token");
                for (const auto &[k, v] : run.rgn_n) std::printf(" %s=%.1f", k.c_str(), v / generate), inl += v;
                std::printf(" rest=%.1f total=%.1f layers=%d\n", (g_regions - inl) / generate, g_regions / generate, run.layers);
                const int TB = std::min(64, omp_get_max_threads());
                for (const auto &[k, v] : run.sub_p)
                    std::printf("RUNNER_DS41_SUBREGIONS %-15s regions=%.1f region_ms=%.3f wait_ms_per_thread=%.3f\n", k.c_str(), v.n / generate, 1e3 * v.s / generate,
                                1e3 * v.w / TB / generate);
                std::printf("RUNNER_DS41_REGIONS region_ms_per_token");   // thread 0 inside regions per phase (the rest: serial + comm)
                for (const auto &[k, v] : run.rgn_s) std::printf(" %s=%.3f", k.c_str(), 1e3 * v / generate);
                std::printf("\n");
            }
            std::fflush(stdout);
        }
        if (comm.rank == 0 && generate > 0) {
            std::printf("RUNNER_DS41_GENTIMERS");
            for (const auto &[k, v] : run.timers) {
                const auto it = timers_eval.find(k);
                std::printf(" %s=%.3f", k.c_str(), 1e3 * (v - (it == timers_eval.end() ? 0.0 : it->second)) / generate);
            }
            std::printf(" gemv=%.3f\n", 1e3 * (g_gemv_s - gemv_eval) / generate);
            for (const auto &[k, g] : g_gstat) {
                const auto it = gstat_eval.find(k);
                const GStat e = it == gstat_eval.end() ? GStat{} : it->second;
                std::printf("RUNNER_DS41_GENGEMV %-14s rows_ms=%.3f GBps=%.1f\n", k.c_str(), 1e3 * (g.rows - e.rows) / generate,
                            g.rows > e.rows ? (g.bytes - e.bytes) / (g.rows - e.rows) / 1e9 : 0.0);
            }
            std::fflush(stdout);
        }
        if (comm.rank == 0 && generate > 0)   // plain decode: argmax + broadcast per token, no scoring
            std::printf("RUNNER_DS41_GEN tokens=%d from_pos=%d ms_per_token=%.3f tok_per_s=%.2f\n", generate, int(ids.size()),
                        1e3 * (now() - tg) / generate, generate / (now() - tg));
        {   // every rank (its own out file): where a slow rank shows up as low comm (the others wait for it)
            const double nt = double(ids.size() + gen.size());
            std::printf("RUNNER_DS41_RANK %d ms_per_token=%.3f comm=%.3f gemv=%.3f mhc=%.3f ffn.experts=%.3f attn.core=%.3f\n", comm.rank,
                        1e3 * es / double(std::max<size_t>(ids.size(), 1)),
                        1e3 * run.timers["comm"] / nt, 1e3 * g_gemv_s / nt, 1e3 * run.timers["mhc"] / nt, 1e3 * run.timers["ffn.experts"] / nt,
                        1e3 * run.timers["attn.core"] / nt);
            std::printf("RUNNER_DS41_RANKT %d", comm.rank);
            for (const auto &[k, v] : run.timers) std::printf(" %s=%.3f", k.c_str(), 1e3 * v / nt);
            std::printf("\n");
            std::fflush(stdout);
        }
        if (comm.rank == 0) {
            std::cout << "RUNNER_DS41 ranks=" << comm.world << " layers=" << run.layers << " tokens=" << ids.size() << " ppl="
                      << (scored ? std::exp(nll / scored) : 0.0) << " top1=" << (scored ? double(top1) / scored : 0.0) << " ms_per_token="
                      << 1e3 * es / double(std::max<size_t>(ids.size(), 1)) << "\n";
            if (kl_n)
                std::cout << "RUNNER_DS41_KL chunk=" << kl_chunk << " positions=" << kl_n << " kl=" << kl_sum / kl_n << " top1_agree="
                          << double(kl_top) / kl_n << " ppl_ref=" << std::exp(kl_nll_ref / kl_n) << " ppl=" << std::exp(kl_nll / kl_n) << "\n";
            if (run.bal_n)
                std::cout << "RUNNER_DS41_BALANCE busiest_routed=" << run.bal_routed / run.bal_n << " busiest_total=" << run.bal_total / run.bal_n
                          << " mean=" << run.bal_mean / run.bal_n << " shared_balance=" << !run.L[0].shared_all.empty() << " shared_load_s=" << run.shared_load_s << "\n";
            std::cout << "RUNNER_DS41_TIMERS";
            for (const auto &[k, v] : run.timers) std::cout << " " << k << "=" << 1e3 * v / double(ids.size() + gen.size());
            std::cout << " gemv=" << 1e3 * g_gemv_s / double(ids.size() + gen.size()) << "\n";
            for (const auto &[k, g] : g_gstat)   // per token: calls, ms prep, ms rows, MB, GB/s over the team pass
                std::printf("RUNNER_DS41_GEMV %-14s calls=%.1f prep_ms=%.3f rows_ms=%.3f MB=%.2f GBps=%.1f\n", k.c_str(), g.calls / double(ids.size() + gen.size()),
                            1e3 * g.prep / double(ids.size() + gen.size()), 1e3 * g.rows / double(ids.size() + gen.size()),
                            g.bytes / 1e6 / double(ids.size() + gen.size()), g.rows > 0 ? g.bytes / g.rows / 1e9 : 0.0);
            std::fflush(stdout);
            if (!gen.empty()) {
                std::cout << "RUNNER_DS41_GENERATED";
                for (int t : gen) std::cout << " " << t;
                std::cout << "\n";
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "ds41_cpu_runner rank " << comm.rank << ": " << e.what() << "\n";
        code = 1;
    }
    comm.finish();
    return code;
}
