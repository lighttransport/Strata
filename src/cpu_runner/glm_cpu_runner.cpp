// src/cpu_runner/glm_cpu_runner.cpp - GLM-5.3-Flash FP4-Local on CPU nodes (A64FX first), one process per node.
//
// The trunk is tensor-parallel over the ranks (KDA and MLA heads, dense-FFN and shared-expert rows, lm_head rows);
// every routed expert is sliced over up to 4 ranks (gate/up rows, matching down columns); routers, mHC, norms and
// the MLA latent/indexer path are computed on every rank from bitwise-identical inputs. Two collectives per layer:
// the mixer and the FFN partial outputs, reduced deterministically (reduce-scatter then all-gather) so every rank
// holds the same residual. Weights stay in the checkpoint's numbers: `exact` streams E2M1 x E4M3 experts and E4M3
// trunk tensors with F32 activations; `fast` streams W4A8 experts and requantizes the E4M3 trunk to int8 per (row, 32)
// (kernels: src/kernels/cpu/glm_fp4_sve.c). The forward follows tools/glm_fp4_cpu_reference.cpp, the golden reference.
//
//   glm_cpu_runner MODEL_DIR --eval IDS [--numerics exact|fast] [--context N] [--tokens N] [--layers N]
//                  [--save-logits FILE] [--reference GOLDEN_DIR] [--dump-at T,...] [--dump-dir DIR] [--threads N]
//                  [--image-dir DIR]   (per-rank packed image: written on the first load, replayed afterwards)
//
// Build (MPI optional; without it the runner is one rank):
//   c++ -std=c++20 -O3 -fopenmp -Iinclude src/cpu_runner/glm_cpu_runner.cpp src/kernels/cpu/glm_fp4_sve.c
//       src/host/glm_fp4.c src/host/glm_names.c src/host/glm_env.c src/host/glm_expert_pack.c -o glm_cpu_runner
//   A64FX: clang++ -stdlib=libc++ -mcpu=a64fx -msve-vector-bits=512 ... -DSTRATA_CPU_RUNNER_MPI (MPI wrapper)
#include "strata/core/model.hpp"
#include "strata/kernels/cpu/glm_fp4_sve.h"
#include "strata/kernels/glm_reference.hpp"

#include <omp.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <type_traits>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <thread>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#ifdef STRATA_CPU_RUNNER_MPI
#include <mpi.h>
#endif
#ifdef STRATA_CPU_RUNNER_UTOFU
extern "C" int cpu_utofu_init(int rank, int world, int max_count);
extern "C" int cpu_utofu_max(void);   // the longest vector one call takes (the big region's count; 0: tree mode)
extern "C" void cpu_utofu_allreduce(float *x, int n);
extern "C" const char *cpu_utofu_mode(void);
extern "C" int cpu_utofu_banks(int nb, int max);   // staged all-reduce regions (collective)
extern "C" int cpu_utofu_bank_max(void);
extern "C" void cpu_utofu_shards(int count, int *off);
extern "C" float *cpu_utofu_bank_send(int bank, int d);
extern "C" const float *cpu_utofu_bank_result(int bank, int d);
extern "C" void cpu_utofu_bank_allreduce(int bank, int count);
#else   // no staged banks without uTofu (Comm::banks stays 0, so these are never called)
static void cpu_utofu_shards(int, int *) {}
static float *cpu_utofu_bank_send(int, int) { return nullptr; }
static const float *cpu_utofu_bank_result(int, int) { return nullptr; }
static void cpu_utofu_bank_allreduce(int, int) {}
#endif

namespace ref = strata::kernels::glm_reference;

// The hot exp / sigmoid / SiLU / SwiGLU loops through the vector kernels of glm_fp4_sve.c (FEXPA based, 1-6 ulp from
// libm; default since 2026-10-24: 8K eval inside the noise band of single-site perturbations, ppl 56.03 / next-token
// 3521 against 56.44-58.06 / 3489-3506; STRATA_RUNNER_FAST_EXP=0: libm, the earlier bitwise results)
// STRATA_RUNNER_FAST_EXP_MASK limits it to sites: 1 KDA, 2 SwiGLU, 4 router, 8 MLA softmax (default all)
// STRATA_RUNNER_ATTN_F32=1: MLA attention scores and latent averages in fp32 (approximate; the gather path, contexts
// beyond the selection size, and the latent average everywhere)
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
typedef svfloat32_t vf32 __attribute__((arm_sve_vector_bits(512)));
#endif
bool attn_f32() {
    static const bool on = std::getenv("STRATA_RUNNER_ATTN_F32") && std::string(std::getenv("STRATA_RUNNER_ATTN_F32")) == "1";
    return on;
}
// STRATA_RUNNER_INDEX_SKIP=0: score the indexer's pools even when the selection keeps all of them
// pipelined prefill passes (step_multi_pipe), default since 2026-10-25: 2K 218 vs 202 tok/s, 8K 199 vs 181, bitwise
bool prefill_pipe() {
    static const bool on = !(std::getenv("STRATA_RUNNER_PREFILL_PIPE") && std::string(std::getenv("STRATA_RUNNER_PREFILL_PIPE")) == "0");
    return on;
}
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
// the 8 horizontal sums of v[0..8) (lane j: svaddv(v[j])) by a transpose tree in FADDV's recursive pairwise order
// ((e0 + e1) + (e2 + e3)) + ((e4 + e5) + (e6 + e7)): bit for bit svaddv's results (checked on A64FX over 24K dots), in
// 21 vector operations instead of 8 reductions (STRATA_RUNNER_TREE_SUM=0: svaddv)
typedef svfloat64_t vf64t __attribute__((arm_sve_vector_bits(512)));
static inline svfloat64_t hsum8_f64(const vf64t *v) {
    const svbool_t pd = svptrue_b64();
    vf64t l1[4], l2[2];
    for (int j = 0; j < 4; ++j) l1[j] = svadd_f64_x(pd, svuzp1_f64(v[2 * j], v[2 * j + 1]), svuzp2_f64(v[2 * j], v[2 * j + 1]));
    for (int j = 0; j < 2; ++j) l2[j] = svadd_f64_x(pd, svuzp1_f64(l1[2 * j], l1[2 * j + 1]), svuzp2_f64(l1[2 * j], l1[2 * j + 1]));
    return svadd_f64_x(pd, svuzp1_f64(l2[0], l2[1]), svuzp2_f64(l2[0], l2[1]));
}
bool tree_sum_on() {
    static const bool on = !(std::getenv("STRATA_RUNNER_TREE_SUM") && std::string(std::getenv("STRATA_RUNNER_TREE_SUM")) == "0");
    return on;
}
#endif
int avg_heads() {   // the prefix average's heads per item (STRATA_RUNNER_AVG_HEADS: 1 or 2, default 2)
    static const int v = std::getenv("STRATA_RUNNER_AVG_HEADS") ? std::clamp(std::atoi(std::getenv("STRATA_RUNNER_AVG_HEADS")), 1, 2) : 2;
    return v;
}
bool gather_block_on() {   // MLA scores beyond the selection size in (token, 3 heads, 4 positions) blocks
    static const bool on = !(std::getenv("STRATA_RUNNER_GATHER_BLOCK") && std::string(std::getenv("STRATA_RUNNER_GATHER_BLOCK")) == "0");
    return on;
}
bool index_skip() {
    static const bool on = !(std::getenv("STRATA_RUNNER_INDEX_SKIP") && std::string(std::getenv("STRATA_RUNNER_INDEX_SKIP")) == "0");
    return on;
}
bool fast_exp(int site = 15) {
    static const bool on = !(std::getenv("STRATA_RUNNER_FAST_EXP") && std::string(std::getenv("STRATA_RUNNER_FAST_EXP")) == "0");
    static const int mask = std::getenv("STRATA_RUNNER_FAST_EXP_MASK") ? std::atoi(std::getenv("STRATA_RUNNER_FAST_EXP_MASK")) : 15;
    return on && (mask & site);
}

// ref::mhc with the Sinkhorn iterations on one 16-lane vector (the 4 x 4 matrix, lane 4i + j): each lane's row or
// column sum is eps plus the four elements in ref::mhc's order (svtbl brings element j / i of its row / column to
// every lane), then one vector division, so every value has ref::mhc's bits. ref::mhc ran 32 dependent scalar
// divisions per iteration on the thread that holds every site's critical path (STRATA_RUNNER_MHC_SVE=0: ref::mhc)
void mhc_coef(const float *p, const float *base, const float *scale, float *c, int iters, float eps) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    static const bool on = !(std::getenv("STRATA_RUNNER_MHC_SVE") && std::string(std::getenv("STRATA_RUNNER_MHC_SVE")) == "0");
    if (on) {
        for (int i = 0; i < 4; ++i) {   // ref::mhc's first steps, unchanged
            c[i] = ref::sigmoid(p[i] * scale[0] + base[i]) + eps;
            c[4 + i] = 2 * ref::sigmoid(p[4 + i] * scale[1] + base[4 + i]);
        }
        for (int i = 0; i < 4; ++i) {
            float mx = -INFINITY, sum = 0;
            for (int j = 0; j < 4; ++j) mx = std::max(mx, p[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]);
            for (int j = 0; j < 4; ++j) {
                c[8 + i * 4 + j] = std::exp(p[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j] - mx);
                sum += c[8 + i * 4 + j];
            }
            for (int j = 0; j < 4; ++j) c[8 + i * 4 + j] = c[8 + i * 4 + j] / sum + eps;
        }
        const svbool_t pw = svptrue_b32();
        const svuint32_t lane = svindex_u32(0, 1), row = svlsl_n_u32_x(pw, svlsr_n_u32_x(pw, lane, 2), 2), col = svand_n_u32_x(pw, lane, 3);
        typedef svuint32_t vu32 __attribute__((arm_sve_vector_bits(512)));
        vu32 ri[4], ci[4];
        for (int k = 0; k < 4; ++k) ri[k] = svadd_n_u32_x(pw, row, k), ci[k] = svadd_n_u32_x(pw, col, 4 * k);
        svfloat32_t M = svld1_f32(pw, c + 8);
        const svfloat32_t e = svdup_n_f32(eps);
        for (int n = 0; n < iters; ++n) {
            if (n > 0) {
                svfloat32_t sum = e;
                for (int k = 0; k < 4; ++k) sum = svadd_f32_x(pw, sum, svtbl_f32(M, ri[k]));
                M = svdiv_f32_x(pw, M, sum);
            }
            svfloat32_t sum = e;
            for (int k = 0; k < 4; ++k) sum = svadd_f32_x(pw, sum, svtbl_f32(M, ci[k]));
            M = svdiv_f32_x(pw, M, sum);
        }
        svst1_f32(pw, c + 8, M);
        return;
    }
#endif
    ref::mhc(p, base, scale, c, iters, eps);
}
using strata::artifact::GlmModelOpt;
using strata::artifact::ModelOptFormat;
using strata::artifact::ModelOptPacked;
using Vec = std::vector<float>;

namespace {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// FPCR.FZ on the calling thread: single/double subnormals become zero. The KDA state decays by up to e^-5 per token and
// its smallest entries pass through the subnormal range, which A64FX processes far slower than normal numbers. Half
// precision (FZ16, the E4M3 decode's subnormals) is not touched; the decode is re-checked after setting it.
void flush_subnormals() {
#if defined(__aarch64__)
    uint64_t f;
    asm volatile("mrs %0, fpcr" : "=r"(f));
    f |= uint64_t(1) << 24;
    asm volatile("msr fpcr, %0" ::"r"(f));
#endif
}

// ------------------------------------------------------------------------------------------------- team regions
// A team region is team_run(f): f runs on every thread of the team (`omp for`, `omp single` and team_barrier() inside
// bind to it). Without an open team that is an OpenMP parallel region (fork + join 4.2-4.4 us at 47 threads). Inside
// with_team(work) (default; STRATA_RUNNER_PERSISTENT=0 forks per region) one region stays open: thread 0 runs `work`,
// the other threads wait for a generation flag, and team_run publishes f, runs it with them and ends with the team's
// barrier (~1.6 us). Serial code of `work` must not contain worksharing constructs or team barriers (thread 0 alone
// would meet the open team's); every decode-time region is a team_run.
inline void team_barrier();
struct TeamDispatch {
    alignas(256) std::atomic<uint64_t> gen{0};
    alignas(256) void (*fn)(void *) = nullptr;
    void *ctx = nullptr;
    bool stop = false;
    bool open = false;   // with_team's region is open
    bool busy = false;   // thread 0 is inside a dispatched f (a nested team_run then forks as OpenMP would)
};
TeamDispatch g_team;
template <class F> void team_run(F &&f) {
    if (g_team.open && !g_team.busy && omp_get_level() == 1 && omp_get_thread_num() == 0) {
        using Fn = std::remove_reference_t<F>;
        g_team.fn = [](void *c) { (*static_cast<Fn *>(c))(); };
        g_team.ctx = const_cast<void *>(static_cast<const void *>(&f));
        g_team.busy = true;
        g_team.gen.fetch_add(1, std::memory_order_release);
        f();
        team_barrier();
        g_team.busy = false;
        return;
    }
#pragma omp parallel
    f();
}
bool persistent_team() {
    static const bool on = !(std::getenv("STRATA_RUNNER_PERSISTENT") && std::string(std::getenv("STRATA_RUNNER_PERSISTENT")) == "0");
    return on;
}
// `work` on thread 0 of one open region (persistent_team(): 22.2 vs 23.6 ms per token on 12 nodes), or plainly
template <class W> void with_team(W &&work) {
    if (!persistent_team() || omp_in_parallel()) { work(); return; }
    std::exception_ptr err;
    g_team.stop = false, g_team.busy = false;
    const uint64_t g0 = g_team.gen.load();
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            g_team.open = true;
            try { work(); } catch (...) { err = std::current_exception(); }
            g_team.open = false;
            g_team.stop = true;
            g_team.gen.fetch_add(1, std::memory_order_release);
        } else {
            uint64_t seen = g0;
            for (;;) {
                uint64_t g;
                while ((g = g_team.gen.load(std::memory_order_acquire)) == seen) {}
                seen = g;
                if (g_team.stop) break;
                g_team.fn(g_team.ctx);
                team_barrier();
            }
        }
    }
    if (err) std::rethrow_exception(err);
}

// ------------------------------------------------------------------------------------------------- communication
struct Comm {
    int rank = 0, world = 1;
    bool utofu = false;   // deterministic uTofu tree all-reduce (comm_utofu.c) instead of MPI
    int ar_max = 16384;   // floats per uTofu all-reduce call (STRATA_RUNNER_AR_MAX); longer vectors go in pieces
    double seconds = 0;
    std::vector<float> scratch;
    void init(int *argc, char ***argv) {
#ifdef STRATA_CPU_RUNNER_MPI
        int provided = 0;
        MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
#ifdef STRATA_CPU_RUNNER_UTOFU
        const char *mode = std::getenv("STRATA_RUNNER_COMM");
        if (world > 1 && !(mode && std::string(mode) == "mpi")) {
            if (const char *e = std::getenv("STRATA_RUNNER_AR_MAX")) ar_max = std::max(4096, std::atoi(e));
            const int rc = cpu_utofu_init(rank, world, ar_max);
            int ok = rc == 0, all = 0;
            MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            utofu = all == 1;
            if (rank == 0) std::cerr << "RUNNER comm " << (utofu ? cpu_utofu_mode() : "mpi (utofu init failed)") << " rc " << rc << '\n';
            // staged all-reduces for the pipelined prefill: 2 banks per pass part (attention / FFN output), for
            // STRATA_RUNNER_PIPE_PARTS parts (default 2), of STRATA_RUNNER_AR_BANK floats (default 1M: a part of
            // 256 tokens; 0: none)
            const char *eb = std::getenv("STRATA_RUNNER_AR_BANK");
            const int bank = eb ? std::atoi(eb) : 1 << 20;
            const int parts = std::getenv("STRATA_RUNNER_PIPE_PARTS") ? std::clamp(std::atoi(std::getenv("STRATA_RUNNER_PIPE_PARTS")), 2, 4) : 2;
            if (utofu && world > 1 && bank > 0) {
                const int brc = cpu_utofu_banks(2 * parts, bank);
                banks = brc == 0 ? cpu_utofu_bank_max() : 0;
                nbanks = banks ? 2 * parts : 0;
                if (rank == 0) std::cerr << "RUNNER comm staged banks " << nbanks << " x " << banks << " floats rc " << brc << '\n';
            }
        }
#endif
#else
        (void)argc, (void)argv;
#endif
    }
    void finish() {
        ar_shutdown();
#ifdef STRATA_CPU_RUNNER_MPI
        MPI_Finalize();
#endif
    }
    // ---- asynchronous all-reduces (pipelined prefill passes): a thread on a core of its own (STRATA_RUNNER_COMM_CORE,
    // default the core after the team's) runs the uTofu all-reduces the main thread posts, in posting order; every
    // rank posts the same sequence. While it runs, every all-reduce goes through it (one thread on the queues).
    struct ArReq { float *x; int n; int bank = -1; };   // bank >= 0: a staged all-reduce in that bank
    int banks = 0;   // floats per staged bank (0: none)
    int nbanks = 0;
    std::array<ArReq, 64> ar_ring{};
    std::atomic<long> ar_posted{0}, ar_done{0};
    std::atomic<bool> ar_stop{false};
    std::thread ar_thread;
    double ar_thread_busy = 0;
    bool ar_async_ok() const { return utofu && world > 1; }
    void ar_start() {
        int core = std::getenv("STRATA_RUNNER_COMM_CORE") ? std::atoi(std::getenv("STRATA_RUNNER_COMM_CORE")) : -1;
        if (core < 0) {   // the core after the team's (OMP_PLACES=cores, close: the team from the main thread's core)
            cpu_set_t cs;
            CPU_ZERO(&cs);
            int first = 0;
            if (sched_getaffinity(0, sizeof cs, &cs) == 0)
                for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &cs)) { first = c; break; }
            core = first + omp_get_max_threads();
        }
        ar_thread = std::thread([this, core] {
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(core, &cs);
            pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
            while (!ar_stop.load(std::memory_order_relaxed)) {
                const long d = ar_done.load(std::memory_order_relaxed);
                if (d < ar_posted.load(std::memory_order_acquire)) {
                    const ArReq r = ar_ring[d % 64];
                    const double t = now();
                    if (r.bank >= 0) cpu_utofu_bank_allreduce(r.bank, r.n);
                    else allreduce_now(r.x, r.n);
                    ar_thread_busy += now() - t;
                    ar_done.store(d + 1, std::memory_order_release);
                }
            }
        });
    }
    void ar_shutdown() {   // after a prefill: decode's small all-reduces go direct again (no hand-off)
        if (!ar_thread.joinable()) return;
        ar_wait(ar_posted.load());
        ar_stop.store(true);
        ar_thread.join();
        ar_stop.store(false);
    }
    long allreduce_async(float *x, int n) {   // a ticket for ar_wait
        if (!ar_async_ok() || world == 1) { allreduce(x, n); return 0; }
        if (!ar_thread.joinable()) ar_start();
        const long id = ar_posted.load(std::memory_order_relaxed);
        while (id - ar_done.load(std::memory_order_acquire) >= long(ar_ring.size())) {}
        ar_ring[id % ar_ring.size()] = {x, n};
        ar_posted.store(id + 1, std::memory_order_release);
        return id + 1;
    }
    void ar_wait(long ticket) { while (ar_done.load(std::memory_order_acquire) < ticket) {} }
    // a staged all-reduce of n floats already written to bank's send slots (staged_layout); the results stay in
    // the bank (staged_result) until the bank's next use
    long allreduce_staged(int bank, int n) {
        if (!ar_thread.joinable()) ar_start();
        const long id = ar_posted.load(std::memory_order_relaxed);
        while (id - ar_done.load(std::memory_order_acquire) >= long(ar_ring.size())) {}
        ar_ring[id % ar_ring.size()] = {nullptr, n, bank};
        ar_posted.store(id + 1, std::memory_order_release);
        return id + 1;
    }
    bool staged_ok(int n) const { return banks > 0 && n <= banks; }
    void staged_shards(int n, std::vector<int> &off) const { off.resize(world + 1); cpu_utofu_shards(n, off.data()); }
    float *staged_send(int bank, int d) const { return cpu_utofu_bank_send(bank, d); }
    const float *staged_result(int bank, int d) const { return cpu_utofu_bank_result(bank, d); }
    // sum over ranks; every element is reduced on one rank and then broadcast, so all ranks hold identical bits
    void allreduce(float *x, int n) {
        if (world == 1) return;
        if (ar_thread.joinable()) { ar_wait(allreduce_async(x, n)); return; }   // after the pipelined ones, in order
        allreduce_now(x, n);
    }
    void allreduce_now(float *x, int n) {
#ifdef STRATA_CPU_RUNNER_UTOFU
        if (utofu) {
            const double t = now();
            // rsag mode picks its region and pieces itself (the 16K one for decode-sized vectors, the big one for longer)
            const int piece = cpu_utofu_max() > 0 ? cpu_utofu_max() : ar_max;
            for (int o = 0; o < n; o += piece) cpu_utofu_allreduce(x + o, std::min(piece, n - o));
            seconds += now() - t;
            return;
        }
#endif
#ifdef STRATA_CPU_RUNNER_MPI
        const double t = now();
        const int chunk = (n + world - 1) / world;
        scratch.assign(size_t(chunk) * world * 2, 0.f);
        std::copy(x, x + n, scratch.begin());
        float *part = scratch.data() + size_t(chunk) * world;
        MPI_Reduce_scatter_block(scratch.data(), part, chunk, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allgather(part, chunk, MPI_FLOAT, scratch.data(), chunk, MPI_FLOAT, MPI_COMM_WORLD);
        std::copy(scratch.begin(), scratch.begin() + n, x);
        seconds += now() - t;
#endif
    }
    // rank 0 receives every rank's rows [begin, end) of y (row-partitioned vector of `total` floats)
    void gather(float *y, int begin, int end, int total) {
        if (world == 1) return;
#ifdef STRATA_CPU_RUNNER_MPI
        std::vector<int> counts(world), displs(world);
        int mine[2] = {begin, end - begin};
        std::vector<int> all(2 * world);
        MPI_Allgather(mine, 2, MPI_INT, all.data(), 2, MPI_INT, MPI_COMM_WORLD);
        for (int r = 0; r < world; ++r) displs[r] = all[2 * r], counts[r] = all[2 * r + 1];
        (void)total;
        MPI_Gatherv(rank ? y + begin : MPI_IN_PLACE, end - begin, MPI_FLOAT, y, counts.data(), displs.data(), MPI_FLOAT, 0,
                    MPI_COMM_WORLD);
#endif
    }
    // the global argmax of n vectors whose slices [begin, begin + count) live on the ranks: each rank's first maximum,
    // then MPI_MAXLOC (the lower index on equal values), so the token std::max_element picks over the full logits
    double argmax_local_s = 0, argmax_comm_s = 0;   // the local scans and the exchange (timers)
    void argmax(const float *const *part, int n, int count, int begin, int *out) {
        struct VI { float v; int i; };
        std::vector<VI> loc(n), res(n);
        const double t_l = now();
        if (count >= 4096) {   // the scan over the team: per thread (max, its first index), combined in thread order
            const int team = omp_get_max_threads();
            std::vector<VI> th_best(size_t(team) * n * 8);   // [thread][k], one 64-byte slot each
            team_run([&] {
                const int t = omp_get_thread_num(), nt = omp_get_num_threads();
                const int j0 = int(int64_t(count) * t / nt), j1 = int(int64_t(count) * (t + 1) / nt);
                for (int k = 0; k < n; ++k) {
                    const float *p = part[k];
                    int b = j0;
                    for (int j = j0 + 1; j < j1; ++j)
                        if (p[j] > p[b]) b = j;
                    th_best[(size_t(t) * n + k) * 8] = j1 > j0 ? VI{p[b], b} : VI{-INFINITY, -1};
                }
            });
            for (int k = 0; k < n; ++k) {   // the first maximum: a later thread wins only when strictly larger
                VI best{-INFINITY, -1};
                for (int t = 0; t < team; ++t) {
                    const VI c = th_best[(size_t(t) * n + k) * 8];
                    if (c.i >= 0 && (best.i < 0 || c.v > best.v)) best = c;
                }
                loc[k] = {best.v, begin + std::max(0, best.i)};
            }
        } else
            for (int k = 0; k < n; ++k) {
                const float *p = part[k];
                int b = 0;
                for (int j = 1; j < count; ++j)
                    if (p[j] > p[b]) b = j;
                loc[k] = {count ? p[b] : -INFINITY, begin + b};
            }
        argmax_local_s += now() - t_l;
        struct CommTimer { double &acc; double t0; ~CommTimer() { acc += now() - t0; } } comm_timer{argmax_comm_s, now()};
#ifdef STRATA_CPU_RUNNER_UTOFU
        if (utofu && world > 1 && !std::getenv("STRATA_RUNNER_ARGMAX_MPI")) {
            // every rank's (max, index) in its own slot of zeros, summed by the uTofu all-reduce (exact: one value plus
            // zeros; indices < 2^24), then MAXLOC's choice: the largest value, the lowest index among equal ones
            std::vector<float> g(size_t(n) * world * 2, 0.f);
            for (int k = 0; k < n; ++k) g[(size_t(k) * world + rank) * 2] = loc[k].v, g[(size_t(k) * world + rank) * 2 + 1] = float(loc[k].i);
            allreduce(g.data(), int(g.size()));
            for (int k = 0; k < n; ++k) {
                res[k] = {-INFINITY, 0};
                bool first = true;
                for (int r = 0; r < world; ++r) {
                    const float v = g[(size_t(k) * world + r) * 2];
                    const int i = int(g[(size_t(k) * world + r) * 2 + 1]);
                    if (first || v > res[k].v || (v == res[k].v && i < res[k].i)) res[k] = {v, i}, first = false;
                }
            }
            for (int k = 0; k < n; ++k) out[k] = res[k].i;
            return;
        }
#endif
#ifdef STRATA_CPU_RUNNER_MPI
        if (world > 1) MPI_Allreduce(loc.data(), res.data(), n, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        else res = loc;
#else
        res = loc;
#endif
        for (int k = 0; k < n; ++k) out[k] = res[k].i;
    }
    // argmax and its softmax probability of one vector sliced over the ranks: every rank's (max, its first index, sum
    // of exp(l - max)) gathered; the global max with the lowest index among equal maxima (as argmax), p = 1 / sum_r
    // s_r exp(m_r - M)
    int argmax_prob(const float *part, int count, int begin, double &prob) {
        float mx = -INFINITY;
        int b = 0;
        for (int j = 0; j < count; ++j)
            if (part[j] > mx) mx = part[j], b = j;
        double se = 0;
        for (int j = 0; j < count; ++j) se += std::exp(double(part[j]) - mx);
        double mine[3] = {mx, double(begin + b), se};
        std::vector<double> all(3 * size_t(world));
#ifdef STRATA_CPU_RUNNER_MPI
        if (world > 1) MPI_Allgather(mine, 3, MPI_DOUBLE, all.data(), 3, MPI_DOUBLE, MPI_COMM_WORLD);
        else std::copy(mine, mine + 3, all.begin());
#else
        std::copy(mine, mine + 3, all.begin());
#endif
        double M = -INFINITY;
        int idx = 0;
        for (int r = 0; r < world; ++r)
            if (all[3 * r] > M || (all[3 * r] == M && all[3 * r + 1] < idx)) M = all[3 * r], idx = int(all[3 * r + 1]);
        double S = 0;
        for (int r = 0; r < world; ++r) S += all[3 * r + 2] * std::exp(all[3 * r] - M);
        prob = 1 / S;
        return idx;
    }
    void allreduce_max(double *x, int n) {   // element-wise maximum over ranks (exact)
#ifdef STRATA_CPU_RUNNER_MPI
        if (world > 1) {
            mpi_reserve(size_t(n) * 8, size_t(n) * 8);
            std::memcpy(mpi_sbuf, x, size_t(n) * 8);
            MPI_Allreduce(mpi_sbuf, mpi_rbuf, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            std::memcpy(x, mpi_rbuf, size_t(n) * 8);
        }
#endif
        (void)x, (void)n;
    }
    // Large MPI payloads go through these buffers, which are never freed: Fujitsu MPI over Tofu keeps the network
    // registration of a buffer's pages, so a freed (unmapped) heap buffer whose addresses come back with other pages
    // crashed the next MPI_Allgather (segfault in mca_btl_tofu_handle_tcq). A larger request maps a new pair and keeps
    // the old one mapped.
    char *mpi_sbuf = nullptr, *mpi_rbuf = nullptr;
    size_t mpi_scap = 0, mpi_rcap = 0;
    static char *mpi_map(size_t n) {
        void *p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("runner: MPI buffer mmap failed");
        return static_cast<char *>(p);
    }
    void mpi_reserve(size_t s, size_t r) {
        if (s > mpi_scap) mpi_scap = std::max(s, 2 * mpi_scap), mpi_sbuf = mpi_map(mpi_scap);
        if (r > mpi_rcap) mpi_rcap = std::max(r, 2 * mpi_rcap), mpi_rbuf = mpi_map(mpi_rcap);
    }
    // doubles: scount[r] to rank r from send + sdisp[r], rcount[r] from rank r into recv + rdisp[r]
    void alltoallv(const double *send, const int *scount, const int *sdisp, double *recv, const int *rcount, const int *rdisp) {
        if (world > 1 && coll_ar()) {   // every rank's whole send buffer gathered; each rank keeps the pieces meant for it
            int smax = 0;
            for (int r = 0; r < world; ++r) smax = std::max(smax, sdisp[r] + scount[r]);
            std::vector<double> all(size_t(smax) * world);
            std::vector<int> disp_all(size_t(world) * world);   // every rank's sdisp, to find its piece for me
            allgather(sdisp, world * int(sizeof(int)), disp_all.data());
            allgather(send, smax * int(sizeof(double)), all.data());
            for (int r = 0; r < world; ++r)
                std::memcpy(recv + rdisp[r], all.data() + size_t(r) * smax + disp_all[size_t(r) * world + rank], size_t(rcount[r]) * sizeof(double));
            return;
        }
#ifdef STRATA_CPU_RUNNER_MPI
        if (world > 1) {
            const size_t sn = size_t(sdisp[world - 1] + scount[world - 1]) * sizeof(double), rn = size_t(rdisp[world - 1] + rcount[world - 1]) * sizeof(double);
            mpi_reserve(sn, rn);
            std::memcpy(mpi_sbuf, send, sn);
            MPI_Alltoallv(mpi_sbuf, scount, sdisp, MPI_DOUBLE, mpi_rbuf, rcount, rdisp, MPI_DOUBLE, MPI_COMM_WORLD);
            std::memcpy(recv, mpi_rbuf, rn);
            return;
        }
#endif
        std::memcpy(recv + rdisp[0], send + sdisp[0], size_t(scount[0]) * sizeof(double));
    }
    // allgather (and alltoallv through it) as one uTofu all-reduce of zero-padded slots, the bytes carried as 16-bit
    // halves in floats (exact through sums with zeros and flush-to-zero). Default with uTofu: MPI_Allgather took
    // 1-3 ms for 4-512 KB and crashed (segfault in the Tofu BTL) when called between the runner's uTofu all-reduces;
    // STRATA_RUNNER_COLL=mpi: MPI's
    bool coll_ar() const {
        static const bool on = !(std::getenv("STRATA_RUNNER_COLL") && std::string(std::getenv("STRATA_RUNNER_COLL")) == "mpi");
        return on && utofu;
    }
    std::vector<float> ar_slots;
    void allgather_ar(const void *send, int bytes, void *recv) {
        const int words = (bytes + 1) / 2;   // 16-bit halves
        ar_slots.assign(size_t(words) * world, 0.f);
        const uint8_t *src = static_cast<const uint8_t *>(send);
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (int i = 0; i < words; ++i) {
                const int lo = src[2 * i], hi = 2 * i + 1 < bytes ? src[2 * i + 1] : 0;
                ar_slots[size_t(rank) * words + i] = float(lo | (hi << 8));
            }
        });
        allreduce(ar_slots.data(), int(ar_slots.size()));
        uint8_t *dst = static_cast<uint8_t *>(recv);
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (long ri = 0; ri < long(world) * words; ++ri) {
                const int r = int(ri / words), i = int(ri % words);
                const int v = int(ar_slots[ri]);
                dst[size_t(r) * bytes + 2 * i] = uint8_t(v & 255);
                if (2 * i + 1 < bytes) dst[size_t(r) * bytes + 2 * i + 1] = uint8_t(v >> 8);
            }
        });
    }
    void allgather(const void *send, int bytes, void *recv) {   // every rank's `bytes` bytes, in rank order
        if (world > 1 && coll_ar()) { allgather_ar(send, bytes, recv); return; }
#ifdef STRATA_CPU_RUNNER_MPI
        if (world > 1) {
            mpi_reserve(size_t(bytes), size_t(bytes) * world);
            std::memcpy(mpi_sbuf, send, size_t(bytes));
            MPI_Allgather(mpi_sbuf, bytes, MPI_BYTE, mpi_rbuf, bytes, MPI_BYTE, MPI_COMM_WORLD);
            std::memcpy(recv, mpi_rbuf, size_t(bytes) * world);
            return;
        }
#endif
        std::memcpy(recv, send, size_t(bytes));
    }
    void bcast(int *v, int n) {   // rank 0's values to every rank (tokens chosen from rank 0's logits)
#ifdef STRATA_CPU_RUNNER_MPI
        if (world > 1) MPI_Bcast(v, n, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        (void)v, (void)n;
    }
    void barrier() {
#ifdef STRATA_CPU_RUNNER_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    // per-key min and max over ranks of a timer map (same keys on every rank)
    void spread(const std::map<std::string, double> &t, std::vector<double> &lo, std::vector<double> &hi) {
        std::vector<double> v;
        for (const auto &kv : t) v.push_back(kv.second);
        lo = hi = v;
#ifdef STRATA_CPU_RUNNER_MPI
        if (world > 1) {
            MPI_Allreduce(v.data(), lo.data(), int(v.size()), MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(v.data(), hi.data(), int(v.size()), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }
#endif
    }
    // true when every rank holds the same 64-bit value (replicated state must stay bitwise identical)
    bool same_on_all(uint64_t v) {
#ifdef STRATA_CPU_RUNNER_MPI
        if (world > 1) {
            unsigned long long lo = v, hi = v, x = v;
            MPI_Allreduce(&x, &lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&x, &hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            return lo == hi;
        }
#endif
        (void)v;
        return true;
    }
    double max_over_ranks(double v) {
#ifdef STRATA_CPU_RUNNER_MPI
        double out = v;
        MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return out;
#else
        return v;
#endif
    }
};

// threads per CMG in the team's placement (STRATA_RUNNER_CMG_THREADS, default 12: 47 threads on 48 cores, close):
// the flag barrier's groups and the mHC partial sums go by CMG (at most 8 groups)
int cmg_threads() {
    static const int v = std::getenv("STRATA_RUNNER_CMG_THREADS") ? std::max(6, std::atoi(std::getenv("STRATA_RUNNER_CMG_THREADS"))) : 12;
    return v;
}
// ------------------------------------------------------------------------------------------------- packed matrices
enum class Kind { NvQ, NvP, Mx, Bfp, Q8, B16 };

// Packed weights in memory first touched by the threads that read them: tile t lives on the CMG of the thread that
// gemv() gives it (same static split), which A64FX's per-CMG HBM needs (one CMG holding every page measured 65-88
// GB/s on a 47-thread GEMV instead of 220-270). The pages must be fresh: heap memory recycled from the packing buffers
// was already touched by the loading thread, which put every page of every matrix on its CMG, so matrices come from
// an arena of anonymous mappings that is only ever bump-allocated (weights live as long as the process).
uint8_t *arena_alloc(size_t n) {
    static uint8_t *at = nullptr;
    static size_t left = 0;
    n = (n + 255) & ~size_t(255);
    if (n > left) {
        const size_t chunk = std::max(n, size_t(256) << 20);
        void *p = mmap(nullptr, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("runner: weight arena mmap failed");
        at = static_cast<uint8_t *>(p), left = chunk;
    }
    uint8_t *r = at;
    at += n, left -= n;
    return r;
}
struct Placed {
    uint8_t *p = nullptr;   // arena memory, never freed
    size_t n = 0;
    size_t size() const { return n; }
    const uint8_t *data() const { return p; }
};
struct Matrix {
    Kind kind = Kind::B16;
    int rows = 0, padded = 0, cols = 0;   // logical rows, rows rounded up to 16, columns
    float global = 1;
    int rot = 0;   // thread rotation of the tile split: matrices with tiles % threads != 0 put their extra tile on
                   // different threads, so a fused region of several such matrices stays balanced
    Placed data;
    // chunk-major copies of an mHC projection (STRATA_RUNNER_MHC_REPLICATE: 1 one per CMG (default), 2 one shared,
    // 0 none: the tile-major original):
    // block (chunk, stream, tile) at ((chunk * 4 + stream) * tiles + tile) blocks, so a thread's blocks are one run
    std::vector<Placed> cmg_copy;
    const uint8_t *cmg_data(int th) const {   // the copy thread th reads (its CMG's)
        return cmg_copy.empty() ? data.data() : cmg_copy[std::min(int(cmg_copy.size()) - 1, th / cmg_threads())].data();
    }
    bool whole_k = false;   // never split along K (a router slice of 1-2 tiles: its fused MoE region needs whole tiles)
    // STRATA_RUNNER_PREFILL_TRUNK=q8: the same rows requantized to int8 per (row, 32), used by passes of more than 4
    // tokens instead of this (exact) matrix; decode keeps this one
    std::shared_ptr<Matrix> q8;
    size_t bytes() const { return data.size(); }
};

// columns per K block and bytes per block of one 16-row tile, per layout (a K range of a tile is contiguous)
int block_cols(Kind k) { return k == Kind::NvQ || k == Kind::NvP || k == Kind::Q8 ? 64 : k == Kind::Bfp ? 128 : 32; }
size_t block_bytes(Kind k) {
    switch (k) {
        case Kind::NvQ: case Kind::NvP: return GF4_SB_BYTES;
        case Kind::Mx: return 32 * 16 + 16;
        case Kind::Bfp: return 128 * 16 + 16;
        case Kind::Q8: return 2 * GF4_Q8_BLOCK_BYTES;
        case Kind::B16: return 32 * 32;
    }
    return 0;
}

// a matrix with very few 16-row tiles (the mHC projection: 2) would leave most threads idle: gemv_multi splits it along
// K as well, into `chunks` column ranges of `width` columns, and adds the chunks' partial sums in chunk order
// afterwards; work item i = tile * chunks + chunk, items split evenly. Only below team / 8 tiles: matrices with more
// share their region with others (router + shared expert, 16 expert slices of 12 tiles), and splitting those measured
// 1.9-2.3x slower than whole tiles (shorter runs per call, the partial-sum pass)
struct KSplit {
    int chunks = 0, width = 0;   // chunks 0: rows only
};
// The mHC projections (24 x 4·hidden BF16: two tiles) are read in 256-column blocks, block b with both tiles by the
// thread whose share of the blocks holds b (site() below); place() puts each block on that thread's CMG (512-column
// blocks, half the partials for the one-thread sum, measured no better: 28.9-29.9 vs 28.1-28.7 ms per token).
constexpr int MHC_BLOCK = 256;
bool mhc_layout(const Matrix &m) { return m.kind == Kind::B16 && m.padded == 32 && m.cols % MHC_BLOCK == 0 && m.cols >= 8192; }
// the site's work unit: 32 hidden positions in all four streams (one BF16 K block per stream), so a thread updates,
// projects and combines only residual values it wrote itself at the previous site (with 256-value blocks of the
// concatenated streams, three of the four inputs of every update came from other CMGs)
constexpr int MHC_CHUNK = 32;
int mhc_block_begin(int blocks, int t, int n) { return int(int64_t(blocks) * t / n); }

bool ksplit_enabled() {   // STRATA_RUNNER_KSPLIT=0 turns the K split off (layout experiments; results then differ)
    static const bool on = !(std::getenv("STRATA_RUNNER_KSPLIT") && std::string(std::getenv("STRATA_RUNNER_KSPLIT")) == "0");
    return on;
}
KSplit ksplit(const Matrix &m, int team) {
    const int tiles = m.padded / 16, bc = block_cols(m.kind), blocks = m.cols / bc;
    KSplit s;
    if (tiles * 8 >= team || blocks < 2 || !ksplit_enabled() || mhc_layout(m) || m.whole_k) return s;
    const int want = std::min(blocks, (team + tiles - 1) / tiles);
    s.width = (blocks + want - 1) / want * bc;
    s.chunks = (m.cols + s.width - 1) / s.width;
    if (s.chunks < 2) s.chunks = 0;
    return s;
}

// copy a tile-major packed buffer into first-touched memory with gemv()'s thread split
int next_rotation() {   // load order is the same when an image is replayed, so the rotations are too
    static int counter = 0;
    return counter++ % std::max(1, omp_get_max_threads());
}

Placed place(const std::vector<uint8_t> &packed, const Matrix &m) {
    Placed out;
    out.n = packed.size();
    out.p = arena_alloc(out.n);   // fresh pages: none is touched before the copy below
    const int tiles = m.padded / 16;
    const size_t per = out.n / size_t(tiles);
    const KSplit ks = ksplit(m, omp_get_max_threads());
    const size_t step = size_t(ks.width / block_cols(m.kind)) * block_bytes(m.kind);
    auto item_start = [&](int i) {   // first byte of K-split work item i (items in byte order)
        return i >= tiles * ks.chunks ? out.n : size_t(i / ks.chunks) * per + size_t(i % ks.chunks) * step;
    };
#pragma omp parallel
    {
        const int n = omp_get_num_threads();
        if (mhc_layout(m)) {   // site()'s chunk split: a chunk's four stream blocks, both tiles, on its thread
            const int chunks = m.cols / 4 / MHC_CHUNK, t = omp_get_thread_num();
            const size_t bb = block_bytes(m.kind);   // one 32-column block of one tile
            for (int c = mhc_block_begin(chunks, t, n); c < mhc_block_begin(chunks, t + 1, n); ++c)
                for (int j = 0; j < 4; ++j)
                    for (int tile = 0; tile < tiles; ++tile) {
                        const size_t o = size_t(tile) * per + size_t(j * chunks + c) * bb;
                        std::memcpy(out.p + o, packed.data() + o, bb);
                    }
        } else if (ks.chunks) {   // gemv_multi's item split
            const int items = tiles * ks.chunks, t = omp_get_thread_num();
            const size_t b0 = item_start(int(int64_t(items) * t / n)), b1 = item_start(int(int64_t(items) * (t + 1) / n));
            if (b1 > b0) std::memcpy(out.p + b0, packed.data() + b0, b1 - b0);
        } else {
            const int t = (omp_get_thread_num() + m.rot) % n;
            const int t0 = int(int64_t(tiles) * t / n), t1 = int(int64_t(tiles) * (t + 1) / n);
            if (t1 > t0) std::memcpy(out.p + per * t0, packed.data() + per * t0, per * size_t(t1 - t0));
        }
    }
    return out;
}

// which routed-expert projections the W4A8 numerics apply to: STRATA_RUNNER_W4A8_ONLY=gate_up|down (default both;
// an accuracy experiment: the others stay W4A16). Part of the image name when set.
std::string w4a8_only() { const char *e = std::getenv("STRATA_RUNNER_W4A8_ONLY"); return e ? e : ""; }
bool w4a8_gate_up() { return w4a8_only() != "down"; }
bool w4a8_down() { return w4a8_only() != "gate_up"; }

// int8 activation block (values per scale) for the W4A8 expert kernels: 64, or 16 with --numerics w4a8g16 (set
// by main before loading); the Q8 trunk kernels always take 64
int g_act_group = 64;

struct Activation {   // one token: F32 values and, when an int8 kernel needs them, the Q8 form
    Vec x;
    std::vector<int8_t> q;
    Vec qs;
    int group = 64;   // values per entry of qs
    void set(const float *v, int n, bool quant, int grp = 64) {
        x.assign(v, v + n);
        if (quant) {
            group = grp;
            q.resize(n);
            qs.resize(n / group);
            // gf4_quantize_q8g per 64-value block, blocks over the team (a serial 4096-value pass is ~40 us on A64FX)
            const int blocks = n / 64;
            if (blocks >= 16) {
                team_run([&] {
#pragma omp for schedule(static) nowait
                    for (int b = 0; b < blocks; ++b) gf4_quantize_q8g(x.data() + 64 * b, 64, group, q.data() + 64 * b, qs.data() + 64 * b / group);
                });
            } else gf4_quantize_q8g(x.data(), n, group, q.data(), qs.data());
        }
    }
};

bool needs_q8(const Matrix &m) { return m.kind == Kind::NvQ || m.kind == Kind::Q8; }

// y[r] for the matrix's logical rows (one job of gemv_multi below)
struct Job;
void gemv_multi(const std::vector<Job> &jobs);
void gemv(const Matrix &m, const Activation &a, float *y);

// tokens per multi-token pass: verification passes take at most 4, prefill chunks up to this
constexpr int MAX_PASS = 2048;
// token n's activation of a job: four in place, more (a prefill chunk) in a vector, so a job of a verification pass
// copies without an allocation
struct ActList {
    std::array<const Activation *, 4> in{};
    std::vector<const Activation *> more;
    const Activation *&operator[](int i) {
        if (i < 4) return in[i];
        if (int(more.size()) < i - 3) more.resize(i - 3);
        return more[i - 4];
    }
    const Activation *operator[](int i) const { return i < 4 ? in[i] : more[i - 4]; }
};

// independent GEMVs in one parallel region: every thread runs its gemv() share of each matrix in turn (so the
// first-touch placement still holds) and the team forks once instead of once per matrix
struct Job {
    const Matrix *m;
    const Activation *a;   // nt activations, contiguous (one token: a single one), unless `as` is given
    float *y;              // token n's output at y + n * ldy
    int nt = 1, ldy = 0;
    ActList as{};          // token n's activation (when as[0] is set)
    const int8_t *xt = nullptr;   // W4A8 prefill: the activations transposed once (gf4_w4a8_gemm_prep), shared by the team
    const float *xs = nullptr;
};

// STRATA_RUNNER_W4A8_GEMM=0: W4A8 tiles of a prefill pass through the 4-token kernel (default: gf4_w4a8_gemm, 16
// tokens per vector, bitwise the same, from STRATA_RUNNER_W4A8_GEMM_MIN tokens, default 8)
bool w4a8_gemm_on() {
    static const bool on = !(std::getenv("STRATA_RUNNER_W4A8_GEMM") && std::string(std::getenv("STRATA_RUNNER_W4A8_GEMM")) == "0");
    return on;
}
int w4a8_gemm_min() {
    static const int n = std::getenv("STRATA_RUNNER_W4A8_GEMM_MIN") ? std::atoi(std::getenv("STRATA_RUNNER_W4A8_GEMM_MIN")) : 8;
    return n;
}
// STRATA_RUNNER_W4A8_XT=0: each gf4_w4a8_gemm call transposes its tokens' activations itself (default: an expert
// use's transposed once in the plan, for all its tiles over the team; 12 gate/up tiles per slice give a thread about
// one tile per call)
bool w4a8_xt_on() {
    static const bool on = !(std::getenv("STRATA_RUNNER_W4A8_XT") && std::string(std::getenv("STRATA_RUNNER_W4A8_XT")) == "0");
    return w4a8_gemm_on() && on;
}
// uninitialized 256-byte aligned memory that grows and stays (transposed activations: one buffer for all the
// uses of a pass; per-use vectors regrew, and faulted, whenever a use had more tokens than before)
template <class T> struct AlignedBuf {
    std::unique_ptr<T[]> p; size_t n = 0;
    T *get(size_t want) {
        if (want > n) n = want + want / 4, p.reset(new T[n + 256 / sizeof(T)]);
        return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(p.get()) + 255) & ~uintptr_t(255));
    }
};
// gf4_w4a8_gemm_prep of acts a[0..nt) (a job's list; cols values, one scale group) into xt / xs
void w4a8_xt_prep(const ActList &a, int nt, int cols, int8_t *xt, float *xs) {
    thread_local std::vector<gf4_q8act> q;
    q.resize(nt);
    for (int n = 0; n < nt; ++n)
        q[n] = gf4_q8act{cols, const_cast<int8_t *>(a[n]->q.data()), const_cast<float *>(a[n]->qs.data()), a[n]->group};
    gf4_w4a8_gemm_prep(q.data(), nt, cols, xt, xs);
}

// one tile (or, with k0/k1, columns [k0, k1) of one tile as a sub-matrix) into out[0..16)
// nt tokens (*a[0..nt)) at once: the kernels read each weight once for all of them, with the same arithmetic per
// token as one at a time; token n's rows go to out + n * ldy
void tile_kernel(const Matrix &m, const Activation *const *a, int nt, int t0, int t1, int k0, int k1, float *out, int ldy) {
    if (nt > 4 && m.q8 && k0 == 0 && k1 == m.cols && !a[0]->q.empty() && a[0]->group == 64) {   // prefill: the int8 copy
        tile_kernel(*m.q8, a, nt, t0, t1, k0, k1, out, ldy);
        return;
    }
    if (nt > 4) {   // a prefill chunk: each tile for 4 tokens at a time while it is in this core's cache (the kernels
                    // take at most 4; the arithmetic per token is the same)
        // STRATA_RUNNER_GEMM_DECODE=0: the exact float layouts also through the 4-token kernels (default: their
        // weights decoded once per tile for all tokens, gf4_tile_many, bitwise the same)
        static const bool decode_once = !(std::getenv("STRATA_RUNNER_GEMM_DECODE") && std::string(std::getenv("STRATA_RUNNER_GEMM_DECODE")) == "0");
        if (decode_once && (m.kind == Kind::Mx || m.kind == Kind::Bfp || m.kind == Kind::B16)) {
            const int cols = k1 - k0, kind = m.kind == Kind::Mx ? 0 : m.kind == Kind::Bfp ? 1 : 2;
            const size_t tile_bytes = m.data.size() / (m.padded / 16);
            const uint8_t *base = m.data.data() + size_t(k0 / block_cols(m.kind)) * block_bytes(m.kind);
            const float *x[MAX_PASS];
            for (int n = 0; n < nt; ++n) x[n] = a[n]->x.data() + k0;
            const bool part = k0 != 0 || k1 != m.cols;   // a K-split item: one tile, rows at out
            // tiles in pairs through gf4_tile_many2 (2 tiles x 3 tokens per register block, bitwise the same;
            // STRATA_RUNNER_MANY2=0: one at a time)
            static const bool many2 = !(std::getenv("STRATA_RUNNER_MANY2") && std::string(std::getenv("STRATA_RUNNER_MANY2")) == "0");
            int tile = t0;
            if (many2 && !part)
                for (; tile + 1 < t1; tile += 2)
                    gf4_tile_many2(kind, base + size_t(tile) * tile_bytes, base + size_t(tile + 1) * tile_bytes, cols, x, nt, out + 16 * tile, out + 16 * (tile + 1), ldy);
            for (; tile < t1; ++tile)
                gf4_tile_many(kind, base + size_t(tile) * tile_bytes, cols, x, nt, out + (part ? 0 : 16 * tile), ldy);
            return;
        }
        if (w4a8_gemm_on() && m.kind == Kind::NvQ && nt >= w4a8_gemm_min() && k0 == 0 && k1 == m.cols) {   // gf4_w4a8_gemm
            thread_local std::vector<gf4_q8act> q;
            q.resize(nt);
            for (int n = 0; n < nt; ++n)
                q[n] = gf4_q8act{m.cols, const_cast<int8_t *>(a[n]->q.data()), const_cast<float *>(a[n]->qs.data()), a[n]->group};
            gf4_w4a8_gemm(m.data.data(), m.cols, m.global, t0, t1, q.data(), nt, out, ldy);
            return;
        }
        for (int tile = t0; tile < t1; ++tile)
            for (int b = 0; b < nt; b += 4) tile_kernel(m, a + b, std::min(4, nt - b), tile, tile + 1, k0, k1, out + size_t(b) * ldy, ldy);
        return;
    }
    const int cols = k1 - k0;
    const uint8_t *base = m.data.data();
    if (k0 != 0 || k1 != m.cols) base += size_t(t0) * (m.data.size() / (m.padded / 16)) + size_t(k0 / block_cols(m.kind)) * block_bytes(m.kind),
                                 t1 -= t0, t0 = 0;
    const float *x[4];
    gf4_q8act q[4];
    for (int n = 0; n < nt; ++n) {
        const Activation &an = *a[n];
        x[n] = an.x.data() + k0;
        q[n] = gf4_q8act{cols, nullptr, nullptr, 0};
        if (needs_q8(m)) {
            if (m.kind == Kind::Q8 && an.group != 64) throw std::runtime_error("runner: Q8 trunk kernels take 64-value activation blocks");
            q[n] = gf4_q8act{cols, const_cast<int8_t *>(an.q.data()) + k0, const_cast<float *>(an.qs.data()) + k0 / an.group, an.group};
        }
    }
    switch (m.kind) {
        case Kind::NvQ: gf4_w4a8(base, cols, m.global, t0, t1, q, nt, out, ldy); break;
        case Kind::NvP: gf4_w4a16p(base, cols, m.global, t0, t1, x, nt, out, ldy); break;
        case Kind::Mx: gf4_mxfp8(base, cols, t0, t1, x, nt, out, ldy); break;
        case Kind::Bfp: gf4_bfp8(base, cols, t0, t1, x, nt, out, ldy); break;
        case Kind::Q8: gf4_q8(base, cols, t0, t1, q, nt, out, ldy); break;
        case Kind::B16: gf4_bf16k(base, cols, t0, t1, x, nt, out, ldy); break;
    }
}

void job_acts(const Job &j, const Activation **p) {   // p[0..nt): the job's activations
    for (int n = 0; n < j.nt; ++n) p[n] = j.as[0] ? j.as[n] : j.a + n;
}

// A long pass's job with fewer tiles than threads (router 18, shared expert ~11, per-head absorbs): (tile, token
// block) items over the team instead of whole tiles, blocks of a multiple of 4 tokens, as few as fill the team
// (STRATA_RUNNER_TOKEN_SPLIT=0: whole tiles). Calls tile_kernel per item: the same arithmetic per token.
struct TokenSplit { int blocks = 0, width = 0; };
TokenSplit token_split(const Matrix &m, int nt, int team) {
    static const bool on = !(std::getenv("STRATA_RUNNER_TOKEN_SPLIT") && std::string(std::getenv("STRATA_RUNNER_TOKEN_SPLIT")) == "0");
    const int tiles = m.padded / 16;
    TokenSplit ts;
    if (!on || nt <= 4 || tiles >= team) return ts;
    const int want = std::min((team + tiles - 1) / tiles, (nt + 3) / 4);
    ts.width = ((nt + want - 1) / want + 3) / 4 * 4;
    ts.blocks = (nt + ts.width - 1) / ts.width;
    if (ts.blocks < 2) ts.blocks = 0;
    return ts;
}
// thread t of n's items of a token-split job: tile_kernel per (tile, block), outputs at out (token stride ldy)
void token_split_run(const Matrix &m, const TokenSplit &ts, const Activation *const *ap, int nt, int t, int n, float *out, int ldy) {
    const int tiles = m.padded / 16, items = tiles * ts.blocks, v = (t + m.rot) % n;
    for (int i = int(int64_t(items) * v / n); i < int(int64_t(items) * (v + 1) / n); ++i) {
        const int tile = i / ts.blocks, b0 = (i % ts.blocks) * ts.width, bn = std::min(ts.width, nt - b0);
        tile_kernel(m, ap + b0, bn, tile, tile + 1, 0, m.cols, out + size_t(b0) * ldy, ldy);
    }
}

// The same tree without atomics: each thread raises its own arrival flag (its own 256-byte line), each CMG's first
// thread polls its members' flags, thread 0 polls the CMG leaders', and the release goes down the same two levels.
struct FlagBarrier {
    struct alignas(256) Flag { std::atomic<int> v{0}; };
    Flag arrive[64], garrive[16], grelease[16], release;
    int n = 0, gsize = 12, groups = 1;
    alignas(256) int local[64][64] = {};
    void init(int team, int per_group = 12) { n = team, gsize = per_group, groups = (team + per_group - 1) / per_group; }
    void wait(int t) {
        int &ls = local[t][0];
        ls ^= 1;
        const int g = t / gsize, g0 = g * gsize, g1 = std::min(n, g0 + gsize);
        if (t != g0) {
            arrive[t].v.store(ls, std::memory_order_release);
            while (grelease[g].v.load(std::memory_order_acquire) != ls) {}
            return;
        }
        for (int u = g0 + 1; u < g1; ++u)
            while (arrive[u].v.load(std::memory_order_acquire) != ls) {}
        if (g) {
            garrive[g].v.store(ls, std::memory_order_release);
            while (release.v.load(std::memory_order_acquire) != ls) {}
        } else {
            for (int h = 1; h < groups; ++h)
                while (garrive[h].v.load(std::memory_order_acquire) != ls) {}
            release.v.store(ls, std::memory_order_release);
        }
        grelease[g].v.store(ls, std::memory_order_release);
    }
};

FlagBarrier g_bar;   // the team's barrier in the hot regions (main: g_bar.init(team); STRATA_RUNNER_FLAG_BARRIER=0 keeps OpenMP's)
// a barrier of the current team: the flag barrier when the team is the one it was set up for, OpenMP's otherwise
inline void team_barrier() {
    if (omp_get_num_threads() == g_bar.n) g_bar.wait(omp_get_thread_num());
    else {
#pragma omp barrier
    }
}

// gemv_multi's work for one team: plan outside a region (output and K-split partial buffers), then every thread of
// the team calls run(t, n) once, inside a region of its own (gemv_multi) or one the caller opened (the fused layers).
// run() contains one barrier when a job has a K split and none otherwise; outputs are complete for the caller only
// after its next barrier (or the region's end).
struct ShareItem { int j, tile, ntile, b0, bn; };
// f(item) for thread t of n's items; false: not a long pass (or switched off), the caller splits as before
template <class F>
bool balanced_items(const std::vector<Job> &jobs, int t, int n, F &&f) {
    // the flattened (job, tile pair, token) space in equal contiguous cost ranges, one per thread: token j of a pair
    // goes to the thread whose range holds its midpoint, so a thread gets a token range of 1-3 pairs. Whole items
    // (pair x token block) left the busiest thread ceil(items / n): 7 of 6.13 on 47 threads (87.5 %; 48 threads,
    // 6 of 6, ran 13 % faster in total)
    static const bool on = !(std::getenv("STRATA_RUNNER_GEMV_BALANCE") && std::string(std::getenv("STRATA_RUNNER_GEMV_BALANCE")) == "0");
    if (!on || n < 2 || jobs.empty()) return false;
    double total = 0;
    for (const auto &j : jobs) {
        if (j.nt <= 4) return false;
        total += double(j.m->padded / 16) * j.nt * j.m->cols;
    }
    if (total <= 0) return true;
    const double lo = total * t / n, hi = total * (t + 1) / n;
    double start = 0;
    for (int ji = 0; ji < int(jobs.size()); ++ji) {
        const auto &j = jobs[ji];
        const int tiles = j.m->padded / 16, nt = j.nt;
        for (int tl = 0; tl < tiles; tl += 2) {
            const int ntl = std::min(2, tiles - tl);
            const double u = double(ntl) * j.m->cols, end = start + u * nt;
            if (end > lo && start < hi) {
                const int j0 = std::clamp(int(std::ceil((lo - start) / u - 0.5)), 0, nt);
                const int j1 = std::clamp(int(std::ceil((hi - start) / u - 0.5)), 0, nt);
                if (j1 > j0) f(ShareItem{ji, tl, ntl, j0, j1 - j0});
            }
            start = end;
            if (start >= hi) return true;
        }
    }
    return true;
}
struct GemvPlan {
    struct Split { int chunks = 0, width = 0; std::vector<float> part; };   // see ksplit()
    std::vector<Job> jobs;
    std::vector<std::vector<float>> tmp;   // padded outputs of matrices whose rows are not a multiple of 16
    std::vector<float *> out;
    std::vector<Split> split;
    bool any_split = false;
    std::vector<int> ostride;   // token stride of out[j]
    explicit GemvPlan(std::vector<Job> j) : jobs(std::move(j)), tmp(jobs.size()), out(jobs.size()), split(jobs.size()), ostride(jobs.size()) {
        const int team = omp_get_max_threads();
        for (size_t i = 0; i < jobs.size(); ++i) {
            const Matrix &m = *jobs[i].m;
            const int nt = jobs[i].nt;
            if (nt < 1 || nt > MAX_PASS || (nt > 1 && jobs[i].ldy < m.rows)) throw std::runtime_error("runner: GEMV job token count or stride");
            out[i] = jobs[i].y, ostride[i] = nt > 1 ? jobs[i].ldy : m.padded;
            if (m.padded != m.rows) { tmp[i].assign(size_t(nt) * m.padded, 0.f); out[i] = tmp[i].data(), ostride[i] = m.padded; }
            const KSplit ks = ksplit(m, team);
            if (ks.chunks) {   // part[chunk][token][row]
                split[i].chunks = ks.chunks, split[i].width = ks.width;
                split[i].part.assign(size_t(ks.chunks) * nt * m.padded, 0.f);
                any_split = true;
            }
        }
        for (size_t i = 0; i < jobs.size(); ++i)
            if (!split[i].chunks) whole.push_back(jobs[i]), whole_of.push_back(int(i));
    }
    std::vector<Job> whole;      // the jobs without a K split (balanced over the team in long passes)
    std::vector<int> whole_of;   // their indices in jobs
    void run(int t, int n) {
        const Activation *ap[MAX_PASS];
        // a long pass: the jobs without a K split as balanced token ranges over the team (see balanced_items), the
        // K-split ones (few-row matrices) through their (tile, chunk) items below
        int cur = -1;
        const bool bal = !whole.empty() && balanced_items(whole, t, n, [&](const ShareItem &it) {
            const int jj = whole_of[it.j];
            const Job &jb = jobs[jj];
            const Matrix &m = *jb.m;
            if (jj != cur) job_acts(jb, ap), cur = jj;
            tile_kernel(m, ap + it.b0, it.bn, it.tile, it.tile + it.ntile, 0, m.cols, out[jj] + size_t(it.b0) * ostride[jj], ostride[jj]);
            if (out[jj] != jb.y)   // this item's rows of a padded output
                for (int k = it.b0; k < it.b0 + it.bn; ++k)
                    for (int r = 16 * it.tile; r < std::min(16 * (it.tile + it.ntile), m.rows); ++r)
                        jb.y[size_t(k) * jb.ldy + r] = out[jj][size_t(k) * ostride[jj] + r];
        });
        if (bal && !any_split) return;
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (bal && !split[j].chunks) continue;   // done above
            const Matrix &m = *jobs[j].m;
            const Job &jb = jobs[j];
            const int tiles = m.padded / 16, nt = jb.nt;
            job_acts(jb, ap);
            if (split[j].chunks) {
                auto &sp = split[j];
                const int items = tiles * sp.chunks;
                const int i0 = int(int64_t(items) * t / n), i1 = int(int64_t(items) * (t + 1) / n);
                for (int i = i0; i < i1; ++i) {
                    const int tile = i / sp.chunks, c = i % sp.chunks;
                    tile_kernel(m, ap, nt, tile, tile + 1, c * sp.width, std::min(m.cols, (c + 1) * sp.width),
                                sp.part.data() + size_t(c) * nt * m.padded + 16 * tile, m.padded);
                }
                continue;
            }
            if (const TokenSplit ts = token_split(m, nt, n); ts.blocks) {
                token_split_run(m, ts, ap, nt, t, n, out[j], ostride[j]);
                if (out[j] != jb.y) {   // this thread's items' rows of a padded output
                    const int items = tiles * ts.blocks, v = (t + m.rot) % n;
                    for (int i = int(int64_t(items) * v / n); i < int(int64_t(items) * (v + 1) / n); ++i) {
                        const int tile = i / ts.blocks, b0 = (i % ts.blocks) * ts.width, bn = std::min(ts.width, nt - b0);
                        for (int k = b0; k < b0 + bn; ++k)
                            for (int r = 16 * tile; r < std::min(16 * tile + 16, m.rows); ++r) jb.y[size_t(k) * jb.ldy + r] = out[j][size_t(k) * ostride[j] + r];
                    }
                }
                continue;
            }
            const int v = (t + m.rot) % n;   // the placement's thread split
            const int t0 = int(int64_t(tiles) * v / n), t1 = int(int64_t(tiles) * (v + 1) / n);
            if (t1 > t0) {
                tile_kernel(m, ap, nt, t0, t1, 0, m.cols, out[j], ostride[j]);
                if (out[j] != jb.y)   // this thread's rows of a padded output
                    for (int k = 0; k < nt; ++k)
                        for (int r = 16 * t0; r < std::min(16 * t1, m.rows); ++r) jb.y[size_t(k) * jb.ldy + r] = out[j][size_t(k) * ostride[j] + r];
            }
        }
        if (any_split) {
            team_barrier();
            for (size_t j = 0; j < jobs.size(); ++j) {
                if (!split[j].chunks) continue;
                const auto &sp = split[j];
                const Job &jb = jobs[j];
                const int rows = jb.m->padded, logical = jb.m->rows, nt = jb.nt;
                const int r0 = int(int64_t(rows) * t / n), r1 = int(int64_t(rows) * (t + 1) / n);
                for (int k = 0; k < nt; ++k)
                    for (int r = r0; r < std::min(r1, logical); ++r) {
                        float v = 0;
                        for (int c = 0; c < sp.chunks; ++c) v += sp.part[(size_t(c) * nt + k) * rows + r];
                        jb.y[size_t(k) * jb.ldy + r] = v;
                    }
            }
        }
    }
};

void gemv_multi(const std::vector<Job> &jobs) {
    GemvPlan plan(jobs);
    team_run([&] { plan.run(omp_get_thread_num(), omp_get_num_threads()); });
}

void gemv(const Matrix &m, const Activation &a, float *y) { gemv_multi({{&m, &a, y}}); }

// Read (one load per 256-byte line) the bytes thread t of n will stream in the next GEMV region of these matrices: its
// tile share, or its K-split item share, as gemv_multi / gemv_share / read_norm split them. Run by the idle threads
// while thread 0 is in an all-reduce, it turns the next region's first, cold pass into L2 hits (one call per region
// reaches 60-70 % of the long-run kernel rates cold). The data is only read: no effect on any result.
void touch_share(const std::vector<const Matrix *> &ms, int t, int n) {
    uint64_t sink = 0;
    for (const Matrix *mp : ms) {
        if (!mp || !mp->rows) continue;
        const Matrix &m = *mp;
        const int tiles = m.padded / 16;
        const size_t per = m.bytes() / size_t(tiles);
        const KSplit ks = ksplit(m, n);
        size_t b0, b1;
        if (ks.chunks) {
            const size_t step = size_t(ks.width / block_cols(m.kind)) * block_bytes(m.kind);
            const int items = tiles * ks.chunks;
            auto start = [&](int i) { return i >= items ? m.bytes() : size_t(i / ks.chunks) * per + size_t(i % ks.chunks) * step; };
            b0 = start(int(int64_t(items) * t / n)), b1 = start(int(int64_t(items) * (t + 1) / n));
        } else {
            const int v = (t + m.rot) % n;
            b0 = per * size_t(int64_t(tiles) * v / n), b1 = per * size_t(int64_t(tiles) * (v + 1) / n);
        }
        const volatile uint8_t *d = m.data.data();
        for (size_t b = b0 & ~size_t(255); b < b1; b += 256) sink += d[b];
    }
    static thread_local uint64_t keep;
    keep += sink;
}

// L2 prefetch of thread t of n's tile share of these jobs (the bytes gemv_share will read): issued before a phase
// whose kernel is compute-bound, for the phase after it (opt-in STRATA_RUNNER_PREFETCH_NEXT=1: measured neutral)
void prefetch_share(const std::vector<Job> &jobs, int t, int n) {
    static const bool on = std::getenv("STRATA_RUNNER_PREFETCH_NEXT") && std::string(std::getenv("STRATA_RUNNER_PREFETCH_NEXT")) == "1";
    if (!on) return;
    for (const auto &j : jobs) {
        const Matrix &m = *j.m;
        const int tiles = m.padded / 16, v = (t + m.rot) % n;
        const size_t per = m.bytes() / size_t(tiles);
        const size_t b0 = per * size_t(int64_t(tiles) * v / n), b1 = per * size_t(int64_t(tiles) * (v + 1) / n);
        for (size_t b = b0 & ~size_t(255); b < b1; b += 256) __builtin_prefetch(m.data.data() + b, 0, 2);
    }
}

void prefetch_mats(const std::vector<const Matrix *> &ms, int t, int n) {   // the same for whole-tile matrices
    std::vector<Job> jobs;
    for (const Matrix *m : ms)
        if (m && m->rows && !ksplit(*m, n).chunks && !mhc_layout(*m)) jobs.push_back({m, nullptr, nullptr});
    prefetch_share(jobs, t, n);
}

// thread t of n's whole tiles of each job, inside a team region the caller already opened (gemv_multi's split and
// placement for matrices without a K split); every y holds the matrix's padded rows
// Long passes (every job > 4 tokens): one item list over all the jobs, (tile pair, token block) items in (job, tile,
// block) order, contiguous ranges of equal cost per thread. Whole tiles per job left the busiest thread with 4 tiles
// of 3 x 48 against 3.06 on average (KDA q/k/v: 692 GMAC/s on 47 threads, 10.7 MAC/cycle per thread on 24).
// tile_kernel per item: a token's arithmetic does not depend on its group, so the bits are the same.
// STRATA_RUNNER_GEMV_BALANCE=0: whole tiles per job (and token_split for few-tile jobs)
bool gemv_share_balanced(const std::vector<Job> &jobs, int t, int n) {
    const Activation *ap[MAX_PASS];
    int cur = -1;
    return balanced_items(jobs, t, n, [&](const ShareItem &it) {
        const Job &j = jobs[it.j];
        if (it.j != cur) job_acts(j, ap), cur = it.j;
        const int ldy = j.nt > 1 ? j.ldy : j.m->padded;
        tile_kernel(*j.m, ap + it.b0, it.bn, it.tile, it.tile + it.ntile, 0, j.m->cols, j.y + size_t(it.b0) * ldy, ldy);
    });
}

void gemv_share(const std::vector<Job> &jobs, int t, int n);
// The routed experts of a long pass (many jobs of a few tokens each, weights streamed from HBM every pass): every
// tile stays in the CMG whose thread placed it (gemv_share's split: weights read from the CMG's own memory), and
// the CMG's tiles go in equal contiguous cost ranges, tile by tile, to its threads. The placement split gave the
// busiest thread 9 % more than the mean (gate / up). STRATA_RUNNER_EXPERT_BALANCE=0: gemv_share
void gemv_share_cmg(const std::vector<Job> &jobs, int t, int n) {
    static const bool on = !(std::getenv("STRATA_RUNNER_EXPERT_BALANCE") && std::string(std::getenv("STRATA_RUNNER_EXPERT_BALANCE")) == "0");
    const int gsz = cmg_threads(), g = t / gsz, m0 = g * gsz, m1 = std::min(n, m0 + gsz), mc = m1 - m0;
    if (!on || n < 2 || mc < 2) { gemv_share(jobs, t, n); return; }
    // this CMG's tiles: per job and member thread, the placement's tile range (job order, then member order)
    struct R { int j, t0, t1; };
    thread_local std::vector<R> rs;
    rs.clear();
    double total = 0;
    for (int ji = 0; ji < int(jobs.size()); ++ji) {
        const Matrix &m = *jobs[ji].m;
        const int tiles = m.padded / 16;
        for (int u = m0; u < m1; ++u) {
            const int v = (u + m.rot) % n, a = int(int64_t(tiles) * v / n), b = int(int64_t(tiles) * (v + 1) / n);
            if (b > a) rs.push_back({ji, a, b}), total += double(b - a) * jobs[ji].nt * m.cols;
        }
    }
    if (total <= 0) return;
    const int li = t - m0;
    const Activation *ap[MAX_PASS];
    int cur = -1;
    const double lo = total * li / mc, hi = total * (li + 1) / mc;
    double start = 0;
    for (const R &r : rs) {
        const Job &j = jobs[r.j];
        const double u = double(j.nt) * j.m->cols, end = start + u * (r.t1 - r.t0);
        if (end > lo && start < hi) {   // tile k of the range goes to the thread whose range holds its midpoint
            const int k0 = std::clamp(int(std::ceil((lo - start) / u - 0.5)), 0, r.t1 - r.t0);
            const int k1 = std::clamp(int(std::ceil((hi - start) / u - 0.5)), 0, r.t1 - r.t0);
            if (k1 > k0) {
                if (r.j != cur) job_acts(j, ap), cur = r.j;
                if (j.xt) {   // the use's transposed activations (the plan's gf4_w4a8_gemm_prep)
                    thread_local std::vector<gf4_q8act> q;
                    q.resize(j.nt);
                    for (int i = 0; i < j.nt; ++i)
                        q[i] = gf4_q8act{j.m->cols, const_cast<int8_t *>(ap[i]->q.data()), const_cast<float *>(ap[i]->qs.data()), ap[i]->group};
                    gf4_w4a8_gemm_t(j.m->data.data(), j.m->cols, j.m->global, r.t0 + k0, r.t0 + k1, q.data(), j.nt, j.xt, j.xs, j.y, j.ldy);
                } else
                    tile_kernel(*j.m, ap, j.nt, r.t0 + k0, r.t0 + k1, 0, j.m->cols, j.y, j.nt > 1 ? j.ldy : j.m->padded);
            }
        }
        start = end;
        if (start >= hi) break;
    }
}

void gemv_share(const std::vector<Job> &jobs, int t, int n) {
    if (gemv_share_balanced(jobs, t, n)) return;
    const Activation *ap[MAX_PASS];
    for (const auto &j : jobs) {
        const Matrix &m = *j.m;
        const int tiles = m.padded / 16, v = (t + m.rot) % n;
        if (const TokenSplit ts = token_split(m, j.nt, n); ts.blocks) {
            job_acts(j, ap);
            token_split_run(m, ts, ap, j.nt, t, n, j.y, j.ldy);
            continue;
        }
        const int t0 = int(int64_t(tiles) * v / n), t1 = int(int64_t(tiles) * (v + 1) / n);
        if (t1 <= t0) continue;
        job_acts(j, ap);
        tile_kernel(m, ap, j.nt, t0, t1, 0, m.cols, j.y, j.nt > 1 ? j.ldy : m.padded);
    }
}

Vec gemv(const Matrix &m, const Vec &x) {
    Activation a;
    a.set(x.data(), int(x.size()), needs_q8(m));
    if (int(x.size()) != m.cols) throw std::runtime_error("runner: GEMV width mismatch");
    Vec y(m.rows);
    gemv(m, a, y.data());
    return y;
}

// ref::kda over the team in (head, 16-column group) items. Every column of a head's state is independent; a group's
// 16 columns run in the 16 lanes of one vector, each lane with the reference's operation order for its column, and
// with the rounding the one-column loop got from clang on A64FX: its two sums vectorized as a multiply then an ordered
// add (fadda: unfused), the state update as a fused multiply-add. So a lane's bits equal one column at a time. State layout [head][group][row][16]: an item owns one contiguous 8 KB block (no line
// shared between threads: row-major state with several items per 256-byte line cost 1.1 ms per 6-head layer), and 48
// items keep the team busy. One column per item ran two serial chains of 128 dependent FMAs per column (2.0 ms per
// token); 64-column groups with compiler vectorization were slower (2.5 ms: 12 items).
constexpr int KDA_GROUP = 16;
struct KdaScratch { std::vector<float> nq, nk, decay, qs, ks; };
// one column group of one head for one token: S = S0 * decay per row, the delta rule, out = S^T q (state rows of 16 lanes)
static inline void kda_group_step(const float *S0, float *S, const float *kh, const float *qh, const float *eh, const float *v, float sb,
                                  float *out, int dim) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    const svbool_t pg = svptrue_b32();
    svfloat32_t mem = svdup_n_f32(0);
    for (int i = 0; i < dim; ++i) {
        const svfloat32_t s0 = svmul_n_f32_x(pg, svld1_f32(pg, S0 + i * KDA_GROUP), eh[i]);   // S[i] *= eh[i]
        svst1_f32(pg, S + i * KDA_GROUP, s0);
        mem = svadd_f32_x(pg, mem, svmul_n_f32_x(pg, s0, kh[i]));                             // mem += S[i] * kh[i]
    }
    const svfloat32_t delta = svmul_n_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, v), mem), sb);
    svfloat32_t o = svdup_n_f32(0);
    for (int i = 0; i < dim; ++i) {
        const svfloat32_t s1 = svmla_n_f32_x(pg, svld1_f32(pg, S + i * KDA_GROUP), delta, kh[i]);   // s += kh[i] * delta
        svst1_f32(pg, S + i * KDA_GROUP, s1);
        o = svadd_f32_x(pg, o, svmul_n_f32_x(pg, s1, qh[i]));                                         // o += s * qh[i]
    }
    svst1_f32(pg, out, o);
#else  /* the same per lane, one column at a time */
    float mem[KDA_GROUP] = {0}, delta[KDA_GROUP], o[KDA_GROUP] = {0};
    for (int i = 0; i < dim; ++i)
        for (int c = 0; c < KDA_GROUP; ++c) {
            float &sv = S[i * KDA_GROUP + c];
            sv = S0[i * KDA_GROUP + c] * eh[i];
            mem[c] += sv * kh[i];
        }
    for (int c = 0; c < KDA_GROUP; ++c) delta[c] = (v[c] - mem[c]) * sb;
    for (int i = 0; i < dim; ++i)
        for (int c = 0; c < KDA_GROUP; ++c) {
            float &sv = S[i * KDA_GROUP + c];
            sv += kh[i] * delta[c];
            o[c] += sv * qh[i];
        }
    for (int c = 0; c < KDA_GROUP; ++c) out[c] = o[c];
#endif
}
// G consecutive column groups of one head for one token: kda_group_step per group, the groups' chains interleaved (a
// group's two sums are chains of `dim` dependent adds; one group at a time ran at the add latency, not its rate).
// Group g's state block is G0 + g * dim * 16, its v and out lanes + 16 g; per lane the same operations
template <int G>
static inline void kda_groups_step(const float *S0, float *S, const float *kh, const float *qh, const float *eh, const float *v, float sb,
                                   float *out, int dim) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    const svbool_t pg = svptrue_b32();
    const size_t bs = size_t(dim) * KDA_GROUP;
    vf32 mem[G], o[G], delta[G];
    for (int g = 0; g < G; ++g) mem[g] = svdup_n_f32(0), o[g] = svdup_n_f32(0);
    for (int i = 0; i < dim; ++i) {
        const float e = eh[i], k = kh[i];
#pragma GCC unroll 4
        for (int g = 0; g < G; ++g) {
            const svfloat32_t s0 = svmul_n_f32_x(pg, svld1_f32(pg, S0 + g * bs + i * KDA_GROUP), e);
            svst1_f32(pg, S + g * bs + i * KDA_GROUP, s0);
            mem[g] = svadd_f32_x(pg, mem[g], svmul_n_f32_x(pg, s0, k));
        }
    }
    for (int g = 0; g < G; ++g) delta[g] = svmul_n_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, v + 16 * g), mem[g]), sb);
    for (int i = 0; i < dim; ++i) {
        const float k = kh[i], q = qh[i];
#pragma GCC unroll 4
        for (int g = 0; g < G; ++g) {
            const svfloat32_t s1 = svmla_n_f32_x(pg, svld1_f32(pg, S + g * bs + i * KDA_GROUP), delta[g], k);
            svst1_f32(pg, S + g * bs + i * KDA_GROUP, s1);
            o[g] = svadd_f32_x(pg, o[g], svmul_n_f32_x(pg, s1, q));
        }
    }
    for (int g = 0; g < G; ++g) svst1_f32(pg, out + 16 * g, o[g]);
#else
    for (int g = 0; g < G; ++g)
        kda_group_step(S0 + size_t(g) * dim * KDA_GROUP, S + size_t(g) * dim * KDA_GROUP, kh, qh, eh, v + 16 * g, sb, out + 16 * g, dim);
#endif
}
// half a column group (8 of its 16 lanes, from lane lo): per lane the operations of kda_group_step, so the bits are
// those of the whole group; 6-head ranks have 48 groups for 47 threads, 96 halves split the recurrence evenly
static bool kda_halves() {   // STRATA_RUNNER_KDA_HALVES=1: the recurrence in 8-lane halves when groups > threads
    static const bool on = std::getenv("STRATA_RUNNER_KDA_HALVES") && std::string(std::getenv("STRATA_RUNNER_KDA_HALVES")) == "1";
    return on;
}
static inline void kda_half_step(const float *S0, float *S, const float *kh, const float *qh, const float *eh, const float *v, float sb,
                                 float *out, int dim, int lo) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    const svbool_t pg = svwhilelt_b32(0, 8);   // lanes lo .. lo + 7 (pointers offset by lo)
    svfloat32_t mem = svdup_n_f32(0);
    for (int i = 0; i < dim; ++i) {
        const svfloat32_t s0 = svmul_n_f32_x(pg, svld1_f32(pg, S0 + i * KDA_GROUP + lo), eh[i]);   // S[i] *= eh[i]
        svst1_f32(pg, S + i * KDA_GROUP + lo, s0);
        mem = svadd_f32_x(pg, mem, svmul_n_f32_x(pg, s0, kh[i]));                             // mem += S[i] * kh[i]
    }
    const svfloat32_t delta = svmul_n_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, v + lo), mem), sb);
    svfloat32_t o = svdup_n_f32(0);
    for (int i = 0; i < dim; ++i) {
        const svfloat32_t s1 = svmla_n_f32_x(pg, svld1_f32(pg, S + i * KDA_GROUP + lo), delta, kh[i]);   // s += kh[i] * delta
        svst1_f32(pg, S + i * KDA_GROUP + lo, s1);
        o = svadd_f32_x(pg, o, svmul_n_f32_x(pg, s1, qh[i]));                                         // o += s * qh[i]
    }
    svst1_f32(pg, out + lo, o);
#else
    (void)lo;
    if (lo == 0) kda_group_step(S0, S, kh, qh, eh, v, sb, out, dim);
#endif
}
// the recurrence's work for every thread of a team region that is already open (kda_parallel opens one; the fused
// KDA layer calls it inside its own): the per-head sums on one thread, then two work-shared loops
double g_kda_tp[3];   // thread 0's times after the sums and the normalization (kda.core's split)
void kda_core_team(KdaScratch &w, const float *state_in, float *state, const float *q, const float *k, const float *v, const float *g,
                   const float *beta, float *out, int heads, int dim) {   // state_in == state: in place
    std::vector<float> &nq = w.nq, &nk = w.nk, &decay = w.decay, &qs = w.qs, &ks = w.ks;
    if (nq.size() < size_t(heads) * dim) {   // sized once (every team member sees the sizes after the barrier)
        team_barrier();
        if (omp_get_thread_num() == 0) nq.resize(size_t(heads) * dim), nk.resize(size_t(heads) * dim), decay.resize(size_t(heads) * dim), qs.resize(heads), ks.resize(heads);
        team_barrier();
    }
    for (int h = omp_get_thread_num(); h < heads; h += omp_get_num_threads()) {   // heads over the team, the same order per head
        float sq = 1e-6f, sk = 1e-6f;
        for (int i = 0; i < dim; ++i) {
            sq += q[h * dim + i] * q[h * dim + i];
            sk += k[h * dim + i] * k[h * dim + i];
        }
        qs[h] = sq, ks[h] = sk;
    }
    team_barrier();
    if (omp_get_thread_num() == 0) g_kda_tp[0] = now();
    const int groups = dim / KDA_GROUP;
    {
    // normalized q, k and the decay per element over the team (a serial pass was ~40 us: one exp, two sqrt and two
    // divisions per element), then the column groups
#pragma omp for schedule(static) nowait
    for (int hi = 0; hi < heads * dim; ++hi) {
        const int h = hi / dim;
        nq[hi] = q[hi] / std::sqrt(qs[h]) / std::sqrt((float)dim);
        nk[hi] = k[hi] / std::sqrt(ks[h]);
        decay[hi] = std::exp(g[hi]);
    }
    team_barrier();
    if (omp_get_thread_num() == 0) g_kda_tp[1] = now();
    const bool halves = kda_halves() && heads * groups > omp_get_num_threads();   // 96 halves for 48 groups (opt-in)
#pragma omp for schedule(static) nowait
    for (int it = 0; it < (halves ? 2 : 1) * heads * groups; ++it) {
        const int hg = halves ? it / 2 : it, lo = halves ? 8 * (it % 2) : 0;
        const int h = hg / groups, j0 = (hg % groups) * KDA_GROUP;
        float *S = state + size_t(hg) * dim * KDA_GROUP;   // row i: S[i * 16 + lane], lane = column j0 + lane
        const float *S0 = state_in + size_t(hg) * dim * KDA_GROUP;   // the state before this token (S itself in place)
        const float *kh = nk.data() + size_t(h) * dim, *qh = nq.data() + size_t(h) * dim, *eh = decay.data() + size_t(h) * dim;
        if (halves) kda_half_step(S0, S, kh, qh, eh, v + h * dim + j0, ref::sigmoid(beta[h]), out + h * dim + j0, dim, lo);
        else kda_group_step(S0, S, kh, qh, eh, v + h * dim + j0, ref::sigmoid(beta[h]), out + h * dim + j0, dim);
    }
    team_barrier();
    }
}

void kda_parallel(float *state, const float *q, const float *k, const float *v, const float *g, const float *beta, float *out,
                  int heads, int dim) {
    KdaScratch w;
    team_run([&] { kda_core_team(w, state, state, q, k, v, g, beta, out, heads, dim); });
}

// ------------------------------------------------------------------------------------------------- weight loading
struct Loader {
    const strata::core::ModelArtifact &a;
    const GlmModelOpt &opt;
    const strata::artifact::Safetensors &st;
    bool w4a8;      // routed experts: W4A8 SDOT (rt16q) instead of exact W4A16 (rt16p)
    bool trunk_q8;  // E4M3 trunk tensors requantized to int8 per (row, 32) instead of exact
    size_t total = 0;
    // Per-rank image of everything this rank loads, in load order: written on the first load (`out`), replayed by later
    // runs of the same layout (`in`) instead of reading and packing the checkpoint again.
    std::ofstream *out = nullptr;
    std::ifstream *in = nullptr;
    bool replaying() const { return in != nullptr; }
    template <class T> void put(const T *p, size_t n) { if (out) out->write(reinterpret_cast<const char *>(p), sizeof(T) * n); }
    bool whole_k = false;    // the next matrices are never K-split (Matrix::whole_k)
    bool force_q8 = false;   // the next FP8 matrices requantized to int8 (the prefill trunk copies)
    int in_fd = -1;          // the image again, for dropping its page cache behind the reads (-1: keep it)
    size_t in_done = 0, in_dropped = 0;
    template <class T> void get(T *p, size_t n) {
        if (!in->read(reinterpret_cast<char *>(p), sizeof(T) * n)) throw std::runtime_error("runner: truncated image");
        in_done += sizeof(T) * n;
        if (in_fd >= 0 && in_done - in_dropped >= (size_t(256) << 20)) {   // the page cache would take the HBM first touch needs
            posix_fadvise(in_fd, 0, off_t(in_done & ~size_t(65535)), POSIX_FADV_DONTNEED);
            in_dropped = in_done;
        }
    }
    Matrix record(Matrix m) {
        if (out) {
            const int32_t h[4] = {int32_t(m.kind), m.rows, m.padded, m.cols};
            const uint64_t n = m.data.size();
            put(h, 4), put(&m.global, 1), put(&n, 1), put(m.data.data(), n);
        }
        return m;
    }
    Matrix replay() {
        Matrix m;
        int32_t h[4];
        uint64_t n;
        get(h, 4), get(&m.global, 1), get(&n, 1);
        m.kind = Kind(h[0]), m.rows = h[1], m.padded = h[2], m.cols = h[3];
        std::vector<uint8_t> buf(n);
        get(buf.data(), n);
        m.rot = take_rotation();
        m.whole_k = whole_k;
        m.data = place(buf, m);
        total += n;
        return m;
    }
    Vec vec_record(Vec v) {
        if (out) { const uint64_t n = v.size(); put(&n, 1), put(v.data(), n); }
        return v;
    }
    Vec vec_replay() {
        uint64_t n;
        get(&n, 1);
        Vec v(n);
        get(v.data(), n);
        return v;
    }
    // rows [r0, r1) of a checkpoint matrix: element bytes by pread (a mapped shared file system pays one round trip per
    // page fault), scales decoded whole, global scale
    struct Source {
        ModelOptPacked p;
        int r0 = 0, r1 = 0;
        size_t pitch = 0;
        std::vector<uint8_t> bytes;
        size_t rows() const { return p.rows; }
        size_t columns() const { return p.columns; }
        const uint8_t *row(int r) const { return bytes.data() + size_t(r - r0) * pitch; }
    };
    Source load(const std::string &name, int r0 = 0, int r1 = -1) const {
        const auto mm = opt.matrix(name);
        Source s;
        if (in) {   // replay: shape only, nothing is read
            s.p = ModelOptPacked{mm.format, mm.rows, mm.columns, 1, {}, {}};
            s.r0 = r0, s.r1 = r1 < 0 ? int(mm.rows) : r1;
            return s;
        }
        s.p = opt.load(mm, true);   // maps the shard without touching the weights; decodes the scales
        s.r0 = r0, s.r1 = r1 < 0 ? int(mm.rows) : r1;
        s.pitch = mm.format == ModelOptFormat::Nvfp4 ? mm.columns / 2 : mm.format == ModelOptFormat::Bf16 ? mm.columns * 2
                : mm.format == ModelOptFormat::F32 ? mm.columns * 4 : mm.columns;
        s.bytes.resize(size_t(s.r1 - s.r0) * s.pitch);
        st.read(*mm.weights, uint64_t(s.r0) * s.pitch, s.bytes.data(), s.bytes.size());
        return s;
    }
    static float value(const Source &s, int r, int c) {
        const auto &p = s.p;
        if (p.format == ModelOptFormat::Mxfp8) return strata_fp4_e4m3(s.row(r)[c]) * strata_fp4_e8m0(p.scales[size_t(r) * (p.columns / 32) + c / 32]);
        if (p.format == ModelOptFormat::BlockFp8) {
            float scale;
            std::memcpy(&scale, p.scales.data() + 4 * (size_t(r / 128) * ((p.columns + 127) / 128) + c / 128), 4);
            return strata_fp4_e4m3(s.row(r)[c]) * scale;
        }
        throw std::runtime_error("runner: requantization source must be E4M3");
    }
    static std::string src(int layer, const std::string &rest) { return "model.language_model.layers." + std::to_string(layer) + "." + rest; }

    // a packed matrix before placement: what one rank packs for another in the cooperative expert load
    struct Packed {
        Matrix m;   // header fields only
        std::vector<uint8_t> buf;
    };
    // rows [r0, r1) x columns [c0, c1) of a checkpoint matrix (rows inside the loaded source), packed for its kernel
    Matrix slice(const Source &src_rows, int r0, int r1, int c0, int c1, bool fast_requant) {
        if (in) return replay();
        return finish(pack(src_rows, r0, r1, c0, c1, fast_requant));
    }
    // placement (first touch by the GEMV split), accounting and the image record, in load order
    // the next matrix's thread rotation: the counter's, or rot_next once when set (the counter advances either way,
    // so every other matrix keeps its rotation)
    int rot_next = -1;
    int take_rotation() {
        const int r = next_rotation();
        const int v = rot_next >= 0 ? rot_next : r;
        rot_next = -1;
        return v;
    }
    Matrix finish(Packed pk) {
        Matrix m = std::move(pk.m);
        m.rot = take_rotation();
        m.whole_k = whole_k;
        m.data = place(pk.buf, m);
        total += m.bytes();
        return record(std::move(m));
    }
    Packed pack(const Source &src_rows, int r0, int r1, int c0, int c1, bool fast_requant) const {
        const auto &p = src_rows.p;
        if (r0 < src_rows.r0 || r1 > src_rows.r1) throw std::runtime_error("runner: slice outside the loaded rows");
        Matrix m;
        m.rows = r1 - r0, m.cols = c1 - c0, m.padded = (m.rows + 15) / 16 * 16;
        const int R = m.padded, C = m.cols;
        std::vector<uint8_t> buf;
        if (p.format == ModelOptFormat::Nvfp4) {
            if (c0 % 64 || C % 64) throw std::runtime_error("runner: NVFP4 column slice must be 64-aligned");
            std::vector<uint8_t> codes(size_t(R) * C / 2, 0), scales(size_t(R) * C / 16, 0);
            for (int r = 0; r < m.rows; ++r) {
                std::memcpy(&codes[size_t(r) * C / 2], src_rows.row(r0 + r) + c0 / 2, C / 2);
                std::memcpy(&scales[size_t(r) * C / 16], p.scales.data() + size_t(r0 + r) * p.columns / 16 + c0 / 16, C / 16);
            }
            m.kind = w4a8 ? Kind::NvQ : Kind::NvP;
            m.global = p.global;
            buf.resize(gf4_nvfp4_packed_bytes(R, C));
            if ((w4a8 ? gf4_pack_rt16q : gf4_pack_rt16p)(codes.data(), scales.data(), R, C, buf.data()))
                throw std::runtime_error("runner: NVFP4 scale outside [0, 448]");
        } else if (p.format == ModelOptFormat::Bf16) {
            std::vector<uint16_t> w(size_t(R) * C, 0);
            for (int r = 0; r < m.rows; ++r) std::memcpy(&w[size_t(r) * C], src_rows.row(r0 + r) + size_t(c0) * 2, size_t(C) * 2);
            m.kind = Kind::B16;
            buf.resize(size_t(R) * C * 2);
            gf4_pack_b16k(w.data(), R, C, buf.data());
        } else if (fast_requant && (trunk_q8 || force_q8)) {
            Vec w(size_t(R) * C, 0.f);
#pragma omp parallel for schedule(static)
            for (int r = 0; r < m.rows; ++r)
                for (int c = 0; c < C; ++c) w[size_t(r) * C + c] = value(src_rows, r0 + r, c0 + c);
            m.kind = Kind::Q8;
            buf.resize(gf4_q8k_packed_bytes(R, C));
            gf4_pack_q8k(w.data(), R, C, buf.data());
        } else if (p.format == ModelOptFormat::Mxfp8) {
            if (c0 % 32 || C % 32) throw std::runtime_error("runner: MXFP8 column slice must be 32-aligned");
            std::vector<uint8_t> w(size_t(R) * C, 0), sc(size_t(R) * C / 32, 127);
            for (int r = 0; r < m.rows; ++r) {
                std::memcpy(&w[size_t(r) * C], src_rows.row(r0 + r) + c0, C);
                std::memcpy(&sc[size_t(r) * C / 32], p.scales.data() + size_t(r0 + r) * (p.columns / 32) + c0 / 32, C / 32);
            }
            for (auto v : sc) if (v == 0 || v == 255) throw std::runtime_error("runner: MXFP8 scale outside the decoded range");
            m.kind = Kind::Mx;
            buf.resize(gf4_e8k_packed_bytes(R, C, 1));
            gf4_pack_e8k_mx(w.data(), sc.data(), R, C, buf.data());
        } else if (p.format == ModelOptFormat::BlockFp8) {
            if (r0 % 128 || c0 % 128 || C % 128) throw std::runtime_error("runner: block FP8 slice must be 128-aligned");
            std::vector<uint8_t> w(size_t(R) * C, 0);
            const int rb = (R + 127) / 128, cb = C / 128, pcb = int((p.columns + 127) / 128);
            Vec sc(size_t(rb) * cb, 0.f);
            for (int r = 0; r < m.rows; ++r) std::memcpy(&w[size_t(r) * C], src_rows.row(r0 + r) + c0, C);
            for (int i = 0; i < rb; ++i)
                for (int j = 0; j < cb; ++j)
                    if (r0 / 128 + i < int((p.rows + 127) / 128))
                        std::memcpy(&sc[size_t(i) * cb + j], p.scales.data() + 4 * (size_t(r0 / 128 + i) * pcb + c0 / 128 + j), 4);
            m.kind = Kind::Bfp;
            buf.resize(gf4_e8k_packed_bytes(R, C, 0));
            gf4_pack_e8k_bfp8(w.data(), sc.data(), R, C, buf.data());
        } else throw std::runtime_error("runner: unsupported matrix format");
        return Packed{std::move(m), std::move(buf)};
    }
    Matrix whole(const std::string &name, bool fast_requant = true) {
        const auto s = load(name);
        return slice(s, 0, int(s.rows()), 0, int(s.columns()), fast_requant);
    }
    // rows [r0, r1), every column
    Matrix rows(const std::string &name, int r0, int r1, bool fast_requant = true) {
        const auto s = load(name, r0, r1);
        return slice(s, r0, r1, 0, int(s.columns()), fast_requant);
    }
    size_t row_count(const std::string &name) const { return opt.matrix(name).rows; }
    // F32 copy of a small tensor through the canonical binding (norms, biases, conv, mHC scalars)
    Vec f32(const std::string &canonical) {
        if (in) return vec_replay();
        return vec_record(f32_checkpoint(canonical));
    }
    Vec f32_checkpoint(const std::string &canonical) const {
        const auto &t = a.at(canonical);
        Vec out(t.tensor->elements());
        const uint8_t *d = t.data();
        if (t.tensor->type == 0) std::memcpy(out.data(), d, out.size() * 4);
        else if (t.tensor->type == 30)
            for (size_t i = 0; i < out.size(); ++i) {
                const uint32_t v = (uint32_t(d[2 * i]) | (uint32_t(d[2 * i + 1]) << 8)) << 16;
                std::memcpy(&out[i], &v, 4);
            }
        else throw std::runtime_error("runner: unexpected small tensor type " + canonical);
        return out;
    }
};

struct Span { int begin = 0, end = 0; int size() const { return end - begin; } };
Span split(int units, int world, int rank) {   // contiguous, the first `units % world` ranks one unit more
    const int base = units / world, extra = units % world;
    const int begin = rank * base + std::min(rank, extra);
    return {begin, begin + base + (rank < extra)};
}
// split() with the remainder on ranks off, off + 1, ... instead of 0, 1, ... (the ranges follow (rank - off) mod world)
Span split_rot(int units, int world, int rank, int off) { return split(units, world, ((rank - off) % world + world) % world); }

// ------------------------------------------------------------------------------------------------- the model
struct KdaLayer {
    Matrix q, k, v, f_a, g_a, f_b, g_b, beta, o;   // q/k/v/f_b/g_b: owned heads' rows; o: owned heads' columns
    Vec conv[3], dt, a_neg, onorm;
    // the recurrent state ([head][group][row][16]) in placed arena buffers: `state` is the current one; a multi-token
    // pass with snapshots leaves the state after each of its tokens in its own buffer of `pool` (rollback = pointer)
    float *state = nullptr;
    size_t state_n = 0;
    std::array<float *, 4> pool{};
    Vec hist[3];
};
struct MlaLayer {
    Matrix q_a, q_b, kv_a, iq, iw, ik, igate, o;   // q_b: owned heads' rows; o: owned heads' columns
    // front sharding (STRATA_RUNNER_MLA_SHARD, default on): q_a, kv_a, iw, ik, igate (stage 1) and iq (stage 2) hold
    // this rank's rows [r0, r1) of the full matrix; the full outputs are one all-reduce of zero-padded rows per stage
    bool sharded = false;
    int full[6] = {0, 0, 0, 0, 0, 0}, r0[6] = {0, 0, 0, 0, 0, 0};   // q_a, kv_a, iw, ik, igate, iq
    std::vector<Matrix> kb, vb;                     // per owned head: K absorbed (512 x 256), V (256 x 512)
    Vec q_a_norm, kv_a_norm, k_norm_w, k_norm_b, ape;
    Vec keys, gates, pooled, cache;
    // hybrid latent cache (contexts beyond STRATA_RUNNER_CACHE_HOT positions): `cache` is a ring of the last `ring`
    // rows on every rank and `cold` this rank's 256-position blocks (block b on rank b % world); ring 0: `cache`
    // holds every position (the replicated cache)
    Vec cold;
    int ring = 0;
};
constexpr int COLD_BLOCK = 256;
struct FfnDense { Matrix gate, up, down; };
struct ExpertSlice { int expert = -1, part = 0; Matrix gate, up, down; };
struct MoeLayer {
    Matrix router, sgate, sup, sdown;
    int router_r0 = 0;   // router sharded (STRATA_RUNNER_ROUTER_SHARD=1, opt-in): its rows [r0, r0 + rows) of the experts
    Vec bias;
    std::vector<ExpertSlice> slices;   // this rank's (expert, part) slices
    int quant = -1;                    // any int8-activation kernel among the shared expert and slices (cached: a scan touches 288 slices)
    std::vector<int> owner_index;       // expert -> index into slices or -1
};
// ref::router with a top-k selection instead of a stable sort of all experts (35 us per call on A64FX): the k largest
// sigmoid(logit) + bias in descending order, ties to the lower expert index (the stable sort's order), same weights
// the selection of route_topk on scores and keys computed already (scores[i] = sigmoid(logit), key = score + bias)
void route_select(const float *scores, const float *key, int ne, int k, float scale, int *ids, float *weights) {
    int count = 0;
    float thr = 0;   // key[ids[k - 1]] once the list is full (kept in a register: the same comparisons)
    for (int i = 0; i < ne; ++i) {
        if (count == k && !(key[i] > thr)) continue;
        int p = count < k ? count++ : k - 1;
        while (p > 0 && key[ids[p - 1]] < key[i]) ids[p] = ids[p - 1], --p;
        ids[p] = i;
        if (count == k) thr = key[ids[k - 1]];
    }
    float sum = 0;
    for (int i = 0; i < k; ++i) sum += scores[ids[i]];
    for (int i = 0; i < k; ++i) weights[i] = scale * scores[ids[i]] / (sum + 1e-20f);
}
void route_topk(const float *logits, const float *bias, int ne, int k, float scale, int *ids, float *weights) {
    std::vector<float> scores(ne), key(ne);
    for (int i = 0; i < ne; ++i) scores[i] = ref::sigmoid(logits[i]), key[i] = scores[i] + bias[i];
    int count = 0;
    for (int i = 0; i < ne; ++i) {
        if (count == k && !(key[i] > key[ids[k - 1]])) continue;
        int p = count < k ? count++ : k - 1;
        while (p > 0 && key[ids[p - 1]] < key[i]) ids[p] = ids[p - 1], --p;
        ids[p] = i;
    }
    float sum = 0;
    for (int i = 0; i < k; ++i) sum += scores[ids[i]];
    for (int i = 0; i < k; ++i) weights[i] = scale * scores[ids[i]] / (sum + 1e-20f);
}

// Indexer scores of `pools` pooled keys: score[b] = sum_h iw[h] * max(dot(iq_h, pooled_b) / sqrt(dim), 0) / sqrt(heads),
// each dot a double sum over d in order. The reference loop ran one serial chain of 128 dependent double FMAs per (pool,
// head): ~80 us per MLA layer at 32 positions, growing with the context. Here eight heads share a vector (each lane
// one head, its d order unchanged) and two pools run together for more independent chains; the head sum stays scalar
// in head order, so every score keeps its bits.
// the head sum of one pool's dots, shared by both paths (explicit fmaf in head order, never vectorized)
static float index_head_sum(const double *dots, const float *iw, int heads, float rs, float rh) {
    float score = 0;
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
    for (int h = 0; h < heads; ++h) score = std::fmaf(iw[h], std::max(float(dots[h]) / rs, 0.f), score);
    return score / rh;
}
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
// pools b0 and b0 + 1 (b0 alone at the end) of a 32-head indexer: iqt is iq transposed to [d][h] in double
static inline void index_pair32(const double *iqt, const float *iw, const float *pooled, int b0, int pools, int dim, float rs, float rh, float *scores) {
    constexpr int HV = 4;
    typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
    const int nb = std::min(2, pools - b0);
    const svbool_t pg = svptrue_b64();
    vf64 acc0[HV], acc1[HV];
    for (int v = 0; v < HV; ++v) acc0[v] = svdup_n_f64(0), acc1[v] = svdup_n_f64(0);
    const float *p0 = pooled + size_t(b0) * dim, *p1 = pooled + size_t(b0 + nb - 1) * dim;
    for (int d = 0; d < dim; ++d) {
        const double x0 = double(p0[d]), x1 = double(p1[d]);
        const double *q = iqt + size_t(d) * 32;
#pragma GCC unroll 4
        for (int v = 0; v < HV; ++v) {
            const svfloat64_t qv = svld1_f64(pg, q + 8 * v);
            acc0[v] = svmla_n_f64_x(pg, acc0[v], qv, x0);   // the products of two floats are exact in double:
            acc1[v] = svmla_n_f64_x(pg, acc1[v], qv, x1);   // fused or not, the same sums
        }
    }
    double dots[2][32];
    for (int v = 0; v < HV; ++v) svst1_f64(pg, dots[0] + 8 * v, acc0[v]), svst1_f64(pg, dots[1] + 8 * v, acc1[v]);
    for (int j = 0; j < nb; ++j) scores[b0 + j] = index_head_sum(dots[j], iw, 32, rs, rh);
}
#endif
void index_scores(const float *iq, const float *iw, const float *pooled, int pools, int heads, int dim, float *scores) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    if (heads == 32) {   // 4 vectors of 8 heads, compile-time, so the eight chains stay in registers
        std::vector<double> iqt(size_t(dim) * 32);   // [d][h]
        for (int h = 0; h < 32; ++h)
            for (int d = 0; d < dim; ++d) iqt[size_t(d) * 32 + h] = double(iq[h * dim + d]);
        const float rs = std::sqrt(float(dim)), rh = std::sqrt(float(heads));
        if (!(pools >= 64))
            for (int b0 = 0; b0 < pools; b0 += 2) index_pair32(iqt.data(), iw, pooled, b0, pools, dim, rs, rh, scores);
        else
            team_run([&] {
#pragma omp for schedule(static) nowait
                for (int b0 = 0; b0 < pools; b0 += 2) index_pair32(iqt.data(), iw, pooled, b0, pools, dim, rs, rh, scores);
            });
        return;
    }
#endif
    const float rs = std::sqrt(float(dim)), rh = std::sqrt(float(heads));
    if (!(pools >= 64))
        for (int b = 0; b < pools; ++b) {
            std::vector<double> dots(heads);
            for (int h = 0; h < heads; ++h) {
                double dot = 0;
                for (int d = 0; d < dim; ++d) dot += double(iq[h * dim + d]) * pooled[b * dim + d];
                dots[h] = dot;
            }
            scores[b] = index_head_sum(dots.data(), iw, heads, rs, rh);
        }
    else
    team_run([&] {
#pragma omp for schedule(static) nowait
        for (int b = 0; b < pools; ++b) {
            std::vector<double> dots(heads);
            for (int h = 0; h < heads; ++h) {
                double dot = 0;
                for (int d = 0; d < dim; ++d) dot += double(iq[h * dim + d]) * pooled[b * dim + d];
                dots[h] = dot;
            }
            scores[b] = index_head_sum(dots.data(), iw, heads, rs, rh);
        }
    });
}

struct Layer {
    bool kda = true, dense = false;
    Matrix hc_attn, hc_ffn;
    Vec hc_attn_base, hc_attn_scale, hc_ffn_base, hc_ffn_scale, attn_norm, ffn_norm;
    KdaLayer kd;
    MlaLayer ml;
    FfnDense ffn;
    MoeLayer moe;
};

struct Runner {
    strata::core::ModelArtifact &artifact;
    const strata::core::ModelDescriptor &m;
    Comm &comm;
    std::string numerics;   // exact | fast (both below) | w4a8, w4a8g16 (experts only) | q8trunk (trunk only)
    bool w4a8, trunk_q8;
    int capacity, layers_run;
    bool prefill_q8 = std::getenv("STRATA_RUNNER_PREFILL_TRUNK") && std::string(std::getenv("STRATA_RUNNER_PREFILL_TRUNK")) == "q8";
    // the int8 prefill copy of an FP8 trunk matrix (prefill_q8): the same rows and columns, the same thread rotation
    // (so its pages sit with the threads that read them), loaded right after the matrix (images replay in order)
    template <class Reload> void add_q8(Loader &ld, Matrix &m, Reload reload) {
        if (!prefill_q8 || trunk_q8 || (m.kind != Kind::Mx && m.kind != Kind::Bfp) || !m.rows) return;
        ld.force_q8 = true, ld.rot_next = m.rot;
        m.q8 = std::make_shared<Matrix>(reload());
        ld.force_q8 = false;
    }
    bool q8_act(const Matrix &mt, int T) const { return needs_q8(mt) || (T > 4 && mt.q8 != nullptr); }
    // STRATA_RUNNER_CACHE_HOT=W: contexts beyond W positions keep a hybrid latent cache (default 65536; 0: replicated)
    int cache_hot = std::getenv("STRATA_RUNNER_CACHE_HOT") ? std::atoi(std::getenv("STRATA_RUNNER_CACHE_HOT")) : 65536;
    // ... and only for capacities from STRATA_RUNNER_CACHE_HYBRID_MIN positions (default 300000: below it the
    // replicated fp32 cache fits, ~7 GiB per rank, and is exact and faster)
    int cache_hybrid_min = std::getenv("STRATA_RUNNER_CACHE_HYBRID_MIN") ? std::atoi(std::getenv("STRATA_RUNNER_CACHE_HYBRID_MIN")) : 300000;
    int cold_owner(int pos) const { return (pos / COLD_BLOCK) % comm.world; }
    size_t cold_slot(int pos) const { return size_t(pos / COLD_BLOCK / comm.world) * COLD_BLOCK + pos % COLD_BLOCK; }
    void cache_put(MlaLayer &M, int pos, const float *row) {   // the latent row of position pos into the cache
        const int lat = m.kv_rank;
        if (!M.ring) { std::copy(row, row + lat, M.cache.begin() + size_t(pos) * lat); return; }
        std::copy(row, row + lat, M.cache.begin() + size_t(pos % M.ring) * lat);
        if (cold_owner(pos) == comm.rank) std::copy(row, row + lat, M.cold.begin() + cold_slot(pos) * lat);
    }
    Span heads;          // owned attention heads (KDA and MLA)
    int parts;           // slices per routed expert
    std::vector<Layer> L;
    Matrix head;         // lm_head rows owned
    Matrix draft_head;   // the same rows requantized to int8 for the draft block's picks only (STRATA_RUNNER_DRAFT_HEAD=exact: none)
    Span vocab;
    Vec out_norm;
    std::vector<uint16_t> embedding;   // BF16 [vocab][hidden] in memory: a mapped row on a shared file system is a remote
                                       // page fault per new token (50-400 ms measured)
    int position = 0;
    uint64_t last_hash = 0;   // residual after the last layer of the last step
    std::string image_dir;   // per-rank packed image cache (node-local), empty = off
    std::map<std::string, double> timers;
    std::vector<int32_t> route_ids;
    // the MTP draft block (--mtp): layer index = main layer count, a plain residual (no mHC): x = eh_proj([enorm(emb) |
    // hnorm(h)]), x += MLA(attn_norm(x)), x += MoE(ffn_norm(x)), h' = shared_head_norm(x), logits = lm_head(h'). Cell q
    // pairs token q with the main model's output-normed hidden of q - 1 (zero at q = 0); a depth-2 step pairs the
    // next draft with h' (Strata's GPU decoder, src/program/glm_decode.cpp mtp_step / sync_mtp).
    struct MtpBlock {
        bool on = false;
        Layer layer;
        Vec enorm, hnorm, head_norm;
        Matrix eh;     // eh_proj rows eh_rows of this rank (the full row vector is one all-reduce of the zero-padded part)
        Span eh_rows;
        int position = 0;   // next cell
    } mtp;
    struct MtpState { int position = 0; Vec keys, gates; };
    MtpState mtp_save() const { return {mtp.position, mtp.layer.ml.keys, mtp.layer.ml.gates}; }
    void mtp_restore(const MtpState &st) { mtp.position = st.position, mtp.layer.ml.keys = st.keys, mtp.layer.ml.gates = st.gates; }
    // the same for context c without switching to it (its MTP state is in ctx_store unless it is active)
    MtpState mtp_save_ctx(int c) const {
        if (c == active_ctx) return mtp_save();
        const auto &cs = ctx_store[c];
        return {cs.mtp_position, cs.mtp_layer.keys, cs.mtp_layer.gates};
    }
    void mtp_restore_ctx(int c, const MtpState &st) {
        if (c == active_ctx) { mtp_restore(st); return; }
        auto &cs = ctx_store[c];
        cs.mtp_position = st.position, cs.mtp_layer.keys = st.keys, cs.mtp_layer.gates = st.gates;
    }
    int position_ctx(int c) const { return c == active_ctx ? position : ctx_store[c].position; }
    // MTP cells of several contexts in one pass: groups (context, tokens) in token order, cells[g] group g's first
    // cell. The draft block's projections and MoE run over every token; its attention per context, with that
    // context's draft layer state (mla_back's grouped path): each context's results are those of its own calls
    std::vector<int> mtp_cells;
    std::vector<Vec> mtp_forward_ctx(const std::vector<int> &tokens, const std::vector<Vec> &prev, const std::vector<std::pair<int, int>> &groups,
                                     const std::vector<int> &cells, std::vector<Vec> *logits) {
        if (groups.size() == 1) {
            switch_ctx(groups[0].first);
            return mtp_forward(tokens, prev, cells[0], logits);
        }
        pass_groups = groups, mtp_cells = cells;
        auto h = mtp_forward(tokens, prev, cells[0], logits);
        pass_groups.clear(), mtp_cells.clear();
        return h;
    }
    bool want_mtp = false;
    // STRATA_RUNNER_REMAINDER_SPREAD=1: split remainders on different ranks per tensor family (the head split puts the
    // 4 extra heads on ranks 0-3; every other split did too, so ranks 0-3 carried every extra unit)
    bool rem_spread = std::getenv("STRATA_RUNNER_REMAINDER_SPREAD") && std::string(std::getenv("STRATA_RUNNER_REMAINDER_SPREAD")) == "1";
    int rot(int off) const { return rem_spread ? off % std::max(1, comm.world) : 0; }
    bool mla_shard = !(std::getenv("STRATA_RUNNER_MLA_SHARD") && std::string(std::getenv("STRATA_RUNNER_MLA_SHARD")) == "0");
    // router rows over the ranks too (opt-in: its all-reduce, 0.87 ms per token, costs more than the 0.15 ms it saves)
    bool router_shard = mla_shard && std::getenv("STRATA_RUNNER_ROUTER_SHARD") && std::string(std::getenv("STRATA_RUNNER_ROUTER_SHARD")) == "1";
    std::vector<Vec> last_hidden;   // the last pass's output-normed hidden per token (the MTP block's input)
    std::vector<Vec> last_part;     // the last pass's logits for this rank's vocabulary rows, per token
    bool full_logits = true;        // gather full logits to rank 0 (false: greedy generation takes argmax_last())
    std::vector<int> argmax_last() {   // the last pass's greedy token per position, on every rank
        std::vector<const float *> pp;
        for (const auto &v : last_part) pp.push_back(v.data());
        std::vector<int> out(pp.size());
        comm.argmax(pp.data(), int(pp.size()), vocab.end - vocab.begin, vocab.begin, out.data());
        return out;
    }

    Runner(strata::core::ModelArtifact &a, Comm &c, const std::string &mode, int cap, int layers)
        : artifact(a), m(a.descriptor()), comm(c), numerics(mode), w4a8(mode == "fast" || mode == "w4a8" || mode == "w4a8g16"),
          trunk_q8(mode == "fast" || mode == "q8trunk"), capacity(cap), layers_run(layers) {}

    // routed expert e is cut into `parts` row slices of whole 64-row units (64 keeps the down projection's column
    // slice on NVFP4 superblocks); slice p lives on owner(e, p). parts == ranks: every rank holds one slice of every
    // expert (balanced work, 2-3 units each at 12 ranks); fewer parts: whole slices on parts of the ranks.
    int owner(int expert, int part) const {
        return parts == comm.world ? (expert + part) % comm.world : (expert + part * (comm.world / parts)) % comm.world;
    }
    Span slice_rows(int intermediate, int part) const {
        const Span u = split(intermediate / 64, parts, part);
        return {u.begin * 64, u.end * 64};
    }

    const strata::core::LayerDescriptor &desc_of(int l) const {
        return l < int(m.layers.size()) ? m.layers[l] : m.draft_layers.at(size_t(l) - m.layers.size());
    }
    // one layer's share of this rank: the main layers, and (index = main layer count) the MTP block
    void load_layer(Loader &ld, int l, Layer &Y) {
        const int H = m.hidden, D = m.linear_dim, hb = heads.begin, he = heads.end;
        const std::string p = strata::host::blk_prefix(l);
        const auto &desc = desc_of(l);
        const bool draft = l >= int(m.layers.size());   // the MTP block: a plain residual, no mHC
        Y.kda = desc.mixer == strata::core::MixerKind::Kda;
        Y.dense = desc.ffn == strata::core::FfnKind::Dense;
        if (!draft) {
            Y.hc_attn = ld.whole(Loader::src(l, "hc_attn_fn"));
            Y.hc_ffn = ld.whole(Loader::src(l, "hc_ffn_fn"));
            Y.hc_attn_base = ld.f32(p + "hc_attn_base.weight"), Y.hc_attn_scale = ld.f32(p + "hc_attn_scale.weight");
            Y.hc_ffn_base = ld.f32(p + "hc_ffn_base.weight"), Y.hc_ffn_scale = ld.f32(p + "hc_ffn_scale.weight");
        }
        Y.attn_norm = ld.f32(p + "attn_norm.weight"), Y.ffn_norm = ld.f32(p + "ffn_norm.weight");
        if (Y.kda) {
            auto &K = Y.kd;
            const char *qkv[] = {"q", "k", "v"};
            for (int i = 0; i < 3; ++i) {
                auto &mq = i == 0 ? K.q : i == 1 ? K.k : K.v;
                mq = ld.rows(Loader::src(l, std::string("self_attn.") + qkv[i] + "_proj.weight"), hb * D, he * D);
                add_q8(ld, mq, [&] { return ld.rows(Loader::src(l, std::string("self_attn.") + qkv[i] + "_proj.weight"), hb * D, he * D); });
                const auto conv = ld.f32(p + "ssm_conv1d_" + qkv[i] + ".weight");
                K.conv[i].assign(conv.begin() + size_t(hb) * D * m.conv_kernel, conv.begin() + size_t(he) * D * m.conv_kernel);
                K.hist[i].assign(size_t(he - hb) * D * (m.conv_kernel - 1), 0.f);
            }
            K.f_a = ld.whole(Loader::src(l, "self_attn.f_a_proj.weight"));
            K.g_a = ld.whole(Loader::src(l, "self_attn.g_a_proj.weight"));
            K.f_b = ld.rows(Loader::src(l, "self_attn.f_b_proj.weight"), hb * D, he * D);
            K.g_b = ld.rows(Loader::src(l, "self_attn.g_b_proj.weight"), hb * D, he * D);
            K.beta = ld.whole(Loader::src(l, "self_attn.b_proj.weight"));
            const auto o = ld.load(Loader::src(l, "self_attn.o_proj.weight"));
            K.o = ld.slice(o, 0, int(o.rows()), hb * D, he * D, true);
            add_q8(ld, K.o, [&] { return ld.slice(o, 0, int(o.rows()), hb * D, he * D, true); });
            const auto dt = ld.f32(p + "ssm_dt.bias"), av = ld.f32(p + "ssm_a");
            K.dt.assign(dt.begin() + size_t(hb) * D, dt.begin() + size_t(he) * D);
            K.a_neg.assign(av.begin() + hb, av.begin() + he);
            if (std::all_of(av.begin(), av.end(), [](float v) { return v > 0; })) for (auto &v : K.a_neg) v = -v;
            K.onorm = ld.f32(p + "ssm_norm.weight");
            if (D % KDA_GROUP) throw std::runtime_error("runner: KDA head dimension must be a multiple of 16");
            K.state_n = size_t(he - hb) * D * D;
            K.pool[0] = K.state = reinterpret_cast<float *>(arena_alloc(K.state_n * 4));
            {   // zeroed by the threads that walk each 16-column block (kda_core_team's static split): placed on their CMG
                const size_t per = size_t(D) * KDA_GROUP, items = K.state_n / per;
#pragma omp parallel for schedule(static)
                for (size_t it = 0; it < items; ++it) std::memset(K.state + it * per, 0, per * 4);
            }
        } else {
            auto &M = Y.ml;
            const std::string front[6] = {"self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.indexer.weights_proj.weight",
                                          "self_attn.indexer.wk.weight", "self_attn.indexer.index_kpool_compress_gate", "self_attn.indexer.wq_b.weight"};
            Matrix *dst[6] = {&M.q_a, &M.kv_a, &M.iw, &M.ik, &M.igate, &M.iq};
            M.sharded = mla_shard && comm.world > 1;
            for (int i = 0; i < 6; ++i) {
                const std::string nm = Loader::src(l, front[i]);
                const int rows = int(ld.row_count(nm));
                M.full[i] = rows;
                if (!M.sharded) { *dst[i] = ld.whole(nm); continue; }
                const int unit = ld.opt.matrix(nm).format == ModelOptFormat::BlockFp8 ? 128 : 16;   // whole scale blocks
                static const int front_off[6] = {0, 8, 6, 4, 4, 4};
                const Span u = split_rot(rows / unit, comm.world, comm.rank, rot(front_off[i]));
                M.r0[i] = u.begin * unit;
                if (u.size()) *dst[i] = ld.rows(nm, u.begin * unit, u.end * unit);
            }
            const int hd = int(ld.row_count(Loader::src(l, "self_attn.q_b_proj.weight"))) / m.attention_heads;   // 256
            M.q_b = ld.rows(Loader::src(l, "self_attn.q_b_proj.weight"), hb * hd, he * hd);
            add_q8(ld, M.q_b, [&] { return ld.rows(Loader::src(l, "self_attn.q_b_proj.weight"), hb * hd, he * hd); });
            const auto o = ld.load(Loader::src(l, "self_attn.o_proj.weight"));
            M.o = ld.slice(o, 0, int(o.rows()), hb * hd, he * hd, true);
            add_q8(ld, M.o, [&] { return ld.slice(o, 0, int(o.rows()), hb * hd, he * hd, true); });
            // kv_b [heads * (256 K + 256 V)][512]: K transposed per head (latent-major) for the absorbed query
            const int per = int(ld.row_count(Loader::src(l, "self_attn.kv_b_proj.weight"))) / m.attention_heads, half = per / 2;
            auto kvb = ld.load(Loader::src(l, "self_attn.kv_b_proj.weight"), hb * per, he * per);
            if (kvb.p.format == ModelOptFormat::Mxfp8) {   // the MTP block's kv_b: to BF16, exactly (an E4M3 value
                Loader::Source b;                            // times a power of two has at most four significant bits)
                b.p = ModelOptPacked{ModelOptFormat::Bf16, kvb.p.rows, kvb.p.columns, 1, {}, {}};
                b.r0 = kvb.r0, b.r1 = kvb.r1, b.pitch = kvb.p.columns * 2;
                if (!ld.replaying()) {
                    b.bytes.resize(size_t(b.r1 - b.r0) * b.pitch);
                    for (int r = b.r0; r < b.r1; ++r)
                        for (size_t c = 0; c < kvb.p.columns; ++c) {
                            const float f = Loader::value(kvb, r, int(c));
                            uint32_t u;
                            std::memcpy(&u, &f, 4);
                            if (u & 0xFFFF) throw std::runtime_error("runner: MXFP8 kv_b value not exact in BF16");
                            const uint16_t h = uint16_t(u >> 16);
                            std::memcpy(&b.bytes[size_t(r - b.r0) * b.pitch + c * 2], &h, 2);
                        }
                }
                kvb = std::move(b);
            }
            if (kvb.p.format != ModelOptFormat::Bf16) throw std::runtime_error("runner: kv_b is expected in BF16 or MXFP8");
            const int lat = int(kvb.columns());
            for (int h = hb; h < he; ++h) {
                Loader::Source kt;
                kt.p = ModelOptPacked{ModelOptFormat::Bf16, size_t(lat), size_t(half), 1, {}, {}};
                kt.r0 = 0, kt.r1 = lat, kt.pitch = size_t(half) * 2;
                if (!ld.replaying()) {
                    kt.bytes.resize(size_t(lat) * half * 2);
                    for (int r = 0; r < half; ++r)
                        for (int c = 0; c < lat; ++c) std::memcpy(&kt.bytes[(size_t(c) * half + r) * 2], kvb.row(h * per + r) + size_t(c) * 2, 2);
                }
                M.kb.push_back(ld.slice(kt, 0, lat, 0, half, false));
                M.vb.push_back(ld.slice(kvb, h * per + half, h * per + per, 0, lat, false));
            }
            M.q_a_norm = ld.f32(p + "attn_q_a_norm.weight"), M.kv_a_norm = ld.f32(p + "attn_kv_a_norm.weight");
            M.k_norm_w = ld.f32(p + "indexer.k_norm.weight"), M.k_norm_b = ld.f32(p + "indexer.k_norm.bias");
            M.ape = ld.f32(p + "indexer_compressor_ape.weight");
            M.keys.assign(size_t(m.index_pool) * m.index_dim, 0.f);
            M.gates.assign(size_t(m.index_pool) * m.index_dim, 0.f);
            M.pooled.assign(size_t(capacity / m.index_pool + 1) * m.index_dim, 0.f);
            if (cache_hot > 0 && capacity > cache_hot && capacity >= cache_hybrid_min) {
                M.ring = cache_hot;
                M.cache.assign(size_t(cache_hot) * m.kv_rank, 0.f);
                M.cold.assign(size_t((capacity + COLD_BLOCK - 1) / COLD_BLOCK / comm.world + 1) * COLD_BLOCK * m.kv_rank, 0.f);
            } else M.cache.assign(size_t(capacity) * m.kv_rank, 0.f);
        }
        if (Y.dense) {
            const Span f = split(int(ld.row_count(Loader::src(l, "mlp.gate_proj.weight"))) / 16, comm.world, comm.rank);
            Y.ffn.gate = ld.rows(Loader::src(l, "mlp.gate_proj.weight"), f.begin * 16, f.end * 16);
            add_q8(ld, Y.ffn.gate, [&] { return ld.rows(Loader::src(l, "mlp.gate_proj.weight"), f.begin * 16, f.end * 16); });
            Y.ffn.up = ld.rows(Loader::src(l, "mlp.up_proj.weight"), f.begin * 16, f.end * 16);
            add_q8(ld, Y.ffn.up, [&] { return ld.rows(Loader::src(l, "mlp.up_proj.weight"), f.begin * 16, f.end * 16); });
            const auto d = ld.load(Loader::src(l, "mlp.down_proj.weight"));
            Y.ffn.down = ld.slice(d, 0, H, f.begin * 16, f.end * 16, true);
            add_q8(ld, Y.ffn.down, [&] { return ld.slice(d, 0, H, f.begin * 16, f.end * 16, true); });
        } else {
            auto &E = Y.moe;
            if (router_shard && comm.world > 1) {   // 16-row units, the full logits by one all-reduce of zero-padded rows
                const std::string nm = Loader::src(l, "mlp.gate.weight");
                const Span u = split(int(ld.row_count(nm)) / 16, comm.world, comm.rank);
                E.router_r0 = u.begin * 16;
                ld.whole_k = true;
                E.router = ld.rows(nm, u.begin * 16, u.end * 16);
                ld.whole_k = false;
            } else
                E.router = ld.whole(Loader::src(l, "mlp.gate.weight"));
            E.bias = ld.f32(p + "exp_probs_b.bias");
            // 128-row units (block-FP8 scale blocks)
            const Span s = split_rot(int(ld.row_count(Loader::src(l, "mlp.shared_experts.gate_proj.weight"))) / 128, comm.world, comm.rank, rot(4));
            if (s.size()) {
                // STRATA_RUNNER_ROT_SPREAD (default 1): the shared expert's gate and up rotated so their tiles' threads
                // overlap the router's (and each other's) least: the three share one region, and with fewer tiles than
                // threads the counter's consecutive rotations stacked tiles of all three on the same threads
                static const bool spread = !(std::getenv("STRATA_RUNNER_ROT_SPREAD") && std::string(std::getenv("STRATA_RUNNER_ROT_SPREAD")) == "0");
                const int team = omp_get_max_threads(), tiles_s = (s.end - s.begin) * 128 / 16;
                std::vector<int> load(team, 0);
                auto owners = [&](int tiles, int rot_, std::vector<int> &add) {
                    for (int t = 0; t < team; ++t) {
                        const int v = (t + rot_) % team;
                        add[t] += int(int64_t(tiles) * (v + 1) / team) - int(int64_t(tiles) * v / team);
                    }
                };
                auto best_rot = [&]() {   // the rotation with the smallest maximum tiles per thread, then the fewest shared
                    int best = 0;
                    long best_key = LONG_MAX;
                    for (int r = 0; r < team; ++r) {
                        std::vector<int> add(load);
                        owners(tiles_s, r, add);
                        long mx = 0, shared = 0;
                        for (int t = 0; t < team; ++t) mx = std::max<long>(mx, add[t]), shared += add[t] > 1;
                        if (mx * 1000 + shared < best_key) best_key = mx * 1000 + shared, best = r;
                    }
                    return best;
                };
                if (spread) {
                    owners(E.router.padded / 16, E.router.rot, load);
                    const int rg = best_rot();
                    owners(tiles_s, rg, load);
                    ld.rot_next = rg;
                }
                E.sgate = ld.rows(Loader::src(l, "mlp.shared_experts.gate_proj.weight"), s.begin * 128, s.end * 128);
                add_q8(ld, E.sgate, [&] { return ld.rows(Loader::src(l, "mlp.shared_experts.gate_proj.weight"), s.begin * 128, s.end * 128); });
                if (spread) {
                    const int ru = best_rot();
                    owners(tiles_s, ru, load);
                    ld.rot_next = ru;
                }
                E.sup = ld.rows(Loader::src(l, "mlp.shared_experts.up_proj.weight"), s.begin * 128, s.end * 128);
                add_q8(ld, E.sup, [&] { return ld.rows(Loader::src(l, "mlp.shared_experts.up_proj.weight"), s.begin * 128, s.end * 128); });
                const auto sd = ld.load(Loader::src(l, "mlp.shared_experts.down_proj.weight"));
                E.sdown = ld.slice(sd, 0, H, s.begin * 128, s.end * 128, true);
                add_q8(ld, E.sdown, [&] { return ld.slice(sd, 0, H, s.begin * 128, s.end * 128, true); });
            }
            E.owner_index.assign(m.experts, -1);
            const int inter = desc.intermediate;
            if (cooperative && !ld.replaying()) load_experts_cooperative(ld, l, E);
            else
                for (int e = 0; e < m.experts; ++e)
                    for (int part = 0; part < parts; ++part) {
                        if (owner(e, part) != comm.rank) continue;
                        const std::string x = "mlp.experts." + std::to_string(e) + ".";
                        const Span rr = slice_rows(inter, part);
                        ExpertSlice es;
                        es.expert = e, es.part = part;
                        const bool keep = ld.w4a8;
                        ld.w4a8 = keep && w4a8_gate_up();
                        es.gate = ld.rows(Loader::src(l, x + "gate_proj.weight"), rr.begin, rr.end, false);
                        es.up = ld.rows(Loader::src(l, x + "up_proj.weight"), rr.begin, rr.end, false);
                        ld.w4a8 = keep && w4a8_down();
                        const auto d = ld.load(Loader::src(l, x + "down_proj.weight"));
                        es.down = ld.slice(d, 0, H, rr.begin, rr.end, false);
                        ld.w4a8 = keep;
                        E.owner_index[e] = int(E.slices.size());
                        E.slices.push_back(std::move(es));
                    }
        }
    }

    bool cooperative = false;   // STRATA_RUNNER_COOP_LOAD (default on with MPI): readers share the checkpoint reads

#ifdef STRATA_CPU_RUNNER_MPI
    // One MoE layer's routed experts, read once: rank e % world reads expert e's three matrices whole, packs every
    // owner's slices and sends them (MPI_Alltoallv); owners place and record them in (expert, slice) order, the order
    // the per-rank load and an image replay use. Each rank reads 1/world of the experts as whole tensors instead of
    // a down projection per owned slice.
    void load_experts_cooperative(Loader &ld, int l, MoeLayer &E) {
        const int W = comm.world, H = m.hidden, inter = desc_of(l).intermediate;
        std::vector<std::vector<uint8_t>> out(W);
        auto put = [](std::vector<uint8_t> &b, const void *p, size_t n) {
            const auto *c = static_cast<const uint8_t *>(p);
            b.insert(b.end(), c, c + n);
        };
        for (int e = comm.rank; e < m.experts; e += W) {
            const std::string x = "mlp.experts." + std::to_string(e) + ".";
            const auto g = ld.load(Loader::src(l, x + "gate_proj.weight")), u = ld.load(Loader::src(l, x + "up_proj.weight")),
                       d = ld.load(Loader::src(l, x + "down_proj.weight"));
            for (int part = 0; part < parts; ++part) {
                const int o = owner(e, part);
                const Span rr = slice_rows(inter, part);
                const bool keep = ld.w4a8;
                ld.w4a8 = keep && w4a8_gate_up();
                Loader::Packed pk[3] = {ld.pack(g, rr.begin, rr.end, 0, H, false), ld.pack(u, rr.begin, rr.end, 0, H, false), {}};
                ld.w4a8 = keep && w4a8_down();
                pk[2] = ld.pack(d, 0, H, rr.begin, rr.end, false);
                ld.w4a8 = keep;
                const int32_t head[2] = {e, part};
                put(out[o], head, sizeof head);
                for (auto &q : pk) {
                    const int32_t h[4] = {int32_t(q.m.kind), q.m.rows, q.m.padded, q.m.cols};
                    const uint64_t n = q.buf.size();
                    put(out[o], h, sizeof h), put(out[o], &q.m.global, 4), put(out[o], &n, 8), put(out[o], q.buf.data(), n);
                }
            }
        }
        std::vector<int> sc(W), rc(W), sd(W), rd(W);
        for (int r = 0; r < W; ++r) {
            if (out[r].size() > size_t(1) << 30) throw std::runtime_error("runner: cooperative load message too large");
            sc[r] = int(out[r].size());
        }
        MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::vector<uint8_t> send, recv;
        for (int r = 0; r < W; ++r) sd[r] = int(send.size()), send.insert(send.end(), out[r].begin(), out[r].end());
        int64_t total_in = 0;
        for (int r = 0; r < W; ++r) rd[r] = int(total_in), total_in += rc[r];
        if (total_in > (int64_t(1) << 31) - 1) throw std::runtime_error("runner: cooperative load receive too large");
        recv.resize(size_t(total_in));
        MPI_Alltoallv(send.data(), sc.data(), sd.data(), MPI_BYTE, recv.data(), rc.data(), rd.data(), MPI_BYTE, MPI_COMM_WORLD);
        struct Item { int e, part; Loader::Packed pk[3]; };
        std::vector<Item> items;
        size_t at = 0;
        auto get = [&](void *p, size_t n) {
            if (at + n > recv.size()) throw std::runtime_error("runner: truncated cooperative load message");
            std::memcpy(p, recv.data() + at, n);
            at += n;
        };
        while (at < recv.size()) {
            Item it;
            int32_t head[2];
            get(head, sizeof head);
            it.e = head[0], it.part = head[1];
            for (auto &q : it.pk) {
                int32_t h[4];
                uint64_t n;
                get(h, sizeof h), get(&q.m.global, 4), get(&n, 8);
                q.m.kind = Kind(h[0]), q.m.rows = h[1], q.m.padded = h[2], q.m.cols = h[3];
                q.buf.resize(n);
                get(q.buf.data(), n);
            }
            items.push_back(std::move(it));
        }
        std::sort(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.e != b.e ? a.e < b.e : a.part < b.part; });
        for (auto &it : items) {
            ExpertSlice es;
            es.expert = it.e, es.part = it.part;
            es.gate = ld.finish(std::move(it.pk[0]));
            es.up = ld.finish(std::move(it.pk[1]));
            es.down = ld.finish(std::move(it.pk[2]));
            E.owner_index[it.e] = int(E.slices.size());
            E.slices.push_back(std::move(es));
        }
    }
#else
    void load_experts_cooperative(Loader &, int, MoeLayer &) { throw std::runtime_error("runner: cooperative load needs MPI"); }
#endif

    void load() {
        const double t0 = now();
        {
            const char *c = std::getenv("STRATA_RUNNER_COOP_LOAD");
#ifdef STRATA_CPU_RUNNER_MPI
            cooperative = comm.world > 1 && !(c && std::string(c) == "0");
#else
            (void)c;
#endif
        }
        Loader ld{artifact, *artifact.exl3().modelopt, artifact.exl3().tensors, w4a8, trunk_q8};
        std::ifstream image_in;
        std::ofstream image_out;
        std::filesystem::path image, partial;
        const uint64_t magic = 0x31474d494c464721ull;   // "!GFLIMG1"
        // numerics code: 1 exact, 0 fast (the codes of the first images), 2 w4a8, 3 q8trunk
        const int32_t code = !w4a8 && !trunk_q8 ? 1 : w4a8 && trunk_q8 ? 0 : w4a8 ? 2 : 3;
        // (parts before the header: it was stored unset, so images were rejected whenever that garbage changed)
        parts = std::min(4, comm.world);
        if (const char *e = std::getenv("STRATA_RUNNER_EXPERT_PARTS")) parts = std::max(1, std::min(comm.world, std::atoi(e)));
        if (comm.world % parts) throw std::runtime_error("runner: rank count must be a multiple of the expert slice count");
        const int32_t layout[6] = {comm.world, comm.rank, layers_run, code, parts, want_mtp ? 4 : 3};   // last: layout version (4: + MTP)
        // (the per-matrix rotation is not stored: it follows the load order, which a replay repeats)
        // page cache of images (ours and others) holds HBM: with zone_reclaim off, weight pages whose CMG is full of
        // cached file pages land on another CMG (STRATA_RUNNER_IMAGE_KEEPCACHE=1 keeps the cache)
        const bool drop_cache = !std::getenv("STRATA_RUNNER_IMAGE_KEEPCACHE");
        auto drop_file = [](const std::filesystem::path &f) {
            const int fd = open(f.c_str(), O_RDONLY);
            if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED), close(fd);
        };
        if (!image_dir.empty()) {
            std::filesystem::create_directories(image_dir);
            if (drop_cache)
                for (const auto &e : std::filesystem::directory_iterator(image_dir))
                    if (e.is_regular_file()) drop_file(e.path());
            image = std::filesystem::path(image_dir) / ("glm-fp4l-" + numerics + (w4a8_only().empty() ? "" : "-" + w4a8_only()) + (want_mtp ? "-mtp" : "") + (mla_shard ? router_shard ? "-ms2" : "-ms" : "") + (rem_spread ? "-rs" : "") + (prefill_q8 && !trunk_q8 ? "-pq8" : "") + "-r" + std::to_string(comm.rank) +
                                                        "of" + std::to_string(comm.world) + "-L" + std::to_string(layers_run) + ".img");
            partial = image.string() + ".partial";
            image_in.open(image, std::ios::binary);
            bool ok = false;
            std::string why = "missing";
            if (image_in) {
                uint64_t mg = 0;
                int32_t lay[6] = {0, 0, 0, 0, 0, 0};
                ok = image_in.read(reinterpret_cast<char *>(&mg), 8) && image_in.read(reinterpret_cast<char *>(lay), sizeof lay) &&
                     mg == magic && std::equal(lay, lay + 6, layout);
                if (!ok) {
                    why = "header";
                    for (int i = 0; i < 6; ++i) why += " " + std::to_string(lay[i]) + (lay[i] == layout[i] ? "" : "!=" + std::to_string(layout[i]));
                    image_in.close();
                }
            }
            if (!ok) std::cerr << "RUNNER image rank " << comm.rank << " not replayable (" << why << "): " << image.string() << '\n';
            // every rank replays or every rank loads: a load reads experts cooperatively (collectives a replaying
            // rank never joins; a run whose image write failed on some ranks left the others' images complete)
            if (comm.max_over_ranks(ok ? 0 : 1) > 0 && ok) ok = false, image_in.close();
            if (ok) {
                ld.in = &image_in;
                if (drop_cache) ld.in_fd = open(image.c_str(), O_RDONLY);
            }
            else {
                image_out.open(partial, std::ios::binary | std::ios::trunc);
                image_out.write(reinterpret_cast<const char *>(&magic), 8);
                image_out.write(reinterpret_cast<const char *>(layout), sizeof layout);
                ld.out = &image_out;
            }
        }
        heads = split(m.attention_heads, comm.world, comm.rank);
        L.resize(layers_run);
        for (int l = 0; l < layers_run; ++l) {
            if (std::getenv("STRATA_RUNNER_LOAD_TRACE")) std::cerr << "RUNNER_LOAD_TRACE rank " << comm.rank << " layer " << l << " start " << now() - t0 << " s\n";
            load_layer(ld, l, L[l]);
            if (comm.rank == 0 && (l % 5 == 0 || l + 1 == layers_run))
                std::cerr << "RUNNER load layer " << l << " rank0 MiB " << ld.total / 1048576.0 << " " << now() - t0 << " s\n";
        }
        vocab = split_rot(int(ld.row_count("lm_head.weight")) / 16, comm.world, comm.rank, rot(4));
        vocab = {vocab.begin * 16, vocab.end * 16};
        head = ld.rows("lm_head.weight", vocab.begin, vocab.end);
        out_norm = ld.f32("output_norm.weight");
        {
            const auto &e = artifact.exl3().tensors.at("model.language_model.embed_tokens.weight");
            if (e.dtype != "BF16" || e.bytes != uint64_t(m.vocab) * m.hidden * 2) throw std::runtime_error("runner: unexpected embedding");
            embedding.resize(e.bytes / 2);
            if (ld.in) ld.get(embedding.data(), embedding.size());
            else {
                artifact.exl3().tensors.read(e, 0, embedding.data(), e.bytes);
                ld.put(embedding.data(), embedding.size());
            }
        }
        if (want_mtp) {
            const int lm = int(m.layers.size());
            if (m.draft_layers.empty()) throw std::runtime_error("runner: --mtp needs the checkpoint's draft block");
            load_layer(ld, lm, mtp.layer);
            const std::string p = strata::host::blk_prefix(lm);
            mtp.enorm = ld.f32(p + "nextn.enorm.weight"), mtp.hnorm = ld.f32(p + "nextn.hnorm.weight");
            mtp.head_norm = ld.f32(p + "nextn.shared_head_norm.weight");
            const Span t = split_rot(m.hidden / 16, comm.world, comm.rank, rot(8));
            mtp.eh_rows = {t.begin * 16, t.end * 16};
            mtp.eh = ld.rows(Loader::src(lm, "eh_proj.weight"), mtp.eh_rows.begin, mtp.eh_rows.end);
            mtp.on = true;
            if (comm.rank == 0) std::cerr << "RUNNER loaded the MTP block, " << mtp.layer.moe.slices.size() << " expert slices\n";
        }
        if (ld.out) {
            image_out.close();
            if (!image_out) throw std::runtime_error("runner: image write failed");
            std::filesystem::rename(partial, image);
            if (drop_cache) {   // written back first: DONTNEED leaves dirty pages
                const int fd = open(image.c_str(), O_RDONLY);
                if (fd >= 0) fdatasync(fd), posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED), close(fd);
            }
        }
        if (ld.in_fd >= 0) posix_fadvise(ld.in_fd, 0, 0, POSIX_FADV_DONTNEED), close(ld.in_fd), ld.in_fd = -1;
        if (std::getenv("STRATA_RUNNER_NUMA_REPORT")) {   // this process's resident pages per NUMA node (numa_maps)
            std::ifstream nm("/proc/self/numa_maps");
            std::string line;
            std::map<int, double> pages;
            double kb_page = 64;
            while (std::getline(nm, line)) {
                std::istringstream ls(line);
                std::string w;
                double kp = 4;
                std::vector<std::pair<int, double>> here;
                while (ls >> w) {
                    if (w.rfind("kernelpagesize_kB=", 0) == 0) kp = std::atof(w.c_str() + 18);
                    else if (w.size() > 2 && w[0] == 'N' && std::isdigit(w[1]) && w.find('=') != std::string::npos)
                        here.push_back({std::atoi(w.c_str() + 1), std::atof(w.c_str() + w.find('=') + 1)});
                }
                for (auto [nd, c] : here) pages[nd] += c * kp / 1048576.0;   // GiB
                kb_page = kp;
            }
            std::ostringstream o;
            char host[256] = {0};
            gethostname(host, sizeof host - 1);
            o << "RUNNER_NUMA rank=" << comm.rank << " host=" << host << " GiB";
            for (auto [nd, g] : pages) o << " N" << nd << "=" << g;
            std::cerr << o.str() << '\n';
            (void)kb_page;
        }
        const std::string mhc_copies = std::getenv("STRATA_RUNNER_MHC_REPLICATE") ? std::getenv("STRATA_RUNNER_MHC_REPLICATE") : "1";
        if (mhc_copies != "0") {   // default one per CMG (21.41 vs 21.67 ms per token; 360 MiB per rank)
            // one copy of each mHC projection per CMG, first touched by that CMG's threads (a 64 KB page of the shared
            // copy holds chunks of two CMGs, so half of every thread's blocks were remote whatever place() did)
            const int team = omp_get_max_threads(), cg = cmg_threads(), groups = (team + cg - 1) / cg;
            const bool shared = mhc_copies == "2";
            auto replicate = [&](Matrix &fm) {
                if (!mhc_layout(fm)) return;
                const int tiles = fm.padded / 16, nc = fm.cols / 4 / MHC_CHUNK;
                const size_t bb = block_bytes(fm.kind), tile_bytes = fm.bytes() / size_t(tiles);
                fm.cmg_copy.resize(shared ? 1 : groups);
                for (auto &c : fm.cmg_copy) c.n = fm.data.size(), c.p = arena_alloc(c.n);
                std::vector<int> owner(nc);   // site()'s thread for each chunk
                for (int u = 0; u < team; ++u)
                    for (int ch = mhc_block_begin(nc, u, team); ch < mhc_block_begin(nc, u + 1, team); ++ch) owner[ch] = u;
                team_run([&] {   // every copy written (first touched) by its CMG's threads: own chunks by their owner
                    const int th = omp_get_thread_num(), g = shared ? 0 : th / cg, r = th % cg, gn = std::min(cg, team - (th / cg) * cg);
                    for (int ch = 0; ch < nc; ++ch) {
                        const bool mine = owner[ch] == th;
                        const bool foreign = !shared && owner[ch] / cg != g && ch % gn == r;
                        if (!mine && !foreign) continue;
                        for (int j = 0; j < 4; ++j)
                            for (int tile = 0; tile < tiles; ++tile)
                                std::memcpy(fm.cmg_copy[g].p + (size_t(ch * 4 + j) * tiles + tile) * bb,
                                            fm.data.data() + size_t(tile) * tile_bytes + size_t(j * nc + ch) * bb, bb);
                    }
                });
            };
            size_t extra = 0;
            for (auto &Y : L) replicate(Y.hc_attn), replicate(Y.hc_ffn), extra += (Y.hc_attn.cmg_copy.size() * Y.hc_attn.bytes() + Y.hc_ffn.cmg_copy.size() * Y.hc_ffn.bytes());
            if (comm.rank == 0) std::cerr << "RUNNER mHC projections replicated per CMG: " << groups << " copies, " << extra / 1048576.0 << " MiB\n";
        }
        if (want_mtp && head.kind == Kind::Mx && !(std::getenv("STRATA_RUNNER_DRAFT_HEAD") && std::string(std::getenv("STRATA_RUNNER_DRAFT_HEAD")) == "exact")) {
            // drafts only propose tokens (every kept token is the main model's exact pick), so their head may be
            // approximate: the MXFP8 rows decoded and requantized to int8 per (row, 32), the trunk's fast-mode format
            const int R = head.padded, C = head.cols, nb = C / 32;
            std::vector<float> w(size_t(R) * C);
            team_run([&] {
#pragma omp for schedule(static) nowait
                for (int t = 0; t < R / 16; ++t)
                    for (int b = 0; b < nb; ++b) {
                        const uint8_t *o = head.data.data() + (size_t(t) * nb + b) * block_bytes(Kind::Mx);
                        for (int r = 0; r < 16; ++r) {
                            const float sc = gf4_ue8m0(o[512 + r]) * head.global;
                            for (int k = 0; k < 32; ++k) w[size_t(16 * t + r) * C + 32 * b + k] = gf4_e4m3(o[16 * k + r]) * sc;
                        }
                    }
            });
            std::vector<uint8_t> buf(gf4_q8k_packed_bytes(R, C));
            gf4_pack_q8k(w.data(), R, C, buf.data());
            draft_head.kind = Kind::Q8, draft_head.rows = head.rows, draft_head.padded = R, draft_head.cols = C, draft_head.global = 1;
            draft_head.rot = next_rotation();
            draft_head.data = place(buf, draft_head);
            if (comm.rank == 0) std::cerr << "RUNNER draft head: int8 copy, " << buf.size() / 1048576.0 << " MiB per rank\n";
        }
        const double mib = comm.max_over_ranks(ld.total / 1048576.0);
        if (comm.rank == 0)
            std::cerr << "RUNNER loaded" << (ld.in ? " (image)" : ld.out ? " (image written)" : "") << " " << layers_run << " layers, ranks " << comm.world << ", heads per rank " << heads.size()
                      << ", expert slices " << parts << ", max MiB per rank " << mib << ", " << now() - t0 << " s\n";
        memory_budget(mib);
    }

    // the per-context buffers (MLA latent caches, index keys and pools, KDA states and histories) and the node's
    // available memory after loading, max / min over ranks; a warning below 2 GiB of headroom
    void memory_budget(double weights_mib) {
        double ctx = 0;
        auto add = [&](const MlaLayer &M) { ctx += double(M.cache.size() + M.cold.size() + M.pooled.size() + M.keys.size() + M.gates.size()) * 4; };
        for (const auto &Y : L) {
            if (Y.kda) ctx += double(Y.kd.state_n) * 4 + double(Y.kd.hist[0].size() * 3) * 4;
            else add(Y.ml);
        }
        if (mtp.on) add(mtp.layer.ml);
        double avail = 0;
        {
            std::ifstream mi("/proc/meminfo");
            std::string k;
            double v;
            std::string unit;
            while (mi >> k >> v >> unit)
                if (k == "MemAvailable:") { avail = v * 1024; break; }
        }
        const double ctx_mib = comm.max_over_ranks(ctx / 1048576.0), avail_mib = -comm.max_over_ranks(-avail / 1048576.0);
        if (comm.rank == 0) {
            std::cerr << "RUNNER memory per rank: weights " << weights_mib << " MiB, context buffers " << ctx_mib << " MiB (capacity " << capacity
                      << " positions, " << ctx_mib * 1024 / std::max(1, capacity) << " KiB per position), available after load " << avail_mib << " MiB\n";
            if (avail_mib < 2048) std::cerr << "RUNNER WARNING: under 2 GiB of memory left after load\n";
        }
    }

    // the all-reduce on thread 0 while the other threads touch their share of the next region's weights: opt-in
    // (STRATA_RUNNER_COMM_TOUCH=1); measured neutral with dist barriers (28.8 vs 29.0 ms per token) and 1.7 ms slower
    // with the default ones
    void allreduce_touch(Vec &y, const std::vector<const Matrix *> &next) {
        static const bool on = std::getenv("STRATA_RUNNER_COMM_TOUCH") && std::string(std::getenv("STRATA_RUNNER_COMM_TOUCH")) == "1";
        if (!on || comm.world == 1) { comm.allreduce(y.data(), m.hidden); return; }
        team_run([&] {
            const int t = omp_get_thread_num(), n = omp_get_num_threads();
            if (t == 0) comm.allreduce(y.data(), m.hidden);
            else touch_share(next, t, n);
        });
    }
    // the first region after the attention all-reduce of layer l (the FFN mHC site and the dense FFN or the router +
    // shared expert), and after its FFN all-reduce (the next layer's mHC site and first projections, or the head)
    std::vector<const Matrix *> after_attn(int l) const {
        const auto &Y = L[l];
        if (Y.dense) return {&Y.hc_ffn, &Y.ffn.gate, &Y.ffn.up};
        return {&Y.hc_ffn, &Y.moe.router, &Y.moe.sgate, &Y.moe.sup};
    }
    std::vector<const Matrix *> after_ffn(int l) const {
        if (l + 1 >= layers_run) return {&head};
        const auto &Y = L[l + 1];
        if (Y.kda) return {&Y.hc_attn, &Y.kd.q, &Y.kd.k, &Y.kd.v, &Y.kd.f_a, &Y.kd.g_a, &Y.kd.beta};
        return {&Y.hc_attn, &Y.ml.q_a, &Y.ml.kv_a, &Y.ml.iw, &Y.ml.ik, &Y.ml.igate};
    }

    // sub-phase timers: adds the time since `t` to timers[what] and restarts `t`
    static inline thread_local bool lap_off = false;   // inside a parallel region: no timers (the map is shared)
    void lap(double &t, const char *what) {
        const double n = now();
        if (!lap_off) timers[what] += n - t;
        t = n;
    }

    // ---------------------------------------------------------------------------------------------- forward
    // sum of squares of n values in eight chains (one serial double chain is latency-bound on A64FX)
    static double sumsq(const float *x, size_t n) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
        // the loop below in 8 double lanes (lane j: the fused x^2 + acc_j over i, as clang contracts it) and FADDV's
        // reduction tree, which is the loop's ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)): the same bits (checked
        // on A64FX over 6K blocks); clang compiled the loop to 128-bit NEON, ~56 cycles per 32 values
        if (n % 8 == 0) {
            typedef svfloat64_t vd __attribute__((arm_sve_vector_bits(512)));
            const svbool_t pd = svptrue_b64();
            vd acc = svdup_n_f64(0);
            for (size_t i = 0; i < n; i += 8) {
                const vd v = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(x + i))));
                acc = svmla_f64_x(pd, acc, v, v);
            }
            return svaddv_f64(pd, acc);
        }
#endif
        double acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const size_t n8 = n / 8 * 8;
        for (size_t i = 0; i < n8; i += 8)
            for (int j = 0; j < 8; ++j) acc[j] += double(x[i + j]) * x[i + j];
        double s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (size_t i = n8; i < n; ++i) s += double(x[i]) * x[i];
        return s;
    }
    Vec norm(const Vec &x, const Vec &w) const {
        // sum of squares in fixed 1024-value blocks, blocks over the team for the 4 x hidden mHC input, partials added
        // in block order
        const size_t n = x.size(), nb = (n + 1023) / 1024;
        std::vector<double> part(nb);
        if (!(n >= 8192))
            for (size_t b = 0; b < nb; ++b) part[b] = sumsq(x.data() + b * 1024, std::min(n, b * 1024 + 1024) - b * 1024);
        else
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (size_t b = 0; b < nb; ++b) part[b] = sumsq(x.data() + b * 1024, std::min(n, b * 1024 + 1024) - b * 1024);
        });
        double s = 0;
        for (double v : part) s += v;
        const float inv = float(1 / std::sqrt(s / n + m.rms_epsilon));
        Vec y(n);
        if (!(n >= 8192))
            for (size_t i = 0; i < n; ++i) y[i] = x[i] * inv * (w.empty() ? 1 : w[i]);
        else
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; ++i) y[i] = x[i] * inv * (w.empty() ? 1 : w[i]);
        });
        return y;
    }
    std::pair<Vec, Vec> read(const Matrix &fn, const Vec &base, const Vec &scale, const Vec &r) {
        double t = now(), tl = t;
        const auto nr = norm(r, {});
        lap(tl, "mhc.norm");
        auto projected = gemv(fn, nr);
        timers["mhc_gemv"] += now() - t;
        t = now();
        Vec c(24), x(m.hidden, 0.f);
        mhc_coef(projected.data(), base.data(), scale.data(), c.data(), m.sinkhorn_iterations, m.hc_epsilon);
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (int d = 0; d < m.hidden; ++d)
                for (int i = 0; i < 4; ++i) x[d] += c[i] * r[i * m.hidden + d];
        });
        timers["mhc"] += now() - t;
        return {x, c};
    }
    // one mHC read site and the following RMS norm in one team region (read() + norm() took four regions and serial
    // steps, ~80 us per site): the sum of squares of r and fn . r per 1024-column block (projected = inv * the block
    // partials summed in block order, where read() projected the normalized r), the coefficients on one thread, then
    // x = sum_i c[i] r_i per 256-value block with its sum of squares, then x * inv(x) * w. Deterministic, so every rank
    // gets the same bits; STRATA_RUNNER_MHC_FUSED=0 keeps read() + norm().
    std::pair<Vec, Vec> read_norm(const Matrix &fn, const Vec &base, const Vec &scale, const Vec &r, const Vec &w) {
        static const bool fused = !(std::getenv("STRATA_RUNNER_MHC_FUSED") && std::string(std::getenv("STRATA_RUNNER_MHC_FUSED")) == "0");
        const int H = m.hidden, n = int(r.size()), KB = 1024, nkb = n / KB, XB = 256;
        if (!fused || fn.kind != Kind::B16 || n % KB || H % XB || fn.cols != n || fn.rows > 32 || fn.cols % 32) {
            auto [x, c] = read(fn, base, scale, r);
            return {norm(x, w), c};
        }
        const double t = now();
        const int tiles = fn.padded / 16, team = omp_get_max_threads();
        const size_t tile_bytes = fn.bytes() / size_t(tiles);
        // the projection in fn's own K split, so each thread reads the weights place() put on its CMG
        const KSplit ks = ksplit(fn, team);
        const int chunks = ks.chunks ? ks.chunks : 1, width = ks.chunks ? ks.width : fn.cols, items = tiles * chunks;
        std::vector<float> part(size_t(chunks) * fn.padded);
        std::vector<double> ss_r(nkb), ss_x(H / XB);
        Vec c(24), x(H), y(H);
        float inv_x = 0;
        double tp[5] = {0, 0, 0, 0, 0};   // thread 0's clock after each step (sub-phase timers)
        team_run([&] {
            if (omp_get_thread_num() == 0) tp[0] = now();
#pragma omp for schedule(static) nowait
            for (int b = 0; b < nkb; ++b) ss_r[b] = sumsq(r.data() + size_t(b) * KB, KB);
            {
                const int th = omp_get_thread_num(), nth = omp_get_num_threads();
                const int i0 = int(int64_t(items) * th / nth), i1 = int(int64_t(items) * (th + 1) / nth);
                for (int it = i0; it < i1; ++it) {
                    const int tile = it / chunks, c = it % chunks, k0 = c * width, k1 = std::min(fn.cols, k0 + width);
                    const uint8_t *w0 = fn.data.data() + size_t(tile) * tile_bytes + size_t(k0 / block_cols(Kind::B16)) * block_bytes(Kind::B16);
                    const float *xs[1] = {r.data() + k0};
                    gf4_bf16k(w0, k1 - k0, 0, 1, xs, 1, part.data() + size_t(c) * fn.padded + 16 * tile, fn.padded);
                }
            }
#pragma omp barrier
            if (omp_get_thread_num() == 0) tp[1] = now();
#pragma omp single
            {
                double s = 0;
                for (double v : ss_r) s += v;
                const float inv = float(1 / std::sqrt(s / n + m.rms_epsilon));
                float projected[32];
                for (int row = 0; row < fn.rows; ++row) {
                    float v = 0;
                    for (int c = 0; c < chunks; ++c) v += part[size_t(c) * fn.padded + row];
                    projected[row] = v * inv;
                }
                mhc_coef(projected, base.data(), scale.data(), c.data(), m.sinkhorn_iterations, m.hc_epsilon);
            }
            if (omp_get_thread_num() == 0) tp[2] = now();
#pragma omp for schedule(static)
            for (int b = 0; b < H / XB; ++b) {
                for (int d = b * XB; d < (b + 1) * XB; ++d) {
                    float v = 0;
                    for (int i = 0; i < 4; ++i) v += c[i] * r[size_t(i) * H + d];
                    x[d] = v;
                }
                ss_x[b] = sumsq(x.data() + size_t(b) * XB, XB);
            }
            if (omp_get_thread_num() == 0) tp[3] = now();
#pragma omp single
            {
                double s = 0;
                for (double v : ss_x) s += v;
                inv_x = float(1 / std::sqrt(s / H + m.rms_epsilon));
            }
            if (omp_get_thread_num() == 0) tp[4] = now();
#pragma omp for schedule(static)
            for (int d = 0; d < H; ++d) y[d] = x[d] * inv_x * (w.empty() ? 1 : w[d]);
        });
        const double te = now();
        timers["mhc.site"] += te - t;
        timers["mhc.s0_fork"] += tp[0] - t;
        timers["mhc.s1_proj"] += tp[1] - tp[0];
        timers["mhc.s2_coef"] += tp[2] - tp[1];
        timers["mhc.s3_comb"] += tp[3] - tp[2];
        timers["mhc.s4_inv"] += tp[4] - tp[3];
        timers["mhc.s5_scale"] += te - tp[4];
        return {y, c};
    }
    // One mHC read site, the following RMS norm, and optionally the residual update before it, in one team region:
    // each thread takes its MHC_BLOCK-value blocks of the 4 x hidden residual, writes them (r' = c_w[4+j] y_w + sum_i
    // c_w[8+4i+j] r_i, write()'s arithmetic) when a write is given, and while they are in its cache adds their sum of
    // squares and fn . block for both tiles (fn placed by these blocks); one thread sums the block partials in block
    // order (projected = inv * fn . r'), computes the coefficients, then x = sum_i c[i] r'_i per 256 values with its sum of
    // squares, and the norm. Fusing the update saved a fork and the remote reads of a residual just written by other threads.
    float site_projected[32];
    alignas(256) float site_projected_pad[32 * 64];
    struct SiteScratch { std::vector<float> part, th_part, g_part; std::vector<double> ss_r, ss_x, th_ss, g_ss; Vec x; };
    SiteScratch site_w;
    std::pair<Vec, Vec> site(Vec &r, const Vec *cw, const Vec *yw, const Matrix &fn, const Vec &base, const Vec &scale, const Vec &w,
                             const std::vector<const Matrix *> &next = {}) {   // next: the following phase's weights (L2 prefetch)
        static const bool fused = !(std::getenv("STRATA_RUNNER_MHC_FUSED") && std::string(std::getenv("STRATA_RUNNER_MHC_FUSED")) == "0");
        const int H = m.hidden, n = int(r.size()), DC = MHC_CHUNK, nc = H / DC, nq = 4 * nc;   // nq: (stream, chunk) units
        static const bool mhc_prefetch = !(std::getenv("STRATA_RUNNER_MHC_PREFETCH") && std::string(std::getenv("STRATA_RUNNER_MHC_PREFETCH")) == "0");
        static const bool site_trace = std::getenv("STRATA_RUNNER_SITE_TRACE") != nullptr;   // per-thread lateness (diagnostic)
        if (site_trace && site_tr.empty()) site_tr.assign(64 * 32, 0.0);
        if (!fused || !mhc_layout(fn) || fn.cols != n || n != 4 * H || fn.rows > 32 || H % DC) {
            if (cw) write(r, *cw, *yw);
            return read_norm(fn, base, scale, r, w);
        }
        const double t = now();
        const int tiles = fn.padded / 16;
        const size_t tile_bytes = fn.bytes() / size_t(tiles), bb = block_bytes(Kind::B16);
        auto &W = site_w;
        W.part.resize(size_t(nq) * fn.padded), W.ss_r.resize(nq), W.ss_x.resize(nc), W.x.resize(H);
        W.th_ss.resize(64 * 32), W.th_part.resize(64 * 64), W.g_ss.resize(8 * 32), W.g_part.resize(8 * 64);
        if (cw && r_next.size() != r.size()) r_next.resize(r.size());
        const float *src = cw ? r_next.data() : r.data();
        Vec c(24), y(H);
        float inv_x = 0;
        double tp[5] = {0, 0, 0, 0, 0};   // thread 0's clock after each step (sub-phase timers)
        double tp_rows = 0;
        team_run([&] {
            const int th = omp_get_thread_num(), nth = omp_get_num_threads();
            if (th == 0) tp[0] = now();
            if (site_trace) site_tr[size_t(th) * 32] = now();
            if (!next.empty()) prefetch_mats(next, th, nth);
            const int c0 = mhc_block_begin(nc, th, nth), c1 = mhc_block_begin(nc, th + 1, nth);
            const uint8_t *fnd = fn.cmg_data(th);
            const bool cm = !fn.cmg_copy.empty();
            auto blk = [&](int ch, int j, int tile) {   // block (tile, stream j's chunk ch) in the layout this thread reads
                return cm ? fnd + (size_t(ch * 4 + j) * tiles + tile) * bb : fnd + size_t(tile) * tile_bytes + size_t(j * nc + ch) * bb;
            };
            if ((c1 - c0) * 4 > 64) std::abort();   // the unit buffers below (128 chunks over >= 8 threads)
            alignas(256) float lpart[64 * 32];
            double lss[64];
            if (mhc_prefetch)   // every weight block and source piece of this thread's units at once: their misses overlap
                for (int ch = c0; ch < c1; ++ch)
                    for (int j = 0; j < 4; ++j) {
                        const int q = j * nc + ch;
                        for (int tile = 0; tile < tiles; ++tile)
                            for (size_t o = 0; o < bb; o += 256) __builtin_prefetch(blk(ch, j, tile) + o, 0, 3);
                        if (cw) __builtin_prefetch(r.data() + size_t(j) * H + size_t(ch) * DC, 0, 3);
                        else __builtin_prefetch(src + size_t(q) * DC, 0, 3);
                    }
            for (int ch = c0; ch < c1; ++ch)
                for (int j = 0; j < 4; ++j) {
                    const int q = j * nc + ch;   // stream j, positions [32 ch, 32 ch + 32)
                    if (cw) {
                        const Vec &cc = *cw, &yy = *yw;
                        for (int d = ch * DC; d < (ch + 1) * DC; ++d) {
                            float v = cc[4 + j] * yy[d];
                            for (int i = 0; i < 4; ++i) v += cc[8 + i * 4 + j] * r[size_t(i) * H + d];
                            r_next[size_t(j) * H + d] = v;
                        }
                    }
                    const int u = (ch - c0) * 4 + j;   // this thread's unit: its partials stay in its own buffer
                    lss[u] = sumsq(src + size_t(q) * DC, DC);
                    const float *xs[1] = {src + size_t(q) * DC};
                    for (int tile = 0; tile < tiles; ++tile) gf4_bf16k(blk(ch, j, tile), DC, 0, 1, xs, 1, lpart + size_t(u) * 32 + 16 * tile, 32);
                }
            {   // this thread's units summed (chunk-major, stream order), one partial per thread for the rows below
                double sq = 0;
                float pr[32] = {0};
                for (int ch = c0; ch < c1; ++ch)
                    for (int j = 0; j < 4; ++j) {
                        const int u = (ch - c0) * 4 + j;
                        sq += lss[u];
                        for (int row = 0; row < fn.rows; ++row) pr[row] += lpart[size_t(u) * 32 + row];
                    }
                W.th_ss[size_t(th) * 32] = sq;   // one 256-byte line per thread (8-byte slots shared lines: false sharing)
                std::copy(pr, pr + 32, W.th_part.begin() + size_t(th) * 64);
            }
            if (site_trace) site_tr[size_t(th) * 32 + 1] = now();
            team_barrier();
            if (th == 0) tp[1] = now();
            if (site_trace && th == 0) {   // the latest start and the latest phase-A end against thread 0's start
                double ls = 0, le = 0, own = site_tr[1] - tp[0];
                int ws = 0, we = 0;
                for (int u = 0; u < nth; ++u) {
                    const double s_ = site_tr[size_t(u) * 32] - tp[0], e_ = site_tr[size_t(u) * 32 + 1] - tp[0];
                    if (s_ > ls) ls = s_, ws = u;
                    if (e_ > le) le = e_, we = u;
                }
                site_stat[0] += ls, site_stat[1] += le, site_stat[2] += own, site_stat[3] += 1;
                site_late_start[ws] += 1, site_late_end[we] += 1;
            }
            // the per-thread partials summed per CMG by its first thread (its members' lines are in its L2), then the
            // four CMG sums by thread 0 (24 row threads reading all 47 threads' lines cost 2-20 us per site, depending
            // on where the lines were), the coefficients on thread 0
            const int gsz = cmg_threads(), g = th / gsz;
            if (th % gsz == 0) {
                double sq = 0;
                float pr[32] = {0};
                for (int u = th; u < std::min(nth, th + gsz); ++u) {
                    sq += W.th_ss[size_t(u) * 32];
                    for (int row = 0; row < fn.rows; ++row) pr[row] += W.th_part[size_t(u) * 64 + row];
                }
                W.g_ss[size_t(g) * 32] = sq;
                std::copy(pr, pr + 32, W.g_part.begin() + size_t(g) * 64);
            }
            team_barrier();
            if (th == 0) {
                tp_rows = now();
                double sq = 0;
                float pr[32] = {0};
                for (int h = 0; h * gsz < nth; ++h) {
                    sq += W.g_ss[size_t(h) * 32];
                    for (int row = 0; row < fn.rows; ++row) pr[row] += W.g_part[size_t(h) * 64 + row];
                }
                const float inv = float(1 / std::sqrt(sq / n + m.rms_epsilon));
                for (int row = 0; row < fn.rows; ++row) site_projected[row] = pr[row] * inv;
                mhc_coef(site_projected, base.data(), scale.data(), c.data(), m.sinkhorn_iterations, m.hc_epsilon);
            }
            team_barrier();
            if (th == 0) tp[2] = now();
            for (int ch = c0; ch < c1; ++ch) {   // this thread's own chunks again: their values are in its cache
                for (int d = ch * DC; d < (ch + 1) * DC; ++d) {
                    float v = 0;
                    for (int i = 0; i < 4; ++i) v += c[i] * src[size_t(i) * H + d];
                    W.x[d] = v;
                }
                W.ss_x[ch] = sumsq(W.x.data() + size_t(ch) * DC, DC);
            }
            team_barrier();
            if (th == 0) tp[3] = now();
            if (omp_get_thread_num() == 0)
            {
                double sq = 0;
                for (double v : W.ss_x) sq += v;
                inv_x = float(1 / std::sqrt(sq / H + m.rms_epsilon));
            }
            team_barrier();
            if (th == 0) tp[4] = now();
            for (int d = c0 * DC; d < c1 * DC; ++d) y[d] = W.x[d] * inv_x * (w.empty() ? 1 : w[d]);
        });
        if (cw) r.swap(r_next);
        const double te = now();
        timers["mhc.site"] += te - t;
        timers["mhc.s0_fork"] += tp[0] - t;
        timers["mhc.s1_proj"] += tp[1] - tp[0];
        timers["mhc.s2_coef"] += tp[2] - tp[1];
        timers["mhc.s2a_rows"] += tp_rows - tp[1];
        timers["mhc.s3_comb"] += tp[3] - tp[2];
        timers["mhc.s4_inv"] += tp[4] - tp[3];
        timers["mhc.s5_scale"] += te - tp[4];
        return {y, c};
    }
    // site() for the T tokens of a multi-token pass in one region: per block, the update and sum of squares per token
    // and one projection call for all T tokens (fn read once); the same arithmetic per token as T site() calls
    std::vector<Vec> r_next_multi, sm_x;
    std::vector<double> sm_th_ss, sm_g_ss;
    std::vector<float> sm_th_part, sm_g_part;
    std::vector<float> sm_part;
    std::vector<double> sm_ss_r, sm_ss_x;
    // site() for the T tokens of a long pass with whole tokens per thread (token t on thread t % team): each thread
    // replays the team's split of site() for its tokens, virtual thread by virtual thread (its units in (chunk, stream)
    // order, then the per-CMG sums, then the CMG sums in order), so every sum has site()'s order and the result its
    // bits; no barrier inside the site, and fn's blocks come from the thread's CMG copy
    std::vector<Vec> st_rn;   // per token: the updated residual (swapped with r)
    std::array<double, 64> mhc_thr_total{};   // every thread's site_tokens time (RUNNER_MHC_THREADS at the prefill end)
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
    // one BF16 block (32 columns x 16 rows) for NT tokens with gf4_bf16k's arithmetic (bf16_tile, the default chain
    // form: chain c over columns 8c .. 8c + 7, then (q0 + q1) + (q2 + q3) added to a zero sum), added to acc[n]:
    // the bits of gf4_bf16k(m, 32, 0, 1, x, NT, lp) followed by acc += lp, without a call and a store per block
    typedef svfloat32_t vf32s __attribute__((arm_sve_vector_bits(512)));
    template <int NT>
    static inline void mhc_block_acc(const uint8_t *m, const float *const *x, vf32s *acc) {
        const svbool_t pw = svptrue_b32();
        // the weights two columns per 64-byte load, widened by a zip with zero (bf16 bits as the high half of each
        // f32: the same values as the shift; one-column extending loads were slower)
        const svuint16_t z0 = svdup_n_u16(0);
        vf32s w[32];
        for (int k = 0; k < 32; k += 2) {
            const svuint16_t v = svld1_u16(svptrue_b16(), reinterpret_cast<const uint16_t *>(m + size_t(k) * 32));
            w[k] = svreinterpret_f32_u16(svzip1_u16(z0, v)), w[k + 1] = svreinterpret_f32_u16(svzip2_u16(z0, v));
        }
        vf32s q[NT][4];
        for (int n = 0; n < NT; ++n) for (int c = 0; c < 4; ++c) q[n][c] = svdup_n_f32(0);
#pragma GCC unroll 2
        for (int h = 0; h < 2; ++h)
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i)
#pragma GCC unroll 4
                for (int c = 0; c < 4; ++c) {
                    const int k = 8 * c + 4 * h + i;
#pragma GCC unroll 4
                    for (int n = 0; n < NT; ++n) q[n][c] = svmla_f32_x(pw, q[n][c], w[k], svdup_n_f32(x[n][k]));
                }
        for (int n = 0; n < NT; ++n) {
            const svfloat32_t t = svadd_f32_x(pw, svdup_n_f32(0), svadd_f32_x(pw, svadd_f32_x(pw, q[n][0], q[n][1]), svadd_f32_x(pw, q[n][2], q[n][3])));
            acc[n] = svadd_f32_x(pw, acc[n], t);
        }
    }
#endif
    std::vector<std::pair<Vec, Vec>> site_tokens(std::vector<Vec> &r, const std::vector<Vec> *cw, const std::vector<Vec> *yw, const Matrix &fn,
                                                 const Vec &base, const Vec &scale, const Vec &w) {
        const double t0 = now();
        const int T = int(r.size()), H = m.hidden, n = int(r[0].size()), DC = MHC_CHUNK, nc = H / DC;
        const int tiles = fn.padded / 16;
        const size_t tile_bytes = fn.bytes() / size_t(tiles), bb = block_bytes(Kind::B16);
        std::vector<std::pair<Vec, Vec>> res(T);
        if (cw && st_rn.size() < size_t(T)) st_rn.resize(T);
        // the projection's blocks with their row sums in registers (STRATA_RUNNER_MHC_UNITS=0: one gf4_bf16k call and
        // scalar row adds per block, as before; bitwise the same)
        static const bool fused_units = !(std::getenv("STRATA_RUNNER_MHC_UNITS") && std::string(std::getenv("STRATA_RUNNER_MHC_UNITS")) == "0");
        (void)fused_units;
        static const bool write_first = !(std::getenv("STRATA_RUNNER_MHC_WRITE_FIRST") && std::string(std::getenv("STRATA_RUNNER_MHC_WRITE_FIRST")) == "0");
        double st_t[4] = {0, 0, 0, 0};   // thread 0: projection, per-token tail, its end (sub-timers)
        double st_wf = 0;
        std::array<double, 64> th_busy{};  // every thread's time in the region (the balance by CMG)
        team_run([&] {
            const double tb0 = now();
            const int th = omp_get_thread_num(), nth = omp_get_num_threads(), gsz = cmg_threads();
            const uint8_t *fnd = fn.cmg_data(th);
            const bool cm = !fn.cmg_copy.empty();
            if (th == 0) st_t[3] = cm;
            auto blk = [&](int ch, int j, int tile) {
                return cm ? fnd + (size_t(ch * 4 + j) * tiles + tile) * bb : fnd + size_t(tile) * tile_bytes + size_t(j * nc + ch) * bb;
            };
            // this thread's tokens, up to MHC_TOK (STRATA_RUNNER_MHC_TOK: 4, 8 or 12; default 4) per pass over fn's
            // blocks, in kernel groups of 4 (12 streamed the weights once per 12 tokens but was slower, 0.391 vs 0.314
            // ms: the passes' residuals no longer stay in cache). Dynamic groups within a CMG were slower too (0.407
            // vs 0.334 ms: the owner's L1 holds its tokens' residuals)
            constexpr int MT = 12;
            static const int mtok = std::getenv("STRATA_RUNNER_MHC_TOK") ? std::clamp(std::atoi(std::getenv("STRATA_RUNNER_MHC_TOK")) / 4 * 4, 4, MT) : 4;
            int mine[MT], k = 0;
            for (int t0_ = th; t0_ < T; t0_ += mtok * nth) {
                k = 0;
                for (int t = t0_; t < T && k < mtok; t += nth) mine[k++] = t;
                const int g4 = k;
                const float *src[MT];
                for (int a = 0; a < g4; ++a) {
                    const int t = mine[a];
                    if (cw && st_rn[t].size() != size_t(n)) st_rn[t].resize(n);
                    src[a] = cw ? st_rn[t].data() : r[t].data();
                }
                float tot_pr[MT][32] = {}, gpr[MT][32] = {};
                double tot_sq[MT] = {}, gsq[MT] = {};
                const double ta = th == 0 ? now() : 0;
                // the group's residual updates first, whole streams (STRATA_RUNNER_MHC_WRITE_FIRST=0: per unit, just
                // before its projection): the projection then reads values that are not in flight in the store queue
                // (A64FX forwards stores to the reloads poorly); each value is computed as before
                if (cw && write_first)
                    for (int a = 0; a < g4; ++a) {
                        const int t = mine[a];
                        const Vec &cc = (*cw)[t], &yy = (*yw)[t];
                        float *rn = st_rn[t].data();
                        const float *rt = r[t].data();
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                        if (H % 16 == 0) {   // the same multiply, then the same fused multiply-adds, 16 lanes at a time
                            const svbool_t pw = svptrue_b32();
                            for (int j = 0; j < 4; ++j) {
                                const svfloat32_t cy = svdup_n_f32(cc[4 + j]), c0 = svdup_n_f32(cc[8 + j]), c1 = svdup_n_f32(cc[12 + j]),
                                                  c2 = svdup_n_f32(cc[16 + j]), c3 = svdup_n_f32(cc[20 + j]);
                                for (int d = 0; d < H; d += 16) {
                                    svfloat32_t v = svmul_f32_x(pw, cy, svld1_f32(pw, yy.data() + d));
                                    v = svmla_f32_x(pw, v, c0, svld1_f32(pw, rt + d));
                                    v = svmla_f32_x(pw, v, c1, svld1_f32(pw, rt + size_t(H) + d));
                                    v = svmla_f32_x(pw, v, c2, svld1_f32(pw, rt + size_t(2) * H + d));
                                    v = svmla_f32_x(pw, v, c3, svld1_f32(pw, rt + size_t(3) * H + d));
                                    svst1_f32(pw, rn + size_t(j) * H + d, v);
                                }
                            }
                            continue;
                        }
#endif
                        for (int j = 0; j < 4; ++j)
                            for (int d = 0; d < H; ++d) {
                                float v = cc[4 + j] * yy[d];
                                for (int i = 0; i < 4; ++i) v += cc[8 + i * 4 + j] * rt[size_t(i) * H + d];
                                rn[size_t(j) * H + d] = v;
                            }
                    }
                if (th == 0) st_t[1] -= 0, st_wf += now() - ta;   // thread 0: the residual update (write-first)
                for (int u = 0; u < nth; ++u) {   // the team's threads in order, as site()
                    if (u % gsz == 0)
                        for (int a = 0; a < g4; ++a) { gsq[a] = 0; for (int row = 0; row < 32; ++row) gpr[a][row] = 0; }
                    float pr[MT][32] = {};
                    double sq[MT] = {};
                    for (int ch = mhc_block_begin(nc, u, nth); ch < mhc_block_begin(nc, u + 1, nth); ++ch)
                        for (int j = 0; j < 4; ++j) {
                            const int q = j * nc + ch;
                            const float *xs[MT];
                            for (int a = 0; a < g4; ++a) {
                                const int t = mine[a];
                                if (cw && !write_first) {
                                    const Vec &cc = (*cw)[t], &yy = (*yw)[t];
                                    float *rn = st_rn[t].data();
                                    for (int d = ch * DC; d < (ch + 1) * DC; ++d) {
                                        float v = cc[4 + j] * yy[d];
                                        for (int i = 0; i < 4; ++i) v += cc[8 + i * 4 + j] * r[t][size_t(i) * H + d];
                                        rn[size_t(j) * H + d] = v;
                                    }
                                }
                                sq[a] += sumsq(src[a] + size_t(q) * DC, DC);
                                xs[a] = src[a] + size_t(q) * DC;
                            }
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                            if (fused_units && DC == 32 && tiles <= 2) {   // the rows' sums in registers (lanes = rows)
                                for (int tile = 0; tile < tiles; ++tile)
                                    for (int a0 = 0; a0 < g4; a0 += 4) {   // the block stays in L1 for every group
                                        const int gn = std::min(4, g4 - a0);
                                        vf32s acc[4];
                                        for (int a = 0; a < gn; ++a) acc[a] = svld1_f32(svptrue_b32(), pr[a0 + a] + 16 * tile);
                                        switch (gn) {
                                        case 1: mhc_block_acc<1>(blk(ch, j, tile), xs + a0, acc); break;
                                        case 2: mhc_block_acc<2>(blk(ch, j, tile), xs + a0, acc); break;
                                        case 3: mhc_block_acc<3>(blk(ch, j, tile), xs + a0, acc); break;
                                        default: mhc_block_acc<4>(blk(ch, j, tile), xs + a0, acc); break;
                                        }
                                        for (int a = 0; a < gn; ++a) svst1_f32(svptrue_b32(), pr[a0 + a] + 16 * tile, acc[a]);
                                    }
                                continue;
                            }
#endif
                            for (int a0 = 0; a0 < g4; a0 += 4) {
                                const int gn = std::min(4, g4 - a0);
                                alignas(64) float lp[4 * 32];
                                for (int tile = 0; tile < tiles; ++tile) gf4_bf16k(blk(ch, j, tile), DC, 0, 1, xs + a0, gn, lp + 16 * tile, 32);
                                for (int a = 0; a < gn; ++a)
                                    for (int row = 0; row < fn.rows; ++row) pr[a0 + a][row] += lp[a * 32 + row];
                            }
                        }
                    for (int a = 0; a < g4; ++a) {
                        gsq[a] += sq[a];
                        for (int row = 0; row < fn.rows; ++row) gpr[a][row] += pr[a][row];
                    }
                    if (u % gsz == gsz - 1 || u == nth - 1)
                        for (int a = 0; a < g4; ++a) {
                            tot_sq[a] += gsq[a];
                            for (int row = 0; row < fn.rows; ++row) tot_pr[a][row] += gpr[a][row];
                        }
                }
                const double tb = th == 0 ? now() : 0;
                if (th == 0) st_t[0] += tb - ta;
                for (int a = 0; a < g4; ++a) {   // coefficients, combination, norm: site()'s steps for one token
                    const int t = mine[a];
                    Vec c(24), x(H), y(H);
                    float projected[32];
                    const float inv = float(1 / std::sqrt(tot_sq[a] / n + m.rms_epsilon));
                    for (int row = 0; row < fn.rows; ++row) projected[row] = tot_pr[a][row] * inv;
                    mhc_coef(projected, base.data(), scale.data(), c.data(), m.sinkhorn_iterations, m.hc_epsilon);
                    double sqx = 0;
                    for (int ch = 0; ch < nc; ++ch) {
                        for (int d = ch * DC; d < (ch + 1) * DC; ++d) {
                            float v = 0;
                            for (int i = 0; i < 4; ++i) v += c[i] * src[a][size_t(i) * H + d];
                            x[d] = v;
                        }
                        sqx += sumsq(x.data() + size_t(ch) * DC, DC);
                    }
                    const float inv_x = float(1 / std::sqrt(sqx / H + m.rms_epsilon));
                    for (int d = 0; d < H; ++d) y[d] = x[d] * inv_x * (w.empty() ? 1 : w[d]);
                    if (cw) r[t].swap(st_rn[t]);
                    res[t] = {std::move(y), std::move(c)};
                }
                if (th == 0) st_t[1] += now() - tb;
            }
            if (th == 0) st_t[2] = now();
            th_busy[th] = now() - tb0;
        });
        const double te = now();
        timers["mhc.site"] += te - t0;
        timers["mhc.site_tokens"] += te - t0;
        timers["mhc.st_proj"] += st_t[0], timers["mhc.st_tail"] += st_t[1], timers["mhc.st_write"] += st_wf;
        timers["mhc.st_wait"] += te - st_t[2];   // thread 0 done -> region end: the slowest thread's lead
        timers["mhc.st_cmg"] += st_t[3];         // sites whose fn has CMG copies (count; > 0: local reads)
        {
            const int nth = omp_get_max_threads(), gsz = cmg_threads();
            double mx = 0, mean = 0, cm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            int cn[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int i = 0; i < nth && i < 64; ++i) mx = std::max(mx, th_busy[i]), mean += th_busy[i], cm[std::min(7, i / gsz)] += th_busy[i], ++cn[std::min(7, i / gsz)];
            timers["mhc.thr_max"] += mx, timers["mhc.thr_mean"] += mean / nth;
            for (int i = 0; i < nth && i < 64; ++i) mhc_thr_total[i] += th_busy[i];
            for (int g = 0; g < 4; ++g) if (cn[g]) timers[std::string("mhc.thr_cmg") + char('0' + g)] += cm[g] / cn[g];
        }
        return res;
    }
    // tokens per batched site region (STRATA_RUNNER_SITE_MAX: fewer). 64: at C = 16 (passes of ~37 tokens) mhc 0.795
    // -> 0.673 ms per token against site_tokens (whole tokens per thread, each streaming all of fn for one token)
    static constexpr int SITE_MAX = 64;
    std::vector<std::pair<Vec, Vec>> site_multi(std::vector<Vec> &r, const std::vector<Vec> *cw, const std::vector<Vec> *yw, const Matrix &fn,
                                                const Vec &base, const Vec &scale, const Vec &w) {
        const int T = int(r.size()), H = m.hidden, n = int(r[0].size()), DC = MHC_CHUNK, nc = H / DC, nq = 4 * nc;
        std::vector<std::pair<Vec, Vec>> res(T);
        static const int site_max = std::getenv("STRATA_RUNNER_SITE_MAX") ? std::clamp(std::atoi(std::getenv("STRATA_RUNNER_SITE_MAX")), 4, SITE_MAX) : SITE_MAX;
        // STRATA_RUNNER_SITE_TOKENS=0: long passes in groups of site_max tokens (default: site_tokens)
        static const bool by_tokens = !(std::getenv("STRATA_RUNNER_SITE_TOKENS") && std::string(std::getenv("STRATA_RUNNER_SITE_TOKENS")) == "0");
        if (T > site_max && by_tokens && mhc_layout(fn) && fn.cols == n && n == 4 * H && fn.rows <= 32 && H % DC == 0)
            return site_tokens(r, cw, yw, fn, base, scale, w);
        if (T > site_max && mhc_layout(fn)) {   // a prefill chunk: groups of site_max tokens per region
            for (int g0 = 0; g0 < T; g0 += site_max) {
                const int g1 = std::min(T, g0 + site_max);
                std::vector<Vec> rg(std::make_move_iterator(r.begin() + g0), std::make_move_iterator(r.begin() + g1)), cg, yg;
                if (cw) cg.assign(cw->begin() + g0, cw->begin() + g1), yg.assign(yw->begin() + g0, yw->begin() + g1);
                auto sg = site_multi(rg, cw ? &cg : nullptr, cw ? &yg : nullptr, fn, base, scale, w);
                for (int t = g0; t < g1; ++t) r[t] = std::move(rg[t - g0]), res[t] = std::move(sg[t - g0]);
            }
            return res;
        }
        if (T == 1 || !mhc_layout(fn) || fn.cols != n || n != 4 * H || fn.rows > 32 || H % DC || T > SITE_MAX) {
            for (int t = 0; t < T; ++t) res[t] = site(r[t], cw ? &(*cw)[t] : nullptr, yw ? &(*yw)[t] : nullptr, fn, base, scale, w);
            return res;
        }
        const double t0 = now();
        const int tiles = fn.padded / 16;
        const size_t tile_bytes = fn.bytes() / size_t(tiles), bb = block_bytes(Kind::B16);
        auto &part = sm_part;   // [token][unit][row]; persistent like the other buffers below
        auto &ss_r = sm_ss_r, &ss_x = sm_ss_x;
        if (part.size() < size_t(T) * nq * fn.padded) part.resize(size_t(T) * nq * fn.padded);
        if (ss_r.size() < size_t(T) * nq) ss_r.resize(size_t(T) * nq);
        if (ss_x.size() < size_t(T) * nc) ss_x.resize(size_t(T) * nc);
        if (cw) {
            r_next_multi.resize(T);
            for (auto &v : r_next_multi) v.resize(n);
        }
        std::vector<const float *> src(T);
        for (int t = 0; t < T; ++t) src[t] = cw ? r_next_multi[t].data() : r[t].data();
        if (sm_th_ss.size() < size_t(T) * 64 * 32)
            sm_th_ss.resize(size_t(T) * 64 * 32), sm_th_part.resize(size_t(T) * 64 * 64), sm_g_ss.resize(size_t(T) * 8 * 32), sm_g_part.resize(size_t(T) * 8 * 64);
        auto &x = sm_x;
        if (x.size() < size_t(T)) x.resize(T);
        for (auto &v : x) if (v.size() != size_t(H)) v.resize(H);
        std::vector<Vec> c(T, Vec(24)), y(T);   // y sized inside the region, a token per thread (a serial zero fill
                                                // of T x H values was ~1.7 us per token)
        std::vector<float> inv_x(T);
        static const bool mhc_prefetch = !(std::getenv("STRATA_RUNNER_MHC_PREFETCH") && std::string(std::getenv("STRATA_RUNNER_MHC_PREFETCH")) == "0");
        float projected[SITE_MAX][32];
        // phase clocks (thread 0 after each barrier) and every thread's projection time, by pass size (mhs<T bucket>.*)
        double sm_ts[6] = {0, 0, 0, 0, 0, 0};
        std::array<double, 64> sm_pb{}, sm_wb{};
        const double t_reg = now();
        team_run([&] {
            const int th = omp_get_thread_num(), nth = omp_get_num_threads();
            const double tb0 = now();
            for (int t = th; t < T; t += nth) y[t].resize(H);   // written after the barriers below
            const int c0 = mhc_block_begin(nc, th, nth), c1 = mhc_block_begin(nc, th + 1, nth);
            const uint8_t *fnd = fn.cmg_data(th);
            const bool cm = !fn.cmg_copy.empty();
            auto blk = [&](int ch, int j, int tile) {   // block (tile, stream j's chunk ch) in the layout this thread reads
                return cm ? fnd + (size_t(ch * 4 + j) * tiles + tile) * bb : fnd + size_t(tile) * tile_bytes + size_t(j * nc + ch) * bb;
            };
            if ((c1 - c0) * 4 > 64) std::abort();
            // [unit][token][row] and [unit][token]: this thread's own buffers (first touched by it)
            static thread_local std::vector<float> lpart_v;
            static thread_local std::vector<double> lss_v;
            if (lpart_v.size() < size_t(c1 - c0) * 4 * T * 32) lpart_v.resize(size_t(c1 - c0) * 4 * T * 32), lss_v.resize(size_t(c1 - c0) * 4 * T);
            float *lpart = lpart_v.data();
            double *lss = lss_v.data();
            // this thread's chunks of every token's residual streams (and of y with a residual update): T separate
            // vectors, written by other threads, often on other CMGs; prefetched at once, the misses overlap instead of
            // coming one chunk after another. STRATA_RUNNER_MHC_PREFETCH_R=0: off
            static const bool write_first_m = !(std::getenv("STRATA_RUNNER_MHC_WRITE_FIRST") && std::string(std::getenv("STRATA_RUNNER_MHC_WRITE_FIRST")) == "0");
            static const bool prefetch_r = !(std::getenv("STRATA_RUNNER_MHC_PREFETCH_R") && std::string(std::getenv("STRATA_RUNNER_MHC_PREFETCH_R")) == "0");
            // STRATA_RUNNER_MHC_PF_DIST=D > 0 (default 2; 0: all at once): the first D tokens here, then token t + D's
            // lines during token t's update (all T tokens at once is ~27 lines per token, beyond the misses a core
            // keeps in flight; at 36 tokens the update took 62 -> 51 us)
            static const int pf_dist = std::getenv("STRATA_RUNNER_MHC_PF_DIST") ? std::atoi(std::getenv("STRATA_RUNNER_MHC_PF_DIST")) : 2;
            auto prefetch_tok = [&](int t) {
                for (int i = 0; i < 4; ++i) {
                    const float *rt = (cw ? r[t].data() : src[t]) + size_t(i) * H;
                    for (size_t o = size_t(c0) * DC; o < size_t(c1) * DC; o += 64) __builtin_prefetch(rt + o, 0, 3);
                    if (cw)
                        for (size_t o = size_t(c0) * DC; o < size_t(c1) * DC; o += 64) __builtin_prefetch(r_next_multi[t].data() + size_t(i) * H + o, 1, 3);
                    if (cw && i == 0)
                        for (size_t o = size_t(c0) * DC; o < size_t(c1) * DC; o += 64) __builtin_prefetch((*yw)[t].data() + o, 0, 3);
                }
            };
            const bool pf_pipe = prefetch_r && pf_dist > 0 && cw && write_first_m;
            if (pf_pipe)
                for (int t = 0; t < std::min(T, pf_dist); ++t) prefetch_tok(t);
            else if (prefetch_r)
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < 4; ++i) {
                        const float *rt = (cw ? r[t].data() : src[t]) + size_t(i) * H;
                        for (size_t o = size_t(c0) * DC; o < size_t(c1) * DC; o += 64) __builtin_prefetch(rt + o, 0, 3);
                        if (cw)   // the update's output lines, for write
                            for (size_t o = size_t(c0) * DC; o < size_t(c1) * DC; o += 64) __builtin_prefetch(r_next_multi[t].data() + size_t(i) * H + o, 1, 3);
                        if (cw && i == 0)
                            for (size_t o = size_t(c0) * DC; o < size_t(c1) * DC; o += 64) __builtin_prefetch((*yw)[t].data() + o, 0, 3);
                    }
            if (mhc_prefetch)   // site()'s prefetch of this thread's weight blocks
                for (int ch = c0; ch < c1; ++ch)
                    for (int j = 0; j < 4; ++j)
                        for (int tile = 0; tile < tiles; ++tile)
                            for (size_t o = 0; o < bb; o += 256) __builtin_prefetch(blk(ch, j, tile) + o, 0, 3);
            // two barriers instead of five: each thread takes token th % T and a slice of its chunks; it sums every
            // thread's partials itself (per CMG, then the CMG sums: the same order), computes that token's coefficients
            // (the same values on each thread of the token), its chunks of x with their sums of squares, then after the
            // barrier inv_x from all chunks and its slice of y. STRATA_RUNNER_MHC_MERGE=0: the phases below
            static const bool merge_m = !(std::getenv("STRATA_RUNNER_MHC_MERGE") && std::string(std::getenv("STRATA_RUNNER_MHC_MERGE")) == "0");
            // STRATA_RUNNER_MHC_MERGE_MAX: the most tokens merged (default all: at 24-48 tokens 151-262 -> 139-233 us)
            static const int merge_max = std::getenv("STRATA_RUNNER_MHC_MERGE_MAX") ? std::atoi(std::getenv("STRATA_RUNNER_MHC_MERGE_MAX")) : SITE_MAX;
            const bool merged = merge_m && T <= merge_max && nth >= T && (nth + cmg_threads() - 1) / cmg_threads() <= 8;
            // the residual update of this thread's chunks for every token first, 16 lanes at a time (the same multiply,
            // then the same fused multiply-adds): the projection then reads values no longer in the store queue (A64FX
            // forwards stores to reloads poorly; see site_tokens). STRATA_RUNNER_MHC_WRITE_FIRST=0: per unit, as before
            bool updated = false;
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
            if (cw && write_first_m && DC % 16 == 0) {
                const svbool_t pw = svptrue_b32();
                for (int t = 0; t < T; ++t) {
                    if (pf_pipe && t + pf_dist < T) prefetch_tok(t + pf_dist);
                    const Vec &cc = (*cw)[t], &yy = (*yw)[t];
                    const float *rt = r[t].data();
                    float *rn = r_next_multi[t].data();
                    for (int j = 0; j < 4; ++j) {
                        const svfloat32_t cy = svdup_n_f32(cc[4 + j]), k0 = svdup_n_f32(cc[8 + j]), k1 = svdup_n_f32(cc[12 + j]),
                                          k2 = svdup_n_f32(cc[16 + j]), k3 = svdup_n_f32(cc[20 + j]);
                        for (int d = c0 * DC; d < c1 * DC; d += 16) {
                            svfloat32_t v = svmul_f32_x(pw, cy, svld1_f32(pw, yy.data() + d));
                            v = svmla_f32_x(pw, v, k0, svld1_f32(pw, rt + d));
                            v = svmla_f32_x(pw, v, k1, svld1_f32(pw, rt + size_t(H) + d));
                            v = svmla_f32_x(pw, v, k2, svld1_f32(pw, rt + size_t(2) * H + d));
                            v = svmla_f32_x(pw, v, k3, svld1_f32(pw, rt + size_t(3) * H + d));
                            svst1_f32(pw, rn + size_t(j) * H + d, v);
                        }
                    }
                }
                updated = true;
            }
            sm_wb[th] = now() - tb0;
            // the projection with the row sums in registers (mhc_block_acc, as site_tokens): tokens in groups of 4, this
            // thread's units inside (its blocks stay in L1); per token the same unit order, so the same sums as the
            // per-unit partials below. STRATA_RUNNER_MHC_UNITS=0: gf4_bf16k per unit into lpart, then the sums
            static const bool fused_m = !(std::getenv("STRATA_RUNNER_MHC_UNITS") && std::string(std::getenv("STRATA_RUNNER_MHC_UNITS")) == "0");
            if (fused_m && updated == bool(cw) && DC == 32 && tiles <= 2) {
                for (int a0 = 0; a0 < T; a0 += 4) {
                    const int gn = std::min(4, T - a0);
                    vf32s acc[2][4];
                    double sq[4] = {0, 0, 0, 0};
                    for (int tile = 0; tile < 2; ++tile)
                        for (int a = 0; a < 4; ++a) acc[tile][a] = svdup_n_f32(0);
                    for (int ch = c0; ch < c1; ++ch)
                        for (int j = 0; j < 4; ++j) {
                            const int q = j * nc + ch;
                            const float *xs[4];
                            for (int a = 0; a < gn; ++a) {
                                xs[a] = src[a0 + a] + size_t(q) * DC;
                                sq[a] += sumsq(xs[a], DC);
                            }
                            for (int tile = 0; tile < tiles; ++tile) switch (gn) {
                                case 1: mhc_block_acc<1>(blk(ch, j, tile), xs, acc[tile]); break;
                                case 2: mhc_block_acc<2>(blk(ch, j, tile), xs, acc[tile]); break;
                                case 3: mhc_block_acc<3>(blk(ch, j, tile), xs, acc[tile]); break;
                                default: mhc_block_acc<4>(blk(ch, j, tile), xs, acc[tile]); break;
                                }
                        }
                    for (int a = 0; a < gn; ++a) {
                        const int t = a0 + a;
                        if (!merged) sm_th_ss[(size_t(t) * 64 + th) * 32] = sq[a];   // one line per (token, thread), as site()
                        float *pp = sm_th_part.data() + (size_t(t) * 64 + th) * 64;
                        std::memcpy(pp + 32, &sq[a], sizeof(double));   // and next to the rows (the merged sums' read)
                        svst1_f32(svptrue_b32(), pp, acc[0][a]);
                        svst1_f32(svptrue_b32(), pp + 16, tiles == 2 ? svfloat32_t(acc[1][a]) : svdup_n_f32(0));
                    }
                }
                sm_pb[th] = now() - tb0;
            } else {
#endif
            for (int ch = c0; ch < c1; ++ch)
                for (int j = 0; j < 4; ++j) {
                    const int q = j * nc + ch;
                    const float *xs[SITE_MAX];
                    for (int t = 0; t < T; ++t) {
                        if (cw && !updated) {
                            const Vec &cc = (*cw)[t], &yy = (*yw)[t];
                            float *rn = r_next_multi[t].data();
                            for (int d = ch * DC; d < (ch + 1) * DC; ++d) {
                                float v = cc[4 + j] * yy[d];
                                for (int i = 0; i < 4; ++i) v += cc[8 + i * 4 + j] * r[t][size_t(i) * H + d];
                                rn[size_t(j) * H + d] = v;
                            }
                        }
                        lss[((ch - c0) * 4 + j) * T + t] = sumsq(src[t] + size_t(q) * DC, DC);
                        xs[t] = src[t] + size_t(q) * DC;
                    }
                    const int u = (ch - c0) * 4 + j;   // [unit][token][32] in this thread's own buffer
                    for (int tile = 0; tile < tiles; ++tile)   // fn read once for all T tokens (the kernel takes 4 at a time)
                        for (int b = 0; b < T; b += 4)
                            gf4_bf16k(blk(ch, j, tile), DC, 0, 1, xs + b, std::min(4, T - b), lpart + (size_t(u) * T + b) * 32 + 16 * tile, 32);
                }
            sm_pb[th] = now() - tb0;
            for (int t = 0; t < T; ++t) {   // per thread and token, as site()
                double sq = 0;
                float pr[32] = {0};
                for (int ch = c0; ch < c1; ++ch)
                    for (int j = 0; j < 4; ++j) {
                        const int u = (ch - c0) * 4 + j;
                        sq += lss[u * T + t];
                        for (int row = 0; row < fn.rows; ++row) pr[row] += lpart[(size_t(u) * T + t) * 32 + row];
                    }
                sm_th_ss[(size_t(t) * 64 + th) * 32] = sq;   // one line per (token, thread), as site()
                std::copy(pr, pr + 32, sm_th_part.begin() + (size_t(t) * 64 + th) * 64);
                std::memcpy(sm_th_part.data() + (size_t(t) * 64 + th) * 64 + 32, &sq, sizeof(double));
            }
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
            }
#endif
            team_barrier();
            if (th == 0) sm_ts[0] = now();
            const int gsz = cmg_threads(), g = th / gsz;   // site()'s order: per CMG, then the CMG sums
            if (merged) {
                const int t = th % T, k = th / T, kn = (nth - t + T - 1) / T;
                const int e0 = mhc_block_begin(nc, k, kn), e1 = mhc_block_begin(nc, k + 1, kn);
                for (int u = 0; u < nth; ++u)
                    __builtin_prefetch(&sm_th_part[(size_t(t) * 64 + u) * 64], 0, 3);   // rows and sum of squares: one line
                for (int i = 0; i < 4; ++i)
                    for (size_t o = size_t(e0) * DC; o < size_t(e1) * DC; o += 64) __builtin_prefetch(src[t] + size_t(i) * H + o, 0, 3);
                double gss[8];
                float gpr[8][32];
                for (int h = 0; h * gsz < nth; ++h) {
                    double sq = 0;
                    float pr[32] = {0};
                    for (int u = h * gsz; u < std::min(nth, h * gsz + gsz); ++u) {
                        double v;
                        std::memcpy(&v, sm_th_part.data() + (size_t(t) * 64 + u) * 64 + 32, sizeof(double));
                        sq += v;
                        for (int row = 0; row < fn.rows; ++row) pr[row] += sm_th_part[(size_t(t) * 64 + u) * 64 + row];
                    }
                    gss[h] = sq;
                    std::copy(pr, pr + 32, gpr[h]);
                }
                double sq = 0;
                float pr[32] = {0};
                for (int h = 0; h * gsz < nth; ++h) {
                    sq += gss[h];
                    for (int row = 0; row < fn.rows; ++row) pr[row] += gpr[h][row];
                }
                const float inv = float(1 / std::sqrt(sq / n + m.rms_epsilon));
                float pj[32], cc[24];
                for (int row = 0; row < fn.rows; ++row) pj[row] = pr[row] * inv;
                mhc_coef(pj, base.data(), scale.data(), cc, m.sinkhorn_iterations, m.hc_epsilon);
                if (k == 0) std::copy(cc, cc + 24, c[t].begin());
                if (th == 0) sm_ts[1] = sm_ts[2] = now();
                const float *s0 = src[t];
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                if (DC == 32) {   // as the vector combination below
                    typedef svfloat64_t vd __attribute__((arm_sve_vector_bits(512)));
                    const svbool_t pw = svptrue_b32(), pd = svptrue_b64();
                    const svfloat32_t k0 = svdup_n_f32(cc[0]), k1 = svdup_n_f32(cc[1]), k2 = svdup_n_f32(cc[2]), k3 = svdup_n_f32(cc[3]);
                    for (int ch = e0; ch < e1; ++ch) {
                        vd acc = svdup_n_f64(0);
                        for (int d = ch * DC; d < (ch + 1) * DC; d += 16) {
                            svfloat32_t v = svmla_f32_x(pw, svdup_n_f32(0), k0, svld1_f32(pw, s0 + d));
                            v = svmla_f32_x(pw, v, k1, svld1_f32(pw, s0 + size_t(H) + d));
                            v = svmla_f32_x(pw, v, k2, svld1_f32(pw, s0 + size_t(2) * H + d));
                            v = svmla_f32_x(pw, v, k3, svld1_f32(pw, s0 + size_t(3) * H + d));
                            svst1_f32(pw, x[t].data() + d, v);
                            const vd lo = svcvt_f64_f32_x(pd, svzip1_f32(v, v)), hi = svcvt_f64_f32_x(pd, svzip2_f32(v, v));
                            acc = svmla_f64_x(pd, acc, lo, lo), acc = svmla_f64_x(pd, acc, hi, hi);
                        }
                        ss_x[size_t(t) * nc + ch] = svaddv_f64(pd, acc);
                    }
                } else
#endif
                for (int ch = e0; ch < e1; ++ch) {
                    for (int d = ch * DC; d < (ch + 1) * DC; ++d) {
                        float v = 0;
                        for (int i = 0; i < 4; ++i) v += cc[i] * s0[size_t(i) * H + d];
                        x[t][d] = v;
                    }
                    ss_x[size_t(t) * nc + ch] = sumsq(x[t].data() + size_t(ch) * DC, DC);
                }
                team_barrier();
                if (th == 0) sm_ts[3] = sm_ts[4] = now();
                double sx = 0;
                for (int ch = 0; ch < nc; ++ch) sx += ss_x[size_t(t) * nc + ch];
                const float ix = float(1 / std::sqrt(sx / H + m.rms_epsilon));
                for (int d = e0 * DC; d < e1 * DC; ++d) y[t][d] = x[t][d] * ix * (w.empty() ? 1 : w[d]);
                return;
            }
            if (th % gsz == 0)
                for (int t = 0; t < T; ++t) {
                    double sq = 0;
                    float pr[32] = {0};
                    for (int u = th; u < std::min(nth, th + gsz); ++u) {
                        sq += sm_th_ss[(size_t(t) * 64 + u) * 32];
                        for (int row = 0; row < fn.rows; ++row) pr[row] += sm_th_part[(size_t(t) * 64 + u) * 64 + row];
                    }
                    sm_g_ss[(size_t(t) * 8 + g) * 32] = sq;
                    std::copy(pr, pr + 32, sm_g_part.begin() + (size_t(t) * 8 + g) * 64);
                }
            team_barrier();
            if (th == 0) sm_ts[1] = now();
#pragma omp for schedule(static, 1) nowait
            for (int t = 0; t < T; ++t) {
                double sq = 0;
                float pr[32] = {0};
                for (int h = 0; h * gsz < nth; ++h) {
                    sq += sm_g_ss[(size_t(t) * 8 + h) * 32];
                    for (int row = 0; row < fn.rows; ++row) pr[row] += sm_g_part[(size_t(t) * 8 + h) * 64 + row];
                }
                const float inv = float(1 / std::sqrt(sq / n + m.rms_epsilon));
                for (int row = 0; row < fn.rows; ++row) projected[t][row] = pr[row] * inv;
                mhc_coef(projected[t], base.data(), scale.data(), c[t].data(), m.sinkhorn_iterations, m.hc_epsilon);
            }
            team_barrier();
            if (th == 0) sm_ts[2] = now();
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
            // 16 lanes at a time (the same fused chain from 0) and sumsq's 8 double lanes taken from the registers
            // (zip puts value k in the low half of container k, as ld1uw does), not reloaded from the stores just made
            if (DC == 32) {
                typedef svfloat64_t vd __attribute__((arm_sve_vector_bits(512)));
                const svbool_t pw = svptrue_b32(), pd = svptrue_b64();
                for (int t = 0; t < T; ++t) {
                    const svfloat32_t k0 = svdup_n_f32(c[t][0]), k1 = svdup_n_f32(c[t][1]), k2 = svdup_n_f32(c[t][2]), k3 = svdup_n_f32(c[t][3]);
                    const float *s0 = src[t];
                    for (int ch = c0; ch < c1; ++ch) {
                        vd acc = svdup_n_f64(0);
                        for (int d = ch * DC; d < (ch + 1) * DC; d += 16) {
                            svfloat32_t v = svmla_f32_x(pw, svdup_n_f32(0), k0, svld1_f32(pw, s0 + d));
                            v = svmla_f32_x(pw, v, k1, svld1_f32(pw, s0 + size_t(H) + d));
                            v = svmla_f32_x(pw, v, k2, svld1_f32(pw, s0 + size_t(2) * H + d));
                            v = svmla_f32_x(pw, v, k3, svld1_f32(pw, s0 + size_t(3) * H + d));
                            svst1_f32(pw, x[t].data() + d, v);
                            const vd lo = svcvt_f64_f32_x(pd, svzip1_f32(v, v)), hi = svcvt_f64_f32_x(pd, svzip2_f32(v, v));
                            acc = svmla_f64_x(pd, acc, lo, lo), acc = svmla_f64_x(pd, acc, hi, hi);
                        }
                        ss_x[size_t(t) * nc + ch] = svaddv_f64(pd, acc);
                    }
                }
            } else
#endif
            for (int ch = c0; ch < c1; ++ch)
                for (int t = 0; t < T; ++t) {
                    for (int d = ch * DC; d < (ch + 1) * DC; ++d) {
                        float v = 0;
                        for (int i = 0; i < 4; ++i) v += c[t][i] * src[t][size_t(i) * H + d];
                        x[t][d] = v;
                    }
                    ss_x[size_t(t) * nc + ch] = sumsq(x[t].data() + size_t(ch) * DC, DC);
                }
            team_barrier();
            if (th == 0) sm_ts[3] = now();
#pragma omp for schedule(static, 1) nowait
            for (int t = 0; t < T; ++t) {
                double sq = 0;
                for (int ch = 0; ch < nc; ++ch) sq += ss_x[size_t(t) * nc + ch];
                inv_x[t] = float(1 / std::sqrt(sq / H + m.rms_epsilon));
            }
            team_barrier();
            if (th == 0) sm_ts[4] = now();
            for (int t = 0; t < T; ++t)
                for (int d = c0 * DC; d < c1 * DC; ++d) y[t][d] = x[t][d] * inv_x[t] * (w.empty() ? 1 : w[d]);
        });
        sm_ts[5] = now();
        for (int t = 0; t < T; ++t) {
            if (cw) r[t].swap(r_next_multi[t]);
            res[t] = {std::move(y[t]), std::move(c[t])};
        }
        timers["mhc.site"] += now() - t0;
        static const bool mhs_timers = std::getenv("STRATA_RUNNER_MHC_PHASES") && std::string(std::getenv("STRATA_RUNNER_MHC_PHASES")) == "1";
        if (mhs_timers) {   // STRATA_RUNNER_MHC_PHASES=1: the phase clocks above by pass size (mhs<T bucket>.*)
            const std::string b = "mhs" + std::to_string((T + 3) / 4 * 4) + ".";
            const int nth = omp_get_max_threads();
            double mx = 0, mean = 0;
            for (int i = 0; i < nth && i < 64; ++i) mx = std::max(mx, sm_pb[i]), mean += sm_pb[i] / nth;
            static const char *ph[5] = {"a_proj", "b_cmg", "c_coef", "d_comb", "e_invx"};
            double prev = t_reg;
            for (int i = 0; i < 5; ++i) timers[b + ph[i]] += sm_ts[i] - prev, prev = sm_ts[i];
            timers[b + "f_out"] += sm_ts[5] - prev;
            timers[b + "setup"] += t_reg - t0, timers[b + "post"] += now() - sm_ts[5];
            double wmean = 0;
            for (int i = 0; i < nth && i < 64; ++i) wmean += sm_wb[i] / nth;
            timers[b + "proj_max"] += mx, timers[b + "proj_mean"] += mean, timers[b + "proj_write"] += wmean;
            timers[b + "calls"] += 1e-3;
        }
        return res;
    }
    Vec r_next;   // write()'s output buffer, swapped with the residual (no allocation or zero fill per call)
    void write(Vec &r, const Vec &c, const Vec &y) {
        if (r_next.size() != r.size()) r_next.resize(r.size());
        Vec &out = r_next;
        team_run([&] {
#pragma omp for collapse(2) schedule(static) nowait
            for (int j = 0; j < 4; ++j)
                for (int d = 0; d < m.hidden; ++d) {
                    float v = c[4 + j] * y[d];
                    for (int i = 0; i < 4; ++i) v += c[8 + i * 4 + j] * r[i * m.hidden + d];
                    out[j * m.hidden + d] = v;
                }
        });
        r.swap(r_next);
    }
    // the KDA layer in one team region (kda() below forks six times with serial steps between): q/k/v/f_a/g_a/beta,
    // f_b/g_b, conv + SiLU and the decay gate, the recurrence, the output norm and gate, o_proj. Same arithmetic per
    // element as kda(), so the results are bitwise equal (STRATA_RUNNER_KDA_FUSED=0 keeps kda()).
    KdaScratch kda_w;
    Vec kda_fused(KdaLayer &K, const Vec &x) {
        const double t_in = now();
        const int D = m.linear_dim, nh = heads.size(), n = nh * D, Kc = m.conv_kernel;
        Activation ax, af, afa, aga, ay;
        ax.set(x.data(), m.hidden, trunk_q8);
        af.set(x.data(), m.hidden, false);
        std::array<Vec, 3> qkv, raws;
        for (auto &r : raws) r.resize(n);
        for (auto &v : qkv) v.resize(n);
        afa.x.resize(K.f_a.rows), aga.x.resize(K.g_a.rows), ay.x.resize(n);
        const bool oq = needs_q8(K.o);
        if (oq) ay.q.resize(n), ay.qs.resize(n / 64), ay.group = 64;
        Vec beta_all(K.beta.rows), decay(n), gate(n), y(n), beta(nh), out(K.o.rows);
        std::vector<float> inv(nh);
        GemvPlan p1({{&K.q, &ax, raws[0].data()}, {&K.k, &ax, raws[1].data()}, {&K.v, &ax, raws[2].data()},
                     {&K.f_a, &af, afa.x.data()}, {&K.g_a, &af, aga.x.data()}, {&K.beta, &af, beta_all.data()}});
        GemvPlan p2({{&K.f_b, &afa, decay.data()}, {&K.g_b, &aga, gate.data()}});
        GemvPlan p3({{&K.o, &ay, out.data()}});
        double tp[5] = {0, 0, 0, 0, 0};
        team_run([&] {
            const int t = omp_get_thread_num(), nth = omp_get_num_threads();
            p1.run(t, nth);
            team_barrier();
            if (t == 0) tp[0] = now();
            p2.run(t, nth);
            prefetch_share(p3.jobs, t, nth);   // o_proj into L2 during the conv and the recurrence
            team_barrier();
            if (t == 0) tp[1] = now();
#pragma omp for schedule(static) nowait
            for (int pi = 0; pi < 4 * n; ++pi) {   // per-channel conv1d + SiLU of q, k, v and the decay gate
                const int part = pi / n, i = pi % n;
                if (part == 3) {
                    decay[i] = m.gate_lower_bound * ref::sigmoid(-K.a_neg[i / D] * (decay[i] + K.dt[i]));
                    continue;
                }
                const Vec &raw = raws[part];
                const auto &w = K.conv[part];
                auto &hist = K.hist[part];
                float sum = raw[i] * w[i * Kc + Kc - 1];
                for (int j = 0; j < Kc - 1; ++j) sum += hist[i * (Kc - 1) + j] * w[i * Kc + j];
                for (int j = 0; j < Kc - 2; ++j) hist[i * (Kc - 1) + j] = hist[i * (Kc - 1) + j + 1];
                hist[i * (Kc - 1) + Kc - 2] = raw[i];
                qkv[part][i] = sum * ref::sigmoid(sum);
            }
            team_barrier();
            if (t == 0) tp[2] = now();
#pragma omp single nowait
            for (int h = 0; h < nh; ++h) beta[h] = beta_all[heads.begin + h];
            kda_core_team(kda_w, K.state, K.state, qkv[0].data(), qkv[1].data(), qkv[2].data(), decay.data(), beta.data(), y.data(), nh, D);
            if (t == 0) tp[3] = now();
            for (int h = t; h < nh; h += nth) {   // heads over the team (the same order per head)
                double sum = 0;
                for (int i = 0; i < D; ++i) sum += double(y[h * D + i]) * y[h * D + i];
                inv[h] = float(1 / std::sqrt(sum / D + m.rms_epsilon));
            }
            team_barrier();
#pragma omp for schedule(static) nowait
            for (int b = 0; b < n / 64; ++b) {   // output norm and gate per 64-value block (whole int8 blocks for o_proj)
                for (int j = b * 64; j < b * 64 + 64; ++j) ay.x[j] = y[j] * inv[j / D] * K.onorm[j % D] * ref::sigmoid(gate[j]);
                if (oq) gf4_quantize_q8(ay.x.data() + b * 64, 64, ay.q.data() + b * 64, ay.qs.data() + b);
            }
            team_barrier();
            if (t == 0) tp[4] = now();
            p3.run(t, nth);
        });
        const double t_out = now();
        timers["kda.gemv1"] += tp[0] - t_in;
        timers["kda.gemv2"] += tp[1] - tp[0];
        timers["kda.conv"] += tp[2] - tp[1];
        timers["kda.core"] += tp[3] - tp[2];
        timers["kda.core_sums"] += g_kda_tp[0] - tp[2];
        timers["kda.core_norm"] += g_kda_tp[1] - g_kda_tp[0];
        timers["kda.gate"] += tp[4] - tp[3];
        timers["kda.o"] += t_out - tp[4];
        timers["kda_proj"] += (tp[2] - t_in) + (t_out - tp[4]);
        timers["kda_small"] += tp[4] - tp[2];
        return out;
    }
    bool kda_fusable(const KdaLayer &K) const {
        static const bool on = !(std::getenv("STRATA_RUNNER_KDA_FUSED") && std::string(std::getenv("STRATA_RUNNER_KDA_FUSED")) == "0");
        const int n = int(heads.size()) * int(m.linear_dim);
        return on && n % 64 == 0 && K.o.cols == n;
    }
    Vec kda(KdaLayer &K, const Vec &x) {
        if (kda_fusable(K)) return kda_fused(K, x);
        double t = now();
        const int D = m.linear_dim, nh = heads.size(), n = nh * D, Kc = m.conv_kernel;
        Activation ax, af;
        ax.set(x.data(), m.hidden, trunk_q8);
        af.set(x.data(), m.hidden, false);
        std::array<Vec, 3> qkv, raws;
        for (auto &r : raws) r.resize(n);
        Vec fa(K.f_a.rows), ga(K.g_a.rows), beta_all(K.beta.rows);
        double tl = t;
        gemv_multi({{&K.q, &ax, raws[0].data()}, {&K.k, &ax, raws[1].data()}, {&K.v, &ax, raws[2].data()},
                    {&K.f_a, &af, fa.data()}, {&K.g_a, &af, ga.data()}, {&K.beta, &af, beta_all.data()}});
        lap(tl, "kda.gemv1");
        Activation afa, aga;
        afa.set(fa.data(), int(fa.size()), false);
        aga.set(ga.data(), int(ga.size()), false);
        Vec decay(n), gate(n);
        gemv_multi({{&K.f_b, &afa, decay.data()}, {&K.g_b, &aga, gate.data()}});
        lap(tl, "kda.gemv2");
        for (auto &v : qkv) v.resize(n);
        // per-channel conv1d + SiLU of q, k, v and the decay gate: independent channels over the team
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (int pi = 0; pi < 4 * n; ++pi) {
                const int part = pi / n, i = pi % n;
                if (part == 3) {
                    decay[i] = m.gate_lower_bound * ref::sigmoid(-K.a_neg[i / D] * (decay[i] + K.dt[i]));
                    continue;
                }
                const Vec &raw = raws[part];
                const auto &w = K.conv[part];
                auto &hist = K.hist[part];
                float sum = raw[i] * w[i * Kc + Kc - 1];
                for (int j = 0; j < Kc - 1; ++j) sum += hist[i * (Kc - 1) + j] * w[i * Kc + j];
                for (int j = 0; j < Kc - 2; ++j) hist[i * (Kc - 1) + j] = hist[i * (Kc - 1) + j + 1];
                hist[i * (Kc - 1) + Kc - 2] = raw[i];
                qkv[part][i] = sum * ref::sigmoid(sum);
            }
        });
        Vec beta(beta_all.begin() + heads.begin, beta_all.begin() + heads.end);
        lap(tl, "kda.conv");
        timers["kda_proj"] += now() - t;
        t = now();
        Vec y(n);
        tl = t;
        kda_parallel(K.state, qkv[0].data(), qkv[1].data(), qkv[2].data(), decay.data(), beta.data(), y.data(), nh, D);
        lap(tl, "kda.core");
        std::vector<float> inv(nh);
        for (int h = 0; h < nh; ++h) {
            double sum = 0;
            for (int i = 0; i < D; ++i) sum += double(y[h * D + i]) * y[h * D + i];
            inv[h] = float(1 / std::sqrt(sum / D + m.rms_epsilon));
        }
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (int j = 0; j < n; ++j) y[j] = y[j] * inv[j / D] * K.onorm[j % D] * ref::sigmoid(gate[j]);
        });
        timers["kda_small"] += now() - t;
        t = now();
        auto out = gemv(K.o, y);
        timers["kda_proj"] += now() - t;
        return out;
    }
    // one token's MLA attention at position `pos` (index keys and pooling, pool selection, cache append, the absorbed
    // attention over the selected positions): its values for o_proj. Shared by the one-token and the multi-token
    // passes; tokens of one pass call it in position order.
    // one token's index step at position `pos` (keys and pooling, pool selection) and its cache row: the positions
    // its attention reads. Tokens of a pass call it in position order.
    std::vector<int> mla_select(MlaLayer &M, const Vec &iq, const Vec &iw, const Vec &ik, const Vec &gate, const Vec &row, int pos) {
        if (pos >= capacity) throw std::runtime_error("runner: position " + std::to_string(pos) + " beyond the context capacity " + std::to_string(capacity));
        double t = now(), tl = t;
        const int dim = m.index_dim, P = m.index_pool, tokens = pos + 1, pools = tokens / P, tail = pos % P;
        double mean = 0, var = 0;
        for (float v : ik) mean += v;
        mean /= dim;
        for (float v : ik) var += (v - mean) * (v - mean);
        const float inv = float(1 / std::sqrt(var / dim + 1e-6));
        for (int i = 0; i < dim; ++i) {
            M.keys[tail * dim + i] = float(ik[i] - mean) * inv * M.k_norm_w[i] + M.k_norm_b[i];
            M.gates[tail * dim + i] = gate[i];
        }
        if (tokens % P == 0) ref::pool(M.keys.data(), M.gates.data(), M.ape.data(), M.pooled.data() + (pools - 1) * dim, P, dim);
        Vec scores(pools);
        // while every complete pool is kept (contexts up to the selection size) the scores choose nothing: skipped
        if (pools > int(m.index_top_k / P) || !index_skip()) index_scores(iq.data(), iw.data(), M.pooled.data(), pools, int(m.index_heads), dim, scores.data());
        // the top index_top_k / P pools by score, ties to the lower index, in ascending order: the same set as a stable
        // sort by descending score cut to that length (a serial sort of up to 2048 pools was ~200 us per MLA layer at
        // 8K context); nth_element finds the last pool kept, one pass collects the pools at least as good
        const int keep = std::min(int(m.index_top_k / P), pools);
        std::vector<int> order;
        order.reserve(keep);
        if (keep == pools) {
            for (int b = 0; b < pools; ++b) order.push_back(b);
        } else if (keep > 0) {
            auto better = [&](int a1, int a2) { return scores[a1] > scores[a2] || (scores[a1] == scores[a2] && a1 < a2); };
            std::vector<int> idx(pools);
            std::iota(idx.begin(), idx.end(), 0);
            std::nth_element(idx.begin(), idx.begin() + (keep - 1), idx.end(), better);
            const int last = idx[keep - 1];
            for (int b = 0; b < pools; ++b)
                if (!better(last, b)) order.push_back(b);
        }
        std::vector<int> ids;
        for (int b : order)
            for (int i = 0; i < P; ++i) ids.push_back(b * P + i);
        for (int i = pools * P; i < tokens; ++i) ids.push_back(i);
        tl = t;
        lap(tl, "mla.index");
        cache_put(M, pos, row.data());
        if (M.ring && pos >= M.ring) throw std::runtime_error("runner: the hybrid cache needs the batched indexer (STRATA_RUNNER_INDEX_BATCH=1)");
        return ids;
    }
    // mla_select for T consecutive positions (iq, iw, ik, gate: [T][rows]): the key, pool and cache updates in position
    // order (`after(tk)` after each: the caller's snapshot), then every position's pool scores in one team region and
    // the selections one position per thread. Each position scores the pools complete at its position, as T calls
    // of mla_select would (the same values: a pool does not change once complete)
    struct LayerState;
    // st: a context's index state and cache instead of the layer's (a several-context pass selecting all contexts in
    // parallel; small passes within the selection size only: no nested team regions)
    std::vector<std::vector<int>> mla_select_multi(MlaLayer &M, const Vec &iq, const Vec &iw, const Vec &ik, const Vec &gate,
                                                   const std::vector<Vec> &row, int pos0, const std::function<void(int)> &after,
                                                   LayerState *st = nullptr) {
        double t = now(), tl = t;
        Vec &Mkeys = st ? st->keys : M.keys, &Mgates = st ? st->gates : M.gates, &Mpooled = st ? st->pooled : M.pooled;
        auto put = [&](int pos, const float *r) {
            if (!st) { cache_put(M, pos, r); return; }
            std::copy(r, r + m.kv_rank, st->cache.begin() + size_t(pos) * m.kv_rank);
        };
        const int T = int(row.size()), dim = m.index_dim, P = m.index_pool, IH = int(m.index_heads);
        const int iq_r = int(iq.size()) / T, iw_r = int(iw.size()) / T;
        if (int(ik.size()) != T * dim || int(gate.size()) != T * dim) throw std::runtime_error("runner: indexer key size");
        if (pos0 + T > capacity) throw std::runtime_error("runner: positions up to " + std::to_string(pos0 + T - 1) + " beyond the context capacity " + std::to_string(capacity));
        std::vector<int> pools(T);
        // a long pass without snapshots: the keys, cache rows and complete pools over the team (each value computed as
        // below; a pool from its P keys in ring order, the ring left as the loop below leaves it).
        // STRATA_RUNNER_INDEX_KEYS_PAR=0: the loop below
        static const bool keys_par = !(std::getenv("STRATA_RUNNER_INDEX_KEYS_PAR") && std::string(std::getenv("STRATA_RUNNER_INDEX_KEYS_PAR")) == "0");
        const bool kpar = keys_par && !after && T > 8;
        if (kpar) {
            std::vector<float> Kt(size_t(T) * dim), Gt(size_t(T) * dim);
            std::vector<int> ends;   // positions that complete a pool
            for (int tk = 0; tk < T; ++tk) {
                pools[tk] = (pos0 + tk + 1) / P;
                if ((pos0 + tk + 1) % P == 0) ends.push_back(pos0 + tk);
            }
            team_run([&] {
#pragma omp for schedule(static) nowait
                for (int tk = 0; tk < T; ++tk) {
                    const float *ikt = ik.data() + size_t(tk) * dim, *gt = gate.data() + size_t(tk) * dim;
                    double mean = 0, var = 0;
                    for (int i = 0; i < dim; ++i) mean += ikt[i];
                    mean /= dim;
                    for (int i = 0; i < dim; ++i) var += (ikt[i] - mean) * (ikt[i] - mean);
                    const float inv = float(1 / std::sqrt(var / dim + 1e-6));
                    for (int i = 0; i < dim; ++i) {
                        Kt[size_t(tk) * dim + i] = float(ikt[i] - mean) * inv * M.k_norm_w[i] + M.k_norm_b[i];
                        Gt[size_t(tk) * dim + i] = gt[i];
                    }
                    put(pos0 + tk, row[tk].data());
                }
                team_barrier();
                std::vector<float> kb(size_t(P) * dim), gb(size_t(P) * dim);
#pragma omp for schedule(static) nowait
                for (int e = 0; e < int(ends.size()); ++e) {
                    const int last = ends[e], first = last + 1 - P;
                    for (int ps = first; ps <= last; ++ps) {   // ring slot ps % P: this pass's key, or the ring's (earlier)
                        const int sl = ps % P;
                        const float *ks = ps >= pos0 ? Kt.data() + size_t(ps - pos0) * dim : Mkeys.data() + size_t(sl) * dim;
                        const float *gs = ps >= pos0 ? Gt.data() + size_t(ps - pos0) * dim : Mgates.data() + size_t(sl) * dim;
                        std::copy(ks, ks + dim, kb.begin() + size_t(sl) * dim);
                        std::copy(gs, gs + dim, gb.begin() + size_t(sl) * dim);
                    }
                    ref::pool(kb.data(), gb.data(), M.ape.data(), Mpooled.data() + size_t((last + 1) / P - 1) * dim, P, dim);
                }
            });
            for (int tk = std::max(0, T - P); tk < T; ++tk) {   // the ring as the loop leaves it
                const int sl = (pos0 + tk) % P;
                std::copy(Kt.begin() + size_t(tk) * dim, Kt.begin() + size_t(tk + 1) * dim, Mkeys.begin() + size_t(sl) * dim);
                std::copy(Gt.begin() + size_t(tk) * dim, Gt.begin() + size_t(tk + 1) * dim, Mgates.begin() + size_t(sl) * dim);
            }
        }
        for (int tk = 0; tk < T && !kpar; ++tk) {
            const int pos = pos0 + tk, tokens = pos + 1, tail = pos % P;
            const float *ikt = ik.data() + size_t(tk) * dim, *gt = gate.data() + size_t(tk) * dim;
            double mean = 0, var = 0;
            for (int i = 0; i < dim; ++i) mean += ikt[i];
            mean /= dim;
            for (int i = 0; i < dim; ++i) var += (ikt[i] - mean) * (ikt[i] - mean);
            const float inv = float(1 / std::sqrt(var / dim + 1e-6));
            for (int i = 0; i < dim; ++i) {
                Mkeys[tail * dim + i] = float(ikt[i] - mean) * inv * M.k_norm_w[i] + M.k_norm_b[i];
                Mgates[tail * dim + i] = gt[i];
            }
            pools[tk] = tokens / P;
            if (tokens % P == 0) ref::pool(Mkeys.data(), Mgates.data(), M.ape.data(), Mpooled.data() + (pools[tk] - 1) * dim, P, dim);
            put(pos, row[tk].data());
            if (after) after(tk);
        }
        lap(tl, "mla.ix_keys");   // (mla.index_scores now from here)
        std::vector<Vec> scores(T);
        for (int tk = 0; tk < T; ++tk) scores[tk].resize(pools[tk]);
        // the tokens whose selection keeps every complete pool need no scores (mla_select's skip)
        std::vector<int> spools(pools);
        if (index_skip())
            for (int tk = 0; tk < T; ++tk)
                if (pools[tk] <= int(m.index_top_k / P)) spools[tk] = 0;
        // long contexts (STRATA_RUNNER_INDEX_SHARD_MIN complete pools): the pools in
        // 32-pool blocks over the ranks; each rank scores its blocks (the same kernels per pool), keeps its best
        // index_top_k / P, and the candidates of all ranks are merged by the same order. Every pool of the global
        // selection is among its owner's best, so the selection is the unsharded one
        // (default: 4096 pools = 16K positions in long passes, where the exchange is shared by many tokens, 16384 pools
        // = 64K positions in verification passes)
        static const int shard_env = std::getenv("STRATA_RUNNER_INDEX_SHARD_MIN") ? std::atoi(std::getenv("STRATA_RUNNER_INDEX_SHARD_MIN")) : -1;
        const int shard_min = shard_env >= 0 ? shard_env : T > 4 ? 4096 : 16384;
        const int keepn = int(m.index_top_k / P);
        std::vector<char> shard(T, 0);
        bool any_shard = false;
        for (int tk = 0; tk < T; ++tk)
            if (comm.world > 1 && shard_min > 0 && spools[tk] > keepn && spools[tk] >= shard_min) shard[tk] = 1, any_shard = true;
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
        if (IH == 32) {
            std::vector<long> off(T + 1, 0);                 // pair items per position
            for (int tk = 0; tk < T; ++tk) off[tk + 1] = off[tk] + (shard[tk] ? 0 : (spools[tk] + 1) / 2);   // sharded tokens: own_items only
            // [t][d][h], only for the tokens that score (none in a pass within the selection size)
            std::vector<double> iqt;
            int n_score = 0;
            for (int tk = 0; tk < T; ++tk) n_score += spools[tk] > 0;
            if (n_score) {
                iqt.resize(size_t(T) * dim * 32);
                auto tr = [&](int tk) {
                    if (!spools[tk]) return;
                    const float *q = iq.data() + size_t(tk) * iq_r;
                    for (int h = 0; h < 32; ++h)
                        for (int d = 0; d < dim; ++d) iqt[(size_t(tk) * dim + d) * 32 + h] = double(q[h * dim + d]);
                };
                if (T < 8) for (int tk = 0; tk < T; ++tk) tr(tk);
                else
                    team_run([&] {
#pragma omp for schedule(static) nowait
                        for (int tk = 0; tk < T; ++tk) tr(tk);
                    });
            }
            const float rs = std::sqrt(float(dim)), rh = std::sqrt(float(IH));
            std::vector<std::pair<int, int>> own_items;   // sharded tokens: (token, first pool) of this rank's pairs
            if (any_shard)   // this rank's 32-pool blocks only, 16 pairs each
                for (int tk = 0; tk < T; ++tk)
                    for (int b0 = 32 * comm.rank; shard[tk] && b0 < spools[tk]; b0 += 32 * comm.world)
                        for (int b = b0; b < std::min(spools[tk], b0 + 32); b += 2) own_items.push_back({tk, b});
            auto item = [&](long gi) {
                int tk = int(std::upper_bound(off.begin(), off.end(), gi) - off.begin()) - 1, b0 = int(2 * (gi - off[tk]));
                if (tk >= T) { const auto &it = own_items[gi - off[T]]; tk = it.first, b0 = it.second; }
                index_pair32(iqt.data() + size_t(tk) * dim * 32, iw.data() + size_t(tk) * iw_r, Mpooled.data(), b0, spools[tk], dim, rs, rh,
                             scores[tk].data());
            };
            off.push_back(off[T] + long(own_items.size()));   // the sharded items after every token's own range
            const long n_items = off[T + 1];
            if (n_items < 32)
                for (long gi = 0; gi < n_items; ++gi) item(gi);
            else
                team_run([&] {
#pragma omp for schedule(static) nowait
                    for (long gi = 0; gi < n_items; ++gi) item(gi);
                });
        } else
#endif
            for (int tk = 0; tk < T; ++tk) {
                if (!shard[tk]) {
                    index_scores(iq.data() + size_t(tk) * iq_r, iw.data() + size_t(tk) * iw_r, Mpooled.data(), spools[tk], IH, dim, scores[tk].data());
                    continue;
                }
                for (int b0 = 32 * comm.rank; b0 < spools[tk]; b0 += 32 * comm.world)   // this rank's 32-pool blocks
                        index_scores(iq.data() + size_t(tk) * iq_r, iw.data() + size_t(tk) * iw_r, Mpooled.data() + size_t(b0) * dim, std::min(32, spools[tk] - b0), IH, dim,
                                     scores[tk].data() + b0);
            }
        lap(tl, "mla.index_scores");
        std::vector<std::vector<int>> merged(T);   // sharded tokens: the selected pools in ascending order
        if (any_shard) {
            using Cand = std::pair<float, int>;   // (score, pool); pool -1: none
            std::vector<Cand> mine(size_t(T) * keepn, Cand{-INFINITY, -1}), all(size_t(comm.world) * T * keepn);
            auto local_best = [&](int tk) {
                if (!shard[tk]) return;
                const Vec &sc = scores[tk];
                std::vector<int> idx;
                for (int b0 = 32 * comm.rank; b0 < spools[tk]; b0 += 32 * comm.world)
                    for (int b = b0; b < std::min(spools[tk], b0 + 32); ++b) idx.push_back(b);
                auto better = [&](int a1, int a2) { return sc[a1] > sc[a2] || (sc[a1] == sc[a2] && a1 < a2); };
                const int k = std::min<int>(keepn, int(idx.size()));
                if (k > 0 && k < int(idx.size())) std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), better);
                for (int i = 0; i < k; ++i) mine[size_t(tk) * keepn + i] = {sc[idx[i]], idx[i]};
            };
            team_run([&] {
#pragma omp for schedule(static, 1) nowait
                for (int tk = 0; tk < T; ++tk) local_best(tk);
            });
            const double tc = now();
            comm.allgather(mine.data(), int(mine.size() * sizeof(Cand)), all.data());
            timers["comm_index"] += now() - tc;
            team_run([&] {
#pragma omp for schedule(static, 1) nowait
                for (int tk = 0; tk < T; ++tk) {
                    if (!shard[tk]) continue;
                    std::vector<Cand> c;
                    for (int r = 0; r < comm.world; ++r)
                        for (int i = 0; i < keepn; ++i)
                            if (const Cand &e = all[(size_t(r) * T + tk) * keepn + i]; e.second >= 0) c.push_back(e);
                    auto better = [](const Cand &a, const Cand &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
                    if (int(c.size()) > keepn) std::nth_element(c.begin(), c.begin() + (keepn - 1), c.end(), better), c.resize(keepn);
                    auto &o = merged[tk];
                    for (const auto &e : c) o.push_back(e.second);
                    std::sort(o.begin(), o.end());
                }
            });
        }
        // the top index_top_k / P pools by score, ties to the lower index, in ascending order (mla_select's selection)
        std::vector<std::vector<int>> sel(T);
        auto select = [&](int tk) {
            const int np = pools[tk], keep = std::min(int(m.index_top_k / P), np);
            const Vec &sc = scores[tk];
            std::vector<int> order;
            order.reserve(keep);
            if (shard[tk]) order = merged[tk];
            else if (keep == np) {
                for (int b = 0; b < np; ++b) order.push_back(b);
            } else if (keep > 0) {
                auto better = [&](int a1, int a2) { return sc[a1] > sc[a2] || (sc[a1] == sc[a2] && a1 < a2); };
                std::vector<int> idx(np);
                std::iota(idx.begin(), idx.end(), 0);
                std::nth_element(idx.begin(), idx.begin() + (keep - 1), idx.end(), better);
                const int last = idx[keep - 1];
                for (int b = 0; b < np; ++b)
                    if (!better(last, b)) order.push_back(b);
            }
            auto &ids = sel[tk];
            const int tokens = pos0 + tk + 1;
            ids.reserve(size_t(order.size()) * P + P);
            for (int b : order)
                for (int i = 0; i < P; ++i) ids.push_back(b * P + i);
            for (int i = np * P; i < tokens; ++i) ids.push_back(i);
        };
        if (T > 8 || (T > 1 && pools[T - 1] > int(m.index_top_k / P)))   // one position per thread (a long pass's lists of
                                                                          // up to index_top_k positions, or nth_element)
            team_run([&] {
#pragma omp for schedule(static, 1) nowait
                for (int tk = 0; tk < T; ++tk) select(tk);
            });
        else
            for (int tk = 0; tk < T; ++tk) select(tk);
        tl = t;
        lap(tl, "mla.index");
        return sel;
    }
    // the absorbed attention of T tokens (q[t], the positions ids[t] each reads): per head one GEMV call for all
    // tokens (K and V absorbed read once), the score, softmax and average passes over every (token, head) at once;
    // the same arithmetic per (token, head) as one token at a time. values[t]: heads x head dim.
    const float *const *mla_tcache = nullptr;   // see mla_attention
    Vec ma_aq, ma_val, ma_av;   // mla_attention's work arrays
    std::vector<Activation> ma_qh, ma_avs;   // and its per (token, head) activations (their x kept allocated)
    void mla_attention(MlaLayer &M, const std::vector<const Vec *> &q, const std::vector<std::vector<int>> &ids, std::vector<Vec> &values) {
        double t0 = now(), tl = t0;
        const int T = int(q.size()), lat = m.kv_rank, nh = heads.size(), hd = M.kb.empty() ? 0 : M.kb[0].cols;
        const float scale = 1 / std::sqrt(float(hd));
        // the per (token, head) activations persistent and filled by the team (STRATA_RUNNER_MLA_ACTS_PAR=0: fresh
        // ones per call, filled by thread 0; at 16 contexts ~110 us per expand call)
        static const bool acts_par = !(std::getenv("STRATA_RUNNER_MLA_ACTS_PAR") && std::string(std::getenv("STRATA_RUNNER_MLA_ACTS_PAR")) == "0");
        std::vector<Activation> qh_fresh, avs_fresh;
        if (!acts_par) qh_fresh.resize(size_t(T) * nh), avs_fresh.resize(size_t(T) * nh);
        else if (ma_qh.size() < size_t(T) * nh) ma_qh.resize(size_t(T) * nh), ma_avs.resize(size_t(T) * nh);
        auto &qh = acts_par ? ma_qh : qh_fresh;
        auto &avs = acts_par ? ma_avs : avs_fresh;
        auto fill = [&](std::vector<Activation> &dst, auto &&src_of, int len) {   // dst[t * nh + h] = src_of(t, h)[0, len)
            if (acts_par && T > 4)
                team_run([&] {
#pragma omp for schedule(static) nowait
                    for (int i = 0; i < T * nh; ++i) dst[i].set(src_of(i / nh, i % nh), len, false);
                });
            else
                for (int i = 0; i < T * nh; ++i) dst[i].set(src_of(i / nh, i % nh), len, false);
        };
        // [h][t][lat], [t][h][hd], [t][h][lat]: persistent (each element is written before it is read; a zero-filled
        // allocation per call cost ~0.08 ms per generated token at 16 contexts)
        auto &aq_all = ma_aq, &val_all = ma_val, &av_all = ma_av;
        if (aq_all.size() < size_t(nh) * T * lat) aq_all.resize(size_t(nh) * T * lat);
        if (val_all.size() < size_t(T) * nh * hd) val_all.resize(size_t(T) * nh * hd);
        if (av_all.size() < size_t(T) * nh * lat) av_all.resize(size_t(T) * nh * lat);
        std::vector<Job> absorb;
        fill(qh, [&](int t, int h) { return q[t]->data() + size_t(h) * hd; }, hd);
        for (int h = 0; h < nh; ++h) {
            Job j{&M.kb[h], nullptr, aq_all.data() + size_t(h) * T * lat, T, lat};
            for (int t = 0; t < T; ++t) j.as[t] = &qh[size_t(t) * nh + h];
            absorb.push_back(std::move(j));
        }
        lap(tl, "mla.qprep");
        gemv_multi(absorb);
        lap(tl, "mla.absorb");
        std::vector<long> S(T), poff(T + 1, 0), boff(T + 1, 0);   // positions, prob offsets, score item offsets
        for (int t = 0; t < T; ++t) {
            S[t] = long(ids[t].size());
            poff[t + 1] = poff[t] + long(nh) * S[t];
            boff[t + 1] = boff[t] + ((S[t] + 7) / 8) * nh;
        }
        std::vector<double> prob(poff[T]), mx(size_t(T) * nh, -1e300), total(size_t(T) * nh, 0.0);
        // per-thread maxima of the scores (order-free), one 256-byte line per thread and (token, head) group of 32
        const int nth_max = omp_get_max_threads(), tnh = T * nh, mrow = (tnh + 31) / 32 * 32;
        std::vector<double> pmx(size_t(nth_max) * mrow, -1e300);
        // a long pass where every token reads all positions up to its own (contexts within the selection size): items of
        // 4 tokens x 4 positions per head, so each cache row and query is loaded once for 4 dots (the dots as below:
        // per lane in order over i, then svaddv)
        // mla_tcache (a pass over several contexts): token t's latent cache at mla_tcache[t]; the prefix and union
        // kernels share cache rows between tokens, so such a pass takes the per-token (gather) kernels
        auto cbase = [&](int t) { return mla_tcache ? mla_tcache[t] : M.cache.data(); };
        bool prefix = T > 4 && lat % 8 == 0 && !mla_tcache;
        for (int t = 0; t < T && prefix; ++t)
            prefix = !ids[t].empty() && ids[t].front() == 0 && ids[t].back() + 1 == int(ids[t].size());
        // a long pass beyond the selection size: groups of 4 tokens walk the sorted union of their selections, each
        // cache row read once for the tokens that selected it; every token still meets its positions in its own order
        // (STRATA_RUNNER_UNION_BLOCK=1, opt-in: bitwise, but slower at 8K, 155.5 vs 167.4 tok/s: the 4 tokens' unions
        // are much larger than one selection)
        static const bool union_block = std::getenv("STRATA_RUNNER_UNION_BLOCK") && std::string(std::getenv("STRATA_RUNNER_UNION_BLOCK")) == "1";
        const bool uni = !prefix && union_block && T > 4 && lat % 32 == 0 && !mla_tcache;
        const int ntg_u = (T + 3) / 4;
        std::vector<std::vector<int>> u_pos(uni ? ntg_u : 0), u_idx(uni ? ntg_u : 0);   // union rows; [a][u] index or -1
        if (uni)
            team_run([&] {
#pragma omp for schedule(dynamic, 1) nowait
                for (int tg = 0; tg < ntg_u; ++tg) {
                    const int t0 = 4 * tg, tn = std::min(4, T - t0);
                    auto &U = u_pos[tg];
                    for (int a = 0; a < tn; ++a) U.insert(U.end(), ids[t0 + a].begin(), ids[t0 + a].end());
                    std::sort(U.begin(), U.end());
                    U.erase(std::unique(U.begin(), U.end()), U.end());
                    auto &X = u_idx[tg];
                    X.assign(size_t(4) * U.size(), -1);
                    for (int a = 0; a < tn; ++a) {
                        const auto &v = ids[t0 + a];
                        size_t j = 0;
                        for (size_t u = 0; u < U.size() && j < v.size(); ++u)
                            if (U[u] == v[j]) X[size_t(a) * U.size() + u] = int(j++);
                    }
                }
            });
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
        static const bool score_block = !(std::getenv("STRATA_RUNNER_SCORE_BLOCK") && std::string(std::getenv("STRATA_RUNNER_SCORE_BLOCK")) == "0");
        if (uni) {   // 4 tokens x 4 union rows per item and head (the prefix blocks' dots)
            std::vector<long> uoff(ntg_u + 1, 0);
            for (int tg = 0; tg < ntg_u; ++tg) uoff[tg + 1] = uoff[tg] + long((u_pos[tg].size() + 3) / 4) * nh;
            team_run([&] {
                typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
                const svbool_t pd = svptrue_b64();
                const int th = omp_get_thread_num();
#pragma omp for schedule(static) nowait
                for (long gi = 0; gi < uoff[ntg_u]; ++gi) {
                    const int tg = int(std::upper_bound(uoff.begin(), uoff.end(), gi) - uoff.begin()) - 1;
                    const long bi = gi - uoff[tg];
                    const int h = int(bi % nh);
                    const auto &U = u_pos[tg];
                    const auto &X = u_idx[tg];
                    const long u0 = (bi / nh) * 4, nu = long(U.size());
                    const int t0 = 4 * tg, tn = std::min(4, T - t0);
                    const float *ap[4], *cp[4];
                    for (int a = 0; a < 4; ++a) ap[a] = aq_all.data() + (size_t(h) * T + t0 + std::min(a, tn - 1)) * lat;
                    for (int k = 0; k < 4; ++k) cp[k] = M.cache.data() + size_t(U[std::min<long>(u0 + k, nu - 1)]) * lat;
                    vf64 acc[4][4];
                    for (int a = 0; a < 4; ++a) for (int k = 0; k < 4; ++k) acc[a][k] = svdup_n_f64(0);
                    for (int i = 0; i < lat; i += 8) {
                        vf64 cv[4];
                        for (int k = 0; k < 4; ++k) cv[k] = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(cp[k] + i))));
#pragma GCC unroll 4
                        for (int a = 0; a < 4; ++a) {
                            const svfloat64_t qv = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(ap[a] + i))));
#pragma GCC unroll 4
                            for (int k = 0; k < 4; ++k) acc[a][k] = svmla_f64_x(pd, acc[a][k], qv, cv[k]);
                        }
                    }
                    for (int a = 0; a < tn; ++a) {
                        const int t = t0 + a;
                        double *pr = prob.data() + poff[t] + size_t(h) * S[t];
                        double &m_ = pmx[size_t(th) * mrow + size_t(t) * nh + h];
                        for (int k = 0; k < 4 && u0 + k < nu; ++k) {
                            const int si = X[size_t(a) * nu + u0 + k];
                            if (si < 0) continue;
                            pr[si] = svaddv_f64(pd, acc[a][k]) * scale, m_ = std::max(m_, pr[si]);
                        }
                    }
                }
            });
        } else
        if (prefix && score_block) {
            // an item: SCORE_PB consecutive 4-position blocks for the same 4 tokens and head (STRATA_RUNNER_SCORE_PB,
            // default 4): the queries stream from L2 once per SCORE_PB blocks instead of once per block (the kernel was
            // bound by that stream: 1 byte per MAC; with L1-resident operands it ran 1.7x faster). Each block as before
            static const int spb = std::getenv("STRATA_RUNNER_SCORE_PB") ? std::max(1, std::atoi(std::getenv("STRATA_RUNNER_SCORE_PB"))) : 4;
            const int ntg = (T + 3) / 4;
            const long npb = (S[T - 1] + 3) / 4, npg = (npb + spb - 1) / spb, items = npg * ntg * nh;
            team_run([&] {
                typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
                const svbool_t pd = svptrue_b64();
                const int th = omp_get_thread_num();
#pragma omp for schedule(static) nowait
                for (long gi = 0; gi < items; ++gi)
                for (long pb = (gi / (long(ntg) * nh)) * spb, pb1 = std::min(npb, pb + spb); pb < pb1; ++pb) {
                    const int tg = int((gi / nh) % ntg), h = int(gi % nh);
                    const int t0 = 4 * tg, tn = std::min(4, T - t0);
                    const long s0 = 4 * pb;
                    if (s0 >= S[t0 + tn - 1]) break;   // past the group's longest selection
                    const float *ap[4], *cp[4];
                    for (int a = 0; a < 4; ++a) ap[a] = aq_all.data() + (size_t(h) * T + t0 + std::min(a, tn - 1)) * lat;
                    for (int k = 0; k < 4; ++k) cp[k] = M.cache.data() + size_t(std::min<long>(s0 + k, S[t0 + tn - 1] - 1)) * lat;
                    vf64 acc[4][4];
                    for (int a = 0; a < 4; ++a) for (int k = 0; k < 4; ++k) acc[a][k] = svdup_n_f64(0);
                    for (int i = 0; i < lat; i += 8) {
                        vf64 cv[4];
                        for (int k = 0; k < 4; ++k) cv[k] = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(cp[k] + i))));
#pragma GCC unroll 4
                        for (int a = 0; a < 4; ++a) {
                            const svfloat64_t qv = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(ap[a] + i))));
#pragma GCC unroll 4
                            for (int k = 0; k < 4; ++k) acc[a][k] = svmla_f64_x(pd, acc[a][k], qv, cv[k]);
                        }
                    }
                    double hs[16];
                    if (tree_sum_on())
                        for (int g = 0; g < 2; ++g) {   // dots (2g, k), (2g + 1, k)
                            vf64t v[8];
                            for (int j = 0; j < 8; ++j) v[j] = acc[2 * g + j / 4][j % 4];
                            svst1_f64(pd, hs + 8 * g, hsum8_f64(v));
                        }
                    else
                        for (int a = 0; a < 4; ++a) for (int k = 0; k < 4; ++k) hs[4 * a + k] = svaddv_f64(pd, acc[a][k]);
                    for (int a = 0; a < tn; ++a) {
                        const int t = t0 + a;
                        const long St = S[t];
                        double *pr = prob.data() + poff[t] + size_t(h) * St;
                        double &m_ = pmx[size_t(th) * mrow + size_t(t) * nh + h];
                        for (int k = 0; k < 4 && s0 + k < St; ++k) pr[s0 + k] = hs[4 * a + k] * scale, m_ = std::max(m_, pr[s0 + k]);
                    }
                }
            });
        } else
#endif
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
        // a long pass beyond the selection size: per item one token x 3 heads x 4 positions, over GATHER_PB (4)
        // consecutive position blocks: each gathered cache row feeds 3 heads' dots in registers and the 3 query rows
        // stay in L1 across the blocks (the item below loaded both for every dot: 9 loads per 8 FMAs). Each dot as
        // below: 8 latent lanes in order over i, then svaddv. STRATA_RUNNER_GATHER_BLOCK=0: the item below
        if (gather_block_on() && !prefix && !attn_f32() && lat % 8 == 0 && T > 4) {
            constexpr int HG = 3, PB = 4;
            const int ng = (nh + HG - 1) / HG;
            std::vector<long> goff(T + 1, 0);   // items per token: position groups x head groups
            for (int t = 0; t < T; ++t) goff[t + 1] = goff[t] + long((S[t] + 4 * PB - 1) / (4 * PB)) * ng;
            team_run([&] {
                typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
                const svbool_t pd = svptrue_b64();
                const int th = omp_get_thread_num();
#pragma omp for schedule(static) nowait
                for (long gi = 0; gi < goff[T]; ++gi) {
                    const int t = int(std::upper_bound(goff.begin(), goff.end(), gi) - goff.begin()) - 1;
                    const long bi = gi - goff[t], St = S[t];
                    const int h0 = int(bi % ng) * HG, hn = std::min(HG, nh - h0);
                    const float *ap[HG];
                    for (int j = 0; j < HG; ++j) ap[j] = aq_all.data() + (size_t(h0 + std::min(j, hn - 1)) * T + t) * lat;
                    for (long s0 = (bi / ng) * 4 * PB, s1 = std::min(St, s0 + 4 * PB); s0 < s1; s0 += 4) {
                        const int cnt = int(std::min<long>(4, St - s0));
                        const float *cp[4];
                        for (int k = 0; k < 4; ++k) cp[k] = cbase(t) + size_t(ids[t][std::min(s0 + k, St - 1)]) * lat;
                        vf64 acc[HG][4];
                        for (int j = 0; j < HG; ++j) for (int k = 0; k < 4; ++k) acc[j][k] = svdup_n_f64(0);
                        for (int i = 0; i < lat; i += 8) {
                            vf64 cv[4];
                            for (int k = 0; k < 4; ++k) cv[k] = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(cp[k] + i))));
#pragma GCC unroll 3
                            for (int j = 0; j < HG; ++j) {
                                const svfloat64_t qv = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(ap[j] + i))));
#pragma GCC unroll 4
                                for (int k = 0; k < 4; ++k) acc[j][k] = svmla_f64_x(pd, acc[j][k], qv, cv[k]);
                            }
                        }
                        double hs[16];
                        if (tree_sum_on()) {   // dots (j, k) at 4 j + k: heads 0-1, then head 2 (its 4 sums twice)
                            vf64t v[8];
                            for (int j = 0; j < 8; ++j) v[j] = acc[j / 4][j % 4];
                            svst1_f64(pd, hs, hsum8_f64(v));
                            for (int j = 0; j < 8; ++j) v[j] = acc[2][j % 4];
                            svst1_f64(pd, hs + 8, hsum8_f64(v));
                        } else
                            for (int j = 0; j < HG; ++j) for (int k = 0; k < 4; ++k) hs[4 * j + k] = svaddv_f64(pd, acc[j][k]);
                        for (int j = 0; j < hn; ++j) {
                            const int h = h0 + j;
                            double *pr = prob.data() + poff[t] + size_t(h) * St;
                            double &m_ = pmx[size_t(th) * mrow + size_t(t) * nh + h];
                            for (int k = 0; k < cnt; ++k) pr[s0 + k] = hs[4 * j + k] * scale, m_ = std::max(m_, pr[s0 + k]);
                        }
                    }
                }
            });
        } else
#endif
        // eight (head, position) dots per item as eight independent chains (each in its order over i; a product of two
        // floats is exact in double); items position-block-major within a token, so the heads of the same positions run
        // on neighbouring threads and share a CMG's L2 for those cache rows
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (long gi = 0; gi < boff[T]; ++gi) {
                const int t = int(std::upper_bound(boff.begin(), boff.end(), gi) - boff.begin()) - 1;
                const long bi = gi - boff[t], St = S[t];
                const int h = int(bi % nh);
                const long s0 = (bi / nh) * 8;
                const int cnt = int(std::min<long>(8, St - s0));
                const float *cp[8];
                const float *ap = aq_all.data() + (size_t(h) * T + t) * lat;
                for (int k = 0; k < 8; ++k) cp[k] = cbase(t) + size_t(ids[t][std::min(s0 + k, St - 1)]) * lat;
                double *pr = prob.data() + poff[t] + size_t(h) * St;
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                if (attn_f32() && lat % 16 == 0) {   // approximate: fp32 dots, 16 latent values per vector
                    const svbool_t pw = svptrue_b32();
                    vf32 acc[8];
                    for (int k = 0; k < 8; ++k) acc[k] = svdup_n_f32(0);
                    for (int i = 0; i < lat; i += 16) {
                        const svfloat32_t qv = svld1_f32(pw, ap + i);
#pragma GCC unroll 8
                        for (int k = 0; k < 8; ++k) acc[k] = svmla_f32_x(pw, acc[k], qv, svld1_f32(pw, cp[k] + i));
                    }
                    double &m_ = pmx[size_t(omp_get_thread_num()) * mrow + size_t(t) * nh + h];
                    for (int k = 0; k < cnt; ++k) pr[s0 + k] = double(svaddv_f32(pw, acc[k])) * scale, m_ = std::max(m_, pr[s0 + k]);
                    continue;
                }
                // SVE: eight latent values per double vector, eight positions' chains, one final horizontal sum each (the
                // only reordering of these sums: lanes, then svaddv; their products are exact)
                if (lat % 8 == 0) {
                    typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
                    const svbool_t pd = svptrue_b64();
                    vf64 acc[8];
                    for (int k = 0; k < 8; ++k) acc[k] = svdup_n_f64(0);
                    for (int i = 0; i < lat; i += 8) {
                        const svfloat64_t qv = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(ap + i))));
#pragma GCC unroll 8
                        for (int k = 0; k < 8; ++k) {
                            const svfloat64_t cv = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(cp[k] + i))));
                            acc[k] = svmla_f64_x(pd, acc[k], qv, cv);
                        }
                    }
                    double &m_ = pmx[size_t(omp_get_thread_num()) * mrow + size_t(t) * nh + h];
                    for (int k = 0; k < cnt; ++k) pr[s0 + k] = svaddv_f64(pd, acc[k]) * scale, m_ = std::max(m_, pr[s0 + k]);
                    continue;
                }
#endif
                double acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
                for (int i = 0; i < lat; ++i)
                    for (int k = 0; k < 8; ++k) acc[k] += double(ap[i]) * cp[k][i];
                double &m_ = pmx[size_t(omp_get_thread_num()) * mrow + size_t(t) * nh + h];
                for (int k = 0; k < cnt; ++k) pr[s0 + k] = acc[k] * scale, m_ = std::max(m_, pr[s0 + k]);
            }
        });
        for (int u = 0; u < nth_max; ++u)   // the maximum does not depend on the order
            for (int th_ = 0; th_ < tnh; ++th_) mx[th_] = std::max(mx[th_], pmx[size_t(u) * mrow + th_]);
        if (fast_exp(8))
            team_run([&] {   // per (token, head) row: subtract the maximum, then the vector exp
#pragma omp for schedule(static) nowait
                for (int th_ = 0; th_ < tnh; ++th_) {
                    const int t = th_ / nh, h = th_ % nh;
                    double *ph = prob.data() + poff[t] + size_t(h) * S[t];
                    const double mv = mx[size_t(t) * nh + h];
                    for (long si = 0; si < S[t]; ++si) ph[si] -= mv;
                    gf4_vexp_f64(ph, ph, S[t]);
                }
            });
        else
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (long gi = 0; gi < poff[T]; ++gi) {
                const int t = int(std::upper_bound(poff.begin(), poff.end(), gi) - poff.begin()) - 1;
                const long h = (gi - poff[t]) / S[t];
                prob[gi] = std::exp(prob[gi] - mx[size_t(t) * nh + h]);
            }
        });
        team_run([&] {   // each (token, head) total in position order on one thread (was one thread for all)
#pragma omp for schedule(static) nowait
            for (int th_ = 0; th_ < tnh; ++th_) {
                const int t = th_ / nh, h = th_ % nh;
                double tot = 0;
                const double *ph = prob.data() + poff[t] + size_t(h) * S[t];
                for (long si = 0; si < S[t]; ++si) tot += ph[si];
                total[th_] = tot;
            }
        });
        lap(tl, "mla.score");
        // latent average in 64-latent blocks per (token, head): each item walks the selected cache rows contiguously
        // (per element the summation order over positions is unchanged); head-major items
        const int lb = lat / 64;
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
        // the prefix case (every token reads all positions up to its own): items of 4 tokens x 32 latents per head, each
        // cache row read once for the 4 tokens; per element the same products and order over positions as below
        static const bool avg_block = !(std::getenv("STRATA_RUNNER_AVG_BLOCK") && std::string(std::getenv("STRATA_RUNNER_AVG_BLOCK")) == "0");
        if (uni) {   // the union walk: 4 tokens x 3 heads x 16 latents per item, each union row read once for all of them
            const int nb16 = lat / 16, nhg = (nh + 2) / 3;
            team_run([&] {
                typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
                const svbool_t pd = svptrue_b64();
#pragma omp for schedule(static) nowait
                for (int it = 0; it < ntg_u * nhg * nb16; ++it) {
                    const int tg = it / (nhg * nb16), h0 = 3 * ((it / nb16) % nhg), i0 = (it % nb16) * 16;
                    const int t0 = 4 * tg, tn = std::min(4, T - t0), hn = std::min(3, nh - h0);
                    const auto &U = u_pos[tg];
                    const auto &X = u_idx[tg];
                    const size_t nu = U.size();
                    vf64 acc[4][3][2];
                    for (int a = 0; a < 4; ++a) for (int j = 0; j < 3; ++j) acc[a][j][0] = acc[a][j][1] = svdup_n_f64(0);
                    const double *ph[4][3];   // rows past the group's tokens / heads: clamped (their sums are not stored)
                    for (int a = 0; a < 4; ++a)
                        for (int j = 0; j < 3; ++j) {
                            const int t = t0 + std::min(a, tn - 1);
                            ph[a][j] = prob.data() + poff[t] + size_t(h0 + std::min(j, hn - 1)) * S[t];
                        }
                    for (size_t u = 0; u < nu; ++u) {
                        const float *c = M.cache.data() + size_t(U[u]) * lat + i0;
                        if (u + 8 < nu) __builtin_prefetch(M.cache.data() + size_t(U[u + 8]) * lat + i0, 0, 3);
                        const vf64 c0 = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(c))));
                        const vf64 c1 = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(c + 8))));
#pragma GCC unroll 4
                        for (int a = 0; a < 4; ++a) {
                            const int si = X[size_t(a) * nu + u];   // -1: not this token's (or past the group)
                            if (si < 0) continue;
#pragma GCC unroll 3
                            for (int j = 0; j < 3; ++j) {
                                const svfloat64_t pr = svdup_n_f64(ph[a][j][si]);
                                acc[a][j][0] = svmla_f64_x(pd, acc[a][j][0], pr, c0);
                                acc[a][j][1] = svmla_f64_x(pd, acc[a][j][1], pr, c1);
                            }
                        }
                    }
                    for (int a = 0; a < tn; ++a)
                        for (int j = 0; j < hn; ++j) {
                            const int t = t0 + a, h = h0 + j;
                            double sum[16];
                            svst1_f64(pd, sum, acc[a][j][0]);
                            svst1_f64(pd, sum + 8, acc[a][j][1]);
                            for (int i = 0; i < 16; ++i) av_all[(size_t(t) * nh + h) * lat + i0 + i] = float(sum[i] / total[size_t(t) * nh + h]);
                        }
                }
            });
        } else
        // the prefix average with each cache row segment shared by 2 heads (STRATA_RUNNER_AVG_HEADS=2, default):
        // 4 tokens x 2 heads x 16 latents per item (16 accumulators as 4 x 32), each chain as below
        if (prefix && avg_block && lat % 16 == 0 && avg_heads() == 2) {
            const int ntg = (T + 3) / 4, nhp = (nh + 1) / 2, nb16 = lat / 16;
            team_run([&] {
                typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
                const svbool_t pd = svptrue_b64();
#pragma omp for schedule(static) nowait
                for (int it = 0; it < ntg * nhp * nb16; ++it) {
                    const int tg = it / (nhp * nb16), h0 = ((it / nb16) % nhp) * 2, i0 = (it % nb16) * 16;
                    const int t0 = 4 * tg, tn = std::min(4, T - t0), hn = std::min(2, nh - h0);
                    vf64 acc[4][2][2];
                    for (int a = 0; a < 4; ++a) for (int j = 0; j < 2; ++j) acc[a][j][0] = acc[a][j][1] = svdup_n_f64(0);
                    const long smax = S[t0 + tn - 1];
                    const double *ph[4][2];
                    for (int a = 0; a < tn; ++a)
                        for (int j = 0; j < 2; ++j) ph[a][j] = prob.data() + poff[t0 + a] + size_t(h0 + std::min(j, hn - 1)) * S[t0 + a];
                    for (long si = 0; si < smax; ++si) {
                        const float *c = M.cache.data() + size_t(si) * lat + i0;
                        const vf64 c0 = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(c))));
                        const vf64 c1 = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(c + 8))));
                        for (int a = 0; a < tn; ++a) {
                            if (si >= S[t0 + a]) continue;
#pragma GCC unroll 2
                            for (int j = 0; j < 2; ++j) {
                                const svfloat64_t pr = svdup_n_f64(ph[a][j][si]);
                                acc[a][j][0] = svmla_f64_x(pd, acc[a][j][0], pr, c0);
                                acc[a][j][1] = svmla_f64_x(pd, acc[a][j][1], pr, c1);
                            }
                        }
                    }
                    for (int a = 0; a < tn; ++a)
                        for (int j = 0; j < hn; ++j) {
                            const int t = t0 + a, h = h0 + j;
                            double sum[16];
                            svst1_f64(pd, sum, acc[a][j][0]), svst1_f64(pd, sum + 8, acc[a][j][1]);
                            for (int i = 0; i < 16; ++i) av_all[(size_t(t) * nh + h) * lat + i0 + i] = float(sum[i] / total[size_t(t) * nh + h]);
                        }
                }
            });
        } else
        if (prefix && avg_block && lat % 32 == 0) {
            const int ntg = (T + 3) / 4, nb32 = lat / 32;
            team_run([&] {
                typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
                const svbool_t pd = svptrue_b64();
#pragma omp for schedule(static) nowait
                for (int it = 0; it < ntg * nh * nb32; ++it) {
                    const int tg = it / (nh * nb32), h = (it / nb32) % nh, i0 = (it % nb32) * 32;
                    const int t0 = 4 * tg, tn = std::min(4, T - t0);
                    vf64 acc[4][4];
                    for (int a = 0; a < 4; ++a) for (int v = 0; v < 4; ++v) acc[a][v] = svdup_n_f64(0);
                    const long smax = S[t0 + tn - 1];
                    const double *ph[4];
                    for (int a = 0; a < tn; ++a) ph[a] = prob.data() + poff[t0 + a] + size_t(h) * S[t0 + a];
                    for (long si = 0; si < smax; ++si) {
                        const float *c = M.cache.data() + size_t(si) * lat + i0;
                        vf64 cv[4];
                        for (int v = 0; v < 4; ++v) cv[v] = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(c + 8 * v))));
                        for (int a = 0; a < tn; ++a) {
                            if (si >= S[t0 + a]) continue;
                            const svfloat64_t pr = svdup_n_f64(ph[a][si]);
                            for (int v = 0; v < 4; ++v) acc[a][v] = svmla_f64_x(pd, acc[a][v], pr, cv[v]);
                        }
                    }
                    for (int a = 0; a < tn; ++a) {
                        const int t = t0 + a;
                        double sum[32];
                        for (int v = 0; v < 4; ++v) svst1_f64(pd, sum + 8 * v, acc[a][v]);
                        for (int i = 0; i < 32; ++i) av_all[(size_t(t) * nh + h) * lat + i0 + i] = float(sum[i] / total[size_t(t) * nh + h]);
                    }
                }
            });
        } else
        if (gather_block_on() && !prefix && !attn_f32() && lat % 32 == 0 && T > 4) {
            // beyond the selection size: per item one token x 3 heads x 32 latents over the token's positions, each
            // gathered row segment read once for the 3 heads (the item below read it per head: ~4 bytes per MAC). Per
            // (token, head, latent) the chain below: fused multiply-adds over the positions in order
            constexpr int HG = 3;
            const int ng = (nh + HG - 1) / HG, nb32 = lat / 32;
            team_run([&] {
                typedef svfloat64_t vf64 __attribute__((arm_sve_vector_bits(512)));
                const svbool_t pd = svptrue_b64();
#pragma omp for schedule(static) nowait
                for (int it = 0; it < T * ng * nb32; ++it) {
                    const int t = it / (ng * nb32), h0 = ((it / nb32) % ng) * HG, i0 = (it % nb32) * 32, hn = std::min(HG, nh - h0);
                    const long St = S[t];
                    const double *ph[HG];
                    for (int j = 0; j < HG; ++j) ph[j] = prob.data() + poff[t] + size_t(h0 + std::min(j, hn - 1)) * St;
                    vf64 acc[HG][4];
                    for (int j = 0; j < HG; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = svdup_n_f64(0);
                    for (long si = 0; si < St; ++si) {
                        const float *c = cbase(t) + size_t(ids[t][si]) * lat + i0;
                        if (si + 8 < St) __builtin_prefetch(cbase(t) + size_t(ids[t][si + 8]) * lat + i0, 0, 3);
                        vf64 cv[4];
                        for (int v = 0; v < 4; ++v) cv[v] = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(c + 8 * v))));
#pragma GCC unroll 3
                        for (int j = 0; j < HG; ++j) {
                            const svfloat64_t pr = svdup_n_f64(ph[j][si]);
                            for (int v = 0; v < 4; ++v) acc[j][v] = svmla_f64_x(pd, acc[j][v], pr, cv[v]);
                        }
                    }
                    for (int j = 0; j < hn; ++j) {
                        const int h = h0 + j;
                        double sum[32];
                        for (int v = 0; v < 4; ++v) svst1_f64(pd, sum + 8 * v, acc[j][v]);
                        for (int i = 0; i < 32; ++i) av_all[(size_t(t) * nh + h) * lat + i0 + i] = float(sum[i] / total[size_t(t) * nh + h]);
                    }
                }
            });
        } else
#endif
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (int thb = 0; thb < T * nh * lb; ++thb) {
                const int t = thb / (nh * lb), h = (thb / lb) % nh, i0 = (thb % lb) * 64;
                double sum[64] = {0};
                const double *ph = prob.data() + poff[t] + size_t(h) * S[t];
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                if (attn_f32()) {   // approximate: fp32 weighted sums, 4 vectors of 16
                    const svbool_t pw = svptrue_b32();
                    vf32 a0 = svdup_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
                    for (long si = 0; si < S[t]; ++si) {
                        const float *c = cbase(t) + size_t(ids[t][si]) * lat + i0;
                        if (si + 8 < S[t]) __builtin_prefetch(cbase(t) + size_t(ids[t][si + 8]) * lat + i0, 0, 3);
                        const float pr = float(ph[si]);
                        a0 = svmla_n_f32_x(pw, a0, svld1_f32(pw, c), pr), a1 = svmla_n_f32_x(pw, a1, svld1_f32(pw, c + 16), pr);
                        a2 = svmla_n_f32_x(pw, a2, svld1_f32(pw, c + 32), pr), a3 = svmla_n_f32_x(pw, a3, svld1_f32(pw, c + 48), pr);
                    }
                    float tmp[64];
                    svst1_f32(pw, tmp, a0), svst1_f32(pw, tmp + 16, a1), svst1_f32(pw, tmp + 32, a2), svst1_f32(pw, tmp + 48, a3);
                    for (int i = 0; i < 64; ++i) sum[i] = tmp[i];
                } else
#endif
                for (long si = 0; si < S[t]; ++si) {
                    const float *c = cbase(t) + size_t(ids[t][si]) * lat + i0;
                    if (si + 8 < S[t]) __builtin_prefetch(cbase(t) + size_t(ids[t][si + 8]) * lat + i0, 0, 3);   // random rows: misses overlap
                    const double pr = ph[si];
                    for (int i = 0; i < 64; ++i) sum[i] += pr * c[i];
                }
                for (int i = 0; i < 64; ++i) av_all[(size_t(t) * nh + h) * lat + i0 + i] = float(sum[i] / total[size_t(t) * nh + h]);
            }
        });
        lap(tl, "mla.avg");
        const double te0 = now();
        std::vector<Job> expand;
        fill(avs, [&](int t, int h) { return av_all.data() + (size_t(t) * nh + h) * lat; }, lat);
        for (int h = 0; h < nh; ++h) {
            Job j{&M.vb[h], nullptr, val_all.data() + size_t(h) * hd, T, nh * hd};
            for (int t = 0; t < T; ++t) j.as[t] = &avs[size_t(t) * nh + h];
            expand.push_back(std::move(j));
        }
        const double te1 = now();
        {
            GemvPlan plan(expand);
            timers["mla.ex_plan"] += now() - te1;
            const double te2 = now();
            team_run([&] { plan.run(omp_get_thread_num(), omp_get_num_threads()); });
            timers["mla.ex_run"] += now() - te2;
            timers["mla.ex_split"] += 1e-3 * plan.any_split;   // calls with a K split (count)
        }
        const double te3 = now();
        values.resize(T);
        auto copy_out = [&](int t) { values[t].assign(val_all.begin() + size_t(t) * nh * hd, val_all.begin() + size_t(t + 1) * nh * hd); };
        if (acts_par && T > 4)
            team_run([&] {
#pragma omp for schedule(static) nowait
                for (int t = 0; t < T; ++t) copy_out(t);
            });
        else
            for (int t = 0; t < T; ++t) copy_out(t);
        timers["mla.ex_jobs"] += te1 - te0, timers["mla.ex_copy"] += now() - te3, timers["mla.ex_calls"] += 1e-3;
        lap(tl, "mla.expand");
    }
    // mla_attention with the hybrid cache, for passes reaching beyond the ring: positions from hot_lo on are read from
    // the ring by every rank for its own heads; older ones only by the rank holding their block, for all heads, with
    // the absorbed queries of all heads gathered first. Per (token, head) the softmax uses the global maximum over
    // every selected position (one max all-reduce), the cold ranks send their exp sums and weighted latent sums to the
    // head's rank (one all-to-all), which adds them to its hot part in rank order. Deterministic; not bitwise the
    // replicated cache's (a different summation order over positions). Tokens in blocks of 64 (buffer sizes).
    void mla_attention_hybrid(MlaLayer &M, const std::vector<const Vec *> &q, const std::vector<std::vector<int>> &ids, std::vector<Vec> &values, int hot_lo) {
        double t0 = now(), tl = t0;
        const int T = int(q.size()), lat = m.kv_rank, nh = heads.size(), hd = M.kb[0].cols, W = comm.world, HA = int(m.attention_heads);
        const double scale = 1 / std::sqrt(float(hd));
        const int hmax = (HA + W - 1) / W;
        std::vector<int> head_rank(HA), head_li(HA), hbeg(W + 1, 0);
        for (int r = 0; r < W; ++r) {
            const Span sp = split(HA, W, r);
            hbeg[r] = sp.begin, hbeg[r + 1] = sp.end;
            for (int g = sp.begin; g < sp.end; ++g) head_rank[g] = r, head_li[g] = g - sp.begin;
        }
        values.assign(T, Vec());
        const int TB = 64;
        for (int b0 = 0; b0 < T; b0 += TB) {
            const int tn = std::min(TB, T - b0);
            std::vector<Activation> qh(size_t(tn) * nh), avs(size_t(tn) * nh);
            Vec aq(size_t(hmax) * tn * lat, 0.f), val(size_t(tn) * nh * hd), av(size_t(tn) * nh * lat);   // aq [h][t][lat], this rank's heads
            std::vector<Job> absorb;
            for (int h = 0; h < nh; ++h) {
                Job j{&M.kb[h], nullptr, aq.data() + size_t(h) * tn * lat, tn, lat};
                for (int t = 0; t < tn; ++t) {
                    qh[size_t(t) * nh + h].set(q[b0 + t]->data() + size_t(h) * hd, hd, false);
                    j.as[t] = &qh[size_t(t) * nh + h];
                }
                absorb.push_back(j);
            }
            gemv_multi(absorb);
            lap(tl, "mla.absorb");
            // every head's absorbed queries: this rank's heads in a zero buffer of all heads, one all-reduce (exact: one
            // value plus zeros)
            std::vector<float> aq_all(size_t(HA) * tn * lat, 0.f);   // [global head][t][lat]
            std::copy(aq.begin(), aq.begin() + size_t(nh) * tn * lat, aq_all.begin() + size_t(heads.begin) * tn * lat);
            double tc = now();
            comm.allreduce(aq_all.data(), int(aq_all.size()));
            timers["comm_mla_cold"] += now() - tc;
            auto aqg = [&](int g, int t) { return aq_all.data() + (size_t(g) * tn + t) * lat; };
            std::vector<std::vector<int>> hot(tn), cold(tn);   // hot: every rank; cold: the ones this rank holds
            for (int t = 0; t < tn; ++t)
                for (int pp : ids[b0 + t]) {
                    if (pp >= hot_lo) hot[t].push_back(pp);
                    else if (cold_owner(pp) == comm.rank) cold[t].push_back(pp);
                }
            auto dot = [&](const float *a, const float *c) {
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
                if (lat % 8 == 0) {   // eight latent values per double vector, then the horizontal sum
                    const svbool_t pd = svptrue_b64();
                    svfloat64_t acc = svdup_n_f64(0), acc2 = svdup_n_f64(0);
                    for (int i = 0; i < lat; i += 16) {
                        acc = svmla_f64_x(pd, acc, svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(a + i)))),
                                          svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(c + i)))));
                        acc2 = svmla_f64_x(pd, acc2, svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(a + i + 8)))),
                                           svcvt_f64_f32_x(pd, svreinterpret_f32_u64(svld1uw_u64(pd, reinterpret_cast<const uint32_t *>(c + i + 8)))));
                    }
                    return svaddv_f64(pd, svadd_f64_x(pd, acc, acc2)) * scale;
                }
#endif
                double acc = 0;
                for (int i = 0; i < lat; ++i) acc += double(a[i]) * c[i];
                return acc * scale;
            };
            auto ring_row = [&](int pp) { return M.cache.data() + size_t(pp % M.ring) * lat; };
            auto cold_row = [&](int pp) { return M.cold.data() + cold_slot(pp) * lat; };
            // scores: hot for this rank's heads, cold for every head; the maxima per (token, global head)
            std::vector<std::vector<double>> hs(size_t(tn) * nh), cs(size_t(tn) * HA);
            std::vector<double> mx(size_t(tn) * HA, -1e300);
            // items of 16 rows of one token: each row read once for every head it is scored for
            std::vector<std::array<int, 3>> items;   // (token, 0 hot / 1 cold, first row)
            for (int t = 0; t < tn; ++t) {
                for (int h = 0; h < nh; ++h) hs[size_t(t) * nh + h].resize(hot[t].size());
                for (int g = 0; g < HA; ++g) cs[size_t(t) * HA + g].resize(cold[t].size());
                for (size_t i = 0; i < hot[t].size(); i += 16) items.push_back({t, 0, int(i)});
                for (size_t i = 0; i < cold[t].size(); i += 16) items.push_back({t, 1, int(i)});
            }
            team_run([&] {
#pragma omp for schedule(dynamic, 4) nowait
                for (size_t it = 0; it < items.size(); ++it) {
                    const auto [t, kind, i0] = items[it];
                    const auto &rows = kind ? cold[t] : hot[t];
                    for (int i = i0; i < std::min<int>(int(rows.size()), i0 + 16); ++i) {
                        if (kind) {
                            const float *c = cold_row(rows[i]);
                            for (int g = 0; g < HA; ++g) cs[size_t(t) * HA + g][i] = dot(aqg(g, t), c);
                        } else {
                            const float *c = ring_row(rows[i]);
                            for (int h = 0; h < nh; ++h) hs[size_t(t) * nh + h][i] = dot(aqg(heads.begin + h, t), c);
                        }
                    }
                }
            });
            for (int t = 0; t < tn; ++t) {
                for (int g = 0; g < HA; ++g)
                    for (double v : cs[size_t(t) * HA + g]) mx[size_t(t) * HA + g] = std::max(mx[size_t(t) * HA + g], v);
                for (int h = 0; h < nh; ++h)
                    for (double v : hs[size_t(t) * nh + h]) mx[size_t(t) * HA + heads.begin + h] = std::max(mx[size_t(t) * HA + heads.begin + h], v);
            }
            tc = now();
            comm.allreduce_max(mx.data(), int(mx.size()));
            // cold partials per (token, global head): [exp sum, latent sums] in double, summed over the ranks by one
            // all-reduce as floats (rank order)
            const int rec = lat + 1;
            std::vector<float> part(size_t(tn) * HA * rec, 0.f);
            team_run([&] {
#pragma omp for schedule(dynamic, 4) nowait
                for (int it = 0; it < tn * HA; ++it) {
                    const int t = it / HA, g = it % HA;
                    const double mv = mx[size_t(t) * HA + g];
                    const auto &v = cs[size_t(t) * HA + g];
                    std::vector<double> o(rec, 0.0);
                    for (size_t i = 0; i < v.size(); ++i) {
                        const double e = std::exp(v[i] - mv);
                        o[0] += e;
                        const float *c = cold_row(cold[t][i]);
                        for (int j = 0; j < lat; ++j) o[1 + j] += e * c[j];
                    }
                    float *pp = part.data() + (size_t(t) * HA + g) * rec;
                    for (int j = 0; j < rec; ++j) pp[j] = float(o[j]);
                }
            });
            comm.allreduce(part.data(), int(part.size()));
            timers["comm_mla_cold"] += now() - tc;
            // this rank's heads: hot part, then the cold partials of ranks 0 .. W-1 in order
            team_run([&] {
#pragma omp for schedule(dynamic, 1) nowait
                for (int it = 0; it < tn * nh; ++it) {
                    const int t = it / nh, h = it % nh, g = heads.begin + h;
                    const double mv = mx[size_t(t) * HA + g];
                    std::vector<double> acc(rec, 0.0);
                    const auto &v = hs[size_t(t) * nh + h];
                    for (size_t i = 0; i < v.size(); ++i) {
                        const double e = std::exp(v[i] - mv);
                        acc[0] += e;
                        const float *c = ring_row(hot[t][i]);
                        for (int j = 0; j < lat; ++j) acc[1 + j] += e * c[j];
                    }
                    const float *pr = part.data() + (size_t(t) * HA + g) * rec;
                    for (int j = 0; j < rec; ++j) acc[j] += pr[j];
                    for (int j = 0; j < lat; ++j) av[(size_t(t) * nh + h) * lat + j] = float(acc[1 + j] / acc[0]);
                }
            });
            lap(tl, "mla.hybrid");
            std::vector<Job> expand;
            for (int h = 0; h < nh; ++h) {
                Job j{&M.vb[h], nullptr, val.data() + size_t(h) * hd, tn, nh * hd};
                for (int t = 0; t < tn; ++t) {
                    avs[size_t(t) * nh + h].set(av.data() + (size_t(t) * nh + h) * lat, lat, false);
                    j.as[t] = &avs[size_t(t) * nh + h];
                }
                expand.push_back(j);
            }
            gemv_multi(expand);
            for (int t = 0; t < tn; ++t) values[b0 + t].assign(val.begin() + size_t(t) * nh * hd, val.begin() + size_t(t + 1) * nh * hd);
            lap(tl, "mla.expand");
        }
        timers["mla.hybrid_total"] += now() - t0;
    }
    Vec mla_attend(MlaLayer &M, const Vec &q, const Vec &iq, const Vec &iw, const Vec &ik, const Vec &gate, const Vec &row, int pos) {
        std::vector<std::vector<int>> ids{mla_select(M, iq, iw, ik, gate, row, pos)};
        std::vector<Vec> values;
        mla_attention(M, {&q}, ids, values);
        return std::move(values[0]);
    }
    Vec mla(MlaLayer &M, const Vec &x) { return std::move(mla_multi(M, {x})[0]); }   // the same arithmetic as one token alone
    Vec ffn_dense(FfnDense &F, const Vec &x, float limit) {
        const double t = now();
        Activation ax;
        ax.set(x.data(), m.hidden, trunk_q8);
        Vec g(F.gate.rows), u(F.up.rows);
        gemv_multi({{&F.gate, &ax, g.data()}, {&F.up, &ax, u.data()}});
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (size_t i = 0; i < g.size(); ++i) g[i] = ref::swiglu(g[i], u[i], limit);
        });
        auto out = gemv(F.down, g);
        timers["dense"] += now() - t;
        return out;
    }
    // the MoE layer in one team region (moe() below forks three times and runs routing, SwiGLU setup and the weighted
    // sum on one thread): router + shared gate/up, top-k on one thread, routed gate/up, SwiGLU (and int8 blocks for
    // W4A8 down projections) over the team, down projections, the weighted sum over the team. Same arithmetic per
    // element as moe(), so the results are bitwise equal. Needs every matrix without a K split.
    std::vector<Vec> moe_g, moe_u, moe_y;
    std::vector<Activation> moe_acts;
    Activation moe_ax, moe_af;   // moe_fused's per-layer buffers, persistent (no allocation per layer)
    Vec moe_logits, moe_sg, moe_su, moe_ys, moe_rscore, moe_rkey, moe_routing;
    std::vector<int> moe_ids, moe_mine, moe_start;
    std::vector<Job> moe_first, moe_up_jobs, moe_down_jobs;
    std::vector<float *> moe_gp;
    std::vector<const float *> moe_upp;
    std::vector<float> moe_lim;
    static bool moe_quant(MoeLayer &E) {
        if (E.quant < 0) {
            bool q = E.sgate.rows && needs_q8(E.sgate);
            for (const auto &sl : E.slices) q = q || needs_q8(sl.gate);
            E.quant = q;
        }
        return E.quant;
    }
    Vec moe_fused(MoeLayer &E, const Vec &x, const strata::core::LayerDescriptor &desc) {
        const double t_in = now();
        const int team = omp_get_max_threads();
        const bool quant = moe_quant(E);
        Activation &ax = moe_ax, &af = moe_af;
        ax.set(x.data(), m.hidden, quant, g_act_group);
        af.set(x.data(), m.hidden, false);
        const bool rsh = E.router.rows != m.experts;
        // persistent buffers: every element read below is written first, except the router logits of other ranks'
        // rows (sharded router: zeroed) and the shared expert's output on a rank without a shared slice (zeroed)
        Vec &logits = moe_logits, &sg = moe_sg, &su = moe_su, &ys = moe_ys;
        if (rsh) logits.assign(m.experts + 16, 0.f);
        else logits.resize(std::max<size_t>(logits.size(), E.router.padded));
        sg.resize(std::max<size_t>(sg.size(), E.sgate.padded)), su.resize(std::max<size_t>(su.size(), E.sup.padded));
        if (E.sgate.rows) ys.resize(m.hidden);
        else ys.assign(m.hidden, 0.f);
        Vec sum(m.hidden);
        auto &first = moe_first;
        first.assign(1, Job{&E.router, &af, logits.data() + E.router_r0});
        if (E.sgate.rows) first.push_back({&E.sgate, &ax, sg.data()}), first.push_back({&E.sup, &ax, su.data()});
        auto &ids = moe_ids, &mine = moe_mine;
        ids.resize(m.top_k), mine.clear();
        Vec &routing = moe_routing;
        routing.resize(m.top_k);
        auto &g = moe_g, &u = moe_u, &y = moe_y;   // persistent: no allocation or zero fill per layer
        auto &acts = moe_acts;
        auto &up_jobs = moe_up_jobs, &down_jobs = moe_down_jobs;
        up_jobs.clear(), down_jobs.clear();
        auto &start = moe_start;   // SwiGLU value ranges per activation (routed slices, then the shared expert)
        start.assign(1, 0);
        auto &gp = moe_gp;
        auto &upp = moe_upp;
        auto &lim = moe_lim;
        gp.clear(), upp.clear(), lim.clear();
        double t_route = 0, t_up = 0, t_act = 0, t_down = 0, t_reg = 0, t_first = 0;
        Vec &rscore = moe_rscore, &rkey = moe_rkey;
        rscore.resize(m.experts), rkey.resize(m.experts);
        const double t_setup = now();
        team_run([&] {
            const int t = omp_get_thread_num(), n = omp_get_num_threads();
            if (t == 0) t_reg = now();
            gemv_share(first, t, n);
            if (t == 0) t_first = now();
            if (rsh) {   // the sharded logits summed over the ranks, then the scores over the team
                team_barrier();
                if (t == 0) {
                    const double tc = now();
                    comm.allreduce(logits.data(), m.experts);
                    timers["comm_router"] += now() - tc;
                }
                team_barrier();
                const int r0 = int(int64_t(m.experts) * t / n), r1 = int(int64_t(m.experts) * (t + 1) / n);
                for (int r = r0; r < r1; ++r) rscore[r] = ref::sigmoid(logits[r]), rkey[r] = rscore[r] + E.bias[r];
            } else {   // this thread's router rows: sigmoid scores and selection keys (the top-k on one thread only selects)
                const int tiles = E.router.padded / 16, v = (t + E.router.rot) % n;
                const int r0 = 16 * int(int64_t(tiles) * v / n), r1 = std::min(int(m.experts), 16 * int(int64_t(tiles) * (v + 1) / n));
                for (int r = r0; r < r1; ++r) rscore[r] = ref::sigmoid(logits[r]), rkey[r] = rscore[r] + E.bias[r];
            }
            team_barrier();
            if (omp_get_thread_num() == 0)
            {
                t_route = now();
                route_select(rscore.data(), rkey.data(), m.experts, m.top_k, m.expert_scale, ids.data(), routing.data());
                timers["moe.topk_select"] += now() - t_route;
                route_ids.insert(route_ids.end(), ids.begin(), ids.end());
                for (int i = 0; i < m.top_k; ++i)
                    if (E.owner_index[ids[i]] >= 0) mine.push_back(i);
                for (int i : mine)   // the chosen slices' descriptors are cold: their misses overlap
                    for (size_t o = 0; o < sizeof(ExpertSlice); o += 64) __builtin_prefetch(reinterpret_cast<const char *>(&E.slices[E.owner_index[ids[i]]]) + o);
                const size_t k_n = mine.size();
                if (g.size() < k_n) g.resize(k_n), u.resize(k_n), y.resize(k_n);
                if (acts.size() < k_n + 1) acts.resize(k_n + 1);
                for (size_t k = 0; k < k_n; ++k) y[k].resize(m.hidden);   // every row is written by its down projection
                for (size_t k = 0; k < k_n; ++k) {
                    auto &sl = E.slices[E.owner_index[ids[mine[k]]]];
                    g[k].resize(sl.gate.padded), u[k].resize(sl.up.padded);
                    up_jobs.push_back({&sl.gate, &ax, g[k].data()}), up_jobs.push_back({&sl.up, &ax, u[k].data()});
                    down_jobs.push_back({&sl.down, &acts[k], y[k].data()});
                    acts[k].x.resize(sl.gate.rows);
                    acts[k].group = g_act_group;
                    if (needs_q8(sl.down)) acts[k].q.resize(sl.gate.rows), acts[k].qs.resize(sl.gate.rows / g_act_group);
                    else acts[k].q.clear();
                    gp.push_back(g[k].data()), upp.push_back(u[k].data()), lim.push_back(desc.swiglu_limit);
                    start.push_back(start.back() + sl.gate.rows);
                }
                if (E.sgate.rows) {
                    down_jobs.push_back({&E.sdown, &acts[k_n], ys.data()});
                    acts[k_n].x.resize(E.sgate.rows);
                    acts[k_n].group = 64;
                    if (needs_q8(E.sdown)) acts[k_n].q.resize(E.sgate.rows), acts[k_n].qs.resize(E.sgate.rows / 64);
                    else acts[k_n].q.clear();
                    gp.push_back(sg.data()), upp.push_back(su.data()), lim.push_back(desc.shared_swiglu_limit);
                    start.push_back(start.back() + E.sgate.rows);
                }
                t_up = now();
            }
            team_barrier();
            prefetch_share(down_jobs, t, n);   // the down projections into L2 while gate/up computes
            gemv_share(up_jobs, t, n);
            team_barrier();
            if (t == 0) t_act = now();
            // SwiGLU per 64-value block (whole Q8 blocks for int8 down projections)
#pragma omp for schedule(static) nowait
            for (int b = 0; b < start.back() / 64; ++b) {
                const int i0 = b * 64, k = int(std::upper_bound(start.begin(), start.end(), i0) - start.begin()) - 1, j0 = i0 - start[k];
                Activation &a = acts[k];   // k == mine.size(): the shared expert
                for (int j = j0; j < j0 + 64; ++j) a.x[j] = ref::swiglu(gp[k][j], upp[k][j], lim[k]);
                if (!a.q.empty()) gf4_quantize_q8g(a.x.data() + j0, 64, a.group, a.q.data() + j0, a.qs.data() + j0 / a.group);
            }
            team_barrier();
            if (t == 0) t_down = now();
            gemv_share(down_jobs, t, n);
            team_barrier();
#pragma omp for schedule(static) nowait
            for (int d = 0; d < m.hidden; ++d) {
                float v = 0;
                for (size_t k = 0; k < mine.size(); ++k) v += routing[mine[k]] * y[k][d];
                sum[d] = v + ys[d];
            }
        });
        (void)team;
        const double t_out = now();
        timers["moe.route_gemv"] += t_route - t_in;
        timers["moe.r_setup"] += t_setup - t_in, timers["moe.r_dispatch"] += t_reg - t_setup, timers["moe.r_own"] += t_first - t_reg;
        timers["moe.r_wait"] += t_route - t_first;
        timers["moe.topk"] += t_up - t_route;
        timers["moe.up"] += t_act - t_up;
        timers["moe.act"] += t_down - t_act;
        timers["moe.down"] += t_out - t_down;
        timers["router"] += t_up - t_in;
        timers["experts"] += t_out - t_up;
        return sum;
    }
    bool moe_fusable(const MoeLayer &E) const {
        static const bool on = !(std::getenv("STRATA_RUNNER_MOE_FUSED") && std::string(std::getenv("STRATA_RUNNER_MOE_FUSED")) == "0");
        if (!on) return false;
        const int team = omp_get_max_threads();
        auto ok = [&](const Matrix &w) { return !w.rows || (!ksplit(w, team).chunks && w.padded == w.rows); };
        if (!ok(E.router) || !ok(E.sgate) || !ok(E.sup) || !ok(E.sdown) || (E.sgate.rows && E.sgate.rows % 64)) return false;
        if (E.router.rows != m.experts && E.router.rows % 16) return false;
        for (const auto &sl : E.slices)
            if (!ok(sl.gate) || !ok(sl.up) || !ok(sl.down) || sl.gate.rows % 64) return false;
        return true;
    }
    Vec moe(MoeLayer &E, const Vec &x, const strata::core::LayerDescriptor &desc) {
        if (moe_fusable(E)) return moe_fused(E, x, desc);
        double t = now(), t0 = t;
        Activation ax, af;
        const bool quant = moe_quant(E);   // int8 activations only when a kernel of this layer takes them
        ax.set(x.data(), m.hidden, quant, g_act_group);
        af.set(x.data(), m.hidden, false);
        Vec logits(m.experts + 16, 0.f), sg(E.sgate.rows), su(E.sup.rows);
        std::vector<Job> first{{&E.router, &af, logits.data() + E.router_r0}};
        if (E.sgate.rows) first.push_back({&E.sgate, &ax, sg.data()}), first.push_back({&E.sup, &ax, su.data()});
        gemv_multi(first);   // router and the shared expert's gate/up: both need only x
        if (E.router.rows != m.experts) comm.allreduce(logits.data(), m.experts);
        std::vector<int> ids(m.top_k);
        Vec routing(m.top_k);
        lap(t0, "moe.route_gemv");
        route_topk(logits.data(), E.bias.data(), m.experts, m.top_k, m.expert_scale, ids.data(), routing.data());
        route_ids.insert(route_ids.end(), ids.begin(), ids.end());
        timers["router"] += now() - t;
        timers["moe.topk"] += now() - t0;
        t = now();
        std::vector<int> mine;   // positions i of the routed experts this rank holds a slice of
        for (int i = 0; i < m.top_k; ++i)
            if (E.owner_index[ids[i]] >= 0) mine.push_back(i);
        std::vector<Vec> g(mine.size()), u(mine.size()), y(mine.size());
        std::vector<Job> up_jobs, down_jobs;
        for (size_t k = 0; k < mine.size(); ++k) {
            auto &s = E.slices[E.owner_index[ids[mine[k]]]];
            g[k].resize(s.gate.rows), u[k].resize(s.up.rows), y[k].resize(m.hidden);
            up_jobs.push_back({&s.gate, &ax, g[k].data()}), up_jobs.push_back({&s.up, &ax, u[k].data()});
        }
        double tl = t;
        gemv_multi(up_jobs);
        lap(tl, "moe.up");
        // SwiGLU of every routed slice and the shared expert in one team pass (one exp per element: ~50 us serial)
        std::vector<std::pair<float *, const float *>> gu;
        std::vector<float> limit;
        std::vector<int> start{0};
        for (size_t k = 0; k < mine.size(); ++k) gu.push_back({g[k].data(), u[k].data()}), limit.push_back(desc.swiglu_limit), start.push_back(start.back() + int(g[k].size()));
        if (E.sgate.rows) gu.push_back({sg.data(), su.data()}), limit.push_back(desc.shared_swiglu_limit), start.push_back(start.back() + int(sg.size()));
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (int i = 0; i < start.back(); ++i) {
                const int k = int(std::upper_bound(start.begin(), start.end(), i) - start.begin()) - 1, j = i - start[k];
                gu[k].first[j] = ref::swiglu(gu[k].first[j], gu[k].second[j], limit[k]);
            }
        });
        std::vector<Activation> acts(mine.size() + 1);
        for (size_t k = 0; k < mine.size(); ++k) {
            auto &s = E.slices[E.owner_index[ids[mine[k]]]];
            acts[k].set(g[k].data(), int(g[k].size()), needs_q8(s.down), g_act_group);
            down_jobs.push_back({&s.down, &acts[k], y[k].data()});
        }
        Vec ys(m.hidden, 0.f);
        if (E.sgate.rows) {
            acts.back().set(sg.data(), int(sg.size()), needs_q8(E.sdown));
            down_jobs.push_back({&E.sdown, &acts.back(), ys.data()});
        }
        lap(tl, "moe.act");
        gemv_multi(down_jobs);
        lap(tl, "moe.down");
        Vec sum(m.hidden, 0.f);
        for (size_t k = 0; k < mine.size(); ++k)
            for (int d = 0; d < m.hidden; ++d) sum[d] += routing[mine[k]] * y[k][d];
        for (int d = 0; d < m.hidden; ++d) sum[d] += ys[d];
        lap(tl, "moe.sum");
        timers["experts"] += now() - t;
        return sum;
    }
    // one position; returns the full logits on rank 0 (or nothing when the model is truncated)
    Vec step(int token, const std::filesystem::path &dump) {
        const double t_step = now();
        struct StepTimer {   // the whole step, for the untimed remainder
            Runner &r; double t0;
            ~StepTimer() { r.timers["step"] += now() - t0; }
        } step_timer{*this, t_step};
        if (token < 0 || token >= m.vocab) throw std::runtime_error("runner: token id outside the vocabulary");
        Vec emb(m.hidden);
        const uint16_t *ed = embedding.data() + size_t(token) * m.hidden;
        for (int i = 0; i < m.hidden; ++i) {
            const uint32_t v = uint32_t(ed[i]) << 16;
            std::memcpy(&emb[i], &v, 4);
        }
        Vec r(4 * m.hidden);
        for (int i = 0; i < 4; ++i) std::copy(emb.begin(), emb.end(), r.begin() + i * m.hidden);
        auto save = [&](const char *kind, int l) {
            if (dump.empty() || comm.rank) return;
            std::ofstream o(dump / (std::string(kind) + "-" + std::to_string(l) + ".bin"), std::ios::binary);
            o.write(reinterpret_cast<const char *>(r.data()), r.size() * 4);
        };
        Vec c_ffn, y_ffn;   // the previous layer's FFN residual update, done inside this layer's first site
        for (int l = 0; l < layers_run; ++l) {
            auto &Y = L[l];
            auto [x, c] = site(r, l ? &c_ffn : nullptr, l ? &y_ffn : nullptr, Y.hc_attn, Y.hc_attn_base, Y.hc_attn_scale, Y.attn_norm,
                               l ? after_ffn(l - 1) : std::vector<const Matrix *>{});
            if (l) save("layer", l - 1);
            auto y = Y.kda ? kda(Y.kd, x) : mla(Y.ml, x);
            double t = now();
            allreduce_touch(y, after_attn(l));
            timers["comm"] += now() - t;
            timers[Y.kda ? "comm_kda" : "comm_mla"] += now() - t;
            auto f = site(r, &c, &y, Y.hc_ffn, Y.hc_ffn_base, Y.hc_ffn_scale, Y.ffn_norm, after_attn(l));   // attention update + FFN site
            save("attn", l);
            x = f.first;
            y = Y.dense ? ffn_dense(Y.ffn, x, m.layers[l].swiglu_limit) : moe(Y.moe, x, m.layers[l]);
            t = now();
            allreduce_touch(y, after_ffn(l));
            timers["comm"] += now() - t;
            timers[Y.dense ? "comm_dense" : "comm_moe"] += now() - t;
            c_ffn = f.second, y_ffn = y;
            if (l + 1 == layers_run) {   // no site follows the last layer's update
                t = now();
                write(r, c_ffn, y_ffn);
                timers["mhc.write"] += now() - t;
                save("layer", l);
            }
        }
        ++position;
        last_hash = 1469598103934665603ull;   // FNV-1a over the residual's bytes
        for (float v : r) {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            last_hash = (last_hash ^ u) * 1099511628211ull;
        }
        if (layers_run < int(m.layers.size())) return {};
        const double t = now();
        Vec collapsed(m.hidden);
        for (int d = 0; d < m.hidden; ++d) collapsed[d] = (r[d] + r[m.hidden + d] + r[2 * m.hidden + d] + r[3 * m.hidden + d]) * .25f;
        const auto hn = norm(collapsed, out_norm);
        if (mtp.on) last_hidden.assign(1, hn);
        Vec part = gemv(head, hn);
        if (!full_logits) {
            last_part.assign(1, std::move(part));
            timers["head"] += now() - t;
            return {};
        }
        Vec logits(comm.rank == 0 ? m.vocab : vocab.end, 0.f);
        std::copy(part.begin(), part.end(), logits.begin() + vocab.begin);
        comm.gather(logits.data(), vocab.begin, vocab.end, m.vocab);
        timers["head"] += now() - t;
        if (comm.rank) return {};
        return logits;
    }

    // ------------------------------------------------------------------------------------------- draft rollback
    // What a position changes in the main model besides rows a later pass overwrites (attention caches, completed
    // index pools): the KDA recurrent states and conv histories, and each MLA layer's index key/gate tail (a rejected
    // position can overwrite a slot the current, unfinished pool still needs). step_multi(..., snaps) fills snaps[j]
    // with the state after token j (j < T - 1), taken right after that token's step in each layer; restore() rolls
    // the model back to it.
    struct MainState {
        int position = 0;
        std::vector<float *> kda_state;   // per layer: arena memory, first touched by the threads that own the blocks
        std::vector<Vec> kda_hist, mla_keys, mla_gates;   // per layer (empty for the other kind)
    };
    std::vector<float> kb_qkv, kb_nq, kb_nk, kb_ed, kb_y, kb_qs, kb_ks, kb_beta, kb_inv;   // kda_multi's batched phases
    std::vector<double> site_tr;                  // STRATA_RUNNER_SITE_TRACE: [thread][start, phase-A end] per site
    double site_stat[4] = {0, 0, 0, 0};           // sums of the latest start, latest phase-A end, thread 0's phase A; count
    std::array<int, 64> site_late_start{}, site_late_end{};
    std::vector<std::vector<float>> kb_hist;
    double kb_tp[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // thread 0's clock after each batched phase
    std::vector<MainState> *snap_sink = nullptr;
    float *snap_ptr = nullptr;   // the KDA snapshot being written (shared by the team)
    int cur_layer = 0;
    MainState &snap_in(std::vector<MainState> &sink, int j) {
        auto &st = sink[j];
        if (st.kda_state.size() != L.size()) st.kda_state.assign(L.size(), nullptr), st.kda_hist.resize(L.size()), st.mla_keys.resize(L.size()), st.mla_gates.resize(L.size());
        return st;
    }
    MainState &snap_at(int j) {
        auto &st = (*snap_sink)[j];
        if (st.kda_state.size() != L.size()) st.kda_state.assign(L.size(), nullptr), st.kda_hist.resize(L.size()), st.mla_keys.resize(L.size()), st.mla_gates.resize(L.size());
        return st;
    }
    std::vector<MainState> snap_pool;   // reused by every verification pass (the arena buffers stay placed)

    // ------------------------------------------------------------------------------------------- resident contexts
    // Several sequences in one runner (--contexts N): a context's per-sequence state (KDA states, their rollback
    // buffers and conv histories, MLA index key/gate tails, pools and latent caches, the draft block's, the
    // positions) lives in ctx_store[c] while another context is active; switch_ctx swaps the layers' fields with the
    // store (pointers and vectors, no copies), so every kernel works on the active context unchanged
    struct LayerState { float *state = nullptr; std::array<float *, 4> pool{}; Vec hist[3], keys, gates, pooled, cache, cold; };
    struct CtxState { std::vector<LayerState> layers; LayerState mtp_layer; int position = 0, mtp_position = 0; };
    std::vector<CtxState> ctx_store;
    int active_ctx = 0;
    static void swap_layer(Layer &Y, LayerState &s) {
        if (Y.kda) {
            std::swap(Y.kd.state, s.state), std::swap(Y.kd.pool, s.pool);
            for (int i = 0; i < 3; ++i) Y.kd.hist[i].swap(s.hist[i]);
        } else Y.ml.keys.swap(s.keys), Y.ml.gates.swap(s.gates), Y.ml.pooled.swap(s.pooled), Y.ml.cache.swap(s.cache), Y.ml.cold.swap(s.cold);
    }
    void swap_ctx(CtxState &cs) {
        for (size_t l = 0; l < L.size(); ++l) swap_layer(L[l], cs.layers[l]);
        if (mtp.on) swap_layer(mtp.layer, cs.mtp_layer);
        std::swap(position, cs.position), std::swap(mtp.position, cs.mtp_position);
    }
    void switch_ctx(int c) {
        if (c == active_ctx) return;
        swap_ctx(ctx_store[active_ctx]);   // the active context's state into its slot
        swap_ctx(ctx_store[c]);            // c's state into the layers
        active_ctx = c;
    }
    // contexts 1 .. n - 1 with fresh (zero) state shaped like context 0's; context 0 stays active
    void add_contexts(int n) {
        ctx_store.resize(n);
        auto fresh = [&](const Layer &Y, LayerState &s) {
            if (Y.kda) {
                const auto &K = Y.kd;
                s.state = reinterpret_cast<float *>(arena_alloc(K.state_n * 4)), s.pool = {s.state, nullptr, nullptr, nullptr};
                const size_t per = size_t(m.linear_dim) * KDA_GROUP, items = K.state_n / per;
                float *st = s.state;
                team_run([&] {   // placed like context 0's (load_layer): zeroed by the threads that walk each block
#pragma omp for schedule(static) nowait
                    for (size_t it = 0; it < items; ++it) std::memset(st + it * per, 0, per * 4);
                });
                for (int i = 0; i < 3; ++i) s.hist[i].assign(K.hist[i].size(), 0.f);
            } else {
                const auto &M = Y.ml;
                s.keys.assign(M.keys.size(), 0.f), s.gates.assign(M.gates.size(), 0.f), s.pooled.assign(M.pooled.size(), 0.f), s.cache.assign(M.cache.size(), 0.f);
                s.cold.assign(M.cold.size(), 0.f);
            }
        };
        for (int c = 1; c < n; ++c) {
            auto &cs = ctx_store[c];
            cs.layers.resize(L.size());
            for (size_t l = 0; l < L.size(); ++l) fresh(L[l], cs.layers[l]);
            if (mtp.on) fresh(mtp.layer, cs.mtp_layer);
        }
        ctx_store[0].layers.resize(L.size());
    }
    // a pass over several contexts (step_multi with pass_groups set): (context, tokens) groups in pass order, each
    // context's tokens its next positions; group_snaps[c] gets context c's snapshots as snaps does for one
    std::vector<std::pair<int, int>> pass_groups;
    std::vector<std::vector<MainState>> *group_snaps = nullptr;

    void restore(const MainState &st) {
        for (size_t l = 0; l < L.size(); ++l) {   // pointers and the small MLA key tails on this thread
            auto &Y = L[l];
            if (Y.kda) Y.kd.state = st.kda_state[l];   // the buffer the pass left the state after that token in
            else Y.ml.keys = st.mla_keys[l], Y.ml.gates = st.mla_gates[l];
        }
        team_run([&] {   // the KDA conv histories (~0.9 MB) over the team
#pragma omp for schedule(static) nowait
            for (int li = 0; li < int(L.size()) * 3; ++li) {
                auto &Y = L[li / 3];
                if (!Y.kda) continue;
                const int i = li % 3;
                const size_t hn = Y.kd.hist[0].size();
                std::copy(st.kda_hist[li / 3].begin() + i * hn, st.kda_hist[li / 3].begin() + (i + 1) * hn, Y.kd.hist[i].begin());
            }
        });
        position = st.position;
    }

    // ------------------------------------------------------------------------------------------- multi-token pass
    // T = 1..4 consecutive positions in one pass (MTP verification, prefill chunks): every GEMV reads its weights once
    // for all T tokens (the kernels' arithmetic per token is the same for any token count), while the mHC sites, the
    // KDA convolution and recurrence, MLA attention and routing run per token in position order with the one-token
    // code. A pass is therefore bitwise equal to T calls of step(); main's --multi checks that.
    static std::vector<Activation> acts_of(const std::vector<Vec> &xs, int n, bool quant, int group = 64) {
        std::vector<Activation> a(xs.size());
        const int T = int(xs.size());
        if (T <= 4 || n % 64) {
            for (int t = 0; t < T; ++t) a[t].set(xs[t].data(), n, quant, group);
            return a;
        }
        // a prefill chunk: every token's buffers, copy and int8 blocks in one team region, a contiguous block of tokens
        // per thread (Activation::set's per-block quantization; the buffers were allocated and zero-filled on the main
        // thread: ~2 us per token and layer; token t on thread t % team scattered the 3-4 tokens a kernel group
        // reads over the CMGs: the GEMVs slowed down more than this saved)
        const int nb = n / 64;
        // STRATA_RUNNER_ACTS_PAR=1: each thread allocates its block of tokens (opt-in: the setup drops 0.16 -> 0.05 ms
        // per prompt token, but the GEMVs reading those buffers slow down more: 198.2 vs 200.1 tok/s at 2K)
        static const bool par = std::getenv("STRATA_RUNNER_ACTS_PAR") && std::string(std::getenv("STRATA_RUNNER_ACTS_PAR")) == "1";
        if (!par) {
            for (auto &v : a) {
                v.x.resize(n);
                if (quant) v.group = group, v.q.resize(n), v.qs.resize(n / group);
            }
            team_run([&] {
#pragma omp for schedule(static) nowait
                for (int i = 0; i < T * nb; ++i) {
                    const int t = i / nb, b = i % nb;
                    std::copy(xs[t].begin() + 64 * b, xs[t].begin() + 64 * b + 64, a[t].x.begin() + 64 * b);
                    if (quant) gf4_quantize_q8g(a[t].x.data() + 64 * b, 64, group, a[t].q.data() + 64 * b, a[t].qs.data() + 64 * b / group);
                }
            });
            return a;
        }
        team_run([&] {
            const int th = omp_get_thread_num(), nth = omp_get_num_threads();
            for (int t = int(int64_t(T) * th / nth); t < int(int64_t(T) * (th + 1) / nth); ++t) {
                auto &v = a[t];
                v.x.assign(xs[t].begin(), xs[t].begin() + n);
                if (!quant) continue;
                v.group = group, v.q.resize(n), v.qs.resize(n / group);
                for (int b = 0; b < nb; ++b) gf4_quantize_q8g(v.x.data() + 64 * b, 64, group, v.q.data() + 64 * b, v.qs.data() + 64 * b / group);
            }
        });
        return a;
    }
    // acts_of into a persistent per-call-site pool (long passes): the buffers are allocated (on the main thread) once,
    // then only copied and quantized over the team; acts_of allocated and zero-filled T buffers on the main thread per
    // call (~0.16 ms per prompt token over the MoE and KDA layers). A site's pool is used only inside its function.
    // STRATA_RUNNER_ACTS_POOL=0: acts_of
    std::array<std::vector<Activation>, 8> act_pool;
    std::vector<Activation> &acts_pool(int site, const std::vector<Vec> &xs, int n, bool quant, int group = 64) {
        static const bool on = !(std::getenv("STRATA_RUNNER_ACTS_POOL") && std::string(std::getenv("STRATA_RUNNER_ACTS_POOL")) == "0");
        auto &a = act_pool[site];
        const int T = int(xs.size());
        if (!on || T <= 4 || n % 64) { a = acts_of(xs, n, quant, group); return a; }
        if (a.size() != size_t(T)) a.resize(T);
        for (auto &v : a) {
            if (v.x.size() != size_t(n)) v.x.resize(n);
            if (quant) v.group = group, v.q.resize(n), v.qs.resize(n / group);
            else v.q.clear(), v.qs.clear();   // an empty q selects the float kernels
        }
        const int nb = n / 64;
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (int i = 0; i < T * nb; ++i) {
                const int t = i / nb, b = i % nb;
                std::copy(xs[t].begin() + 64 * b, xs[t].begin() + 64 * b + 64, a[t].x.begin() + 64 * b);
                if (quant) gf4_quantize_q8g(a[t].x.data() + 64 * b, 64, group, a[t].q.data() + 64 * b, a[t].qs.data() + 64 * b / group);
            }
        });
        return a;
    }
    static std::vector<Vec> split_rows(const Vec &all, int T, int stride, int rows) {
        std::vector<Vec> out(T);
        for (int t = 0; t < T; ++t) out[t].assign(all.begin() + size_t(t) * stride, all.begin() + size_t(t) * stride + rows);
        return out;
    }
    std::array<Vec, 3> km_raw;
    Vec km_decay, km_gate;
    Vec km_fa, km_ga, km_beta, km_out;              // kda_multi's work arrays (each element is written before it is read)
    std::vector<Activation> km_afa, km_aga, km_ay;  // its activations (allocated per call: ~0.13 ms per generated token at 16 contexts)
    static void act_rows(std::vector<Activation> &v, int T, int n, bool q8) {   // T activations of n values (q8: the Q8 form too)
        if (v.size() < size_t(T)) v.resize(T);
        for (int t = 0; t < T; ++t) {
            Activation &a = v[t];
            if (a.x.size() != size_t(n)) a.x.resize(n);
            if (q8) {
                if (a.q.size() != size_t(n)) a.q.resize(n), a.qs.resize(n / 64);
                a.group = 64;
            } else a.q.clear(), a.qs.clear();
        }
    }
    std::vector<Vec> kda_multi(KdaLayer &K, const std::vector<Vec> &xs) {
        const double t_in = now();
        const int T = int(xs.size()), D = m.linear_dim, nh = heads.size(), n = nh * D, Kc = m.conv_kernel, H = m.hidden;
        // one activation array: an int8 kernel reads q / qs, a float kernel x (the second, float-only copy was the same)
        auto &ax = acts_pool(0, xs, H, trunk_q8 || q8_act(K.q, int(xs.size())));
        auto &af = ax;
        const double t_acts = now();
        double t_p1 = 0, t_p2 = 0;
        const int fr = K.f_a.rows, gr = K.g_a.rows, br = K.beta.rows;
        auto &raw_all = km_raw;   // persistent buffers (every element is written before it is read)
        auto &fa_all = km_fa, &ga_all = km_ga, &beta_all = km_beta;
        if (fa_all.size() < size_t(T) * fr) fa_all.resize(size_t(T) * fr);
        if (ga_all.size() < size_t(T) * gr) ga_all.resize(size_t(T) * gr);
        if (beta_all.size() < size_t(T) * br) beta_all.resize(size_t(T) * br);
        for (auto &v : raw_all) if (v.size() < size_t(T) * n) v.resize(size_t(T) * n);
        // the whole layer in one team region, as kda_fused: the projections for all tokens, then conv, recurrence,
        // output norm and gate token by token, then o_proj for all tokens
        GemvPlan p1({{&K.q, ax.data(), raw_all[0].data(), T, n}, {&K.k, ax.data(), raw_all[1].data(), T, n},
                     {&K.v, ax.data(), raw_all[2].data(), T, n}, {&K.f_a, af.data(), fa_all.data(), T, fr},
                     {&K.g_a, af.data(), ga_all.data(), T, gr}, {&K.beta, af.data(), beta_all.data(), T, br}});
        const double t_s1 = now();
        auto &afa = km_afa, &aga = km_aga;
        act_rows(afa, T, fr, false), act_rows(aga, T, gr, false);
        auto &decay_all = km_decay, &gate_all = km_gate;
        if (decay_all.size() < size_t(T) * n) decay_all.resize(size_t(T) * n), gate_all.resize(size_t(T) * n);
        GemvPlan p2({{&K.f_b, afa.data(), decay_all.data(), T, n}, {&K.g_b, aga.data(), gate_all.data(), T, n}});
        const double t_s2 = now();
        double t_mid = 0, t_o = 0;
        const bool oq = q8_act(K.o, T);
        auto &ay = km_ay;
        act_rows(ay, T, n, oq);
        std::array<Vec, 3> qkv;
        for (auto &v : qkv) v.resize(n);
        Vec y(n), beta(nh);
        std::vector<float> inv(nh);
        auto &out_all = km_out;
        if (out_all.size() < size_t(T) * K.o.rows) out_all.resize(size_t(T) * K.o.rows);
        const double t_s3 = now();
        GemvPlan p3({{&K.o, ay.data(), out_all.data(), T, K.o.rows}});
        const double t_s4 = now();
        static const bool kda_batch = !(std::getenv("STRATA_RUNNER_KDA_BATCH") && std::string(std::getenv("STRATA_RUNNER_KDA_BATCH")) == "0");
        // the contexts of the pass (pass_groups; else one group, the active context): each group's tokens chain through
        // its context's state, conv histories and rollback buffers; the projections run over every token at once
        struct KG { int g0, g1; float **state; std::array<float *, 4> *pool; Vec *hist; std::vector<MainState> *sink; };
        std::vector<KG> kg;
        const bool grouped = pass_groups.size() > 1 && cur_layer < int(L.size()) && &K == &L[cur_layer].kd;
        if (grouped) {
            if (!kda_batch) throw std::runtime_error("runner: several contexts need the batched KDA phases");
            int g0 = 0;
            for (const auto &[c, len] : pass_groups) {
                LayerState *ls = c == active_ctx ? nullptr : &ctx_store[c].layers[cur_layer];
                kg.push_back({g0, g0 + len, ls ? &ls->state : &K.state, ls ? &ls->pool : &K.pool, ls ? ls->hist : K.hist, group_snaps ? &(*group_snaps)[c] : nullptr});
                g0 += len;
            }
        } else kg.push_back({0, T, &K.state, &K.pool, K.hist, snap_sink});
        const bool any_snap = std::any_of(kg.begin(), kg.end(), [](const KG &g) { return g.sink != nullptr; });
        if (kda_batch) {
            if (kb_qkv.size() < size_t(T) * 3 * n) kb_qkv.resize(size_t(T) * 3 * n);
            if (kb_nq.size() < size_t(T) * n) kb_nq.resize(size_t(T) * n), kb_nk.resize(size_t(T) * n), kb_ed.resize(size_t(T) * n), kb_y.resize(size_t(T) * n);
            kb_qs.resize(T * nh), kb_ks.resize(T * nh), kb_beta.resize(T * nh), kb_inv.resize(T * nh);
            if (any_snap) {
                if (kb_hist.size() < size_t(T)) kb_hist.resize(T);
                for (int tk = 0; tk + 1 < T; ++tk) kb_hist[tk].resize(3 * K.hist[0].size());
            }
        }
        // state buffers per token: in place without snapshots; with them token 0 updates the current state in place and
        // token j > 0 reads token j - 1's and writes a spare buffer of the pool, so the state after every token stays
        std::vector<const float *> sin(T);
        std::vector<float *> sout(T);
        for (const auto &g : kg) {
            float *st = *g.state;
            sin[g.g0] = sout[g.g0] = st;
            for (int tk = g.g0 + 1; tk < g.g1; ++tk) {
                if (!g.sink) { sin[tk] = sout[tk] = st; continue; }
                float *spare = nullptr;
                for (auto &b : *g.pool) {
                    if (!b) b = reinterpret_cast<float *>(arena_alloc(K.state_n * 4));   // first touched by the recurrence's writes
                    if (b != st && std::find(sout.begin() + g.g0, sout.begin() + tk, b) == sout.begin() + tk) { spare = b; break; }
                }
                sin[tk] = sout[tk - 1], sout[tk] = spare;
            }
        }
        const double t_setup = now();
        team_run([&] {
        const int th = omp_get_thread_num(), nth = omp_get_num_threads();
        p1.run(th, nth);
        team_barrier();
        if (th == 0) t_p1 = now();
        if (omp_get_thread_num() == 0)
        for (int t = 0; t < T; ++t) {
            std::copy(fa_all.begin() + size_t(t) * fr, fa_all.begin() + size_t(t + 1) * fr, afa[t].x.begin());
            std::copy(ga_all.begin() + size_t(t) * gr, ga_all.begin() + size_t(t + 1) * gr, aga[t].x.begin());
        }
        team_barrier();
        if (th == 0) t_p2 = now();
        p2.run(th, nth);
        team_barrier();
        if (th == 0) t_mid = now();
        if (kda_batch) {   // the same per-element arithmetic, phase by phase for all tokens (6 barriers, not ~7 per token)
            const int groups = D / KDA_GROUP;
            const bool fx = fast_exp(1) && n % 64 == 0;
            // several contexts: conv items per (part, context, channel), so a thread walks one context's histories
            // contiguously (items per channel over every context interleaved 16 contexts' misses)
            // STRATA_RUNNER_KDA_CONV_CTX=0: per (part, channel)
            static const bool conv_ctx = !(std::getenv("STRATA_RUNNER_KDA_CONV_CTX") && std::string(std::getenv("STRATA_RUNNER_KDA_CONV_CTX")) == "0");
            const int ncg = conv_ctx && kg.size() > 1 ? int(kg.size()) : 1;
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
            // fast exp, a 4-tap conv: items of 16 channels, the same multiply and fused multiply-adds per lane
            // (svld3 / svld4 the stride-3 histories and stride-4 weights). STRATA_RUNNER_KDA_CONV_VEC=0: per channel
            static const bool conv_vec = !(std::getenv("STRATA_RUNNER_KDA_CONV_VEC") && std::string(std::getenv("STRATA_RUNNER_KDA_CONV_VEC")) == "0");
            if (conv_vec && fx && Kc == 4 && n % 16 == 0 && D % 16 == 0) {
                const int nb = n / 16;
                const svbool_t pw = svptrue_b32();
#pragma omp for schedule(static) nowait
                for (int pi2 = 0; pi2 < 3 * ncg * nb + nb; ++pi2) {
                    const bool dec = pi2 >= 3 * ncg * nb;
                    const int part = dec ? 3 : pi2 / (ncg * nb), gq = dec ? 0 : (pi2 / nb) % ncg, i0 = 16 * (dec ? pi2 - 3 * ncg * nb : pi2 % nb);
                    if (part == 3) {   // the sigmoid's argument, -a * (decay + dt)
                        const svfloat32_t na = svdup_n_f32(-K.a_neg[i0 / D]), dt = svld1_f32(pw, K.dt.data() + i0);
                        for (int tk = 0; tk < T; ++tk) {
                            float *dc = decay_all.data() + size_t(tk) * n + i0;
                            svst1_f32(pw, dc, svmul_f32_x(pw, na, svadd_f32_x(pw, svld1_f32(pw, dc), dt)));
                        }
                        continue;
                    }
                    const svfloat32x4_t wv = svld4_f32(pw, K.conv[part].data() + size_t(i0) * 4);
                    const svfloat32_t w0 = svget4_f32(wv, 0), w1 = svget4_f32(wv, 1), w2 = svget4_f32(wv, 2), w3 = svget4_f32(wv, 3);
                    for (size_t gk = ncg > 1 ? size_t(gq) : 0; gk < (ncg > 1 ? size_t(gq) + 1 : kg.size()); ++gk) {
                        const auto &g = kg[gk];
                        float *hp = g.hist[part].data() + size_t(i0) * 3;
                        const svfloat32x3_t hv = svld3_f32(pw, hp);
                        svfloat32_t h0 = svget3_f32(hv, 0), h1 = svget3_f32(hv, 1), h2 = svget3_f32(hv, 2);
                        for (int tk = g.g0; tk < g.g1; ++tk) {
                            const svfloat32_t r = svld1_f32(pw, raw_all[part].data() + size_t(tk) * n + i0);
                            svfloat32_t sum = svmul_f32_x(pw, r, w3);
                            sum = svmla_f32_x(pw, sum, h0, w0), sum = svmla_f32_x(pw, sum, h1, w1), sum = svmla_f32_x(pw, sum, h2, w2);
                            h0 = h1, h1 = h2, h2 = r;
                            svst1_f32(pw, kb_qkv.data() + (size_t(tk) * 3 + part) * n + i0, sum);
                            if (g.sink && tk + 1 < g.g1)   // the conv history after this token, for a rollback to it
                                svst3_f32(pw, kb_hist[tk].data() + size_t(part) * g.hist[part].size() + size_t(i0) * 3, svcreate3_f32(h0, h1, h2));
                        }
                        svst3_f32(pw, hp, svcreate3_f32(h0, h1, h2));
                    }
                }
            } else
#endif
#pragma omp for schedule(static) nowait
            for (int pi2 = 0; pi2 < 3 * ncg * n + n; ++pi2) {   // conv (each channel's tokens in order) and the decay
                const bool dec = pi2 >= 3 * ncg * n;
                const int part = dec ? 3 : pi2 / (ncg * n), gq = dec ? 0 : (pi2 / n) % ncg, i = dec ? pi2 - 3 * ncg * n : pi2 % n;
                if (part == 3) {
                    for (int tk = 0; tk < T; ++tk) {
                        float &dc = decay_all[size_t(tk) * n + i];
                        if (fx) dc = -K.a_neg[i / D] * (dc + K.dt[i]);   // the sigmoid's argument (vector pass below)
                        else dc = m.gate_lower_bound * ref::sigmoid(-K.a_neg[i / D] * (dc + K.dt[i]));
                    }
                    continue;
                }
                const auto &w = K.conv[part];
                for (size_t gk = ncg > 1 ? size_t(gq) : 0; gk < (ncg > 1 ? size_t(gq) + 1 : kg.size()); ++gk) {   // each context's tokens through its own history
                    const auto &g = kg[gk];
                    auto &hist = g.hist[part];
                    for (int tk = g.g0; tk < g.g1; ++tk) {
                        const float r = raw_all[part][size_t(tk) * n + i];
                        float sum = r * w[i * Kc + Kc - 1];
                        for (int j = 0; j < Kc - 1; ++j) sum += hist[i * (Kc - 1) + j] * w[i * Kc + j];
                        for (int j = 0; j < Kc - 2; ++j) hist[i * (Kc - 1) + j] = hist[i * (Kc - 1) + j + 1];
                        hist[i * (Kc - 1) + Kc - 2] = r;
                        kb_qkv[(size_t(tk) * 3 + part) * n + i] = fx ? sum : sum * ref::sigmoid(sum);
                        if (g.sink && tk + 1 < g.g1)   // the conv history after this token, for a rollback to it
                            for (int j = 0; j < Kc - 1; ++j) kb_hist[tk][size_t(part) * hist.size() + i * (Kc - 1) + j] = hist[i * (Kc - 1) + j];
                    }
                }
            }
            team_barrier();
            if (th == 0) kb_tp[5] = now();
            if (fx) {   // fast exp: SiLU of the conv outputs, the decay, its exp, the output gates' sigmoid, 64 values per item
                const int bq = T * 3 * n / 64, bd = T * n / 64;
#pragma omp for schedule(static) nowait
                for (int b = 0; b < bq + 2 * bd; ++b) {
                    if (b < bq) gf4_vsilu(kb_qkv.data() + 64 * size_t(b), kb_qkv.data() + 64 * size_t(b), 64);
                    else if (b < bq + bd) {
                        float *dc = decay_all.data() + 64 * size_t(b - bq);
                        gf4_vsigmoid(dc, dc, 64);
                        for (int j = 0; j < 64; ++j) dc[j] *= m.gate_lower_bound;
                        gf4_vexp(kb_ed.data() + 64 * size_t(b - bq), dc, 64);
                    } else {
                        float *gt = gate_all.data() + 64 * size_t(b - bq - bd);
                        gf4_vsigmoid(gt, gt, 64);
                    }
                }
                team_barrier();
            }
            if (th == 0) kb_tp[0] = now();
#pragma omp for schedule(static) nowait
            for (int th_ = 0; th_ < T * nh; ++th_) {   // kda_core_team's per-head sums (same order)
                const int tk = th_ / nh, h = th_ % nh;
                const float *q = kb_qkv.data() + size_t(tk) * 3 * n, *k = q + n;
                float qs = 1e-6f, ks = 1e-6f;
                for (int i = 0; i < D; ++i) qs += q[h * D + i] * q[h * D + i], ks += k[h * D + i] * k[h * D + i];
                kb_qs[th_] = qs, kb_ks[th_] = ks;
                kb_beta[th_] = beta_all[size_t(tk) * br + heads.begin + h];
            }
            team_barrier();
            if (th == 0) kb_tp[1] = now();
#pragma omp for schedule(static) nowait
            for (int ti = 0; ti < T * n; ++ti) {
                const int tk = ti / n, hi = ti % n, h = hi / D;
                const float *q = kb_qkv.data() + size_t(tk) * 3 * n, *k = q + n;
                kb_nq[ti] = q[hi] / std::sqrt(kb_qs[tk * nh + h]) / std::sqrt((float)D);
                kb_nk[ti] = k[hi] / std::sqrt(kb_ks[tk * nh + h]);
                if (!fx) kb_ed[ti] = std::exp(decay_all[size_t(tk) * n + hi]);
            }
            team_barrier();
            if (th == 0) kb_tp[2] = now();
            const bool halves = kda_halves() && nh * groups > nth;   // 48 groups on 47 threads: 96 halves (opt-in)
            // several contexts: items per (column group, context), each walking only its context's tokens (the
            // contexts are independent; one item walking every context's tokens chained ~34 state loads in a row).
            // STRATA_RUNNER_KDA_REC_CTX=0: items per column group over all tokens
            static const bool rec_ctx = !(std::getenv("STRATA_RUNNER_KDA_REC_CTX") && std::string(std::getenv("STRATA_RUNNER_KDA_REC_CTX")) == "0");
            const int ngr = rec_ctx && kg.size() > 1 ? int(kg.size()) : 1;
            // G column groups of a head per item, their chains interleaved (kda_groups_step): G from the rounds of
            // items per thread times a group step's cost at G (1, 1.1, 1.3 add-latency units; STRATA_RUNNER_KDA_REC_G:
            // fixed 1, 2 or 4)
            static const int rec_g = std::getenv("STRATA_RUNNER_KDA_REC_G") ? std::atoi(std::getenv("STRATA_RUNNER_KDA_REC_G")) : 0;
            int G = 1;
            if (!halves) {
                double best = 1e30;
                for (int g : {1, 2, 4}) {
                    if (groups % g || (rec_g && g != rec_g)) continue;
                    const int items = nh * (groups / g) * ngr;
                    const double cost = double((items + nth - 1) / nth) * (g == 1 ? 1.0 : g == 2 ? 1.1 : 1.3);
                    if (cost < best) best = cost, G = g;
                }
            }
            const int gpi = groups / G;   // items per head
#pragma omp for schedule(static) nowait
            for (int it2 = 0; it2 < (halves ? 2 : 1) * nh * gpi * ngr; ++it2) {   // each column group's tokens in order
                const int it = it2 / ngr, gq = it2 % ngr;
                const int hg = halves ? it / 2 : (it / gpi) * groups + (it % gpi) * G, lo = halves ? 8 * (it % 2) : 0;
                const int h = hg / groups, j0 = (hg % groups) * KDA_GROUP;
                const int tk0 = ngr > 1 ? kg[gq].g0 : 0, tk1 = ngr > 1 ? kg[gq].g1 : T;
                for (int tk = tk0; tk < tk1; ++tk) {
                    const float *S0 = sin[tk] + size_t(hg) * D * KDA_GROUP;
                    float *S = sout[tk] + size_t(hg) * D * KDA_GROUP;
                    const float *kh = kb_nk.data() + size_t(tk) * n + h * D, *qh = kb_nq.data() + size_t(tk) * n + h * D, *eh = kb_ed.data() + size_t(tk) * n + h * D;
                    const float *vv = kb_qkv.data() + (size_t(tk) * 3 + 2) * n + h * D + j0;
                    float *oy = kb_y.data() + size_t(tk) * n + h * D + j0;
                    const float sb = ref::sigmoid(kb_beta[tk * nh + h]);
                    if (halves) kda_half_step(S0, S, kh, qh, eh, vv, sb, oy, D, lo);
                    else if (G == 4) kda_groups_step<4>(S0, S, kh, qh, eh, vv, sb, oy, D);
                    else if (G == 2) kda_groups_step<2>(S0, S, kh, qh, eh, vv, sb, oy, D);
                    else kda_group_step(S0, S, kh, qh, eh, vv, sb, oy, D);
                }
            }
            team_barrier();
            if (th == 0) kb_tp[3] = now();
#pragma omp for schedule(static) nowait
            for (int th_ = 0; th_ < T * nh; ++th_) {
                const int tk = th_ / nh, h = th_ % nh;
                const float *y = kb_y.data() + size_t(tk) * n;
                double sum = 0;
                for (int i = 0; i < D; ++i) sum += double(y[h * D + i]) * y[h * D + i];
                kb_inv[th_] = float(1 / std::sqrt(sum / D + m.rms_epsilon));
            }
            team_barrier();
            if (th == 0) kb_tp[4] = now();
#pragma omp for schedule(static) nowait
            for (int tb = 0; tb < T * (n / 64); ++tb) {
                const int tk = tb / (n / 64), b = tb % (n / 64);
                Activation &a = ay[tk];
                const float *y = kb_y.data() + size_t(tk) * n, *gate = gate_all.data() + size_t(tk) * n;
                if (fx)
                    for (int j = b * 64; j < b * 64 + 64; ++j) a.x[j] = y[j] * kb_inv[tk * nh + j / D] * K.onorm[j % D] * gate[j];
                else
                    for (int j = b * 64; j < b * 64 + 64; ++j) a.x[j] = y[j] * kb_inv[tk * nh + j / D] * K.onorm[j % D] * ref::sigmoid(gate[j]);
                if (oq) gf4_quantize_q8(a.x.data() + b * 64, 64, a.q.data() + b * 64, a.qs.data() + b);
            }
            if (any_snap && th == 0)
                for (const auto &g : kg)
                    for (int tk = g.g0; g.sink && tk + 1 < g.g1; ++tk) {
                        auto &st = snap_in(*g.sink, tk - g.g0);
                        st.kda_state[cur_layer] = sout[tk];
                        st.kda_hist[cur_layer].assign(kb_hist[tk].begin(), kb_hist[tk].begin() + 3 * K.hist[0].size());
                    }
        } else
        for (int tk = 0; tk < T; ++tk) {   // conv, recurrence, output norm and gate: token by token (kda_fused's code)
            const float *raws[3] = {raw_all[0].data() + size_t(tk) * n, raw_all[1].data() + size_t(tk) * n, raw_all[2].data() + size_t(tk) * n};
            float *decay = decay_all.data() + size_t(tk) * n;
            const float *gate = gate_all.data() + size_t(tk) * n;
            Activation &a = ay[tk];
            {
#pragma omp for schedule(static) nowait
                for (int pi = 0; pi < 4 * n; ++pi) {
                    const int part = pi / n, i = pi % n;
                    if (part == 3) {
                        decay[i] = m.gate_lower_bound * ref::sigmoid(-K.a_neg[i / D] * (decay[i] + K.dt[i]));
                        continue;
                    }
                    const float *raw = raws[part];
                    const auto &w = K.conv[part];
                    auto &hist = K.hist[part];
                    float sum = raw[i] * w[i * Kc + Kc - 1];
                    for (int j = 0; j < Kc - 1; ++j) sum += hist[i * (Kc - 1) + j] * w[i * Kc + j];
                    for (int j = 0; j < Kc - 2; ++j) hist[i * (Kc - 1) + j] = hist[i * (Kc - 1) + j + 1];
                    hist[i * (Kc - 1) + Kc - 2] = raw[i];
                    qkv[part][i] = sum * ref::sigmoid(sum);
                }
                team_barrier();
#pragma omp single nowait
                for (int h = 0; h < nh; ++h) beta[h] = beta_all[size_t(tk) * br + heads.begin + h];
                kda_core_team(kda_w, sin[tk], sout[tk], qkv[0].data(), qkv[1].data(), qkv[2].data(), decay, beta.data(), y.data(), nh, D);
                if (omp_get_thread_num() == 0)
                for (int h = 0; h < nh; ++h) {
                    double sum = 0;
                    for (int i = 0; i < D; ++i) sum += double(y[h * D + i]) * y[h * D + i];
                    inv[h] = float(1 / std::sqrt(sum / D + m.rms_epsilon));
                }
                team_barrier();
#pragma omp for schedule(static) nowait
                for (int b = 0; b < n / 64; ++b) {
                    for (int j = b * 64; j < b * 64 + 64; ++j) a.x[j] = y[j] * inv[j / D] * K.onorm[j % D] * ref::sigmoid(gate[j]);
                    if (oq) gf4_quantize_q8(a.x.data() + b * 64, 64, a.q.data() + b * 64, a.qs.data() + b);
                }
                team_barrier();
                if (snap_sink && tk + 1 < T) {   // the state after this token stays in sout[tk]; the conv history is copied
                    if (omp_get_thread_num() == 0)
                    {
                        auto &st = snap_at(tk);
                        st.kda_state[cur_layer] = sout[tk];
                        const size_t hn = K.hist[0].size();
                        st.kda_hist[cur_layer].resize(3 * hn);
                        for (int i = 0; i < 3; ++i) std::copy(K.hist[i].begin(), K.hist[i].end(), st.kda_hist[cur_layer].begin() + i * hn);
                    }
                    team_barrier();
                }
            }
        }
        team_barrier();
        if (th == 0) t_o = now();
        p3.run(th, nth);
        });
        for (const auto &g : kg) *g.state = sout[g.g1 - 1];   // the state after each context's last token (restore() picks an earlier one)
        const double t_out = now();
        timers["kda_proj"] += (t_mid - t_in) + (t_out - t_o);
        timers["kdap.acts"] += t_acts - t_in, timers["kdap.setup"] += t_setup - t_acts, timers["kdap.p1"] += t_p1 - t_setup;
        timers["kdap.copy"] += t_p2 - t_p1, timers["kdap.p2"] += t_mid - t_p2, timers["kdap.o"] += t_out - t_o;
        timers["kdap.s_p1"] += t_s1 - t_acts, timers["kdap.s_p2"] += t_s2 - t_s1, timers["kdap.s_bufs"] += t_s3 - t_s2;
        timers["kdap.s_p3"] += t_s4 - t_s3, timers["kdap.s_state"] += t_setup - t_s4;
        timers["kda_small"] += t_o - t_mid;
        if (kda_batch) {
            static const char *nm[5] = {"kdam.conv", "kdam.sums", "kdam.norm", "kdam.rec", "kdam.inv"};
            double prev = t_mid;
            for (int i = 0; i < 5; ++i) timers[nm[i]] += kb_tp[i] - prev, prev = kb_tp[i];
            timers["kdam.gate"] += t_o - prev;
            timers["kdam.conv_loop"] += kb_tp[5] - t_mid;
        }
        return split_rows(out_all, T, K.o.rows, K.o.rows);
    }
    // the MLA projections of a pass (and their all-reduces when the front is sharded): no cache or position state, so
    // a pipelined pass runs both halves' fronts before either half's attention
    // In two stages, each ending in an all-reduce (sharded fronts) that async posts on the comm thread: stage 1 the
    // x projections, stage 2 (after its wait) the norms and q_b / iq; mla_back waits for stage 2's.
    struct MlaFront { std::vector<Vec> qr, row; Vec q, iq, iw, ik, gate, qa, kva, buf; long tk1 = 0, tk2 = 0; int T = 0; bool async = false; };
    MlaFront mla_front(MlaLayer &M, const std::vector<Vec> &xs) {
        MlaFront F;
        mla_front1(M, xs, F, false);
        mla_front2(M, F);
        return F;
    }
    void mla_front1(MlaLayer &M, const std::vector<Vec> &xs, MlaFront &Fr, bool async) {
        const double t = now();
        const int T = int(xs.size()), H = m.hidden;
        Fr.T = T, Fr.async = async;
        auto &ax = acts_pool(1, xs, H, trunk_q8);
        auto &aq_x = ax;   // float kernels read the x of the same activations (the front stays exact: K-split matrices)
        // the indexer projections are BF16 in the main layers (float activations) but FP8 in the draft block, which
        // fast mode requantizes to int8: those take the int8 activations
        auto ix = [&](const Matrix &w) { return needs_q8(w) ? ax.data() : aq_x.data(); };
        const int qa_r = M.full[0], kv_r = M.full[1], iw_r = M.full[2], ik_r = M.full[3], ig_r = M.full[4];
        auto &qa = Fr.qa, &kva = Fr.kva;
        qa.assign(size_t(T) * qa_r, 0.f), kva.assign(size_t(T) * kv_r, 0.f);
        auto &iw = Fr.iw, &ik = Fr.ik, &gate = Fr.gate;
        iw.assign(size_t(T) * iw_r, 0.f), ik.assign(size_t(T) * ik_r, 0.f), gate.assign(size_t(T) * ig_r, 0.f);
        if (!M.sharded)
            gemv_multi({{&M.q_a, ax.data(), qa.data(), T, qa_r}, {&M.kv_a, ax.data(), kva.data(), T, kv_r},
                        {&M.iw, ix(M.iw), iw.data(), T, iw_r}, {&M.ik, ix(M.ik), ik.data(), T, ik_r},
                        {&M.igate, ix(M.igate), gate.data(), T, ig_r}});
        else {   // this rank's rows into a zero row vector per token, then one all-reduce (exact: one value plus zeros)
            const int F = qa_r + kv_r + iw_r + ik_r + ig_r, off[5] = {0, qa_r, qa_r + kv_r, qa_r + kv_r + iw_r, qa_r + kv_r + iw_r + ik_r};
            auto &buf = Fr.buf;
            buf.assign(size_t(T) * F, 0.f);
            const Matrix *mats[5] = {&M.q_a, &M.kv_a, &M.iw, &M.ik, &M.igate};
            std::vector<Job> jobs;
            for (int i = 0; i < 5; ++i)
                if (mats[i]->rows) jobs.push_back({mats[i], i < 2 ? ax.data() : ix(*mats[i]), buf.data() + off[i] + M.r0[i], T, F});
            gemv_multi(jobs);
            const double tc = now();
            if (async) Fr.tk1 = comm.allreduce_async(buf.data(), T * F);
            else comm.allreduce(buf.data(), T * F);
            timers["comm_mla_front"] += now() - tc;
        }
        timers["mla_proj"] += now() - t;
    }
    void mla_front2(MlaLayer &M, MlaFront &F) {
        const double t = now();
        const int T = F.T;
        const int qa_r = M.full[0], kv_r = M.full[1], iw_r = M.full[2], ik_r = M.full[3], ig_r = M.full[4];
        auto &qa = F.qa, &kva = F.kva;
        if (M.sharded) {
            const double tc = now();
            comm.ar_wait(F.tk1);
            timers["comm_mla_front"] += now() - tc;
            const int Fw = qa_r + kv_r + iw_r + ik_r + ig_r, off[5] = {0, qa_r, qa_r + kv_r, qa_r + kv_r + iw_r, qa_r + kv_r + iw_r + ik_r};
            auto &buf = F.buf;
            Vec *outs[5] = {&qa, &kva, &F.iw, &F.ik, &F.gate};
            const int sizes[5] = {qa_r, kv_r, iw_r, ik_r, ig_r};
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < 5; ++i)
                    std::copy(buf.begin() + size_t(t) * Fw + off[i], buf.begin() + size_t(t) * Fw + off[i] + sizes[i], outs[i]->begin() + size_t(t) * sizes[i]);
            Vec().swap(buf);
        }
        auto &qr = F.qr, &row = F.row;
        qr.resize(T), row.resize(T);
        for (int tk = 0; tk < T; ++tk) {
            qr[tk] = norm(Vec(qa.begin() + size_t(tk) * qa_r, qa.begin() + size_t(tk + 1) * qa_r), M.q_a_norm);
            row[tk] = norm(Vec(kva.begin() + size_t(tk) * kv_r, kva.begin() + size_t(tk + 1) * kv_r), M.kv_a_norm);
        }
        auto &aqr = acts_pool(2, qr, int(qr[0].size()), trunk_q8 || q8_act(M.q_b, T));
        auto &aqrf = aqr;
        const int qb_r = M.q_b.rows, iq_r = M.full[5];
        auto &q = F.q, &iq = F.iq;
        q.assign(size_t(T) * qb_r, 0.f), iq.assign(size_t(T) * iq_r, 0.f);
        auto iqa = needs_q8(M.iq) ? aqr.data() : aqrf.data();
        if (!M.sharded) gemv_multi({{&M.q_b, aqr.data(), q.data(), T, qb_r}, {&M.iq, iqa, iq.data(), T, iq_r}});
        else {
            std::vector<Job> jobs{{&M.q_b, aqr.data(), q.data(), T, qb_r}};
            if (M.iq.rows) jobs.push_back({&M.iq, iqa, iq.data() + M.r0[5], T, iq_r});
            gemv_multi(jobs);
            const double tc = now();
            if (F.async) F.tk2 = comm.allreduce_async(iq.data(), T * iq_r);
            else comm.allreduce(iq.data(), T * iq_r);
            timers["comm_mla_front"] += now() - tc;
        }
        Vec().swap(qa), Vec().swap(kva);
        timers["mla_proj"] += now() - t;
    }
    std::vector<Vec> mla_multi(MlaLayer &M, const std::vector<Vec> &xs, int pos0 = -1) {   // pos0: first token's position
        auto F = mla_front(M, xs);
        return mla_back(M, F, pos0);
    }
    std::vector<Vec> mla_back(MlaLayer &M, MlaFront &F, int pos0 = -1) {
        if (pos0 < 0) pos0 = position;
        double t = now();
        const int T = int(F.qr.size()), H = m.hidden;
        (void)H;
        const int iw_r = M.full[2], ik_r = M.full[3], ig_r = M.full[4], qb_r = M.q_b.rows, iq_r = M.full[5];
        (void)iw_r, (void)ik_r, (void)ig_r, (void)qb_r, (void)iq_r;
        if (F.tk2) {
            const double tc = now();
            comm.ar_wait(F.tk2);
            timers["comm_mla_front"] += now() - tc;
        }
        auto &qr = F.qr, &row = F.row;
        auto &q = F.q, &iq = F.iq, &iw = F.iw, &ik = F.ik, &gate = F.gate;
        (void)qr, (void)row, (void)q, (void)iq, (void)iw, (void)ik, (void)gate;
        std::vector<Vec> values;
        std::vector<std::vector<int>> sel(T);
        std::vector<Vec> qs(T);
        std::vector<const Vec *> qp(T);
        static const bool select_multi = !(std::getenv("STRATA_RUNNER_INDEX_BATCH") && std::string(std::getenv("STRATA_RUNNER_INDEX_BATCH")) == "0");
        auto snapshot = [&](int tk) {   // the index key tail after position tk, for a rollback to it
            if (snap_sink && tk + 1 < T && cur_layer < int(L.size()) && &M == &L[cur_layer].ml) {
                auto &st = snap_at(tk);
                st.mla_keys[cur_layer] = M.keys, st.mla_gates[cur_layer] = M.gates;
            }
        };
        for (int tk = 0; tk < T; ++tk) qs[tk] = Vec(q.begin() + size_t(tk) * qb_r, q.begin() + size_t(tk + 1) * qb_r), qp[tk] = &qs[tk];
        const bool mtp_grouped = mtp.on && &M == &mtp.layer.ml && mtp_cells.size() == pass_groups.size();
        if (pass_groups.size() > 1 && ((cur_layer < int(L.size()) && &M == &L[cur_layer].ml) || mtp_grouped)) {
            // several contexts: selection and attention per context, with that context's index tails, pools and caches
            // swapped into this layer and its own positions; the projections above and below ran over every token.
            // The MTP block's layer (mtp_forward_ctx): each context's draft layer state and its own first cell
            if (!(select_multi && ik_r == m.index_dim && ig_r == m.index_dim)) throw std::runtime_error("runner: several contexts need the batched indexer");
            values.resize(T);
            // the groups' attention in one mla_attention call (STRATA_RUNNER_GROUP_ATTN=0: one call per group): the
            // per-group calls cost ~190 us each for ~2 tokens (their team regions); a cache row's address is the
            // context's own (each context's cache keeps its buffer when swapped out)
            static const bool group_attn = !(std::getenv("STRATA_RUNNER_GROUP_ATTN") && std::string(std::getenv("STRATA_RUNNER_GROUP_ATTN")) == "0");
            std::vector<int> ga_tok;
            std::vector<std::vector<int>> ga_ids;
            std::vector<const float *> ga_cache;
            // every context's selection at once, one group per item, when no group needs a team region of its own
            // (at most 8 tokens, within the selection size: no pool scores, no hybrid cache) and the attention is
            // grouped: each group with its own context's state (STRATA_RUNNER_GROUP_SELECT=0: one after another)
            static const bool group_select = !(std::getenv("STRATA_RUNNER_GROUP_SELECT") && std::string(std::getenv("STRATA_RUNNER_GROUP_SELECT")) == "0");
            bool par_sel = group_select && group_attn && !M.ring && index_skip();
            for (size_t gi = 0; gi < pass_groups.size() && par_sel; ++gi) {
                const auto [c, len] = pass_groups[gi];
                const int pc = mtp_grouped ? mtp_cells[gi] : c == active_ctx ? position : ctx_store[c].position;
                if (len > 8 || pc + len > int(m.index_top_k)) par_sel = false;
            }
            if (par_sel) {
                std::vector<std::vector<std::vector<int>>> sgs(pass_groups.size());
                std::vector<int> gstart(pass_groups.size() + 1, 0);
                for (size_t gi = 0; gi < pass_groups.size(); ++gi) gstart[gi + 1] = gstart[gi] + pass_groups[gi].second;
                const double tsel = now();
                team_run([&] {
                    lap_off = true;
#pragma omp for schedule(dynamic, 1) nowait
                    for (int gi = 0; gi < int(pass_groups.size()); ++gi) {
                        const auto [c, len] = pass_groups[gi];
                        const int g0 = gstart[gi], g1 = g0 + len;
                        LayerState *ls = c == active_ctx ? nullptr : mtp_grouped ? &ctx_store[c].mtp_layer : &ctx_store[c].layers[cur_layer];
                        const int pc = mtp_grouped ? mtp_cells[gi] : ls ? ctx_store[c].position : position;
                        auto sub = [&](const Vec &all, int rows) { return Vec(all.begin() + size_t(g0) * rows, all.begin() + size_t(g1) * rows); };
                        std::vector<Vec> rg(row.begin() + g0, row.begin() + g1);
                        std::vector<MainState> *sink = group_snaps && !mtp_grouped ? &(*group_snaps)[c] : nullptr;
                        const Vec &gk = ls ? ls->keys : M.keys, &gg = ls ? ls->gates : M.gates;
                        auto snap_g = [&](int j) {
                            if (sink && j + 1 < len) { auto &st = snap_in(*sink, j); st.mla_keys[cur_layer] = gk, st.mla_gates[cur_layer] = gg; }
                        };
                        sgs[gi] = mla_select_multi(M, sub(iq, iq_r), sub(iw, iw_r), sub(ik, ik_r), sub(gate, ig_r), rg, pc,
                                                   sink ? std::function<void(int)>(snap_g) : std::function<void(int)>(), ls);
                    }
                    lap_off = false;
                });
                timers["mla.index"] += now() - tsel;
                for (size_t gi = 0; gi < pass_groups.size(); ++gi) {
                    const auto [c, len] = pass_groups[gi];
                    LayerState *ls = c == active_ctx ? nullptr : mtp_grouped ? &ctx_store[c].mtp_layer : &ctx_store[c].layers[cur_layer];
                    for (int t2 = 0; t2 < len; ++t2)
                        ga_tok.push_back(gstart[gi] + t2), ga_ids.push_back(std::move(sgs[gi][t2])), ga_cache.push_back(ls ? ls->cache.data() : M.cache.data());
                }
            }
            int g0 = 0;
            for (size_t gi = 0; gi < pass_groups.size() && !par_sel; ++gi) {
                const auto [c, len] = pass_groups[gi];
                const int g1 = g0 + len;
                LayerState *ls = c == active_ctx ? nullptr : mtp_grouped ? &ctx_store[c].mtp_layer : &ctx_store[c].layers[cur_layer];
                Layer &LY = mtp_grouped ? mtp.layer : L[cur_layer];
                if (ls) swap_layer(LY, *ls);
                const int pc = mtp_grouped ? mtp_cells[gi] : ls ? ctx_store[c].position : position;
                auto sub = [&](const Vec &all, int rows) { return Vec(all.begin() + size_t(g0) * rows, all.begin() + size_t(g1) * rows); };
                std::vector<Vec> rg(row.begin() + g0, row.begin() + g1);
                std::vector<MainState> *sink = group_snaps && !mtp_grouped ? &(*group_snaps)[c] : nullptr;
                auto snap_g = [&](int j) {
                    if (sink && j + 1 < len) { auto &st = snap_in(*sink, j); st.mla_keys[cur_layer] = M.keys, st.mla_gates[cur_layer] = M.gates; }
                };
                // (the callback only when snapshots are taken: an empty one lets the keys go over the team)
                auto sg = mla_select_multi(M, sub(iq, iq_r), sub(iw, iw_r), sub(ik, ik_r), sub(gate, ig_r), rg, pc,
                                           sink ? std::function<void(int)>(snap_g) : std::function<void(int)>());
                if (M.ring && pc + len > M.ring) {   // the hybrid cache: this group's attention on its own
                    std::vector<const Vec *> qg(qp.begin() + g0, qp.begin() + g1);
                    std::vector<Vec> vg;
                    mla_attention_hybrid(M, qg, sg, vg, pc + len - M.ring);
                    for (int t2 = 0; t2 < len; ++t2) values[g0 + t2] = std::move(vg[t2]);
                } else if (group_attn) {   // the attention of every such group in one call below, per-token caches
                    for (int t2 = 0; t2 < len; ++t2) ga_tok.push_back(g0 + t2), ga_ids.push_back(std::move(sg[t2])), ga_cache.push_back(M.cache.data());
                } else {
                    std::vector<const Vec *> qg(qp.begin() + g0, qp.begin() + g1);
                    std::vector<Vec> vg;
                    mla_attention(M, qg, sg, vg);
                    for (int t2 = 0; t2 < len; ++t2) values[g0 + t2] = std::move(vg[t2]);
                }
                if (ls) swap_layer(LY, *ls);
                g0 = g1;
            }
            if (!ga_tok.empty()) {   // one attention call for the groups' tokens, each reading its own context's cache
                std::vector<const Vec *> qg;
                for (int t2 : ga_tok) qg.push_back(qp[t2]);
                std::vector<Vec> vg;
                mla_tcache = ga_cache.data();
                mla_attention(M, qg, ga_ids, vg);
                mla_tcache = nullptr;
                for (size_t i = 0; i < ga_tok.size(); ++i) values[ga_tok[i]] = std::move(vg[i]);
            }
        } else if (select_multi && ik_r == m.index_dim && ig_r == m.index_dim)
            sel = mla_select_multi(M, iq, iw, ik, gate, row, pos0,
                                   snap_sink && cur_layer < int(L.size()) && &M == &L[cur_layer].ml ? std::function<void(int)>(snapshot) : std::function<void(int)>());
        else
            for (int tk = 0; tk < T; ++tk) {   // index steps and cache rows in position order, then attention for all tokens
                auto slice = [&](const Vec &all, int rows) { return Vec(all.begin() + size_t(tk) * rows, all.begin() + size_t(tk + 1) * rows); };
                sel[tk] = mla_select(M, slice(iq, iq_r), slice(iw, iw_r), slice(ik, ik_r), slice(gate, ig_r), row[tk], pos0 + tk);
                snapshot(tk);
            }
        if (values.empty()) {
            if (M.ring && pos0 + T > M.ring) mla_attention_hybrid(M, qp, sel, values, pos0 + T - M.ring);
            else if (T <= 128) mla_attention(M, qp, sel, values);
            else {   // long passes in blocks of 128 tokens: the score and probability arrays stay 128 x positions
                values.resize(T);
                for (int b0 = 0; b0 < T; b0 += 128) {
                    const int b1 = std::min(T, b0 + 128);
                    std::vector<const Vec *> qb(qp.begin() + b0, qp.begin() + b1);
                    std::vector<std::vector<int>> sb(sel.begin() + b0, sel.begin() + b1);
                    std::vector<Vec> vb;
                    mla_attention(M, qb, sb, vb);
                    for (int t2 = b0; t2 < b1; ++t2) values[t2] = std::move(vb[t2 - b0]);
                }
            }
        }
        timers["mla_attn"] += now() - t;
        t = now();
        auto &av = acts_pool(5, values, int(values[0].size()), q8_act(M.o, T));
        Vec out(size_t(T) * M.o.rows);
        gemv_multi({{&M.o, av.data(), out.data(), T, M.o.rows}});
        timers["mla_proj"] += now() - t;
        return split_rows(out, T, M.o.rows, M.o.rows);
    }
    std::vector<Vec> dense_multi(FfnDense &F, const std::vector<Vec> &xs, float limit) {
        const double t = now();
        const int T = int(xs.size()), gr = F.gate.rows;
        auto &ax = acts_pool(4, xs, m.hidden, trunk_q8 || q8_act(F.gate, T));
        Vec g(size_t(T) * gr), u(size_t(T) * gr);
        gemv_multi({{&F.gate, ax.data(), g.data(), T, gr}, {&F.up, ax.data(), u.data(), T, gr}});
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (size_t i = 0; i < g.size(); ++i) g[i] = ref::swiglu(g[i], u[i], limit);
        });
        auto &ag = acts_pool(6, split_rows(g, T, gr, gr), gr, q8_act(F.down, T));
        Vec out(size_t(T) * F.down.rows);
        gemv_multi({{&F.down, ag.data(), out.data(), T, F.down.rows}});
        timers["dense"] += now() - t;
        return split_rows(out, T, F.down.rows, F.down.rows);
    }
    struct MoeUse {   // a slice in a pass; gp, upp, yp, ap: its gate/up/down outputs and SwiGLU activations (its own
                      // vectors, or in a long pass rows of the flat buffers below)
        int slice = -1; std::vector<int> tokens; std::vector<Activation> act; Vec g, u, y;
        float *gp = nullptr, *upp = nullptr, *yp = nullptr; Activation *ap = nullptr;
    };
    struct FlatBuf {   // uninitialized floats that grow and stay (first touched by the kernels that write them)
        std::unique_ptr<float[]> p; size_t n = 0;
        float *get(size_t want) { if (want > n) p.reset(new float[want]), n = want; return p.get(); }
    };
    FlatBuf moe_fg, moe_fu, moe_fy;
    std::vector<int> moe_cnt, moe_flat_tok, moe_slice_of, moe_fill;   // the long-pass plan's counting sort   // long passes: every (use, token) row of gate, up and down outputs
    std::vector<Activation> moe_fact;
    std::vector<int> moe_row_off;
    std::vector<MoeUse> moe_use_pool;
    std::vector<int> moe_use_of, moe_block_off;   // long passes: slice -> use, each use's first SwiGLU block
    AlignedBuf<int8_t> moe_xt, moe_xtd;   // long passes: the uses' transposed activations for gate / up and for down
    AlignedBuf<float> moe_xs, moe_xsd;
    std::vector<size_t> moe_xblk;         // each use's first 16-token block in them (gf4_w4a8_gemm_blocks of its tokens)
    // moe_multi in one team region (as moe_fused for one token): every GEMV share, the routing on one thread, SwiGLU
    // and the weighted sums over the team; the same arithmetic per token as moe_multi
    std::vector<Vec> moe_multi_fused(MoeLayer &E, const std::vector<Vec> &xs, const strata::core::LayerDescriptor &desc) {
        const double t_in = now();
        const int T = int(xs.size()), H = m.hidden;
        const bool quant = moe_quant(E);
        auto &ax = acts_pool(3, xs, H, quant, g_act_group);
        auto &af = ax;   // the router's float kernel reads x
        const double t_acts = now();
        double t_gemv = 0;   // thread 0 after the first GEMVs (router, shared gate / up)
        const int er = m.experts, sr = E.sgate.rows;
        const bool rsh = E.router.rows != m.experts;
        Vec logits(size_t(T) * er + 16, 0.f), sg(size_t(T) * std::max(sr, 1)), su(size_t(T) * std::max(sr, 1)), rscore(size_t(T) * er), rkey(size_t(T) * er);
        std::vector<Job> first{{&E.router, af.data(), logits.data() + E.router_r0, T, er}};
        if (sr) first.push_back({&E.sgate, ax.data(), sg.data(), T, sr}), first.push_back({&E.sup, ax.data(), su.data(), T, sr});
        std::vector<std::vector<int>> ids(T, std::vector<int>(m.top_k)), mine(T);
        std::vector<Vec> routing(T, Vec(m.top_k)), sum(T, Vec(H));
        auto &pool = moe_use_pool;   // persistent: buffers grow once, no allocation or zero fill per layer
        int n_uses = 0;
        std::vector<Job> up, down;
        std::vector<std::array<int, 3>> blocks;   // (use or -1 for the shared expert, token, first row)
        std::vector<Activation> sact(T);
        Vec ys(size_t(T) * H, 0.f);
        std::vector<std::vector<const float *>> yk(T);
        double t_route = 0, t_plan = 0, t_pl[3] = {0, 0, 0}, te[6] = {0, 0, 0, 0, 0, 0}, t_pu[2] = {0, 0}, t_xd = 0;
        std::vector<double> th_busy(2 * size_t(omp_get_max_threads()), 0.0);   // per thread: its gate/up, down GEMV time
        int flat_w = 0;   // long passes: values per gate/up row of the flat buffers
        // passes of at least this many tokens plan over the team into the flat buffers (STRATA_RUNNER_PLAN_TEAM_MIN,
        // default 2: verification passes too; the plan only builds lists, the arithmetic is the same)
        static const int plan_team_min = std::getenv("STRATA_RUNNER_PLAN_TEAM_MIN") ? std::max(1, std::atoi(std::getenv("STRATA_RUNNER_PLAN_TEAM_MIN"))) : 2;
        team_run([&] {
            const int th = omp_get_thread_num(), nth = omp_get_num_threads();
            gemv_share(first, th, nth);
            if (th == 0) t_gemv = now();
            if (rsh) {
                team_barrier();
                if (th == 0) {
                    const double tc = now();
                    comm.allreduce(logits.data(), T * er);
                    timers["comm_router"] += now() - tc;
                }
                team_barrier();
            } else if (token_split(E.router, T, nth).blocks)
                team_barrier();   // the router's rows came from (tile, token block) items of any thread
            {   // this thread's router rows (sharded: experts over the team): scores and keys per token
                const int tiles = E.router.padded / 16, v = (th + E.router.rot) % nth;
                const int r0 = rsh ? int(int64_t(er) * th / nth) : 16 * int(int64_t(tiles) * v / nth);
                const int r1 = rsh ? int(int64_t(er) * (th + 1) / nth) : std::min(int(m.experts), 16 * int(int64_t(tiles) * (v + 1) / nth));
                for (int tk = 0; tk < T; ++tk) {
                    if (fast_exp(4) && r1 > r0) gf4_vsigmoid(rscore.data() + size_t(tk) * er + r0, logits.data() + size_t(tk) * er + r0, r1 - r0);
                    for (int r = r0; r < r1; ++r) {
                        const size_t i = size_t(tk) * er + r;
                        if (!fast_exp(4)) rscore[i] = ref::sigmoid(logits[i]);
                        rkey[i] = rscore[i] + E.bias[r];
                    }
                }
            }
            team_barrier();
            if (T >= plan_team_min) {   // each token's selection on its own thread (the same per-token arithmetic)
                if (th == 0) t_route = now();
#pragma omp for schedule(static) nowait
                for (int tk = 0; tk < T; ++tk) {
                    route_select(rscore.data() + size_t(tk) * er, rkey.data() + size_t(tk) * er, m.experts, m.top_k, m.expert_scale, ids[tk].data(), routing[tk].data());
                    for (int i = 0; i < m.top_k; ++i)
                        if (E.owner_index[ids[tk][i]] >= 0) mine[tk].push_back(i);
                }
                team_barrier();
                if (th == 0) {   // the uses in slice order (as the map below), their tokens in token order
                    for (int tk = 0; tk < T; ++tk) route_ids.insert(route_ids.end(), ids[tk].begin(), ids[tk].end());
                    t_pl[0] = now();
                    // a counting sort into one array (each use's tokens in token order): per-use push_backs into
                    // separately allocated vectors cost ~0.1 ms per call in cache misses on this thread
                    auto &use_of = moe_use_of;
                    const int ns = int(E.slices.size());
                    use_of.assign(ns, -1);
                    auto &cnt = moe_cnt, &flat = moe_flat_tok, &slice_of = moe_slice_of;
                    cnt.assign(ns, 0);
                    for (int tk = 0; tk < T; ++tk)
                        for (int i : mine[tk]) ++cnt[E.owner_index[ids[tk][i]]];
                    slice_of.clear();
                    for (int si = 0; si < ns; ++si)   // numbered in slice order
                        if (cnt[si]) use_of[si] = n_uses++, slice_of.push_back(si);
                    if (pool.size() < size_t(n_uses)) pool.resize(n_uses);
                    moe_row_off.assign(size_t(n_uses) + 1, 0);
                    for (int ui = 0; ui < n_uses; ++ui) moe_row_off[ui + 1] = moe_row_off[ui] + cnt[slice_of[ui]];
                    const size_t rows_all = size_t(moe_row_off[n_uses]);
                    flat.resize(rows_all);
                    moe_fill.assign(moe_row_off.begin(), moe_row_off.end() - 1);
                    for (int tk = 0; tk < T; ++tk)
                        for (int i : mine[tk]) flat[moe_fill[use_of[E.owner_index[ids[tk][i]]]]++] = tk;
                    t_pu[0] = now();
                    for (int ui = 0; ui < n_uses; ++ui) flat_w = std::max(flat_w, E.slices[slice_of[ui]].gate.rows);
                    moe_fg.get(rows_all * flat_w), moe_fu.get(rows_all * flat_w), moe_fy.get(rows_all * H);
                    if (moe_fact.size() < rows_all) moe_fact.resize(rows_all);
                    t_pu[1] = now();
                    up.resize(2 * size_t(n_uses)), down.resize(size_t(n_uses) + (sr ? 1 : 0));
                    moe_block_off.assign(size_t(n_uses) + 2, 0);
                    for (int ui = 0; ui < n_uses; ++ui)
                        moe_block_off[ui + 1] = moe_block_off[ui] + (moe_row_off[ui + 1] - moe_row_off[ui]) * (E.slices[moe_slice_of[ui]].gate.rows / 64);
                    moe_block_off[n_uses + 1] = moe_block_off[n_uses] + (sr ? T * (sr / 64) : 0);
                    blocks.resize(moe_block_off[n_uses + 1]);
                    if (T > 4 && w4a8_xt_on()) {   // the transposed activations' blocks (gate / up: H values, down: flat_w)
                        moe_xblk.assign(size_t(n_uses) + 1, 0);
                        for (int ui = 0; ui < n_uses; ++ui) {
                            const int nt = moe_row_off[ui + 1] - moe_row_off[ui];
                            moe_xblk[ui + 1] = moe_xblk[ui] + (nt >= w4a8_gemm_min() ? gf4_w4a8_gemm_blocks(nt) : 0);
                        }
                        const size_t nb = moe_xblk[n_uses];
                        moe_xt.get(nb * 16 * H), moe_xs.get(nb * H), moe_xtd.get(nb * 16 * flat_w), moe_xsd.get(nb * flat_w);
                    }
                    t_pl[1] = now();
                }
                team_barrier();
#pragma omp for schedule(dynamic, 4) nowait
                for (int ui = 0; ui < n_uses; ++ui) {   // each use's buffers, jobs and SwiGLU blocks (the code below, per use)
                    MoeUse &us = pool[ui];
                    us.slice = moe_slice_of[ui];
                    us.tokens.assign(moe_flat_tok.begin() + moe_row_off[ui], moe_flat_tok.begin() + moe_row_off[ui + 1]);
                    const auto &sl = E.slices[us.slice];
                    const int nt = int(us.tokens.size());
                    // rows [moe_row_off[ui], + nt) of the flat buffers (flat_w values per gate/up row: the widest slice)
                    const size_t r0 = size_t(moe_row_off[ui]);
                    us.gp = moe_fg.p.get() + r0 * flat_w, us.upp = moe_fu.p.get() + r0 * flat_w, us.yp = moe_fy.p.get() + r0 * H;
                    us.ap = moe_fact.data() + r0;
                    for (int k = 0; k < nt; ++k) {
                        Activation &a = us.ap[k];
                        if (a.x.capacity() < size_t(flat_w)) a.x.reserve(flat_w), a.q.reserve(flat_w), a.qs.reserve(flat_w / 16);
                        a.x.resize(sl.gate.rows), a.group = g_act_group;
                        if (needs_q8(sl.down)) a.q.resize(sl.gate.rows), a.qs.resize(sl.gate.rows / g_act_group);
                        else a.q.clear();
                    }
                    Job jg{&sl.gate, nullptr, us.gp, nt, sl.gate.rows}, ju{&sl.up, nullptr, us.upp, nt, sl.up.rows};
                    for (int k = 0; k < nt; ++k) jg.as[k] = ju.as[k] = &ax[us.tokens[k]];
                    down[ui] = {&sl.down, us.ap, us.yp, nt, H};
                    if (T > 4 && w4a8_xt_on() && moe_xblk[ui + 1] > moe_xblk[ui]) {   // (the uses of 8+ tokens)
                        const size_t bk = moe_xblk[ui];
                        if (sl.gate.kind == Kind::NvQ && sl.up.kind == Kind::NvQ && sl.gate.cols == H && sl.up.cols == H && ax[0].group == g_act_group) {
                            int8_t *xt = moe_xt.get(0) + bk * 16 * H;
                            float *xs = moe_xs.get(0) + bk * H;
                            w4a8_xt_prep(jg.as, nt, H, xt, xs);
                            jg.xt = ju.xt = xt, jg.xs = ju.xs = xs;
                        }
                        if (sl.down.kind == Kind::NvQ && sl.down.cols <= flat_w)   // filled after the SwiGLU blocks
                            down[ui].xt = moe_xtd.get(0) + bk * 16 * flat_w, down[ui].xs = moe_xsd.get(0) + bk * flat_w;
                    }
                    up[2 * ui] = std::move(jg), up[2 * ui + 1] = std::move(ju);
                    int b = moe_block_off[ui];
                    for (int k = 0; k < nt; ++k)
                        for (int j0 = 0; j0 < sl.gate.rows; j0 += 64) blocks[b++] = {ui, k, j0};
                }
                team_barrier();   // the output buffers are grown: the row pointers below stay valid
#pragma omp for schedule(static) nowait
                for (int tk = 0; tk < T; ++tk) {   // the shared expert's activation and blocks, the token's routed rows
                    if (sr) {
                        Activation &a = sact[tk];
                        a.x.resize(sr), a.group = 64;
                        if (q8_act(E.sdown, T)) a.q.resize(sr), a.qs.resize(sr / 64);
                        int b = moe_block_off[n_uses] + tk * (sr / 64);
                        for (int j0 = 0; j0 < sr; j0 += 64) blocks[b++] = {-1, tk, j0};
                    }
                    for (int i : mine[tk]) {
                        const auto &u2 = pool[moe_use_of[E.owner_index[ids[tk][i]]]];
                        const int row = int(std::lower_bound(u2.tokens.begin(), u2.tokens.end(), tk) - u2.tokens.begin());
                        yk[tk].push_back(u2.yp + size_t(row) * H);
                    }
                }
                team_barrier();
                if (th == 0) {
                    if (sr) down[n_uses] = {&E.sdown, sact.data(), ys.data(), T, H};
                    t_pl[2] = t_plan = now();
                }
            }
            if (T < plan_team_min && omp_get_thread_num() == 0)
            {
                t_route = now();
                for (int tk = 0; tk < T; ++tk) {
                    route_select(rscore.data() + size_t(tk) * er, rkey.data() + size_t(tk) * er, m.experts, m.top_k, m.expert_scale, ids[tk].data(), routing[tk].data());
                    for (int i = 0; i < m.top_k; ++i)
                        if (E.owner_index[ids[tk][i]] >= 0) mine[tk].push_back(i);
                    route_ids.insert(route_ids.end(), ids[tk].begin(), ids[tk].end());
                }
                t_pl[0] = now();
                std::map<int, std::vector<int>> users;
                for (int tk = 0; tk < T; ++tk)
                    for (int i : mine[tk]) users[E.owner_index[ids[tk][i]]].push_back(tk);
                if (pool.size() < users.size()) pool.resize(users.size());
                for (auto &[si, tks] : users) {
                    MoeUse &us = pool[n_uses++];
                    us.slice = si, us.tokens = tks;
                    const auto &sl = E.slices[si];
                    auto grow = [](auto &v, size_t want) { if (v.size() < want) v.resize(want); };
                    grow(us.g, tks.size() * sl.gate.rows), grow(us.u, tks.size() * sl.up.rows), grow(us.y, tks.size() * size_t(H));
                    if (us.act.size() < tks.size()) us.act.resize(tks.size());
                    us.gp = us.g.data(), us.upp = us.u.data(), us.yp = us.y.data(), us.ap = us.act.data();
                    for (size_t k = 0; k < tks.size(); ++k) {
                        Activation &a = us.act[k];
                        a.x.resize(sl.gate.rows), a.group = g_act_group;
                        if (needs_q8(sl.down)) a.q.resize(sl.gate.rows), a.qs.resize(sl.gate.rows / g_act_group);
                        else a.q.clear();
                    }
                }
                t_pl[1] = now();
                for (int ui = 0; ui < n_uses; ++ui) {
                    auto &us = pool[ui];
                    const auto &sl = E.slices[us.slice];
                    const int nt = int(us.tokens.size());
                    Job jg{&sl.gate, nullptr, us.g.data(), nt, sl.gate.rows}, ju{&sl.up, nullptr, us.u.data(), nt, sl.up.rows};
                    for (int k = 0; k < nt; ++k) jg.as[k] = ju.as[k] = &ax[us.tokens[k]];
                    up.push_back(std::move(jg)), up.push_back(std::move(ju));
                    down.push_back({&sl.down, us.act.data(), us.y.data(), nt, H});
                    for (int k = 0; k < nt; ++k)
                        for (int j0 = 0; j0 < sl.gate.rows; j0 += 64) blocks.push_back({ui, k, j0});
                }
                if (sr) {
                    for (int tk = 0; tk < T; ++tk) {
                        Activation &a = sact[tk];
                        a.x.resize(sr), a.group = 64;
                        if (needs_q8(E.sdown)) a.q.resize(sr), a.qs.resize(sr / 64);
                        for (int j0 = 0; j0 < sr; j0 += 64) blocks.push_back({-1, tk, j0});
                    }
                    down.push_back({&E.sdown, sact.data(), ys.data(), T, H});
                }
                t_pl[2] = now();
                for (int tk = 0; tk < T; ++tk)
                    for (int i : mine[tk]) {
                        const int si = E.owner_index[ids[tk][i]];
                        for (int ui = 0; ui < n_uses; ++ui)
                            if (const auto &u2 = pool[ui]; u2.slice == si) {
                                const int row = int(std::find(u2.tokens.begin(), u2.tokens.end(), tk) - u2.tokens.begin());
                                yk[tk].push_back(u2.y.data() + size_t(row) * H);
                            }
                    }
                t_plan = now();
            }
            team_barrier();
            if (th == 0) te[0] = now();
            const double tu0 = now();
            prefetch_share(down, th, nth);
            if (T > 4) gemv_share_cmg(up, th, nth); else gemv_share(up, th, nth);   // (decode passes: the placement split)
            th_busy[2 * th] = now() - tu0;
            if (th == 0) te[1] = now();
            team_barrier();
            if (th == 0) te[2] = now();
#pragma omp for schedule(static) nowait
            for (size_t bi = 0; bi < blocks.size(); ++bi) {
                const auto [ui, k, j0] = blocks[bi];
                if (ui >= 0) {
                    MoeUse &us = pool[ui];
                    const int rows = E.slices[us.slice].gate.rows;
                    Activation &a = us.ap[k];
                    if (fast_exp(2)) gf4_vswiglu(a.x.data() + j0, us.gp + size_t(k) * rows + j0, us.upp + size_t(k) * rows + j0, 64, desc.swiglu_limit);
                    else
                        for (int j = j0; j < j0 + 64; ++j) a.x[j] = ref::swiglu(us.gp[size_t(k) * rows + j], us.upp[size_t(k) * rows + j], desc.swiglu_limit);
                    if (!a.q.empty()) gf4_quantize_q8g(a.x.data() + j0, 64, a.group, a.q.data() + j0, a.qs.data() + j0 / a.group);
                } else {
                    Activation &a = sact[k];
                    if (fast_exp(2)) gf4_vswiglu(a.x.data() + j0, sg.data() + size_t(k) * sr + j0, su.data() + size_t(k) * sr + j0, 64, desc.shared_swiglu_limit);
                    else
                        for (int j = j0; j < j0 + 64; ++j) a.x[j] = ref::swiglu(sg[size_t(k) * sr + j], su[size_t(k) * sr + j], desc.shared_swiglu_limit);
                    if (!a.q.empty()) gf4_quantize_q8g(a.x.data() + j0, 64, 64, a.q.data() + j0, a.qs.data() + j0 / 64);
                }
            }
            team_barrier();
            if (th == 0) te[3] = now();
            if (T > 4 && w4a8_xt_on() && T >= plan_team_min) {   // each use's SwiGLU rows transposed once for down
#pragma omp for schedule(dynamic, 4) nowait
                for (int ui = 0; ui < n_uses; ++ui)
                    if (down[ui].xt) {
                        const MoeUse &us = pool[ui];
                        const int nt = down[ui].nt, cols = down[ui].m->cols;
                        thread_local std::vector<gf4_q8act> q;
                        q.resize(nt);
                        for (int k = 0; k < nt; ++k)
                            q[k] = gf4_q8act{cols, us.ap[k].q.data(), us.ap[k].qs.data(), us.ap[k].group};
                        gf4_w4a8_gemm_prep(q.data(), nt, cols, const_cast<int8_t *>(down[ui].xt), const_cast<float *>(down[ui].xs));
                    }
                team_barrier();
                if (th == 0) t_xd = now() - te[3], te[3] = now();
            }
            const double td0 = now();
            if (T > 4) gemv_share_cmg(down, th, nth); else gemv_share(down, th, nth);
            th_busy[2 * th + 1] = now() - td0;
            if (th == 0) te[4] = now();
            team_barrier();
            if (th == 0) te[5] = now();
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE_BITS == 512
            if (H % 1024 == 0) {   // (token, 1024 values) items, 16 lanes: the same zero start, one fused multiply-add
                                   // per routed expert in order, then + ys (the scalar loop's contracted arithmetic)
                const int nq = H / 1024;
                const svbool_t pw = svptrue_b32();
#pragma omp for schedule(static) nowait
                for (int it = 0; it < T * nq; ++it) {
                    const int tk = it / nq, d0 = (it % nq) * 1024, nk = int(yk[tk].size());
                    float rk[16];
                    for (int k = 0; k < nk && k < 16; ++k) rk[k] = routing[tk][mine[tk][k]];
                    if (nk > 16) std::abort();
                    const float *ysr = ys.data() + size_t(tk) * H;
                    float *out = sum[tk].data();
                    for (int d = d0; d < d0 + 1024; d += 16) {
                        svfloat32_t v = svdup_n_f32(0);
                        for (int k = 0; k < nk; ++k) v = svmla_f32_x(pw, v, svdup_n_f32(rk[k]), svld1_f32(pw, yk[tk][k] + d));
                        svst1_f32(pw, out + d, svadd_f32_x(pw, v, svld1_f32(pw, ysr + d)));
                    }
                }
            } else
#endif
#pragma omp for schedule(static) nowait
            for (int td = 0; td < T * H; ++td) {
                const int tk = td / H, d = td % H;
                float v = 0;
                for (size_t k = 0; k < yk[tk].size(); ++k) v += routing[tk][mine[tk][k]] * yk[tk][k][d];
                sum[tk][d] = v + ys[size_t(tk) * H + d];
            }
        });
        timers["router"] += t_route - t_in;
        timers["router.acts"] += t_acts - t_in, timers["router.gemv"] += t_gemv - t_acts, timers["router.rest"] += t_route - t_gemv;
        const double t_end = now();
        timers["experts"] += t_end - t_route;
        // thread 0: gate/up GEMVs (its own share, then the barrier wait), SwiGLU, down (own, wait), weighted sum
        timers["experts.up"] += te[1] - te[0], timers["experts.up_wait"] += te[2] - te[1], timers["experts.act"] += te[3] - te[2];
        timers["experts.down"] += te[4] - te[3], timers["experts.down_wait"] += te[5] - te[4], timers["experts.sum"] += t_end - te[5];
        timers["experts.xt_down"] += t_xd;
        {   // the slowest thread and the mean (balance of the expert GEMVs)
            const int nth = omp_get_max_threads();
            double mu = 0, md = 0, xu = 0, xd = 0;
            for (int i = 0; i < nth; ++i) mu += th_busy[2 * i], md += th_busy[2 * i + 1], xu = std::max(xu, th_busy[2 * i]), xd = std::max(xd, th_busy[2 * i + 1]);
            timers["experts.up_max"] += xu, timers["experts.up_mean"] += mu / nth, timers["experts.down_max"] += xd, timers["experts.down_mean"] += md / nth;
        }
        timers["experts.plan"] += t_plan - t_route;   // thread 0's routing and job list (the others wait)
        timers["experts.plan_route"] += t_pl[0] - t_route, timers["experts.plan_uses"] += t_pl[1] - t_pl[0];
        timers["experts.plan_jobs"] += t_pl[2] - t_pl[1], timers["experts.plan_rows"] += t_plan - t_pl[2];
        if (t_pu[0] > 0) timers["experts.pu_lists"] += t_pu[0] - t_pl[0], timers["experts.pu_bufs"] += t_pu[1] - t_pu[0], timers["experts.pu_jobs"] += t_pl[1] - t_pu[1];
        return sum;
    }
    std::vector<Vec> moe_multi(MoeLayer &E, const std::vector<Vec> &xs, const strata::core::LayerDescriptor &desc) {
        if (moe_fusable(E)) return moe_multi_fused(E, xs, desc);
        const double t_in = now();
        const int T = int(xs.size()), H = m.hidden;
        const bool quant = moe_quant(E);
        auto ax = acts_of(xs, H, quant, g_act_group), af = acts_of(xs, H, false);
        const int er = m.experts, sr = E.sgate.rows;
        Vec logits(size_t(T) * er + 16, 0.f), sg(size_t(T) * std::max(sr, 1)), su(size_t(T) * std::max(sr, 1));
        std::vector<Job> first{{&E.router, af.data(), logits.data() + E.router_r0, T, er}};
        if (sr) first.push_back({&E.sgate, ax.data(), sg.data(), T, sr}), first.push_back({&E.sup, ax.data(), su.data(), T, sr});
        gemv_multi(first);
        if (E.router.rows != m.experts) comm.allreduce(logits.data(), T * er);
        // routing per token (moe_fused's arithmetic: sigmoid score, score + bias key, the same selection)
        std::vector<std::vector<int>> ids(T, std::vector<int>(m.top_k)), mine(T);
        std::vector<Vec> routing(T, Vec(m.top_k));
        for (int tk = 0; tk < T; ++tk) {
            Vec sc(m.experts), key(m.experts);
            for (int r = 0; r < m.experts; ++r) sc[r] = ref::sigmoid(logits[size_t(tk) * er + r]), key[r] = sc[r] + E.bias[r];
            route_select(sc.data(), key.data(), m.experts, m.top_k, m.expert_scale, ids[tk].data(), routing[tk].data());
            route_ids.insert(route_ids.end(), ids[tk].begin(), ids[tk].end());
            for (int i = 0; i < m.top_k; ++i)
                if (E.owner_index[ids[tk][i]] >= 0) mine[tk].push_back(i);
        }
        timers["router"] += now() - t_in;
        const double t_ex = now();
        // the slices this pass needs, each with the tokens routed to it in token order
        std::map<int, std::vector<int>> users;   // slice index -> tokens
        for (int tk = 0; tk < T; ++tk)
            for (int i : mine[tk]) users[E.owner_index[ids[tk][i]]].push_back(tk);
        struct Use { int slice; std::vector<int> tokens; std::vector<Activation> act; Vec g, u, y; };
        std::vector<Use> uses;
        for (auto &[si, tks] : users) {
            Use us;
            us.slice = si, us.tokens = tks;
            const auto &sl = E.slices[si];
            us.g.resize(tks.size() * sl.gate.rows), us.u.resize(tks.size() * sl.up.rows), us.y.resize(tks.size() * size_t(H));
            us.act.resize(tks.size());
            for (auto &a : us.act) {
                a.x.resize(sl.gate.rows), a.group = g_act_group;
                if (needs_q8(sl.down)) a.q.resize(sl.gate.rows), a.qs.resize(sl.gate.rows / g_act_group);
            }
            uses.push_back(std::move(us));
        }
        std::vector<Job> up;
        for (auto &us : uses) {   // each slice once for the tokens routed to it (their activations by pointer)
            const auto &sl = E.slices[us.slice];
            const int nt = int(us.tokens.size());
            Job jg{&sl.gate, nullptr, us.g.data(), nt, sl.gate.rows}, ju{&sl.up, nullptr, us.u.data(), nt, sl.up.rows};
            for (int k = 0; k < nt; ++k) jg.as[k] = ju.as[k] = &ax[us.tokens[k]];
            up.push_back(jg), up.push_back(ju);
        }
        gemv_multi(up);
        // SwiGLU per 64-value block (and its int8 block for W4A8) over the team, as moe_fused: (use, token, block)
        std::vector<std::array<int, 3>> blocks;
        for (int ui = 0; ui < int(uses.size()); ++ui)
            for (int k = 0; k < int(uses[ui].tokens.size()); ++k)
                for (int j0 = 0; j0 < E.slices[uses[ui].slice].gate.rows; j0 += 64) blocks.push_back({ui, k, j0});
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (size_t bi = 0; bi < blocks.size(); ++bi) {
                const auto [ui, k, j0] = blocks[bi];
                Use &us = uses[ui];
                const int rows = E.slices[us.slice].gate.rows;
                Activation &a = us.act[k];
                for (int j = j0; j < j0 + 64; ++j) a.x[j] = ref::swiglu(us.g[size_t(k) * rows + j], us.u[size_t(k) * rows + j], desc.swiglu_limit);
                if (!a.q.empty()) gf4_quantize_q8g(a.x.data() + j0, 64, a.group, a.q.data() + j0, a.qs.data() + j0 / a.group);
            }
        });
        std::vector<Job> down;
        for (auto &us : uses) down.push_back({&E.slices[us.slice].down, us.act.data(), us.y.data(), int(us.tokens.size()), H});
        std::vector<Activation> sact(T);
        Vec ys(size_t(T) * H, 0.f);
        if (sr) {
            for (int tk = 0; tk < T; ++tk) {
                Activation &a = sact[tk];
                a.x.resize(sr), a.group = 64;
                if (needs_q8(E.sdown)) a.q.resize(sr), a.qs.resize(sr / 64);
                for (int j = 0; j < sr; ++j) a.x[j] = ref::swiglu(sg[size_t(tk) * sr + j], su[size_t(tk) * sr + j], desc.shared_swiglu_limit);
                if (!a.q.empty())
                    for (int j0 = 0; j0 < sr; j0 += 64) gf4_quantize_q8g(a.x.data() + j0, 64, 64, a.q.data() + j0, a.qs.data() + j0 / 64);
            }
            down.push_back({&E.sdown, sact.data(), ys.data(), T, H});
        }
        gemv_multi(down);
        std::vector<Vec> sum(T, Vec(H));
        std::vector<std::vector<const float *>> yk(T);   // token tk's routed outputs in its routing order
        for (int tk = 0; tk < T; ++tk)
            for (int i : mine[tk]) {
                const int si = E.owner_index[ids[tk][i]];
                const Use *us = nullptr;
                for (const auto &u2 : uses) if (u2.slice == si) us = &u2;
                const int row = int(std::find(us->tokens.begin(), us->tokens.end(), tk) - us->tokens.begin());
                yk[tk].push_back(us->y.data() + size_t(row) * H);
            }
        team_run([&] {
#pragma omp for schedule(static) nowait
            for (int td = 0; td < T * H; ++td) {   // weighted sum in the routing order, then the shared expert
                const int tk = td / H, d = td % H;
                float v = 0;
                for (size_t k = 0; k < yk[tk].size(); ++k) v += routing[tk][mine[tk][k]] * yk[tk][k][d];
                sum[tk][d] = v + ys[size_t(tk) * H + d];
            }
        });
        timers["experts"] += now() - t_ex;
        return sum;
    }
    std::vector<Vec> ffn_multi(int l, const std::vector<Vec> &xs) {
        auto &Y = L[l];
        return Y.dense ? dense_multi(Y.ffn, xs, m.layers[l].swiglu_limit) : moe_multi(Y.moe, xs, m.layers[l]);
    }
    Vec ar_buf;   // allreduce_rows' staging buffer (persistent: no allocation per call)
    void allreduce_rows(std::vector<Vec> &ys) {   // one all-reduce of all tokens' rows (each element summed in rank order)
        const int T = int(ys.size()), H = m.hidden;
        if (ar_buf.size() < size_t(T) * H) ar_buf.resize(size_t(T) * H);
        Vec &buf = ar_buf;
        const double t0 = now();
        if (T > 4)   // a long pass: the row copies over the team (2 MB each way at 128 tokens)
            team_run([&] {
#pragma omp for schedule(static) nowait
                for (int t = 0; t < T; ++t) std::copy(ys[t].begin(), ys[t].end(), buf.begin() + size_t(t) * H);
            });
        else
            for (int t = 0; t < T; ++t) std::copy(ys[t].begin(), ys[t].end(), buf.begin() + size_t(t) * H);
        const double t1 = now();
        comm.allreduce(buf.data(), T * H);
        const double t2 = now();
        if (T > 4)
            team_run([&] {
#pragma omp for schedule(static) nowait
                for (int t = 0; t < T; ++t) std::copy(buf.begin() + size_t(t) * H, buf.begin() + size_t(t + 1) * H, ys[t].begin());
            });
        else
            for (int t = 0; t < T; ++t) std::copy(buf.begin() + size_t(t) * H, buf.begin() + size_t(t + 1) * H, ys[t].begin());
        timers["comm.copies"] += (t1 - t0) + (now() - t2), timers["comm.wire"] += t2 - t1;
    }
    // T cells of the MTP block from cell `cell`: tokens[t] with prev[t] (the hidden it pairs with); returns the
    // post-shared_head_norm hiddens, and rank 0's draft logits per token in *logits when asked
    std::vector<Vec> mtp_forward(const std::vector<int> &tokens, const std::vector<Vec> &prev, int cell, std::vector<Vec> *logits) {
        const double t0 = now();
        const int T = int(tokens.size()), H = m.hidden;
        auto &Y = mtp.layer;
        std::vector<Vec> joined(T, Vec(2 * size_t(H)));
        for (int t = 0; t < T; ++t) {
            Vec emb(H);
            const uint16_t *ed = embedding.data() + size_t(tokens[t]) * H;
            for (int i = 0; i < H; ++i) {
                const uint32_t v = uint32_t(ed[i]) << 16;
                std::memcpy(&emb[i], &v, 4);
            }
            const Vec en = norm(emb, mtp.enorm), hn = norm(prev[t], mtp.hnorm);   // embedding half first
            std::copy(en.begin(), en.end(), joined[t].begin());
            std::copy(hn.begin(), hn.end(), joined[t].begin() + H);
        }
        auto aj = acts_of(joined, 2 * H, needs_q8(mtp.eh));
        const int rows = mtp.eh.rows;
        Vec part(size_t(T) * std::max(rows, 1));
        if (rows) gemv_multi({{&mtp.eh, aj.data(), part.data(), T, rows}});
        std::vector<Vec> cur(T, Vec(H, 0.f));
        for (int t = 0; t < T; ++t) std::copy(part.begin() + size_t(t) * rows, part.begin() + size_t(t + 1) * rows, cur[t].begin() + mtp.eh_rows.begin);
        allreduce_rows(cur);   // the other ranks' rows are zero here: the sum is the row vector itself
        const double t_front = now();
        // STRATA_RUNNER_MTP_DETAIL=1: the shared timers' growth inside the draft block's attention and MoE, as mtpa.* / mtpm.*
        static const bool mtp_detail = std::getenv("STRATA_RUNNER_MTP_DETAIL") && std::string(std::getenv("STRATA_RUNNER_MTP_DETAIL")) == "1";
        std::map<std::string, double> snap;
        auto diff_into = [&](const char *pre) {
            for (const auto &[k, v] : timers)
                if (k.compare(0, 3, "mtp") && k.compare(0, 3, "mhs")) {
                    const auto it = snap.find(k);
                    const double d = v - (it == snap.end() ? 0 : it->second);
                    if (d > 0) timers[pre + k] += d;   // new keys only start with the prefix: no rehash of k's iterator
                }
        };
        if (mtp_detail) snap = timers;
        std::vector<Vec> x(T);
        for (int t = 0; t < T; ++t) x[t] = norm(cur[t], Y.attn_norm);
        auto y = mla_multi(Y.ml, x, cell);
        allreduce_rows(y);
        const double t_attn = now();
        if (mtp_detail) diff_into("mtpa."), snap = timers;
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < H; ++d) cur[t][d] += y[t][d];
            x[t] = norm(cur[t], Y.ffn_norm);
        }
        y = moe_multi(Y.moe, x, m.draft_layers.at(0));
        allreduce_rows(y);
        const double t_moe = now();
        if (mtp_detail) diff_into("mtpm.");
        timers["mtp.front"] += t_front - t0, timers["mtp.attn"] += t_attn - t_front, timers["mtp.moe"] += t_moe - t_attn;
        timers["mtp.calls"] += 1e-3, timers["mtp.tokens"] += 1e-3 * T;
        std::vector<Vec> h(T);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < H; ++d) cur[t][d] += y[t][d];
            h[t] = norm(cur[t], mtp.head_norm);
        }
        if (mtp_cells.size() == pass_groups.size() && pass_groups.size() > 1)   // mtp_forward_ctx: each context's next cell
            for (size_t gi = 0; gi < pass_groups.size(); ++gi) {
                const auto [c, len] = pass_groups[gi];
                (c == active_ctx ? mtp.position : ctx_store[c].mtp_position) = mtp_cells[gi] + len;
            }
        else mtp.position = cell + T;
        if (logits) {
            const Matrix &dh = draft_head.rows ? draft_head : head;
            auto ah = acts_of(h, H, needs_q8(dh));
            Vec hp(size_t(T) * dh.rows);
            gemv_multi({{&dh, ah.data(), hp.data(), T, dh.rows}});
            logits->clear();
            if (!full_logits) {   // this rank's rows only: argmax_last() picks the drafts
                last_part = split_rows(hp, T, head.rows, head.rows);
                timers["mtp.head"] += now() - t_moe;
                timers["mtp"] += now() - t0;
                return h;
            }
            for (int k = 0; k < T; ++k) {
                Vec lg(comm.rank == 0 ? m.vocab : vocab.end, 0.f);
                std::copy(hp.begin() + size_t(k) * head.rows, hp.begin() + size_t(k + 1) * head.rows, lg.begin() + vocab.begin);
                comm.gather(lg.data(), vocab.begin, vocab.end, m.vocab);
                if (comm.rank == 0) logits->push_back(std::move(lg));
            }
        }
        timers["mtp"] += now() - t0;
        return h;
    }

    bool head_last = false;   // step_multi without full logits: the head for the pass's last token only (prefill)
    // the pass: rank 0 gets each token's logits (nothing when the model is truncated); dumps[t] as step()'s dump
    // A prefill pass in two halves whose all-reduces overlap the other half's compute (default; STRATA_RUNNER_PREFILL_PIPE=0: off):
    // per layer the attention of half A then B (A's tokens come first: the KDA states and MLA caches see the same
    // order), each half's output all-reduced by the comm thread while the next half computes, then the FFN site and
    // FFN of A then B the same way. The arithmetic per token is step_multi's, so the bits are the same.
    std::array<Vec, 8> pipe_buf;
    std::array<std::vector<Vec>, 8> pipe_y;
    std::vector<Vec> step_multi_pipe(const std::vector<int> &tokens, const std::vector<std::filesystem::path> &dumps = {}) {
        // K parts (STRATA_RUNNER_PIPE_PARTS, 2..4, default 2): each part's all-reduces overlap the other parts' work
        static const int KP = std::getenv("STRATA_RUNNER_PIPE_PARTS") ? std::clamp(std::atoi(std::getenv("STRATA_RUNNER_PIPE_PARTS")), 2, 4) : 2;
        const int T = int(tokens.size()), H = m.hidden, base = position, K = std::min(KP, std::max(2, T / 16));
        int off[4], len[4];
        for (int h = 0; h < K; ++h) off[h] = int(int64_t(T) * h / K), len[h] = int(int64_t(T) * (h + 1) / K) - off[h];
        const double busy0 = comm.ar_thread_busy;
        std::vector<Vec> rh[4];
        for (int h = 0; h < K; ++h) {
            rh[h].resize(len[h]);
            team_run([&] {   // first touched by each token's owner in site_tokens
                const int th = omp_get_thread_num(), nth = omp_get_num_threads();
                for (int t = th; t < len[h]; t += nth) rh[h][t].assign(4 * H, 0.f);
            });
            for (int t = 0; t < len[h]; ++t) {
                const int tok = tokens[off[h] + t];
                if (tok < 0 || tok >= m.vocab) throw std::runtime_error("runner: token id outside the vocabulary");
                const uint16_t *ed = embedding.data() + size_t(tok) * H;
                for (int i = 0; i < H; ++i) {
                    const uint32_t v = uint32_t(ed[i]) << 16;
                    float f;
                    std::memcpy(&f, &v, 4);
                    for (int k = 0; k < 4; ++k) rh[h][t][k * H + i] = f;
                }
            }
        }
        // staged all-reduces (STRATA_RUNNER_AR_STAGED=0: off): the team writes the rows straight into the bank's
        // send slots (shard by shard) and reads the sums from its result slots; the comm thread only sums and Puts
        // (its copies in and out were most of its 1.45 ms per prompt token, memory-bound on one core)
        static const bool staged_on = !(std::getenv("STRATA_RUNNER_AR_STAGED") && std::string(std::getenv("STRATA_RUNNER_AR_STAGED")) == "0");
        std::vector<int> soff[8];
        bool staged[8] = {false, false, false, false, false, false, false, false};
        // f(t, d, i0, i1): row t's elements [i0, i1) belong to shard d (element t * H + i)
        auto rows_by_shard = [&](const std::vector<int> &off, int t, auto &&f) {
            const int e0 = t * H, e1 = e0 + H;
            int d = int(std::upper_bound(off.begin(), off.end(), e0) - off.begin()) - 1;
            for (int e = e0; e < e1; ++d) {
                const int stop = std::min(e1, off[d + 1]);
                f(d, e - e0, stop - e0, e - off[d]);
                e = stop;
            }
        };
        auto stage = [&](std::vector<Vec> &ys, Vec &buf, int slot) {   // rows into the buffer, the all-reduce posted
            const int n = int(ys.size());
            staged[slot] = staged_on && comm.staged_ok(n * H) && slot < comm.nbanks;
            if (staged[slot]) {
                comm.staged_shards(n * H, soff[slot]);
                const auto &off = soff[slot];
                team_run([&] {
#pragma omp for schedule(static) nowait
                    for (int t = 0; t < n; ++t)
                        rows_by_shard(off, t, [&](int d, int i0, int i1, int o) {
                            std::copy(ys[t].begin() + i0, ys[t].begin() + i1, comm.staged_send(slot, d) + o);
                        });
                });
                return comm.allreduce_staged(slot, n * H);
            }
            if (buf.size() < size_t(n) * H) buf.resize(size_t(n) * H);
            team_run([&] {
#pragma omp for schedule(static) nowait
                for (int t = 0; t < n; ++t) std::copy(ys[t].begin(), ys[t].end(), buf.begin() + size_t(t) * H);
            });
            return comm.allreduce_async(buf.data(), n * H);
        };
        // the summed rows go to persistent per-token vectors first touched by the token's owner in site_tokens
        // (thread t % team): the sublayer outputs were allocated on the main thread's CMG, so the other CMGs' threads
        // read them remotely in every site
        auto unstage = [&](long tk, int n, Vec &buf, int slot) -> std::vector<Vec> & {
            const double tw = now();
            comm.ar_wait(tk);
            const double dw = now() - tw;
            timers["comm.wait"] += dw;
            // by output (attention / FFN, part) and the layer kind whose output it is
            const bool ffn = slot >= K;
            const int lk = ffn ? cur_layer - 1 : cur_layer;   // an FFN output is waited for at the next layer
            timers[std::string("comm.w_") + (lk >= 0 && lk < int(L.size()) && !L[lk].kda ? "mla_" : "kda_") + (ffn ? "ffn" : "att") + char('A' + slot % K)] += dw;
            auto &own = pipe_y[slot];
            if (own.size() != size_t(n)) own.resize(n);
            team_run([&] {
                const int th = omp_get_thread_num(), nth = omp_get_num_threads();
                for (int t = th; t < n; t += nth) {
                    if (own[t].size() != size_t(H)) own[t].assign(H, 0.f);
                    if (staged[slot])
                        rows_by_shard(soff[slot], t, [&](int d, int i0, int i1, int o) {
                            const float *src = comm.staged_result(slot, d) + o;
                            std::copy(src, src + (i1 - i0), own[t].begin() + i0);
                        });
                    else std::copy(buf.begin() + size_t(t) * H, buf.begin() + size_t(t + 1) * H, own[t].begin());
                }
            });
            return own;
        };
        auto save = [&](const char *kind, int l, int h) {   // step_multi's residual dumps
            if (dumps.empty() || comm.rank) return;
            for (int t = 0; t < len[h]; ++t) {
                if (dumps[off[h] + t].empty()) continue;
                std::ofstream o(dumps[off[h] + t] / (std::string(kind) + "-" + std::to_string(l) + ".bin"), std::ios::binary);
                o.write(reinterpret_cast<const char *>(rh[h][t].data()), rh[h][t].size() * 4);
            }
        };
        std::vector<Vec> c_ffn[4], y_ffn[4];
        long tk_ffn[4] = {0, 0, 0, 0};
        for (int l = 0; l < layers_run; ++l) {
            auto &Y = L[l];
            cur_layer = l;
            std::vector<Vec> c[4], ya[4];
            long tk_att[4] = {0, 0, 0, 0};
            MlaFront fr[4];
            for (int h = 0; h < K; ++h) {
                std::vector<Vec> *yf = l ? &unstage(tk_ffn[h], len[h], pipe_buf[K + h], K + h) : nullptr;
                auto st = site_multi(rh[h], l ? &c_ffn[h] : nullptr, yf, Y.hc_attn, Y.hc_attn_base, Y.hc_attn_scale, Y.attn_norm);
                if (l) save("layer", l - 1, h);
                std::vector<Vec> x(len[h]);
                c[h].resize(len[h]);
                for (int t = 0; t < len[h]; ++t) x[t] = std::move(st[t].first), c[h][t] = std::move(st[t].second);
                position = base + off[h];
                if (Y.kda) {
                    ya[h] = kda_multi(Y.kd, x);
                    tk_att[h] = stage(ya[h], pipe_buf[h], h);
                } else
                    mla_front1(Y.ml, x, fr[h], true);   // the fronts' all-reduces each overlap the other half's work
            }
            if (!Y.kda) {
                for (int h = 0; h < K; ++h) mla_front2(Y.ml, fr[h]);
                for (int h = 0; h < K; ++h) {
                    position = base + off[h];
                    ya[h] = mla_back(Y.ml, fr[h]);
                    tk_att[h] = stage(ya[h], pipe_buf[h], h);
                }
            }
            for (int h = 0; h < K; ++h) {
                auto &yo = unstage(tk_att[h], len[h], pipe_buf[h], h);
                auto st = site_multi(rh[h], &c[h], &yo, Y.hc_ffn, Y.hc_ffn_base, Y.hc_ffn_scale, Y.ffn_norm);
                save("attn", l, h);
                std::vector<Vec> x(len[h]);
                c_ffn[h].resize(len[h]);
                for (int t = 0; t < len[h]; ++t) x[t] = std::move(st[t].first), c_ffn[h][t] = std::move(st[t].second);
                y_ffn[h] = ffn_multi(l, x);
                tk_ffn[h] = stage(y_ffn[h], pipe_buf[K + h], K + h);
            }
        }
        for (int h = 0; h < K; ++h) {
            auto &yf = unstage(tk_ffn[h], len[h], pipe_buf[K + h], K + h);
            for (int t = 0; t < len[h]; ++t) write(rh[h][t], c_ffn[h][t], yf[t]);
            save("layer", layers_run - 1, h);
        }
        position = base + T;
        timers["comm.thread"] += comm.ar_thread_busy - busy0;   // the comm thread's all-reduce time in this pass
        std::vector<Vec> r;
        r.reserve(T);
        for (int h = 0; h < K; ++h) for (auto &v : rh[h]) r.push_back(std::move(v));
        last_hash = 1469598103934665603ull;   // as step_multi
        for (float v : r[T - 1]) {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            last_hash = (last_hash ^ u) * 1099511628211ull;
        }
        return pass_head(r);
    }
    // the end of a pass: the last residual's hidden through out_norm (all tokens' kept for the draft block), the head
    // for every token (or the last one, head_last), the logits to rank 0 or this rank's rows in last_part
    std::vector<Vec> pass_head(std::vector<Vec> &r) {
        const int T = int(r.size()), H = m.hidden;
        if (layers_run < int(m.layers.size())) return {};
        const double t = now();
        std::vector<Vec> hn(T);
        for (int k = 0; k < T; ++k) {
            Vec collapsed(H);
            for (int d = 0; d < H; ++d) collapsed[d] = (r[k][d] + r[k][H + d] + r[k][2 * H + d] + r[k][3 * H + d]) * .25f;
            hn[k] = norm(collapsed, out_norm);
        }
        if (mtp.on) last_hidden = hn;
        if (head_last && !full_logits) hn.erase(hn.begin(), hn.end() - 1);   // a prefill chunk: only its last token's
        const int TH = int(hn.size());
        auto ah = acts_of(hn, H, needs_q8(head));
        Vec part(size_t(TH) * head.rows);
        gemv_multi({{&head, ah.data(), part.data(), TH, head.rows}});
        std::vector<Vec> out;
        if (!full_logits) {
            last_part = split_rows(part, TH, head.rows, head.rows);
            timers["head"] += now() - t;
            return out;
        }
        for (int k = 0; k < T; ++k) {
            Vec logits(comm.rank == 0 ? m.vocab : vocab.end, 0.f);
            std::copy(part.begin() + size_t(k) * head.rows, part.begin() + size_t(k + 1) * head.rows, logits.begin() + vocab.begin);
            comm.gather(logits.data(), vocab.begin, vocab.end, m.vocab);
            if (comm.rank == 0) out.push_back(std::move(logits));
        }
        timers["head"] += now() - t;
        return out;
    }
    std::vector<Vec> step_multi(const std::vector<int> &tokens, const std::vector<std::filesystem::path> &dumps,
                                std::vector<MainState> *snaps = nullptr) {
        const int T = int(tokens.size()), H = m.hidden;
        snap_sink = snaps;
        if (snaps) {
            if (snaps->size() < size_t(std::max(0, T - 1))) snaps->resize(std::max(0, T - 1));
            for (int j = 0; j + 1 < T; ++j) (*snaps)[j].position = position + j + 1;
        }
        if (pass_groups.size() > 1) {   // several contexts: their snapshot slots and positions
            int sum = 0;
            for (const auto &[c, n] : pass_groups) {
                sum += n;
                if (!group_snaps) continue;
                switch_ctx(c);
                auto &gs = (*group_snaps)[c];
                if (gs.size() < size_t(std::max(0, n - 1))) gs.resize(std::max(0, n - 1));
                for (int j = 0; j + 1 < n; ++j) gs[j].position = position + j + 1;
            }
            if (sum != T || snaps) throw std::runtime_error("runner: context groups do not cover the pass");
        }
        if (T < 1 || T > MAX_PASS) throw std::runtime_error("runner: multi-token pass of 1.." + std::to_string(MAX_PASS) + " tokens");
        std::vector<Vec> r(T);
        if (T > 4)   // a long pass: token t's residual allocated (first touched) by thread t % team, its owner in site_tokens
            team_run([&] {
                const int th = omp_get_thread_num(), nth = omp_get_num_threads();
                for (int t = th; t < T; t += nth) r[t].assign(4 * H, 0.f);
            });
        else
            for (auto &v : r) v.assign(4 * H, 0.f);
        for (int t = 0; t < T; ++t) {
            if (tokens[t] < 0 || tokens[t] >= m.vocab) throw std::runtime_error("runner: token id outside the vocabulary");
            const uint16_t *ed = embedding.data() + size_t(tokens[t]) * H;
            for (int i = 0; i < H; ++i) {
                const uint32_t v = uint32_t(ed[i]) << 16;
                float f;
                std::memcpy(&f, &v, 4);
                for (int k = 0; k < 4; ++k) r[t][k * H + i] = f;
            }
        }
        auto save = [&](const char *kind, int l, int t) {
            if (dumps.empty() || dumps[t].empty() || comm.rank) return;
            std::ofstream o(dumps[t] / (std::string(kind) + "-" + std::to_string(l) + ".bin"), std::ios::binary);
            o.write(reinterpret_cast<const char *>(r[t].data()), r[t].size() * 4);
        };
        std::vector<Vec> c_ffn(T), y_ffn(T);
        for (int l = 0; l < layers_run; ++l) {
            auto &Y = L[l];
            cur_layer = l;
            std::vector<Vec> x(T), c(T);
            {
                auto s = site_multi(r, l ? &c_ffn : nullptr, l ? &y_ffn : nullptr, Y.hc_attn, Y.hc_attn_base, Y.hc_attn_scale, Y.attn_norm);
                for (int t = 0; t < T; ++t) {
                    x[t] = std::move(s[t].first), c[t] = std::move(s[t].second);
                    if (l) save("layer", l - 1, t);
                }
            }
            // several contexts (pass_groups): the layers run their projections over every token and their recurrence /
            // selection and attention per context
            auto y = Y.kda ? kda_multi(Y.kd, x) : mla_multi(Y.ml, x);
            double tc = now();
            allreduce_rows(y);
            timers["comm"] += now() - tc;
            {
                auto s = site_multi(r, &c, &y, Y.hc_ffn, Y.hc_ffn_base, Y.hc_ffn_scale, Y.ffn_norm);
                for (int t = 0; t < T; ++t) {
                    save("attn", l, t);
                    x[t] = std::move(s[t].first), c_ffn[t] = std::move(s[t].second);
                }
            }
            auto y2 = ffn_multi(l, x);
            tc = now();
            allreduce_rows(y2);
            timers["comm"] += now() - tc;
            y_ffn = std::move(y2);
            if (l + 1 == layers_run)
                for (int t = 0; t < T; ++t) {
                    write(r[t], c_ffn[t], y_ffn[t]);
                    save("layer", l, t);
                }
        }
        if (pass_groups.size() > 1)
            for (const auto &[c, n] : pass_groups) switch_ctx(c), position += n;
        else position += T;
        snap_sink = nullptr;
        last_hash = 1469598103934665603ull;   // FNV-1a over the last token's residual, as step()
        for (float v : r[T - 1]) {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            last_hash = (last_hash ^ u) * 1099511628211ull;
        }
        return pass_head(r);
    }
};

std::vector<int> read_ids(const std::string &path, size_t limit) {
    std::ifstream in(path);
    std::string text((std::istreambuf_iterator<char>(in)), {});
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream is(text);
    std::vector<int> ids;
    for (int t; is >> t && ids.size() < limit;) ids.push_back(t);
    return ids;
}


// --generate N: greedy decoding after the --eval tokens as the prompt; with --mtp K (1..3), K-deep MTP drafts verified
// by one main pass of K + 1 positions per round (the longest prefix where the main model's argmax agrees is kept,
// then its argmax at the last kept position). The main model's state rolls back to the kept prefix; the MTP block's
// to its first cell of the round (that cell paired the exact main hidden), and the kept drafts' cells are recomputed
// from the verified hiddens. The tokens equal plain greedy decoding's: rank 0 prints them for comparison.
int generate(Runner &runner, Comm &comm, const std::vector<int> &prompt, int count, int depth) {
    const auto &m = runner.m;
    runner.full_logits = false;   // greedy tokens from each rank's vocabulary slice (Comm::argmax), no gather
    auto pick = [&] {
        const double ta = now();
        const int tk = runner.argmax_last().at(0);
        runner.timers["glue.argmax"] += now() - ta;
        return tk;
    };
    std::vector<Vec> hid;
    double t0 = now();
    // STRATA_RUNNER_PREFILL_CHUNK=N: the prompt in multi-token passes of N positions (default 1024: 2K 245.5 / 8K 216.7
    // tok/s against 215.4 at 128, 2026-10-25; 1: one step() per token); the tokens and states are the same (a pass
    // is bitwise equal to its single steps)
    const size_t chunk = std::getenv("STRATA_RUNNER_PREFILL_CHUNK") ? std::clamp(std::atoi(std::getenv("STRATA_RUNNER_PREFILL_CHUNK")), 1, MAX_PASS) : 1024;
    if (chunk == 1)
        for (int tok : prompt) {
            runner.step(tok, {});
            if (depth) hid.push_back(runner.last_hidden.at(0));
        }
    else
        for (size_t q0 = 0; q0 < prompt.size(); q0 += chunk) {
            const std::vector<int> tk(prompt.begin() + q0, prompt.begin() + std::min(prompt.size(), q0 + chunk));
            runner.head_last = true;
            // the pass in two halves with overlapped all-reduces (passes of 64+ tokens; STRATA_RUNNER_PREFILL_PIPE=0: off)
            static const bool pipe = prefill_pipe();
            if (pipe && tk.size() >= 64) runner.step_multi_pipe(tk);
            else runner.step_multi(tk, {});
            runner.head_last = false;
            if (depth) hid.insert(hid.end(), runner.last_hidden.begin(), runner.last_hidden.end());
            const size_t done = q0 + tk.size();
            if (comm.rank == 0 && done % 8192 < chunk)   // long prompts: progress every 8192 positions
                std::cerr << "RUNNER prefill " << done << " / " << prompt.size() << " positions, " << now() - t0 << " s, "
                          << double(done) / (now() - t0) << " tok/s\n";
        }
    if (chunk > 1) runner.moe_use_pool.clear();   // decode builds its own (smaller, placed by its own plan)
    comm.ar_shutdown();
    const double prefill_main_s = now() - t0;
    if (depth) {   // the MTP block's cells over the prompt: token q with the hidden of q - 1
        const size_t mc = std::max<size_t>(4, chunk);
        for (size_t q0 = 0; q0 < prompt.size(); q0 += mc) {
            std::vector<int> tk;
            std::vector<Vec> pv;
            for (size_t q = q0; q < std::min(prompt.size(), q0 + mc); ++q) tk.push_back(prompt[q]), pv.push_back(q ? hid[q - 1] : Vec(m.hidden, 0.f));
            runner.mtp_forward(tk, pv, int(q0), nullptr);
        }
    }
    const double prefill_s = now() - t0;
    if (std::getenv("STRATA_RUNNER_PREFILL_TIMERS") && comm.rank == 0) {   // rank 0's phase timers of the prefill, per prompt token
        std::ostringstream o;
        o << "RUNNER_PREFILL_TIMERS chunk=" << chunk << " main_s=" << prefill_main_s << " ms_per_token";
        for (const auto &[k, v] : runner.timers) o << ' ' << k << '=' << v / double(prompt.size()) * 1e3;
        std::cout << o.str() << '\n';
        std::ostringstream m;   // every thread's mHC site time, ms per prompt token
        m << "RUNNER_MHC_THREADS";
        for (int i = 0; i < omp_get_max_threads() && i < 64; ++i) m << ' ' << runner.mhc_thr_total[i] / double(prompt.size()) * 1e3;
        std::cout << m.str() << '\n';
    }
    runner.timers.clear();   // decode only
    comm.argmax_local_s = comm.argmax_comm_s = 0;
    std::vector<int> out;
    int t = pick(), passes = 0, drafted = 0, accepted_total = 0;
    // STRATA_RUNNER_MTP_TAU1 / TAU2: verify the first / later drafts only when the MTP block's probability for them is
    // at least that (0: always, the default)
    const double tau1 = std::getenv("STRATA_RUNNER_MTP_TAU1") ? std::atof(std::getenv("STRATA_RUNNER_MTP_TAU1")) : 0;
    const double tau2 = std::getenv("STRATA_RUNNER_MTP_TAU2") ? std::atof(std::getenv("STRATA_RUNNER_MTP_TAU2")) : 0;
    // STRATA_RUNNER_MTP_ADAPT=1: the depth per round from 1 to --mtp K, the one with the most expected tokens per ms:
    // tokens(k) = 1 + sum_i<=k prod_j<=i a_j with a_j the moving acceptance of draft j given drafts < j accepted,
    // cost(k) = base + k * step (12 nodes: a round of 1 / 2 / 3 positions with its drafts ~22 / 32 / 42 ms). The tokens
    // do not depend on the depth (every kept token is the main model's greedy choice), only the speed does.
    const bool adapt = std::getenv("STRATA_RUNNER_MTP_ADAPT") && std::string(std::getenv("STRATA_RUNNER_MTP_ADAPT")) == "1";
    // STRATA_RUNNER_MTP_LAZY=1: keep the draft block's cells of accepted drafts from drafting (made from the draft
    // chain's hidden, not the verified one) instead of recomputing them; drafts only, the tokens do not change
    const bool lazy = std::getenv("STRATA_RUNNER_MTP_LAZY") && std::string(std::getenv("STRATA_RUNNER_MTP_LAZY")) == "1";
    const double a_base = std::getenv("STRATA_RUNNER_MTP_BASE") ? std::atof(std::getenv("STRATA_RUNNER_MTP_BASE")) : 22.0;
    const double a_step = std::getenv("STRATA_RUNNER_MTP_STEP") ? std::atof(std::getenv("STRATA_RUNNER_MTP_STEP")) : 10.0;
    double acc_rate[4] = {0.9, 0.9, 0.9, 0.9};   // moving conditional acceptance per draft position (optimistic start)
    int depth_hist[4] = {0, 0, 0, 0}, rounds = 0;
    auto choose_depth = [&] {
        if (!adapt) return depth;
        int best = 1;
        double best_v = 0, run = 1, tok = 1;
        for (int k = 1; k <= depth; ++k) {
            run *= acc_rate[k - 1], tok += run;
            const double v = tok / (a_base + k * a_step);
            if (v > best_v) best_v = v, best = k;
        }
        if (++rounds % 16 == 0) best = std::min(depth, best + 1);   // one deeper now and then: that position gets judged
        return best;
    };
    double phase[4] = {0, 0, 0, 0};   // draft, verify, rollback, resync
    Vec h_last = depth ? hid.back() : Vec();
    // the re-sync joins the next depth-1 draft (as generate_multi; STRATA_RUNNER_MTP_FUSE_RS=0: its own pass)
    static const bool fuse_rs = !(std::getenv("STRATA_RUNNER_MTP_FUSE_RS") && std::string(std::getenv("STRATA_RUNNER_MTP_FUSE_RS")) == "0");
    std::vector<int> pend_tok;
    std::vector<Vec> pend_prev;
    int pend_cell = 0;
    t0 = now();
    while (int(out.size()) < count) {
        if (depth == 0) {
            out.push_back(t);
            runner.step(t, {});
            t = pick();
            ++passes;
            continue;
        }
        const int p1 = runner.position;
        double tr = now();
        std::vector<int> win{t};
        std::vector<Vec> lg;
        int di = 0;   // the draft's row in the call's logits (the last; after a pending re-sync's cells)
        std::vector<Vec> hd;
        if (!pend_tok.empty()) {
            pend_tok.push_back(t), pend_prev.push_back(h_last);
            hd = runner.mtp_forward(pend_tok, pend_prev, pend_cell, &lg);
            di = int(pend_tok.size()) - 1, hd = {hd.back()};
            pend_tok.clear(), pend_prev.clear();
        } else hd = runner.mtp_forward({t}, {h_last}, p1, &lg);
        const auto anchor = runner.mtp_save();
        auto draft = [&](double tau) {   // the draft token, or -1 when its MTP probability is under tau
            double prob = 1;
            const double ta = now();
            const int tk = tau > 0 ? comm.argmax_prob(runner.last_part[di].data(), runner.vocab.end - runner.vocab.begin, runner.vocab.begin, prob)
                                   : runner.argmax_last().at(di);
            di = 0;
            runner.timers["glue.argmax"] += now() - ta;
            return prob >= tau ? tk : -1;
        };
        const int depth_now = choose_depth();
        ++depth_hist[depth_now - 1];
        int dk = draft(tau1);
        if (dk >= 0) win.push_back(dk);
        std::vector<Runner::MtpState> after_cell;   // lazy: the draft block after the cell of win[j], j >= 1
        for (int k = 2; k <= depth_now && dk >= 0; ++k) {
            hd = runner.mtp_forward({win.back()}, {hd[0]}, p1 + k - 1, &lg);
            if (lazy) after_cell.push_back(runner.mtp_save());
            dk = draft(tau2);
            if (dk >= 0) win.push_back(dk);
        }
        const int drafts = int(win.size()) - 1;
        auto &snaps = runner.snap_pool;   // reused: its arena buffers are allocated once
        phase[0] += now() - tr, tr = now();
        runner.step_multi(win, {}, &snaps);
        phase[1] += now() - tr, tr = now();
        const auto hid_win = runner.last_hidden;
        const auto best = runner.argmax_last();   // the main model's token after each window position, on every rank
        int acc = 0;   // accepted drafts
        while (acc < drafts && best[acc] == win[acc + 1]) ++acc;
        for (int j = 0; j < drafts && j <= acc; ++j)   // draft j was judged (the ones before it accepted)
            acc_rate[j] = 0.9 * acc_rate[j] + 0.1 * (j < acc ? 1.0 : 0.0);
        int res[2] = {acc, best[acc]};
        for (int j = 0; j <= acc; ++j) out.push_back(win[j]);
        if (acc < drafts) runner.restore(snaps[acc]);
        if (!lazy || acc == 0) runner.mtp_restore(anchor);
        else if (acc < drafts) runner.mtp_restore(after_cell[acc - 1]);   // the kept drafts' cells from drafting
        phase[2] += now() - tr, tr = now();
        if (lazy && acc > 0) {   // only the last accepted draft's cell is missing (it was never drafted from)
            if (acc == drafts) runner.mtp_forward({win[acc]}, {hid_win[acc - 1]}, p1 + acc, nullptr);
        } else if (acc > 0 && fuse_rs) {
            pend_tok.assign(win.begin() + 1, win.begin() + 1 + acc);
            pend_prev.assign(hid_win.begin(), hid_win.begin() + acc);
            pend_cell = p1 + 1;
        } else if (acc > 0) {
            std::vector<int> tk(win.begin() + 1, win.begin() + 1 + acc);
            std::vector<Vec> pv(hid_win.begin(), hid_win.begin() + acc);
            runner.mtp_forward(tk, pv, p1 + 1, nullptr);
        }
        phase[3] += now() - tr;
        h_last = hid_win[acc];
        t = res[1];
        ++passes, drafted += drafts, accepted_total += acc;
    }
    const double decode_s = now() - t0;
    out.resize(count);
    runner.timers["glue.argmax_local"] = comm.argmax_local_s, runner.timers["glue.argmax_comm"] = comm.argmax_comm_s;
    if (std::getenv("STRATA_RUNNER_SITE_TRACE") && runner.site_stat[3] > 0) {
        const double n = runner.site_stat[3];
        std::ostringstream o;
        o << "RUNNER_SITE_TRACE rank=" << comm.rank << " sites=" << n << " us latest_start=" << runner.site_stat[0] / n * 1e6
          << " latest_phaseA_end=" << runner.site_stat[1] / n * 1e6 << " thread0_phaseA=" << runner.site_stat[2] / n * 1e6 << " late_start_threads";
        for (int u = 0; u < 64; ++u) if (runner.site_late_start[u] > n / 20) o << ' ' << u << ':' << runner.site_late_start[u];
        o << " late_end_threads";
        for (int u = 0; u < 64; ++u) if (runner.site_late_end[u] > n / 20) o << ' ' << u << ':' << runner.site_late_end[u];
        std::cerr << o.str() << '\n';
    }
    if (std::getenv("STRATA_RUNNER_TIMERS_ALL")) {   // every rank's own timers per generated token
        char host[256] = {0};
        gethostname(host, sizeof host - 1);
        std::ostringstream o;
        o << "RUNNER_RANK_TIMERS rank=" << comm.rank << " host=" << host;
        for (const auto &[k, v] : runner.timers) o << ' ' << k << '=' << v / count * 1e3;
        std::cerr << o.str() << '\n';
    }
    if (comm.rank == 0) {   // rank 0's phase timers per generated token (prefill included)
        std::ostringstream tm;
        tm << "RUNNER_GEN_TIMERS ms_per_token";
        for (const auto &[k, v] : runner.timers) tm << ' ' << k << '=' << v / count * 1e3;
        std::cout << tm.str() << '\n';
    }
    if (comm.rank == 0) {
        std::ostringstream o;
        o << "RUNNER_GENERATED";
        for (int v : out) o << ' ' << v;
        std::cout << o.str() << '\n';
        std::cout << "RUNNER_GEN tokens=" << count << " depth=" << depth << " prompt=" << prompt.size() << " prefill_s=" << prefill_s
                  << " ms_per_token=" << decode_s / count * 1e3 << " passes=" << passes << " accepted=" << accepted_total << "/" << drafted
                  << " tokens_per_pass=" << double(count) / passes << " ms_per_pass draft=" << phase[0] / passes * 1e3
                  << " verify=" << phase[1] / passes * 1e3 << " rollback=" << phase[2] / passes * 1e3 << " resync=" << phase[3] / passes * 1e3;
        if (adapt) std::cout << " adapt_depths=" << depth_hist[0] << "/" << depth_hist[1] << "/" << depth_hist[2];
        std::cout << '\n';
    }
    return 0;
}

// --generate N with --contexts C: C prompts (the ids in C segments) decoded together. Each round drafts every
// unfinished context's MTP window (its own adaptive depth when STRATA_RUNNER_MTP_ADAPT=1), verifies all windows in
// one pass (one group per context: the attention per context, everything else over all tokens), then accepts and
// rolls back each context as generate() does, so each context's tokens are its own greedy decoding.
int generate_multi(Runner &runner, Comm &comm, const std::vector<int> &ids, int C, int count, int depth) {
    const auto &m = runner.m;
    runner.full_logits = false;
    const size_t seg = ids.size() / size_t(C);
    runner.add_contexts(C);
    struct Seq { std::vector<int> out; int t = 0, p1 = 0, drafted = 0, accepted = 0; Vec h_last; double acc_rate[4] = {0.9, 0.9, 0.9, 0.9}; int rounds = 0; };
    std::vector<Seq> sq(C);
    const size_t chunk = std::getenv("STRATA_RUNNER_PREFILL_CHUNK") ? std::clamp(std::atoi(std::getenv("STRATA_RUNNER_PREFILL_CHUNK")), 1, MAX_PASS) : 128;
    double t0 = now();
    for (int c = 0; c < C; ++c) {   // each context's prompt alone, as generate()
        runner.switch_ctx(c);
        const std::vector<int> prompt(ids.begin() + long(size_t(c) * seg), ids.begin() + long(size_t(c + 1) * seg));
        std::vector<Vec> hid;
        for (size_t q0 = 0; q0 < prompt.size(); q0 += chunk) {
            const std::vector<int> tk(prompt.begin() + long(q0), prompt.begin() + long(std::min(prompt.size(), q0 + chunk)));
            runner.head_last = true;
            runner.step_multi(tk, {});
            runner.head_last = false;
            if (depth) hid.insert(hid.end(), runner.last_hidden.begin(), runner.last_hidden.end());
        }
        sq[c].t = runner.argmax_last().at(0);
        if (depth) {
            const size_t mc = std::max<size_t>(4, chunk);
            for (size_t q0 = 0; q0 < prompt.size(); q0 += mc) {
                std::vector<int> tk;
                std::vector<Vec> pv;
                for (size_t q = q0; q < std::min(prompt.size(), q0 + mc); ++q) tk.push_back(prompt[q]), pv.push_back(q ? hid[q - 1] : Vec(m.hidden, 0.f));
                runner.mtp_forward(tk, pv, int(q0), nullptr);
            }
            sq[c].h_last = hid.back();
        }
    }
    runner.moe_use_pool.clear();
    const double prefill_s = now() - t0;
    runner.timers.clear();
    const bool adapt = std::getenv("STRATA_RUNNER_MTP_ADAPT") && std::string(std::getenv("STRATA_RUNNER_MTP_ADAPT")) == "1";
    const double a_base = std::getenv("STRATA_RUNNER_MTP_BASE") ? std::atof(std::getenv("STRATA_RUNNER_MTP_BASE")) : 22.0;
    const double a_step = std::getenv("STRATA_RUNNER_MTP_STEP") ? std::atof(std::getenv("STRATA_RUNNER_MTP_STEP")) : 10.0;
    auto choose_depth = [&](Seq &q) {   // generate()'s policy, per context
        if (!adapt) return depth;
        int best = 1;
        double best_v = 0, run = 1, tok = 1;
        for (int k = 1; k <= depth; ++k) {
            run *= q.acc_rate[k - 1], tok += run;
            const double v = tok / (a_base + k * a_step);
            if (v > best_v) best_v = v, best = k;
        }
        if (++q.rounds % 16 == 0) best = std::min(depth, best + 1);
        return best;
    };
    std::vector<std::vector<Runner::MainState>> gsnap(C);
    int passes = 0;
    // a context's MTP re-sync (its accepted drafts' cells) joins its next depth-1 draft: the cells follow on, so one
    // draft-block pass instead of two (a call streams the block's weights: ~2.6 ms at C = 16); a context that is done
    // needs none. STRATA_RUNNER_MTP_FUSE_RS=0: a re-sync pass after each verify
    static const bool fuse_rs = !(std::getenv("STRATA_RUNNER_MTP_FUSE_RS") && std::string(std::getenv("STRATA_RUNNER_MTP_FUSE_RS")) == "0");
    std::vector<std::vector<int>> pend_tok(C);
    std::vector<std::vector<Vec>> pend_prev(C);
    std::vector<int> pend_cell(C, 0);
    t0 = now();
    for (;;) {
        std::vector<int> all;
        std::vector<std::vector<int>> win(C);
        std::vector<Runner::MtpState> anchor(C);
        std::vector<std::pair<int, int>> groups;
        // the drafts of every unfinished context, depth by depth, one draft-block pass per depth over all contexts
        // still drafting (mtp_forward_ctx; each context's MTP state sees its own calls in their order: the same
        // drafts as one context at a time). STRATA_RUNNER_MTP_BATCH=0: one context at a time
        static const bool mtp_batch = !(std::getenv("STRATA_RUNNER_MTP_BATCH") && std::string(std::getenv("STRATA_RUNNER_MTP_BATCH")) == "0");
        if (mtp_batch && depth) {
            std::vector<int> act, dn(C, 0);
            std::vector<Vec> hd(C);
            for (int c = 0; c < C; ++c)
                if (int(sq[c].out.size()) < count) act.push_back(c), sq[c].p1 = runner.position_ctx(c), win[c] = {sq[c].t}, dn[c] = choose_depth(sq[c]);
            for (int k = 1; !act.empty(); ++k) {
                std::vector<int> tok, cells;
                std::vector<Vec> prev;
                std::vector<std::pair<int, int>> g;
                for (int c : act)
                    if (dn[c] >= k) {
                        if (k == 1 && !pend_tok[c].empty()) {   // the pending re-sync first, then the draft's cell
                            tok.insert(tok.end(), pend_tok[c].begin(), pend_tok[c].end());
                            prev.insert(prev.end(), pend_prev[c].begin(), pend_prev[c].end());
                            cells.push_back(pend_cell[c]), g.push_back({c, int(pend_tok[c].size()) + 1});
                            pend_tok[c].clear(), pend_prev[c].clear();
                        } else cells.push_back(sq[c].p1 + k - 1), g.push_back({c, 1});
                        tok.push_back(win[c].back()), prev.push_back(k == 1 ? sq[c].h_last : hd[c]);
                    }
                if (g.empty()) break;
                std::vector<Vec> lg;
                const auto h = runner.mtp_forward_ctx(tok, prev, g, cells, &lg);
                const double ta = now();
                const auto am = runner.argmax_last();
                runner.timers["mtp.argmax"] += now() - ta;
                for (size_t i = 0, e = 0; i < g.size(); ++i) {   // each group's last token is the draft
                    const int c = g[i].first;
                    e += g[i].second;
                    hd[c] = h[e - 1];
                    win[c].push_back(am.at(e - 1));
                    if (k == 1) anchor[c] = runner.mtp_save_ctx(c);
                }
            }
            for (int c : act) all.insert(all.end(), win[c].begin(), win[c].end()), groups.push_back({c, int(win[c].size())});
        } else
        for (int c = 0; c < C; ++c) {
            if (int(sq[c].out.size()) >= count) continue;
            runner.switch_ctx(c);
            Seq &q = sq[c];
            q.p1 = runner.position;
            win[c] = {q.t};
            if (depth) {
                std::vector<Vec> lg;
                auto hd = runner.mtp_forward({q.t}, {q.h_last}, q.p1, &lg);
                anchor[c] = runner.mtp_save();
                const int dn = choose_depth(q);
                win[c].push_back(runner.argmax_last().at(0));
                for (int k = 2; k <= dn; ++k) {
                    hd = runner.mtp_forward({win[c].back()}, {hd[0]}, q.p1 + k - 1, &lg);
                    win[c].push_back(runner.argmax_last().at(0));
                }
            }
            all.insert(all.end(), win[c].begin(), win[c].end());
            groups.push_back({c, int(win[c].size())});
        }
        if (groups.empty()) break;
        if (groups.size() == 1) {   // one context left: an ordinary pass with its snapshots
            runner.switch_ctx(groups[0].first);
            runner.step_multi(all, {}, &gsnap[groups[0].first]);
        } else {
            runner.pass_groups = groups, runner.group_snaps = &gsnap;
            runner.step_multi(all, {});
            runner.pass_groups.clear(), runner.group_snaps = nullptr;
        }
        const auto hid_all = runner.last_hidden;
        const auto best = runner.argmax_last();
        std::vector<int> rs_tok, rs_cells;
        std::vector<Vec> rs_prev;
        std::vector<std::pair<int, int>> rs_g;
        int off = 0;
        for (const auto &[c, n] : groups) {
            runner.switch_ctx(c);
            Seq &q = sq[c];
            const int drafts = n - 1;
            int acc = 0;
            while (acc < drafts && best[off + acc] == win[c][acc + 1]) ++acc;
            for (int j = 0; j < drafts && j <= acc; ++j) q.acc_rate[j] = 0.9 * q.acc_rate[j] + 0.1 * (j < acc ? 1.0 : 0.0);
            for (int j = 0; j <= acc; ++j) q.out.push_back(win[c][j]);
            if (acc < drafts) runner.restore(gsnap[c][acc]);
            if (depth && mtp_batch) {   // the MTP re-syncs of every context in one pass below
                runner.mtp_restore_ctx(c, anchor[c]);
                if (acc > 0 && fuse_rs) {
                    pend_tok[c].assign(win[c].begin() + 1, win[c].begin() + 1 + acc);
                    pend_prev[c].assign(hid_all.begin() + off, hid_all.begin() + off + acc);
                    pend_cell[c] = q.p1 + 1;
                } else if (acc > 0) {
                    rs_tok.insert(rs_tok.end(), win[c].begin() + 1, win[c].begin() + 1 + acc);
                    rs_prev.insert(rs_prev.end(), hid_all.begin() + off, hid_all.begin() + off + acc);
                    rs_g.push_back({c, acc}), rs_cells.push_back(q.p1 + 1);
                }
                q.h_last = hid_all[off + acc];
            } else if (depth) {
                runner.mtp_restore(anchor[c]);
                if (acc > 0) {
                    std::vector<int> tk(win[c].begin() + 1, win[c].begin() + 1 + acc);
                    std::vector<Vec> pv(hid_all.begin() + off, hid_all.begin() + off + acc);
                    runner.mtp_forward(tk, pv, q.p1 + 1, nullptr);
                }
                q.h_last = hid_all[off + acc];
            }
            q.t = best[off + acc];
            q.drafted += drafts, q.accepted += acc;
            off += n;
        }
        if (!rs_g.empty()) runner.mtp_forward_ctx(rs_tok, rs_prev, rs_g, rs_cells, nullptr);
        ++passes;
    }
    const double decode_s = now() - t0;
    if (comm.rank == 0) {
        int drafted = 0, accepted = 0;
        for (int c = 0; c < C; ++c) {
            sq[c].out.resize(count);
            std::ostringstream o;
            o << "RUNNER_GENERATED_CTX " << c;
            for (int v : sq[c].out) o << ' ' << v;
            std::cout << o.str() << '\n';
            drafted += sq[c].drafted, accepted += sq[c].accepted;
        }
        std::ostringstream tm;
        tm << "RUNNER_GEN_TIMERS ms_per_token";
        for (const auto &[k, v] : runner.timers) tm << ' ' << k << '=' << v / double(C * count) * 1e3;
        std::cout << tm.str() << '\n';
        std::cout << "RUNNER_GEN_MULTI contexts=" << C << " tokens=" << count << " depth=" << depth << " prompt=" << seg << " prefill_s=" << prefill_s
                  << " ms_per_token=" << decode_s / double(C * count) * 1e3 << " tok_s_total=" << double(C * count) / decode_s
                  << " passes=" << passes << " ms_per_pass=" << decode_s / passes * 1e3 << " accepted=" << accepted << "/" << drafted << '\n';
    }
    return 0;
}

double kl_and_top(const Vec &ref_logits, const Vec &got, bool &top_equal) {
    const auto mr = *std::max_element(ref_logits.begin(), ref_logits.end()), mg = *std::max_element(got.begin(), got.end());
    double zr = 0, zg = 0;
    for (size_t i = 0; i < got.size(); ++i) zr += std::exp(double(ref_logits[i]) - mr), zg += std::exp(double(got[i]) - mg);
    double kl = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double pr = std::exp(double(ref_logits[i]) - mr) / zr;
        const double lr = double(ref_logits[i]) - mr - std::log(zr), lg = double(got[i]) - mg - std::log(zg);
        if (pr > 0) kl += pr * (lr - lg);
    }
    top_equal = (std::max_element(ref_logits.begin(), ref_logits.end()) - ref_logits.begin()) ==
                (std::max_element(got.begin(), got.end()) - got.begin());
    return kl;
}

}   // namespace

// A two-level sense-reversing barrier for one team pinned close on A64FX (thread t on CMG t / group): threads meet on
// their CMG's counter (one 256-byte line in that CMG's L2), the last of each CMG meets the other CMGs' last on a
// top counter, and each CMG is released through its own line. Every thread calls wait(t) with its team index.
struct TeamBarrier {
    struct alignas(256) Line { std::atomic<int> count{0}; std::atomic<int> sense{0}; };
    Line group[16], top;
    int n = 0, gsize = 12, groups = 1;
    alignas(256) int local[64][64] = {};   // per-thread sense, one line apart
    void init(int team, int per_group = 12) { n = team, gsize = per_group, groups = (team + per_group - 1) / per_group; }
    void wait(int t) {
        int &ls = local[t][0];
        ls ^= 1;
        const int g = t / gsize, members = std::min(gsize, n - g * gsize);
        Line &G = group[g];
        if (G.count.fetch_add(1, std::memory_order_acq_rel) == members - 1) {
            G.count.store(0, std::memory_order_relaxed);
            if (top.count.fetch_add(1, std::memory_order_acq_rel) == groups - 1) {
                top.count.store(0, std::memory_order_relaxed);
                top.sense.store(ls, std::memory_order_release);
            } else
                while (top.sense.load(std::memory_order_acquire) != ls) {}
            G.sense.store(ls, std::memory_order_release);
        } else
            while (G.sense.load(std::memory_order_acquire) != ls) {}
    }
};


// --bench-gemv: the runner's GEMV path on synthetic matrices of the decode shapes, no model (one node)
int bench_gemv() {
    std::mt19937 rng(7);
    auto make = [&](Kind kind, int rows, int cols) {
        Matrix m;
        m.kind = kind, m.rows = rows, m.padded = (rows + 15) / 16 * 16, m.cols = cols;
        const int R = m.padded;
        std::vector<uint8_t> buf;
        if (kind == Kind::NvP || kind == Kind::NvQ) {
            std::vector<uint8_t> codes(size_t(R) * cols / 2), scales(size_t(R) * cols / 16, 0x38);
            for (auto &c : codes) c = uint8_t(rng());
            buf.resize(gf4_nvfp4_packed_bytes(R, cols));
            (kind == Kind::NvP ? gf4_pack_rt16p : gf4_pack_rt16q)(codes.data(), scales.data(), R, cols, buf.data());
        } else if (kind == Kind::Mx) {
            std::vector<uint8_t> w(size_t(R) * cols), e(size_t(R) * cols / 32, 127);
            for (auto &c : w) c = uint8_t(rng() % 0x77);
            buf.resize(gf4_e8k_packed_bytes(R, cols, 1));
            gf4_pack_e8k_mx(w.data(), e.data(), R, cols, buf.data());
        } else {
            std::vector<uint16_t> w(size_t(R) * cols);
            for (auto &c : w) c = uint16_t(0x3C00 | (rng() & 0x7F));
            buf.resize(size_t(R) * cols * 2);
            gf4_pack_b16k(w.data(), R, cols, buf.data());
        }
        m.rot = next_rotation();
        m.data = place(buf, m);
        return m;
    };
    Vec x(16384);
    for (auto &v : x) v = float(int(rng() % 200) - 100) * 0.01f;
    struct Case { const char *name; Kind kind; int rows, cols, jobs; };
    const Case cases[] = {{"router_b16", Kind::B16, 288, 4096, 1},     {"hc_b16", Kind::B16, 24, 16384, 1},
                          {"expert_up_nvp", Kind::NvP, 192, 4096, 16}, {"expert_down_nvp", Kind::NvP, 4096, 192, 8},
                          {"kda_qkv_mx", Kind::Mx, 768, 4096, 3},      {"big_nvp", Kind::NvP, 2048, 4096, 1},
                          {"expert_up_nvq", Kind::NvQ, 192, 4096, 16}, {"kda_o_mx", Kind::Mx, 4096, 768, 1},
                          {"expert_down_nvq", Kind::NvQ, 4096, 192, 8},
                          {"huge_mx", Kind::Mx, 16384, 4096, 1},       {"huge_nvp", Kind::NvP, 16384, 4096, 1},
                          {"huge_nvq", Kind::NvQ, 16384, 4096, 1}};
    const char *only = std::getenv("STRATA_BENCH_ONLY");
    {
#pragma omp parallel
        { __asm__ __volatile__("" ::: "memory"); }   // thread creation outside the timing
        const double t = now();
        for (int i = 0; i < 4000; ++i) {
#pragma omp parallel
            { __asm__ __volatile__("" ::: "memory"); }
        }
        std::cerr << "BENCH omp_fork us " << (now() - t) / 4000 * 1e6 << " threads " << omp_get_max_threads() << '\n';
        static TeamBarrier tb;
        static FlagBarrier fb;
        double t_omp = 0, t_team = 0, t_flag = 0;
#pragma omp parallel
        {
            const int th = omp_get_thread_num();
#pragma omp single
            tb.init(omp_get_num_threads()), fb.init(omp_get_num_threads());
            for (int rep = 0; rep < 2; ++rep) {
#pragma omp barrier
                double t1 = now();
                for (int i = 0; i < 20000; ++i) {
#pragma omp barrier
                }
                double t2 = now();
                for (int i = 0; i < 20000; ++i) tb.wait(th);
                double t3 = now();
                for (int i = 0; i < 20000; ++i) fb.wait(th);
                double t4 = now();
                if (th == 0 && rep == 1) t_omp = (t2 - t1) / 20000, t_team = (t3 - t2) / 20000, t_flag = (t4 - t3) / 20000;
            }
        }
        std::cerr << "BENCH barrier us omp " << t_omp * 1e6 << " team " << t_team * 1e6 << " flag " << t_flag * 1e6 << '\n';
    }
    for (const auto &c : cases) {
        if (only && !std::strstr(only, c.name)) continue;
        const size_t one = Matrix{}.bytes() + (c.kind == Kind::NvP || c.kind == Kind::NvQ ? gf4_nvfp4_packed_bytes((c.rows + 15) / 16 * 16, c.cols)
                                               : c.kind == Kind::Mx ? gf4_e8k_packed_bytes((c.rows + 15) / 16 * 16, c.cols, 1)
                                                                    : size_t((c.rows + 15) / 16 * 16) * c.cols * 2);
        const int sets = int(std::max<size_t>(1, (size_t(96) << 20) / (one * c.jobs)));   // cold: cycle through > L2 (4 x 8 MiB)
        std::vector<std::vector<Matrix>> ms(sets);
        for (auto &set : ms)
            for (int j = 0; j < c.jobs; ++j) set.push_back(make(c.kind, c.rows, c.cols));
        {   // NUMA node of each page of the first matrix (move_pages with no target nodes only queries)
            const Matrix &m0 = ms[0][0];
            const size_t page = size_t(sysconf(_SC_PAGESIZE));
            std::vector<void *> pages;
            for (size_t b = 0; b < m0.bytes(); b += page) pages.push_back(const_cast<uint8_t *>(m0.data.data()) + b);
            std::vector<int> st(pages.size(), -99);
            const long rc = syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr, st.data(), 0);
            std::map<int, int> hist;
            for (int v : st) hist[v]++;
            std::cerr << "BENCH " << c.name << " pages " << pages.size() << " page_bytes " << page << " rc " << rc << " nodes";
            for (auto [node, cnt] : hist) std::cerr << ' ' << node << ':' << cnt;
            std::cerr << '\n';
        }
        // STRATA_BENCH_NT=N: every job for N tokens (a prefill chunk's GEMM), default 1
        const int NT = std::getenv("STRATA_BENCH_NT") ? std::clamp(std::atoi(std::getenv("STRATA_BENCH_NT")), 1, MAX_PASS) : 1;
        std::vector<Activation> av(NT);
        for (auto &a : av) a.set(x.data(), c.cols, c.kind == Kind::NvQ);
        std::vector<Vec> ys(c.jobs, Vec(size_t(NT) * c.rows));
        auto run = [&](int set) {
            std::vector<Job> jobs;
            for (int j = 0; j < c.jobs; ++j) jobs.push_back({&ms[set][j], av.data(), ys[j].data(), NT, c.rows});
            gemv_multi(jobs);
        };
        if (sets > 1) {   // every set's jobs in one region: long per-thread runs, as the kernel bench's stream mode
            std::vector<Job> all;
            std::vector<Vec> yall(size_t(sets) * c.jobs, Vec(size_t(NT) * c.rows));
            for (int st = 0; st < sets; ++st)
                for (int j = 0; j < c.jobs; ++j) all.push_back({&ms[st][j], av.data(), yall[size_t(st) * c.jobs + j].data(), NT, c.rows});
            gemv_multi(all);
            const double t = now();
            for (int i = 0; i < 20; ++i) gemv_multi(all);
            const double per = (now() - t) / 20;
            std::cerr << "BENCH " << c.name << " fused_cold jobs " << all.size() << " nt " << NT << " us " << per * 1e6 << " GBps " << one * all.size() / per * 1e-9
                      << " GMACps " << double(c.rows) * c.cols * NT * all.size() / per * 1e-9 << '\n';
        }
        for (int mode = 0; mode < 2; ++mode) {   // 0 warm (one set), 1 cold
            const int reps = mode ? std::max(200, 2 * sets) : 1000;
            run(0);
            const double t = now();
            for (int i = 0; i < reps; ++i) run(mode ? i % sets : 0);
            const double per = (now() - t) / reps;
            std::cerr << "BENCH " << c.name << (mode ? " cold" : " warm") << " jobs " << c.jobs << " rows " << c.rows << " cols " << c.cols
                      << " ksplit " << ksplit(ms[0][0], omp_get_max_threads()).chunks << " nt " << NT << " us " << per * 1e6 << " GBps " << one * c.jobs / per * 1e-9
                      << " GMACps " << double(c.rows) * c.cols * NT * c.jobs / per * 1e-9 << '\n';
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    // LLVM OpenMP's distributed barrier, unless the environment chooses: on 12 A64FX nodes (47 threads) it cut a fork
    // from 5.3 to 4.3 us and a barrier from 9.0 to 6.7 us, and the decode step from 33-35 to 28.5-29.9 ms (wider
    // hyper or tree branching was slower). Read when the runtime starts, so before the first OpenMP construct.
    setenv("KMP_FORKJOIN_BARRIER_PATTERN", "dist,dist", 0);
    setenv("KMP_PLAIN_BARRIER_PATTERN", "dist,dist", 0);
    setenv("KMP_REDUCTION_BARRIER_PATTERN", "dist,dist", 0);
    if (!(std::getenv("STRATA_RUNNER_FLAG_BARRIER") && std::string(std::getenv("STRATA_RUNNER_FLAG_BARRIER")) == "0"))
        g_bar.init(omp_get_max_threads(), cmg_threads());   // 1.6 us per barrier against 7.5 for OpenMP's (dist) at 47 threads
    if (argc >= 2 && std::string(argv[1]) == "--bench-gemv") return bench_gemv();
    Comm comm;
    comm.init(&argc, &argv);
    int code = 0;
    try {
        if (argc < 2) throw std::invalid_argument("usage: glm_cpu_runner MODEL_DIR --eval IDS [options]");
        std::string eval, save_logits, reference, numerics = "exact", dump_dir, image_dir;
        size_t tokens_limit = SIZE_MAX;
        int context = 4096, layers = -1, multi = 0;   // multi: positions per pass (step_multi), 0 = step()
        int generate_n = 0, mtp_depth = 0;   // --generate N tokens after the prompt, with --mtp K-deep drafts
        int contexts = 1;   // --contexts N: N resident sequences (eval: the ids in N segments, passes over all of them)
        std::set<size_t> dump_at;
        for (int i = 2; i + 1 < argc; i += 2) {
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "--eval") eval = v;
            else if (k == "--save-logits") save_logits = v;
            else if (k == "--reference") reference = v;
            else if (k == "--numerics") numerics = v;
            else if (k == "--context") context = std::stoi(v);
            else if (k == "--tokens") tokens_limit = std::stoul(v);
            else if (k == "--layers") layers = std::stoi(v);
            else if (k == "--dump-dir") dump_dir = v;
            else if (k == "--image-dir") image_dir = v;
            else if (k == "--threads") omp_set_num_threads(std::stoi(v));
            else if (k == "--multi") multi = std::stoi(v);
            else if (k == "--generate") generate_n = std::stoi(v);
            else if (k == "--mtp") mtp_depth = std::stoi(v);
            else if (k == "--contexts") contexts = std::stoi(v);
            else if (k == "--dump-at") {
                std::istringstream is(v);
                for (std::string t; std::getline(is, t, ',');) dump_at.insert(std::stoul(t));
            } else throw std::invalid_argument("unknown option " + k);
        }
        if (numerics != "exact" && numerics != "fast" && numerics != "w4a8" && numerics != "q8trunk" && numerics != "w4a8g16")
            throw std::invalid_argument("--numerics exact|fast|w4a8|w4a8g16|q8trunk");
        if (numerics == "w4a8g16") g_act_group = 16;   // W4A8 experts with one activation scale per K16 group, exact trunk
        if (!std::getenv("STRATA_RUNNER_SUBNORMALS")) {
#pragma omp parallel
            flush_subnormals();
            flush_subnormals();
            if (gf4_selftest_e4m3()) throw std::runtime_error("E4M3 vector decode changed under flush-to-zero");
        }
        if (eval.empty()) throw std::invalid_argument("--eval IDS is required");
        strata::core::ModelArtifact artifact{std::filesystem::path(argv[1])};
        if (!artifact.is_modelopt()) throw std::invalid_argument("the CPU runner reads the FP4-Local (ModelOpt) checkpoint directory");
        const auto &m = artifact.descriptor();
        if (layers < 0 || layers > int(m.layers.size())) layers = int(m.layers.size());
        const auto ids = read_ids(eval, tokens_limit);
        if (ids.empty()) throw std::invalid_argument("empty token file");
        if (mtp_depth < 0 || mtp_depth > 3) throw std::invalid_argument("--mtp 0..3");
        Runner runner(artifact, comm, numerics, std::max(context, int(ids.size()) + generate_n + 8), layers);
        runner.image_dir = image_dir;
        runner.want_mtp = mtp_depth > 0 || (std::getenv("STRATA_RUNNER_LOAD_MTP") && std::string(std::getenv("STRATA_RUNNER_LOAD_MTP")) == "1");   // 1: same image for plain runs
        runner.load();
        if (std::getenv("STRATA_RUNNER_PROBE_MTP") && runner.mtp.on) {   // one MTP cell from the first prompt token
            std::vector<Vec> lg;
            const auto h = runner.mtp_forward({ids[0]}, {Vec(m.hidden, 0.f)}, 0, &lg);
            double sq = 0;
            for (float v : h[0]) sq += double(v) * v;
            if (comm.rank == 0)
                std::cerr << "RUNNER_PROBE_MTP token " << ids[0] << " draft " << (std::max_element(lg[0].begin(), lg[0].end()) - lg[0].begin())
                          << " hidden_rms " << std::sqrt(sq / m.hidden) << '\n';
        }
        if (contexts > 1 && generate_n == 0) {   // eval: context c reads segment c; each pass takes --multi positions of every context
            if (multi < 1) throw std::invalid_argument("--contexts needs --multi");
            const size_t seg = ids.size() / size_t(contexts);
            runner.full_logits = false;
            with_team([&] { runner.add_contexts(contexts); });
            double total = 0;
            for (size_t s0 = 0; s0 < seg; s0 += size_t(multi)) {
                std::vector<int> tk;
                std::vector<std::filesystem::path> dumps;
                runner.pass_groups.clear();
                for (int c = 0; c < contexts; ++c) {
                    const int n = int(std::min(seg - s0, size_t(multi)));
                    for (int j = 0; j < n; ++j) {
                        tk.push_back(ids[size_t(c) * seg + s0 + j]);
                        std::filesystem::path d;
                        if (dump_at.count(s0 + j) && !dump_dir.empty()) {
                            d = std::filesystem::path(dump_dir) / ("dump-c" + std::to_string(c) + "-" + std::to_string(s0 + j));
                            if (comm.rank == 0) std::filesystem::create_directories(d);
                        }
                        dumps.push_back(d);
                    }
                    runner.pass_groups.push_back({c, n});
                }
                comm.barrier();
                const double t = now();
                with_team([&] { runner.step_multi(tk, dumps); });
                comm.barrier();
                total += now() - t;
            }
            runner.pass_groups.clear();
            if (comm.rank == 0)
                std::cout << "RUNNER_CTX_EVAL contexts=" << contexts << " positions=" << seg << " multi=" << multi << " ms_per_token=" << total / double(seg * contexts) * 1e3 << '\n';
            comm.finish();
            return 0;
        }
        if (generate_n > 0) {
            if (layers != int(m.layers.size())) throw std::invalid_argument("--generate needs every layer");
            with_team([&] { code = contexts > 1 ? generate_multi(runner, comm, ids, contexts, generate_n, mtp_depth) : generate(runner, comm, ids, generate_n, mtp_depth); });
            comm.finish();
            return code;
        }
        if (std::getenv("STRATA_RUNNER_PROBE_COLL")) {   // which MPI collectives work next to the uTofu all-reduce (rank 0 prints)
            for (int kb : {4, 64, 512, 4096}) {
                const int bytes = kb * 1024;
                std::vector<char> snd(bytes, char(comm.rank)), rcv(size_t(bytes) * comm.world);
                comm.barrier();
                double t = now();
                comm.allgather(snd.data(), bytes, rcv.data());
                const double t_ag = now() - t;
                bool ok = true;
                for (int r = 0; r < comm.world; ++r) ok = ok && rcv[size_t(r) * bytes] == char(r) && rcv[size_t(r) * bytes + bytes - 1] == char(r);
                std::vector<double> sd(size_t(bytes) / 8), rd(size_t(bytes) / 8);
                const int per = int(sd.size()) / comm.world;
                std::vector<int> sc(comm.world, per), sp(comm.world), rc(comm.world, per), rp(comm.world);
                for (int r = 0; r < comm.world; ++r) sp[r] = rp[r] = r * per;
                for (size_t i = 0; i < sd.size(); ++i) sd[i] = comm.rank * 1000 + int(i / std::max(1, per));
                comm.barrier();
                t = now();
                comm.alltoallv(sd.data(), sc.data(), sp.data(), rd.data(), rc.data(), rp.data());
                const double t_a2a = now() - t;
                bool ok2 = true;
                for (int r = 0; r < comm.world; ++r) ok2 = ok2 && rd[size_t(r) * per] == r * 1000 + comm.rank;
                const bool all_ok = comm.max_over_ranks((ok && ok2) ? 0 : 1) == 0;
                if (comm.rank == 0)
                    std::cerr << "RUNNER_PROBE_COLL bytes " << bytes << " allgather us " << t_ag * 1e6 << " alltoallv us " << t_a2a * 1e6 << (all_ok ? " OK" : " WRONG") << '\n';
            }
        }
        if (std::getenv("STRATA_RUNNER_PROBE_COMM")) {   // back-to-back all-reduces of 1, 2 and 4 hidden vectors, all ranks
          for (int mult : {1, 2, 4, 16, 64}) {
            Vec v(size_t(mult) * m.hidden, 1.f / comm.world);
            comm.barrier();
            for (int i = 0; i < 20; ++i) comm.allreduce(v.data(), int(v.size()));
            comm.barrier();
            double t = now();
            for (int i = 0; i < 200; ++i) comm.allreduce(v.data(), int(v.size()));
            const double per = (now() - t) / 200;
            comm.barrier();
            t = now();
            for (int i = 0; i < 200; ++i) { comm.allreduce(v.data(), int(v.size())); comm.barrier(); }
            const double with_barrier = (now() - t) / 200;
            const double worst = comm.max_over_ranks(per);
            if (comm.rank == 0)
                std::cerr << "RUNNER_PROBE allreduce " << v.size() << " floats us " << per * 1e6 << " max_over_ranks " << worst * 1e6
                          << " with_barrier " << with_barrier * 1e6 << '\n';
          }
        }
        if (std::getenv("STRATA_RUNNER_PROBE") && comm.rank == 0) {
            // us per call of the runner's own GEMV path on loaded matrices (overheads outside the kernels)
            auto probe = [&](const char *what, const Matrix &w) {
                Vec x(w.cols, 0.01f), y(w.rows);
                Activation a;
                a.set(x.data(), w.cols, needs_q8(w));
                gemv(w, a, y.data());
                double t = now();
                for (int i = 0; i < 200; ++i) gemv(w, a, y.data());
                const double per_call = (now() - t) / 200;
                t = now();
                for (int i = 0; i < 200; ++i) y = gemv(w, x);
                std::cerr << "RUNNER_PROBE " << what << " rows " << w.rows << " cols " << w.cols << " us_per_call " << per_call * 1e6
                          << " with_activation_us " << (now() - t) / 200 * 1e6 << " GBps " << w.bytes() / per_call * 1e-9 << '\n';
            };
            probe("hc_attn", runner.L[0].hc_attn);
            if (runner.L[0].kda) probe("kda_q", runner.L[0].kd.q), probe("kda_f_b", runner.L[0].kd.f_b), probe("kda_o", runner.L[0].kd.o);
            if (runner.L[0].dense) probe("dense_gate", runner.L[0].ffn.gate);
            Vec r(4 * m.hidden, 0.01f);
            double t = now();
            for (int i = 0; i < 200; ++i) r = runner.norm(r, {});
            std::cerr << "RUNNER_PROBE norm16384 us " << (now() - t) / 200 * 1e6 << '\n';
            auto &Y = runner.L[0];
            Vec rr(4 * m.hidden, 0.01f);
            t = now();
            for (int i = 0; i < 200; ++i) {
                auto [xx, c] = runner.read(Y.hc_attn, Y.hc_attn_base, Y.hc_attn_scale, rr);
                runner.write(rr, c, xx);
            }
            std::cerr << "RUNNER_PROBE read_write us " << (now() - t) / 200 * 1e6 << '\n';
            Vec emb(m.hidden, 0.01f);
            t = now();
            for (int i = 0; i < 20; ++i) emb = runner.kda(Y.kd, emb);
            std::cerr << "RUNNER_PROBE kda_layer us " << (now() - t) / 20 * 1e6 << '\n';
        }
        std::ifstream golden;
        if (!reference.empty() && comm.rank == 0) golden.open(std::filesystem::path(reference) / "logits.bin", std::ios::binary);
        std::ofstream saved;
        if (!save_logits.empty() && comm.rank == 0) saved.open(save_logits, std::ios::binary);
        double kl_sum = 0, decode_s = 0, nll_sum = 0;
        int top_hits = 0, compared = 0, next_hits = 0, scored = 0;
        std::vector<Vec> pending;   // --multi: the pass's logits not scored yet
        double pass_dt = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            std::filesystem::path dump;
            if (dump_at.count(i) && !dump_dir.empty()) {
                dump = std::filesystem::path(dump_dir) / ("dump-" + std::to_string(i));
                if (comm.rank == 0) std::filesystem::create_directories(dump);
            }
            Vec logits;
            double dt;
            if (multi > 0) {   // one pass per `multi` positions; each token's line carries the pass time / tokens
                const size_t start = i - i % size_t(multi);
                if (i == start) {
                    std::vector<int> chunk;
                    std::vector<std::filesystem::path> dumps;
                    for (size_t j = start; j < std::min(ids.size(), start + size_t(multi)); ++j) {
                        chunk.push_back(ids[j]);
                        std::filesystem::path d;
                        if (dump_at.count(j) && !dump_dir.empty()) {
                            d = std::filesystem::path(dump_dir) / ("dump-" + std::to_string(j));
                            if (comm.rank == 0) std::filesystem::create_directories(d);
                        }
                        dumps.push_back(d);
                    }
                    comm.barrier();
                    const double t = now();
                    static const bool pipe = prefill_pipe();
                    with_team([&] { pending = pipe && chunk.size() >= 64 ? runner.step_multi_pipe(chunk, dumps) : runner.step_multi(chunk, dumps); });
                    comm.barrier();
                    pass_dt = (now() - t) / double(chunk.size());
                    decode_s += now() - t;
                }
                dt = pass_dt;
                if (!pending.empty()) logits = std::move(pending[i - start]);
            } else {
                comm.barrier();
                const double t = now();
                with_team([&] { logits = runner.step(ids[i], dump); });
                comm.barrier();
                dt = now() - t;
                decode_s += dt;
            }
            if (i < 4 || i + 1 == ids.size())
                if (!comm.same_on_all(runner.last_hash) && comm.rank == 0)
                    std::cerr << "RUNNER_DIVERGED residual differs between ranks at position " << i << '\n';
            if (comm.rank) continue;
            if (logits.empty()) {
                std::cerr << "RUNNER token " << i << " id " << ids[i] << " ms " << dt * 1e3 << '\n';
                continue;
            }
            if (saved) saved.write(reinterpret_cast<const char *>(logits.data()), logits.size() * 4);
            const auto best = std::max_element(logits.begin(), logits.end()) - logits.begin();
            std::ostringstream line;
            line << "RUNNER token " << i << " id " << ids[i] << " argmax " << best << " ms " << dt * 1e3;
            if (i + 1 < ids.size()) {   // teacher-forced next-token score: negative log-likelihood and top-1
                double z = 0;
                const double mx = logits[best];
                for (float v : logits) z += std::exp(double(v) - mx);
                const double nll = -(double(logits[ids[i + 1]]) - mx - std::log(z));
                nll_sum += nll, ++scored, next_hits += best == ids[i + 1];
                line << " nll " << nll;
            }
            if (golden) {
                Vec g(m.vocab);
                if (golden.read(reinterpret_cast<char *>(g.data()), g.size() * 4)) {
                    bool top = false;
                    const double kl = kl_and_top(g, logits, top);
                    kl_sum += kl, top_hits += top, ++compared;
                    line << " kl " << kl << " top1 " << top;
                }
            }
            std::cerr << line.str() << '\n';
        }
        if (std::getenv("STRATA_RUNNER_PROBE_PHASES")) {
            // us per call of each per-layer phase in isolation, after the evaluation (kda/mla advance their state:
            // nothing is decoded afterwards); every rank runs the same items, rank 0 prints its own and the slowest
            const auto kept = runner.timers;   // the phase calls below must not count in RUNNER_TIMERS
            std::vector<std::pair<std::string, double>> items;
            auto time = [&](const std::string &what, int reps, const std::function<void()> &f) {
                f();
                const double t = now();
                for (int i = 0; i < reps; ++i) f();
                items.push_back({what, (now() - t) / reps});
            };
            int lk = -1, lm = -1;
            for (int l = 0; l < layers; ++l) {
                if (runner.L[l].kda && !runner.L[l].dense && lk < 0) lk = l;
                if (!runner.L[l].kda && !runner.L[l].dense && lm < 0) lm = l;
            }
            Vec h(m.hidden, 0.01f), r4(4 * m.hidden, 0.01f), y(64 * 1024);
            for (int i = 0; i < m.hidden; ++i) h[i] = 0.01f * float((i * 37) % 101 - 50);
            time("omp_empty", 2000, [] {
#pragma omp parallel
                { __asm__ __volatile__("" ::: "memory"); }   // keeps the region from being elided
            });
            time("omp_barrier", 2000, [] {
#pragma omp parallel
                {
#pragma omp barrier
                }
            });
            Activation a;
            time("set_f32_hidden", 500, [&] { a.set(h.data(), m.hidden, false); });
            time("set_q8_hidden", 500, [&] { a.set(h.data(), m.hidden, true); });
            time("norm_hidden", 500, [&] { (void)runner.norm(h, {}); });
            time("norm_4hidden", 500, [&] { (void)runner.norm(r4, {}); });
            if (lk >= 0) {
                auto &Y = runner.L[lk];
                auto &E = Y.moe;
                Activation a4;
                a4.set(r4.data(), 4 * m.hidden, needs_q8(Y.hc_attn));
                time("hc_gemv", 500, [&] { gemv(Y.hc_attn, a4, y.data()); });
                Vec c;
                time("mhc_read", 500, [&] { c = runner.read(Y.hc_attn, Y.hc_attn_base, Y.hc_attn_scale, r4).second; });
                time("mhc_write", 500, [&] { runner.write(r4, c, h); });
                Activation af;
                af.set(h.data(), m.hidden, false);
                time("router_gemv", 500, [&] { gemv(E.router, af, y.data()); });
                Vec logits(E.router.rows);
                gemv(E.router, af, logits.data());
                std::vector<int> ids(m.top_k);
                Vec w(m.top_k);
                time("router_topk", 500, [&] { ref::router(logits.data(), E.bias.data(), m.experts, m.top_k, m.expert_scale, ids.data(), w.data()); });
                if (E.sgate.rows) {
                    Activation ax;
                    ax.set(h.data(), m.hidden, needs_q8(E.sgate));
                    Vec sg(E.sgate.rows), su(E.sup.rows);
                    time("router_shared_multi", 500, [&] { gemv_multi({{&E.router, &af, logits.data()}, {&E.sgate, &ax, sg.data()}, {&E.sup, &ax, su.data()}}); });
                }
                time("moe_layer", 100, [&] { (void)runner.moe(E, h, m.layers[lk]); });
                time("kda_layer", 100, [&] { (void)runner.kda(Y.kd, h); });
            }
            if (lm >= 0) time("mla_layer", 100, [&] { (void)runner.mla(runner.L[lm].ml, h); });
            if (runner.L[0].dense) time("dense_layer", 100, [&] { (void)runner.ffn_dense(runner.L[0].ffn, h, m.layers[0].swiglu_limit); });
            for (auto &[what, sec] : items) {
                const double worst = comm.max_over_ranks(sec);
                if (comm.rank == 0) std::cerr << "RUNNER_PHASE " << what << " us " << sec * 1e6 << " max_over_ranks " << worst * 1e6 << '\n';
            }
            runner.route_ids.clear();
            runner.timers = kept;
        }
        runner.timers["comm_inner"] = comm.seconds;
        if (std::getenv("STRATA_RUNNER_TIMERS_ALL")) {   // every rank's own timers (rank skew), one stderr line each
            char host[256] = {0};
            gethostname(host, sizeof host - 1);
            std::ostringstream o;
            o << "RUNNER_RANK_TIMERS rank=" << comm.rank << " host=" << host;
            for (const auto &[k, v] : runner.timers) o << ' ' << k << '=' << v / ids.size() * 1e3;
            std::cerr << o.str() << '\n';
        }
        std::vector<double> lo, hi;
        comm.spread(runner.timers, lo, hi);
        if (comm.rank == 0) {
            std::ostringstream sp;
            sp << "RUNNER_SPREAD ms_per_token min/max over ranks";
            size_t k = 0;
            for (const auto &kv : runner.timers) {
                sp << ' ' << kv.first << '=' << lo[k] / ids.size() * 1e3 << '/' << hi[k] / ids.size() * 1e3;
                ++k;
            }
            std::cout << sp.str() << '\n';
            std::ostringstream s;
            s << "RUNNER_SUMMARY numerics=" << numerics << " ranks=" << comm.world << " layers=" << layers << " tokens=" << ids.size()
              << " ms_per_token=" << decode_s / ids.size() * 1e3;
            if (compared) s << " mean_kl=" << kl_sum / compared << " top1=" << top_hits << "/" << compared;
            if (scored) s << " ppl=" << std::exp(nll_sum / scored) << " next_top1=" << next_hits << "/" << scored;
            std::cout << s.str() << '\n';
            std::ostringstream tm;
            tm << "RUNNER_TIMERS ms_per_token";
            for (const auto &[k, v] : runner.timers) tm << ' ' << k << '=' << v / ids.size() * 1e3;
            tm << " comm_inner=" << comm.seconds / ids.size() * 1e3;
            std::cout << tm.str() << '\n';
        }
    } catch (const std::exception &e) {
        std::cerr << "RUNNER_ERROR rank " << comm.rank << ": " << e.what() << '\n';
        code = 1;
    }
    comm.finish();
    return code;
}
