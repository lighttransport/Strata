// include/strata/cpu_runner/infra.hpp - the CPU runners' shared infrastructure, from the GLM-5.3-Flash A64FX runner
// (src/cpu_runner/glm_cpu_runner.cpp, e4dc39fc): the persistent OpenMP team (team_run / with_team / team_barrier), the
// CMG-grouped flag barrier, and Comm (MPI + uTofu all-reduce, comm thread, staged banks, argmax, gathers, broadcasts).
// Header-only; one translation unit per runner. Build flags as the GLM runner: -DSTRATA_CPU_RUNNER_MPI
// -DSTRATA_CPU_RUNNER_UTOFU, link comm_utofu.c and -ltofucom.
#pragma once
#include <omp.h>
#include "strata/host/glm_env.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
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
extern "C" int cpu_utofu_xc_init(int shard_cap);   // reduce-scatter / all-gather region (collective)
extern "C" int cpu_utofu_xc_cap(void);
extern "C" int cpu_utofu_reduce_scatter(const float *x, const int *off, float *mine);
extern "C" int cpu_utofu_allgather(float *x, const int *off, int canon);
extern "C" int cpu_utofu_alltoall(const float *send, const int *soff, float *recv, const int *roff);
#else   // no staged banks without uTofu (Comm::banks stays 0, so these are never called)
inline void cpu_utofu_shards(int, int *) {}
inline float *cpu_utofu_bank_send(int, int) { return nullptr; }
inline const float *cpu_utofu_bank_result(int, int) { return nullptr; }
inline void cpu_utofu_bank_allreduce(int, int) {}
#endif


namespace strata::cpu_runner {

inline double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// FPCR.FZ on the calling thread: single/double subnormals become zero. The KDA state decays by up to e^-5 per token and
// its smallest entries pass through the subnormal range, which A64FX processes far slower than normal numbers. Half
// precision (FZ16, the E4M3 decode's subnormals) is not touched; the decode is re-checked after setting it.
inline void flush_subnormals() {
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
// barrier profile (off unless a runner sets g_barprof): per team thread its time in team_barrier and the count; thread
// 0's time inside dispatched regions and their count (the rest of its time is serial work while the team waits)
struct alignas(256) BarProf { double wait = 0, n = 0; };
inline bool g_barprof = false;
inline BarProf g_barp[64];
inline double g_region_s = 0, g_regions = 0;
struct TeamDispatch {
    alignas(256) std::atomic<uint64_t> gen{0};
    alignas(256) void (*fn)(void *) = nullptr;
    void *ctx = nullptr;
    bool stop = false;
    bool open = false;   // with_team's region is open
    bool busy = false;   // thread 0 is inside a dispatched f (a nested team_run then forks as OpenMP would)
};
inline TeamDispatch g_team;
template <class F> void team_run(F &&f) {
    if (g_team.open && !g_team.busy && omp_get_level() == 1 && omp_get_thread_num() == 0) {
        const double r0 = g_barprof ? now() : 0;
        using Fn = std::remove_reference_t<F>;
        g_team.fn = [](void *c) { (*static_cast<Fn *>(c))(); };
        g_team.ctx = const_cast<void *>(static_cast<const void *>(&f));
        g_team.busy = true;
        g_team.gen.fetch_add(1, std::memory_order_release);
        f();
        team_barrier();
        g_team.busy = false;
        if (g_barprof) g_region_s += now() - r0, g_regions += 1;
        return;
    }
#pragma omp parallel
    f();
}
inline bool persistent_team() {
    static const bool on = !(strata_env_equals("STRATA_RUNNER_PERSISTENT", "0"));
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
    // ---- reduce-scatter and all-gather with caller-given shard bounds off[0..world] (shard r on rank r). Both give the
    // all-reduce's bits: reduce_scatter's shard is summed in rank order (as rsag); allgather's values are each rank's
    // own + 0.0f (the all-reduce of a vector zero outside each rank's shard). Without the uTofu region: the all-reduce.
    bool xc = false;
    bool xc_init(int shard_cap) {   // collective
#ifdef STRATA_CPU_RUNNER_UTOFU
        if (utofu && world > 1) {
            const int rc = cpu_utofu_xc_init(shard_cap);
            int ok = rc == 0, all = 0;
            MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            xc = all == 1;
            if (rank == 0) std::cerr << "RUNNER comm xc (reduce-scatter / all-gather) shard cap " << shard_cap << " rc " << rc << '\n';
        }
#else
        (void) shard_cap;
#endif
        return xc;
    }
    int xc_cap() const {
#ifdef STRATA_CPU_RUNNER_UTOFU
        return xc ? cpu_utofu_xc_cap() : 0;
#else
        return 0;
#endif
    }
    void reduce_scatter(const float *x, const int *off, float *mine) {
        const double t = now();
#ifdef STRATA_CPU_RUNNER_UTOFU
        if (xc) {
            if (ar_thread.joinable() && !on_ar_thread()) ar_wait(ar_posted.load());
            if (cpu_utofu_reduce_scatter(x, off, mine)) { std::cerr << "RUNNER uTofu reduce-scatter failed\n"; MPI_Abort(MPI_COMM_WORLD, 3); }
            seconds += now() - t;
            return;
        }
#endif
        std::vector<float> tmp(x, x + off[world]);
        allreduce(tmp.data(), off[world]);
        std::copy(tmp.begin() + off[rank], tmp.begin() + off[rank + 1], mine);
        (void) t;
    }
    // canon false: the bits as sent (no + 0.0f; integers carried in floats, values never summed elsewhere). The fallback
    // all-reduce gives those bits only for canon (and finite values)
    void allgather_shards(float *x, const int *off, bool canon = true) {
        const double t = now();
#ifdef STRATA_CPU_RUNNER_UTOFU
        if (xc) {
            if (ar_thread.joinable() && !on_ar_thread()) ar_wait(ar_posted.load());
            if (cpu_utofu_allgather(x, off, canon ? 1 : 0)) { std::cerr << "RUNNER uTofu all-gather failed\n"; MPI_Abort(MPI_COMM_WORLD, 3); }
            seconds += now() - t;
            return;
        }
#endif
        if (!canon && world > 1) throw std::runtime_error("allgather_shards: the exact gather needs the uTofu region");
        std::fill(x, x + off[rank], 0.f);
        std::fill(x + off[rank + 1], x + off[world], 0.f);
        allreduce(x, off[world]);
        (void) t;
    }
    // segment d of send ([soff[d], soff[d+1])) to rank d; recv's segment s from rank s; the bits as sent
    void alltoall(const float *send, const int *soff, float *recv, const int *roff) {
        const double t = now();
#ifdef STRATA_CPU_RUNNER_UTOFU
        if (xc) {
            if (ar_thread.joinable() && !on_ar_thread()) ar_wait(ar_posted.load());
            if (cpu_utofu_alltoall(send, soff, recv, roff)) { std::cerr << "RUNNER uTofu all-to-all failed\n"; MPI_Abort(MPI_COMM_WORLD, 3); }
            seconds += now() - t;
            return;
        }
#endif
        if (world > 1) throw std::runtime_error("alltoall: needs the uTofu region");
        std::copy(send + soff[0], send + soff[1], recv + roff[0]);
        (void) t;
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
    struct ArReq { float *x; int n; int bank = -1; std::function<void()> *fn = nullptr; };   // bank >= 0: a staged all-reduce in that bank; fn: a collective to run
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
                    if (r.fn) (*r.fn)();
                    else if (r.bank >= 0) cpu_utofu_bank_allreduce(r.bank, r.n);
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
    bool on_ar_thread() const { return ar_thread.joinable() && std::this_thread::get_id() == ar_thread.get_id(); }
    // a collective run by the comm thread in posting order (fn must stay alive until ar_wait(ticket))
    long post_fn(std::function<void()> *fn) {
        if (!ar_async_ok() || world == 1) { (*fn)(); return 0; }
        if (!ar_thread.joinable()) ar_start();
        const long id = ar_posted.load(std::memory_order_relaxed);
        while (id - ar_done.load(std::memory_order_acquire) >= long(ar_ring.size())) {}
        ar_ring[id % ar_ring.size()] = {nullptr, 0, -1, fn};
        ar_posted.store(id + 1, std::memory_order_release);
        return id + 1;
    }
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
        if (ar_thread.joinable() && !on_ar_thread()) { ar_wait(allreduce_async(x, n)); return; }   // after the pipelined ones, in order
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
        if (utofu && world > 1 && !strata_env_is_set("STRATA_RUNNER_ARGMAX_MPI")) {
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
        static const bool on = !(strata_env_equals("STRATA_RUNNER_COLL", "mpi"));
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
inline int cmg_threads() {
    static const int v = std::getenv("STRATA_RUNNER_CMG_THREADS") ? std::max(6, std::atoi(std::getenv("STRATA_RUNNER_CMG_THREADS"))) : 12;
    return v;
}
// ------------------------------------------------------------------------------------------------- packed matrices
struct FlagBarrier {
    struct alignas(256) Flag { std::atomic<int> v{0}; };
    Flag arrive[64], garrive[16], grelease[16], release;
    Flag carrive[64], crelease[16];   // group_wait's own flags (its own parity: local[t][1])
    int n = 0, gsize = 12, groups = 1;
    // STRATA_RUNNER_BAR_FLAT=1: every thread waits on the global release (no per-group release hop); with
    // STRATA_RUNNER_BAR_GROUP=N (3..12) arrival groups of N (bar_bench, 47 threads: 1.47-1.89 us -> ~1.05 at N = 4)
    bool flat = false;
    alignas(256) int local[64][64] = {};
    void init(int team, int per_group = 12) {
        flat = strata_env_is_one("STRATA_RUNNER_BAR_FLAT");
        if (flat && std::getenv("STRATA_RUNNER_BAR_GROUP")) per_group = std::max(3, std::min(12, std::atoi(std::getenv("STRATA_RUNNER_BAR_GROUP"))));
        n = team, gsize = per_group, groups = (team + per_group - 1) / per_group;
        if (groups > 16) throw std::runtime_error("FlagBarrier: more than 16 groups");
    }
    void wait(int t) {
        int &ls = local[t][0];
        ls ^= 1;
        const int g = t / gsize, g0 = g * gsize, g1 = std::min(n, g0 + gsize);
        if (t != g0) {
            arrive[t].v.store(ls, std::memory_order_release);
            if (flat) while (release.v.load(std::memory_order_acquire) != ls) {}
            else while (grelease[g].v.load(std::memory_order_acquire) != ls) {}
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
        if (!flat) grelease[g].v.store(ls, std::memory_order_release);
    }
    // a barrier of thread t's group only (threads t / gsize), independent of wait()'s flags and parity
    void group_wait(int t) {
        int &ls = local[t][1];
        ls ^= 1;
        const int g = t / gsize, g0 = g * gsize, g1 = std::min(n, g0 + gsize);
        if (t != g0) {
            carrive[t].v.store(ls, std::memory_order_release);
            while (crelease[g].v.load(std::memory_order_acquire) != ls) {}
            return;
        }
        for (int u = g0 + 1; u < g1; ++u)
            while (carrive[u].v.load(std::memory_order_acquire) != ls) {}
        crelease[g].v.store(ls, std::memory_order_release);
    }
};

inline FlagBarrier g_bar;   // the team's barrier in the hot regions (main: g_bar.init(team); STRATA_RUNNER_FLAG_BARRIER=0 keeps OpenMP's)
// a barrier of the current team: the flag barrier when the team is the one it was set up for, OpenMP's otherwise
inline void team_barrier() {
    if (omp_get_num_threads() == g_bar.n) {
        const int t = omp_get_thread_num();
        if (!g_barprof || t >= 64) g_bar.wait(t);
        else {
            const double t0 = now();
            g_bar.wait(t);
            g_barp[t].wait += now() - t0, g_barp[t].n += 1;
        }
    } else {
#pragma omp barrier
    }
}

// a barrier of the calling thread's CMG group (the flag barrier's groups of cmg_threads()); the team's barrier when the
// flag barrier is not the team's or its groups are not the CMGs
inline void cmg_barrier() {
    if (omp_get_num_threads() == g_bar.n && g_bar.gsize == cmg_threads()) g_bar.group_wait(omp_get_thread_num());
    else team_barrier();
}

// gemv_multi's work for one team: plan outside a region (output and K-split partial buffers), then every thread of

}  // namespace strata::cpu_runner
