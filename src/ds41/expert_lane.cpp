// src/ds41/expert_lane.cpp - DeepSeek-V4.1's CPU expert lane (strata/ds41/expert_lane.hpp).
#include "strata/ds41/expert_lane.hpp"
#include "strata/host/glm_env.h"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <sys/mman.h>
#include <unordered_map>
#include <fstream>
#include <map>
#include <pthread.h>
#include <sched.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace strata::ds41 {
namespace cpu = strata::kernels::cpu;

namespace {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// Keep four vectors live across routes. No FMA: match collect()'s multiply then add.
__attribute__((target("sse2,no-avx,no-fma")))
void collect_vectors(float* y, const float* const* src, const float* weights, int n, int hidden) {
    if (!n) { std::fill(y, y + hidden, 0.f); return; }
    int j = 0;
    for (; j + 16 <= hidden; j += 16) {
        const __m128 w0 = _mm_set1_ps(weights[0]);
        __m128 a = _mm_mul_ps(_mm_loadu_ps(src[0] + j), w0);
        __m128 b = _mm_mul_ps(_mm_loadu_ps(src[0] + j + 4), w0);
        __m128 c = _mm_mul_ps(_mm_loadu_ps(src[0] + j + 8), w0);
        __m128 d = _mm_mul_ps(_mm_loadu_ps(src[0] + j + 12), w0);
        for (int k = 1; k < n; ++k) {
            const __m128 w = _mm_set1_ps(weights[k]);
            a = _mm_add_ps(a, _mm_mul_ps(_mm_loadu_ps(src[k] + j), w));
            b = _mm_add_ps(b, _mm_mul_ps(_mm_loadu_ps(src[k] + j + 4), w));
            c = _mm_add_ps(c, _mm_mul_ps(_mm_loadu_ps(src[k] + j + 8), w));
            d = _mm_add_ps(d, _mm_mul_ps(_mm_loadu_ps(src[k] + j + 12), w));
        }
        _mm_storeu_ps(y + j, a); _mm_storeu_ps(y + j + 4, b);
        _mm_storeu_ps(y + j + 8, c); _mm_storeu_ps(y + j + 12, d);
    }
    for (; j < hidden; ++j) {
        float v = src[0][j] * weights[0];
        for (int k = 1; k < n; ++k) v += src[k][j] * weights[k];
        y[j] = v;
    }
}
#endif
inline void pause() {
#if defined(__x86_64__)
    _mm_pause();
#endif
}
int read_int(const std::string& path, int fallback) {
    std::ifstream f(path);
    int v = fallback;
    return f >> v ? v : fallback;
}
// the package (socket) of a logical CPU
int cpu_package(int cpu) { return read_int("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/physical_package_id", 0); }
// a logical CPU's hyperthread siblings (its thread_siblings_list, "a,b" or "a-b")
std::vector<int> siblings(int cpu) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
    std::string text;
    std::vector<int> out;
    if (!(f >> text)) return out;
    size_t i = 0;
    while (i < text.size()) {
        size_t j = text.find(',', i);
        if (j == std::string::npos) j = text.size();
        const std::string part = text.substr(i, j - i);
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str()), b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) out.push_back(c);
        i = j + 1;
    }
    return out;
}
// the NUMA node of a logical CPU (its cpuN/nodeK link; 0 when there is none)
int cpu_node(int cpu) {
    for (int n = 0; n < 64; ++n)
        if (access(("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node" + std::to_string(n)).c_str(), F_OK) == 0) return n;
    return 0;
}
// the package of the memory node holding the page at p (-1: unknown)
int page_package(const void* p) {
    int node = -1;
    // get_mempolicy(&node, nullptr, 0, p, MPOL_F_NODE | MPOL_F_ADDR)
    if (syscall(SYS_get_mempolicy, &node, nullptr, 0UL, p, 3UL) != 0 || node < 0) return -1;
    static std::map<int, int> of_node;
    auto it = of_node.find(node);
    if (it != of_node.end()) return it->second;
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string text;
    const int pkg = f >> text ? cpu_package(std::atoi(text.c_str())) : -1;
    of_node[node] = pkg;
    return pkg;
}
}  // namespace

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif

// STRATA_DS41_MISS_READ: the reader pool. A job is one piece (<= 1 MiB) of a cold expert's rows; its group's ready
// counter falls to 0 when every piece is in the page cache.
struct ExpertLane::MissReader {
    struct Job { const uint8_t* p; size_t len; std::atomic<int>* ready; };
    uint64_t gap = 1376, calls = 0, base = 0;   // base: calls before the last reset
    std::unordered_map<const void*, uint64_t> last;            // gate rows -> last call that used the expert
    std::vector<std::unique_ptr<std::atomic<int>>> flags;      // per call, reused (all 0 when a call ends)
    size_t used = 0;
    std::mutex m;
    std::condition_variable cv;
    std::deque<Job> q;
    bool stop = false;
    std::vector<std::thread> pool;
    std::atomic<uint64_t> pieces{0}, bytes{0}, read_ns{0}, wait_ns{0};
    uint64_t cold = 0;
    explicit MissReader(int n) {
        if (const char* v = std::getenv("STRATA_DS41_MISS_GAP")) gap = uint64_t(std::max(1, std::atoi(v)));
        for (int i = 0; i < n; ++i) pool.emplace_back([this] { loop(); });
    }
    ~MissReader() {
        { std::lock_guard<std::mutex> g(m); stop = true; }
        cv.notify_all();
        for (auto& t : pool) t.join();
    }
    std::atomic<int>* flag() {
        if (used == flags.size()) flags.push_back(std::make_unique<std::atomic<int>>(0));
        return flags[used++].get();
    }
    void push(const uint8_t* p, size_t len, std::atomic<int>* ready, std::vector<Job>& out) {
        const size_t piece = size_t(1) << 20;
        for (size_t o = 0; o < len; o += piece) {
            out.push_back({p + o, std::min(piece, len - o), ready});
            ready->fetch_add(1, std::memory_order_relaxed);
        }
    }
    void submit(std::vector<Job>& jobs) {
        if (jobs.empty()) return;
        { std::lock_guard<std::mutex> g(m); for (auto& j : jobs) q.push_back(j); }
        cv.notify_all();
    }
    void loop() {
        static const size_t page = size_t(sysconf(_SC_PAGESIZE));
        std::vector<unsigned char> vec;
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> g(m);
                cv.wait(g, [&] { return stop || !q.empty(); });
                if (stop && q.empty()) return;
                j = q.front();
                q.pop_front();
            }
            const auto t0 = std::chrono::steady_clock::now();
            const uintptr_t a = uintptr_t(j.p) & ~uintptr_t(page - 1), b = uintptr_t(j.p) + j.len;
            const size_t len = b - a, np = (len + page - 1) / page;
            vec.resize(np);
            bool missing = mincore(reinterpret_cast<void*>(a), len, vec.data()) != 0;   // unknown: read it
            for (size_t i = 0; i < np && !missing; ++i) missing = !(vec[i] & 1);
            if (missing) {
                madvise(reinterpret_cast<void*>(a), len, MADV_POPULATE_READ);
                pieces.fetch_add(1, std::memory_order_relaxed);
                bytes.fetch_add(len, std::memory_order_relaxed);
            }
            read_ns.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
            j.ready->fetch_sub(1, std::memory_order_acq_rel);
        }
    }
};

ExpertLane::MissStats ExpertLane::miss_stats() const {
    MissStats s;
    if (!miss_) return s;
    s.calls = miss_->calls - miss_->base;
    s.cold = miss_->cold;
    s.read_pieces = miss_->pieces.load();
    s.read_bytes = miss_->bytes.load();
    s.read_s = double(miss_->read_ns.load()) * 1e-9;
    s.wait_s = double(miss_->wait_ns.load()) * 1e-9;
    return s;
}

void ExpertLane::miss_reset() {
    if (!miss_) return;
    miss_->base = miss_->calls;   // (calls itself keeps counting: the last-use marks are call numbers)
    miss_->cold = 0;
    miss_->pieces = miss_->bytes = miss_->read_ns = miss_->wait_ns = 0;
}

size_t ExpertLane::page_out(const LaneLayer& layer, int experts, int hidden, int ff, double fraction, unsigned seed) {
    static const size_t page = size_t(sysconf(_SC_PAGESIZE));
    const size_t gate_bytes = layer.gate_stride ? layer.gate_stride : layer.f.gu_row * size_t(ff);
    const size_t down_bytes = layer.down_stride ? layer.down_stride : layer.f.d_row * size_t(hidden);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    size_t asked = 0;
    auto out = [&](const uint8_t* p, size_t len) {   // whole pages inside the range only
        const uintptr_t a = (uintptr_t(p) + page - 1) & ~uintptr_t(page - 1), b = (uintptr_t(p) + len) & ~uintptr_t(page - 1);
        if (b > a && madvise(reinterpret_cast<void*>(a), b - a, MADV_PAGEOUT) == 0) asked += b - a;
    };
    for (int e = 0; e < experts; ++e) {
        if (u(rng) >= fraction) continue;
        out(layer.gate + size_t(e) * gate_bytes, gate_bytes);
        out(layer.up + size_t(e) * gate_bytes, gate_bytes);
        out(layer.down + size_t(e) * down_bytes, down_bytes);
    }
    return asked;
}

// MISS_READ: mark this call's groups warm or cold, queue the cold ones' pieces, run the cold groups last
void ExpertLane::miss_mark(const LaneLayer& layer) {
    MissReader& r = *miss_;
    ++r.calls;
    r.used = 0;
    const size_t gate_bytes = layer.gate_stride ? layer.gate_stride : layer.f.gu_row * size_t(ff_);
    const size_t down_bytes = layer.down_stride ? layer.down_stride : layer.f.d_row * size_t(hidden_);
    std::vector<MissReader::Job> jobs;
    std::unordered_map<int, std::atomic<int>*> this_call;   // VERIFY_EXACT: one expert in several groups
    for (Group& g : groups_) {
        g.ready = nullptr;
        auto seen = this_call.find(g.expert);
        if (seen != this_call.end()) { g.ready = seen->second; continue; }
        const uint8_t* gate = layer.gate + size_t(g.expert) * gate_bytes;
        auto it = r.last.find(gate);
        const bool cold = it == r.last.end() || r.calls - it->second > r.gap;
        r.last[gate] = r.calls;
        if (!cold) { this_call[g.expert] = nullptr; continue; }
        ++r.cold;
        g.ready = r.flag();
        r.push(gate, gate_bytes, g.ready, jobs);
        r.push(layer.up + size_t(g.expert) * gate_bytes, gate_bytes, g.ready, jobs);
        r.push(layer.down + size_t(g.expert) * down_bytes, down_bytes, g.ready, jobs);
        this_call[g.expert] = g.ready;
    }
    r.submit(jobs);
    std::stable_partition(groups_.begin(), groups_.end(), [](const Group& g) { return g.ready == nullptr; });
}

void ExpertLane::miss_wait(const Group& g) {
    if (!g.ready || g.ready->load(std::memory_order_acquire) == 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    while (g.ready->load(std::memory_order_acquire) > 0) pause();
    miss_->wait_ns.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
}

size_t ExpertLane::place_bands(const LaneLayer& layer, int experts, int hidden, int ff, int nodes) {
    static const size_t page = size_t(sysconf(_SC_PAGESIZE));
    const size_t gate_bytes = layer.gate_stride ? layer.gate_stride : layer.f.gu_row * size_t(ff);
    const size_t down_bytes = layer.down_stride ? layer.down_stride : layer.f.d_row * size_t(hidden);
    std::vector<void*> pages;
    std::vector<int> to, status;
    size_t asked = 0;
    auto flush = [&] {
        if (pages.empty()) return;
        status.assign(pages.size(), 0);
        // move_pages(0, count, pages, nodes, status, MPOL_MF_MOVE)
        syscall(SYS_move_pages, 0, (unsigned long)pages.size(), pages.data(), to.data(), status.data(), 2);
        pages.clear();
        to.clear();
    };
    // the pages of rows [0, rows) of row_bytes each at base: a page goes with the band holding its first byte
    auto matrix = [&](const uint8_t* base, size_t row_bytes, int rows, int e) {
        const uintptr_t lo = uintptr_t(base), hi = lo + row_bytes * size_t(rows);
        for (uintptr_t a = (lo + page - 1) & ~uintptr_t(page - 1); a + page <= hi; a += page) {
            const int band = int((a - lo) / row_bytes / 128);
            pages.push_back(reinterpret_cast<void*>(a));
            to.push_back((band + e) % nodes);
            asked += page;
            if (pages.size() >= 65536) flush();
        }
    };
    for (int e = 0; e < experts; ++e) {
        matrix(layer.gate + size_t(e) * gate_bytes, layer.f.gu_row, ff, e);
        matrix(layer.up + size_t(e) * gate_bytes, layer.f.gu_row, ff, e);
        matrix(layer.down + size_t(e) * down_bytes, layer.f.d_row, hidden, e);
    }
    flush();
    return asked;
}

// LANE_BANDS: each node's threads (threads 1.. in begin()) numbered within their node
void ExpertLane::band_teams() {
    const int first = async_ ? 1 : 0;
    band_size_.assign(size_t(nodes_), 0);
    band_index_.assign(size_t(threads_), 0);
    for (int t = first; t < threads_; ++t) band_index_[size_t(t)] = band_size_[size_t(node_of_[size_t(t)])]++;
}

ExpertLane::ExpertLane(int hidden, int ff, int threads, bool pin_threads)
    : hidden_(hidden), ff_(ff), threads_(std::max(1, threads)) {
    // STRATA_DS41_LANE_SLEEP (1): idle workers block on the start counter; 0: they yield (as before)
    sleep_ = !(std::getenv("STRATA_DS41_LANE_SLEEP") && std::string(std::getenv("STRATA_DS41_LANE_SLEEP")) == "0");
    cpu_of_.assign(size_t(threads_), -1);
    if (pin_threads) {
        // one thread per physical core; threads beyond them on the cores' hyperthread siblings (not the same CPU twice)
        const auto cores = cpu::physical_cores(false);
        std::vector<int> cpus(cores.begin(), cores.end());
        for (int c : cores)
            for (int sib : siblings(c))
                if (std::find(cpus.begin(), cpus.end(), sib) == cpus.end()) cpus.push_back(sib);
        // STRATA_DS41_LANE_CPU_OFFSET=k: skip the first k cores (a second lane in the process, STRATA_DS41_DUAL)
        const size_t off = std::getenv("STRATA_DS41_LANE_CPU_OFFSET") ? size_t(std::max(0, std::atoi(std::getenv("STRATA_DS41_LANE_CPU_OFFSET")))) : 0;
        for (int t = 0; t < threads_ && !cpus.empty(); ++t) cpu_of_[size_t(t)] = cpus[(size_t(t) + off) % cpus.size()];
    }
    // STRATA_DS41_LANE_NUMA=1 (pinned threads): each socket's threads take the experts whose rows sit in its memory
    numa_ = pin_threads && std::getenv("STRATA_DS41_LANE_NUMA") && std::string(std::getenv("STRATA_DS41_LANE_NUMA")) == "1";
    pkg_of_.assign(size_t(threads_), 0);
    for (int t = 0; t < threads_; ++t) pkg_of_[size_t(t)] = cpu_of_[size_t(t)] >= 0 ? cpu_package(cpu_of_[size_t(t)]) : 0;
    packages_ = 1 + *std::max_element(pkg_of_.begin(), pkg_of_.end());
    if (packages_ < 2) numa_ = false;
    // STRATA_DS41_LANE_BANDS=1 (pinned threads, 2+ NUMA nodes): node teams over the bands placed in their memory
    node_of_.assign(size_t(threads_), 0);
    for (int t = 0; t < threads_; ++t) node_of_[size_t(t)] = cpu_of_[size_t(t)] >= 0 ? cpu_node(cpu_of_[size_t(t)]) : 0;
    nodes_ = 1 + *std::max_element(node_of_.begin(), node_of_.end());
    bands_ = pin_threads && nodes_ > 1 && std::getenv("STRATA_DS41_LANE_BANDS") && std::string(std::getenv("STRATA_DS41_LANE_BANDS")) == "1";
    if (bands_) {   // every node needs threads, with and without the caller (begin() leaves thread 0 out)
        for (int first = 0; first < 2 && bands_; ++first)
            for (int n = 0; n < nodes_ && bands_; ++n)
                bands_ = std::any_of(node_of_.begin() + first, node_of_.end(), [n](int x) { return x == n; });
        if (!bands_) std::fprintf(stderr, "ds41 expert lane: LANE_BANDS off (a NUMA node has no lane thread)\n");
    }
    if (bands_) numa_ = false;
    if (const char* v = std::getenv("STRATA_DS41_MISS_READ"); v && std::atoi(v) > 0) miss_ = std::make_unique<MissReader>(std::atoi(v));
    for (int i = 1; i < threads_; ++i) pool_.emplace_back([this, i] { pin(i); worker(i); });
    pin(0);
}

ExpertLane::~ExpertLane() {
    if (strata_env_is_one("STRATA_DS41_LANE_STATS") && stat_calls_ > 0) {
        for (int t = 0; t < threads_ && t < 64; ++t)
            std::fprintf(stderr, "DS41_LANE_STATS thread %2d per call ms work %.3f %.3f %.3f wait %.3f %.3f %.3f\n", t,
                         stat_work_[t][0] / stat_calls_ * 1e-6, stat_work_[t][1] / stat_calls_ * 1e-6, stat_work_[t][2] / stat_calls_ * 1e-6,
                         stat_wait_[t][0] / stat_calls_ * 1e-6, stat_wait_[t][1] / stat_calls_ * 1e-6, stat_wait_[t][2] / stat_calls_ * 1e-6);
    }
    if (strata_env_is_one("STRATA_DS41_LANE_STATS") && merge_calls_ > 0)
        std::fprintf(stderr, "DS41_LANE_STATS merge per call ms: check %.3f lane wait %.3f merge %.3f (calls %ld)\n", merge_check_ns_ / merge_calls_ * 1e-6,
                     merge_wait_ns_ / merge_calls_ * 1e-6, merge_work_ns_ / merge_calls_ * 1e-6, merge_calls_);
    stop_.store(true);
    start_.fetch_add(1, std::memory_order_acq_rel);   // wake sleepers; they see stop_
    start_.notify_all();
    for (auto& t : pool_) t.join();
}

void ExpertLane::pin(int thread) {
    if (cpu_of_[size_t(thread)] < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_of_[size_t(thread)], &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

void ExpertLane::barrier() {
    // Read the job's mode before publishing arrival. After the final arrival
    // the caller may already prepare a job with another participant count.
    const int participants = async_ ? threads_ - 1 : threads_;
    const unsigned s = sense_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == participants) {
        arrived_.store(0, std::memory_order_relaxed);
        sense_.store(s + 1, std::memory_order_release);
    } else {
        while (sense_.load(std::memory_order_acquire) == s) pause();
    }
}

void ExpertLane::worker(int thread) {
    // STRATA_DS41_LANE_SPIN (16384): pauses before an idle worker sleeps (spinning workers slow the thread that
    // enqueues the GPU's work when they share its cores)
    static const unsigned spin_max = [] { const char* v = std::getenv("STRATA_DS41_LANE_SPIN"); return v ? unsigned(std::max(1, std::atoi(v))) : (1u << 14); }();
    unsigned seen = 0;
    for (;;) {
        unsigned spins = 0;
        while (start_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            if (++spins < spin_max) pause();
            else if (sleep_) start_.wait(seen, std::memory_order_acquire);   // idle (GPU work, streamed prefill): sleep
            else std::this_thread::yield();
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = start_.load(std::memory_order_acquire);
        // The caller can begin an async job as soon as a sync job's last
        // barrier releases. Its workers must not credit that new job here.
        const bool job_async = async_;
        phase_work(thread);
        if (job_async) {
            const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            for (int64_t seen = end_ns_.load(std::memory_order_relaxed); seen < now && !end_ns_.compare_exchange_weak(seen, now);) {}
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
}

// The three phases of one layer for one thread; each ends at a barrier every thread reaches.
void ExpertLane::phase_work(int t) {
    if (job_fn_) {   // parallel_for(): this thread's part, then the barrier the caller waits at
        job_fn_(job_ctx_, t, threads_);
        barrier();
        return;
    }
    const bool job_async = async_;
    const int T = job_async ? threads_ - 1 : threads_;   // begin(): threads 1.. only
    if (job_async) --t;
    const cpu::NativeFmt& f = layer_->f;
    static const bool skip_unused = strata_env_is_one("STRATA_DS41_LANE_SKIP_UNUSED");
    const size_t act_bytes = f.act_bytes, h_bytes = f.h_bytes;
    const int abs_t = job_async ? t + 1 : t;
    if (bands_) {   // LANE_BANDS: this node's bands of every group, its team's share of them
        const int node = node_of_[size_t(abs_t)], tn = band_size_[size_t(node)], ti = band_index_[size_t(abs_t)];
        // the (group, band) items of this node, in order; this thread takes items [n * ti / tn, n * (ti + 1) / tn)
        auto items = [&](int rows, auto&& fn) {
            const int nb = (rows + 127) / 128;
            int n = 0;
            for (const Group& g : groups_)
                for (int b = 0; b < nb; ++b) n += (b + g.expert) % nodes_ == node;
            const int i0 = int(int64_t(n) * ti / tn), i1 = int(int64_t(n) * (ti + 1) / tn);
            int i = 0;
            for (const Group& g : groups_) {
                for (int b = 0; b < nb && i < i1; ++b) {
                    if ((b + g.expert) % nodes_ != node) continue;
                    if (i++ < i0) continue;
                    fn(g, b * 128, std::min(rows, (b + 1) * 128));
                }
                if (i >= i1) break;
            }
        };
        items(ff_, [&](const Group& g, int r0, int r1) {
            if (g.ready) miss_wait(g);
            const uint8_t* gate = g.gate;
            const uint8_t* up = g.up;
            for (size_t j0 = 0; j0 < g.entries.size(); j0 += cpu::MAXT) {
                const size_t ng = std::min<size_t>(cpu::MAXT, g.entries.size() - j0);
                const void* acts[cpu::MAXT];
                float* ffs[cpu::MAXT];
                for (size_t j = 0; j < ng; ++j) {
                    acts[j] = act_.data() + size_t(g.entries[j0 + j] / top_k_) * act_bytes;
                    ffs[j] = hidden_buf_.data() + size_t(g.entries[j0 + j]) * size_t(ff_);
                }
                cpu::native_gu_rows(f, gate, acts, int(ng), ffs, r0, r1, up);
            }
        });
        barrier();
        const int entries = nt_ * top_k_;
        for (int e = entries * t / T; e < entries * (t + 1) / T; ++e)
            if (!skip_unused || ids_[e] >= 0)
                cpu::native_quant_h(f, hidden_buf_.data() + size_t(e) * size_t(ff_), hq_.data() + size_t(e) * h_bytes);
        barrier();
        items(hidden_, [&](const Group& g, int r0, int r1) {
            const uint8_t* gate = g.gate;
            const uint8_t* down = g.down;
            for (size_t j0 = 0; j0 < g.entries.size(); j0 += cpu::MAXT) {
                const size_t ng = std::min<size_t>(cpu::MAXT, g.entries.size() - j0);
                const void* hqs[cpu::MAXT];
                float* outs[cpu::MAXT];
                for (size_t j = 0; j < ng; ++j) {
                    hqs[j] = hq_.data() + size_t(g.entries[j0 + j]) * h_bytes;
                    outs[j] = out_.data() + size_t(g.entries[j0 + j]) * size_t(hidden_);
                }
                cpu::native_down_rows(f, gate, hqs, int(ng), outs, r0, r1, down);
            }
        });
        barrier();
        return;
    }
    // with teams (LANE_NUMA): this thread's share of its team's groups only
    static const bool stats = strata_env_is_one("STRATA_DS41_LANE_STATS");
    auto now_ns = [] { return double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); };
    double t_prev = stats ? now_ns() : 0.0;
    auto sync = [&](int ph) {   // barrier (per thread: work since the last one, then the wait)
        if (!stats) { barrier(); return; }
        const double a = now_ns();
        barrier();
        const double b = now_ns();
        stat_work_[abs_t & 63][ph] += a - t_prev;
        stat_wait_[abs_t & 63][ph] += b - a;
        t_prev = b;
    };
    const int team = numa_ ? pkg_of_[size_t(abs_t)] : -1;
    const int tn = numa_ ? team_size_[size_t(team)] : T, ti = numa_ ? team_index_[size_t(abs_t)] : t;
    // 1. gate/up: this thread's rows of every selected expert (all of its tokens at once)
    {
        const int r0 = int(int64_t(ff_) * ti / tn), r1 = int(int64_t(ff_) * (ti + 1) / tn);
        for (const Group& g : groups_) {
            if (r0 == r1) break;
            if (numa_ && g.team != team) continue;
            if (g.ready) miss_wait(g);   // MISS_READ: a cold expert's pieces are in
            const uint8_t* gate = g.gate;
            const uint8_t* up = g.up;
            for (size_t j0 = 0; j0 < g.entries.size(); j0 += cpu::MAXT) {   // the row kernels take up to MAXT tokens
                const size_t ng = std::min<size_t>(cpu::MAXT, g.entries.size() - j0);
                const void* acts[cpu::MAXT];
                float* ffs[cpu::MAXT];
                for (size_t j = 0; j < ng; ++j) {
                    acts[j] = act_.data() + size_t(g.entries[j0 + j] / top_k_) * act_bytes;
                    ffs[j] = hidden_buf_.data() + size_t(g.entries[j0 + j]) * size_t(ff_);
                }
                cpu::native_gu_rows(f, gate, acts, int(ng), ffs, r0, r1, up);
            }
        }
    }
    sync(0);
    // 2. the hidden of every entry, quantized for the down rows
    {
        const int entries = nt_ * top_k_;
        for (int e = entries * t / T; e < entries * (t + 1) / T; ++e)
            if (!skip_unused || ids_[e] >= 0)
                cpu::native_quant_h(f, hidden_buf_.data() + size_t(e) * size_t(ff_), hq_.data() + size_t(e) * h_bytes);
    }
    sync(1);
    // 3. down: this thread's hidden rows of every selected expert
    {
        const int r0 = int(int64_t(hidden_) * ti / tn), r1 = int(int64_t(hidden_) * (ti + 1) / tn);
        for (const Group& g : groups_) {
            if (r0 == r1) break;
            if (numa_ && g.team != team) continue;
            const uint8_t* gate = g.gate;
            const uint8_t* down = g.down;
            for (size_t j0 = 0; j0 < g.entries.size(); j0 += cpu::MAXT) {
                const size_t ng = std::min<size_t>(cpu::MAXT, g.entries.size() - j0);
                const void* hqs[cpu::MAXT];
                float* outs[cpu::MAXT];
                for (size_t j = 0; j < ng; ++j) {
                    hqs[j] = hq_.data() + size_t(g.entries[j0 + j]) * h_bytes;
                    outs[j] = out_.data() + size_t(g.entries[j0 + j]) * size_t(hidden_);
                }
                cpu::native_down_rows(f, gate, hqs, int(ng), outs, r0, r1, down);
            }
        }
    }
    sync(2);
    if (stats && abs_t == (job_async ? 1 : 0)) ++stat_calls_;
}

void ExpertLane::prepare(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, int top_k) {
    if (layer.f.n_embd != hidden_ || layer.f.n_ff != ff_) throw std::invalid_argument("ds41 expert lane: format geometry");
    layer_ = &layer;
    ids_ = ids;  // valid through run() or begin()/finish(), including worker quantization
    nt_ = nt;
    top_k_ = top_k;
    // group the routes by expert (the order of the groups does not change any value)
    // Exact verification normally uses one group per route so each token
    // rounds as when alone. The opt-in grouping probe measures weight reuse.
    static const bool split = strata_env_is_one("STRATA_DS41_VERIFY_EXACT") &&
                              !strata_env_is_one("STRATA_DS41_LANE_GROUP_EXACT");
    groups_.clear();
    if (split) {
        for (int e = 0; e < nt * top_k; ++e)
            if (ids[e] >= 0) groups_.push_back({ids[e], {e}});
    } else {
        std::map<int, std::vector<int>> by_expert;
        for (int e = 0; e < nt * top_k; ++e)
            if (ids[e] >= 0) by_expert[ids[e]].push_back(e);   // negative: handled elsewhere (the GPU tier)
        for (auto& [expert, entries] : by_expert) groups_.push_back({expert, std::move(entries)});
    }
    const size_t gu_stride = layer.gate_stride ? layer.gate_stride : layer.f.gu_row * size_t(ff_);
    const size_t d_stride = layer.down_stride ? layer.down_stride : layer.f.d_row * size_t(hidden_);
    for (Group& g : groups_) {
        if (layer.expert_spans) {
            if (size_t(g.expert) >= layer.expert_span_count)
                throw std::out_of_range("ds41 expert lane: cached expert index");
            const auto& span = layer.expert_spans[g.expert];
            if (!span[0] || !span[1] || !span[2])
                throw std::invalid_argument("ds41 expert lane: missing cached span");
            g.gate = span[0]; g.up = span[1]; g.down = span[2];
        } else {
            g.gate = layer.gate + size_t(g.expert) * gu_stride;
            g.up = layer.up + size_t(g.expert) * gu_stride;
            g.down = layer.down + size_t(g.expert) * d_stride;
        }
    }
    if (miss_ && layer.expert_spans)
        throw std::invalid_argument("ds41 expert lane: cached spans do not use mapped-page readers");
    if (miss_) miss_mark(layer);
    if (numa_) assign_teams(layer);
    if (bands_) band_teams();
    act_.resize(size_t(nt) * layer.f.act_bytes);
    hidden_buf_.resize(size_t(nt) * size_t(top_k) * size_t(ff_));
    out_.resize(size_t(nt) * size_t(top_k) * size_t(hidden_));
    hq_.resize(size_t(nt) * size_t(top_k) * layer.f.h_bytes);
    for (int t = 0; t < nt; ++t)
        cpu::native_quant_act(layer.f, z + size_t(t) * size_t(hidden_), act_.data() + size_t(t) * layer.f.act_bytes);
}

// LANE_NUMA: each group to the package holding its gate rows, then groups move from the most to the least loaded
// team (per thread) while that narrows the gap; a group costs 1 + 0.25 per extra token (the weights are decoded once)
void ExpertLane::assign_teams(const LaneLayer& layer) {
    const int first = async_ ? 1 : 0;
    team_size_.assign(size_t(packages_), 0);
    team_index_.assign(size_t(threads_), 0);
    for (int t = first; t < threads_; ++t) team_index_[size_t(t)] = team_size_[size_t(pkg_of_[size_t(t)])]++;
    const size_t gate_bytes = layer.gate_stride ? layer.gate_stride : layer.f.gu_row * size_t(ff_);
    std::vector<double> load(size_t(packages_), 0.0);
    auto cost = [](const Group& g) { return 1.0 + 0.25 * double(g.entries.size() - 1); };
    for (Group& g : groups_) {
        const uint8_t* gate = layer.gate + size_t(g.expert) * gate_bytes;
        auto it = page_pkg_.find(gate);
        if (it == page_pkg_.end()) it = page_pkg_.emplace(gate, page_package(gate)).first;
        int p = it->second;
        if (p < 0 || p >= packages_ || team_size_[size_t(p)] == 0) p = int(std::min_element(load.begin(), load.end()) - load.begin());
        g.team = p;
        load[size_t(p)] += cost(g);
    }
    for (;;) {   // rebalance: per-thread load
        int hi = 0, lo = 0;
        auto per = [&](int p) { return team_size_[size_t(p)] ? load[size_t(p)] / team_size_[size_t(p)] : 1e30; };
        for (int p = 1; p < packages_; ++p) {
            if (per(p) > per(hi)) hi = p;
            if (per(p) < per(lo)) lo = p;
        }
        if (hi == lo) break;
        Group* best = nullptr;
        for (Group& g : groups_)
            if (g.team == hi && (!best || cost(g) < cost(*best))) best = &g;
        if (!best) break;
        const double c = cost(*best);
        const double after = std::max((load[size_t(hi)] - c) / team_size_[size_t(hi)], (load[size_t(lo)] + c) / team_size_[size_t(lo)]);
        if (after >= per(hi)) break;
        best->team = lo;
        load[size_t(hi)] -= c;
        load[size_t(lo)] += c;
    }
}

void ExpertLane::parallel_for(void (*fn)(void*, int, int), void* ctx) {
    if (threads_ < 2 || async_) { fn(ctx, 0, 1); return; }
    job_fn_ = fn;
    job_ctx_ = ctx;
    start_.fetch_add(1, std::memory_order_acq_rel);
    start_.notify_all();
    fn(ctx, 0, threads_);
    barrier();
    job_fn_ = nullptr;
}

void ExpertLane::collect(const int32_t* ids, const float* w, float* y) {
    for (int t = 0; t < nt_; ++t) {
        float* yt = y + size_t(t) * size_t(hidden_);
        bool first = true;
        for (int k = 0; k < top_k_; ++k) {
            if (ids[t * top_k_ + k] < 0) continue;
            const float* o = out_.data() + size_t(t * top_k_ + k) * size_t(hidden_);
            const float wk = w[t * top_k_ + k];
            if (first) for (int j = 0; j < hidden_; ++j) yt[j] = o[j] * wk;
            else for (int j = 0; j < hidden_; ++j) yt[j] += o[j] * wk;
            first = false;
        }
        if (first) std::fill(yt, yt + hidden_, 0.0f);
    }
}

void ExpertLane::run(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, const float* w, int top_k,
                     float* y) {
    if (nt < 1 || top_k < 1) throw std::invalid_argument("ds41 expert lane: token count");
    if (std::all_of(ids, ids + size_t(nt) * size_t(top_k), [](int32_t i) { return i < 0; })) {
        std::fill(y, y + size_t(nt) * size_t(hidden_), 0.0f);
        return;
    }
    async_ = false;
    prepare(layer, z, nt, ids, top_k);
    start_.fetch_add(1, std::memory_order_acq_rel);
    start_.notify_all();
    phase_work(0);
    collect(ids, w, y);
}

void ExpertLane::run_raw(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, int top_k, float* routes) {
    if (nt < 1 || top_k < 1 || !routes) throw std::invalid_argument("ds41 expert lane: token count");
    if (std::all_of(ids, ids + size_t(nt) * size_t(top_k), [](int32_t i) { return i < 0; })) return;
    async_ = false;
    prepare(layer, z, nt, ids, top_k);
    start_.fetch_add(1, std::memory_order_acq_rel);
    start_.notify_all();
    phase_work(0);
    size_t r = 0;
    for (int e = 0; e < nt * top_k; ++e)
        if (ids[e] >= 0) std::copy_n(out_.data() + size_t(e) * size_t(hidden_), hidden_, routes + (r++) * size_t(hidden_));
}

void ExpertLane::begin(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, const float* w, int top_k, float* y) {
    if (nt < 1 || top_k < 1) throw std::invalid_argument("ds41 expert lane: token count");
    ids_ = ids;
    w_ = w;
    y_ = y;
    idle_ = std::all_of(ids, ids + size_t(nt) * size_t(top_k), [](int32_t i) { return i < 0; });
    if (idle_) { nt_ = nt; return; }
    if (threads_ < 2) { async_ = false; prepare(layer, z, nt, ids, top_k); phase_work(0); return; }
    async_ = true;
    done_.store(0, std::memory_order_relaxed);
    begin_t_ = std::chrono::steady_clock::now();
    end_ns_.store(0, std::memory_order_relaxed);
    prepare(layer, z, nt, ids, top_k);
    start_.fetch_add(1, std::memory_order_acq_rel);
    start_.notify_all();
}

void ExpertLane::wait_pending() noexcept {
    if (!idle_ && async_) {
        while (done_.load(std::memory_order_acquire) < threads_ - 1) pause();
        async_ = false;
        const int64_t e = end_ns_.load(std::memory_order_relaxed);
        const int64_t b = std::chrono::duration_cast<std::chrono::nanoseconds>(begin_t_.time_since_epoch()).count();
        if (e > b) async_busy_s_ += double(e - b) * 1e-9;
        ++async_jobs_;
    }
}

void ExpertLane::finish() {
    if (idle_) { std::fill(y_, y_ + size_t(nt_) * size_t(hidden_), 0.0f); return; }
    if (async_) {
        while (done_.load(std::memory_order_acquire) < threads_ - 1) pause();
        async_ = false;
        const int64_t e = end_ns_.load(std::memory_order_relaxed), b = std::chrono::duration_cast<std::chrono::nanoseconds>(begin_t_.time_since_epoch()).count();
        if (e > b) async_busy_s_ += double(e - b) * 1e-9;
        ++async_jobs_;
    }
    collect(ids_, w_, y_);
}

void ExpertLane::finish_with_external2(const int32_t* original_ids, const int32_t* entries, const float* values, int count,
                                       const int32_t* entries2, const float* values2, int count2, int top_k) {
    if (count2 <= 0) { finish_with_external(original_ids, entries, values, count, top_k); return; }
    if (count <= 0) { finish_with_external(original_ids, entries2, values2, count2, top_k); return; }
    if (!original_ids || !entries || !values || !entries2 || !values2 || !ids_ || !w_ || !y_ || nt_ < 1 || nt_ > 4096 ||
        top_k < 1 || top_k > 16 || count + count2 > nt_ * top_k || hidden_ < 1 || hidden_ > 5120)
        throw std::invalid_argument("ds41 expert lane: invalid external route geometry");
    static const bool mstats = strata_env_is_one("STRATA_DS41_LANE_STATS");
    auto ns = [] { return double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); };
    const double m0 = mstats ? ns() : 0.0;
    const int total = nt_ * top_k;
    std::vector<uint8_t> external(size_t(total), 0);
    for (int i = 0; i < count + count2; ++i) {
        const int e = i < count ? entries[i] : entries2[i - count];
        if (e < 0 || e >= total || external[size_t(e)] || ids_[e] >= 0 || original_ids[e] < 0)
            throw std::invalid_argument("ds41 expert lane: invalid external route index");
        external[size_t(e)] = 1;
    }
    for (int i = 0; i < total; ++i)
        if ((external[size_t(i)] ? original_ids[i] < 0 : original_ids[i] != ids_[i]))
            throw std::invalid_argument("ds41 expert lane: inconsistent external routes");
    // Opt-in: consume each source in place, preserving collect()'s route and float order.
    if (strata_env_is_one("STRATA_DS41_MERGE_DIRECT")) {
        const double m1 = mstats ? ns() : 0.0;
        if (!idle_ && async_) {
            while (done_.load(std::memory_order_acquire) < threads_ - 1) pause();
            async_ = false;
            const int64_t e = end_ns_.load(std::memory_order_relaxed);
            const int64_t b = std::chrono::duration_cast<std::chrono::nanoseconds>(begin_t_.time_since_epoch()).count();
            if (e > b) async_busy_s_ += double(e - b) * 1e-9;
            ++async_jobs_;
        }
        if (!idle_ && (top_k_ != top_k || out_.size() < size_t(total) * size_t(hidden_)))
            throw std::logic_error("ds41 expert lane: external output capacity mismatch");
        const double m2 = mstats ? ns() : 0.0;
        std::vector<const float*> sources(size_t(total), nullptr);
        for (int i = 0; i < count; ++i) sources[size_t(entries[i])] = values + size_t(i) * hidden_;
        for (int i = 0; i < count2; ++i) sources[size_t(entries2[i])] = values2 + size_t(i) * hidden_;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        if (strata_env_is_one("STRATA_DS41_MERGE_VECTOR")) {
            // each token is summed in its own route order whichever thread does it: STRATA_DS41_MERGE_PAR (default on) spreads
            // the tokens of a wide pass (batched sessions) over the lane threads, the same bits
            auto body = [&](int part, int parts) {
                for (int t = part; t < nt_; t += parts) {
                    const float* p[16]; float weights[16]; int n = 0;
                    for (int k = 0; k < top_k; ++k) {
                        const int route = t * top_k + k;
                        if (original_ids[route] < 0) continue;
                        p[n] = sources[size_t(route)] ? sources[size_t(route)] : out_.data() + size_t(route) * hidden_;
                        weights[n++] = w_[route];
                    }
                    collect_vectors(y_ + size_t(t) * hidden_, p, weights, n, hidden_);
                }
            };
            static const bool par = !(std::getenv("STRATA_DS41_MERGE_PAR") && std::getenv("STRATA_DS41_MERGE_PAR")[0] == '0');
            if (par && nt_ >= 16 && threads_ > 1 && !async_) {
                parallel_for([](void* c, int part, int parts) { (*static_cast<decltype(body)*>(c))(part, parts); }, &body);
            } else {
                body(0, 1);
            }
            if (mstats) { const double m3 = ns(); merge_check_ns_ += m1 - m0; merge_wait_ns_ += m2 - m1; merge_work_ns_ += m3 - m2; ++merge_calls_; }
            return;
        }
#endif
        for (int t = 0; t < nt_; ++t) {
            float* yt = y_ + size_t(t) * hidden_;
            bool first = true;
            for (int k = 0; k < top_k; ++k) {
                const int route = t * top_k + k;
                if (original_ids[route] < 0) continue;
                const float* o = sources[size_t(route)];
                if (!o) o = out_.data() + size_t(route) * hidden_;
                const float wk = w_[route];
                if (first) for (int j = 0; j < hidden_; ++j) yt[j] = o[j] * wk;
                else for (int j = 0; j < hidden_; ++j) yt[j] += o[j] * wk;
                first = false;
            }
            if (first) std::fill(yt, yt + hidden_, 0.0f);
        }
        return;
    }
    const bool was_idle = idle_;
    finish();
    if (was_idle) {
        top_k_ = top_k;
        out_.assign(size_t(total) * size_t(hidden_), 0.f);
    } else if (top_k_ != top_k || out_.size() < size_t(total) * size_t(hidden_)) {
        throw std::logic_error("ds41 expert lane: external output capacity mismatch");
    }
    for (int i = 0; i < count; ++i) std::copy_n(values + size_t(i) * hidden_, hidden_, out_.data() + size_t(entries[i]) * hidden_);
    for (int i = 0; i < count2; ++i) std::copy_n(values2 + size_t(i) * hidden_, hidden_, out_.data() + size_t(entries2[i]) * hidden_);
    collect(original_ids, w_, y_);
}

void ExpertLane::finish_with_external(const int32_t* original_ids, const int32_t* entries,
                                      const float* values, int count, int top_k) {
    if (!original_ids || !entries || !values || !ids_ || !w_ || !y_ || nt_ < 1 || nt_ > 4096 ||
        top_k < 1 || top_k > 16 || count < 1 || count > nt_ * top_k || hidden_ < 1 || hidden_ > 5120)
        throw std::invalid_argument("ds41 expert lane: invalid external route geometry");
    const int total = nt_ * top_k;
    std::vector<uint8_t> external(size_t(total), 0);
    for (int i = 0; i < count; ++i) {
        const int e = entries[i];
        if (e < 0 || e >= total || external[size_t(e)] || ids_[e] >= 0 || original_ids[e] < 0)
            throw std::invalid_argument("ds41 expert lane: invalid external route index");
        external[size_t(e)] = 1;
    }
    for (int i = 0; i < total; ++i)
        if ((external[size_t(i)] ? original_ids[i] < 0 : original_ids[i] != ids_[i]))
            throw std::invalid_argument("ds41 expert lane: inconsistent external routes");
    const bool was_idle = idle_;
    finish();
    if (was_idle) {
        top_k_ = top_k;
        out_.assign(size_t(total) * size_t(hidden_), 0.f);
    } else if (top_k_ != top_k || out_.size() < size_t(total) * size_t(hidden_)) {
        throw std::logic_error("ds41 expert lane: external output capacity mismatch");
    }
    for (int i = 0; i < count; ++i)
        std::copy_n(values + size_t(i) * hidden_, hidden_, out_.data() + size_t(entries[i]) * hidden_);
    collect(original_ids, w_, y_);
}

}  // namespace strata::ds41
