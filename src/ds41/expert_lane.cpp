// src/ds41/expert_lane.cpp - DeepSeek-V4.1's CPU expert lane (strata/ds41/expert_lane.hpp).
#include "strata/ds41/expert_lane.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cstdio>
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
        for (int t = 0; t < threads_ && !cpus.empty(); ++t) cpu_of_[size_t(t)] = cpus[size_t(t) % cpus.size()];
    }
    // STRATA_DS41_LANE_NUMA=1 (pinned threads): each socket's threads take the experts whose rows sit in its memory
    numa_ = pin_threads && std::getenv("STRATA_DS41_LANE_NUMA") && std::string(std::getenv("STRATA_DS41_LANE_NUMA")) == "1";
    pkg_of_.assign(size_t(threads_), 0);
    for (int t = 0; t < threads_; ++t) pkg_of_[size_t(t)] = cpu_of_[size_t(t)] >= 0 ? cpu_package(cpu_of_[size_t(t)]) : 0;
    packages_ = 1 + *std::max_element(pkg_of_.begin(), pkg_of_.end());
    if (packages_ < 2) numa_ = false;
    for (int i = 1; i < threads_; ++i) pool_.emplace_back([this, i] { pin(i); worker(i); });
    pin(0);
}

ExpertLane::~ExpertLane() {
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
    const unsigned s = sense_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == (async_ ? threads_ - 1 : threads_)) {
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
        phase_work(thread);
        if (async_) done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

// The three phases of one layer for one thread; each ends at a barrier every thread reaches.
void ExpertLane::phase_work(int t) {
    const int T = async_ ? threads_ - 1 : threads_;   // begin(): threads 1.. only
    if (async_) --t;
    const cpu::NativeFmt& f = layer_->f;
    const size_t act_bytes = f.act_bytes, h_bytes = f.h_bytes;
    const size_t gate_bytes = layer_->gate_stride ? layer_->gate_stride : f.gu_row * size_t(ff_);
    const size_t down_bytes = layer_->down_stride ? layer_->down_stride : f.d_row * size_t(hidden_);
    // with teams (LANE_NUMA): this thread's share of its team's groups only
    const int abs_t = async_ ? t + 1 : t;
    const int team = numa_ ? pkg_of_[size_t(abs_t)] : -1;
    const int tn = numa_ ? team_size_[size_t(team)] : T, ti = numa_ ? team_index_[size_t(abs_t)] : t;
    // 1. gate/up: this thread's rows of every selected expert (all of its tokens at once)
    {
        const int r0 = int(int64_t(ff_) * ti / tn), r1 = int(int64_t(ff_) * (ti + 1) / tn);
        for (const Group& g : groups_) {
            if (r0 == r1) break;
            if (numa_ && g.team != team) continue;
            const uint8_t* gate = layer_->gate + size_t(g.expert) * gate_bytes;
            const uint8_t* up = layer_->up + size_t(g.expert) * gate_bytes;
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
    barrier();
    // 2. the hidden of every entry, quantized for the down rows
    {
        const int entries = nt_ * top_k_;
        for (int e = entries * t / T; e < entries * (t + 1) / T; ++e)
            cpu::native_quant_h(f, hidden_buf_.data() + size_t(e) * size_t(ff_), hq_.data() + size_t(e) * h_bytes);
    }
    barrier();
    // 3. down: this thread's hidden rows of every selected expert
    {
        const int r0 = int(int64_t(hidden_) * ti / tn), r1 = int(int64_t(hidden_) * (ti + 1) / tn);
        for (const Group& g : groups_) {
            if (r0 == r1) break;
            if (numa_ && g.team != team) continue;
            const uint8_t* gate = layer_->gate + size_t(g.expert) * gate_bytes;
            const uint8_t* down = layer_->down + size_t(g.expert) * down_bytes;
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
    barrier();
}

void ExpertLane::prepare(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, int top_k) {
    if (layer.f.n_embd != hidden_ || layer.f.n_ff != ff_) throw std::invalid_argument("ds41 expert lane: format geometry");
    layer_ = &layer;
    nt_ = nt;
    top_k_ = top_k;
    // group the routes by expert (the order of the groups does not change any value)
    // STRATA_DS41_VERIFY_EXACT=1: one group per route, so a token's rows round as when it is alone
    static const bool split = std::getenv("STRATA_DS41_VERIFY_EXACT") && std::string(std::getenv("STRATA_DS41_VERIFY_EXACT")) == "1";
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
    if (numa_) assign_teams(layer);
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
    prepare(layer, z, nt, ids, top_k);
    start_.fetch_add(1, std::memory_order_acq_rel);
    start_.notify_all();
}

void ExpertLane::finish() {
    if (idle_) { std::fill(y_, y_ + size_t(nt_) * size_t(hidden_), 0.0f); return; }
    if (async_) {
        while (done_.load(std::memory_order_acquire) < threads_ - 1) pause();
        async_ = false;
    }
    collect(ids_, w_, y_);
}

}  // namespace strata::ds41
