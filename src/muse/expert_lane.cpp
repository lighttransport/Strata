// src/muse/expert_lane.cpp - Muse Spark's CPU expert lane (strata/muse/expert_lane.hpp).
#include "strata/muse/expert_lane.hpp"
#include "strata/muse/reference.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <map>
#include <stdexcept>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace strata::muse {
namespace cpu = strata::kernels::cpu;

namespace {
inline void pause() {
#if defined(__x86_64__)
    _mm_pause();
#endif
}
}  // namespace

namespace {
// CPU lists of each NUMA node (/sys/devices/system/node/nodeN/cpulist), physical cores first: a core's SMT sibling
// is listed after every first thread of the node's cores (Linux numbers the siblings of 0..15 as 16..31 here).
std::vector<std::vector<int>> numa_cpus() {
    std::vector<std::vector<int>> nodes;
    for (int n = 0;; ++n) {
        std::ifstream f("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (!f) break;
        std::string list;
        std::getline(f, list);
        std::vector<int> cpus;
        std::stringstream ss(list);
        for (std::string part; std::getline(ss, part, ',');) {
            const auto dash = part.find('-');
            const int a = std::stoi(part.substr(0, dash)), b = dash == std::string::npos ? a : std::stoi(part.substr(dash + 1));
            for (int c = a; c <= b; ++c) cpus.push_back(c);
        }
        std::vector<int> first, second;   // the first CPU of each core, then the siblings
        for (int c : cpus) {
            std::ifstream t("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            std::string sib;
            std::getline(t, sib);
            const int lead = sib.empty() ? c : std::stoi(sib.substr(0, sib.find_first_of(",-")));
            (lead == c ? first : second).push_back(c);
        }
        first.insert(first.end(), second.begin(), second.end());
        nodes.push_back(first);
    }
    return nodes;
}
}  // namespace

ExpertLane::ExpertLane(const cpu::NativeFmt& f, int latent, int ff, float post_eps, int threads, bool numa)
    : f_(f), latent_(latent), ff_(ff), threads_(std::max(1, threads)), post_eps_(post_eps) {
    if (f.n_embd != latent || f.n_ff != ff) throw std::invalid_argument("muse expert lane: format geometry mismatch");
    cpu_of_.assign(threads_, -1);
    node_of_.assign(threads_, -1);
    if (numa) {
        const auto nodes = numa_cpus();
        if (nodes.size() >= 2) {
            numa_ = true;
            const int n0 = (threads_ + 1) / 2;   // threads [0, n0) on node 0: they hold the first rows
            for (int t = 0; t < threads_; ++t) {
                const int node = t < n0 ? 0 : 1, k = t < n0 ? t : t - n0;
                node_of_[t] = node;
                cpu_of_[t] = nodes[node][(size_t) k % nodes[node].size()];
            }
        }
    }
    for (int i = 1; i < threads_; ++i) pool_.emplace_back([this, i] { pin(i); worker(i); });
    pin(0);
}

void ExpertLane::pin(int thread) {
    if (!numa_ || cpu_of_[thread] < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_of_[thread], &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

std::string ExpertLane::placement() const {
    if (!numa_) return "unpinned";
    std::string s;
    for (int t = 0; t < threads_; ++t) s += (t ? " " : "") + std::to_string(node_of_[t]) + ":" + std::to_string(cpu_of_[t]);
    return s;
}

void ExpertLane::dispatch(int t) {
    if (!adopt_src_) { phase_work(t); return; }
    // adopt: this thread's rows of every expert, first written by this thread
    const int T = threads_;
    const int r0 = (int) ((int64_t) ff_ * t / T), r1 = (int) ((int64_t) ff_ * (t + 1) / T);
    const int d0 = (int) ((int64_t) latent_ * t / T), d1 = (int) ((int64_t) latent_ * (t + 1) / T);
    for (int e = 0; e < adopt_experts_; ++e) {
        const size_t g = ((size_t) e * ff_ + r0) * f_.gu_row, gn = (size_t) (r1 - r0) * f_.gu_row;
        std::memcpy(const_cast<uint8_t*>(adopt_dst_.gate) + g, adopt_src_->gate + g, gn);
        std::memcpy(const_cast<uint8_t*>(adopt_dst_.up) + g, adopt_src_->up + g, gn);
        const size_t d = ((size_t) e * latent_ + d0) * f_.d_row, dn = (size_t) (d1 - d0) * f_.d_row;
        std::memcpy(const_cast<uint8_t*>(adopt_dst_.down) + d, adopt_src_->down + d, dn);
    }
    barrier();
}

LaneLayer ExpertLane::adopt(const LaneLayer& src, int experts) {
    const size_t gbytes = (size_t) experts * ff_ * f_.gu_row, dbytes = (size_t) experts * latent_ * f_.d_row;
    auto map = [&](size_t bytes) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);   // untouched pages
        if (p == MAP_FAILED) throw std::runtime_error("muse expert lane: cannot map expert memory");
        owned_.push_back({p, bytes});
        return static_cast<uint8_t*>(p);
    };
    adopt_dst_ = {map(gbytes), map(gbytes), map(dbytes), src.post_exp};
    adopt_src_ = &src;
    adopt_experts_ = experts;
    start_.fetch_add(1, std::memory_order_acq_rel);
    dispatch(0);
    adopt_src_ = nullptr;
    return adopt_dst_;
}

ExpertLane::~ExpertLane() {
    stop_.store(true);
    for (auto& t : pool_) t.join();
    for (const Mapping& m : owned_) munmap(m.p, m.bytes);
}

void ExpertLane::barrier() {
    const unsigned s = sense_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == threads_) {
        arrived_.store(0, std::memory_order_relaxed);
        sense_.store(s + 1, std::memory_order_release);
    } else {
        while (sense_.load(std::memory_order_acquire) == s) pause();
    }
}

void ExpertLane::worker(int thread) {
    unsigned seen = 0;
    for (;;) {
        unsigned spins = 0;
        while (start_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            if (++spins < (1u << 14)) pause();
            else std::this_thread::yield();   // between layers the GPU trunk runs; do not hold a core forever
        }
        seen = start_.load(std::memory_order_acquire);
        dispatch(thread);
    }
}

// The three phases of one layer for one thread; each ends at a barrier every thread reaches.
void ExpertLane::phase_work(int t) {
    const int T = threads_;
    const size_t act_bytes = f_.act_bytes, h_bytes = f_.h_bytes;
    // 1. gate/up: this thread's rows of every selected expert (all of its tokens at once)
    {
        const int r0 = (int) ((int64_t) ff_ * t / T), r1 = (int) ((int64_t) ff_ * (t + 1) / T);
        for (const Group& g : groups_) {
            if (r0 == r1) break;
            const uint8_t* gate = layer_->gate + (size_t) g.expert * f_.gu_row * ff_;
            const uint8_t* up = layer_->up + (size_t) g.expert * f_.gu_row * ff_;
            for (size_t j0 = 0; j0 < g.entries.size(); j0 += cpu::MAXT) {   // the row kernels take up to MAXT tokens
                const size_t ng = std::min<size_t>(cpu::MAXT, g.entries.size() - j0);
                const void* acts[cpu::MAXT];
                float* ffs[cpu::MAXT];
                for (size_t j = 0; j < ng; ++j) {
                    acts[j] = act_.data() + (size_t) (g.entries[j0 + j] / top_k_) * act_bytes;
                    ffs[j] = hidden_.data() + (size_t) g.entries[j0 + j] * ff_;
                }
                cpu::native_gu_rows(f_, gate, acts, (int) ng, ffs, r0, r1, up);
            }
        }
    }
    barrier();
    // 2. the hidden of every entry, quantized for the down rows
    {
        const int entries = nt_ * top_k_;
        for (int e = entries * t / T; e < entries * (t + 1) / T; ++e)
            cpu::native_quant_h(f_, hidden_.data() + (size_t) e * ff_, hq_.data() + (size_t) e * h_bytes);
    }
    barrier();
    // 3. down: this thread's latent rows of every selected expert
    {
        const int r0 = (int) ((int64_t) latent_ * t / T), r1 = (int) ((int64_t) latent_ * (t + 1) / T);
        for (const Group& g : groups_) {
            if (r0 == r1) break;
            const uint8_t* gate = layer_->gate + (size_t) g.expert * f_.gu_row * ff_;
            const uint8_t* down = layer_->down + (size_t) g.expert * f_.d_row * latent_;
            for (size_t j0 = 0; j0 < g.entries.size(); j0 += cpu::MAXT) {
                const size_t ng = std::min<size_t>(cpu::MAXT, g.entries.size() - j0);
                const void* hqs[cpu::MAXT];
                float* outs[cpu::MAXT];
                for (size_t j = 0; j < ng; ++j) {
                    hqs[j] = hq_.data() + (size_t) g.entries[j0 + j] * h_bytes;
                    outs[j] = out_.data() + (size_t) g.entries[j0 + j] * latent_;
                }
                cpu::native_down_rows(f_, gate, hqs, (int) ng, outs, r0, r1, down);
            }
        }
    }
    barrier();
}

void ExpertLane::run_raw(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, int top_k) {
    if (nt < 1 || top_k < 1) throw std::invalid_argument("muse expert lane: token count");
    layer_ = &layer;
    nt_ = nt;
    top_k_ = top_k;
    // group the routes by expert (the order of the groups does not change any value)
    std::map<int, std::vector<int>> by_expert;
    for (int e = 0; e < nt * top_k; ++e) by_expert[ids[e]].push_back(e);
    groups_.clear();
    for (auto& [expert, entries] : by_expert) {
        groups_.push_back({expert, std::move(entries)});
    }
    act_.resize((size_t) nt * f_.act_bytes);
    hidden_.resize((size_t) nt * top_k * ff_);
    out_.resize((size_t) nt * top_k * latent_);
    hq_.resize((size_t) nt * top_k * f_.h_bytes);
    for (int t = 0; t < nt; ++t) cpu::native_quant_act(f_, z + (size_t) t * latent_, act_.data() + (size_t) t * f_.act_bytes);
    start_.fetch_add(1, std::memory_order_acq_rel);
    dispatch(0);
}

void ExpertLane::combine(float* raw, const float* post_exp, float post_eps, int latent, int nt, const float* w, int top_k,
                         float* y) {
    // each expert's output normalized (ffn_post_exp_norm), then weighted, in route order
    for (int t = 0; t < nt; ++t) {
        float* yt = y + (size_t) t * latent;
        std::fill(yt, yt + latent, 0.f);
        for (int k = 0; k < top_k; ++k) {
            float* o = raw + (size_t) (t * top_k + k) * latent;
            for (int j = 0; j < latent; ++j) o[j] *= post_exp[j];
            ref::rms_norm(o, latent, post_eps);
            const float wk = w[t * top_k + k];
            for (int j = 0; j < latent; ++j) yt[j] += wk * o[j];
        }
    }
}

void ExpertLane::run(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, const float* w, int top_k,
                     float* y) {
    run_raw(layer, z, nt, ids, top_k);
    combine(out_.data(), layer.post_exp, post_eps_, latent_, nt, w, top_k, y);
}

}  // namespace strata::muse
