// src/ds41/expert_lane.cpp - DeepSeek-V4.1's CPU expert lane (strata/ds41/expert_lane.hpp).
#include "strata/ds41/expert_lane.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <map>
#include <pthread.h>
#include <sched.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
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
}  // namespace

ExpertLane::ExpertLane(int hidden, int ff, int threads, bool pin_threads)
    : hidden_(hidden), ff_(ff), threads_(std::max(1, threads)) {
    // STRATA_DS41_LANE_SLEEP (1): idle workers block on the start counter; 0: they yield (as before)
    sleep_ = !(std::getenv("STRATA_DS41_LANE_SLEEP") && std::string(std::getenv("STRATA_DS41_LANE_SLEEP")) == "0");
    cpu_of_.assign(size_t(threads_), -1);
    if (pin_threads) {
        const auto cores = cpu::physical_cores(false);
        for (int t = 0; t < threads_ && !cores.empty(); ++t) cpu_of_[size_t(t)] = cores[size_t(t) % cores.size()];
    }
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
            else if (sleep_) start_.wait(seen, std::memory_order_acquire);   // idle (GPU work, streamed prefill): sleep
            else std::this_thread::yield();
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = start_.load(std::memory_order_acquire);
        phase_work(thread);
    }
}

// The three phases of one layer for one thread; each ends at a barrier every thread reaches.
void ExpertLane::phase_work(int t) {
    const int T = threads_;
    const cpu::NativeFmt& f = layer_->f;
    const size_t act_bytes = f.act_bytes, h_bytes = f.h_bytes;
    const size_t gate_bytes = layer_->gate_stride ? layer_->gate_stride : f.gu_row * size_t(ff_);
    const size_t down_bytes = layer_->down_stride ? layer_->down_stride : f.d_row * size_t(hidden_);
    // 1. gate/up: this thread's rows of every selected expert (all of its tokens at once)
    {
        const int r0 = int(int64_t(ff_) * t / T), r1 = int(int64_t(ff_) * (t + 1) / T);
        for (const Group& g : groups_) {
            if (r0 == r1) break;
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
        const int r0 = int(int64_t(hidden_) * t / T), r1 = int(int64_t(hidden_) * (t + 1) / T);
        for (const Group& g : groups_) {
            if (r0 == r1) break;
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

void ExpertLane::run(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, const float* w, int top_k,
                     float* y) {
    if (nt < 1 || top_k < 1) throw std::invalid_argument("ds41 expert lane: token count");
    if (std::all_of(ids, ids + size_t(nt) * size_t(top_k), [](int32_t i) { return i < 0; })) {
        std::fill(y, y + size_t(nt) * size_t(hidden_), 0.0f);
        return;
    }
    if (layer.f.n_embd != hidden_ || layer.f.n_ff != ff_) throw std::invalid_argument("ds41 expert lane: format geometry");
    layer_ = &layer;
    nt_ = nt;
    top_k_ = top_k;
    // group the routes by expert (the order of the groups does not change any value)
    std::map<int, std::vector<int>> by_expert;
    for (int e = 0; e < nt * top_k; ++e)
        if (ids[e] >= 0) by_expert[ids[e]].push_back(e);   // negative: handled elsewhere (the GPU tier)
    groups_.clear();
    for (auto& [expert, entries] : by_expert) groups_.push_back({expert, std::move(entries)});
    act_.resize(size_t(nt) * layer.f.act_bytes);
    hidden_buf_.resize(size_t(nt) * size_t(top_k) * size_t(ff_));
    out_.resize(size_t(nt) * size_t(top_k) * size_t(hidden_));
    hq_.resize(size_t(nt) * size_t(top_k) * layer.f.h_bytes);
    for (int t = 0; t < nt; ++t)
        cpu::native_quant_act(layer.f, z + size_t(t) * size_t(hidden_), act_.data() + size_t(t) * layer.f.act_bytes);
    start_.fetch_add(1, std::memory_order_acq_rel);
    start_.notify_all();
    phase_work(0);
    for (int t = 0; t < nt; ++t) {
        float* yt = y + size_t(t) * size_t(hidden_);
        bool first = true;
        for (int k = 0; k < top_k; ++k) {
            if (ids[t * top_k + k] < 0) continue;
            const float* o = out_.data() + size_t(t * top_k + k) * size_t(hidden_);
            const float wk = w[t * top_k + k];
            if (first) for (int j = 0; j < hidden_; ++j) yt[j] = o[j] * wk;
            else for (int j = 0; j < hidden_; ++j) yt[j] += o[j] * wk;
            first = false;
        }
        if (first) std::fill(yt, yt + hidden_, 0.0f);
    }
}

}  // namespace strata::ds41
