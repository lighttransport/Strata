// include/strata/muse/expert_lane.hpp - Muse Spark's routed experts on the CPU, row-parallel over persistent threads.
//
// One call per MoE layer: for nt tokens with their top-k routes, y[t] = sum_k w[t][k] * rms_norm(post_exp * E_k(z[t]))
// over the latent width, where E_k is expert k's SwiGLU (gate/up latent -> ff, down ff -> latent). Every thread takes
// its share of the rows of every selected expert, so a layer is three phases (gate/up, the hidden's quantization, down)
// separated by spin barriers, whatever the number of experts. The rows use the engine's native kernels
// (NativeFmt::iq1s_rows for the IQ1_S experts), and a row's arithmetic does not depend on which thread computes it:
// the result is bit-identical for any thread count. The post-expert norm and the weighted sum run on the calling
// thread in route order (k = 0..top_k-1), as strata/muse/reference.hpp and tools/muse_gpu_trunk_check.cpp do.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace strata::muse {

struct LaneLayer {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;   // [experts][rows] of NativeFmt rows
    const float* post_exp = nullptr;                                 // ffn_post_exp_norm weight, latent wide
};

class ExpertLane {
public:
    /// f: the experts' NativeFmt (gate/up latent -> ff, down ff -> latent); threads >= 1 (the caller is one of them).
    /// numa: pin the threads to cores, the first half on NUMA node 0 and the rest on node 1 (physical cores first), so a
    /// thread's rows are always read by the same node; with adopt() the rows also live there. Off: unpinned threads.
    ExpertLane(const strata::kernels::cpu::NativeFmt& f, int latent, int ff, float post_eps, int threads, bool numa = false);
    ~ExpertLane();
    ExpertLane(const ExpertLane&) = delete;
    ExpertLane& operator=(const ExpertLane&) = delete;

    /// z: [nt][latent]; ids, w: [nt][top_k]; y: [nt][latent] (written).
    void run(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, const float* w, int top_k, float* y);
    /// The experts' raw outputs only (before the post-expert norm and the router weight), for a row split: each node
    /// runs its share of every expert's ff rows and the partials are added before combine(). Read them with raw().
    void run_raw(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, int top_k);
    float* raw() { return out_.data(); }   // [nt * top_k][latent], entry t * top_k + k
    /// y[t] = sum_k w[t][k] * rms_norm(post_exp * raw[t][k]), in route order. raw is modified (normalized in place).
    static void combine(float* raw, const float* post_exp, float post_eps, int latent, int nt, const float* w, int top_k, float* y);
    int threads() const { return threads_; }
    /// A copy of a layer's experts (`experts` of them) owned by the lane, each thread's rows first written by that thread,
    /// so with numa they sit on its node. The arithmetic does not change. The source may be dropped afterwards.
    LaneLayer adopt(const LaneLayer& src, int experts);
    /// "node:cores" of every thread, for logs.
    std::string placement() const;

private:
    struct Group { int expert; std::vector<int> entries; };   // entries: t * top_k + k routed to this expert
    void phase_work(int thread);
    void dispatch(int thread);
    void pin(int thread);
    void barrier();
    void worker(int thread);

    strata::kernels::cpu::NativeFmt f_;
    int latent_, ff_, threads_;
    float post_eps_;
    // per call
    const LaneLayer* layer_ = nullptr;
    int nt_ = 0, top_k_ = 0;
    std::vector<Group> groups_;
    std::vector<uint8_t> act_;      // [nt][act_bytes]
    std::vector<float> hidden_, out_;   // [entries][ff], [entries][latent]
    std::vector<uint8_t> hq_;       // [entries][h_bytes]
    // threads
    std::vector<std::thread> pool_;
    std::atomic<unsigned> start_{0};
    std::atomic<int> arrived_{0};
    std::atomic<unsigned> sense_{0};
    std::atomic<bool> stop_{false};
    // NUMA placement
    bool numa_ = false;
    std::vector<int> cpu_of_, node_of_;           // per thread
    // the adopt job, when set instead of the layer phases
    const LaneLayer* adopt_src_ = nullptr;
    LaneLayer adopt_dst_;
    int adopt_experts_ = 0;
    struct Mapping { void* p; size_t bytes; };
    std::vector<Mapping> owned_;
};

}  // namespace strata::muse
