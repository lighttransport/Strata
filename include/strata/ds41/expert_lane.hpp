// include/strata/ds41/expert_lane.hpp - DeepSeek-V4.1's routed experts on the CPU, row-parallel over persistent threads.
//
// One call per MoE layer: for nt tokens with their top-k routes, y[t] = sum_k w[t][k] * E_k(z[t]), E_k expert k's
// clamped SwiGLU (gate/up hidden -> ff, down ff -> hidden). Every thread takes its share of the rows of every selected
// expert, so a layer is three phases (gate/up, the hidden's quantization, down) separated by spin barriers. The rows
// use the engine's native kernels (NativeFmt::iq1s_rows for IQ1_S), and a row's arithmetic does not depend on which
// thread computes it: the result is bit-identical for any thread count. The weighted sum runs on the calling thread
// in route order (k = 0..top_k-1), the order of llama.cpp's ggml_mul + sum over the expert outputs.
// Adapted from the Muse Spark lane (branch worktree-muse-spark-sim, src/muse/expert_lane.cpp).
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace strata::ds41 {

/// One layer's experts as GGUF stores them: [experts][rows] of NativeFmt rows per projection.
struct LaneLayer {
    strata::kernels::cpu::NativeFmt f;
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
};

class ExpertLane {
public:
    /// threads >= 1 (the caller is one of them); pin: pin thread t to the t-th physical core of the machine.
    ExpertLane(int hidden, int ff, int threads, bool pin = false);
    ~ExpertLane();
    ExpertLane(const ExpertLane&) = delete;
    ExpertLane& operator=(const ExpertLane&) = delete;

    /// z: [nt][hidden]; ids, w: [nt][top_k]; y: [nt][hidden] (written). A negative id is a route computed elsewhere: it
    /// adds nothing here (a token with none of its routes here gets zeros).
    void run(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, const float* w, int top_k, float* y);
    int threads() const { return threads_; }

private:
    struct Group { int expert; std::vector<int> entries; };   // entries: t * top_k + k routed to this expert
    void phase_work(int thread);
    void barrier();
    void worker(int thread);
    void pin(int thread);

    int hidden_, ff_, threads_;
    // per call
    const LaneLayer* layer_ = nullptr;
    int nt_ = 0, top_k_ = 0;
    std::vector<Group> groups_;
    std::vector<uint8_t> act_;          // [nt][act_bytes]
    std::vector<float> hidden_buf_, out_;   // [entries][ff], [entries][hidden]
    std::vector<uint8_t> hq_;           // [entries][h_bytes]
    // threads
    std::vector<std::thread> pool_;
    std::vector<int> cpu_of_;
    std::atomic<unsigned> start_{0};
    std::atomic<int> arrived_{0};
    std::atomic<unsigned> sense_{0};
    std::atomic<bool> stop_{false};
    bool sleep_ = true;
};

}  // namespace strata::ds41
