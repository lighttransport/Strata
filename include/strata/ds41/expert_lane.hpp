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
#include <chrono>
#include <map>
#include <memory>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace strata::ds41 {

/// One layer's experts as GGUF stores them: [experts][rows] of NativeFmt rows per projection. The strides are the
/// bytes from one expert to the next (0: f.gu_row * f.n_ff and f.d_row * hidden, the GGUF tensors). A row slice of
/// wider experts sets f.n_ff to its rows and keeps the full strides and f.d_row, so the down rows read their first
/// f.n_ff columns; experts packed one blob each (gate | up | down) set every stride to the blob's size.
struct LaneLayer {
    strata::kernels::cpu::NativeFmt f;
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    size_t gate_stride = 0, down_stride = 0;
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
    /// run() in two halves: begin() starts the work on the other threads (the caller's thread is free until
    /// finish(), which waits for them and writes y). ids, w, z and y must stay valid until finish().
    void begin(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, const float* w, int top_k, float* y);
    void finish();
    /// Finish an asynchronous job and insert bounded externally computed raw
    /// route outputs before collecting in the original route order.
    void finish_with_external(const int32_t* original_ids, const int32_t* entries,
                              const float* values, int count, int top_k);
    /// finish_with_external with two sources of raw route outputs (e.g. a GPU tier's and a remote worker's), each copied
    /// once into place.
    void finish_with_external2(const int32_t* original_ids, const int32_t* entries, const float* values, int count,
                               const int32_t* entries2, const float* values2, int count2, int top_k);
    /// The raw (unweighted) output of every route with ids >= 0, one hidden-wide row each in route order into `routes`
    /// (a remote node's whole experts: the caller merges them in its own collect order).
    void run_raw(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, int top_k, float* routes);
    int threads() const { return threads_; }
    int caller_cpu() const { return cpu_of_.empty() ? -1 : cpu_of_[0]; }   // the core the caller (lane thread 0) computes on

    /// STRATA_DS41_MISS_READ=N (opt-in): queued parallel reads for experts that may have left the page cache. An
    /// expert not used for STRATA_DS41_MISS_GAP calls (default 1376, about 32 tokens of 43 layers) is "cold": its
    /// groups run last, and N reader threads check its pages (mincore) and read the missing ones in 1 MiB pieces
    /// (MADV_POPULATE_READ), N reads in flight instead of one fault at a time on a lane thread. A lane thread waits
    /// for a cold group's reads before its rows. Values are unchanged (the same rows, the same order of sums).
    struct MissStats { uint64_t calls = 0, cold = 0, read_pieces = 0, read_bytes = 0; double wait_s = 0, read_s = 0; };
    MissStats miss_stats() const;
    void miss_reset();   // the counters restart (decode's own)
    /// begin()/finish() jobs: their count and the seconds from begin() to the last thread's end (the lane's own time)
    double async_busy_s() const { return async_busy_s_; }
    long async_jobs() const { return async_jobs_; }
    void async_reset() { async_busy_s_ = 0; async_jobs_ = 0; }
    /// STRATA_DS41_LANE_BANDS=1 (opt-in; Strata PR #4016's band layout, in place): every 128-row band of every expert
    /// matrix lives on NUMA node (band + expert) % nodes, and each node's pinned lane threads compute only the bands in
    /// their own memory. place_bands() moves a layer's mapped pages there (move_pages); returns the bytes asked.
    /// Rows are computed by the same kernels whichever thread runs them: results are bit-identical.
    bool bands() const { return bands_; }
    int nodes() const { return nodes_; }
    static size_t place_bands(const LaneLayer& layer, int experts, int hidden, int ff, int nodes);
    /// MADV_PAGEOUT on a fraction of a layer's experts (a test: STRATA_DS41_MISS_TEST_DROP); returns the bytes asked
    static size_t page_out(const LaneLayer& layer, int experts, int hidden, int ff, double fraction, unsigned seed);

private:
    // entries: t * top_k + k routed to this expert; ready: a cold group's reads (MISS_READ), null when warm
    struct Group { int expert; std::vector<int> entries; int team = 0; std::atomic<int>* ready = nullptr; };
    void phase_work(int thread);
    // fn(ctx, part, parts) on every lane thread (the caller is part 0) and a barrier: the merge across tokens (needs !async_)
    void parallel_for(void (*fn)(void*, int, int), void* ctx);
    void (*job_fn_)(void*, int, int) = nullptr;
    void* job_ctx_ = nullptr;
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
    // STRATA_DS41_LANE_STATS=1: per thread, ns waiting at each of a layer call's three barriers and ns of work before them
    double stat_wait_[64][3] = {}, stat_work_[64][3] = {};
    long stat_calls_ = 0;
    double merge_wait_ns_ = 0, merge_work_ns_ = 0, merge_check_ns_ = 0;
    long merge_calls_ = 0;
    std::atomic<bool> stop_{false};
    bool sleep_ = true;
    // begin()/finish(): threads 1.. do the work; the caller collects
    bool async_ = false, idle_ = false;
    std::atomic<int> done_{0};
    std::chrono::steady_clock::time_point begin_t_;
    std::atomic<int64_t> end_ns_{0};   // the last thread's end, steady_clock ns
    double async_busy_s_ = 0;
    long async_jobs_ = 0;
    const int32_t* ids_ = nullptr;
    const float* w_ = nullptr;
    float* y_ = nullptr;
    void prepare(const LaneLayer& layer, const float* z, int nt, const int32_t* ids, int top_k);
    void assign_teams(const LaneLayer& layer);
    // STRATA_DS41_LANE_NUMA: thread -> package; per call: threads per package, a thread's index in its team;
    // gate rows -> their package (cached)
    bool numa_ = false;
    int packages_ = 1;
    std::vector<int> pkg_of_, team_size_, team_index_;
    std::map<const void*, int> page_pkg_;
    // LANE_BANDS: thread -> NUMA node; per call: threads per node, a thread's index in its node's team
    bool bands_ = false;
    int nodes_ = 1;
    std::vector<int> node_of_, band_size_, band_index_;
    void band_teams();
    void collect(const int32_t* ids, const float* w, float* y);
    // MISS_READ: last call that used an expert (keyed by its gate rows), the reader pool and its queue
    struct MissReader;
    std::unique_ptr<MissReader> miss_;
    void miss_mark(const LaneLayer& layer);
    void miss_wait(const Group& g);
};

}  // namespace strata::ds41
