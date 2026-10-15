#pragma once
#include <cstdint>
#include <future>
#include <exception>
#include <vector>
#include <utility>

namespace strata::spec {
struct LayerPipelineStats { uint64_t proposed = 0, accepted = 0; };
// One unconfirmed input can advance the head. The tail sees confirmed inputs
// only. A backend owns two generation-tagged transport slots and recurrent
// checkpoints; every exit drains outstanding work before restoring the head.
// emit returns true on count/EOS; it may throw on cancellation.
template<class Backend, class Emit>
LayerPipelineStats run_layer_pipeline(Backend &b, int current, Emit emit) {
    LayerPipelineStats stats;
    uint64_t generation = 1;
    int slot = 0, draft = -1;
    auto first = b.start_head(current, slot, generation);
    try { draft = b.draft(current); }
    catch (...) { if (first.valid()) first.wait(); throw; }
    first.get();
    for (;;) {
        b.save_head();
        const int next_slot = slot ^ 1;
        const uint64_t next_generation = generation + 1;
        auto ahead = b.start_head(draft, next_slot, next_generation);
        try {
            const int actual = b.confirm(current, slot, generation);
            ++stats.proposed; stats.accepted += actual == draft;
            if (emit(actual)) {
                ahead.get();
                b.restore_head();
                b.release(next_slot, next_generation);
                return stats;
            }
            bool boundary=false;
            if constexpr(requires { b.window_due(stats.proposed); }) boundary=b.window_due(stats.proposed);
            if(boundary) {
                // Bring both stages to the same confirmed frontier before a
                // policy change, even if this proposal happened to match.
                ahead.get();b.restore_head();b.release(next_slot,next_generation);
                if constexpr(requires { b.rebalance(); }) b.rebalance();
                ahead=b.start_head(actual,next_slot,next_generation);
            }
            int next_draft = -1;
            // The confirmed tail hidden is ready even if the head is still
            // computing. Submit the next draft before joining that head.
            auto drafting = b.start_draft(actual, next_draft);
            try {
                ahead.get();
                if (!boundary && actual != draft) {
                    b.restore_head();
                    b.release(next_slot, next_generation);
                    b.start_head(actual, next_slot, next_generation).get();
                }
            } catch (...) { if (drafting.valid()) drafting.wait(); throw; }
            drafting.get();
            current = actual; draft = next_draft;
            slot = next_slot; generation = next_generation;
        } catch (...) {
            if (ahead.valid()) ahead.wait();
            b.restore_head();
            throw;
        }
    }
}
// Chunks advance in waves. A physical device owns one ordered queue; the
// diagonal gives independent stages concurrent work without reordering a
// layer's tokens. complete() publishes/recycles slots only after all jobs in
// the wave have drained, including on a failed producer or consumer.
template<class Backend>
void run_layer_prefill(Backend &b, size_t panels, size_t stages) {
    if (!panels || !stages) return;
    struct Job { size_t stage, panel; std::future<void> future; };
    for(size_t wave=0;wave<panels+stages-1;++wave) {
        std::vector<Job> jobs;
        jobs.reserve(stages);
        try {
            for(size_t stage=0;stage<stages;++stage) {
                if(wave<stage || wave-stage>=panels) continue;
                const size_t panel=wave-stage;
                jobs.push_back({stage,panel,b.start(stage,panel)});
            }
        } catch (...) {
            for(auto &job:jobs)if(job.future.valid())job.future.wait();
            throw;
        }
        std::exception_ptr error;
        for(auto &job:jobs) {
            try {job.future.get();}catch(...){if(!error)error=std::current_exception();}
        }
        if(error)std::rethrow_exception(error);
        for(auto &job:jobs)b.complete(job.stage,job.panel);
    }
}
} // namespace strata::spec
