// Ordered row recurrence must preserve all speculative rollback prefixes.
#include "strata/kernels/glm_prefill.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
namespace k = strata::kernels;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    float *base = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) { ck(cudaMalloc(&base, (n + 32) * sizeof(float))); }
    ~Buffer() { cudaFree(base); }
    float *data() { return base + 16; }
    void put(const std::vector<float> &values, cudaStream_t stream) {
        if (values.size() != count) throw std::runtime_error("fixture size");
        ck(cudaMemsetAsync(base, 0xa5, (count + 32) * sizeof(float), stream));
        ck(cudaMemcpyAsync(data(), values.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    std::vector<unsigned char> get(cudaStream_t stream) {
        ck(cudaStreamSynchronize(stream));
        std::vector<unsigned char> out((count + 32) * sizeof(float));
        ck(cudaMemcpy(out.data(), base, out.size(), cudaMemcpyDeviceToHost));
        if (!std::all_of(out.begin(), out.begin() + 64, [](auto b) { return b == 0xa5; }) ||
            !std::all_of(out.end() - 64, out.end(), [](auto b) { return b == 0xa5; }))
            throw std::runtime_error("guard overwritten");
        for (size_t i = 0; i < count; ++i) {
            float value;
            std::memcpy(&value, out.data() + 64 + i * sizeof(float), sizeof(float));
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite result");
        }
        return out;
    }
};
int main() {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    try {
        ck(cudaStreamCreate(&stream));
        setenv("STRATA_GLM_KDA_ORDERED_ROWS", "1", 1);
        std::mt19937 rng(281);
        std::normal_distribution<float> random(0.f, .1f);
        size_t checks = 0, graph_checks = 0;
        for (int heads : {1, 3, 64}) for (int tokens : {1, 2, 3, 4, 7, 8, 16})
        for (int mode : {0, 1}) for (int retained : std::set<int>{0, 1, tokens - 1, tokens + 1}) {
            const size_t state_size = size_t(heads) * 128 * 128;
            const size_t n = size_t(tokens) * heads * 128, stride = state_size + 32;
            std::vector<float> q(n), key(n), value(n), decay(n), beta(size_t(tokens) * heads),
                initial(state_size), zero(n), sentinel(stride * std::max(1, retained), 19.f);
            for (auto &x : q) x = random(rng);
            for (auto &x : key) x = random(rng) * (mode ? 1e-8f : 1.f);
            for (auto &x : value) x = random(rng);
            for (size_t i = 0; i < decay.size(); ++i) decay[i] = mode ? (i % 2 ? -16.f : -1e-6f) : -.08f;
            for (size_t i = 0; i < beta.size(); ++i) beta[i] = mode ? (i % 2 ? -50.f : 50.f) : .2f;
            for (auto &x : initial) x = random(rng) * (mode ? 1000.f : .1f);
            Buffer dq(n), dk(n), dv(n), dg(n), db(beta.size()), state(state_size), out(n), snapshots(sentinel.size());
            dq.put(q, stream); dk.put(key, stream); dv.put(value, stream);
            dg.put(decay, stream); db.put(beta, stream);
            auto reset = [&] { state.put(initial, stream); out.put(zero, stream); snapshots.put(sentinel, stream); };
            auto run = [&](int parts) {
                k::glm_kda_chunk(state.data(), dq.data(), dk.data(), dv.data(), dg.data(), db.data(),
                    out.data(), heads, 128, tokens, stream, 128, parts, nullptr, snapshots.data(), stride, retained);
            };
            reset(); run(1);
            const auto expected_state = state.get(stream), expected_out = out.get(stream), expected_snapshots = snapshots.get(stream);
            for (int parts : {2, 4}) {
                reset(); run(parts);
                auto verify = [&] {
                    if (state.get(stream) != expected_state || out.get(stream) != expected_out ||
                        snapshots.get(stream) != expected_snapshots)
                        throw std::runtime_error("ordered recurrence changed state/output/prefix bits");
                    ++checks;
                };
                verify();
                // Replay fixed addresses after restoring initial state. Snapshot gaps and
                // unused slots must remain intact across every graph launch.
                if (heads == 3 && (tokens == 4 || tokens == 8) && retained == tokens - 1) {
                    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    run(parts);
                    ck(cudaStreamEndCapture(stream, &graph));
                    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        reset(); ck(cudaGraphLaunch(exec, stream)); verify(); ++graph_checks;
                    }
                    ck(cudaGraphExecDestroy(exec)); exec = nullptr;
                    ck(cudaGraphDestroy(graph)); graph = nullptr;
                }
                // Restore one retained prefix, then compare the next recurrence with
                // the serial kernel from exactly the same state and next input.
                if (retained > 0) {
                    const size_t kept = std::min(tokens, retained);
                    const float *prefix = snapshots.data() + (kept - 1) * stride;
                    ck(cudaMemcpyAsync(state.data(), prefix, state_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    k::glm_kda_chunk(state.data(), dq.data(), dk.data(), dv.data(), dg.data(), db.data(), out.data(),
                        heads, 128, 1, stream, 128, 1);
                    const auto next_state = state.get(stream), next_out = out.get(stream);
                    ck(cudaMemcpyAsync(state.data(), prefix, state_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    k::glm_kda_chunk(state.data(), dq.data(), dk.data(), dv.data(), dg.data(), db.data(), out.data(),
                        heads, 128, 1, stream, 128, parts);
                    if (state.get(stream) != next_state || out.get(stream) != next_out)
                        throw std::runtime_error("ordered recurrence changed rollback next step");
                    ++checks;
                }
            }
        }
        // Prepared and unordered paths still reject snapshots before launch.
        Buffer dummy(128 * 128);
        for (int parts : {2, 4, 8}) for (bool prepared : {false, true}) {
            setenv("STRATA_GLM_KDA_ORDERED_ROWS", "0", 1);
            bool rejected = false;
            try {
                k::glm_kda_chunk(dummy.data(), dummy.data(), dummy.data(), dummy.data(), dummy.data(),
                    dummy.data(), dummy.data(), 1, 128, 1, stream, 128, parts,
                    prepared ? dummy.data() : nullptr, dummy.data(), 128 * 128, 1);
            } catch (const std::invalid_argument &) { rejected = true; }
            if (!rejected) throw std::runtime_error("unsupported snapshot path accepted");
        }
        ck(cudaStreamDestroy(stream)); stream = nullptr;
        std::cout << "KDA_ORDERED_HISTORY passed=1 bitwise_checks=" << checks
                  << " graph_checks=" << graph_checks << " guards=unchanged\n";
        return 0;
    } catch (const std::exception &e) {
        if (stream) cudaStreamSynchronize(stream);
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
