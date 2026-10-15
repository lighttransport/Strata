// Compact greedy readback: CPU tie/finite semantics, bounds, scratch reuse and graph replay.
#include "strata/kernels/glm.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
namespace k = strata::kernels;
static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    void *p = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&p, bytes)); }
    ~Buffer() { const auto status = cudaFree(p); (void)status; }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};
int main() {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    try {
        check(cudaStreamCreate(&stream));
        std::mt19937 rng(2718);
        int cases = 0, graph_replays = 0;
        constexpr int32_t canary = 0x12345678;
        for (int n : {1,31,32,33,255,256,257,4095,4096,4097,8193,154880,154881}) {
            const auto scratch_bytes = k::glm_greedy_pick_scratch_bytes(n);
            Buffer input((size_t(n) + 8) * sizeof(float));
            Buffer scratch(scratch_bytes + 64);
            Buffer output(6 * sizeof(int32_t));
            auto *logits = static_cast<float *>(input.p) + 4;
            auto *work = static_cast<char *>(scratch.p) + 32;
            auto *result = reinterpret_cast<k::GlmGreedyPick *>(static_cast<int32_t *>(output.p) + 2);
            auto compare = [&](const std::vector<float> &values) {
                int32_t picked[6];
                check(cudaMemcpy(picked, output.p, sizeof(picked), cudaMemcpyDeviceToHost));
                const bool bad = std::any_of(values.begin(), values.end(), [](float x) { return !std::isfinite(x); });
                if (picked[3] != int(bad)) throw std::runtime_error("nonfinite flag mismatch");
                if (!bad && picked[2] != std::max_element(values.begin(), values.end()) - values.begin())
                    throw std::runtime_error("greedy token mismatch");
                for (int i : {0,1,4,5}) if (picked[i] != canary) throw std::runtime_error("result guard overwritten");
                std::vector<unsigned char> bytes(scratch_bytes + 64);
                check(cudaMemcpy(bytes.data(), scratch.p, bytes.size(), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < 32; ++i)
                    if (bytes[i] != 0xa5 || bytes[scratch_bytes + 32 + i] != 0xa5)
                        throw std::runtime_error("scratch guard overwritten");
                std::vector<float> roundtrip(size_t(n) + 8);
                check(cudaMemcpy(roundtrip.data(), input.p, roundtrip.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (std::memcmp(roundtrip.data() + 4, values.data(), size_t(n) * sizeof(float)))
                    throw std::runtime_error("input modified");
                for (int i = 0; i < 4; ++i)
                    if (roundtrip[i] != 1234.5f || roundtrip[size_t(n) + 4 + i] != 1234.5f)
                        throw std::runtime_error("input guard overwritten");
            };
            auto upload = [&](const std::vector<float> &values) {
                std::vector<float> guarded(size_t(n) + 8, 1234.5f);
                std::copy(values.begin(), values.end(), guarded.begin() + 4);
                check(cudaMemcpy(input.p, guarded.data(), guarded.size() * sizeof(float), cudaMemcpyHostToDevice));
                int32_t guards[6] = {canary,canary,canary,canary,canary,canary};
                check(cudaMemcpy(output.p, guards, sizeof(guards), cudaMemcpyHostToDevice));
                check(cudaMemset(scratch.p, 0xa5, scratch_bytes + 64));
            };
            for (int pattern = 0; pattern < 12; ++pattern) {
                std::vector<float> values(n);
                for (auto &v : values) v = float(int(rng() % 257) - 128);
                if (pattern == 1) std::fill(values.begin(), values.end(), -42.f);
                if (pattern == 2) { std::fill(values.begin(), values.end(), -0.f); values.back() = 0.f; }
                if (pattern == 3) { values[0] = std::numeric_limits<float>::max(); values.back() = values[0]; }
                if (pattern == 4) { std::fill(values.begin(), values.end(), -std::numeric_limits<float>::max()); values.back() = -1.f; }
                if (pattern == 5) values[n / 2] = std::numeric_limits<float>::quiet_NaN();
                if (pattern == 6) values.back() = std::numeric_limits<float>::infinity();
                if (pattern == 7) values[0] = -std::numeric_limits<float>::infinity();
                if (pattern == 8) std::fill(values.begin(), values.end(), std::numeric_limits<float>::quiet_NaN());
                if (pattern == 9) { std::fill(values.begin(), values.end(), std::numeric_limits<float>::denorm_min()); values.back() = 3.f * std::numeric_limits<float>::denorm_min(); }
                if (pattern == 10) { std::fill(values.begin(), values.end(), -3.f * std::numeric_limits<float>::denorm_min()); values.back() = -std::numeric_limits<float>::denorm_min(); }
                if (pattern == 11) for (auto &value : values) {
                    uint32_t bits = rng();
                    if ((bits & 0x7f800000u) == 0x7f800000u) bits ^= 0x00800000u;
                    std::memcpy(&value, &bits, sizeof(value));
                }
                upload(values);
                k::glm_greedy_pick(logits,n,work,result,stream);
                check(cudaStreamSynchronize(stream));
                compare(values);
                ++cases;
            }
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            k::glm_greedy_pick(logits,n,work,result,stream);
            check(cudaStreamEndCapture(stream,&graph));
            check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            for (int repeat = 0; repeat < 3; ++repeat) {
                std::vector<float> values(n,-1.f);
                values[(size_t(repeat) * 4096) % n] = 7.f;
                upload(values);
                check(cudaGraphLaunch(executable,stream));
                check(cudaStreamSynchronize(stream));
                compare(values);
                ++graph_replays;
            }
            check(cudaGraphExecDestroy(executable)); executable = nullptr;
            check(cudaGraphDestroy(graph)); graph = nullptr;
        }
        for (int n : {0,-1}) {
            bool rejected = false;
            try { (void)k::glm_greedy_pick_scratch_bytes(n); } catch (const std::invalid_argument &) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid vocabulary accepted");
        }
        bool rejected = false;
        try { k::glm_greedy_pick(nullptr,1,nullptr,nullptr,stream); } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("null buffers accepted");
        check(cudaStreamDestroy(stream)); stream = nullptr;
        std::cout << "PASS greedy_pick cases=" << cases << " graph_replays=" << graph_replays
                  << " tie_finite_guards_scratch_reuse=1\n";
        return 0;
    } catch (const std::exception &e) {
        if (executable) { const auto status = cudaGraphExecDestroy(executable); (void)status; }
        if (graph) { const auto status = cudaGraphDestroy(graph); (void)status; }
        if (stream) { const auto status = cudaStreamDestroy(stream); (void)status; }
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
