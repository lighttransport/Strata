#include "strata/kernels/glm_fp4.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace k = strata::kernels;

static void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}

struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc(&p, n)); }
    ~Buffer() { cudaFree(p); }
};

int main(int argc, char** argv) {
    cudaStream_t s = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 7)
            return 77;
        const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        int cases = 0, graphs = 0;

        for (const auto [N, K] : {std::pair{32, 64}, {128, 512}, {2048, 768}, {2048, 4096},
                                  {4096, 2048}, {4096, 4096}, {8192, 1024}}) {
            const size_t count = size_t(N) * K;
            std::vector<uint8_t> codes(count / 2), scales(count / 16);
            for (size_t i = 0; i < codes.size(); ++i)
                codes[i] = uint8_t(i * 53 + 17);
            for (size_t i = 0; i < scales.size(); ++i)
                scales[i] = uint8_t((i * 7 + 19) % 127);
            Buffer weights(codes.size()), factors(scales.size()), packed(codes.size()),
                ps(scales.size()), output((count + 64) * 2);
            ck(cudaMemcpyAsync(weights.p, codes.data(), codes.size(), cudaMemcpyHostToDevice, s));
            ck(cudaMemcpyAsync(factors.p, scales.data(), scales.size(), cudaMemcpyHostToDevice, s));
            auto* y = static_cast<uint16_t*>(output.p) + 32;
            auto reset = [&] { ck(cudaMemsetAsync(output.p, 0x5a, (count + 64) * 2, s)); };

            for (bool packed_scales : {false, true}) {
                auto prep_scales = [&] {
                    k::glm_fp4_prepack(static_cast<uint8_t*>(weights.p), static_cast<uint8_t*>(factors.p),
                                       static_cast<uint8_t*>(packed.p), static_cast<uint8_t*>(ps.p), N, K, s);
                };
                prep_scales();
                auto run = [&](bool active, float global) {
                    if (active)
                        k::glm_fp4_raw_dequant_f16(static_cast<uint8_t*>(weights.p),
                            static_cast<uint8_t*>(packed_scales ? ps.p : factors.p),
                            global, y, N, K, packed_scales, s);
                    else {
                        k::glm_fp4_prepack(static_cast<uint8_t*>(weights.p),
                            packed_scales ? nullptr : static_cast<uint8_t*>(factors.p),
                            static_cast<uint8_t*>(packed.p),
                            packed_scales ? nullptr : static_cast<uint8_t*>(ps.p), N, K, s);
                        k::glm_fp4_dequant_f16(k::GlmPackedFormat::Nvfp4, packed.p, ps.p,
                                               global, y, N, K, s, true);
                    }
                    ck(cudaGetLastError());
                };
                auto read = [&] {
                    ck(cudaStreamSynchronize(s));
                    std::vector<uint16_t> bytes(count + 64);
                    ck(cudaMemcpy(bytes.data(), output.p, bytes.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < 32; ++i)
                        if (bytes[i] != 0x5a5a || bytes[count + 32 + i] != 0x5a5a)
                            throw std::runtime_error("output guard overwritten");
                    return bytes;
                };

                for (float global : {0.f, .00061f, -.00037f, .002f, 1e-8f}) {
                    reset();
                    run(false, global);
                    const auto expected = read();
                    reset();
                    run(true, global);
                    if (read() != expected)
                        throw std::runtime_error("canonical operands changed FP16 bits");
                    ++cases;
                }

                ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                run(true, .00061f);
                ck(cudaStreamEndCapture(s, &graph));
                ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                for (int pass = 0; pass < 2; ++pass) {
                    for (auto& v : codes)
                        v = uint8_t(v + 37);
                    for (auto& v : scales)
                        v = uint8_t((v + 19) % 127);
                    ck(cudaMemcpyAsync(weights.p, codes.data(), codes.size(), cudaMemcpyHostToDevice, s));
                    ck(cudaMemcpyAsync(factors.p, scales.data(), scales.size(), cudaMemcpyHostToDevice, s));
                    prep_scales();
                    reset();
                    run(false, .00061f);
                    const auto expected = read();
                    reset();
                    ck(cudaGraphLaunch(exec, s));
                    if (read() != expected)
                        throw std::runtime_error("canonical graph replay changed FP16 bits");
                    ++graphs;
                }
                ck(cudaGraphExecDestroy(exec));
                exec = nullptr;
                ck(cudaGraphDestroy(graph));
                graph = nullptr;

                if (bench && N >= 2048 && N * K == 2048 * 4096) {
                    for (int i = 0; i < 32; ++i) {
                        run(false, .00061f);
                        run(true, .00061f);
                    }
                    ck(cudaStreamSynchronize(s));
                    for (int sample = 0; sample < 5; ++sample)
                        for (int at = 0; at < 2; ++at) {
                            const bool active = bool(at ^ (sample & 1));
                            cudaEvent_t first, last;
                            ck(cudaEventCreate(&first));
                            ck(cudaEventCreate(&last));
                            ck(cudaEventRecord(first, s));
                            for (int i = 0; i < 32; ++i)
                                run(active, .00061f);
                            ck(cudaEventRecord(last, s));
                            ck(cudaEventSynchronize(last));
                            float ms;
                            ck(cudaEventElapsedTime(&ms, first, last));
                            ck(cudaEventDestroy(first));
                            ck(cudaEventDestroy(last));
                            std::cout << "RAW_OPERANDS_BENCH N=" << N << " K=" << K
                                      << " packed_scales=" << packed_scales << " active=" << active
                                      << " sample=" << sample << " us=" << ms * 1000 / 32 << '\n';
                        }
                }
            }
        }

        for (const auto [N, K] : {std::pair{40, 64}, {32, 48}, {0, 64}, {32, 0}}) {
            bool rejected = false;
            try {
                k::glm_fp4_raw_dequant_f16(nullptr, nullptr, 1, nullptr, N, K, false, s);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("invalid operand extent accepted");
        }
        ck(cudaStreamDestroy(s));
        s = nullptr;
        std::cout << "RAW_OPERANDS passed=1 bitwise_cases=" << cases << " graph_cases=" << graphs
                  << " guards=unchanged sm=" << prop.major << prop.minor << '\n';
        return 0;
    } catch (const std::exception& e) {
        if (s)
            cudaStreamSynchronize(s);
        if (exec)
            cudaGraphExecDestroy(exec);
        if (graph)
            cudaGraphDestroy(graph);
        if (s)
            cudaStreamDestroy(s);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
