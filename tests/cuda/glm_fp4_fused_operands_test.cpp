#include "strata/kernels/glm_fp4.hpp"
#include "strata/host/glm_fp4.h"
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
    explicit Buffer(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 1))); }
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
        bool bench = argc > 1 && std::string(argv[1]) == "--bench";
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        int checks = 0, graphs = 0;

        for (auto [N, K] : {std::pair{128, 64}, {128, 512}, {2048, 4096}, {4096, 2048}}) {
            int C = K / 16;
            size_t count = size_t(N) * K;
            std::vector<uint8_t> raw(count / 2), bases(N), codes,
                logical(size_t(N) * C), decoded(logical.size());
            std::vector<uint16_t> offsets, exceptions;
            std::vector<uint32_t> pages;
            for (size_t i = 0; i < raw.size(); ++i)
                raw[i] = uint8_t(i * 53 + 17);
            for (int row = 0; row < N; ++row)
                bases[row] = uint8_t(40 + row % 8);
            int tile = 0;
            for (int r0 = 0; r0 < N; r0 += 128)
                for (int c0 = 0; c0 < C; c0 += 4, ++tile) {
                    if (tile % 32 == 0) {
                        pages.push_back(exceptions.size() * 2);
                        offsets.push_back(0);
                    }
                    for (int col = c0; col < c0 + 4; col += 2)
                        for (int row = r0; row < r0 + 128; row += 32) {
                            for (int rr = 0; rr < 32; ++rr) {
                                unsigned lo = (row + rr + col) % 16, hi = (row + rr + col + 1) % 16;
                                codes.push_back(lo | (hi << 4));
                                logical[(row + rr) * C + col] = bases[row + rr] + lo;
                                logical[(row + rr) * C + col + 1] = bases[row + rr] + hi;
                            }
                            exceptions.push_back(0 | (81 << 8));
                            logical[row * C + col] = 81;
                            exceptions.push_back(63 | (11 << 8));
                            logical[(row + 31) * C + col + 1] = 11;
                            offsets.push_back(exceptions.size() - pages.back() / 2);
                        }
                }
            if (strata_fp4_csf32_decode(N, C, bases.data(), bases.size(), codes.data(), codes.size(),
                    offsets.data(), offsets.size(), pages.data(), pages.size(), exceptions.data(),
                    exceptions.size(), decoded.data(), decoded.size()) || decoded != logical)
                throw std::runtime_error("independent CSF fixture disagrees with C11");
            Buffer w(raw.size()), b(bases.size()), c(codes.size()), o(offsets.size() * 2),
                p(pages.size() * 4), e(exceptions.size() * 2), packed(raw.size()),
                scales(logical.size()), flag(4), output((count + 64) * 2);
            auto copy = [&](void* to, const void* from, size_t bytes) {
                ck(cudaMemcpyAsync(to, from, bytes, cudaMemcpyHostToDevice, s));
            };
            copy(w.p, raw.data(), raw.size());
            copy(b.p, bases.data(), bases.size());
            copy(c.p, codes.data(), codes.size());
            copy(o.p, offsets.data(), offsets.size() * 2);
            copy(p.p, pages.data(), pages.size() * 4);
            copy(e.p, exceptions.data(), exceptions.size() * 2);
            auto* y = static_cast<uint16_t*>(output.p) + 32;
            auto run = [&](int mode, float global) {
                if (mode == 2)
                    k::glm_fp4_csf32_dequant_f16(static_cast<uint8_t*>(w.p), static_cast<uint8_t*>(b.p),
                        static_cast<uint8_t*>(c.p), static_cast<uint16_t*>(o.p), static_cast<uint32_t*>(p.p),
                        static_cast<uint16_t*>(e.p), global, y, static_cast<int*>(flag.p), N, K, s);
                else {
                    if (mode == 0)
                        k::glm_fp4_prepack(static_cast<uint8_t*>(w.p), nullptr,
                                           static_cast<uint8_t*>(packed.p), nullptr, N, K, s);
                    k::glm_fp4_csf32_scales(static_cast<uint8_t*>(b.p), static_cast<uint8_t*>(c.p),
                        static_cast<uint16_t*>(o.p), static_cast<uint32_t*>(p.p), static_cast<uint16_t*>(e.p),
                        static_cast<uint8_t*>(scales.p), static_cast<int*>(flag.p), N, K, s);
                    if (mode == 0)
                        k::glm_fp4_dequant_f16(k::GlmPackedFormat::Nvfp4, packed.p, scales.p,
                                              global, y, N, K, s, true);
                    else
                        k::glm_fp4_raw_dequant_f16(static_cast<uint8_t*>(w.p),
                            static_cast<uint8_t*>(scales.p), global, y, N, K, true, s);
                }
                ck(cudaGetLastError());
            };
            auto reset = [&] { ck(cudaMemsetAsync(output.p, 0x5a, (count + 64) * 2, s)); };
            auto read = [&](bool valid = true) {
                ck(cudaStreamSynchronize(s));
                int bad = -1;
                ck(cudaMemcpy(&bad, flag.p, 4, cudaMemcpyDeviceToHost));
                if ((bad == 0) != valid)
                    throw std::runtime_error("CSF validity flag differs");
                std::vector<uint16_t> got(count + 64);
                ck(cudaMemcpy(got.data(), output.p, got.size() * 2, cudaMemcpyDeviceToHost));
                for (int j = 0; j < 32; ++j)
                    if (got[j] != 0x5a5a || got[count + 32 + j] != 0x5a5a)
                        throw std::runtime_error("operand guard overwritten");
                return got;
            };

            for (float global : {0.f, .00061f, -.00037f, .002f, 1e-8f}) {
                reset();
                run(0, global);
                auto expected = read();
                for (int mode : {1, 2}) {
                    reset();
                    run(mode, global);
                    if (read() != expected)
                        throw std::runtime_error("fused operands changed FP16 bits");
                    ++checks;
                }
            }

            ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            run(2, .00061f);
            ck(cudaStreamEndCapture(s, &graph));
            ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            for (int pass = 0; pass < 2; ++pass) {
                for (auto& v : raw)
                    v = uint8_t(v + 37);
                for (auto& v : exceptions)
                    v = uint16_t((v & 255) | ((((v >> 8) + 19) % 127) << 8));
                copy(w.p, raw.data(), raw.size());
                copy(e.p, exceptions.data(), exceptions.size() * 2);
                reset();
                run(0, .00061f);
                auto expected = read();
                reset();
                ck(cudaGraphLaunch(exec, s));
                if (read() != expected)
                    throw std::runtime_error("fused graph changed operand bits");
                ++graphs;
            }
            ck(cudaGraphExecDestroy(exec));
            exec = nullptr;
            ck(cudaGraphDestroy(graph));
            graph = nullptr;

            auto saved = exceptions[0];
            exceptions[0] = 0 | (127 << 8);
            copy(e.p, exceptions.data(), exceptions.size() * 2);
            reset();
            run(2, .00061f);
            read(false);
            exceptions[0] = 200 | (81 << 8);
            copy(e.p, exceptions.data(), exceptions.size() * 2);
            reset();
            run(2, .00061f);
            read(false);
            exceptions[0] = saved;
            copy(e.p, exceptions.data(), exceptions.size() * 2);
            reset();
            run(2, .00061f);
            read();

            if (bench && N >= 2048) {
                for (int i = 0; i < 32; ++i)
                    for (int mode = 0; mode < 3; ++mode)
                        run(mode, .00061f);
                ck(cudaStreamSynchronize(s));
                for (int sample = 0; sample < 5; ++sample)
                    for (int at = 0; at < 3; ++at) {
                        int mode = (at + sample) % 3;
                        cudaEvent_t first, last;
                        ck(cudaEventCreate(&first));
                        ck(cudaEventCreate(&last));
                        ck(cudaEventRecord(first, s));
                        for (int i = 0; i < 32; ++i)
                            run(mode, .00061f);
                        ck(cudaEventRecord(last, s));
                        ck(cudaEventSynchronize(last));
                        float ms;
                        ck(cudaEventElapsedTime(&ms, first, last));
                        ck(cudaEventDestroy(first));
                        ck(cudaEventDestroy(last));
                        std::cout << "FUSED_CSF_BENCH N=" << N << " K=" << K
                                  << " mode=" << mode << " sample=" << sample
                                  << " us=" << ms * 1000 / 32 << '\n';
                    }
            }
        }

        bool rejected = false;
        try {
            k::glm_fp4_csf32_dequant_f16(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       1, nullptr, nullptr, 32, 64, s);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("invalid fused shape accepted");
        ck(cudaStreamDestroy(s));
        s = nullptr;
        std::cout << "FUSED_CSF_OPERANDS passed=1 bitwise_cases=" << checks << " graph_cases=" << graphs
                  << " guards=unchanged invalid_scales_and_slots=rejected sm=" << prop.major << prop.minor << '\n';
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
