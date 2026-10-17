// tests/core/muse_expert_lane_test.cpp - strata/muse/expert_lane.hpp against a plain single-thread loop (the one in
// tools/muse_gpu_trunk_check.cpp), on synthetic IQ1_S experts of Muse Spark's shape (latent 4096, ff 4096): every
// thread count gives the same bits, for 1, 2 and 5 tokens with shared experts and for 12 tokens on one expert (groups
// over MAXT are cut into chunks). --bench T: ms per decode layer (1 token, top-8) on T threads.
#include "strata/muse/expert_lane.hpp"
#include "strata/muse/reference.hpp"

#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace muse = strata::muse;

int main(int argc, char** argv) {
    const int L = 4096, FF = 4096, E = 12, K = 8;
    const float eps = 1e-8f;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, L, FF, f, err)) { std::printf("%s\n", err.c_str()); return 1; }
    f.iq1s_rows = true;
    std::mt19937 rng(20261017);
    auto blob = [&](size_t rows, size_t row_bytes) {   // random grid codes, small finite fp16 block scales
        std::vector<uint8_t> b(rows * row_bytes);
        for (auto& v : b) v = (uint8_t) rng();
        for (size_t o = 0; o < b.size(); o += 50) {
            const uint16_t d = ggml_fp32_to_fp16(0.01f + 0.02f * (rng() % 1000) / 1000.f);
            std::memcpy(&b[o], &d, 2);
        }
        return b;
    };
    const auto gate = blob((size_t) E * FF, f.gu_row), up = blob((size_t) E * FF, f.gu_row), down = blob((size_t) E * L, f.d_row);
    std::vector<float> post(L);
    std::normal_distribution<float> nd;
    for (auto& v : post) v = 1.f + 0.1f * nd(rng);
    muse::LaneLayer layer{gate.data(), up.data(), down.data(), post.data()};

    // the reference: one token and one expert at a time, as the parity tool does
    auto reference = [&](const float* z, int nt, const int32_t* ids, const float* w, float* y) {
        std::vector<uint8_t> act(cpu::kNativeActBytes), hq(cpu::kNativeHBytes);
        std::vector<float> ffv(FF), outv(L);
        for (int t = 0; t < nt; ++t) {
            float* yt = y + (size_t) t * L;
            std::fill(yt, yt + L, 0.f);
            cpu::native_quant_act(f, z + (size_t) t * L, act.data());
            const void* ap = act.data();
            const void* hp = hq.data();
            float* ffp = ffv.data();
            float* op = outv.data();
            for (int k = 0; k < K; ++k) {
                const size_t e = (size_t) ids[t * K + k];
                const uint8_t* g = gate.data() + e * f.gu_row * FF;
                cpu::native_gu_rows(f, g, &ap, 1, &ffp, 0, FF, up.data() + e * f.gu_row * FF);
                cpu::native_quant_h(f, ffv.data(), hq.data());
                cpu::native_down_rows(f, g, &hp, 1, &op, 0, L, down.data() + e * f.d_row * L);
                for (int j = 0; j < L; ++j) outv[j] *= post[j];
                muse::ref::rms_norm(outv.data(), L, eps);
                for (int j = 0; j < L; ++j) yt[j] += w[t * K + k] * outv[j];
            }
        }
    };

    if (argc > 2 && std::string(argv[1]) == "--bench") {
        // --bench T [numa]: numa pins the threads by NUMA node and gives them their rows on their node (adopt)
        const bool numa = argc > 3 && std::string(argv[3]) == "numa";
        muse::ExpertLane lane(f, L, FF, eps, std::atoi(argv[2]), numa);
        const muse::LaneLayer bench_layer = numa ? lane.adopt(layer, E) : layer;
        std::vector<float> z(L), y(L), w(K, 0.125f);
        std::vector<int32_t> ids(K);
        for (auto& v : z) v = nd(rng);
        const int reps = 200;
        double total = 0;
        for (int r = 0; r < reps + 10; ++r) {
            for (int k = 0; k < K; ++k) ids[k] = (r + k) % E;
            const auto t0 = std::chrono::steady_clock::now();
            lane.run(bench_layer, z.data(), 1, ids.data(), w.data(), K, y.data());
            if (r >= 10) total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        // a layer reads 8 experts of 3 x 4096 x 4096 IQ1_S weights
        const double bytes = 8.0 * 3 * FF * f.gu_row;
        std::printf("%d threads%s: %.3f ms per decode layer (top-8), %.1f GB/s of expert rows, x62 layers = %.1f ms per token\n",
                    lane.threads(), numa ? " (NUMA pinned, rows adopted)" : "", total / reps, bytes / (total / reps * 1e-3) / 1e9,
                    62 * total / reps);
        return 0;
    }

    int fails = 0;
    struct Case { int nt; bool one_expert; };
    for (const Case c : {Case{1, false}, Case{2, false}, Case{5, false}, Case{12, true}}) {
        std::vector<float> z((size_t) c.nt * L), w((size_t) c.nt * K);
        std::vector<int32_t> ids((size_t) c.nt * K);
        for (auto& v : z) v = nd(rng);
        for (int t = 0; t < c.nt; ++t) {
            float s = 0;
            for (int k = 0; k < K; ++k) {
                ids[t * K + k] = c.one_expert ? (k == 0 ? 3 : (3 + k) % E) : (int) ((t * 3 + k * 5) % E);
                s += w[t * K + k] = 0.1f + (float) (rng() % 100) / 100.f;
            }
            for (int k = 0; k < K; ++k) w[t * K + k] /= s;
        }
        std::vector<float> want((size_t) c.nt * L);
        reference(z.data(), c.nt, ids.data(), w.data(), want.data());
        for (int threads : {1, 4, 15})
            for (bool numa : {false, true}) {
                muse::ExpertLane lane(f, L, FF, eps, threads, numa);
                const muse::LaneLayer use = numa ? lane.adopt(layer, E) : layer;
                std::vector<float> got((size_t) c.nt * L);
                lane.run(use, z.data(), c.nt, ids.data(), w.data(), K, got.data());
                lane.run(use, z.data(), c.nt, ids.data(), w.data(), K, got.data());   // reuse across calls
                const bool same = !std::memcmp(got.data(), want.data(), got.size() * 4);
                std::printf("%2d tokens%s, %2d threads%s: %s\n", c.nt, c.one_expert ? " (one shared expert)" : "", threads,
                            numa ? " NUMA + adopted rows" : "", same ? "bit-identical" : "FAIL");
                fails += !same;
            }
    }
    std::printf("muse_expert_lane_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
