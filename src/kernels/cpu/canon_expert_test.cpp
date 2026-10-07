// canon_expert_test - the CPU half of the canonical expert arithmetic (canon_expert.hpp).
//   canon_quant_q8k   against ggml's Q8_K from_float: equal scales and codes except where ggml's build fused the
//                     multiply into the rounding add (reported, bounded), and a zero block is all zeros;
//   canon::exp        against std::exp to a few ulp, over the clamped range;
//   canon_route_sum   against the descending fma chain written out per element, zero weights skipped;
//   chain split       a chain continued from the partial sum over the last routes equals the full chain.
#include "strata/kernels/canon_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;

int main() {
    ggml_cpu_init();
    std::mt19937 rng(20261007);
    std::normal_distribution<float> normal(0.f, 2.f);
    bool pass = true;
    // Quantizer.
    const int n = 4096, trials = 200;
    size_t codes = 0, differing = 0, scale_differences = 0;
    std::vector<float> x(n);
    std::vector<block_q8_K> mine(n / 256), theirs(n / 256);
    for (int trial = 0; trial < trials; ++trial) {
        for (auto& v : x) v = normal(rng) * (trial % 7 == 0 ? 1e-3f : 1.f);
        if (trial % 5 == 0) std::fill_n(x.begin() + 512, 256, 0.f);
        std::memset(mine.data(), 0xa5, mine.size() * sizeof(block_q8_K));
        cpu::canon_quant_q8k(x.data(), mine.data(), n);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), theirs.data(), n);
        for (int b = 0; b < n / 256; ++b) {
            if (trial % 5 == 0 && b == 2) {
                const block_q8_K zero{};
                if (std::memcmp(&mine[b], &zero, sizeof(zero))) { std::printf("zero block is not all zeros\n"); pass = false; }
                continue;
            }
            if (std::memcmp(&mine[b].d, &theirs[b].d, 4)) ++scale_differences;
            int sum[16] = {};
            for (int i = 0; i < 256; ++i) {
                ++codes;
                differing += mine[b].qs[i] != theirs[b].qs[i];
                if (std::abs(mine[b].qs[i] - theirs[b].qs[i]) > 1) pass = false;
                sum[i / 16] += mine[b].qs[i];
            }
            for (int i = 0; i < 16; ++i) pass = pass && mine[b].bsums[i] == sum[i];
        }
    }
    std::printf("Q8_K: %zu of %zu codes differ from ggml by one step (ties under a fused multiply-add), %zu scale differences\n",
                differing, codes, scale_differences);
    if (scale_differences || differing > codes / 10000) pass = false;
    // exp and SwiGLU.
    double worst = 0;
    for (int i = 0; i <= 400000; ++i) {
        const float v = -90.f + i * (180.f / 400000);
        const float clamped = std::fmin(88.f, std::fmax(-87.f, v));
        const double reference = std::exp((double)clamped), got = canon::exp(v);
        worst = std::max(worst, std::fabs(got - reference) / reference);
    }
    std::printf("exp: largest relative error %.3g\n", worst);
    if (!(worst < 4e-7)) pass = false;
    const float s = cpu::canon_swiglu(1.5f, -2.f, 10.f), reference = 1.5f / (1.f + std::exp(-1.5f)) * -2.f;
    const float deep = cpu::canon_swiglu(-200.f, 1.f, 0.f);   // exp clamps: a tiny finite value, not NaN
    if (std::fabs(s - reference) > 1e-6f || cpu::canon_swiglu(50.f, 30.f, 10.f) != cpu::canon_swiglu(10.f, 10.f, 10.f) ||
        !std::isfinite(deep) || std::fabs(deep) > 1e-30f) pass = false;
    // Routed sum and the split chain.
    const int K = 8, H = 4096;
    std::vector<float> rows((size_t)K * H), out(H), partial(H);
    for (auto& v : rows) v = normal(rng);
    for (int trial = 0; trial < 50; ++trial) {
        float w[K];
        for (auto& v : w) v = rng() % 5 ? std::fabs(normal(rng)) : 0.f;
        cpu::canon_route_sum(rows.data(), w, out.data(), K, H, 0, H);
        const int split = trial % K;                      // the CPU chain covers routes after `split`
        float tail[K] = {};
        for (int k = split + 1; k < K; ++k) tail[k] = w[k];
        cpu::canon_route_sum(rows.data(), tail, partial.data(), K, H, 0, H);
        for (int r = 0; r < H; ++r) {
            float chain = 0, continued = partial[r];
            for (int k = K - 1; k >= 0; --k)
                if (w[k] != 0.f) chain = std::fma(w[k], rows[(size_t)k * H + r], chain);
            for (int k = split; k >= 0; --k)
                if (w[k] != 0.f) continued = std::fma(w[k], rows[(size_t)k * H + r], continued);
            if (std::memcmp(&chain, &out[r], 4) || std::memcmp(&chain, &continued, 4)) {
                std::printf("routed sum mismatch at trial %d row %d\n", trial, r);
                pass = false;
                break;
            }
        }
    }
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
