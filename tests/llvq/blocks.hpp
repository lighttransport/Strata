// The test blocks of tools/quant_study/llvq_layer's llvq_dump (src/dump.rs), regenerated in C++: 24 SplitMix64
// Gaussians scaled by 10^u (u uniform in [-3, 3]); by i % 97: zero, x1e-300, x1e250, sparse, and two thirds of the
// rest rounded to a quarter or a whole step of the scale (exact ties).
#pragma once
#include "strata/llvq/quantize.hpp"
#include <cmath>
#include <vector>

namespace strata::llvq::test {

inline std::vector<double> dump_blocks(size_t n, uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<double> out(n * 24);
    for (size_t i = 0; i < n; ++i) {
        double* x = &out[i * 24];
        double scale = std::pow(10.0, rng.next_f64() * 6.0 - 3.0);
        for (int j = 0; j < 24; ++j) x[j] = rng.next_gaussian() * scale;
        size_t k = i % 97;
        if (k == 0) for (int j = 0; j < 24; ++j) x[j] = 0.0;
        else if (k == 1) for (int j = 0; j < 24; ++j) x[j] *= 1e-300;
        else if (k == 2) for (int j = 0; j < 24; ++j) x[j] *= 1e250;
        else if (k == 3) { for (int j = 0; j < 24; ++j) if (j % 5 != 0) x[j] = 0.0; }
        else if (k % 3 == 0) for (int j = 0; j < 24; ++j) x[j] = std::round(x[j] / scale * 4.0) / 4.0 * scale;
        else if (k % 3 == 1) for (int j = 0; j < 24; ++j) x[j] = std::round(x[j] / scale);
    }
    return out;
}

// FNV-1a 64 over (word, t) per block, as the golden constants were computed from llvq_dump's output.
struct Fnv {
    uint64_t h = 0xcbf29ce484222325ull;
    void bytes(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
    }
};
// llvq_dump 3000 0x5eed (llvq ddecad31).
constexpr size_t GOLDEN_N = 3000;
constexpr uint64_t GOLDEN_SEED = 0x5eed, GOLDEN_FNV = 0xd476ce307bd71cfaull;

}  // namespace strata::llvq::test
