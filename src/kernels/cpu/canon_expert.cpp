// Built with -ffp-contract=off: every product below is rounded before it is added (see canon_expert.hpp).
#include "strata/kernels/canon_expert.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace strata::kernels::cpu {

namespace {
struct GateStats {
    static constexpr int kLayers = 64, kBins = 96;
    std::string path;
    std::atomic<uint64_t> counts[kLayers][kBins]{};
    std::atomic<int> layer{-1};
    ~GateStats() {
        if (path.empty()) return;
        std::ofstream out(path);
        out << "{\"bin0_log2\":-16,\"bins_per_octave\":4,\"layers\":{";
        bool first = true;
        for (int l = 0; l < kLayers; ++l) {
            uint64_t total = 0;
            for (const auto& c : counts[l]) total += c.load();
            if (!total) continue;
            out << (first ? "" : ",") << '"' << l << "\":[";
            for (int b = 0; b < kBins; ++b) out << (b ? "," : "") << counts[l][b].load();
            out << ']';
            first = false;
        }
        out << "}}\n";
    }
    void add(float magnitude) {
        const int l = layer.load(std::memory_order_relaxed);
        if (l < 0 || l >= kLayers) return;
        const int bin = magnitude > 0 ? (int)std::floor((std::log2(magnitude) + 16.f) * 4.f) : 0;
        counts[l][bin < 0 ? 0 : bin >= kBins ? kBins - 1 : bin].fetch_add(1, std::memory_order_relaxed);
    }
};
GateStats* gate_stats() {
    static GateStats* stats = [] {
        const char* path = std::getenv("STRATA_Q23_GATE_STATS");
        static GateStats instance;
        if (path && *path) instance.path = path;
        return instance.path.empty() ? nullptr : &instance;
    }();
    return stats;
}
}  // namespace

void canon_gate_stats_layer(int layer) {
    if (auto* stats = gate_stats()) stats->layer.store(layer, std::memory_order_relaxed);
}
float canon_swiglu(float gate, float up, float limit, float skip) {
    if (auto* stats = gate_stats()) stats->add(std::fabs(canon::silu(gate, limit)));
    return canon::swiglu(gate, up, limit, skip);
}
bool canon_gate_skipped(float gate, float limit, float skip) {
    return skip > 0 && std::fabs(canon::silu(gate, limit)) < skip;
}

void canon_quant_q8k(const float* x, void* blocks, int n) {
    if (n < 0 || n % QK_K) throw std::invalid_argument("canonical Q8_K: length must be whole 256-value blocks");
    auto* y = static_cast<block_q8_K*>(blocks);
    for (int b = 0; b < n / QK_K; ++b, x += QK_K) {
        float max = 0, amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (amax == 0) {
            std::memset(&y[b], 0, sizeof(block_q8_K));
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            const float value = iscale * x[j] + 12582912.f;
            int32_t bits;
            std::memcpy(&bits, &value, 4);
            const int v = (bits & 0x007fffff) - 0x00400000;
            y[b].qs[j] = (int8_t)(v < 127 ? v : 127);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.f / iscale;
    }
}

void canon_route_sum(const float* rows, const float* weights, float* out, int top_k, int hidden, int first, int last) {
    for (int r = first; r < last; ++r) out[r] = 0.f;
    for (int k = top_k - 1; k >= 0; --k) {
        const float w = weights[k];
        if (w == 0.f) continue;
        const float* row = rows + (size_t)k * hidden;
        for (int r = first; r < last; ++r) out[r] = std::fma(w, row[r], out[r]);
    }
}

}  // namespace strata::kernels::cpu
