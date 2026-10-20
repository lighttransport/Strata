// include/strata/spec/host_sampler.hpp - the sampler chain on the host, for engines whose logits are on the host
// (the GLM decoder's step/verify return full-vocabulary rows).  The same chain and the same draws as the device
// kernel (include/strata/kernels/sampler.hpp, src/kernels/cuda/sampler.cu; sampler_parity pins its host reference):
//
//     penalties -> top_k (at most 64, ties to the lower id) -> top_p -> min_p -> temperature -> Philox draw
//
// The draw for a token is keyed by (seed, counter): a token's counter is its absolute position in the sequence, so a
// verify window, a replay or a resumed prefix draws the same number for the same position.  Speculative decoding
// with a point-mass draft (MTP, lookup): accept the draft iff it equals the target's sample at that position; on a
// mismatch the target's sample is the next token.  That is exactly the target's distribution
// (include/strata/core/spec_prob.hpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace strata::spec {

struct HostSampler {
    float temperature = 0.0f;     // <= 0: greedy (argmax, ties to the lower id)
    int top_k = 64;               // 1..64 as given; 0 or more than 64 keep the widest list, 64
    float top_p = 1.0f;           // 1 disables
    float min_p = 0.0f;           // 0 disables
    int min_keep = 1;
    int penalty_last_n = 0;       // 0 disables the penalties
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    uint64_t seed = 0;

    bool greedy() const { return !(temperature > 0.0f); }

    // The device kernel's Philox draw (sampler_parity's host_philox_uniform).
    static float philox_uniform(uint64_t seed, uint64_t counter) {
        uint32_t c0 = (uint32_t)counter, c1 = (uint32_t)(counter >> 32);
        uint32_t c2 = (uint32_t)seed, c3 = (uint32_t)(seed >> 32);
        for (uint32_t i = 0; i < 10; ++i) {
            const uint32_t hi0 = (uint32_t)(((uint64_t)0x9E3779B9u * c0) >> 32);
            const uint32_t hi1 = (uint32_t)(((uint64_t)0xBB67AE85u * c2) >> 32);
            const uint32_t lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
            const uint32_t n0 = hi1 ^ c1 ^ i, n1 = lo1, n2 = hi0 ^ c3 ^ 0u, n3 = lo0;
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        }
        return (float)(c0 >> 8) * (1.0f / 16777216.0f);
    }

    static int argmax(const float *logits, int nv) {
        return int(std::max_element(logits, logits + nv) - logits);
    }

    // One token from a row of `nv` logits.  `history` (may be empty): the tokens this pick follows, most recent last;
    // only the last penalty_last_n count.  `counter`: the token's absolute position.
    int pick(const float *logits, int nv, uint64_t counter, const std::vector<int> &history = {}) const {
        std::unordered_map<int, int> counts;
        if (penalty_last_n > 0) {
            const size_t from = history.size() > size_t(penalty_last_n) ? history.size() - size_t(penalty_last_n) : 0;
            for (size_t i = from; i < history.size(); ++i)
                if (history[i] >= 0 && history[i] < nv) ++counts[history[i]];
        }
        auto penal = [&](int v) {
            float logit = logits[v];
            if (counts.empty()) return logit;
            const auto it = counts.find(v);
            if (it == counts.end()) return logit;
            if (logit <= 0.0f) logit *= penalty_repeat; else logit /= penalty_repeat;
            return logit - (float)it->second * penalty_freq - penalty_present;
        };
        if (greedy()) {
            int best = 0; float bv = penal(0);
            for (int v = 1; v < nv; ++v) { const float s = penal(v); if (s > bv) { bv = s; best = v; } }
            return best;
        }
        constexpr int KMAX = 64;
        const int k = std::min(nv, (top_k > 0 && top_k < KMAX) ? top_k : KMAX);
        // top_k by (value desc, id asc): a min-heap of the k best seen so far
        struct Item { float value; int id; };
        auto worse = [](const Item &a, const Item &b) { return a.value > b.value || (a.value == b.value && a.id < b.id); };
        std::vector<Item> heap;
        heap.reserve(size_t(k));
        for (int v = 0; v < nv; ++v) {
            const Item it{penal(v), v};
            if ((int)heap.size() < k) { heap.push_back(it); std::push_heap(heap.begin(), heap.end(), worse); }
            else if (it.value > heap.front().value) {       // an equal value has a higher id: the earlier one stays
                std::pop_heap(heap.begin(), heap.end(), worse);
                heap.back() = it;
                std::push_heap(heap.begin(), heap.end(), worse);
            }
        }
        std::sort(heap.begin(), heap.end(), [](const Item &a, const Item &b) {
            return a.value > b.value || (a.value == b.value && a.id < b.id);
        });
        const int n_sel = (int)heap.size();
        int n_keep = n_sel;
        if (top_p < 1.0f) {
            double sum = 0.0;
            for (int i = 0; i < n_sel; ++i) sum += std::exp((double)heap[i].value - (double)heap[0].value);
            double cum = 0.0;
            int cut = n_sel;
            for (int i = 0; i < n_sel; ++i) {
                cum += std::exp((double)heap[i].value - (double)heap[0].value) / sum;
                if (cum >= (double)top_p) { cut = i + 1; break; }
            }
            if (cut < min_keep) cut = min_keep < n_sel ? min_keep : n_sel;
            n_keep = cut;
        }
        if (min_p > 0.0f) {
            const float thresh = heap[0].value + std::log(min_p);
            for (int i = 0; i < n_keep; ++i)
                if (heap[i].value < thresh) { n_keep = i; break; }
        }
        n_keep = std::max(n_keep, 1);
        const float inv_t = 1.0f / temperature;
        float smx = heap[0].value * inv_t;
        for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, heap[i].value * inv_t);
        double sum = 0.0;
        for (int i = 0; i < n_keep; ++i) sum += std::exp((double)(heap[i].value * inv_t) - (double)smx);
        const float u = philox_uniform(seed, counter);
        double cum = 0.0;
        for (int i = 0; i < n_keep; ++i) {
            cum += std::exp((double)(heap[i].value * inv_t) - (double)smx) / sum;
            if ((double)u < cum) return heap[i].id;
        }
        return heap[n_keep - 1].id;
    }
};

}  // namespace strata::spec
