// tests/host/glm_sampler_test.cpp - the host sampler chain (include/strata/spec/host_sampler.hpp) against a serial
// transcription of sampler_parity's reference (the device kernel's chain), plus determinism and distribution checks.
#include "strata/spec/host_sampler.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using strata::spec::HostSampler;

static int fails = 0;
#define EXPECT(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++fails; } } while (0)

// sampler_parity.cpp's sampled_reference, O(k * V) selection, no Gumbel.
static int reference(const std::vector<float> &l, const std::vector<int> &hist, const HostSampler &p, uint64_t counter) {
    auto count = [&](int v) {
        int c = 0;
        const size_t from = hist.size() > size_t(p.penalty_last_n) ? hist.size() - size_t(p.penalty_last_n) : 0;
        for (size_t i = from; p.penalty_last_n > 0 && i < hist.size(); ++i) c += hist[i] == v;
        return c;
    };
    auto penal = [&](float logit, int c) {
        if (c <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        return logit - (float)c * p.penalty_freq - p.penalty_present;
    };
    const int nv = (int)l.size(), k = std::min(nv, (p.top_k > 0 && p.top_k < 64) ? p.top_k : 64);
    std::vector<int> ids; std::vector<float> val; std::vector<char> taken(nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[v]) continue;
            const float s = penal(l[v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[best] = 1; ids.push_back(best); val.push_back(bv);
    }
    int n_keep = (int)ids.size();
    if (p.top_p < 1.0f) {
        double sum = 0, cum = 0; int cut = n_keep;
        for (int i = 0; i < n_keep; ++i) sum += std::exp((double)val[i] - (double)val[0]);
        for (int i = 0; i < n_keep; ++i) { cum += std::exp((double)val[i] - (double)val[0]) / sum; if (cum >= (double)p.top_p) { cut = i + 1; break; } }
        if (cut < p.min_keep) cut = std::min(p.min_keep, n_keep);
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float th = val[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i) if (val[i] < th) { n_keep = i; break; }
    }
    const float inv_t = 1.0f / p.temperature;
    float smx = val[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, val[i] * inv_t);
    double sum = 0, cum = 0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double)(val[i] * inv_t) - (double)smx);
    const float u = HostSampler::philox_uniform(p.seed, counter);
    for (int i = 0; i < n_keep; ++i) { cum += std::exp((double)(val[i] * inv_t) - (double)smx) / sum; if ((double)u < cum) return ids[i]; }
    return ids[n_keep - 1];
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.f, 3.f);
    const int nv = 5000;
    std::vector<float> l(nv);
    std::vector<int> hist;
    for (int i = 0; i < 40; ++i) hist.push_back(int(rng() % 200));
    const HostSampler cases[] = {
        {1.0f, 64, 1.0f, 0.0f}, {0.7f, 20, 0.95f, 0.0f}, {1.3f, 0, 0.9f, 0.05f}, {1.0f, 5, 1.0f, 0.0f},
        {0.8f, 40, 0.8f, 0.0f, 1, 32, 1.1f, 0.2f, 0.3f}};
    int compared = 0;
    for (int trial = 0; trial < 60; ++trial) {
        for (auto &x : l) x = normal(rng);
        l[rng() % 200] += 6.f;                                   // a peak, as real logits have
        for (const auto &base : cases) {
            HostSampler p = base;
            p.seed = 1000 + trial;
            for (uint64_t c = 0; c < 4; ++c) {
                EXPECT(p.pick(l.data(), nv, c, hist) == reference(l, hist, p, c));
                ++compared;
            }
        }
    }
    // the same seed and position draw the same token; a different position may not
    HostSampler s{1.0f, 64, 1.0f, 0.0f};
    s.seed = 42;
    EXPECT(s.pick(l.data(), nv, 17) == s.pick(l.data(), nv, 17));
    // top_k 1 and temperature 0 are the argmax
    HostSampler k1{1.0f, 1, 1.0f, 0.0f};
    EXPECT(k1.pick(l.data(), nv, 3) == HostSampler::argmax(l.data(), nv));
    HostSampler g{};
    EXPECT(g.greedy() && g.pick(l.data(), nv, 3) == HostSampler::argmax(l.data(), nv));
    // frequencies follow softmax(l / T) over a small vocabulary (top_k 64 covers it all)
    std::vector<float> small = {2.0f, 1.0f, 0.5f, 0.0f, -1.0f};
    HostSampler t{0.8f, 64, 1.0f, 0.0f};
    t.seed = 9;
    std::vector<int> freq(small.size(), 0);
    const int draws = 200000;
    for (int c = 0; c < draws; ++c) ++freq[t.pick(small.data(), (int)small.size(), (uint64_t)c)];
    double z = 0;
    for (float x : small) z += std::exp(x / 0.8);
    double worst = 0;
    for (size_t i = 0; i < small.size(); ++i)
        worst = std::max(worst, std::fabs(freq[i] / (double)draws - std::exp(small[i] / 0.8) / z));
    EXPECT(worst < 0.005);
    std::printf("glm_sampler_test: %d picks against the reference, max frequency error %.4f, %d failures\n",
                compared, worst, fails);
    return fails ? 1 : 0;
}
