// tests/core/muse_reference_test.cpp - the Muse Spark reference block (strata/muse/reference.hpp) on random tiny
// weights: the properties the architecture defines, each checked where it can be told apart from a likely mistake.
//
//   1. sliding window: a key older than the window does not reach a sliding-window block's output, and does reach a
//      full-attention block's;
//   2. the router's bias chooses but does not weigh: a large bias on one expert puts it in the top-k, and its weight is
//      its sigmoid share of the chosen ones, not biased;
//   3. the residuals: alpha = 1, beta = 0 leaves the stream as it was;
//   4. RoPE is relative: shifting every position by the same amount leaves a sliding-window block's output unchanged
//      (to float rounding), and the full layers ignore positions altogether;
//   5. the head: logits are capped by the softcap.
#include "strata/muse/reference.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace ref = strata::muse::ref;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-74s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

ref::Dims dims() {
    ref::Dims d;
    d.hidden = 32; d.latent = 16; d.expert_ff = 24; d.experts = 6; d.top_k = 2; d.q_heads = 4; d.kv_heads = 2;
    d.head_dim = 8; d.window = 3;
    return d;
}

std::vector<float> rnd(std::mt19937& g, size_t n, float scale = 0.3f, float mean = 0.f) {
    std::normal_distribution<float> nd(mean, scale);
    std::vector<float> v(n);
    for (auto& e : v) e = nd(g);
    return v;
}

ref::Block block(const ref::Dims& d, bool swa, unsigned seed) {
    std::mt19937 g(seed);
    ref::Block b;
    b.swa = swa;
    const size_t H = d.hidden, QD = (size_t) d.q_heads * d.head_dim, KD = (size_t) d.kv_heads * d.head_dim;
    const size_t L = d.latent, FF = d.expert_ff, E = d.experts;
    b.attn_norm = rnd(g, H, 0.1f, 1.f);
    b.wq = rnd(g, QD * H); b.wk = rnd(g, KD * H); b.wv = rnd(g, KD * H); b.wgate = rnd(g, QD * H); b.wo = rnd(g, H * QD);
    b.q_norm = rnd(g, d.head_dim, 0.1f, 1.f); b.k_norm = std::vector<float>(d.head_dim, 1.f);
    b.attn_alpha = rnd(g, H, 0.05f, 1.f); b.attn_beta = rnd(g, H, 0.05f, 0.5f);
    b.ffn_norm = rnd(g, H, 0.1f, 1.f); b.latent_down = rnd(g, L * H); b.pre_exp_norm = rnd(g, L, 0.1f, 1.f);
    b.router = rnd(g, E * H); b.router_bias = std::vector<float>(E, 0.f); b.post_exp_norm = rnd(g, L, 0.1f, 1.f);
    b.latent_up = rnd(g, H * L); b.post_ffw_norm = rnd(g, H, 0.1f, 1.f);
    b.ffn_alpha = rnd(g, H, 0.05f, 1.f); b.ffn_beta = rnd(g, H, 0.05f, 0.5f);
    b.gate_exps = rnd(g, E * FF * L); b.up_exps = rnd(g, E * FF * L); b.down_exps = rnd(g, E * L * FF);
    return b;
}

// run positions [start, start + n) through one block; `perturb` alters the stream of the step at index 1
std::vector<float> run(const ref::Dims& d, const ref::Block& b, int start, int n, bool perturb, ref::Routes* last = nullptr) {
    std::mt19937 g(7);
    ref::KvCache kv;   // starts at `start`
    std::vector<float> x;
    for (int i = 0; i < n; ++i) {
        x = rnd(g, (size_t) d.hidden, 1.f);
        if (perturb && i == 1) for (auto& e : x) e = -e * 3.f;
        ref::block_step(d, b, kv, start + i, x, last);
    }
    return x;
}

double maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}
}  // namespace

int main() {
    const ref::Dims d = dims();
    {   // 1. the window: 6 steps with window 3, the step at index 1 altered; the last query sees positions 3, 4, 5 only
        const ref::Block swa = block(d, true, 1), full = block(d, false, 1);
        const double ds = maxdiff(run(d, swa, 0, 6, false), run(d, swa, 0, 6, true));
        const double df = maxdiff(run(d, full, 0, 6, false), run(d, full, 0, 6, true));
        check(ds == 0.0, "sliding window: a key older than the window does not reach the output");
        check(df > 1e-4, "full attention: the same key does reach the output");
    }
    {   // 2. the router bias
        ref::Block b = block(d, true, 2);
        ref::Routes plain, biased;
        run(d, b, 0, 1, false, &plain);
        int outside = 0;
        while (outside == plain.experts[0] || outside == plain.experts[1]) ++outside;
        b.router_bias[(size_t) outside] = 10.f;
        run(d, b, 0, 1, false, &biased);
        const bool chosen = biased.experts[0] == outside;
        check(chosen, "a large exp_probs_b puts its expert in the top-k");
        const float sum = biased.weights[0] + biased.weights[1];
        check(std::fabs(sum - 1.f) < 1e-6f && biased.weights[0] > 0 && biased.weights[1] > 0,
              "the chosen weights are renormalized sigmoid shares (sum 1)");
        check(biased.weights[0] < 0.999f, "the bias does not enter the weight (10 + p would make it ~1)");
    }
    {   // 3. alpha 1, beta 0: identity
        ref::Block b = block(d, true, 3);
        std::fill(b.attn_alpha.begin(), b.attn_alpha.end(), 1.f);
        std::fill(b.attn_beta.begin(), b.attn_beta.end(), 0.f);
        std::fill(b.ffn_alpha.begin(), b.ffn_alpha.end(), 1.f);
        std::fill(b.ffn_beta.begin(), b.ffn_beta.end(), 0.f);
        std::mt19937 g(11);
        std::vector<float> x = rnd(g, (size_t) d.hidden, 1.f), x0 = x;
        ref::KvCache kv;
        ref::block_step(d, b, kv, 0, x);
        check(maxdiff(x, x0) == 0.0, "alpha = 1, beta = 0 leaves the stream unchanged");
    }
    {   // 4. relative positions
        const ref::Block swa = block(d, true, 4), full = block(d, false, 4);
        const double ds = maxdiff(run(d, swa, 0, 3, false), run(d, swa, 100, 3, false));
        const double df = maxdiff(run(d, full, 0, 3, false), run(d, full, 100, 3, false));
        check(ds < 1e-4, "RoPE is relative: positions shifted by 100 give the same output (" + std::to_string(ds) + ")");
        check(df == 0.0, "full layers are NoPE: absolute positions do not matter at all");
        // and RoPE is really applied: a sliding-window block differs from the same weights without it
        ref::Block nope = swa;
        nope.swa = false;
        ref::Dims wide = d;
        wide.window = 1 << 20;
        ref::Block swa_wide = swa;
        check(maxdiff(run(wide, swa_wide, 0, 3, false), run(wide, nope, 0, 3, false)) > 1e-4,
              "RoPE changes a sliding-window block's output");
    }
    {   // 5. the softcap
        std::mt19937 g(5);
        const int V = 10;
        std::vector<float> out = rnd(g, (size_t) V * d.hidden, 50.f), on(d.hidden, 1.f), x = rnd(g, d.hidden, 1.f);
        const auto l = ref::logits(d, x, on, out, V, 0.1767767f, 20.f);
        bool capped = true;
        for (float v : l) capped &= std::fabs(v) < 20.f;
        check(capped, "every logit is inside the softcap (|l| < 20)");
    }
    std::printf("muse_reference_test: %d failures\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
