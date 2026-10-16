// include/strata/muse/reference.hpp - Muse Spark Open's forward pass in plain f32 C++, one decode token at a time: the
// oracle the engine's GPU and CPU kernels are checked against (the GLM runner's "naive first" rule), and the place
// where the block's arithmetic is written down once.  Semantics are llama.cpp's src/models/muse-spark.cpp (master,
// 2026-10-16) and build_moe_ffn in src/llama-graph.cpp:
//
//   x = rms_norm(embedding row)                                          (weightless, rms_eps)
//   per block:
//     h  = rms_norm(x) * attn_norm
//     q, k, v, g = Wq h, Wk h, Wv h, Wgate h
//     q, k = per-head rms_norm(.) * attn_q_norm / attn_k_norm             (rms_eps)
//     RoPE (ggml NORM: adjacent pairs) on q, k on sliding-window layers only; full layers are NoPE
//     a  = softmax(q k^T / sqrt(head_dim)) v over the cached positions (sliding window: pos_q - pos_k < window)
//     a  = per-head rms_norm(a)  * sigmoid(g)                              (weightless, rms_eps)
//     o  = rms_norm(Wo a)                                                  (weightless, post_eps)
//     x  = attn_resid_alpha * x + attn_resid_beta * o
//     h  = rms_norm(x) * ffn_norm
//     z  = rms_norm(W_latent_down h) * ffn_pre_exp_norm                   (rms_eps; 4096-wide latent)
//     p  = sigmoid(W_router h)          (the router reads h, the full stream, not z)
//     E  = top-k of p + exp_probs_b     (the bias picks; the weights are p's)
//     w_e = p_e / sum(p_E) * expert_weights_scale                          (expert_weights_norm)
//     y  = sum_e w_e * rms_norm((down_e (silu(gate_e z) * up_e z)) * ffn_post_exp_norm)     (post_eps)
//     f  = rms_norm((W_latent_up y) * post_ffw_norm)                        (post_eps)
//     x  = ffn_resid_alpha * x + ffn_resid_beta * f
//   logits = softcap * tanh(logit_scale * (W_out (rms_norm(x) * output_norm)) / softcap)
//
// Weights are f32 row-major [out][in] (dequantized); nothing here is fast.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

namespace strata::muse::ref {

struct Dims {
    int hidden = 0, latent = 0, expert_ff = 0, experts = 0, top_k = 0, q_heads = 0, kv_heads = 0, head_dim = 0;
    int window = 0;
    float rms_eps = 1e-5f, post_eps = 1e-8f, rope_base = 500000.f, expert_weights_scale = 1.f;
    bool expert_weights_norm = true;
    /// Optional: rounds an activation in place where the engine (or llama.cpp) would lose precision, so a check can
    /// reproduce another implementation's arithmetic instead of exact f32.  Sites: the weight a matvec's input feeds
    /// ("attn_q", "attn_k", "attn_v", "attn_gate", "attn_output", "ffn_latent_down", "ffn_gate_inp", "ffn_gate_exps",
    /// "ffn_up_exps", "ffn_down_exps", "ffn_latent_up"), and in attention "cache_k" / "cache_v" (the stored K and V),
    /// "kq_q" (the query against the cache) and "kq" (the softmax weights against V).
    std::function<void(const char* site, float* x, int n)> round;
};

struct Block {
    bool swa = true;                      // sliding window + RoPE; false: full attention, NoPE
    std::vector<float> attn_norm, wq, wk, wv, wgate, wo, q_norm, k_norm, attn_alpha, attn_beta;
    std::vector<float> ffn_norm, latent_down, pre_exp_norm, router, router_bias, post_exp_norm, latent_up, post_ffw_norm;
    std::vector<float> ffn_alpha, ffn_beta;
    std::vector<float> gate_exps, up_exps, down_exps;   // [expert][ff][latent], [expert][ff][latent], [expert][latent][ff]
    /// Instead of the three vectors above: one expert's gate [ff][latent], up [ff][latent] and down [latent][ff] on
    /// demand (a real layer's 256 experts are 51 GB as f32; a check dequantizes only the routed ones).
    std::function<void(int expert, std::vector<float>& gate, std::vector<float>& up, std::vector<float>& down)> expert;
};

/// One block's K and V of every position since `first_pos`, [position - first_pos][kv_heads * head_dim].
struct KvCache {
    std::vector<float> k, v;
    int first_pos = 0;
    int positions = 0;   // one past the last cached position
};

/// What the router chose for the last token of a block, and the block's intermediate streams (for checks against
/// llama.cpp's named tensors: attn_post_norm, ffn_inp, ffn_moe_out, ffn_post_norm).
struct Routes {
    std::vector<int> experts;
    std::vector<float> weights;
    std::vector<float> attn_post_norm, ffn_inp, moe_out, ffn_post_norm;
};

inline void rms_norm(float* x, int n, float eps, const float* w = nullptr) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float s = 1.f / std::sqrt((float) (ss / n) + eps);
    for (int i = 0; i < n; ++i) x[i] = x[i] * s * (w ? w[i] : 1.f);
}

inline void matvec(const std::vector<float>& w, const float* x, int rows, int cols, float* y, size_t offset = 0) {
    for (int r = 0; r < rows; ++r) {
        double acc = 0;
        const float* row = w.data() + offset + (size_t) r * cols;
        for (int c = 0; c < cols; ++c) acc += (double) row[c] * x[c];
        y[r] = (float) acc;
    }
}

/// matvec of a copy of x rounded for `site` (Dims::round), or of x itself.
inline void mv(const Dims& d, const char* site, const std::vector<float>& w, const float* x, int rows, int cols, float* y,
               size_t offset = 0) {
    if (!d.round) return matvec(w, x, rows, cols, y, offset);
    std::vector<float> r(x, x + cols);
    d.round(site, r.data(), cols);
    matvec(w, r.data(), rows, cols, y, offset);
}

inline void round_at(const Dims& d, const char* site, float* x, int n) {
    if (d.round) d.round(site, x, n);
}

inline float sigmoid(float v) { return 1.f / (1.f + std::exp(-v)); }

/// ggml's RoPE NORM mode (LLAMA_ROPE_TYPE_NORM): pairs (2i, 2i+1), theta_i = pos * base^(-2i / dim).
inline void rope_norm(float* x, int dim, int pos, float base) {
    for (int i = 0; i < dim / 2; ++i) {
        const float theta = (float) pos * std::pow(base, -2.f * (float) i / (float) dim);
        const float c = std::cos(theta), s = std::sin(theta);
        const float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/// One block for the token at position `pos` (the cache holds first_pos .. pos-1 and gets this one appended).
inline void block_step(const Dims& d, const Block& b, KvCache& kv, int pos, std::vector<float>& x, Routes* routes = nullptr) {
    const int H = d.hidden, HD = d.head_dim, QD = d.q_heads * HD, KD = d.kv_heads * HD;
    std::vector<float> h(x);
    rms_norm(h.data(), H, d.rms_eps, b.attn_norm.data());
    std::vector<float> q(QD), k(KD), v(KD), g(QD);
    mv(d, "attn_q", b.wq, h.data(), QD, H, q.data());
    mv(d, "attn_k", b.wk, h.data(), KD, H, k.data());
    mv(d, "attn_v", b.wv, h.data(), KD, H, v.data());
    mv(d, "attn_gate", b.wgate, h.data(), QD, H, g.data());
    for (int hh = 0; hh < d.q_heads; ++hh) {
        rms_norm(&q[(size_t) hh * HD], HD, d.rms_eps, b.q_norm.data());
        if (b.swa) rope_norm(&q[(size_t) hh * HD], HD, pos, d.rope_base);
    }
    for (int hh = 0; hh < d.kv_heads; ++hh) {
        rms_norm(&k[(size_t) hh * HD], HD, d.rms_eps, b.k_norm.data());
        if (b.swa) rope_norm(&k[(size_t) hh * HD], HD, pos, d.rope_base);
    }
    round_at(d, "cache_k", k.data(), KD);
    round_at(d, "cache_v", v.data(), KD);
    kv.k.insert(kv.k.end(), k.begin(), k.end());
    kv.v.insert(kv.v.end(), v.begin(), v.end());
    round_at(d, "kq_q", q.data(), QD);
    if (kv.positions == 0) kv.first_pos = pos;
    kv.positions = pos + 1;
    const int first = b.swa ? std::max(kv.first_pos, pos - d.window + 1) : kv.first_pos;
    std::vector<float> a(QD, 0.f);
    const float scale = 1.f / std::sqrt((float) HD);
    for (int hh = 0; hh < d.q_heads; ++hh) {
        const int kvh = hh / (d.q_heads / d.kv_heads);
        std::vector<float> s((size_t) (pos - first + 1));
        float mx = -INFINITY;
        for (int p = first; p <= pos; ++p) {
            double dot = 0;
            const size_t row = (size_t) (p - kv.first_pos) * KD + (size_t) kvh * HD;
            for (int i = 0; i < HD; ++i) dot += (double) q[(size_t) hh * HD + i] * kv.k[row + i];
            s[(size_t) (p - first)] = (float) dot * scale;
            mx = std::max(mx, s[(size_t) (p - first)]);
        }
        double sum = 0;
        for (float& e : s) { e = std::exp(e - mx); sum += e; }
        for (float& e : s) e = (float) (e / sum);
        round_at(d, "kq", s.data(), (int) s.size());
        for (int p = first; p <= pos; ++p) {
            const float w = s[(size_t) (p - first)];
            const size_t row = (size_t) (p - kv.first_pos) * KD + (size_t) kvh * HD;
            for (int i = 0; i < HD; ++i) a[(size_t) hh * HD + i] += w * kv.v[row + i];
        }
        rms_norm(&a[(size_t) hh * HD], HD, d.rms_eps);
    }
    for (int i = 0; i < QD; ++i) a[(size_t) i] *= sigmoid(g[(size_t) i]);
    std::vector<float> o(H);
    mv(d, "attn_output", b.wo, a.data(), H, QD, o.data());
    rms_norm(o.data(), H, d.post_eps);
    for (int i = 0; i < H; ++i) x[(size_t) i] = b.attn_alpha[(size_t) i] * x[(size_t) i] + b.attn_beta[(size_t) i] * o[(size_t) i];
    if (routes) {
        routes->attn_post_norm = o;
        routes->ffn_inp = x;
    }

    const int L = d.latent, FF = d.expert_ff, E = d.experts;
    h = x;
    rms_norm(h.data(), H, d.rms_eps, b.ffn_norm.data());
    std::vector<float> z(L), p(E);
    mv(d, "ffn_latent_down", b.latent_down, h.data(), L, H, z.data());
    rms_norm(z.data(), L, d.rms_eps, b.pre_exp_norm.data());
    mv(d, "ffn_gate_inp", b.router, h.data(), E, H, p.data());
    for (float& e : p) e = sigmoid(e);
    std::vector<int> order((size_t) E);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) {
        return p[(size_t) i] + b.router_bias[(size_t) i] > p[(size_t) j] + b.router_bias[(size_t) j];
    });
    order.resize((size_t) d.top_k);
    std::vector<float> w((size_t) d.top_k);
    float wsum = 0;
    for (int i = 0; i < d.top_k; ++i) wsum += (w[(size_t) i] = p[(size_t) order[(size_t) i]]);
    for (float& e : w) e = (d.expert_weights_norm ? e / wsum : e) * d.expert_weights_scale;
    std::vector<float> y(L, 0.f), gt(FF), up(FF), out(L);
    std::vector<float> eg, eu, ed;
    for (int i = 0; i < d.top_k; ++i) {
        const int e = order[(size_t) i];
        if (b.expert) {
            b.expert(e, eg, eu, ed);
            mv(d, "ffn_gate_exps", eg, z.data(), FF, L, gt.data());
            mv(d, "ffn_up_exps", eu, z.data(), FF, L, up.data());
        } else {
            mv(d, "ffn_gate_exps", b.gate_exps, z.data(), FF, L, gt.data(), (size_t) e * FF * L);
            mv(d, "ffn_up_exps", b.up_exps, z.data(), FF, L, up.data(), (size_t) e * FF * L);
        }
        for (int j = 0; j < FF; ++j) gt[(size_t) j] = gt[(size_t) j] * sigmoid(gt[(size_t) j]) * up[(size_t) j];
        if (b.expert) mv(d, "ffn_down_exps", ed, gt.data(), L, FF, out.data());
        else mv(d, "ffn_down_exps", b.down_exps, gt.data(), L, FF, out.data(), (size_t) e * L * FF);
        for (int j = 0; j < L; ++j) out[(size_t) j] *= b.post_exp_norm[(size_t) j];
        rms_norm(out.data(), L, d.post_eps);
        for (int j = 0; j < L; ++j) y[(size_t) j] += w[(size_t) i] * out[(size_t) j];
    }
    std::vector<float> f(H);
    mv(d, "ffn_latent_up", b.latent_up, y.data(), H, L, f.data());
    for (int i = 0; i < H; ++i) f[(size_t) i] *= b.post_ffw_norm[(size_t) i];
    rms_norm(f.data(), H, d.post_eps);
    for (int i = 0; i < H; ++i) x[(size_t) i] = b.ffn_alpha[(size_t) i] * x[(size_t) i] + b.ffn_beta[(size_t) i] * f[(size_t) i];
    if (routes) {
        routes->experts = order;
        routes->weights = w;
        routes->moe_out = y;
        routes->ffn_post_norm = f;
    }
}

/// The logits of a final hidden state: output_norm, the head, logit_scale and the tanh softcap.
inline std::vector<float> logits(const Dims& d, const std::vector<float>& x, const std::vector<float>& output_norm,
                                 const std::vector<float>& output, int vocab, float logit_scale, float softcap) {
    std::vector<float> h(x), out((size_t) vocab);
    rms_norm(h.data(), d.hidden, d.rms_eps, output_norm.data());
    matvec(output, h.data(), vocab, d.hidden, out.data());
    for (float& l : out) {
        l *= logit_scale;
        if (softcap > 0) l = softcap * std::tanh(l / softcap);
    }
    return out;
}

/// The stream entering block 0: the embedding row, RMS-normalized without a weight.
inline std::vector<float> embed(const Dims& d, const std::vector<float>& token_embd, int token) {
    std::vector<float> x(token_embd.begin() + (ptrdiff_t) token * d.hidden, token_embd.begin() + (ptrdiff_t) (token + 1) * d.hidden);
    rms_norm(x.data(), d.hidden, d.rms_eps);
    return x;
}

}  // namespace strata::muse::ref
