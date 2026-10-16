// tools/muse_check_common.hpp - what the Muse Spark parity tools share (muse_reference_check, muse_gpu_trunk_check):
// a GGUF's tensors as f32 through ggml's to_float, the reference's blocks built from them with experts dequantized on
// demand, the activation roundings of ggml-cpu (Q8_K / Q8_0) and of the GPU projections (Q8_1), llama.cpp dumps, and
// the per-token comparison statistic.
#pragma once

#include "strata/artifact/muse_model.hpp"
#include "strata/muse/reference.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace muse_check {

namespace ref = strata::muse::ref;

// ggml's quantize_row_q8_K_ref then dequantization (d * q), in place: d = max / -127 from the signed largest value
inline void round_q8k(float* x, int n) {
    for (int b = 0; b + 256 <= n; b += 256) {
        float max = 0, amax = 0;
        for (int j = 0; j < 256; ++j) {
            const float ax = std::fabs(x[b + j]);
            if (ax > amax) { amax = ax; max = x[b + j]; }
        }
        if (!amax) { for (int j = 0; j < 256; ++j) x[b + j] = 0; continue; }
        const float iscale = -127.f / max, d = 1 / iscale;
        for (int j = 0; j < 256; ++j) x[b + j] = d * (float) std::min(127, (int) std::nearbyint(iscale * x[b + j]));
    }
}

// the GPU projections' Q8_1 (native_quantize_q8_1, q8_1_finite.hpp): per 32 values d = amax / 127, q = roundf(x / d),
// and the dot reads d as fp16, so the value the weight meets is q * fp16(d)
inline void round_q8_1(float* x, int n) {
    for (int b = 0; b + 32 <= n; b += 32) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b + j]));
        if (amax == 0) { for (int j = 0; j < 32; ++j) x[b + j] = 0; continue; }
        const float d = amax / 127.0f, dh = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        for (int j = 0; j < 32; ++j) {
            const float q = std::max(-127.f, std::min(127.f, std::round(x[b + j] / d)));
            x[b + j] = q * dh;
        }
    }
}

inline void round_f16(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x[i]));
}

/// rounds to a ggml activation type and back (Q8_K, Q8_0 and F16; F32 unchanged); kQ8_1 for the GPU's Q8_1
constexpr int kQ8_1 = -1;
inline void round_type(int type, float* x, int n) {
    if (type == GGML_TYPE_F32) return;
    if (type == GGML_TYPE_Q8_K) return round_q8k(x, n);
    if (type == kQ8_1) return round_q8_1(x, n);
    if (type == GGML_TYPE_F16) return round_f16(x, n);
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    std::vector<uint8_t> q(ggml_row_size((ggml_type) type, n));
    tr->from_float_ref(x, q.data(), n);
    tr->to_float(q.data(), x, n);
}

// the activation type ggml-cpu quantizes to for a weight type (ggml-cpu.c type_traits_cpu[].vec_dot_type)
inline int cpu_vec_dot_type(int weight_type) {
    switch (weight_type) {
        case GGML_TYPE_F32: return GGML_TYPE_F32;
        case GGML_TYPE_F16: return GGML_TYPE_F16;
        case GGML_TYPE_Q8_0: return GGML_TYPE_Q8_0;
        default: return GGML_TYPE_Q8_K;   // the K-quants and i-quants of this model
    }
}

struct Dump {
    int32_t type = -1;
    int64_t ne[4] = {0, 0, 0, 0};
    std::vector<float> f;
    std::vector<int32_t> i;
    bool ok() const { return type >= 0; }
};

inline Dump load_dump(const std::string& path) {
    Dump d;
    std::ifstream in(path, std::ios::binary);
    if (!in) return d;
    in.read((char*) &d.type, 4);
    in.read((char*) d.ne, 32);
    const size_t n = (size_t) (d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3]);
    if (d.type == 0) { d.f.resize(n); in.read((char*) d.f.data(), (std::streamsize) (n * 4)); }
    else { d.i.resize(n); in.read((char*) d.i.data(), (std::streamsize) (n * 4)); }
    return d;
}

struct Model {
    const strata::GgufModel& m;
    const strata::TensorInfo& t(const std::string& name) const {
        const strata::TensorInfo* p = m.find(name);
        if (!p) throw std::runtime_error("missing " + name);
        return *p;
    }
    const uint8_t* data(const std::string& name) const {
        size_t shard = 0;
        const strata::TensorInfo* p = m.find(name, &shard);
        if (!p) throw std::runtime_error("missing " + name);
        return m.shard(shard).tensor_data(*p);
    }
    /// rows [r0, r0 + rows) of a tensor whose rows are its dimension 0, as f32
    void rows(const std::string& name, int64_t r0, int64_t rows, std::vector<float>& out) const {
        const strata::TensorInfo& p = t(name);
        const int64_t ne0 = (int64_t) p.shape[0];
        const size_t row_bytes = ggml_row_size((ggml_type) p.type, ne0);
        out.resize((size_t) (rows * ne0));
        const uint8_t* base = data(name) + (size_t) r0 * row_bytes;
        if (p.type == GGML_TYPE_F32) std::memcpy(out.data(), base, out.size() * 4);
        else ggml_get_type_traits((ggml_type) p.type)->to_float(base, out.data(), rows * ne0);
    }
    std::vector<float> all(const std::string& name) const {
        const strata::TensorInfo& p = t(name);
        int64_t rows = 1;
        for (size_t k = 1; k < p.shape.size(); ++k) rows *= (int64_t) p.shape[k];
        std::vector<float> v;
        this->rows(name, 0, rows, v);
        return v;
    }
};

struct Stat {
    double max_rel = 0, min_cos = 1;
    void add(const float* a, const float* b, size_t n) {
        double num = 0, den = 0, ab = 0, aa = 0;
        for (size_t i = 0; i < n; ++i) {
            num += (double) (a[i] - b[i]) * (a[i] - b[i]);
            den += (double) b[i] * b[i];
            ab += (double) a[i] * b[i];
            aa += (double) a[i] * a[i];
        }
        max_rel = std::max(max_rel, std::sqrt(num / std::max(den, 1e-30)));
        min_cos = std::min(min_cos, ab / std::sqrt(std::max(aa * den, 1e-30)));
    }
};

inline ref::Dims dims(const strata::muse::MuseConfig& c) {
    ref::Dims d;
    d.hidden = (int) c.hidden; d.latent = (int) c.latent; d.expert_ff = (int) c.expert_ff; d.experts = (int) c.experts;
    d.top_k = c.top_k; d.q_heads = (int) c.q_heads; d.kv_heads = (int) c.kv_heads; d.head_dim = (int) c.head_dim;
    d.window = (int) c.window; d.rms_eps = (float) c.rms_eps; d.post_eps = (float) c.post_eps;
    d.rope_base = (float) c.rope_base_swa; d.expert_weights_scale = (float) c.expert_weights_scale;
    d.expert_weights_norm = c.expert_weights_norm;
    return d;
}

/// The reference's blocks of every layer; experts dequantized when the router first picks them, then cached.
struct Blocks {
    using Expert = std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>;
    std::vector<ref::Block> blocks;
    std::vector<std::map<int, Expert>> cache;
};

inline std::unique_ptr<Blocks> load_blocks(const Model& M, const strata::muse::MuseConfig& c) {
    auto B = std::make_unique<Blocks>();
    B->blocks.resize((size_t) c.layers);
    B->cache.resize((size_t) c.layers);
    Blocks* raw = B.get();
    for (int l = 0; l < c.layers; ++l) {
        ref::Block& b = B->blocks[(size_t) l];
        auto n = [&](const char* s) { return "blk." + std::to_string(l) + "." + s; };
        b.swa = c.swa[(size_t) l] != 0;
        b.attn_norm = M.all(n("attn_norm.weight")); b.wq = M.all(n("attn_q.weight")); b.wk = M.all(n("attn_k.weight"));
        b.wv = M.all(n("attn_v.weight")); b.wgate = M.all(n("attn_gate.weight")); b.wo = M.all(n("attn_output.weight"));
        b.q_norm = M.all(n("attn_q_norm.weight")); b.k_norm = M.all(n("attn_k_norm.weight"));
        b.attn_alpha = M.all(n("attn_resid_alpha.weight")); b.attn_beta = M.all(n("attn_resid_beta.weight"));
        b.ffn_norm = M.all(n("ffn_norm.weight")); b.latent_down = M.all(n("ffn_latent_down.weight"));
        b.pre_exp_norm = M.all(n("ffn_pre_exp_norm.weight")); b.router = M.all(n("ffn_gate_inp.weight"));
        b.router_bias = M.all(n("exp_probs_b.bias")); b.post_exp_norm = M.all(n("ffn_post_exp_norm.weight"));
        b.latent_up = M.all(n("ffn_latent_up.weight")); b.post_ffw_norm = M.all(n("post_ffw_norm.weight"));
        b.ffn_alpha = M.all(n("ffn_resid_alpha.weight")); b.ffn_beta = M.all(n("ffn_resid_beta.weight"));
        b.expert = [&M, &c, raw, l](int e, std::vector<float>& g, std::vector<float>& u, std::vector<float>& dn) {
            auto& slot = raw->cache[(size_t) l];
            auto it = slot.find(e);
            if (it == slot.end()) {
                const std::string p = "blk." + std::to_string(l) + ".";
                std::vector<float> a, bb, cc;
                M.rows(p + "ffn_gate_exps.weight", (int64_t) e * c.expert_ff, c.expert_ff, a);
                M.rows(p + "ffn_up_exps.weight", (int64_t) e * c.expert_ff, c.expert_ff, bb);
                M.rows(p + "ffn_down_exps.weight", (int64_t) e * c.latent, c.latent, cc);
                it = slot.emplace(e, std::make_tuple(std::move(a), std::move(bb), std::move(cc))).first;
            }
            g = std::get<0>(it->second);
            u = std::get<1>(it->second);
            dn = std::get<2>(it->second);
        };
    }
    return B;
}

}  // namespace muse_check
