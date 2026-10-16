// include/strata/artifact/muse_model.hpp - Muse Spark Open (`muse-spark`): the geometry from a GGUF's keys, the tensor
// directory checked against the block rules (strata/host/muse_names.h), the byte footprint, and the two-node expert
// row split that footprint implies.  Header-only and data-free: it reads the shards' headers, so it runs on the first
// megabyte of each shard (tools/muse_gguf_census.py says how to fetch them) as well as on the full download.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/host/glm_names.h"
#include "strata/host/muse_names.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::muse {

struct MuseConfig {
    int layers = 0;
    int64_t hidden = 0, latent = 0, expert_ff = 0, experts = 0;
    int top_k = 0;
    int64_t q_heads = 0, kv_heads = 0, head_dim = 0, window = 0, context = 0, vocab = 0;
    std::vector<uint8_t> swa;            // per layer: 1 = sliding window + RoPE, 0 = full attention, NoPE
    double rope_base = 0, rope_base_swa = 0, rms_eps = 0, post_eps = 0, logit_scale = 0, softcap = 0;
    double expert_weights_scale = 1;
    bool expert_weights_norm = false;
    int gating_func = 0;                 // 2 = sigmoid (llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID)

    strata_muse_dims dims() const { return strata_muse_dims{hidden, latent, expert_ff, experts, q_heads, kv_heads, head_dim}; }
    int full_layers() const {
        int n = 0;
        for (uint8_t s : swa) n += s == 0;
        return n;
    }
};

/// Fills `c` from the metadata shard.  Returns "" or the first problem.
inline std::string read_config(const GgufFile& meta, MuseConfig& c) {
    const MetaValue* arch = meta.get("general.architecture");
    if (!arch || arch->s != STRATA_MUSE_ARCH)
        return "general.architecture is " + std::string(arch ? "'" + arch->s + "'" : "missing") + ", not " STRATA_MUSE_ARCH;
    std::string err;
    auto key = [&](const char* k) -> const MetaValue* {
        const MetaValue* v = meta.get(std::string(STRATA_MUSE_ARCH ".") + k);
        if (!v && err.empty()) err = std::string("missing key " STRATA_MUSE_ARCH ".") + k;
        return v;
    };
    auto i64 = [&](const char* k) -> int64_t { const MetaValue* v = key(k); return v ? (int64_t) v->u : 0; };
    auto f64 = [&](const char* k) -> double { const MetaValue* v = key(k); return v ? v->f : 0.0; };
    c.layers = (int) i64("block_count");
    c.hidden = i64("embedding_length");
    c.latent = i64("moe_latent_size");
    c.expert_ff = i64("expert_feed_forward_length");
    c.experts = i64("expert_count");
    c.top_k = (int) i64("expert_used_count");
    c.q_heads = i64("attention.head_count");
    c.kv_heads = i64("attention.head_count_kv");
    c.head_dim = i64("attention.key_length");
    const int64_t v_dim = i64("attention.value_length");
    c.window = i64("attention.sliding_window");
    c.context = i64("context_length");
    c.rope_base = f64("rope.freq_base");
    c.rope_base_swa = f64("rope.freq_base_swa");
    c.rms_eps = f64("attention.layer_norm_rms_epsilon");
    c.post_eps = f64("post_norm_epsilon");
    c.logit_scale = f64("logit_scale");
    c.softcap = f64("final_logit_softcapping");
    c.expert_weights_scale = f64("expert_weights_scale");
    const MetaValue* norm = key("expert_weights_norm");
    c.expert_weights_norm = norm && norm->u != 0;
    c.gating_func = (int) i64("expert_gating_func");
    const MetaValue* pattern = key("attention.sliding_window_pattern");
    if (!err.empty()) return err;
    if (v_dim != c.head_dim) return "attention.value_length differs from key_length (the kernels assume one head width)";
    if (pattern->type != MetaType::ARRAY || pattern->count != (uint64_t) c.layers || pattern->items.size() != pattern->count)
        return "attention.sliding_window_pattern is not one entry per layer";
    c.swa.clear();
    for (const MetaValue& e : pattern->items) c.swa.push_back(e.u ? 1 : 0);
    if (c.layers <= 0 || c.hidden <= 0 || c.latent <= 0 || c.expert_ff <= 0 || c.experts <= 0 || c.top_k <= 0 ||
        c.top_k > c.experts || c.q_heads <= 0 || c.kv_heads <= 0 || c.q_heads % c.kv_heads || c.head_dim <= 0)
        return "implausible geometry";
    if (c.gating_func != 2) return "expert_gating_func " + std::to_string(c.gating_func) + " (only 2, sigmoid, is implemented)";
    const MetaValue* tokens = meta.get("tokenizer.ggml.tokens");
    c.vocab = tokens ? (int64_t) tokens->count : 0;
    return "";
}

struct Footprint {
    uint64_t experts = 0;            // every routed expert tensor
    uint64_t expert_bytes = 0;       // one expert (gate + up + down) of the largest layer
    uint64_t fixed_layers = 0;       // the blocks' other tensors (the GPU trunk)
    uint64_t fixed_layer_max = 0;    // the largest block's
    uint64_t lm_head = 0, token_embd = 0, other = 0;
    uint64_t total() const { return experts + fixed_layers + lm_head + token_embd + other; }
};

/// Every problem with the tensor directory (missing, extra, wrong shape or format); fills `fp` with the byte counts.
inline std::vector<std::string> check_tensors(const GgufModel& m, const MuseConfig& c, Footprint& fp) {
    std::vector<std::string> problems;
    fp = Footprint{};
    const strata_muse_dims d = c.dims();
    size_t seen = 0;
    auto shape_str = [](const std::vector<uint64_t>& s) {
        std::string out = "[";
        for (size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
        return out + "]";
    };
    for (int l = 0; l < c.layers; ++l) {
        uint64_t layer_fixed = 0, layer_expert = 0;
        for (int k = 0; k < STRATA_MUSE_TENSOR_COUNT; ++k) {
            const auto kind = (strata_muse_tensor) k;
            char name[STRATA_NAME_MAX];
            if (strata_muse_tensor_name(name, sizeof name, l, kind) < 0) continue;
            const TensorInfo* t = m.find(name);
            if (!t) {
                problems.push_back(std::string("missing ") + name);
                continue;
            }
            ++seen;
            int64_t want[3];
            const int nd = strata_muse_shape(&d, kind, want);
            std::vector<uint64_t> w(want, want + nd);
            if (t->shape != w) problems.push_back(std::string(name) + " is " + shape_str(t->shape) + ", expected " + shape_str(w));
            if (!strata_muse_type_ok(kind, (int) t->type))
                problems.push_back(std::string(name) + " is " + t->type_name() + ", which the engine does not read for it");
            const uint64_t bytes = tensor_payload_bytes(*t);
            if (strata_muse_is_expert(kind)) {
                fp.experts += bytes;
                layer_expert += c.experts ? bytes / (uint64_t) c.experts : 0;
            } else {
                layer_fixed += bytes;
            }
        }
        fp.fixed_layers += layer_fixed;
        fp.fixed_layer_max = std::max(fp.fixed_layer_max, layer_fixed);
        fp.expert_bytes = std::max(fp.expert_bytes, layer_expert);
    }
    auto top = [&](const char* name, std::vector<uint64_t> shape, uint64_t* into) {
        const TensorInfo* t = m.find(name);
        if (!t) { problems.push_back(std::string("missing ") + name); return; }
        ++seen;
        if (t->shape != shape) problems.push_back(std::string(name) + " is " + shape_str(t->shape) + ", expected " + shape_str(shape));
        *into += tensor_payload_bytes(*t);
    };
    const uint64_t vocab = c.vocab ? (uint64_t) c.vocab : 0;
    if (const TensorInfo* out = m.find("output.weight"); out && !vocab && out->shape.size() == 2)
        top("output.weight", {(uint64_t) c.hidden, out->shape[1]}, &fp.lm_head);
    else
        top("output.weight", {(uint64_t) c.hidden, vocab}, &fp.lm_head);
    if (const TensorInfo* e = m.find("token_embd.weight"); e && !vocab && e->shape.size() == 2)
        top("token_embd.weight", {(uint64_t) c.hidden, e->shape[1]}, &fp.token_embd);
    else
        top("token_embd.weight", {(uint64_t) c.hidden, vocab}, &fp.token_embd);
    top("output_norm.weight", {(uint64_t) c.hidden}, &fp.other);
    size_t total = 0;
    for (size_t s = 0; s < m.size(); ++s) total += m.shard(s).tensors().size();
    if (total != seen) problems.push_back(std::to_string(total - seen) + " tensors the engine does not know");
    return problems;
}

/// The two-node row split for node B's RAM (bytes available for its expert rows): the share of every expert's rows
/// node B holds, as whole 256-value blocks (strata_muse_row_split), and what each node then keeps.
struct RowSplit {
    double share_b = 0;      // after rounding to whole blocks
    int64_t first_b = 0;     // node B's first ff row
    uint64_t bytes_a = 0, bytes_b = 0;
};

inline RowSplit plan_row_split(const MuseConfig& c, const Footprint& fp, uint64_t node_b_bytes) {
    RowSplit r;
    const double cap = fp.experts ? std::min(1.0, (double) node_b_bytes / (double) fp.experts) : 0.0;
    const int64_t blocks = c.expert_ff / 256;
    // the largest whole-block share that fits node B
    const int64_t b_blocks = (int64_t) std::floor(cap * (double) blocks + 1e-9);
    r.share_b = blocks ? (double) b_blocks / (double) blocks : 0.0;
    r.first_b = strata_muse_row_split(c.expert_ff, 256, r.share_b);
    r.bytes_b = (uint64_t) ((double) fp.experts * r.share_b);
    r.bytes_a = fp.experts - r.bytes_b;
    return r;
}

}  // namespace strata::muse
