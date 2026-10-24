// include/strata/ds41/config.hpp - DeepSeek-V4.1-Flash geometry from a llama.cpp PR #30817 GGUF (arch deepseek41).
//
// The keys and the derived per-layer sources follow that PR's llama_model_deepseek41::load_arch_hparams: each
// compressed layer reads the KV rows of the latest KV-source layer at or before it and the top-k of the latest
// index-source layer; layer `candidate_source` also picks candidate blocks that later index sources score inside.
// Engram: one hash state per Engram layer, n-grams 2..ngram with `heads` heads each, rows of one shared table
// (per_layer_token_embd) at offsets[col] + hash % primes[col].
// Also reads PR #28696's files (llama.cpp ds41 branch; same architecture name): no kv/index source keys - the source
// layers are those that carry the compressor / indexer tensors - and `engram.*` keys with one table per Engram layer
// (blk.L.engram_embd) and a pad id already through the token map.
// And the GSQ-RCO files (ISTA-DASLab, llama.cpp of 2026-10): `engram.layer_ids` with the hash tables as tensors
// (engram_multipliers / engram_primes / engram_offsets / engram_token_map .weight), `engram_*` scalar keys, the tables
// in a separate Engram GGUF (`engram.file`, merged by ModelArtifact), and the candidate-block keys as
// `attention.candidate_*`.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ds41 {

struct Rope {
    float base = 10000, freq_scale = 1, ext_factor = 0, attn_factor = 1, beta_fast = 0, beta_slow = 0;
    int ctx_orig = 0;
};

struct Config {
    int layers = 0, hidden = 0, vocab = 0, heads = 0, head_dim = 0, rope_dim = 0, q_rank = 0;
    int out_groups = 0, out_rank = 0, window = 0, streams = 0, sinkhorn = 0;
    int experts = 0, top_k = 0, ff = 0;
    float rms_eps = 0, hc_eps = 0, expert_scale = 1;
    bool normalize = true;
    int index_heads = 0, index_dim = 0, index_top_k = 0;
    std::vector<int> ratio, kv_src, topk_src;   // per layer; -1 where the layer has no compressed rows
    int candidate_source = -1, candidate_blocks = 0, candidate_block = 0;
    std::vector<float> clamp, shared_clamp;
    Rope rope_plain, rope_compressed;           // layers with ratio 0 / the others
    // Engram
    std::vector<int> engram_layers;
    int ngram = 0, engram_heads = 0, engram_dim = 0;
    std::vector<uint64_t> multipliers, primes, offsets;   // [engram][ngram], [engram][cols], [engram][cols]
    std::vector<int64_t> token_map;
    int64_t engram_pad = 0;
    bool engram_pad_mapped = false;    // #28696: engram_pad is a compressed id already
    bool engram_per_layer = false;     // #28696: blk.L.engram_embd per Engram layer (else per_layer_token_embd)
    int engram_cols() const { return (ngram - 1) * engram_heads; }
    int engram_index(int layer) const {
        for (size_t e = 0; e < engram_layers.size(); ++e)
            if (engram_layers[e] == layer) return int(e);
        return -1;
    }
    const Rope& rope(int layer) const { return ratio[size_t(layer)] ? rope_compressed : rope_plain; }

    /// has(name): whether the artifact holds a tensor (#28696 files: the source layers are found by their tensors);
    /// ints(name): an integer tensor's values (the GSQ-RCO files' Engram hash tables).
    static Config read(const GgufFile& file, const core::ModelDescriptor& m, const std::function<bool(const std::string&)>& has = {},
                       const std::function<std::vector<int64_t>(const std::string&)>& ints = {}) {
        const std::string p = "deepseek41.";
        auto get = [&](const std::string& k) -> const MetaValue& {
            const auto* v = file.get(p + k);
            if (!v) throw std::runtime_error("ds41: missing " + p + k);
            return *v;
        };
        auto integer = [&](const std::string& k) { return int64_t(get(k).u); };
        auto real = [&](const std::string& k) { return float(get(k).num()); };
        auto array = [&](const std::string& k) {   // every element (MetaValue::items holds only a sample)
            get(k);
            std::vector<uint64_t> out;
            for (int64_t x : file.int_array(p + k)) out.push_back(uint64_t(x));
            return out;
        };
        Config c;
        c.layers = int(m.layers.size());
        c.hidden = int(m.hidden);
        c.vocab = int(m.vocab);
        c.heads = int(m.attention_heads);
        c.head_dim = int(m.kv_rank);
        c.rope_dim = int(m.rope_dim);
        c.q_rank = int(m.q_rank);
        c.streams = int(m.streams);
        c.sinkhorn = int(m.sinkhorn_iterations);
        c.rms_eps = m.rms_epsilon;
        c.hc_eps = m.hc_epsilon;
        c.experts = int(m.experts);
        c.top_k = int(m.top_k);
        c.ff = int(m.layers.front().intermediate);
        c.expert_scale = m.expert_scale;
        c.normalize = m.normalize_expert_weights;
        c.index_heads = int(m.index_heads);
        c.index_dim = int(m.index_dim);
        c.index_top_k = int(m.index_top_k);
        for (const auto& l : m.layers) { c.clamp.push_back(l.swiglu_limit); c.shared_clamp.push_back(l.shared_swiglu_limit); }
        c.out_groups = int(integer("attention.output_group_count"));
        c.out_rank = int(integer("attention.output_lora_rank"));
        c.window = int(integer("attention.sliding_window"));
        for (uint64_t r : array("attention.compress_ratios")) c.ratio.push_back(int(r));
        if (int(c.ratio.size()) < c.layers) throw std::runtime_error("ds41: compress_ratios is shorter than block_count");
        c.ratio.resize(size_t(c.layers));
        std::vector<uint64_t> kv, idx;
        if (file.get(p + "attention.kv_source_layers")) {
            kv = array("attention.kv_source_layers");
            idx = array("attention.indexer.source_layers");
        } else {
            if (!has) throw std::runtime_error("ds41: no kv source keys, and no tensor list to find the source layers");
            for (int l = 0; l < c.layers; ++l) {
                const std::string b = "blk." + std::to_string(l) + ".";
                if (has(b + "attn_compressor_kv.weight")) kv.push_back(uint64_t(l));
                if (has(b + "indexer.attn_q_b.weight")) idx.push_back(uint64_t(l));
            }
        }
        c.kv_src.assign(size_t(c.layers), -1);
        c.topk_src.assign(size_t(c.layers), -1);
        int last_kv = -1, last_idx = -1;
        for (int l = 0; l < c.layers; ++l) {
            const bool is_kv = std::find(kv.begin(), kv.end(), uint64_t(l)) != kv.end();
            const bool is_idx = std::find(idx.begin(), idx.end(), uint64_t(l)) != idx.end();
            const int r = c.ratio[size_t(l)];
            if ((is_kv || is_idx) && r == 0) throw std::runtime_error("ds41: a source layer has compress ratio 0");
            if (is_kv) last_kv = l;
            if (is_idx) last_idx = l;
            if (!r) continue;
            if (r > 2 || last_kv < 0 || last_idx < last_kv || c.ratio[size_t(last_kv)] != r || c.ratio[size_t(last_idx)] != r)
                throw std::runtime_error("ds41: layer " + std::to_string(l) + " has no matching kv/index source");
            c.kv_src[size_t(l)] = last_kv;
            c.topk_src[size_t(l)] = last_idx;
        }
        // STRATA_DS41_NO_CANDIDATES=1: ignore the candidate-block keys (every row stays a top-k candidate), for A/B runs
        const bool no_cand = std::getenv("STRATA_DS41_NO_CANDIDATES") && std::string(std::getenv("STRATA_DS41_NO_CANDIDATES")) == "1";
        if (no_cand) {
        } else if (const auto* v = file.get(p + "attention.indexer.candidate_source_layer")) {
            c.candidate_source = int(int64_t(v->u));
            c.candidate_blocks = int(integer("attention.indexer.candidate_top_k_blocks"));
            c.candidate_block = int(integer("attention.indexer.candidate_block_size"));
        } else if (const auto* v2 = file.get(p + "attention.candidate_source_layer_id")) {
            c.candidate_source = int(int64_t(v2->u));
            c.candidate_blocks = int(integer("attention.candidate_topk_blocks"));
            c.candidate_block = int(integer("attention.candidate_block_size"));
        }
        if (c.candidate_source >= 0) {
            if (c.candidate_source < 0 || c.candidate_source >= c.layers || c.topk_src[size_t(c.candidate_source)] != c.candidate_source ||
                c.candidate_blocks < 1 || c.candidate_block < 1)
                throw std::runtime_error("ds41: invalid candidate selection");
        }
        // RoPE: plain layers use rope.freq_base without YaRN, compressed layers compress_rope_freq_base with YaRN
        // (llama.cpp: yarn_ext_factor 1 for a YaRN model; dsv4_rope_attn_factor cancels the YaRN magnitude scale)
        c.rope_plain.base = real("rope.freq_base");
        c.rope_compressed.base = real("attention.compress_rope_freq_base");
        if (const auto* t = file.get(p + "rope.scaling.type"); t && t->s == "yarn") {
            Rope& r = c.rope_compressed;
            r.freq_scale = 1.0f / real("rope.scaling.factor");
            r.ext_factor = 1.0f;
            r.attn_factor = 1.0f / (1.0f + 0.1f * std::log(1.0f / r.freq_scale));
            r.beta_fast = real("rope.scaling.yarn_beta_fast");
            r.beta_slow = real("rope.scaling.yarn_beta_slow");
            r.ctx_orig = int(integer("rope.scaling.original_context_length"));
        }
        // Engram
        if (file.get(p + "ple.layers")) {
            for (uint64_t l : array("ple.layers")) c.engram_layers.push_back(int(l));
            c.ngram = int(integer("ple.ngram_size"));
            c.engram_heads = int(integer("ple.heads_per_ngram"));
            c.engram_dim = int(integer("embedding_length_per_layer_input"));
            c.multipliers = array("ple.layer_multipliers");
            c.offsets = array("ple.head_offsets");
            c.primes = array("ple.head_vocab_sizes");
            c.engram_pad = integer("engram.pad_token_id");
            get("engram.token_map");
            c.token_map = file.int_array(p + "engram.token_map");
            const size_t n = c.engram_layers.size(), cols = size_t(c.engram_cols());
            if (c.ngram < 2 || c.engram_heads < 1 || c.engram_dim < 1 || c.multipliers.size() != n * size_t(c.ngram) ||
                c.primes.size() != n * cols || c.offsets.size() != n * cols || size_t(c.engram_pad) >= c.token_map.size())
                throw std::runtime_error("ds41: inconsistent Engram metadata");
            for (size_t i = 0; i < n * cols; ++i)
                if (c.primes[i] < 2 || c.offsets[i] + c.primes[i] > uint64_t(INT32_MAX))
                    throw std::runtime_error("ds41: Engram bucket range out of range");
        } else if (file.get(p + "engram.layer_ids") && file.get(p + "engram_max_ngram_size")) {   // GSQ-RCO
            for (uint64_t l : array("engram.layer_ids")) c.engram_layers.push_back(int(l));
            c.ngram = int(integer("engram_max_ngram_size"));
            c.engram_heads = int(integer("engram_n_heads"));
            c.engram_dim = int(integer("engram_head_dim"));
            c.engram_pad = integer("engram_pad_token_id");   // a vocabulary id (through the token map)
            c.engram_per_layer = true;
            if (!ints) throw std::runtime_error("ds41: the Engram hash tables are tensors, and no tensor reader was given");
            auto u64 = [&](const char* n) {
                std::vector<uint64_t> out;
                for (int64_t x : ints(n)) out.push_back(uint64_t(x));
                return out;
            };
            c.multipliers = u64("engram_multipliers.weight");
            c.primes = u64("engram_primes.weight");
            c.offsets = u64("engram_offsets.weight");
            c.token_map = ints("engram_token_map.weight");
            const size_t n = c.engram_layers.size(), cols = size_t(c.engram_cols());
            if (c.ngram < 2 || c.engram_heads < 1 || c.engram_dim < 1 || c.multipliers.size() != n * size_t(c.ngram) ||
                c.primes.size() != n * cols || c.offsets.size() != n * cols || c.engram_pad < 0 || size_t(c.engram_pad) >= c.token_map.size())
                throw std::runtime_error("ds41: inconsistent Engram metadata");
            for (size_t i = 0; i < n * cols; ++i)
                if (c.primes[i] < 2 || c.offsets[i] + c.primes[i] > uint64_t(INT32_MAX))
                    throw std::runtime_error("ds41: Engram bucket range out of range");
        } else if (file.get(p + "engram.layer_ids")) {   // #28696
            for (uint64_t l : array("engram.layer_ids")) c.engram_layers.push_back(int(l));
            c.ngram = int(integer("engram.max_ngram_size"));
            c.engram_heads = int(integer("engram.head_count"));
            c.engram_dim = int(integer("engram.key_length"));
            c.multipliers = array("engram.multipliers");
            c.primes = array("engram.primes");
            c.offsets = array("engram.offsets");
            c.engram_pad = integer("engram.pad_id");
            c.engram_pad_mapped = c.engram_per_layer = true;
            get("engram.token_map");
            c.token_map = file.int_array(p + "engram.token_map");
            const size_t n = c.engram_layers.size(), cols = size_t(c.engram_cols());
            if (c.ngram < 2 || c.engram_heads < 1 || c.engram_dim < 1 || c.multipliers.size() != n * size_t(c.ngram) ||
                c.primes.size() != n * cols || c.offsets.size() != n * cols)
                throw std::runtime_error("ds41: inconsistent Engram metadata");
            for (size_t i = 0; i < n * cols; ++i)
                if (c.primes[i] < 2 || c.offsets[i] + c.primes[i] > uint64_t(INT32_MAX))
                    throw std::runtime_error("ds41: Engram bucket range out of range");
        }
        if (c.heads < 1 || c.head_dim < c.rope_dim || c.rope_dim % 2 || c.index_dim < c.rope_dim || c.out_groups < 1 ||
            c.heads % c.out_groups || c.window < 1 || c.streams != 4)
            throw std::runtime_error("ds41: unsupported geometry");
        return c;
    }
};

}  // namespace strata::ds41
