#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <regex>

namespace strata::core {

enum class MixerKind { Gdn, Qsa, Kda, SparseMla };
enum class FfnKind { Dense, Moe };
enum class ResidualKind { Gated, Mhc };

struct LayerDescriptor {
    MixerKind mixer;
    FfnKind ffn;
    int64_t intermediate;
    float swiglu_limit = 0;
    float shared_swiglu_limit = 0;
};

struct ModelDescriptor {
    std::string architecture;
    int64_t hidden = 0, vocab = 0, context = 0;
    int64_t experts = 0, top_k = 0, streams = 0;
    int64_t attention_heads = 0, kv_rank = 0, q_rank = 0;
    int64_t linear_heads = 0, linear_dim = 0, conv_kernel = 0;
    int64_t index_heads = 0, index_dim = 0, index_pool = 0, index_top_k = 0;
    int64_t sinkhorn_iterations = 0;
    int64_t rope_dim = 0, expert_groups = 1, selected_groups = 1, gating_function = 0;
    int64_t shared_intermediate = 0, shared_experts = 0;
    bool normalize_expert_weights = false;
    float rms_epsilon = 0, hc_epsilon = 0, expert_scale = 1, gate_lower_bound = 0;
    ResidualKind residual = ResidualKind::Gated;
    bool has_ngram = false;
    std::vector<LayerDescriptor> layers, draft_layers;

    static ModelDescriptor read(const GgufFile &file) {
        auto integer = [&](const std::string &key) -> int64_t {
            const auto *v = file.get(key);
            if (!v ||
                (v->type != MetaType::U32 && v->type != MetaType::U64 && v->type != MetaType::I32 &&
                 v->type != MetaType::I64) ||
                v->u > (uint64_t)(std::numeric_limits<int64_t>::max)())
                throw std::runtime_error("model: missing or invalid integer " + key);
            return (int64_t)v->u;
        };
        auto real = [&](const std::string &key) -> float {
            const auto *v = file.get(key);
            if (!v || (v->type != MetaType::F32 && v->type != MetaType::F64))
                throw std::runtime_error("model: missing or invalid float " + key);
            if (!std::isfinite(v->f) || !std::isfinite((float)v->f))
                throw std::runtime_error("model: non-finite " + key);
            return (float)v->f;
        };
        const auto *a = file.get("general.architecture");
        if (!a || a->type != MetaType::STRING)
            throw std::runtime_error("model: missing architecture");
        ModelDescriptor m;
        m.architecture = a->s;
        if (m.architecture != "glm5next" && m.architecture != "qwen4exp")
            throw std::runtime_error("model: unsupported architecture " + m.architecture);
        const std::string p = m.architecture + ".";
        const int64_t blocks = integer(p + "block_count");
        m.hidden = integer(p + "embedding_length");
        m.context = integer(p + "context_length");
        m.experts = integer(p + "expert_count");
        m.top_k = integer(p + "expert_used_count");
        m.attention_heads = integer(p + "attention.head_count");
        if (const auto *v = file.get(p + "vocab_size"))
            m.vocab = (int64_t)v->u;
        else if (const auto *v = file.get("tokenizer.ggml.tokens"))
            m.vocab = (int64_t)v->count;
        if (blocks < 1 || blocks > 64 || m.hidden < 1 || m.experts < 1 || m.top_k < 1 ||
            m.top_k > m.experts || m.vocab < 1 || m.context < 1)
            throw std::runtime_error("model: invalid geometry");
        if (m.architecture == "qwen4exp") {
            const auto error = check_architecture(file);
            if (!error.empty())
                throw std::runtime_error(error);
            m.streams = 4;
            m.has_ngram = true;
            for (int64_t l = 0; l < blocks; ++l)
                m.layers.push_back({l % 4 == 3 ? MixerKind::Qsa : MixerKind::Gdn, FfnKind::Moe, 640});
            return m;
        }
        m.residual = ResidualKind::Mhc;
        m.streams = integer(p + "hyper_connection.count");
        m.sinkhorn_iterations = integer(p + "hyper_connection.sinkhorn_iterations");
        m.hc_epsilon = real(p + "hyper_connection.epsilon");
        m.rms_epsilon = real(p + "attention.layer_norm_rms_epsilon");
        m.kv_rank = integer(p + "attention.kv_lora_rank");
        m.q_rank = integer(p + "attention.q_lora_rank");
        m.linear_heads = m.attention_heads;
        m.linear_dim = integer(p + "kda.head_dim");
        m.gate_lower_bound = real(p + "kda.gate_lower_bound");
        m.conv_kernel = integer(p + "ssm.conv_kernel");
        m.index_heads = integer(p + "attention.indexer.head_count");
        m.index_dim = integer(p + "attention.indexer.key_length");
        m.index_pool = integer(p + "attention.indexer.kpool");
        m.index_top_k = integer(p + "attention.indexer.top_k");
        m.expert_scale = real(p + "expert_weights_scale");
        m.rope_dim = integer(p + "rope.dimension_count");
        m.expert_groups = integer(p + "expert_group_count");
        m.selected_groups = integer(p + "expert_group_used_count");
        m.gating_function = integer(p + "expert_gating_func");
        m.shared_intermediate = integer(p + "expert_shared_feed_forward_length");
        m.shared_experts = integer(p + "expert_shared_count");
        const auto *normalized = file.get(p + "expert_weights_norm");
        if (!normalized || normalized->type != MetaType::BOOL)
            throw std::runtime_error("model: missing expert weight normalization");
        m.normalize_expert_weights = normalized->u != 0;
        const int64_t draft = integer(p + "nextn_predict_layers");
        const int64_t dense = integer(p + "leading_dense_block_count");
        const int64_t moe_ff = integer(p + "expert_feed_forward_length");
        const int64_t dense_ff = integer(p + "feed_forward_length");
        const auto *kv = file.get(p + "attention.head_count_kv");
        const auto *clamp = file.get(p + "swiglu_clamp_exp");
        const auto *shared_clamp = file.get(p + "swiglu_clamp_shexp");
        if (!kv || kv->type != MetaType::ARRAY || kv->count != (uint64_t)blocks ||
            kv->items.size() != kv->count || !clamp || clamp->type != MetaType::ARRAY ||
            clamp->count != (uint64_t)blocks || clamp->items.size() != clamp->count)
            throw std::runtime_error("model: incomplete per-layer attention or clamp metadata");
        if (!shared_clamp || shared_clamp->type != MetaType::ARRAY ||
            shared_clamp->count != (uint64_t)blocks || shared_clamp->items.size() != shared_clamp->count)
            throw std::runtime_error("model: incomplete shared-expert clamp metadata");
        if (draft < 0 || draft >= blocks || dense < 0 || dense > blocks - draft || m.streams != 4 ||
            m.sinkhorn_iterations < 1 || m.hc_epsilon <= 0 || m.rms_epsilon <= 0 || m.linear_dim < 1 ||
            m.conv_kernel < 1 || m.kv_rank < 1 || m.q_rank < 1 || m.index_pool < 1 || m.index_dim < 1 ||
            m.index_heads < 1 || m.index_top_k < m.index_pool || m.attention_heads < 1 ||
            m.expert_scale <= 0 || m.gate_lower_bound >= 0 || moe_ff < 1 || dense_ff < 1)
            throw std::runtime_error("model: unsupported GLM geometry");
        for (int64_t l = 0; l < blocks; ++l) {
            const auto heads = kv->items[(size_t)l].num();
            if (heads != 0 && heads != 1)
                throw std::runtime_error("model: unsupported MLA KV head count");
            LayerDescriptor layer{heads == 0 ? MixerKind::Kda : MixerKind::SparseMla,
                                  l < dense ? FfnKind::Dense : FfnKind::Moe, l < dense ? dense_ff : moe_ff,
                                  (float)clamp->items[(size_t)l].num(),
                                  (float)shared_clamp->items[(size_t)l].num()};
            if (!(layer.swiglu_limit > 0) || !std::isfinite(layer.swiglu_limit))
                throw std::runtime_error("model: invalid SwiGLU limit");
            if (!(layer.shared_swiglu_limit > 0) || !std::isfinite(layer.shared_swiglu_limit))
                throw std::runtime_error("model: invalid shared SwiGLU limit");
            (l < blocks - draft ? m.layers : m.draft_layers).push_back(layer);
        }
        return m;
    }
};

struct ArtifactTensor {
    const GgufFile *file;
    const TensorInfo *tensor;
    uint64_t bytes;
    const uint8_t *data() const { return file->tensor_data(*tensor); }
};

struct ModelCensus {
    uint64_t main_experts = 0, main_fixed = 0, draft = 0, total = 0;
    uint64_t selected_expert_bytes = 0;
};

// Owns the mappings; tensor references remain valid for this object's lifetime.
class ModelArtifact {
  public:
    explicit ModelArtifact(const std::filesystem::path &shard) {
        const std::regex split("^(.*)-([0-9]{5})-of-([0-9]{5})\\.gguf$");
        std::smatch match;
        const std::string filename = shard.filename().string();
        std::vector<std::filesystem::path> paths;
        if (std::regex_match(filename, match, split)) {
            const int count = std::stoi(match[3]);
            if (count < 1 || count > 1024)
                throw std::runtime_error("model: invalid split count");
            for (int i = 1; i <= count; ++i) {
                char suffix[40];
                std::snprintf(suffix, sizeof suffix, "-%05d-of-%05d.gguf", i, count);
                paths.push_back(shard.parent_path() / (match[1].str() + suffix));
            }
        } else
            paths.push_back(shard);
        uint64_t declared_tensors = 0;
        for (size_t i = 0; i < paths.size(); ++i) {
            auto file = std::make_unique<GgufFile>(paths[i].string());
            if (i == 0) {
                descriptor_ = ModelDescriptor::read(*file);
                if (const auto *v = file->get("split.tensors.count"))
                    declared_tensors = v->u;
            }
            if (paths.size() > 1) {
                const auto *count = file->get("split.count");
                const auto *number = file->get("split.no");
                const auto *total = file->get("split.tensors.count");
                if (!count || !number || !total || count->u != paths.size() || number->u != i ||
                    total->u != declared_tensors)
                    throw std::runtime_error("model: inconsistent split metadata in " + paths[i].string());
            }
            if (const auto *a = file->get("general.architecture"))
                if (a->s != descriptor_.architecture)
                    throw std::runtime_error("model: shard architecture mismatch");
            std::map<uint64_t, uint64_t> spans;
            for (const auto &t : file->tensors()) {
                int block = 0, bytes = 0;
                if (!block_geometry(t.type, block, bytes) || t.shape.empty() || !t.shape[0] ||
                    t.shape[0] % block)
                    throw std::runtime_error("model: unsupported block geometry for " + t.name);
                uint64_t elements = 1;
                for (const uint64_t d : t.shape) {
                    if (!d || elements > (std::numeric_limits<uint64_t>::max)() / d)
                        throw std::runtime_error("model: invalid extent for " + t.name);
                    elements *= d;
                }
                if (elements / block > (std::numeric_limits<uint64_t>::max)() / bytes)
                    throw std::runtime_error("model: byte count overflow for " + t.name);
                const uint64_t length = elements / block * bytes;
                if (file->data_start() > file->file_size() ||
                    t.offset > file->file_size() - file->data_start() ||
                    length > file->file_size() - file->data_start() - t.offset)
                    throw std::runtime_error("model: truncated tensor " + t.name);
                if (!spans.emplace(t.offset, length).second)
                    throw std::runtime_error("model: duplicate tensor offset " + t.name);
                if (!tensors_.emplace(t.name, ArtifactTensor{file.get(), &t, length}).second)
                    throw std::runtime_error("model: duplicate tensor " + t.name);
            }
            uint64_t end = 0;
            for (const auto &[offset, length] : spans) {
                if (offset < end)
                    throw std::runtime_error("model: overlapping tensor payloads");
                end = offset + length;
            }
            files_.push_back(std::move(file));
        }
        if (declared_tensors && tensors_.size() != declared_tensors)
            throw std::runtime_error("model: split tensor count mismatch");
    }

    const ModelDescriptor &descriptor() const { return descriptor_; }
    const std::map<std::string, ArtifactTensor> &tensors() const { return tensors_; }
    const ArtifactTensor &at(const std::string &name) const {
        const auto i = tensors_.find(name);
        if (i == tensors_.end())
            throw std::runtime_error("model: missing tensor " + name);
        return i->second;
    }
    void shape(const std::string &name, std::initializer_list<uint64_t> expected) const {
        if (at(name).tensor->shape != std::vector<uint64_t>(expected))
            throw std::runtime_error("model: incompatible shape for " + name);
    }
    ModelCensus census() const {
        ModelCensus c;
        for (const auto &[name, t] : tensors_) {
            c.total += t.bytes;
            bool draft = false;
            for (size_t l = 0; l < descriptor_.draft_layers.size(); ++l)
                draft |= name.starts_with("blk." + std::to_string(descriptor_.layers.size() + l) + ".");
            if (draft)
                c.draft += t.bytes;
            else if (name.ends_with("_exps.weight")) {
                c.main_experts += t.bytes;
                c.selected_expert_bytes += t.bytes / descriptor_.experts * descriptor_.top_k;
            } else
                c.main_fixed += t.bytes;
        }
        return c;
    }

  private:
    ModelDescriptor descriptor_;
    std::vector<std::unique_ptr<GgufFile>> files_;
    std::map<std::string, ArtifactTensor> tensors_;
};

} // namespace strata::core
