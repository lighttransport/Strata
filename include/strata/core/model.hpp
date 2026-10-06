#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/glm_native.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <regex>
#include <set>

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

    static ModelDescriptor native(const artifact::Json& root) {
        const auto& c=root.at("text_config");ModelDescriptor m;m.architecture="glm5next";
        auto n=[&](const char* k){return static_cast<int64_t>(c.at(k).integer());};
        m.hidden=n("hidden_size");m.vocab=n("vocab_size");m.context=n("max_position_embeddings");
        m.experts=n("n_routed_experts");m.top_k=n("num_experts_per_tok");m.streams=n("hc_mult");
        m.attention_heads=n("num_attention_heads");m.kv_rank=n("kv_lora_rank");m.q_rank=n("q_lora_rank");
        const auto& a=c.at("linear_attn_config");m.linear_heads=a.at("num_heads").integer();m.linear_dim=a.at("head_dim").integer();
        m.conv_kernel=a.at("short_conv_kernel_size").integer();m.gate_lower_bound=a.at("gate_lower_bound").real();
        m.index_heads=n("index_n_heads");m.index_dim=n("index_head_dim");m.index_pool=n("index_kpool");m.index_top_k=n("index_topk");
        m.sinkhorn_iterations=n("hc_sinkhorn_iters");m.hc_epsilon=c.at("hc_eps").real();m.rms_epsilon=c.at("rms_norm_eps").real();
        m.expert_scale=c.at("routed_scaling_factor").real();m.rope_dim=n("qk_rope_head_dim");m.expert_groups=n("n_group");m.selected_groups=n("topk_group");
        m.gating_function=2;m.normalize_expert_weights=c.at("norm_topk_prob").flag();m.shared_experts=n("n_shared_experts");
        m.shared_intermediate=n("moe_intermediate_size");m.residual=ResidualKind::Mhc;
        for(int64_t l=0;l<n("num_hidden_layers");++l)m.layers.push_back({c.at("layer_types").array.at(l).string()=="linear_attention"?MixerKind::Kda:MixerKind::SparseMla,
            l<n("first_k_dense_replace")?FfnKind::Dense:FfnKind::Moe,l<n("first_k_dense_replace")?n("intermediate_size"):n("moe_intermediate_size"),c.at("swiglu_limit").real(),c.at("swiglu_limit").real()});
        return m;
    }
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
        // Some converters (REAP-pruned GLM-5.3 GGUFs) spell the GLM architecture with a hyphen; its keys use it.
        const std::string p = m.architecture + ".";
        if (m.architecture == "glm5-next") m.architecture = "glm5next";
        if (m.architecture != "glm5next" && m.architecture != "qwen4exp")
            throw std::runtime_error("model: unsupported architecture " + m.architecture);
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
        // Newer converters (e.g. REAP-pruned GGUFs) name these attention.hc.*, ssm.gate_lower_bound and a scalar
        // swiglu_limit; the first name is the original GLM-5.3-Flash GGUF's.
        auto either = [&](const std::string &key, const std::string &other) { return file.get(p + key) ? p + key : p + other; };
        m.streams = integer(either("hyper_connection.count", "attention.hc.mult"));
        m.sinkhorn_iterations = integer(either("hyper_connection.sinkhorn_iterations", "attention.hc.sinkhorn_iters"));
        m.hc_epsilon = real(either("hyper_connection.epsilon", "attention.hc.eps"));
        m.rms_epsilon = real(p + "attention.layer_norm_rms_epsilon");
        m.kv_rank = integer(p + "attention.kv_lora_rank");
        m.q_rank = integer(p + "attention.q_lora_rank");
        m.linear_heads = m.attention_heads;
        m.linear_dim = integer(p + "kda.head_dim");
        m.gate_lower_bound = real(either("kda.gate_lower_bound", "ssm.gate_lower_bound"));
        m.conv_kernel = integer(p + "ssm.conv_kernel");
        m.index_heads = integer(p + "attention.indexer.head_count");
        m.index_dim = integer(p + "attention.indexer.key_length");
        // GLM-5.3-Flash's indexer pools 4 keys; REAP GGUFs omit the key.
        m.index_pool = file.get(p + "attention.indexer.kpool") ? integer(p + "attention.indexer.kpool") : 4;
        m.index_top_k = integer(p + "attention.indexer.top_k");
        m.expert_scale = real(p + "expert_weights_scale");
        m.rope_dim = integer(p + "rope.dimension_count");
        m.expert_groups = integer(p + "expert_group_count");
        m.selected_groups = integer(p + "expert_group_used_count");
        m.gating_function = integer(p + "expert_gating_func");
        m.shared_intermediate = file.get(p + "expert_shared_feed_forward_length")
                                    ? integer(p + "expert_shared_feed_forward_length")
                                    : integer(p + "expert_feed_forward_length");
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
        // A scalar swiglu_limit applies to routed and shared experts of every layer.
        const float scalar_limit = !clamp && !shared_clamp && file.get(p + "swiglu_limit") ? real(p + "swiglu_limit") : 0.f;
        if (!kv || kv->type != MetaType::ARRAY || kv->count != (uint64_t)blocks || kv->items.size() != kv->count ||
            (!scalar_limit && (!clamp || clamp->type != MetaType::ARRAY || clamp->count != (uint64_t)blocks ||
                               clamp->items.size() != clamp->count)))
            throw std::runtime_error("model: incomplete per-layer attention or clamp metadata");
        if (!scalar_limit && (!shared_clamp || shared_clamp->type != MetaType::ARRAY ||
            shared_clamp->count != (uint64_t)blocks || shared_clamp->items.size() != shared_clamp->count))
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
                                  scalar_limit ? scalar_limit : (float)clamp->items[(size_t)l].num(),
                                  scalar_limit ? scalar_limit : (float)shared_clamp->items[(size_t)l].num()};
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
    const artifact::GlmNative::Binding* native = nullptr;
    const uint8_t *data() const { return native ? native->data() : file->tensor_data(*tensor); }
};

struct ModelCensus {
    uint64_t main_experts = 0, main_fixed = 0, draft = 0, total = 0;
    uint64_t selected_expert_bytes = 0;
};

// Owns the mappings; tensor references remain valid for this object's lifetime.
class ModelArtifact {
  public:
    explicit ModelArtifact(const std::filesystem::path &shard) {
        if(std::filesystem::is_directory(shard)) {
            native_=std::make_unique<artifact::GlmNative>(shard);descriptor_=ModelDescriptor::native(native_->model.config);
            for(const auto& [name,b]:native_->bindings)tensors_.emplace(name,ArtifactTensor{nullptr,&b.info,b.info.elements()*(b.info.type==0?4:2),&b});
            return;
        }
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
                if ((a->s == "glm5-next" ? std::string("glm5next") : a->s) != descriptor_.architecture)
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

    bool is_exl3()const { return bool(native_); }
    artifact::GlmExl3& exl3() { if(!native_)throw std::runtime_error("not EXL3");return native_->model; }
    const ModelDescriptor &descriptor() const { return descriptor_; }
    const std::map<std::string, ArtifactTensor> &tensors() const { return tensors_; }
    // Bind an experimental converted expert sidecar to this exact source header
    // and file generation. The original GGUF stays immutable.
    uint64_t source_fingerprint() const {
        uint64_t hash=14695981039346656037ULL;
        auto add=[&](const void* data,size_t size){auto* p=static_cast<const uint8_t*>(data);for(size_t i=0;i<size;++i){hash^=p[i];hash*=1099511628211ULL;}};
        for(const auto& f:files_) {
            const auto size=f->file_size();add(&size,sizeof size);
            const auto stamp=std::filesystem::last_write_time(f->path()).time_since_epoch().count();add(&stamp,sizeof stamp);
            // Metadata-only split shards may omit padding after the last key.
            std::ifstream input(f->path(),std::ios::binary);std::vector<char> header(std::min(f->data_start(),f->file_size()));
            if(!input.read(header.data(),header.size()))throw std::runtime_error("expert pack: cannot fingerprint source");
            add(header.data(),header.size());
        }
        return hash;
    }
    void overlay_experts(const std::filesystem::path& path,const std::set<std::string>& retained={}) {
        if(native_ || !expert_pack_.empty())throw std::invalid_argument("expert pack requires an original GGUF model");
        auto pack=std::make_unique<GgufFile>(path.string());
        if(pack->tensors().empty())throw std::runtime_error("expert pack: empty sidecar");
        const auto* version=pack->get("strata.expert_pack.version"),*source=pack->get("strata.expert_pack.source");
        if(!version||version->type!=MetaType::U32||version->u!=1||!source||source->type!=MetaType::U64||source->u!=source_fingerprint())
            throw std::runtime_error("expert pack: version or source fingerprint mismatch");
        std::set<std::string> names;std::map<uint64_t,uint64_t> spans;
        for(const auto& t:pack->tensors()) {
            if(!names.insert(t.name).second)throw std::runtime_error("expert pack: duplicate tensor");
            const auto& original=at(t.name);std::smatch match;
            if(!std::regex_match(t.name,match,std::regex("blk\\.([0-9]+)\\.ffn_(gate|up|down)_exps\\.weight"))||std::stoi(match[1])<3||std::stoi(match[1])>44||t.shape!=original.tensor->shape||t.shape.size()!=3)
                throw std::runtime_error("expert pack: invalid projection "+t.name);
            // IQ2_XS gate/up -> Q2_K and IQ3_XXS down -> Q3_K (the UD-Q2_K_XL GGUF), or a K-quant source
            // (Q4_K/Q5_K/Q6_K, e.g. the REAP-50 Q4_K_M GGUF) -> the same targets.
            const bool kquant=original.tensor->type==12||original.tensor->type==13||original.tensor->type==14;
            const bool valid=((original.tensor->type==17||kquant) && t.type==10 && match[2]!="down") ||
                             ((original.tensor->type==18||kquant) && t.type==11 && match[2]=="down");
            if(!valid)throw std::runtime_error("expert pack: invalid type transition "+t.name);
            const uint64_t length=t.elements()/256*(t.type==10?84:110);
            if(t.offset%32||pack->data_start()>pack->file_size()||t.offset>pack->file_size()-pack->data_start()||length>pack->file_size()-pack->data_start()-t.offset||!spans.emplace(t.offset,length).second)
                throw std::runtime_error("expert pack: invalid payload bounds");
            const auto* checksum=pack->get("strata.expert_pack.hash."+t.name);
            if(!checksum||checksum->type!=MetaType::U64)throw std::runtime_error("expert pack: missing payload checksum");
        }
        uint64_t end=0;for(auto [offset,length]:spans){if(offset<end)throw std::runtime_error("expert pack: overlapping tensors");end=offset+length;}
        for(const auto& name:retained)if(!names.contains(name))throw std::runtime_error("expert pack: unknown retained projection "+name);
        // Gate/up share a NativeFmt. Their representation must stay paired.
        for(int l=3;l<=44;++l){auto p="blk."+std::to_string(l)+".ffn_";if(names.contains(p+"gate_exps.weight")!=names.contains(p+"up_exps.weight")||retained.contains(p+"gate_exps.weight")!=retained.contains(p+"up_exps.weight"))throw std::runtime_error("expert pack: gate/up must be paired");}
        for(const auto& t:pack->tensors())if(!retained.contains(t.name)) {
            auto& entry=tensors_.at(t.name);original_experts_.emplace(t.name,entry);
            entry={pack.get(),&t,t.elements()/256*(t.type==10?84:110)};
        }
        expert_pack_=path.string();files_.push_back(std::move(pack));
    }
    bool has_expert_pack()const{return !expert_pack_.empty();}
    void discard_original_experts()const {for(const auto& [name,t]:original_experts_)t.file->discard_tensor_pages(*t.tensor,t.bytes);}
    const ArtifactTensor &at(const std::string &name) const {
        auto i = tensors_.find(name);
        if (i == tensors_.end() && name.ends_with(".weight")) {
            // REAP-pruned GLM GGUFs name hc_* tensors without ".weight" and the indexer compressor indexer.kpool_*.
            const std::string stem = name.substr(0, name.size() - 7);
            if (stem.find(".hc_") != std::string::npos) i = tensors_.find(stem);
            else if (const auto k = stem.find("indexer_compressor_"); k != std::string::npos)
                i = tensors_.find(stem.substr(0, k) + "indexer.kpool_" + stem.substr(k + 19));
        }
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
        if(native_) { const auto& n=native_->model;c.main_experts=n.experts_bytes;c.main_fixed=n.fixed_bytes;c.draft=n.draft_bytes;c.total=n.tensors.bytes();c.selected_expert_bytes=c.main_experts/descriptor_.experts*descriptor_.top_k;return c; }
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
    std::unique_ptr<artifact::GlmNative> native_;
    ModelDescriptor descriptor_;
    std::vector<std::unique_ptr<GgufFile>> files_;
    std::map<std::string, ArtifactTensor> tensors_;
    std::map<std::string, ArtifactTensor> original_experts_;
    std::string expert_pack_;
};

} // namespace strata::core
