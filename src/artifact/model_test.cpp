#include "strata/core/model.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

namespace {
using Bytes = std::vector<uint8_t>;
template <class T> void put(Bytes &b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof v);
    std::memcpy(b.data() + n, &v, sizeof v);
}
void str(Bytes &b, const std::string &s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
struct Meta {
    Bytes b;
    uint64_t count = 0;
    void key(const std::string &k, uint32_t type) {
        str(b, k);
        put(b, type);
        ++count;
    }
    void number(const std::string &k, uint32_t v) {
        key(k, 4);
        put(b, v);
    }
    void real(const std::string &k, float v) {
        key(k, 6);
        put(b, v);
    }
    void text(const std::string &k, const std::string &v) {
        key(k, 8);
        str(b, v);
    }
    void array(const std::string &k, const std::vector<float> &v) {
        key(k, 9);
        put<uint32_t>(b, 6);
        put<uint64_t>(b, v.size());
        for (float x : v)
            put(b, x);
    }
};
struct Tensor {
    std::string name;
    uint64_t offset, values;
};
Meta metadata(int shard) {
    Meta m;
    m.number("split.count", 2);
    m.number("split.no", shard);
    m.number("split.tensors.count", 4);
    m.text("general.architecture", "glm5next");
    if (shard)
        return m;
    const std::string p = "glm5next.";
    const std::vector<std::pair<std::string, uint32_t>> ints = {{"block_count", 5},
                                                                {"embedding_length", 32},
                                                                {"vocab_size", 100},
                                                                {"context_length", 1024},
                                                                {"expert_count", 2},
                                                                {"expert_used_count", 1},
                                                                {"attention.head_count", 64},
                                                                {"hyper_connection.count", 4},
                                                                {"hyper_connection.sinkhorn_iterations", 20},
                                                                {"attention.kv_lora_rank", 512},
                                                                {"attention.q_lora_rank", 1536},
                                                                {"kda.head_dim", 128},
                                                                {"ssm.conv_kernel", 4},
                                                                {"attention.indexer.head_count", 32},
                                                                {"attention.indexer.key_length", 128},
                                                                {"attention.indexer.kpool", 4},
                                                                {"attention.indexer.top_k", 2048},
                                                                {"nextn_predict_layers", 2},
                                                                {"leading_dense_block_count", 1},
                                                                {"expert_feed_forward_length", 2048},
                                                                {"feed_forward_length", 12288},
                                                                {"rope.dimension_count", 0},
                                                                {"expert_group_count", 1},
                                                                {"expert_group_used_count", 1},
                                                                {"expert_gating_func", 2},
                                                                {"expert_shared_count", 1},
                                                                {"expert_shared_feed_forward_length", 2048}};
    for (const auto &[k, v] : ints)
        m.number(p + k, v);
    m.real(p + "hyper_connection.epsilon", 1e-6);
    m.real(p + "attention.layer_norm_rms_epsilon", 1e-5);
    m.real(p + "kda.gate_lower_bound", -5);
    m.real(p + "expert_weights_scale", 2.5);
    m.array(p + "attention.head_count_kv", {0, 0, 1, 1, 1});
    m.array(p + "swiglu_clamp_exp", {10, 10, 10, 10, 10});
    m.array(p + "swiglu_clamp_shexp", {10, 10, 10, 10, 10});
    m.key(p + "expert_weights_norm", 7);
    put<uint8_t>(m.b, 1);
    return m;
}
void write(const std::filesystem::path &path, const Meta &m, const std::vector<Tensor> &tensors,
           bool payload = true) {
    Bytes b;
    put<uint32_t>(b, 0x46554747);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, tensors.size());
    put(b, m.count);
    b.insert(b.end(), m.b.begin(), m.b.end());
    uint64_t end = 0;
    for (const auto &t : tensors) {
        str(b, t.name);
        put<uint32_t>(b, 1);
        put(b, t.values);
        put<uint32_t>(b, 0);
        put(b, t.offset);
        end = std::max(end, t.offset + t.values * 4);
    }
    if (!tensors.empty() && payload)
        b.resize((b.size() + 31) / 32 * 32 + end);
    std::ofstream(path, std::ios::binary).write((const char *)b.data(), b.size());
}
void require(bool condition, const char *what) {
    if (!condition)
        throw std::runtime_error(what);
}
template <class F> void rejects(F f, const char *what) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, what);
}
} // namespace
int main() {
    const auto dir =
        std::filesystem::temp_directory_path() /
        ("strata-model-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    const auto first = dir / "fixture-00001-of-00002.gguf", second = dir / "fixture-00002-of-00002.gguf";
    try {
        auto m = metadata(0), s = metadata(1);
        const std::vector<Tensor> tensors = {{"blk.1.ffn_gate_exps.weight", 0, 8},
                                             {"output_norm.weight", 32, 8},
                                             {"blk.3.ffn_gate_exps.weight", 64, 8},
                                             {"blk.4.ffn_gate_exps.weight", 96, 8}};
        write(first, m, {});
        write(second, s, tensors);
        {
            strata::core::ModelArtifact a(second);
            const auto &d = a.descriptor();
            const auto c = a.census();
            require(d.layers.size() == 3 && d.draft_layers.size() == 2,
                    "draft blocks leaked into main geometry");
            require(d.layers[0].ffn == strata::core::FfnKind::Dense &&
                        d.layers[1].ffn == strata::core::FfnKind::Moe,
                    "leading dense layer lost");
            require(d.layers[2].mixer == strata::core::MixerKind::SparseMla &&
                        d.layers[0].mixer == strata::core::MixerKind::Kda,
                    "per-layer mixer array ignored");
            require(c.total == 128 && c.main_experts == 32 && c.main_fixed == 32 && c.draft == 64 &&
                        c.selected_expert_bytes == 16,
                    "main/draft tensor census mismatch");
            require(a.at("output_norm.weight").data() != nullptr,
                    "tensorless first shard lost later tensors");
        }
        auto overlap = tensors;
        overlap[1].offset = 16;
        write(second, s, overlap);
        rejects([&] { strata::core::ModelArtifact a(first); }, "overlapping tensors accepted");
        auto duplicate = tensors;
        duplicate[1].name = duplicate[0].name;
        write(second, s, duplicate);
        rejects([&] { strata::core::ModelArtifact a(first); }, "duplicate tensor name accepted");
        write(second, s, tensors, false);
        rejects([&] { strata::core::ModelArtifact a(first); }, "truncated tensor accepted");
        write(second, metadata(0), tensors);
        rejects([&] { strata::core::ModelArtifact a(first); }, "mismatched split number accepted");
        std::filesystem::remove(second);
        rejects([&] { strata::core::ModelArtifact a(first); }, "missing shard accepted");
        std::filesystem::remove_all(dir);
        std::cout << "model artifact tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::filesystem::remove_all(dir);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
