#include "strata/core/model.hpp"
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--tensors")) {
        std::cerr << "usage: strata-model-inspect <any model shard.gguf> [--tensors]\n";
        return 2;
    }
    try {
        strata::core::ModelArtifact artifact(argv[1]);
        const auto &m = artifact.descriptor();
        const auto c = artifact.census();
        size_t linear = 0, sparse = 0, dense = 0;
        for (const auto &l : m.layers) {
            linear += l.mixer == strata::core::MixerKind::Kda || l.mixer == strata::core::MixerKind::Gdn;
            sparse +=
                l.mixer == strata::core::MixerKind::SparseMla || l.mixer == strata::core::MixerKind::Qsa;
            dense += l.ffn == strata::core::FfnKind::Dense;
        }
        std::cout << "architecture=" << m.architecture << " hidden=" << m.hidden << " vocab=" << m.vocab
                  << " context=" << m.context << " main_layers=" << m.layers.size()
                  << " draft_layers=" << m.draft_layers.size() << " linear_layers=" << linear
                  << " sparse_layers=" << sparse << " dense_ffn_layers=" << dense << " experts=" << m.experts
                  << " top_k=" << m.top_k << '\n';
        std::cout << "tensor_count=" << artifact.tensors().size() << " main_expert_bytes=" << c.main_experts
                  << " main_fixed_bytes=" << c.main_fixed << " draft_bytes=" << c.draft
                  << " total_bytes=" << c.total
                  << " selected_expert_bytes_per_token=" << c.selected_expert_bytes << '\n';
        if (argc == 3)
            for (const auto &[name, t] : artifact.tensors()) {
                std::cout << name << " type=" << t.tensor->type << " bytes=" << t.bytes << " shape=";
                for (auto dim : t.tensor->shape)
                    std::cout << dim << ',';
                std::cout << '\n';
            }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
