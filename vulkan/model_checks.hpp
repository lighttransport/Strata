#pragma once
#include "strata/core/model.hpp"
#include <stdexcept>
namespace strata::vulkan {
// Validate every host-indexed dimension before creating a Vulkan device. A
// different or malformed GGUF must not turn a metadata mismatch into a read
// past a vector, an expert row, or a mapped tensor.
inline const core::ModelDescriptor &
checked_model(const core::ModelArtifact &artifact) {
  const auto &m = artifact.descriptor();
  if (artifact.is_exl3() || m.architecture != "glm5next" || m.streams != 4 ||
      m.rope_dim != 0 || m.linear_dim != 128 || m.linear_heads > 256 ||
      m.attention_heads > 256 || m.hidden > 8192 || m.vocab > 2000000 ||
      m.kv_rank > 4096 || m.q_rank > 8192 || m.index_dim > 4096 ||
      m.index_pool > 64 || m.index_top_k > 2048 || m.conv_kernel > 32 ||
      m.sinkhorn_iterations > 128 || m.top_k > 16 || m.experts != 144 ||
      m.shared_experts != 1)
    throw std::invalid_argument(
        "Vulkan initial runner requires a bounded GLM5next NoPE REAP50 GGUF");
  const uint64_t H = m.hidden, N = m.linear_heads * m.linear_dim,
                 D = m.linear_dim, heads = m.attention_heads;
  auto shape = [&](const std::string &n,
                   std::initializer_list<uint64_t> expected) {
    artifact.shape(n, expected);
  };
  shape("token_embd.weight", {H, uint64_t(m.vocab)});
  shape("output.weight", {H, uint64_t(m.vocab)});
  shape("output_norm.weight", {H});
  for (size_t l = 0; l < m.layers.size(); ++l) {
    const auto p = "blk." + std::to_string(l) + ".";
    for (const char *kind : {"attn", "ffn"}) {
      shape(p + "hc_" + kind + "_fn.weight", {4 * H, 24});
      shape(p + "hc_" + kind + "_base.weight", {24});
      shape(p + "hc_" + kind + "_scale.weight", {3});
      shape(p + kind + "_norm.weight", {H});
    }
    if (m.layers[l].mixer == core::MixerKind::Kda) {
      for (const char *part : {"q", "k", "v"}) {
        shape(p + "attn_" + part + ".weight", {H, N});
        shape(p + "ssm_conv1d_" + part + ".weight",
              {uint64_t(m.conv_kernel), 1, N});
      }
      shape(p + "ssm_a", {uint64_t(m.linear_heads)});
      shape(p + "ssm_dt.bias", {N});
      shape(p + "ssm_beta.weight", {H, uint64_t(m.linear_heads)});
      shape(p + "ssm_norm.weight", {D});
      for (const char *part : {"f", "g"}) {
        shape(p + "ssm_" + part + "_a.weight", {H, D});
        shape(p + "ssm_" + part + "_b.weight", {D, N});
      }
      shape(p + "attn_output.weight", {N, H});
    } else if (m.layers[l].mixer == core::MixerKind::SparseMla) {
      const uint64_t Q = m.q_rank, V = m.kv_rank, I = m.index_dim,
                     IH = m.index_heads;
      shape(p + "attn_q_a.weight", {H, Q});
      shape(p + "attn_q_a_norm.weight", {Q});
      shape(p + "attn_q_b.weight", {Q, heads * 256});
      shape(p + "attn_kv_a_mqa.weight", {H, V});
      shape(p + "attn_kv_a_norm.weight", {V});
      shape(p + "attn_k_b.weight", {256, V, heads});
      shape(p + "attn_v_b.weight", {V, 256, heads});
      shape(p + "attn_output.weight", {heads * 256, H});
      shape(p + "indexer.attn_q_b.weight", {Q, IH * I});
      shape(p + "indexer.attn_k.weight", {H, I});
      shape(p + "indexer.proj.weight", {H, IH});
      shape(p + "indexer.k_norm.weight", {I});
      shape(p + "indexer.k_norm.bias", {I});
      shape(p + "indexer_compressor_gate.weight", {H, I});
      shape(p + "indexer_compressor_ape.weight", {I, uint64_t(m.index_pool)});
    } else
      throw std::invalid_argument("Vulkan unsupported mixer");
    const uint64_t ff = m.layers[l].intermediate;
    if (m.layers[l].ffn == core::FfnKind::Dense) {
      shape(p + "ffn_gate.weight", {H, ff});
      shape(p + "ffn_up.weight", {H, ff});
      shape(p + "ffn_down.weight", {ff, H});
    } else {
      if (ff > 2048)
        throw std::invalid_argument(
            "Vulkan CPU expert scratch geometry exceeded");
      shape(p + "ffn_gate_inp.weight", {H, uint64_t(m.experts)});
      shape(p + "exp_probs_b.bias", {uint64_t(m.experts)});
      shape(p + "ffn_gate_exps.weight", {H, ff, uint64_t(m.experts)});
      shape(p + "ffn_up_exps.weight", {H, ff, uint64_t(m.experts)});
      shape(p + "ffn_down_exps.weight", {ff, H, uint64_t(m.experts)});
      shape(p + "ffn_gate_shexp.weight", {H, uint64_t(m.shared_intermediate)});
      shape(p + "ffn_up_shexp.weight", {H, uint64_t(m.shared_intermediate)});
      shape(p + "ffn_down_shexp.weight", {uint64_t(m.shared_intermediate), H});
    }
  }
  return m;
}
} // namespace strata::vulkan
