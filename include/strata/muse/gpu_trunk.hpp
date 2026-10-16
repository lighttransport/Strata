// include/strata/muse/gpu_trunk.hpp - the Muse Spark block's trunk on one GPU (CUDA or HIP), one decode token at a
// time: everything of a block except the routed experts, which the CPU expert lane computes between the two halves.
//
//   attention_and_route(l): attn_norm, q/k/v/gate, q/k norms, RoPE (sliding-window layers), the f16 KV cache,
//       attention, head norm * sigmoid(gate), attn_output, post-attention norm, residual; ffn_norm, latent_down,
//       pre-expert norm (z), the router (ids, weights).  z, ids and weights are then on the host.
//   finish_moe(l, y): y (the weighted sum of the experts' normalized outputs, latent-wide) -> latent_up,
//       post_ffw_norm, residual.
//   head(): output_norm, the head, logit scale and softcap.
//
// The arithmetic is strata/muse/reference.hpp's with the GPU's precision: projections take Q8_1 activations
// (native_mmvq), K and V are stored as f16.  No CUDA or HIP type appears here, so host tools include it as C++.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::muse {

/// A tensor's bytes in host memory (an mmapped GGUF) and its ggml type; ne0 is the contiguous dimension.
struct HostTensor {
    const void* data = nullptr;
    int type = 0;
    int64_t ne0 = 0, ne1 = 1;
    size_t bytes = 0;
};

struct TrunkLayerHost {
    bool swa = true;
    HostTensor attn_norm, attn_q, attn_k, attn_v, attn_gate, attn_output, attn_q_norm, attn_k_norm, attn_alpha,
        attn_beta, ffn_norm, post_ffw_norm, ffn_alpha, ffn_beta, router, router_bias, latent_down, latent_up,
        pre_exp_norm;
};

struct TrunkConfig {
    int hidden = 0, latent = 0, experts = 0, top_k = 0, q_heads = 0, kv_heads = 0, head_dim = 0, window = 0;
    int context = 0;   // positions the full-attention layers' caches hold
    float rms_eps = 1e-5f, post_eps = 1e-8f, rope_base = 500000.f, expert_weights_scale = 1.f;
    float logit_scale = 1.f, softcap = 0.f;
};

/// Snapshots of a block's intermediate streams for parity checks (copied as the block runs).
enum class TrunkTrace { AttnPostNorm, FfnInp, FfnPostNorm, Hidden };

class GpuTrunk {
public:
    /// Uploads the layers' trunk tensors (and output_norm + output when `head` is non-null) to the current device.
    GpuTrunk(const TrunkConfig& c, const std::vector<TrunkLayerHost>& layers, const HostTensor* output_norm,
             const HostTensor* output);
    ~GpuTrunk();
    GpuTrunk(const GpuTrunk&) = delete;
    GpuTrunk& operator=(const GpuTrunk&) = delete;

    /// The stream entering block 0 at position `pos` (hidden floats, host).
    void begin(const float* x, int pos);
    void attention_and_route(int layer);
    const std::vector<float>& z() const { return z_; }
    const std::vector<int32_t>& ids() const { return ids_; }
    const std::vector<float>& weights() const { return w_; }
    void finish_moe(int layer, const float* y);
    /// The logits of the current stream (vocab floats, host).
    void head(std::vector<float>& logits);
    void trace(TrunkTrace which, std::vector<float>& out) const;
    size_t device_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<float> z_, w_;
    std::vector<int32_t> ids_;
};

}  // namespace strata::muse
