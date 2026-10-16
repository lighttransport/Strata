// include/strata/kernels/muse.hpp - the Muse Spark Open block's GPU kernels that the shared libraries do not have.
//
// The projections are native_mmvq (Q6_K / Q8_0 / Q4_K against Q8_1 activations, the pinned llama.cpp MMVQ) and the
// router is glm_router (sigmoid scores, the bias chooses, renormalized weights: the same rule). What is Muse's own:
// the norms in their four shapes, RoPE on the sliding-window layers, the f16 KV cache (a 2048-slot ring on those
// layers), GQA decode attention, the attention output's per-head norm and sigmoid gate, the alpha/beta residual, the
// router's F32 GEMV, and the head's logit scale and softcap.  The arithmetic is strata/muse/reference.hpp's; each
// kernel's parity is checked through the whole block (tools/muse_gpu_trunk_check.cpp).
//
// Positions are read from a device int so a captured graph replays for every token.  All pointers are device
// pointers; every call enqueues on the given non-null stream, with no allocation and no synchronization.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// y[r] = rms_norm(x[r] * pre) * post over `cols`, row by row (`pre` / `post` may be null; y may be x).
///   attn_norm / ffn_norm / output_norm: pre null, post the weight;  q/k head norms: rows = heads, post the weight;
///   the weightless norms: both null;  post_ffw_norm / ffn_post_exp_norm: pre the weight, post null.
void muse_rms_norm(const float* x, const float* pre, const float* post, float* y, int rows, int cols, float eps,
                   void* stream);

/// RoPE in ggml's NORM layout (pairs 2i, 2i+1), theta_i = pos * base^(-2i / head_dim), on `rows` heads in place.
void muse_rope(float* x, int rows, int head_dim, const int32_t* pos, float base, void* stream);

/// K and V of one token (width = kv_heads * head_dim floats each) into the f16 caches at slot pos (full attention)
/// or pos % ring (a sliding-window layer's ring of `ring` slots).
void muse_kv_append(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, int width,
                    const int32_t* pos, int ring, void* stream);

/// One query token's GQA attention over the cached positions: full layers 0 .. pos, ring layers the last
/// min(pos + 1, ring) slots (softmax does not depend on the order).  q: [q_heads][head_dim], out the same; the caches
/// [slot][kv_heads][head_dim] f16.  head_dim must be 64.
void muse_attention(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int q_heads,
                    int kv_heads, int head_dim, const int32_t* pos, int ring, float scale, void* stream);

/// a[h] = rms_norm(a[h]) * sigmoid(gate[h]) per head (weightless, eps), in place.
void muse_head_norm_gate(float* a, const float* gate, int heads, int head_dim, float eps, void* stream);

/// x = alpha * x + beta * branch, elementwise, in place.
void muse_residual(float* x, const float* branch, const float* alpha, const float* beta, int n, void* stream);

/// y = W x for an F32 [n_out][n_in] weight (the router, 256 x 8192).
void muse_gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, void* stream);

/// logits = softcap * tanh(scale * logits / softcap) (softcap 0: only the scale), in place.
void muse_logit_cap(float* logits, int n, float scale, float softcap, void* stream);

}  // namespace strata::kernels
