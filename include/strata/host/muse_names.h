// include/strata/host/muse_names.h - Muse Spark Open (`muse-spark`): the tensors of a block, their shapes, and how
// two nodes split a routed expert.
//
// Every program that touches a Muse Spark file agrees on these: the loader that binds a block's tensors, the inspect
// tool that checks a download before it is trusted, and - on two nodes - the primary and the expert worker, which
// must cut every expert at the same row so that the partial results they exchange add up.  So they are spelled once,
// here, in C11 like the GLM rules (glm_names.h), and checked by tests/host/muse_names_test.c without a model.
//
// The block, as llama.cpp's src/models/muse-spark.cpp computes it (ggml shapes: dimension 0 varies fastest):
//   attn_norm -> attn_q/k/v, attn_gate (sigmoid output gate) -> q/k RMS norm -> RoPE on the sliding-window layers
//   only (NoPE on the full ones) -> attention -> weightless per-head RMS norm -> * sigmoid(gate) -> attn_output ->
//   weightless RMS norm -> alpha * x + beta * branch (attn_resid_alpha/beta);
//   ffn_norm -> ffn_latent_down (8192 -> 4096) -> ffn_pre_exp_norm -> 8 of 256 experts (router ffn_gate_inp on the
//   8192-wide normed stream, sigmoid + exp_probs_b for the choice, weights renormalized) -> each expert's output
//   * ffn_post_exp_norm then RMS-normalized, THEN weighted and summed -> ffn_latent_up -> * post_ffw_norm then RMS
//   norm -> alpha * x + beta * branch (ffn_resid_alpha/beta).
#ifndef STRATA_HOST_MUSE_NAMES_H_
#define STRATA_HOST_MUSE_NAMES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STRATA_MUSE_ARCH "muse-spark"

typedef enum strata_muse_tensor {
    STRATA_MUSE_ATTN_NORM = 0,
    STRATA_MUSE_ATTN_Q,
    STRATA_MUSE_ATTN_K,
    STRATA_MUSE_ATTN_V,
    STRATA_MUSE_ATTN_GATE,
    STRATA_MUSE_ATTN_OUTPUT,
    STRATA_MUSE_ATTN_Q_NORM,
    STRATA_MUSE_ATTN_K_NORM,
    STRATA_MUSE_ATTN_RESID_ALPHA,
    STRATA_MUSE_ATTN_RESID_BETA,
    STRATA_MUSE_FFN_NORM,
    STRATA_MUSE_POST_FFW_NORM,
    STRATA_MUSE_FFN_RESID_ALPHA,
    STRATA_MUSE_FFN_RESID_BETA,
    STRATA_MUSE_FFN_GATE_INP,
    STRATA_MUSE_EXP_PROBS_B,
    STRATA_MUSE_FFN_LATENT_DOWN,
    STRATA_MUSE_FFN_LATENT_UP,
    STRATA_MUSE_FFN_PRE_EXP_NORM,
    STRATA_MUSE_FFN_POST_EXP_NORM,
    STRATA_MUSE_FFN_GATE_EXPS,
    STRATA_MUSE_FFN_UP_EXPS,
    STRATA_MUSE_FFN_DOWN_EXPS,
    STRATA_MUSE_TENSOR_COUNT
} strata_muse_tensor;

/* The model's dimensions, from the GGUF keys (muse-spark.embedding_length, .moe_latent_size, ...). */
typedef struct strata_muse_dims {
    int64_t hidden;      /* embedding_length, 8192 */
    int64_t latent;      /* moe_latent_size, 4096 */
    int64_t expert_ff;   /* expert_feed_forward_length, 4096 */
    int64_t experts;     /* expert_count, 256 */
    int64_t q_heads;     /* attention.head_count, 128 */
    int64_t kv_heads;    /* attention.head_count_kv, 16 */
    int64_t head_dim;    /* attention.key_length (= value_length), 64 */
} strata_muse_dims;

/* `attn_q.weight` etc.: the suffix after `blk.<layer>.`, or NULL for a kind out of range. */
const char *strata_muse_suffix(strata_muse_tensor kind);

/* `blk.<layer>.<suffix>` -> out (glm_names.h's strata_blk_name).  Returns the length, or -1 if it does not fit. */
int strata_muse_tensor_name(char *out, size_t cap, int layer, strata_muse_tensor kind);

/* The ggml shape of `kind` (shape[0] varies fastest) into shape[0..2].  Returns the number of dimensions (1-3), or 0
   for a kind out of range. */
int strata_muse_shape(const strata_muse_dims *d, strata_muse_tensor kind, int64_t shape[3]);

/* 1 for the routed-expert tensors (ffn_{gate,up,down}_exps), 0 otherwise. */
int strata_muse_is_expert(strata_muse_tensor kind);

/* 1 when the engine reads `kind` in ggml type `type`: F32 for the norms, alphas, betas, router and bias; the
   projections and experts in a quantized or float type the CPU or GPU kernels decode.  0 otherwise. */
int strata_muse_type_ok(strata_muse_tensor kind, int type);

/* Layer `layer` uses RoPE (a sliding-window layer) when swa_pattern[layer] is nonzero; the full-attention layers are
   NoPE.  Returns -1 when layer is outside [0, layers). */
int strata_muse_layer_rope(const uint8_t *swa_pattern, int layers, int layer);

/* The row split of a routed expert between two nodes.  Node 0 (the primary) keeps expert rows [0, first_b) of
   gate and up and the same columns of down (down's K dimension is expert_ff, so a node's partial down output covers
   its own ff rows); node 1 holds [first_b, expert_ff).  The cut is a whole number of `block` values (the IQ/K-quant
   block of down's rows, 256) so each node's down rows are whole blocks, rounded to the nearest block from
   share_b * expert_ff.  Returns first_b, or -1 when share_b is outside [0, 1] or expert_ff is not whole blocks. */
int64_t strata_muse_row_split(int64_t expert_ff, int64_t block, double share_b);

#ifdef __cplusplus
}
#endif

#endif  // STRATA_HOST_MUSE_NAMES_H_
