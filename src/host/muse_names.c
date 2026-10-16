/* src/host/muse_names.c - Muse Spark Open's block tensors, shapes and expert row split (muse_names.h). */
#include "strata/host/muse_names.h"
#include "strata/host/glm_names.h"

/* ggml type ids this file names (ggml.h's enum ggml_type) */
enum {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22,
    T_IQ4_XS = 23, T_IQ1_M = 29, T_BF16 = 30
};

static const char *const kSuffix[STRATA_MUSE_TENSOR_COUNT] = {
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_gate.weight", "attn_output.weight",
    "attn_q_norm.weight", "attn_k_norm.weight", "attn_resid_alpha.weight", "attn_resid_beta.weight",
    "ffn_norm.weight", "post_ffw_norm.weight", "ffn_resid_alpha.weight", "ffn_resid_beta.weight",
    "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_latent_down.weight", "ffn_latent_up.weight",
    "ffn_pre_exp_norm.weight", "ffn_post_exp_norm.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
    "ffn_down_exps.weight",
};

const char *strata_muse_suffix(strata_muse_tensor kind) {
    return (int) kind >= 0 && kind < STRATA_MUSE_TENSOR_COUNT ? kSuffix[kind] : NULL;
}

int strata_muse_tensor_name(char *out, size_t cap, int layer, strata_muse_tensor kind) {
    const char *s = strata_muse_suffix(kind);
    return s ? strata_blk_name(out, cap, layer, s) : -1;
}

static int set(int64_t shape[3], int n, int64_t a, int64_t b, int64_t c) {
    shape[0] = a;
    shape[1] = b;
    shape[2] = c;
    return n;
}

int strata_muse_shape(const strata_muse_dims *d, strata_muse_tensor kind, int64_t shape[3]) {
    const int64_t q = d->q_heads * d->head_dim, kv = d->kv_heads * d->head_dim;
    switch (kind) {
        case STRATA_MUSE_ATTN_NORM: case STRATA_MUSE_ATTN_RESID_ALPHA: case STRATA_MUSE_ATTN_RESID_BETA:
        case STRATA_MUSE_FFN_NORM: case STRATA_MUSE_POST_FFW_NORM: case STRATA_MUSE_FFN_RESID_ALPHA:
        case STRATA_MUSE_FFN_RESID_BETA:
            return set(shape, 1, d->hidden, 1, 1);
        case STRATA_MUSE_ATTN_Q: case STRATA_MUSE_ATTN_GATE: return set(shape, 2, d->hidden, q, 1);
        case STRATA_MUSE_ATTN_K: case STRATA_MUSE_ATTN_V: return set(shape, 2, d->hidden, kv, 1);
        case STRATA_MUSE_ATTN_OUTPUT: return set(shape, 2, q, d->hidden, 1);
        case STRATA_MUSE_ATTN_Q_NORM: case STRATA_MUSE_ATTN_K_NORM: return set(shape, 1, d->head_dim, 1, 1);
        case STRATA_MUSE_FFN_GATE_INP: return set(shape, 2, d->hidden, d->experts, 1);
        case STRATA_MUSE_EXP_PROBS_B: return set(shape, 1, d->experts, 1, 1);
        case STRATA_MUSE_FFN_LATENT_DOWN: return set(shape, 2, d->hidden, d->latent, 1);
        case STRATA_MUSE_FFN_LATENT_UP: return set(shape, 2, d->latent, d->hidden, 1);
        case STRATA_MUSE_FFN_PRE_EXP_NORM: case STRATA_MUSE_FFN_POST_EXP_NORM: return set(shape, 1, d->latent, 1, 1);
        case STRATA_MUSE_FFN_GATE_EXPS: case STRATA_MUSE_FFN_UP_EXPS:
            return set(shape, 3, d->latent, d->expert_ff, d->experts);
        case STRATA_MUSE_FFN_DOWN_EXPS: return set(shape, 3, d->expert_ff, d->latent, d->experts);
        default: return 0;
    }
}

int strata_muse_is_expert(strata_muse_tensor kind) {
    return kind == STRATA_MUSE_FFN_GATE_EXPS || kind == STRATA_MUSE_FFN_UP_EXPS || kind == STRATA_MUSE_FFN_DOWN_EXPS;
}

int strata_muse_type_ok(strata_muse_tensor kind, int type) {
    if ((int) kind < 0 || kind >= STRATA_MUSE_TENSOR_COUNT) return 0;
    switch (kind) {
        case STRATA_MUSE_ATTN_Q: case STRATA_MUSE_ATTN_K: case STRATA_MUSE_ATTN_V: case STRATA_MUSE_ATTN_GATE:
        case STRATA_MUSE_ATTN_OUTPUT: case STRATA_MUSE_FFN_LATENT_DOWN: case STRATA_MUSE_FFN_LATENT_UP:
            /* the dense projections: the GPU's dense formats */
            return type == T_F32 || type == T_F16 || type == T_BF16 || type == T_Q8_0 || type == T_Q4_K ||
                   type == T_Q5_K || type == T_Q6_K;
        case STRATA_MUSE_FFN_GATE_EXPS: case STRATA_MUSE_FFN_UP_EXPS: case STRATA_MUSE_FFN_DOWN_EXPS:
            /* the routed experts: Unsloth's UD mixes (IQ1_S ... Q4_K) and Q8_0 */
            return type == T_IQ1_S || type == T_IQ1_M || type == T_IQ2_XXS || type == T_IQ2_XS || type == T_IQ2_S ||
                   type == T_IQ3_XXS || type == T_IQ3_S || type == T_IQ4_XS || type == T_IQ4_NL || type == T_Q2_K ||
                   type == T_Q3_K || type == T_Q4_K || type == T_Q5_K || type == T_Q6_K || type == T_Q8_0;
        default:
            return type == T_F32;   /* norms, residual alpha/beta, the router and its bias */
    }
}

int strata_muse_layer_rope(const uint8_t *swa_pattern, int layers, int layer) {
    if (!swa_pattern || layer < 0 || layer >= layers) return -1;
    return swa_pattern[layer] ? 1 : 0;
}

int64_t strata_muse_row_split(int64_t expert_ff, int64_t block, double share_b) {
    if (!(share_b >= 0.0 && share_b <= 1.0) || block <= 0 || expert_ff <= 0 || expert_ff % block) return -1;
    const int64_t blocks = expert_ff / block;
    const int64_t b_blocks = (int64_t) (share_b * (double) blocks + 0.5);   /* share_b >= 0: truncation rounds */
    return (blocks - b_blocks) * block;
}
