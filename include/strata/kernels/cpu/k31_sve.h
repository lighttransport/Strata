/* K3.1 Pro (kimi_linear: KDA + gated DSA, latent MoE, block attention residuals) CPU kernels for the A64FX runner.
 * Every kernel has a scalar form (also the oracle on hosts without SVE). The KDA, residual, router and SiTU
 * kernels are ports of the K3 A64FX experiments (k3_kernels.h); the absorbed latent attention and the DSA indexer
 * are new for K3.1. C11; SVE paths need -msve-vector-bits=512. */
#ifndef STRATA_K31_SVE_H
#define STRATA_K31_SVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    K31_HIDDEN = 7168,
    K31_LAYERS = 93,
    K31_HEADS = 96,          /* KDA and MLA heads */
    K31_HEAD_DIM = 128,      /* KDA key/value dim, MLA nope and v dim */
    K31_CONV = 4,            /* KDA short convolution kernel */
    K31_Q_LORA = 1536,
    K31_KV_LORA = 512,
    K31_ROPE = 64,           /* "rope" part of the MLA key; K3.1 uses no rotation (mla_use_nope) */
    K31_LAT = K31_KV_LORA + K31_ROPE,   /* 576 values per token per MLA layer in the latent cache */
    K31_EXPERTS = 896,
    K31_TOPK = 16,
    K31_SHARED = 2,
    K31_LATENT = 3584,       /* routed_expert_hidden_size */
    K31_EXPERT_INTER = 3072,
    K31_IDX_HEADS = 32,
    K31_IDX_DIM = 128,
    K31_IDX_TOPK = 2048,
    K31_RES_BLOCK = 12       /* attention-residual block size (a new block every 12 layers) */
};

/* 0-based layers 3, 7, ..., 91 and 92 are MLA/DSA (24; config lists them 1-based: 4, 8, ..., 92, 93); the rest KDA */
static inline int k31_is_dsa_layer(int layer) { return layer == 92 || (layer >= 0 && layer < 92 && layer % 4 == 3); }

float k31_sigmoid(float x);
uint16_t k31_f32_to_f16(float f);
float k31_f16_to_f32(uint16_t h);

/* out = x * rsqrt(mean(x^2) + eps) * w (sum in double) */
void k31_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
/* LayerNorm with weight and bias (bias may be NULL) */
void k31_layernorm(float *out, const float *x, const float *w, const float *b, int n, float eps);
/* KDA output norm: rmsnorm(x) * w * sigmoid(gate) */
void k31_gated_rmsnorm(float *out, const float *x, const float *gate, const float *w, int n, float eps);
void k31_l2_normalize(float *x, int n, float eps);
void k31_silu_inplace(float *x, int n);

/* SiTU: 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25). k31_situ_fast: FEXPA form (SVE; scalar fallback = exact). */
void k31_situ(float *out, const float *gate, const float *up, int n);
void k31_situ_fast(float *out, const float *gate, const float *up, int n);

/* causal short conv, one step. state [channels][kernel-1] oldest..newest, weight [channels][kernel], no bias */
void k31_conv_step(float *out, const float *x, float *state, const float *weight, int channels, int kernel);
/* log decay = lower_bound * sigmoid(-A * (g + dt_bias)), A per head (stored as A, not A_log), lower_bound -5 */
void k31_kda_log_decay(float *out, const float *g_raw, const float *a, const float *dt_bias, int heads, int key_dim,
                       float lower_bound);
/* one KDA state row task = (head h, value j): row *= decay; delta = beta (v - k.row); row += k delta;
 * out = q.row / sqrt(key_dim). The k.row and q.row sums are in double, as K3 (its error compounds over 69 layers).
 * state [head][value][key]; decay = exp(log_decay) [head][key]. */
void k31_kda_row(float *out, const float *q, const float *k, const float *v, const float *decay, const float *beta,
                 float *state, int key_dim, int value_dim, int task);
/* all heads * value_dim tasks (OpenMP if built with it) */
void k31_kda_step(float *out, const float *q, const float *k, const float *v, const float *decay, const float *beta,
                  float *state, int heads, int key_dim, int value_dim);

/* Block attention residual. Candidates c = blocks[0..count-1] then prefix. score_c = sum_j rmsnorm(c)_j * nw_j * pw_j,
 * softmax over c, out = sum_c p_c c. pw may be NULL (fused GGUF form). */
void k31_attn_res(float *out, const float *prefix, const float *blocks, int block_count, const float *norm_w,
                  const float *proj_w, int hidden, float eps);

/* noaux_tc router: select top-k by sigmoid(logit) + bias (bias may be NULL), weights = sigmoid(logit) of the selected,
 * renormalized to sum 1, times scale. Ties go to the lower expert index. */
void k31_router_topk(const float *logits, const float *bias, int experts, int topk, float scale, int *idx, float *w);

/* DSA indexer scores for one query: score_t = sum_h relu(q_h . k_t) * w_h * w_scale, keys F16 [n][dim].
 * The head sums run in head order. */
void k31_index_scores(const float *q, const float *w, int heads, int dim, const uint16_t *keys, int n, float w_scale,
                      float *scores);
/* indices of the k largest scores (ties: lower index first), ascending index order on return; returns the count
 * (min(k, n)) */
int k31_topk_indices(const float *scores, int n, int k, int *out);

/* Absorbed latent attention for one query over selected tokens of the F16 latent cache [n][576].
 * qa: per head [heads][576] = (W_UK^T q_nope [512], q_rope [64]); score = qa . c * scale; softmax over the selected
 * tokens; out [heads][512] = sum p c[0:512]. sel == NULL: tokens 0..count-1. */
void k31_latent_attention(const float *qa, int heads, const uint16_t *cache, const int *sel, int count, float scale,
                          float *out);

/* y[r] = sum_c W[r][c] x[c] for BF16 W [rows][cols] (sum in float, 4 accumulators) */
void k31_bf16_matvec(const uint16_t *w, int rows, int cols, const float *x, float *y);
/* y[0..n) += alpha * w[0..n) for a BF16 row w */
void k31_bf16_axpy(float *y, const uint16_t *w, float alpha, int n);
/* MXFP4 rows [r0, r1) (GGUF block layout: E8M0 byte + 16 bytes, low nibble element j, high nibble j + 16; cols / 32
 * blocks per row) times a Q8_0 activation laid out contiguously: aq [cols] int8, ad [cols / 32] block scales.
 * y[r] for r in [r0, r1). SVE (512-bit): two blocks per vector, four accumulator chains. */
void k31_mxfp4_rows(const uint8_t *w, int cols, int r0, int r1, const int8_t *aq, const float *ad, float *y);
/* y[r] = sum_c WT[c][r] x[c] for r in [r0, r1): a BF16 matrix stored transposed, WT [k][rows]. Each output summed in
 * c order (column-major matvec: no horizontal sums; for short-row slices such as o_proj columns) */
void k31_bf16_tmatvec(const uint16_t *wt, int rows, int k, const float *x, float *y, int r0, int r1);
/* Planar MXFP4 (rows of stride bytes, 64-aligned: nb*16 nibble bytes, block b at 16 b (low nibble element j, high
 * j + 16), then nb E8M0 bytes; nb % 4 == 0). aqp: the activation per 4-block group g, 128 int8: [b][j] = a[32 (4g+b) + j]
 * then [b][j] = a[32 (4g+b) + 16 + j]; ad4: each block scale repeated 4 times (k31_mxfp4p_act builds both). */
void k31_mxfp4p_act(const int8_t *aq, const float *ad, int nb, int8_t *aqp, float *ad4);
void k31_mxfp4p_rows(const uint8_t *w, size_t stride, int cols, int r0, int r1, const int8_t *aqp, const float *ad4, float *y);
uint16_t k31_f32_to_bf16(float f);
float k31_bf16_to_f32(uint16_t b);

#ifdef __cplusplus
}
#endif
#endif
