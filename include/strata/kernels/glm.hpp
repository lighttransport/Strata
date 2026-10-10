#pragma once

#include <cstdint>

namespace strata::kernels {

// All buffers are device pointers. Calls enqueue on stream without allocating.
void glm_rms_norm(const float *x, const float *weight, float *y, int width, int tokens, float eps,
                  void *stream);
// Same result to rounding with 1024 threads and float4 traffic per row; the sum order differs from glm_rms_norm,
// so a caller uses one or the other for every width it must keep consistent.
void glm_rms_norm_rows(const float *x, const float *weight, float *y, int width, int tokens, float eps,
                       void *stream);
// Aligned rows, width divisible by 32: normalized floats and native Q8_1 activation blocks in one launch.
void glm_rms_norm_rows_q8(const float *x, const float *weight, float *y, void *q8, int width, int tokens,
                         float eps, void *stream);
void glm_layer_norm(const float *x, const float *weight, const float *bias, float *y, int width, float eps,
                    void *stream);
void glm_swiglu(const float *gate, const float *up, float *y, int count, float limit, void *stream);
// `bonus` (optional, one value per expert) is added to the selection score only; weights keep the true scores.
void glm_router(const float *logits, const float *correction, int *ids, float *weights, int experts,
                int top_k, float scale, void *stream, const float *bonus = nullptr);
// One token: glm_router followed by glm_mailbox_publish of the result and `x` into `slot`, as one launch.
struct GlmMailboxView;
void glm_router_publish(const float *logits, const float *correction, int *ids, float *weights, int experts,
                        int top_k, float scale, const float *bonus, GlmMailboxView view, int slot, const float *x,
                        const unsigned *generation, void *stream);

// Four-stream mHC: projected is the 24-wide FP32 projection of normalized streams.
// coefficients layout: pre[4], post[4], comb[4][4] (input stream, output stream).
void glm_mhc_read(const float *streams, const float *projected, const float *base, const float *scale,
                  float *coefficients, float *x, int width, int iterations, float eps, void *stream);
void glm_mhc_write(const float *streams, const float *coefficients, const float *y, float *out, int width,
                   void *stream);
void glm_hyper_head(const float *streams, float *out, int width, void *stream);
// Several decode tokens in one launch each: token t uses streams + t*4*width, projections and coefficients
// + t*24, x/y + t*width; every token computes exactly what the single-token call computes.
void glm_mhc_read_tokens(const float *streams, const float *projected, const float *base, const float *scale,
                         float *coefficients, float *x, int width, int iterations, float eps, int tokens,
                         void *stream);
void glm_mhc_write_tokens(const float *streams, const float *coefficients, const float *y, float *out, int width,
                          int tokens, void *stream);
// Router for several tokens: logits + t*experts, ids/weights + t*top_k.
void glm_router_tokens(const float *logits, const float *correction, int *ids, float *weights, int experts,
                       int top_k, float scale, int tokens, void *stream, const float *bonus = nullptr);
// FP32 24-row mHC projection; caller supplies 24*32 partial sums.
void glm_hc_project(const float *x, const float *weight, float *out, float *scratch,
                    int width, void *stream);

// Convolution history is [channel][kernel-1], oldest first. KDA state is [head][key][value].
void glm_conv(const float *x, const float *weight, float *history, float *y, int channels, int kernel,
              void *stream);
// GGUF ssm_a stores -exp(A_log), not the original Transformers parameter.
void glm_kda_gate(const float *projected, const float *bias, const float *negative_a, float *log_decay,
                  int heads, int dim, float lower_bound, void *stream);
void glm_kda_step(float *state, const float *query, const float *key, const float *value,
                  const float *log_decay, const float *beta_logits, float *out, int heads, int dim,
                  void *stream);
void glm_kda_output(const float *x, const float *gate, const float *weight, float *out, int heads, int dim,
                    float eps, void *stream);

// Learned per-channel softmax pooling over complete pools. Tail tokens stay unpooled.
void glm_index_pool(const float *keys, const float *gates, const float *ape, float *pooled, int pool, int dim,
                    void *stream);
void glm_index_score(const float *query, const float *weights, const float *pooled, float *scores, int heads,
                     int dim, int pools, void *stream);
void glm_index_select(const float *scores, int *selected, int *count, int tokens, int pool, int top_k,
                      void *stream);

// Absorbed NoPE MLA. query=[head][latent], cache=[token][latent], out=[head][latent].
// selected/count describes the raw token IDs returned by the indexer.
void glm_mla(const float *query, const float *cache, const int *selected, const int *count, float *out,
             int heads, int latent, float scale, void *stream);

} // namespace strata::kernels
