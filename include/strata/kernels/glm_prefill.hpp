#pragma once
#include <cstdint>
namespace strata::kernels {
void glm_layer_norm_batch(const float *x, const float *w, const float *b, float *y, int width, int tokens,
                          float eps, void *stream);
void glm_f16(const float *x, uint16_t *y, int64_t count, void *stream);
void glm_conv_batch(const float *x, const float *weight, float *history, float *y, int channels, int kernel,
                    int tokens, void *stream);
void glm_route_batch(const float *logits, const float *bias, int *ids, float *weights, int experts, int top_k,
                     float scale, int tokens, void *stream);
void glm_group_routes(const int *ids, int *bounds, int *dest, int *source, int *cursor, int experts,
                      int top_k, int tokens, void *stream);
void glm_group_bounds(const int *global, int *local, int start, int experts, int offset, int rows,
                      void *stream);
void glm_route_sum(const float *results, const float *weights, float *out, int width, int top_k, int tokens,
                   void *stream);
void glm_mhc_read_batch(const float *r, const float *p, const float *base, const float *scale, float *c,
                        float *x, int width, int iterations, float eps, int tokens, void *stream);
void glm_mhc_write_batch(const float *r, const float *c, const float *y, float *out, int width, int tokens,
                         void *stream);
void glm_kda_gate_batch(const float *x, const float *bias, const float *a, float *y, int heads, int dim,
                        float lower, int tokens, void *stream);
void glm_kda_chunk(float *state, const float *q, const float *key, const float *value, const float *decay,
                   const float *beta, float *out, int heads, int dim, int tokens, void *stream);
void glm_kda_output_batch(const float *x, const float *gate, const float *weight, float *out, int heads,
                          int dim, float eps, int tokens, void *stream);
void glm_index_prepare(const float *keys, const float *gates, const float *ape, float *pending_keys,
                       float *pending_gates, float *pooled, int pos, int tokens, int pool, int dim,
                       void *stream);
void glm_index_reduce(const float *dots, const float *weights, float *scores, int heads, int pools,
                      int queries, int query_pos, int pool, int dim, void *stream);
void glm_index_select_batch(const float *scores, int *ids, int *counts, int pools, int query_pos, int queries,
                            int pool, int top_k, int stride, void *stream);
void glm_mla_gather(const float *cache, const int *ids, const int *counts, float *gathered, int queries,
                    int stride, int latent, void *stream);
void glm_mla_softmax(float *scores, const int *counts, int heads, int stride, int queries, float scale,
                     void *stream);
} // namespace strata::kernels
