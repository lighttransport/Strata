#pragma once
#include <cstdint>
namespace strata::kernels {
void glm_layer_norm_batch(const float *x, const float *w, const float *b, float *y, int width, int tokens,
                          float eps, void *stream);
void glm_f16(const float *x, uint16_t *y, int64_t count, void *stream);
// Contiguous scratch rows, with route indices referring to the full batch.
void glm_gather_f16(const float *x, uint16_t *y, const int *routes, int begin,
                    int rows, int width, void *stream);
void glm_scatter_rows(const float *x, float *y, const int *routes, int begin,
                      int rows, int width, void *stream);
void glm_resident_routes(const int *ids, const unsigned long long *lookup,
                          unsigned long long *pointers, int *starts, int *dest,
                          int *tokens, int *count, void *stream);
// At most eight tokens, eight routes each; group repeated resident addresses.
void glm_resident_routes_batch(const int *ids, const unsigned long long *lookup,
                               unsigned long long *pointers, int *starts, int *dest,
                               int *tokens, int *count, int token_count, void *stream);
void glm_gather_expert_f16_from_half(const uint16_t *x, uint16_t *y, const int *bounds, const int *source,
                                     int first, int groups, int offset, int rows, int width, void *stream,
                                     unsigned expert_mask = 0);
void glm_gather_expert_f16(const float *x, uint16_t *y, const int *bounds, const int *source,
                           int first, int groups, int offset, int rows, int width, void *stream,
                           unsigned expert_mask = 0);
void glm_scatter_expert_rows(const float *x, float *y, const int *bounds, const int *dest,
                             int first, int groups, int offset, int rows, int width, void *stream,
                             unsigned expert_mask = 0);
// Preserve the CPU's separate multiply/add and route order during device reduction.
void glm_moe_reduce(const float *cpu, const float *primary, const float *remote,
                     const int *owners, const float *weights, float *out, int width, void *stream);
void glm_moe_reduce_batch(const float *cpu, const float *primary, const float *remote,
                          const int *owners, const float *weights, float *out, int width,
                          int tokens, void *stream);
void glm_conv_batch(const float *x, const float *weight, float *history, float *y, int channels, int kernel,
                    int tokens, void *stream);
void glm_route_batch(const float *logits, const float *bias, int *ids, float *weights, int experts, int top_k,
                     float scale, int tokens, void *stream, const float *bonus = nullptr);
void glm_group_routes(const int *ids, int *bounds, int *dest, int *source, int *cursor, int experts,
                      int top_k, int tokens, void *stream);
// Stable expert grouping retains increasing original route IDs within each expert.
// scratch needs experts * (ceil(tokens*top_k/256) + 1) integers.
void glm_group_routes_stable(const int *ids, int *bounds, int *dest, int *source, int *scratch,
                            int experts, int top_k, int tokens, void *stream);
void glm_group_bounds(const int *global, int *local, int start, int experts, int offset, int rows,
                      void *stream);
void glm_route_sum(const float *results, const float *weights, float *out, int width, int top_k, int tokens,
                   void *stream);
// Routes of one device's expert groups only; participant 0 also adds the other device's partial sums.
void glm_route_sum_owned(const float *results, const float *weights, const int *ids, const float *partial,
                         float *out, int width, int top_k, int tokens, int primary_groups, int participant,
                         void *stream);
// Copy route rows without arithmetic, preserving their individual FP32 bits.
// gather: route order -> contiguous grouped order; scatter reverses the copy.
void glm_copy_route_rows(const float *source, float *dest, const int *routes, int begin, int rows,
                         int width, bool gather, void *stream);
void glm_mhc_read_batch(const float *r, const float *p, const float *base, const float *scale, float *c,
                        float *x, int width, int iterations, float eps, int tokens, void *stream);
void glm_mhc_write_batch(const float *r, const float *c, const float *y, float *out, int width, int tokens,
                         void *stream);
void glm_kda_gate_batch(const float *x, const float *bias, const float *a, float *y, int heads, int dim,
                        float lower, int tokens, void *stream);
void glm_kda_chunk(float *state, const float *q, const float *key, const float *value, const float *decay,
                   const float *beta, float *out, int heads, int dim, int tokens, void *stream,
                   int columns = 128, int row_parts = 1, const float *prepared_qi = nullptr,
                   float *snapshots = nullptr, long long snapshot_stride = 0, int snapshot_tokens = 0);
void glm_kda_prepare(const float *q, float *key, float *decay, float *beta, float *qi,
                     int heads, int tokens, void *stream);
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
void glm_mla_gather_f16(const float *cache, const int *ids, const int *counts, uint16_t *gathered, int queries,
                        int stride, int latent, void *stream);
void glm_mla_softmax(float *scores, const int *counts, int heads, int stride, int queries, float scale,
                     void *stream);
} // namespace strata::kernels
