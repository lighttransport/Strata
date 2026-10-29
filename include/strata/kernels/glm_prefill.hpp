#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
void glm_bf16(const float *x, uint16_t *y, int64_t n, void *stream);
void glm_gather_expert_bf16(const float *x, uint16_t *y, const int *bounds, const int *source,
                            int first, int groups, int offset, int rows, int width, void *stream,
                            unsigned expert_mask = 0);
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
// Short decode batch with the same output/history bits as consecutive glm_conv calls.
// Saves the history after each of the first snapshot_tokens rows, at snapshot_stride floats per row.
// Input, output, history and snapshot buffers must not overlap.
void glm_conv_batch_history(const float *x, const float *weight, float *history, float *y, int channels,
                            int kernel, int tokens, float *snapshots, long long snapshot_stride,
                            int snapshot_tokens, void *stream);
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
// Opt-in partial prefill: atomically add weighted expert products into one
// vector per token. Floating-point order differs from the route-row reduction.
void glm_scatter_expert_weighted(const float *x, float *out, const float *weights,
                                const int *bounds, const int *dest, int first, int groups,
                                int offset, int rows, int width, int top_k, void *stream,
                                unsigned expert_mask = 0);
// Same products without atomics: one launch per expert, so each token is updated once per launch and the
// launches add in stream order (deterministic). Requires width % 4 == 0 and no concurrent writer of `out`.
void glm_scatter_expert_weighted_ordered(const float *x, float *out, const float *weights,
                                         const int *bounds, const int *dest, int first, int groups,
                                         int offset, int rows, int width, int top_k, void *stream,
                                         unsigned expert_mask = 0);
void glm_sum_selected_rows(const float *rows, const float *weights, float *sum, int width, int top_k, int tokens, void *stream);
// Plain (non-atomic) add of FP16 rows, one row per token: the caller orders it after every other writer of `out`.
void glm_accumulate_token_rows_half(const uint16_t *source, float *out, const int *tokens, int rows, int width, void *stream);
void glm_accumulate_token_rows(const float *source, float *out, const int *tokens,
                               int rows, int width, void *stream);
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
// prepared_qi requires key/decay/beta transformed by glm_kda_prepare. With
// row_parts=1 the recurrence retains the unprepared accumulation order.
// Prepared inputs cannot be combined with rollback snapshots.
// Sequential recurrence for 1..2048 tokens; dim must be 128.
void glm_kda_chunk(float *state, const float *q, const float *key, const float *value, const float *decay,
                   const float *beta, float *out, int heads, int dim, int tokens, void *stream,
                   int columns = 128, int row_parts = 1, const float *prepared_qi = nullptr,
                   float *snapshots = nullptr, long long snapshot_stride = 0, int snapshot_tokens = 0);
void glm_kda_prepare(const float *q, float *key, float *decay, float *beta, float *qi,
                     int heads, int tokens, void *stream);
void glm_kda_output_batch(const float *x, const float *gate, const float *weight, float *out, int heads,
                          int dim, float eps, int tokens, void *stream);
// Position data is updated outside graph capture; consumers retain stable addresses across replay.
void glm_decode_position(int *position, int value, void *stream);
void glm_cache_rows(const float *rows, float *cache, int width, int tokens,
                    const int *position, void *stream);
// FP16 latent cache (STRATA_GLM_KV_FORMAT=f16): the rows rounded as glm_f16 rounds them.
void glm_cache_rows_f16(const float *rows, uint16_t *cache, int width, int tokens,
                        const int *position, void *stream);
void glm_index_prepare(const float *keys, const float *gates, const float *ape, float *pending_keys,
                       float *pending_gates, float *pooled, int pos, int tokens, int pool, int dim,
                       void *stream, const int *device_pos = nullptr, int position_offset = 0);
// The same with FP16 pooled keys (STRATA_GLM_INDEX_KEYS=f16), each rounded as glm_f16 rounds it.
void glm_index_prepare_f16(const float *keys, const float *gates, const float *ape, float *pending_keys,
                           float *pending_gates, uint16_t *pooled, int pos, int tokens, int pool, int dim,
                           void *stream, const int *device_pos = nullptr, int position_offset = 0);
void glm_index_reduce(const float *dots, const float *weights, float *scores, int heads, int pools,
                      int queries, int query_pos, int pool, int dim, void *stream,
                      const int *device_pos = nullptr, int position_offset = 0);
// One block of pooled keys [first, first + count): dots has leading dimension ld (>= count); scores keeps its full
// width pools_total. Each score is the same sum as glm_index_reduce (STRATA_GLM_INDEX_BLOCK_POOLS).
void glm_index_reduce_block(const float *dots, const float *weights, float *scores, int heads, int ld, int first,
                            int count, int pools_total, int queries, int query_pos, int pool, int dim, void *stream,
                            const int *device_pos = nullptr, int position_offset = 0);
void glm_index_select_batch(const float *scores, int *ids, int *counts, int pools, int query_pos, int queries,
                            int pool, int top_k, int stride, void *stream,
                            const int *device_pos = nullptr, int position_offset = 0);
// One-query MLA products over `stride` gathered keys (keys[s][latent], fp32): scores[h][s] = keys[s] . q[h] and
// out[h] = sum_s p[h][s] * keys[s] (p and scores row-major with leading dimension stride). values needs
// glm_mla_one_scratch_bytes of scratch.
std::size_t glm_mla_one_scratch_bytes(int stride, int latent, int heads);
void glm_mla_scores_one(const float *keys, const float *q, float *scores, int stride, int latent, int heads, void *stream);
void glm_mla_values_one(const float *keys, const float *p, float *out, float *scratch, int stride, int latent,
                        int heads, void *stream);
void glm_mla_gather(const float *cache, const int *ids, const int *counts, float *gathered, int queries,
                    int stride, int latent, void *stream);
void glm_mla_gather_f16(const float *cache, const int *ids, const int *counts, uint16_t *gathered, int queries,
                        int stride, int latent, void *stream);
// The same gathers from an FP16 cache. From rows written by glm_f16 / glm_cache_rows_f16, glm_mla_gather_h2h
// returns glm_mla_gather_f16's bits for the FP32 rows; glm_mla_gather_h2f widens the rounded rows.
void glm_mla_gather_h2h(const uint16_t *cache, const int *ids, const int *counts, uint16_t *gathered, int queries,
                        int stride, int latent, void *stream);
void glm_mla_gather_h2f(const uint16_t *cache, const int *ids, const int *counts, float *gathered, int queries,
                        int stride, int latent, void *stream);
// 8-bit latent cache (STRATA_GLM_KV_FORMAT=int8|fp8): a row is `latent` codes, then one FP32 scale per 64 values.
// int8 codes are rint(x / scale), scale = group absmax / 127; fp8 codes are E4M3 (round to nearest even,
// saturating at 448), scale = group absmax / 448. latent % 64 == 0.
constexpr size_t glm_q8_row_bytes(int latent) { return size_t(latent) + size_t(latent) / 16; }
void glm_cache_rows_q8(const float *rows, uint8_t *cache, int latent, int tokens, int position,
                       const int *device_position, bool fp8, void *stream);
// Gathers of the dequantized rows (code * scale), as FP16 and as FP32.
void glm_mla_gather_q8_f16(const uint8_t *cache, bool fp8, const int *ids, const int *counts, uint16_t *gathered,
                           int queries, int stride, int latent, void *stream);
void glm_mla_gather_q8_f32(const uint8_t *cache, bool fp8, const int *ids, const int *counts, float *gathered,
                           int queries, int stride, int latent, void *stream);
void glm_mla_softmax(float *scores, const int *counts, int heads, int stride, int queries, float scale,
                     void *stream);
// Softmax over FP16 scores into FP16 probabilities (STRATA_GLM_MLA_F16_OUT).
void glm_mla_softmax_h2h(const uint16_t *scores, const int *counts, uint16_t *out, int heads, int stride, int queries,
                         float scale, void *stream);
// glm_mla_softmax followed by the FP16 cast of the probabilities, in one pass; `scores` is left unnormalized.
void glm_mla_softmax_f16(float *scores, const int *counts, uint16_t *out, int heads, int stride, int queries,
                         float scale, void *stream);
// Lossy prefill kernels (KDA column tiles) apply only while allowed; set per batch by the decoder.
void glm_set_prefill_lossy(bool allowed);
// Q2_K (type 10) / Q3_K (11) rows of `experts` matrices [rows, cols] -> Q8_0 (34 bytes per 32 values).
void glm_transcode_q8_0(const void *src, int type, void *dst, long long rows, long long cols, int experts,
                        long long src_stride, long long dst_stride, void *stream);
} // namespace strata::kernels
