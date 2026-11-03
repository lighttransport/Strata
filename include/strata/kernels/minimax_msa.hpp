#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Decode/prefill index branch block scores for append-only chronological key storage.
/// q is [queries][index_heads][dim], k is [positions][dim], and output is
/// [queries][index_heads][n_blocks] with score_stride elements per row. Key slots
/// correspond to absolute positions; key_positions is -1 for an unused slot.
void minimax_msa_block_scores(const float* q, const float* k, const int32_t* query_positions,
                              const int32_t* key_positions, int queries, int index_heads, int dim,
                              int keys, int block_size, int n_blocks, float* scores,
                              size_t score_stride, void* stream);

/// Select top_k blocks per query and index head. Local blocks are always retained;
/// ties choose the lower block ID. Output rows have id_stride int32 elements.
void minimax_msa_select_blocks(const float* scores, const int32_t* query_positions,
                               int queries, int index_heads, int n_blocks, int block_size,
                               int top_k, int local_blocks, size_t score_stride,
                               int32_t* ids, size_t id_stride, void* stream);

/// Reference sparse GQA attention over selected blocks. q is [queries][query_heads][dim],
/// k/v are [positions][kv_heads][dim], and selected IDs are [queries][kv_heads][top_k].
/// This first kernel supports float caches and at most 4096 selected token slots per row.
void minimax_msa_attention_f32(const float* q, const float* k, const float* v,
                               const int32_t* query_positions, const int32_t* key_positions,
                               const int32_t* selected_blocks, int queries, int query_heads,
                               int kv_heads, int dim, int keys, int block_size, int top_k,
                               size_t ids_stride, float* out, void* stream);

/// The same with FP16 caches (uint16_t = IEEE half bits): k (index keys) / k, v (K and V) hold 128-wide half rows. dim must be 128.
void minimax_msa_block_scores_h(const float* q, const uint16_t* k, const int32_t* query_positions,
                                const int32_t* key_positions, int queries, int index_heads, int dim, int keys,
                                int block_size, int n_blocks, float* scores, size_t score_stride, void* stream);
/// Block scores against FP8 (E4M3) indexer keys k8 [keys][128] with one float scale per key (a few queries: one-token and window steps).
void minimax_msa_block_scores_q8(const float* q, const uint8_t* k8, const float* kscale, const int32_t* query_positions,
                                 const int32_t* key_positions, int queries, int index_heads, int dim, int keys,
                                 int block_size, int n_blocks, float* scores, size_t score_stride, void* stream);
void minimax_msa_attention_h(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* query_positions,
                             const int32_t* key_positions, const int32_t* selected_blocks, int queries,
                             int query_heads, int kv_heads, int dim, int keys, int block_size, int top_k,
                             size_t ids_stride, float* out, void* stream);

} // namespace strata::kernels
