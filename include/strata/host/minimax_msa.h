#ifndef STRATA_HOST_MINIMAX_MSA_H
#define STRATA_HOST_MINIMAX_MSA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum strata_minimax_msa_status {
    STRATA_MINIMAX_MSA_OK = 0,
    STRATA_MINIMAX_MSA_INVALID_ARGUMENT = 1,
    STRATA_MINIMAX_MSA_NONFINITE_VALUE = 2,
};

/// Reference block scores for MiniMax Sparse Attention. q is [query][index head][dimension],
/// k is [key][dimension], and scores is [query][index head][block]. key_positions may contain
/// -1 for an unused cache slot; other positions must be nonnegative and fit in n_blocks.
/// Each block score is max(dot(q, k)) over its visible keys. Future keys are masked per query.
int strata_minimax_msa_block_scores_f32(const float *q, const float *k,
                                        const int32_t *query_positions, const int32_t *key_positions,
                                        int n_queries, int n_heads, int dim, int n_keys,
                                        int block_size, int n_blocks, float *scores, size_t score_stride);

/// Select top_k block IDs independently for each [query][index head] score row. The current
/// position's block and the preceding local_blocks - 1 blocks are forced into the result, as in
/// MiniMax's local-block bias. Output IDs are ascending; score ties prefer smaller block IDs.
/// ids is [query][index head][top_k], id_stride is measured in int32 elements per row.
int strata_minimax_msa_select_blocks(const float *scores, const int32_t *query_positions,
                                     int n_queries, int n_heads, int n_blocks, int block_size,
                                     int top_k, int local_blocks, size_t score_stride,
                                     int32_t *ids, size_t id_stride);

/// Reference grouped-query sparse attention for selected MSA blocks. q is
/// [query][query_head][dim], k/v are [key][kv_head][dim], and selected blocks
/// are [query][kv_head][top_k]. Attention is causal and scaled by 1/sqrt(dim).
/// Output is [query][query_head][dim].
int strata_minimax_msa_attention_f32(const float *q, const float *k, const float *v,
                                     const int32_t *query_positions, const int32_t *key_positions,
                                     const int32_t *selected_blocks, int n_queries,
                                     int query_heads, int kv_heads, int dim, int n_keys,
                                     int block_size, int n_blocks, int top_k,
                                     size_t ids_stride, float *out);

#ifdef __cplusplus
}
#endif

#endif
