#include "strata/host/minimax_msa.h"

#include <math.h>
#include <stddef.h>

static int valid_score(float x) { return isfinite(x) || (isinf(x) && x < 0.0f); }

int strata_minimax_msa_block_scores_f32(const float *q, const float *k,
                                        const int32_t *query_positions, const int32_t *key_positions,
                                        int n_queries, int n_heads, int dim, int n_keys,
                                        int block_size, int n_blocks, float *scores, size_t score_stride) {
    if (!q || !k || !query_positions || !key_positions || !scores || n_queries <= 0 || n_heads <= 0 ||
        dim <= 0 || n_keys <= 0 || block_size <= 0 || n_blocks <= 0 || score_stride < (size_t)n_blocks)
        return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;

    for (int key = 0; key < n_keys; ++key) {
        const int32_t pos = key_positions[key];
        if (pos < -1 || (pos >= 0 && pos / block_size >= n_blocks)) return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;
    }
    for (int query = 0; query < n_queries; ++query) {
        if (query_positions[query] < 0) return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;
        float *rows = scores + (size_t)query * (size_t)n_heads * score_stride;
        for (int head = 0; head < n_heads; ++head)
            for (int block = 0; block < n_blocks; ++block)
                rows[(size_t)head * score_stride + (size_t)block] = -INFINITY;

        for (int key = 0; key < n_keys; ++key) {
            const int32_t pos = key_positions[key];
            if (pos < 0 || pos > query_positions[query]) continue;
            const size_t block = (size_t)(pos / block_size);
            for (int head = 0; head < n_heads; ++head) {
                const float *qh = q + ((size_t)query * (size_t)n_heads + (size_t)head) * (size_t)dim;
                const float *kh = k + (size_t)key * (size_t)dim;
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    const float qv = qh[d], kv = kh[d];
                    if (!isfinite(qv) || !isfinite(kv)) return STRATA_MINIMAX_MSA_NONFINITE_VALUE;
                    dot += qv * kv;
                }
                float *score = rows + (size_t)head * score_stride + block;
                if (dot > *score) *score = dot;
            }
        }
    }
    return STRATA_MINIMAX_MSA_OK;
}

static int forced_block(int block, int32_t query_pos, int block_size, int local_blocks) {
    const int current = query_pos / block_size;
    return block <= current && current - block < local_blocks;
}

int strata_minimax_msa_select_blocks(const float *scores, const int32_t *query_positions,
                                     int n_queries, int n_heads, int n_blocks, int block_size,
                                     int top_k, int local_blocks, size_t score_stride,
                                     int32_t *ids, size_t id_stride) {
    if (!scores || !query_positions || !ids || n_queries <= 0 || n_heads <= 0 || n_blocks <= 0 ||
        block_size <= 0 || top_k <= 0 || top_k > n_blocks || local_blocks <= 0 || local_blocks > top_k ||
        score_stride < (size_t)n_blocks || id_stride < (size_t)top_k)
        return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;

    for (int query = 0; query < n_queries; ++query) {
        if (query_positions[query] < 0) return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;
        for (int head = 0; head < n_heads; ++head) {
            const float *row = scores + ((size_t)query * (size_t)n_heads + (size_t)head) * score_stride;
            int32_t *out = ids + ((size_t)query * (size_t)n_heads + (size_t)head) * id_stride;
            for (int block = 0; block < n_blocks; ++block)
                if (!valid_score(row[block])) return STRATA_MINIMAX_MSA_NONFINITE_VALUE;

            for (int rank = 0; rank < top_k; ++rank) {
                int best = -1;
                for (int block = 0; block < n_blocks; ++block) {
                    int selected = 0;
                    for (int prior = 0; prior < rank; ++prior) selected |= out[prior] == block;
                    if (selected) continue;
                    if (best < 0) {
                        best = block;
                        continue;
                    }
                    const int block_forced = forced_block(block, query_positions[query], block_size, local_blocks);
                    const int best_forced = forced_block(best, query_positions[query], block_size, local_blocks);
                    if (block_forced != best_forced) {
                        if (block_forced) best = block;
                    } else if (row[block] > row[best] || (row[block] == row[best] && block < best)) {
                        best = block;
                    }
                }
                out[rank] = best;
            }
            // Kernel consumers form a block mask, so return a canonical order independent of ranking.
            for (int i = 1; i < top_k; ++i) {
                const int32_t value = out[i];
                int j = i;
                while (j > 0 && out[j - 1] > value) { out[j] = out[j - 1]; --j; }
                out[j] = value;
            }
        }
    }
    return STRATA_MINIMAX_MSA_OK;
}

static int block_selected(const int32_t *ids, int top_k, int block) {
    for (int i = 0; i < top_k; ++i) if (ids[i] == block) return 1;
    return 0;
}

static int attention_score(const float *q, const float *k, int dim, float *score) {
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
        if (!isfinite(q[d]) || !isfinite(k[d])) return STRATA_MINIMAX_MSA_NONFINITE_VALUE;
        dot += q[d] * k[d];
    }
    *score = dot;
    return STRATA_MINIMAX_MSA_OK;
}

int strata_minimax_msa_attention_f32(const float *q, const float *k, const float *v,
                                     const int32_t *query_positions, const int32_t *key_positions,
                                     const int32_t *selected_blocks, int n_queries,
                                     int query_heads, int kv_heads, int dim, int n_keys,
                                     int block_size, int n_blocks, int top_k,
                                     size_t ids_stride, float *out) {
    if (!q || !k || !v || !query_positions || !key_positions || !selected_blocks || !out ||
        n_queries <= 0 || query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 ||
        dim <= 0 || n_keys <= 0 || block_size <= 0 || n_blocks <= 0 || top_k <= 0 ||
        top_k > n_blocks || ids_stride < (size_t)top_k)
        return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;
    for (int key = 0; key < n_keys; ++key) {
        const int32_t pos = key_positions[key];
        if (pos < -1 || (pos >= 0 && pos / block_size >= n_blocks)) return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;
    }
    const float scale = 1.0f / sqrtf((float)dim);
    const int group = query_heads / kv_heads;
    for (int query = 0; query < n_queries; ++query) {
        if (query_positions[query] < 0) return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;
        const int32_t *query_ids = selected_blocks + (size_t)query * kv_heads * ids_stride;
        for (int kh = 0; kh < kv_heads; ++kh) {
            const int32_t *ids = query_ids + (size_t)kh * ids_stride;
            for (int i = 0; i < top_k; ++i) {
                if (ids[i] < 0 || ids[i] >= n_blocks) return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;
                for (int j = 0; j < i; ++j) if (ids[i] == ids[j]) return STRATA_MINIMAX_MSA_INVALID_ARGUMENT;
            }
            for (int qh = kh * group; qh < (kh + 1) * group; ++qh) {
                const float *qh_ptr = q + ((size_t)query * query_heads + qh) * dim;
                float max_score = -INFINITY;
                for (int key = 0; key < n_keys; ++key) {
                    const int32_t pos = key_positions[key];
                    if (pos < 0 || pos > query_positions[query] ||
                        !block_selected(ids, top_k, pos / block_size)) continue;
                    float score;
                    const int status = attention_score(qh_ptr, k + ((size_t)key * kv_heads + kh) * dim, dim, &score);
                    if (status != STRATA_MINIMAX_MSA_OK) return status;
                    score *= scale;
                    if (score > max_score) max_score = score;
                }
                float denom = 0.0f;
                if (isfinite(max_score)) {
                    for (int key = 0; key < n_keys; ++key) {
                        const int32_t pos = key_positions[key];
                        if (pos < 0 || pos > query_positions[query] ||
                            !block_selected(ids, top_k, pos / block_size)) continue;
                        float score;
                        const int status = attention_score(qh_ptr, k + ((size_t)key * kv_heads + kh) * dim, dim, &score);
                        if (status != STRATA_MINIMAX_MSA_OK) return status;
                        denom += expf(score * scale - max_score);
                    }
                }
                float *dst = out + ((size_t)query * query_heads + qh) * dim;
                for (int d = 0; d < dim; ++d) {
                    float sum = 0.0f;
                    if (denom > 0.0f) {
                        for (int key = 0; key < n_keys; ++key) {
                            const int32_t pos = key_positions[key];
                            if (pos < 0 || pos > query_positions[query] ||
                                !block_selected(ids, top_k, pos / block_size)) continue;
                            float score;
                            const int status = attention_score(qh_ptr, k + ((size_t)key * kv_heads + kh) * dim, dim, &score);
                            if (status != STRATA_MINIMAX_MSA_OK) return status;
                            const float prob = expf(score * scale - max_score) / denom;
                            const float value = v[((size_t)key * kv_heads + kh) * dim + d];
                            if (!isfinite(value)) return STRATA_MINIMAX_MSA_NONFINITE_VALUE;
                            sum += prob * value;
                        }
                    }
                    dst[d] = sum;
                }
            }
        }
    }
    return STRATA_MINIMAX_MSA_OK;
}
