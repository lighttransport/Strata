#pragma once
// Independent decode rows in one launch (STRATA_GLM_BATCH_MIXERS=1): row r belongs to its own sequence and reads
// and writes that sequence's state through p[r]. Each row's arithmetic is that of the matching one-row call
// (glm_conv, glm_kda_gate, glm_kda_chunk with one token, a cache row copy, glm_index_prepare with one token), so a
// row's results do not depend on the other rows. The pointer tables travel as kernel arguments.
namespace strata::kernels {
constexpr int kGlmRowsMax = 32;
struct GlmRowPointers { float *p[kGlmRowsMax]; };
struct GlmRowPositions { int p[kGlmRowsMax]; };
// x and y hold `rows` rows of `channels`; history.p[r] is row r's convolution history.
void glm_conv_rows(const float *x, const float *weight, GlmRowPointers history, float *y, int channels, int kernel,
                   int rows, void *stream);
void glm_kda_gate_rows(const float *projected, const float *bias, const float *negative_a, float *log_decay,
                       int heads, int dim, float lower_bound, int rows, void *stream);
// glm_kda_chunk(tokens=1, row_parts=1, unprepared) for every row; state.p[r] is row r's recurrent state.
void glm_kda_chunk_rows(GlmRowPointers state, const float *q, const float *key, const float *value,
                        const float *decay, const float *beta, float *out, int heads, int dim, int rows,
                        void *stream, int columns = 128);
// Row r of `rows` goes to cache.p[r] + position.p[r] * width.
void glm_cache_rows_each(const float *rows, GlmRowPointers cache, GlmRowPositions position, int width, int count,
                         void *stream);
// glm_index_prepare(tokens=1) for every row at its own position, into its own pending keys, gates and pooled keys.
void glm_index_prepare_rows(const float *keys, const float *gates, const float *ape, GlmRowPointers pending_keys,
                            GlmRowPointers pending_gates, GlmRowPointers pooled, GlmRowPositions position, int rows,
                            int pool, int dim, void *stream);
// glm_index_select_batch for one query per row at its own position: row r's scores at scores + r * pools, its ids
// at ids + r * stride and its count at counts[r].
void glm_index_select_rows(const float *scores, int *ids, int *counts, int pools, GlmRowPositions position, int rows,
                           int pool, int top_k, int stride, void *stream);
// The MLA head absorption for every row with one read of the weights: for head h of `heads`,
// y[t * ld_y + h * out + o] = sum_i weight[(h * out + o) * in + i] * x[t * ld_x + h * in + i] (the layout of the
// strided-batched SGEMM it replaces), summed in a fixed order that does not depend on `rows` (FP32; its rounding
// differs from cuBLAS). in % 64 == 0, out % 16 == 0, rows 1..32.
void glm_mla_absorb_rows(const float *weight, const float *x, float *y, int in, int out, int heads, int rows,
                         int ld_x, int ld_y, void *stream);
} // namespace strata::kernels
