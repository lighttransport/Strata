// include/strata/kernels/ds41.hpp - DeepSeek-V4.1-Flash decode kernels (CUDA/HIP), one token at a time.
//
// The operations follow llama.cpp PR #30817 (src/models/deepseek4.cpp, deepseek41.cpp), the reference this engine is
// checked against: hyper-connection mixes (sigmoid pre/post, Sinkhorn comb), the sliding-window + compressed-row
// attention with per-head sinks, the lightning indexer's score, the ratio-2 compressor's gated merge and the Engram
// gate. All buffers are device pointers; every call enqueues on `stream` and allocates nothing.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// coeff[24] = {pre[4], post[4], comb[16]} from the 24 projected mixes: pre = sigmoid(m * s0 + b) + eps,
/// post = 2 sigmoid(m * s1 + b), comb = Sinkhorn(m * s2 + b) as llama.cpp's build_hc_sinkhorn (comb[dst + 4 src]:
/// softmax over dst, + eps, column normalization, then iterations - 1 row/column normalizations).
/// tokens: rows of 24 mixes / 24 coefficients.
void ds41_hc_coeff(const float* mixes, const float* scale, const float* base, float* coeff, int iterations, float eps,
                   void* stream, int tokens = 1);
/// y[j] = sum_s w[s] h[s][j] over the 4 streams, in stream order (w: 4 values).
/// Per token t: h + t 4n, w + t 24 (a coefficient row), y + t n.
void ds41_hc_pre(const float* h, const float* w, float* y, int n, void* stream, int tokens = 1);
/// out[d][j] = x[j] post[d] + sum_s res[s][j] comb[d + 4 s] (out may not alias res).
/// Per token t: x + t n, res/out + t 4n, post/comb + t 24.
void ds41_hc_post(const float* x, const float* res, const float* post, const float* comb, float* out, int n, void* stream,
                  int tokens = 1);
/// h[s][j] = x[j] for the 4 streams.
void ds41_broadcast_streams(const float* x, float* h, int n, void* stream, int tokens = 1);
/// Rotates `pairs` adjacent pairs at `offset` of each of `rows` rows (stride floats apart) by cs[i] = {cos, sin}
/// (the caller's table holds the sign of sin, so one table rotates forward and its negation back).
/// Rows [g group, (g+1) group) use the table at cs + g cs_step (group 0: one table for every row).
void ds41_rope(float* x, int rows, int stride, int offset, int pairs, const float* cs, void* stream, int group = 0,
               int cs_step = 0);
/// Attention of `heads` query heads of width dim over the window rows [window_rows][dim] and the selected compressed
/// rows comp[sel[i] * comp_stride .. +dim]; keys are the values (one KV head, so each row is read once per 8 heads).
/// softmax(scale q.k) with the head's sink in the denominator only (llama.cpp soft_max_ext with sinks).
/// scores: ds41_attention_scratch(heads) bytes; out: [heads][dim]. heads % 8 == 0, dim % 128 == 0, rows <= 4096.
/// Window row r is slot (window_first + r) % window_ring of `window`.
void ds41_attention(const float* q, int heads, int dim, const float* window, int window_first, int window_ring,
                    int window_rows, const float* comp, int comp_stride, const int* sel, int nsel, const float* sinks,
                    float scale, float* scores, float* out, void* stream);
size_t ds41_attention_scratch(int heads);
/// ds41_attention for `tokens` queries at once (prefill): info[4 t..] = {first window slot, window rows, selected rows,
/// unused} per token, its selected rows at sel + t sel_stride; window row r of token t is slot (first + r) % ring.
/// q, out: [tokens][heads][dim]. dim 256 or 512, heads % 8 == 0.
void ds41_attention_batched(const float* q, int tokens, int heads, int dim, const float* window, int ring, const int* info,
                            const float* comp, int comp_stride, const int* sel, int sel_stride, const float* sinks, float scale,
                            float* out, void* stream);
/// score[t] = sum_h relu(keys[t].q[h]) (w[h] w_scale) for rows t < rows (keys[t] at t * key_stride, width dim).
void ds41_index_score(const float* q, int heads, int dim, const float* keys, int key_stride, int rows, const float* w,
                      float w_scale, float* score, void* stream);
/// The ratio-2 compressor's merge: out[c] = sum_r v_r[c] softmax_r(s_r[c]) over the two positions.
/// ds41_index_score for a pass of `tokens` tokens: token t's q [heads][dim] and w [heads] at t, its
/// (pos0 + t + 1) / ratio scores at score + t * score_stride.
void ds41_index_score_batched(const float* q, int heads, int dim, const float* keys, int key_stride, int pos0, int ratio, int tokens,
                              const float* w, float w_scale, float* score, int score_stride, void* stream);
/// Each token's top k of its visible scores (ties to the lowest index; -inf rows dropped), ascending, into sel [tokens][k]
/// and count [tokens]. mode 1: also the top cand_k blocks of `block` rows by their maxima (the newest pinned) as
/// flags cand [tokens][cand_stride]; mode 2: only rows in cand's flagged blocks.
void ds41_index_topk(const float* score, int score_stride, int tokens, int pos0, int ratio, int k, int* sel, int* count, int mode,
                     uint8_t* cand, int cand_stride, int block, int cand_k, void* stream);
/// info[4 t + 2] = count[t] where the token sees a compressed row, else 0.
void ds41_patch_info(int* info, const int* count, int tokens, int pos0, int ratio, void* stream);
/// ds41_compress2 of consecutive pairs: out[r] from rows 2r and 2r + 1 of v and s ([2 rows][n]).
void ds41_compress2_rows(const float* v, const float* s, float* out, int rows, int n, void* stream);
void ds41_compress2(const float* v0, const float* s0, const float* v1, const float* s1, float* out, int n, void* stream);
/// Engram: kv = [key_0..key_3 (n each), value (n)]; per stream s: d = sum_j rms(h_s)_j rms(key_s)_j g[s n + j] / sqrt(n),
/// gate = sigmoid(sign(d) sqrt(max(|d|, 1e-6))), h_s += value * gate.
void ds41_engram(float* h, const float* kv, const float* g, int n, float eps, void* stream, int tokens = 1);
/// dst[i] = src[idx[i]] for m rows of n floats.
void ds41_gather(const float* src, const int* idx, int m, int n, float* dst, void* stream);
/// dst[idx[i]] += w[i] src[i] for m rows of n floats, in row order.
void ds41_scatter_axpy(const float* src, const int* idx, const float* w, int m, int n, float* dst, void* stream);
/// y += a x
void ds41_axpy(const float* x, float a, float* y, int n, void* stream);
/// y = a + b
void ds41_add(const float* a, const float* b, float* y, int n, void* stream);
/// y = W x, W row-major [rows][in] float.
/// Per token t: x + t in, y + t rows.
void ds41_f32_gemv(const float* w, const float* x, float* y, int rows, int in, void* stream, int tokens = 1);
/// y[i] = x[i] rounded to BF16 (nearest even).
void ds41_to_bf16(const float* x, uint16_t* y, int64_t n, void* stream);
/// y = W x with BF16 W, per token as ds41_f32_gemv.
void ds41_bf16_gemv(const uint16_t* w, const float* x, float* y, int rows, int in, void* stream, int tokens = 1);

}  // namespace strata::kernels
