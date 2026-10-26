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
/// mixes() in one launch: glm_rms_norm_rows(h [tokens][4 n], no weight) -> flat, the F32 rows fn [24][4 n] -> mixes,
/// ds41_hc_coeff -> coeff; bitwise the three. false (nothing launched): unaligned rows.
bool ds41_hc_mixes(const float* h, const float* fn, const float* scale, const float* base, float* flat, float* mixes, float* coeff, int n,
                   int iterations, float eps, float rms_eps, void* stream, int tokens);
/// y[j] = sum_s w[s] h[s][j] over the 4 streams, in stream order (w: 4 values).
/// Per token t: h + t 4n, w + t 24 (a coefficient row), y + t n.
void ds41_hc_pre(const float* h, const float* w, float* y, int n, void* stream, int tokens = 1);
/// out[d][j] = x[j] post[d] + sum_s res[s][j] comb[d + 4 s] (out may not alias res).
/// Per token t: x + t n, res/out + t 4n, post/comb + t 24.
void ds41_hc_post(const float* x, const float* res, const float* post, const float* comb, float* out, int n, void* stream,
                  int tokens = 1);
/// STRATA_DS41_DEFER_RANK: h[d] += y * sum_s post[s] comb[d + 4 s] (comb null: h[d] += y * post[d]); post / comb at the
/// coefficient arrays' offsets 4 / 8, 24 floats per token.
/// STRATA_DS41_GPU_ROUTE: the router's top-k on the GPU (ids / wts [T][K]) and the grouped kernels' table of the routes
/// held in VRAM (slots[e]: the expert's tier blob, 0 when not resident); cpu_topk >= 0 drops non-resident picks of
/// rank >= cpu_topk (weight 0). table: GroupedRunner's layout at capacity cap = T * K.
void ds41_route_table(const float* logits, const float* bias, const unsigned long long* slots, int T, int E, int K, bool normalize,
                      float scale, int cpu_topk, int cap, int* ids, float* wts, void* table, void* stream);
/// ds41_scatter_axpy with the row count m on the device
void ds41_scatter_axpy_dev(const float* src, const int* idx, const float* w, const int* m, int n, float* dst, void* stream);
void ds41_defer_add(const float* y, float* h, const float* post, const float* comb, int n, void* stream, int tokens = 1);
/// ds41_add(moe, peer), ds41_add(., gsum), ds41_add(., y) -> y, then ds41_hc_post(y, ...) in one launch (peer, gsum optional).
void ds41_hc_post_sum(const float* moe, const float* peer, const float* gsum, float* y, const float* res, const float* post, const float* comb,
                      float* out, int n, void* stream, int tokens);
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
                    float scale, float* scores, float* out, void* stream, const int* nsel_dev = nullptr);
/// (nsel_dev: the selected rows' count read on the device, at most nsel - the launch's bound)
size_t ds41_attention_scratch(int heads);
/// ds41_attention for `tokens` queries in one launch (wss, css: token t's window and comp at t wss, t css floats -
/// batched sessions); info[4 t..] = {first slot, window rows, selected rows (with
/// nsel_dev: the bound, the count at nsel_dev[t]), -} on the device; bound >= every token's rows; scores: tokens x
/// ds41_attention_scratch(heads) bytes.
void ds41_attention_multi(const float* q, int tokens, int heads, int dim, const float* window, int ring, const int* info, int bound,
                          const float* comp, int comp_stride, const int* sel, int sel_stride, const int* nsel_dev, const float* sinks,
                          float scale, float* scores, float* out, void* stream, size_t wss = 0, size_t css = 0);
/// ds41_attention for `tokens` queries at once (prefill): info[4 t..] = {first window slot, window rows, selected rows,
/// unused} per token, its selected rows at sel + t sel_stride; window row r of token t is slot (first + r) % ring.
/// q, out: [tokens][heads][dim]. dim 256 or 512, heads % 8 == 0.
void ds41_attention_batched(const float* q, int tokens, int heads, int dim, const float* window, int ring, const int* info,
                            const float* comp, int comp_stride, const int* sel, int sel_stride, const float* sinks, float scale,
                            float* out, void* stream);
/// score[t] = sum_h relu(keys[t].q[h]) (w[h] w_scale) for rows t < rows (keys[t] at t * key_stride, width dim).
void ds41_index_score(const float* q, int heads, int dim, const float* keys, int key_stride, int rows, const float* w,
                      float w_scale, float* score, void* stream, bool half = false);
/// The ratio-2 compressor's merge: out[c] = sum_r v_r[c] softmax_r(s_r[c]) over the two positions.
/// ds41_index_score for a pass of `tokens` tokens: token t's q [heads][dim] and w [heads] at t, its
/// (pos0 + t + 1) / ratio scores at score + t * score_stride.
void ds41_index_score_batched(const float* q, int heads, int dim, const float* keys, int key_stride, int pos0, int ratio, int tokens,
                              const float* w, float w_scale, float* score, int score_stride, void* stream, bool half = false);
/// Each token's top k of its visible scores (ties to the lowest index; -inf rows dropped), ascending, into sel [tokens][k]
/// and count [tokens]. mode 1: also the top cand_k blocks of `block` rows by their maxima (the newest pinned) as
/// flags cand [tokens][cand_stride]; mode 2: only rows in cand's flagged blocks.
void ds41_index_topk(const float* score, int score_stride, int tokens, int pos0, int ratio, int k, int* sel, int* count, int mode,
                     uint8_t* cand, int cand_stride, int block, int cand_k, void* stream, const int* tpos = nullptr);
/// info[4 t + 2] = count[t] where the token sees a compressed row, else 0.
void ds41_patch_info(int* info, const int* count, int tokens, int pos0, int ratio, void* stream);
/// ds41_compress2 of consecutive pairs: out[r] from rows 2r and 2r + 1 of v and s ([2 rows][n]).
void ds41_compress2_rows(const float* v, const float* s, float* out, int rows, int n, void* stream);
/// Device bytes: count blocks of `rows` rows of `width` bytes, block c row r from src + c sstep + r spitch to
/// dst + c dstep + r dpitch (narrow strided rows the copy engine moves slowly).
void ds41_copy_rows(uint8_t* dst, size_t dpitch, size_t dstep, const uint8_t* src, size_t spitch, size_t sstep, int width, int rows,
                    int count, void* stream);
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
/// y[t ldy + r] = W[r] . x[rows[t] ldx] for n <= 8 tokens, each BF16 row of W [out][in] read once (in % 64 == 0).
void ds41_bf16_gemv_mt(const uint16_t* w, const float* x, int ldx, const int* rows, int n, float* y, int ldy, int out, int in,
                       void* stream);
/// ds41_bf16_gemv_mt for `groups` matrices in one launch: group g's matrix at w + g w_gstride (elements), its columns
/// x + g x_goff, its rows y + g y_goff; bitwise the `groups` calls.
void ds41_bf16_gemv_mt_groups(const uint16_t* w, size_t w_gstride, const float* x, int ldx, int x_goff, const int* rows, int n, float* y, int ldy,
                              int y_goff, int out, int in, int groups, void* stream);
/// rows [t][width] of src into ring slot (pos + t) % ring of dst (one launch for a block's keys).
void ds41_ring_rows(const float* src, int rows, int width, int pos, int ring, float* dst, void* stream);
/// The same over MXFP4 rows (ggml block_mxfp4, row_bytes apart; in % 32 == 0).
void ds41_mxfp4_gemv_mt(const uint8_t* w, size_t row_bytes, const float* x, int ldx, const int* rows, int n, float* y, int ldy,
                        int out, int in, void* stream);
/// DSV4 MoE routing per token on the device: sqrt(softplus(logit)) scores, top-k on score + bias, the picked scores
/// normalized (when asked) times scale: ids/w [tokens][k].
void ds41_moe_route(const float* logits, const float* bias, int tokens, int experts, int k, float scale, bool normalize, int* ids, float* w,
                    void* stream);
/// y[p ldy + r] = W_{ids[p]}[r] . x[(p / div) ldx] for `pairs` pairs, MXFP4 experts expert_bytes apart (in % 32 == 0).
void ds41_mxfp4_moe(const uint8_t* w, size_t expert_bytes, size_t row_bytes, const int* ids, int pairs, const float* x, int ldx, int div,
                    float* y, int ldy, int out, int in, void* stream);
/// DSpark's Markov chain on the device: row = w1[tok[i]] as floats (dim BF16); tok[i + 1] = argmax v (the first of equal
/// maxima); conf[i] = sigmoid(cw . [x (hidden); row (dim)]).
void ds41_markov_row(const uint16_t* w1, int dim, const int* tok, int i, float* row, void* stream);
void ds41_argmax_into(const float* v, int n, int* tok, int i, void* stream);
void ds41_confidence(const float* cw, const float* x, int hidden, const float* row, int dim, float* conf, int i, void* stream);
/// keys: with half, an FP16 cache (IEEE half bits; the pointer cast to float*) - for ds41_index_score*.
/// An FP16 cache's selected rows as floats: out[(t topk + j) rw ..] = cache[sel[t topk + j] rw ..] for each token's
/// first min(topk, (pos0 + t + 1) / ratio) picks.
void ds41_gather_sel_h(const uint16_t* cache, int rw, const int* sel, int topk, int tokens, int pos0, int ratio, float* out, void* stream,
                       const int* tpos = nullptr, size_t css = 0);
/// (tpos, css: batched sessions - token t at position tpos[t], its cache at t css elements (FP16) / bytes (Q8_0))
/// Q8_0 caches (rows of row_bytes: per 32 values an FP16 scale and 32 int8): the index scores (keys from value
/// key_off; rows: (pos0 + t + 1) / ratio per token, or fixed_rows > 0 for all), the selected rows' gather as in
/// ds41_gather_sel_h, and float rows [rows][rw] stored as such rows.
void ds41_index_score_q8(const float* q, int heads, int dim, const uint8_t* cache, size_t row_bytes, int key_off, int pos0, int ratio, int tokens,
                         int fixed_rows, const float* w, float w_scale, float* score, int score_stride, void* stream,
                         const int* tpos = nullptr, size_t css = 0);
void ds41_gather_sel_q8(const uint8_t* cache, size_t row_bytes, int rw, const int* sel, int topk, int tokens, int pos0, int ratio, float* out,
                        void* stream, const int* tpos = nullptr,
                        size_t css = 0);
void ds41_quant_rows_q8(const float* x, int rows, int rw, uint8_t* out, size_t row_bytes, void* stream);
/// GEMM index scoring: a Q8_0 cache's keys [row0, row0 + rows) as FP16 [rows][dim]; and the reduction of the
/// products c [tokens][heads][rows] (c[(t heads + h) rows + r] = k_r . q_th) into
/// score[t][row0 + r] = w_scale sum_h w[t][h] relu(c) for each token's (pos0 + t + 1) / ratio visible rows.
void ds41_keys_q8_f16(const uint8_t* cache, size_t row_bytes, int key_off, int dim, int row0, int rows, uint16_t* out, void* stream);
void ds41_index_reduce(const float* c, int rows, int tokens, int heads, const float* w, float w_scale, int pos0, int ratio, int row0,
                       float* score, int score_stride, void* stream);
/// Batched sessions: token t of a pass is session t at position tpos[t] (device). The index scores of each token over
/// its own cache (keys + t kss, (tpos[t] + 1) / ratio rows, max_rows the bound); the compressor step (ratio 1: out = v;
/// ratio 2: even positions stash v, s into stash_* [tokens][n], odd ones merge with it into out); and row t of src
/// [rows][width] to dst + dst_row[t] dst_stride (dst_row[t] < 0: skipped).
void ds41_index_score_sessions(const float* q, int heads, int dim, const float* keys, int key_stride, size_t kss, const int* tpos, int max_rows,
                               int ratio, int tokens, const float* w, float w_scale, float* score, int score_stride, void* stream, bool half = false);
void ds41_compress_sessions(const float* v, const float* s, float* stash_v, float* stash_s, int n, const int* tpos, int ratio, int tokens,
                            float* out, void* stream);
void ds41_scatter_bytes(const void* src, int rows, size_t row_bytes, const int* dst_row, void* dst, size_t dst_pitch, void* stream);
void ds41_scatter_rows(const float* src, int rows, int width, const int* dst_row, float* dst, size_t dst_stride, void* stream);
/// STRATA_DS41_INDEX_FAST: the index scores of `tokens` tokens over a cache of `kind` (0 float, 1 FP16, 2 Q8_0) rows of
/// row_bytes, keys from value key_off; rows (pos0 + t + 1) / ratio per token (tpos: per-token positions, cache at t css
/// bytes; fixed_rows > 0: that many for all), max_rows the grid's bound. A thread per row. false: not covered.
bool ds41_index_score_fast(int kind, const float* q, int heads, int dim, const void* cache, size_t row_bytes, int key_off, int pos0, int ratio,
                           int tokens, int max_rows, int fixed_rows, const float* w, float w_scale, float* score, int score_stride, void* stream,
                           const int* tpos = nullptr, size_t css = 0);
/// dst[b][a][j] = src[a][b][j] (a < na, b < nb, j < width floats): [tokens][groups][w] <-> [groups][tokens][w].
void ds41_swap_blocks(const float* src, float* dst, int na, int nb, int width, void* stream);
/// dst[t ld_dst + j] = mean over the 4 hyper-connection streams of h[t][s][j].
void ds41_stream_mean(const float* h, float* dst, size_t ld_dst, int n, int tokens, void* stream);

}  // namespace strata::kernels
