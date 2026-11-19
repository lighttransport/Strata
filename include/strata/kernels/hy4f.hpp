// include/strata/kernels/hy4f.hpp - Hy4-Flash (GGUF arch hy_v4_flash) decode kernels (CUDA/HIP), one token at a time.
//
// The operations follow llama.cpp PR #32517 (src/models/hy-v4-flash.cpp), the reference this engine is checked
// against. The rest of the layer reuses the DS4.1F / GLM kernels (iq_mmvq, ds41_attention, ds41_rope, glm_rms_norm*).
// All buffers are device pointers; every call enqueues on `stream` and allocates nothing.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// iHC coefficients from the projected mixes (no Sinkhorn): pre[i] = sigmoid(m[i] scale[0] + base[i]) + eps for i < hc;
/// with_post: post[i] = 2-magnitude form sigmoid(m[hc + i] scale[1] + base[hc + i]) * magnitude + eps.
/// out: pre[hc] then post[hc].
void hy4_hc_coeff(const float* mixes, const float* scale, const float* base, float* out, int hc, bool with_post,
                  float magnitude, float eps, void* stream);
/// h[s][j] += x[j] post[s] for the hc streams (the residual is the stream itself: hy_v4 has no cross-stream mix).
void hy4_hc_post(const float* x, const float* post, float* h, int n, int hc, void* stream);
/// y[s][j] = sum over the hc streams h[s'][j] w[s'] (pre-mix), in stream order.
void hy4_hc_pre(const float* h, const float* w, float* y, int n, int hc, void* stream);
/// Per head b: y[b][r] = dequant(W_b[r]) . x[b] for Q8_0 W_b [rows][in] (34 B per 32 values), W_b at b * rows rows,
/// x[b] at b * x_stride floats, y[b] at b * y_stride floats. FP32 activations (no q8_1 rounding).
void hy4_q8_0_heads(const void* w, const float* x, int x_stride, float* y, int y_stride, int rows, int in, int heads,
                    void* stream);
/// MLA query/key rows for ds41_attention (width 640): q[h] = [qa[h] (512) | qpe[h] (64, at qpe + h * qpe_stride) | 0],
/// k = [kv (512) | kpe (64) | 0].
void hy4_mla_pack_q(const float* qa, const float* qpe, int qpe_stride, float* q, int heads, void* stream);
void hy4_mla_pack_k(const float* kv, const float* kpe, float* k, void* stream);
/// y[i] = x[i] sigmoid(g[i])
void hy4_mul_sigmoid(const float* x, const float* g, float* y, int n, void* stream);
/// Engram on the hc streams h [hc][n]: kv = [key_0 .. key_{hc-1}, value] (n each, bias included).
/// d_s = sum_j rms(h_s)_j rms(key_s)_j g[s n + j] / sqrt(n) (rms without weight, eps);
/// gate_s = sigmoid(sign(d_s) sqrt(max(|d_s|, 1e-6))), sign(0) = +1; signal = rms(value) * gamma;
/// h_s[j] += signal[j] (gate_s + hc_scale[s n + j]) when `any` (a token with no valid n-gram adds nothing).
void hy4_engram(float* h, const float* kv, const float* g, const float* gamma, const float* hc_scale, int n, int hc,
                float eps, bool any, void* stream);

/// DSA kpool: one pooled indexer key from a complete group of `kpool` cache rows vg[m] = [value (dim) | gate (dim)]:
/// out[c] = sum_m softmax_m(gate[m][c] + ape[m dim + c]) value[m][c].
void hy4_kpool(const float* vg, const float* ape, float* out, int kpool, int dim, void* stream);

// ---- fused decode kernels (one launch each instead of 2-5)
/// The whole iHC mix: r = rms(h [hc][n] as one row, eps rms_eps); mixes[j] = NVFP4 row j of fn (rows of hc n values) . (h r)
/// for j < rows; coeff = hy4_hc_coeff (rows == 2 hc: pre and post; rows == hc: pre only); x = sum_s pre_s h_s;
/// y = rms(x) * norm_w (eps rms_eps) and its q8_1 blocks (n % 32 == 0). One block; mixes use FP32 activations.
void hy4_hc_mix(const float* h, const void* fn, int rows, const float* scale, const float* base, float* coeff, float magnitude,
                float hc_eps, float rms_eps, const float* norm_w, float* x, float* y, void* y_q8, int n, int hc, void* stream);
/// h_s += rms(x + add) * w * post[s] for the hc streams (add may be null): the post norm and the residual write in one.
void hy4_norm_hc_post(const float* x, const float* add, const float* w, const float* post, float* h, int n, int hc, float eps,
                      void* stream);
/// MLA rows after the projections: q640[h][512..640) = [rope(q[h][nope..nope+64)) | 0] for every head (q640[h][0..512) is
/// the absorbed query, written by hy4_q8_0_heads), and the cache row k = [rms(kvpe[0..512)) * kv_w | rope(kvpe[512..576)) | 0].
/// cs: 32 {cos, sin} pairs (ds41_rope's convention).
void hy4_mla_finish(const float* q, int heads, int q_stride, int nope, const float* kvpe, const float* kv_w, const float* cs, float* q640,
                    float* krow, float eps, void* stream);
/// q8_1 blocks of x[i] sigmoid(g[i]) (n % 32 == 0): the attention gate and the wo input quantization in one.
void hy4_mul_sigmoid_q8(const float* x, const float* g, void* q8, int n, void* stream);

/// MoE routing on the device (one block, E <= 1024, K <= 16): p = sigmoid(logit), the top K of p + bias (ties: lowest id),
/// weights p / max(sum p, 6.1035e-5) (norm) times scale. Routes whose expert has a GPU slot (slots[e] != 0: the blob's
/// address) go to `table`, the grouped-expert table native_expert_grouped / ds41_scatter_axpy_dev read (capacity K:
/// ptr[K] u64, start[K+1], n_groups, dst[K], tok[K], w[K]); the others to the CPU: host = {cpu_ids[K] (-1: on the GPU),
/// w[K] (float bits), ids[K]} as K * 3 ints.
void hy4_route(const float* logits, const float* bias, const unsigned long long* slots, int E, int K, float scale, bool norm,
               void* table, int* host, void* stream);
/// bytes of hy4_route's table for K routes
size_t hy4_route_table_bytes(int K);

}  // namespace strata::kernels
