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

}  // namespace strata::kernels
