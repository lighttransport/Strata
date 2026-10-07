#pragma once
// Device-independent arithmetic for routed Q2_K/Q3_K experts ("canonical" mode): the CPU pool and the CUDA
// resident kernels produce the same bits, so where an expert runs never changes a logit.
//
//   activation   Q8_K per 256 values (canon_quant_q8k / quantize_q8_K_kernel): iscale = -127 / max, where max is
//                the first value of largest magnitude; q = min(127, nearest(fl(iscale * x))); d = 1 / iscale; a
//                zero block is all zeros, bsums included.
//   dot          per 256-value superblock b and lane j < 8, exact in int32:
//                  L_j = sum_{H<8} scale[2H + (j>>2)] * sum_{i<4} q[32H+4j+i] * y[32H+4j+i]
//                  M_j = min[2j] * bsum[2j] + min[2j+1] * bsum[2j+1]                       (Q2_K only)
//                then acc_j = fma(fl(-dmin * yd), M_j, acc_j); acc_j = fma(fl(d * yd), L_j, acc_j), and the row is
//                ((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7)). This is q23_avx2.cpp's kernel, lane for lane.
//   SwiGLU       canon_swiglu below: no libm, every operation rounded once.
//   routed sum   acc = 0; for k = top_k - 1 down to 0: if (w_k != 0) acc = fma(w_k, row_k, acc), on the CPU. A
//                GPU-resident route's row is copied to the host first, so the sum does not depend on which
//                device computed which row.
//
// CUDA translation units must not use --use_fast_math; the CPU one is built with -ffp-contract=off.
#include <cstdint>
#include <cstring>
#if defined(__CUDACC__)
#define STRATA_CANON_FN __device__ __forceinline__
#define STRATA_CANON_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#define STRATA_CANON_MUL(a, b) __fmul_rn((a), (b))
#define STRATA_CANON_ADD(a, b) __fadd_rn((a), (b))
#define STRATA_CANON_SUB(a, b) __fsub_rn((a), (b))
#define STRATA_CANON_DIV(a, b) __fdiv_rn((a), (b))
#else
#include <cmath>
#define STRATA_CANON_FN inline
#define STRATA_CANON_FMA(a, b, c) std::fma((a), (b), (c))
#define STRATA_CANON_MUL(a, b) ((a) * (b))
#define STRATA_CANON_ADD(a, b) ((a) + (b))
#define STRATA_CANON_SUB(a, b) ((a) - (b))
#define STRATA_CANON_DIV(a, b) ((a) / (b))
#endif

namespace strata::kernels::canon {

// exp(x) for x clamped to [-87, 88]: Cephes' expf reduction and polynomial, written as single rounded operations.
STRATA_CANON_FN float exp(float x) {
    x = x < -87.f ? -87.f : x;
    x = x > 88.f ? 88.f : x;
    // 1.5 * 2^23 rounds the product to the nearest integer (ties to even) in the default rounding mode.
    const float n = STRATA_CANON_SUB(STRATA_CANON_ADD(STRATA_CANON_MUL(x, 1.44269504088896341f), 12582912.f), 12582912.f);
    float r = STRATA_CANON_FMA(n, -0.693359375f, x);
    r = STRATA_CANON_FMA(n, 2.12194440e-4f, r);
    float p = STRATA_CANON_FMA(1.9875691500e-4f, r, 1.3981999507e-3f);
    p = STRATA_CANON_FMA(p, r, 8.3334519073e-3f);
    p = STRATA_CANON_FMA(p, r, 4.1665795894e-2f);
    p = STRATA_CANON_FMA(p, r, 1.6666665459e-1f);
    p = STRATA_CANON_FMA(p, r, 5.0000001201e-1f);
    const float y = STRATA_CANON_ADD(STRATA_CANON_FMA(p, STRATA_CANON_MUL(r, r), r), 1.f);
    const uint32_t bits = (uint32_t)((int)n + 127) << 23;
    float scale;
    memcpy(&scale, &bits, 4);
    return STRATA_CANON_MUL(y, scale);
}

// silu(gate) with GLM's clamp from above when limit > 0.
STRATA_CANON_FN float silu(float gate, float limit) {
    if (limit > 0) gate = gate < limit ? gate : limit;
    return STRATA_CANON_DIV(gate, STRATA_CANON_ADD(1.f, exp(-gate)));
}
// silu(gate) * up, with GLM's clamp when limit > 0 (gate from above, up on both sides). skip > 0 (opt-in, lossy):
// a unit whose |silu(gate)| is below it contributes exactly zero, so its up row need not be read.
STRATA_CANON_FN float swiglu(float gate, float up, float limit, float skip = 0.f) {
    const float s = silu(gate, limit);
    if (skip > 0 && (s < 0 ? -s : s) < skip) return 0.f;
    if (limit > 0) {
        up = up < limit ? up : limit;
        up = up > -limit ? up : -limit;
    }
    return STRATA_CANON_MUL(s, up);
}

}  // namespace strata::kernels::canon

#if !defined(__CUDACC__)
namespace strata::kernels::cpu {
/// Out-of-line canon::swiglu for translation units built with other floating-point flags.
float canon_swiglu(float gate, float up, float limit, float skip = 0.f);
/// Whether canon::swiglu returns zero for this gate whatever the up value (skip > 0).
bool canon_gate_skipped(float gate, float limit, float skip);
/// STRATA_Q23_GATE_STATS=path: histograms of |silu(gate)| per layer (quarter octaves from 2^-16), written at exit.
/// The decoder names the layer whose experts the pool is about to run.
void canon_gate_stats_layer(int layer);
/// x (n floats, n a multiple of 256) -> block_q8_K (292 bytes per 256 values), the canonical quantization.
void canon_quant_q8k(const float* x, void* blocks, int n);
/// The canonical routed sum of one token over rows [first, last): rows[k * hidden + r], weights[k], k descending.
void canon_route_sum(const float* rows, const float* weights, float* out, int top_k, int hidden, int first, int last);
}
#endif
