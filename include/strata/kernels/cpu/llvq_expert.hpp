// include/strata/kernels/cpu/llvq_expert.hpp - routed experts as llvq Tetra records (strata-glm-llvq-encode, docs/GLM_LLVQ.md)
// on the CPU expert lane: the native-expert entry points (native_expert.hpp) branch here for kLlvqType.
//
// A record stores W' = W Q^T (llvq's Rotation Q of the input width, one seed for the model), so a row is
// y_r = W'_r . (Q x). The activation is rotated once per input vector and permuted per 24-block into the words' trio
// order (llvq_prepare): the gate/up activation once per token for every expert of a layer, the down activation once
// per expert-token after SwiGLU. Both are plain f32 (act_bytes = 4 n_embd, h_bytes = 4 n_ff).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// The expert "type" of llvq Tetra records: not a ggml type (ggml's types are below GGML_TYPE_COUNT).
constexpr int kLlvqType = 1000;

/// One model's rotation seed (every unit file's header carries it). Builds the decode tables once; call before
/// native_fmt(kLlvqType, ...). Throws without AVX2+FMA (the kernels need them) or for a second, different seed.
void llvq_expert_init(uint64_t seed);
bool llvq_expert_ready() noexcept;
/// Bytes of one record (rows x cols): 24-byte header, f32 row scales, word rows, BF16 tail.
size_t llvq_record_bytes(int64_t rows, int64_t cols) noexcept;

/// x (n floats; n = 4096 or 2048, a power of two) -> Q x, permuted per 24-block into trio order (n floats).
void llvq_prepare(const float* x, int n, float* xp);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows [r0, r1), with GLM's clamp when swiglu_limit > 0.
void llvq_gu_rows(const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff, int r0, int r1,
                  float swiglu_limit);
/// out[t][r] = down_r . hq[t] for rows [r0, r1).
void llvq_down_rows(const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
