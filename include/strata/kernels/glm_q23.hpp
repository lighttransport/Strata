#pragma once
#include "strata/kernels/iq_kernels.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Whether the canonical kernels cover this layout: Q2_K or Q3_K gate/up and down, whole 256-value blocks,
/// rows in multiples of 32, at most 4096 columns.
bool glm_q23_supported(const NativeExpertLayout& L) noexcept;
/// Bytes of scratch glm_q23_expert_grouped needs for `tokens` activations and `cap_entries` entries.
size_t glm_q23_scratch_bytes(const NativeExpertLayout& L, int64_t tokens, int64_t cap_entries);

/// Grouped Q2_K/Q3_K experts in the canonical arithmetic of canon_expert.hpp: each output row equals the CPU
/// pool's canonical row bit for bit. The grouping arguments are native_expert_grouped's (group g's blob at device
/// address grp_ptr[g]; its entries [grp_start[g], grp_start[g+1]) read token ent_tok[e] and write row ent_dst[e] of
/// `out`); `x` holds `tokens` unquantized activations of n_embd floats.
void glm_q23_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                            const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                            int64_t cap_groups, int64_t cap_entries, const float* x, int64_t tokens, void* scratch,
                            float* out, void* stream);

}  // namespace strata::kernels
