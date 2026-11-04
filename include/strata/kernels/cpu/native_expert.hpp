// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows]. Activations use the weight type's
// ggml `vec_dot_type`; dispatch selects a format-specific kernel or ggml's per-token dot. The arithmetic mode
// controls whether grouping tokens may change the selected row arithmetic.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "strata/kernels/cpu/expert_observer.hpp"
#include "strata/kernels/cpu/llvq_expert.hpp"

namespace strata::kernels::cpu {

/// Scratch capacity: GLM's 4096 values as Q8_K (16 x 292 bytes), or as llvq's rotated f32 (16384 bytes).
inline constexpr size_t kNativeActBytes = 16384;
/// Scratch capacity: GLM's 2048 values as Q8_K (8 x 292 bytes), or as llvq's rotated f32 (8192 bytes).
inline constexpr size_t kNativeHBytes = 8192;
/// Widest expert hidden: DeepSeek-V4.1-Flash's 2304 (GLM's 2048). Buffer sizes only: no layout or arithmetic depends on it.
inline constexpr int kNativeFF = 4096;

/// Arithmetic fixed for one pool batch. WindowInvariant preserves the legacy singleton result at every width.
enum class NativeArithmetic { Default, WindowInvariant };

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    // CPU-only Q2_K gate/up + expanded Q3_K down. Convert down bytes with the
    // shared C11 converter, set d_row=n_ff/256*114 and bytes=down_off+d_row*n_embd.
    // Original Q2 gate/up offsets stay unchanged. Explicit selection; no fallback.
    bool q3_expanded_down = false;
    bool nvfp4_k32 = false;         ///< explicit CPU-only repacked NVFP4 rows; all three projections must use K32
    bool lossless = false;          ///< direct magnitude planes for IQ2_S/IQ3_S
    int q23_layout = 0;             ///< 0: GGML rows; 1: tiled direct; 2: tiled lookup (Q2_K/Q3_K only)
    ExpertObserver* observer = nullptr;
    ExpertHiddenTransform* hidden_transform = nullptr;
    int observer_layer = -1;
    bool fuse_h_quant = false;
    bool canon = false;               ///< canonical Q2_K/Q3_K arithmetic (canon_expert.hpp): the bits the GPU tier computes
    float gate_skip = 0;              ///< canonical mode: units with |silu(gate)| below this contribute zero (lossy, opt-in)
    float swiglu_limit = 0;           ///< 0 preserves the Qwen contract; GLM uses 10
    /// > 0: MiniMax's "swigluoai" for the gate/up rows (ggml-cpu dots, then g = min(g, swiglu_limit),
    /// u = clamp(u, -swiglu_limit, swiglu_limit), out = g * sigmoid(swiglu_alpha * g) * (1 + u)).  0 (the default)
    /// keeps every other model's silu(g) * u path exactly as it was.
    float swiglu_alpha = 0;
    /// With swiglu_alpha > 0 only: take the gate/up/down rows from the AVX2 multi-token kernels (iq256_rows) instead of
    /// ggml's vec_dot.  The same arithmetic up to float summation order; opt-in.
    bool iq256_rows = false;
    /// IQ1_S (type 19) gate/up and down rows on the AVX-2 multi-token kernel from one token on (iq_avx2.cpp Fmt32<19>)
    /// instead of ggml-cpu's dot, whose BMI2 pdep decode is microcoded on Zen 1/2. Set by the DeepSeek-V4.1 loader;
    /// STRATA_IQ1S_256=1 sets it for every model. Off: a model already running IQ1_S experts keeps ggml's rounding.
    bool iq1s_rows = false;
    /// NVFP4 (ggml type 40) only: the blob ends with {s_gate, s_up, s_down, 0} at this offset, the per-expert
    /// weight_scale_2 the block format has no room for (tools/iq_pack.py). 0 = no tail. Each scale goes on its own
    /// projection's FP32 output: gate * s_gate, up * s_up, down * s_down. Do not fold s_down into up: the hidden
    /// drops to ~1e-5 and its q8 block scale becomes an FP16 subnormal (expert error 2-12% instead of 1.1%).
    size_t tail_off = 0;
    /// prism.hadamard (hadamard.hpp): the experts of this layer are stored in a rotated basis, and
    /// native_quant_act / native_quant_h apply y = H_block (signs * x) before quantizing.  had_block 0 = none.
    int had_block = 0;
    bool had_gu = false, had_d = false;
    int layer = -1;                     ///< the layer this format belongs to (set by expert_layout_load; STRATA_ACT_DUMP only)
    const float* gu_signs = nullptr;    ///< n_embd signs for x (nullptr: identity)
    const float* d_signs = nullptr;     ///< n_ff signs for h (nullptr: identity)
};
constexpr int kNvfp4Type = 40;
constexpr size_t kNvfp4Tail = 16;

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type. kLlvqType for both types:
/// llvq Tetra records (llvq_expert.hpp; llvq_expert_init first), blob = [gate record | up record | down record].
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// Prepare a complete original Q2_K/Q3_K expert for expanded down scales.
/// Throws on incompatible format, wrong input size, or allocation failure.
/// Both output objects remain unchanged on failure; input may alias output_blob.
/// source and prepared may be the same object. No loader/default selection.
void native_prepare_q3_expanded_down(const NativeFmt& source, const uint8_t* input, size_t input_bytes,
                                     NativeFmt& prepared, std::vector<uint8_t>& output_blob);

/// x (n_embd floats) -> the gate/up activation (act_bytes), rotated first when the layer is Hadamard-folded.
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes), rotated first when the layer is Hadamard-folded.
void native_quant_h(const NativeFmt& f, const float* h, void* dst);
void native_quant_h_rows(const NativeFmt& f,const float* h,void* dst,int first,int last);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// For split storage, scale_blob points to the original complete blob holding the NVFP4 tail.
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1, const uint8_t* separate_up = nullptr, const uint8_t* scale_blob = nullptr);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1, const uint8_t* separate_down = nullptr, const uint8_t* scale_blob = nullptr);

/// Same per-token result as the corresponding legacy nt=1 call on this build and CPU.
/// Kernels that already satisfy this contract retain their grouped execution.
void native_gu_rows_invariant(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt,
                              float* const* ff, int r0, int r1);
void native_down_rows_invariant(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt,
                                float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
