#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
// Native formats and cached FP16 operands, independent of GGML's type IDs.
enum class GlmPackedFormat : int { Nvfp4=0, Mxfp8=1, BlockFp8=2, Bf16=3, F32=4, F16=5, Mxfp8Qpn=6, BlockFp8Qpn=7, F16Qpn=8 };
// FP4 codes/scales become [N/32][K/16][lane]. Eight bytes per code
// fragment, one E4M3 scale per lane.
void glm_fp4_prepack(const uint8_t* codes,const uint8_t* scales,uint8_t* packed_codes,
                    uint8_t* packed_scales,int rows,int columns,void* stream);
// FP8 uses the same lane layout, sixteen bytes per fragment. MXFP8 scales
// expand from one per 32 weights to one per 16; block FP8 scale grids stay
// unchanged. Requires rows%32=0. Original FP8 codes are only permuted.
void glm_fp8_prepack(GlmPackedFormat format,const uint8_t* codes,const void* scales,
                    uint8_t* packed_codes,void* packed_scales,int rows,int columns,void* stream);
// Metadata must pass strata_fp4_csf32_validate before upload. Expands scales
// directly to the prepacked layout; invalid is set for negative/NaN scales.
void glm_fp4_csf32_scales(const uint8_t* bases,const uint8_t* codes,
                          const uint16_t* offsets,const uint32_t* pages,const uint16_t* exceptions,
                          uint8_t* packed_scales,int* invalid,int rows,int columns,void* stream);
// Inputs are FP16, outputs FP32. Both paths consume the same packed bytes.
// SM70 supports 1..8 rows per call; reference supports arbitrary rows.
// FP8 may use 1 (default), 2 or 4 independent accumulator chains. Extra
// chains preserve K partitioning but change the floating-point reduction tree.
void glm_fp4_gemm(GlmPackedFormat format,const void* weights,const void* scales,float global,
                  const uint16_t* x,float* y,int rows,int columns,int tokens,
                  bool sm70,void* stream,float* partials=nullptr,int fp8_accumulators=1);
// Resident narrow expert products. Descriptors are device memory, grouped in
// original expert order. Each count is 1..8; offsets are packed row offsets,
// offset+count<=64. All weights are NVFP4 2048x4096 gate/up, 4096x2048 down.
struct alignas(16) GlmFp4ExpertGroup {
    const uint8_t* weights[3];
    const uint8_t* scales[3];
    float global[3];
    int count,offset;
};
static_assert(sizeof(GlmFp4ExpertGroup)==80);
static_assert(offsetof(GlmFp4ExpertGroup,count)==60);
static_assert(offsetof(GlmFp4ExpertGroup,offset)==64);
void glm_fp4_expert_group_products(const GlmFp4ExpertGroup* groups,int count,bool down,
                                   const uint16_t* input,float* output,float* up,void* stream);
// One writer per token/column, accumulating groups in their original order.
// Preserves separate FP32 multiply/add rounding of per-expert route scatter.
// Same descriptor contract as grouped products; tokens1..8, disjoint buffers.
void glm_fp4_expert_group_scatter(const GlmFp4ExpertGroup* groups,int count,
                                 const int* routes,const float* weights,const float* values,
                                 float* sum,int tokens,void* stream);
struct GlmFp8Projection {const void* weights;const void* scales;float* output;};
// Three independent resident MXFP8 QPN products in one SM70 launch. Uses the
// ordinary single-accumulator, eight-warp reduction without outer K splitting.
// Requires rows%32=0, columns%64=0, tokens1..8; outputs must not alias.
void glm_mxfp8_gemm_three(GlmFp8Projection a,GlmFp8Projection b,GlmFp8Projection c,
                         const uint16_t* x,int rows,int columns,int tokens,void* stream);
// Opt-in SM70 NVFP4 gate/up + SwiGLU + FP16 output. Same two accumulator
// chains and K reduction as gemm; requires rows%32=0, columns%64=0, tokens1..8.
void glm_fp4_gate_up_f16(const uint8_t* gate,const uint8_t* gate_scales,float gate_global,
                         const uint8_t* up,const uint8_t* up_scales,float up_global,
                         const uint16_t* x,uint16_t* hidden,int rows,int columns,int tokens,
                         float limit,void* stream);
// Optional scratch for small fixed projections: 64 K partitions, eight rows.
// Passing it keeps K splitting fixed across all verification widths.
constexpr size_t glm_fp4_partial_bytes(int rows) {return size_t(64)*8*rows*sizeof(float);}
// Bounded prefill reconstruction from the same resident compressed matrix.
void glm_fp4_dequant_f16(GlmPackedFormat format,const void* weights,const void* scales,
                         float global,uint16_t* output,int rows,int columns,void* stream,bool sm70=false);
// Reconstruct exact SM70 FP16 operands directly from canonical FP4 codes.
// Scales may be row-major or the existing packed scale plane. N%32=0, K%64=0.
void glm_fp4_raw_dequant_f16(const uint8_t* weights,const uint8_t* scales,
                            float global,uint16_t* output,int rows,int columns,
                            bool packed_scales,void* stream);
// Fuse CSF32 scale decoding and canonical-code reconstruction into FP16.
// Validated CSF32 metadata, N%128=0 and K%64=0; invalid is reset by this call.
void glm_fp4_csf32_dequant_f16(const uint8_t* weights,const uint8_t* bases,
                              const uint8_t* codes,const uint16_t* offsets,
                              const uint32_t* pages,const uint16_t* exceptions,
                              float global,uint16_t* output,int* invalid,
                              int rows,int columns,void* stream);
// SM70 FP4 prefill: reconstruct tiles in shared memory, then accumulate with
// FP16 WMMA / FP32 accumulators. Same rounded operands as dequant_f16(sm70).
// Requires rows%32=0, columns%64=0, tokens>8, tile_tokens in {32,64}.
void glm_fp4_prefill_mma(const uint8_t* weights,const uint8_t* scales,float global,
                        const uint16_t* x,float* y,int rows,int columns,int tokens,
                        int tile_tokens,void* stream);
// Experimental 128-token / 32-output tile with independent K partitions.
// Not used by the executor. Requires rows%32=0, columns%64=0,
// rows/columns<=8192, 8<tokens<=512, and k_parts in {1,2,4}.
// For k_parts>1, scratch holds k_parts*rows*tokens floats.
void glm_fp4_prefill_split_mma(const uint8_t* weights,const uint8_t* scales,float global,
                              const uint16_t* x,float* y,int rows,int columns,int tokens,
                              int k_parts,float* scratch,void* stream);
// Cache the exact FP16 operands used by the FP8 skinny kernel, in its lane
// layout. Supports native and packed MXFP8/block FP8; requires rows%32=0.
// Output holds rows*columns half values and needs no scale plane.
void glm_fp8_cache_f16_qpn(GlmPackedFormat format,const void* weights,const void* scales,
                            uint16_t* output,int rows,int columns,void* stream);
// Permute row-major FP16 weights to the same lane layout without changing values.
void glm_f16_prepack_qpn(const uint16_t* weights,uint16_t* output,
                         int rows,int columns,void* stream);
// Route indices address [token][route_stride]. Rows within one expert must
// refer to distinct tokens. Scatter calls on one stream preserve expert order
// and round multiply and add separately, matching the host accumulation.
void glm_fp4_route_gather(const float* x,const int* routes,uint16_t* output,
                           int count,int columns,int route_stride,void* stream);
// The same rows from activations already rounded to FP16 with glm_f16.
void glm_fp4_route_gather_half(const uint16_t* x,const int* routes,uint16_t* output,
                                int count,int columns,int route_stride,void* stream);
void glm_fp4_route_scatter(const float* x,const int* routes,const float* weights,float* sum,
                            int count,int columns,int route_stride,void* stream);
}
