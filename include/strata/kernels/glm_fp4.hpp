#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
// Native formats and cached FP16 operands, independent of GGML's type IDs.
enum class GlmPackedFormat : int { Nvfp4=0, Mxfp8=1, BlockFp8=2, Bf16=3, F32=4, F16=5, Mxfp8Qpn=6, BlockFp8Qpn=7 };
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
void glm_fp4_gemm(GlmPackedFormat format,const void* weights,const void* scales,float global,
                  const uint16_t* x,float* y,int rows,int columns,int tokens,
                  bool sm70,void* stream,float* partials=nullptr);
// Optional scratch for small fixed projections: 64 K partitions, eight rows.
// Passing it keeps K splitting fixed across all verification widths.
constexpr size_t glm_fp4_partial_bytes(int rows) {return size_t(64)*8*rows*sizeof(float);}
// Bounded prefill reconstruction from the same resident compressed matrix.
void glm_fp4_dequant_f16(GlmPackedFormat format,const void* weights,const void* scales,
                         float global,uint16_t* output,int rows,int columns,void* stream,bool sm70=false);
// Route indices address [token][route_stride]. Rows within one expert must
// refer to distinct tokens. Scatter calls on one stream preserve expert order
// and round multiply and add separately, matching the host accumulation.
void glm_fp4_route_gather(const float* x,const int* routes,uint16_t* output,
                           int count,int columns,int route_stride,void* stream);
void glm_fp4_route_scatter(const float* x,const int* routes,const float* weights,float* sum,
                            int count,int columns,int route_stride,void* stream);
}
