#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
namespace strata::kernels::cpu {
// FP4 W4A16 dot products: original row-major nibbles, logical E4M3
// group-16 scales, FP32 outputs. x is FP16-rounded FP32, token-major.
// Rows [first,last) can be evaluated independently by the existing pool.
void glm_fp4_rows(const uint8_t* weights,const uint8_t* scales,float global,
                  const float* x,float* y,int columns,int rows,int tokens,
                  int first,int last,bool qpn_scale);
inline constexpr int glm_fp4_weight_table_floats=128*8+8;
// Eight positive E2M1 magnitudes for every E4M3 scale, rounded to FP16 once.
// The extra eight values permit a full AVX-512 load of the last table row.
void glm_fp4_weight_table(float global,bool qpn_scale,float* table);
void glm_fp4_rows_table(const uint8_t* weights,const uint8_t* scales,const float* table,
                        const float* x,float* y,int columns,int rows,int tokens,int first,int last);
// Original CSF32 nibbles are retained in CPU row order, with sparse exception
// masks, ranks and values. This is a lossless layout change for CPU locality.
struct GlmFp4Csf32Rows {
    int rows=0,groups=0;
    std::vector<uint8_t> bases,codes;
    std::vector<uint64_t> masks;
    std::vector<uint32_t> offsets;
    std::vector<uint8_t> exceptions;
    size_t bytes()const {return bases.capacity()+codes.capacity()+masks.capacity()*8+offsets.capacity()*4+exceptions.capacity();}
};
GlmFp4Csf32Rows glm_fp4_csf32_rows(const uint8_t* logical,const uint8_t* bases,
                                   const uint8_t* codes,int rows,int groups);
void glm_fp4_rows_csf32_table(const uint8_t* weights,const GlmFp4Csf32Rows& scales,
                             const float* table,const float* x,float* y,int columns,
                             int rows,int tokens,int first,int last);
// Lossless row-sorted exceptions avoid a mask and rank for every scale word.
// Group indices fit in one byte for the native expert's 128/256 groups.
struct GlmFp4Csf32SparseRows {
    int rows=0,groups=0;
    std::vector<uint8_t> bases,codes;
    std::vector<uint32_t> offsets;
    std::vector<uint16_t> exceptions;
    std::vector<uint32_t> quartets;
    // Optional immutable checkpoint views; their owner must outlive this layout.
    const uint8_t* borrowed_bases=nullptr;
    const uint8_t* borrowed_codes=nullptr;
    size_t bytes()const {return bases.capacity()+codes.capacity()+offsets.capacity()*4+exceptions.capacity()*2+quartets.capacity()*4;}
};
GlmFp4Csf32SparseRows glm_fp4_csf32_sparse_rows(const uint8_t* logical,const uint8_t* bases,
                                              const uint8_t* codes,int rows,int groups,bool borrow=false);
void glm_fp4_cache_scale_quartets(GlmFp4Csf32SparseRows& scales);
void glm_fp4_rows_csf32_sparse_table(const uint8_t* weights,const GlmFp4Csf32SparseRows& scales,
                                    const float* table,const float* x,float* y,int columns,
                                    int rows,int tokens,int first,int last);
// Optional W4A8 approximation. Original FP4 codes stay compressed; activations
// have one signed-int8 scale per 64 values. Weight factors use table[scale*8+1]
// times twice the E2M1 magnitude, instead of rounding each operand to FP16.
struct GlmFp4Q8 {
    int columns=0,tokens=0;
    std::vector<int8_t> values;
    std::vector<float> scales;
    // Optional reusable VNNI operands; preparation does not change quantization.
    std::vector<uint8_t> magnitudes;
    std::vector<uint64_t> negative;
    // -12 times each four-value signed sum cancels the unsigned weight offset.
    // This changes the VNNI operands, not the quantization or floating-point math.
    std::vector<int16_t> dot_bias;
};
GlmFp4Q8 glm_fp4_quantize_q8(const float* x,int columns,int tokens);
void glm_fp4_prepare_q8(GlmFp4Q8& x);
void glm_fp4_prepare_q8_bias(GlmFp4Q8& x);
GlmFp4Q8 glm_fp4_gather_q8(const GlmFp4Q8& x,const int* rows,int tokens);
// In-place FP16 rounding, with the scalar codec's values on every ISA.
void glm_fp4_round_f16(float* x,size_t count);
bool glm_fp4_q8_vnni_available();
void glm_fp4_rows_csf32_q8(const uint8_t* weights,const GlmFp4Csf32SparseRows& scales,
                          const float* table,const GlmFp4Q8& x,float* y,int columns,
                          int rows,int tokens,int first,int last,int row_tile=1,
                          bool local_scales=false,bool exact_policy=false);
}
