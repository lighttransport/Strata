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
}
