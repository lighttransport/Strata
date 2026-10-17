#pragma once
#include <cstdint>
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
}
