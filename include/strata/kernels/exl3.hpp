#pragma once
#include <cstdint>
namespace strata::kernels {
void exl3_dense_q6_gemv(const void* w,const float* x,float* y,int in,int out,void* stream);
void exl3_dense_q6_to_float(const void* w,float* out,int count,void* stream);
void exl3_dense_q8_gemv(const void* w,const float* x,float* y,int in,int out,void* stream);
void exl3_dense_q8_to_float(const void* w,float* out,int count,void* stream);
void exl3_packed_gemv(const uint8_t* packed,const float* x,float* y,int in,int out,void* stream);
void exl3_bf16_gemv(const uint16_t* w,const float* x,float* y,int in,int out,void* stream);
void exl3_bf16_to_float(const uint16_t* w,float* out,int count,void* stream);
// Packed tiles in output-tile-major order; reconstructed matrix is [out,in].
void exl3_reconstruct(const uint8_t* packed,float* weights,int in,int out,void* stream);
void exl3_hadamard(const float* x,float* y,const uint16_t* scale,int width,int tokens,bool pre,void* stream);
}
