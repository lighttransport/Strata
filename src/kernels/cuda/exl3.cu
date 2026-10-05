// EXL3 format equations: Copyright (c) 2025 Turboderp, MIT (third_party/exl3/LICENSE).
#include "strata/kernels/exl3.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace strata::kernels {
__device__ float dense_q6_value(const unsigned char* b,unsigned lane) {
    unsigned byte=lane*6/8,shift=lane*6%8,value=b[2+byte];
    if(shift>2)value|=unsigned(b[3+byte])<<8;
    int q=(value>>shift)&63;if(q&32)q-=64;
    return __half2float(*reinterpret_cast<const half*>(b))*q;
}
__global__ void dense_q6_convert(const unsigned char* w,float* y,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=dense_q6_value(w+size_t(i/32)*26,i%32);
}
__global__ void dense_q6_mv(const unsigned char* w,const float* x,float* y,int in,int out) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=out)return;float sum=0;
    for(int b=0;b<in/32;++b)sum+=dense_q6_value(w+(size_t(row)*(in/32)+b)*26,lane)*x[b*32+lane];
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);if(!lane)y[row]=sum;
}
void exl3_dense_q6_gemv(const void* w,const float* x,float* y,int in,int out,void* stream) {
    dense_q6_mv<<<(out+7)/8,256,0,(cudaStream_t)stream>>>((const unsigned char*)w,x,y,in,out);
}
void exl3_dense_q6_to_float(const void* w,float* out,int count,void* stream) {
    dense_q6_convert<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((const unsigned char*)w,out,count);
}
// 34-byte blocks match cpu::exl3::DenseQ8Block. FP32 activations are unquantized.
__global__ void dense_q8_convert(const unsigned char* w,float* y,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const unsigned char* b=w+size_t(i/32)*34;
    y[i]=__half2float(*reinterpret_cast<const half*>(b))*reinterpret_cast<const signed char*>(b+2)[i%32];
}
__global__ void dense_q8_mv(const unsigned char* w,const float* x,float* y,int in,int out) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=out)return;
    float sum=0;
    for(int b=0;b<in/32;++b) {
        const unsigned char* block=w+(size_t(row)*(in/32)+b)*34;
        float value=__half2float(*reinterpret_cast<const half*>(block))*reinterpret_cast<const signed char*>(block+2)[lane];
        sum+=value*x[b*32+lane];
    }
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(!lane)y[row]=sum;
}
void exl3_dense_q8_gemv(const void* w,const float* x,float* y,int in,int out,void* stream) {
    dense_q8_mv<<<(out+7)/8,256,0,(cudaStream_t)stream>>>((const unsigned char*)w,x,y,in,out);
}
void exl3_dense_q8_to_float(const void* w,float* out,int count,void* stream) {
    dense_q8_convert<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((const unsigned char*)w,out,count);
}
__global__ void bf16_convert(const uint16_t* w,float* y,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=__uint_as_float(unsigned(w[i])<<16);
}
__global__ void bf16_mv(const uint16_t* w,const float* x,float* y,int in,int out) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=out)return;
    float sum=0;for(int i=lane;i<in;i+=32)sum+=__uint_as_float(unsigned(w[row*in+i])<<16)*x[i];
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);if(!lane)y[row]=sum;
}
void exl3_bf16_gemv(const uint16_t* w,const float* x,float* y,int in,int out,void* stream) {
    bf16_mv<<<(out+7)/8,256,0,(cudaStream_t)stream>>>(w,x,y,in,out);
}
void exl3_bf16_to_float(const uint16_t* w,float* out,int count,void* stream) {
    bf16_convert<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(w,out,count);
}
__device__ unsigned offset(int r,int c) {
    // Equivalent to the published 16x16 MMA fragment permutation.
    return ((r&1)) + ((r&6)<<2) + ((r&8)>>2) + ((c&7)<<5) + ((c&8)>>1);
}
// Decode packed MCG weights directly in GEMV; no expanded matrix cache.
__global__ void packed_mv(const uint32_t* packed,const float* x,float* y,int in,int out) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;if(row>=out)return;
    int ot=row/16,c=row%16;float sum=0;
    for(int i=lane;i<in;i+=32) {
        const uint32_t* p=packed+(size_t(ot)*(in/16)+i/16)*16;
        unsigned end=(offset(i%16,c)+257)*2,start=end-16,shift=31-((end-1)&31);
        uint16_t state=uint16_t(((uint64_t(p[(start/32)%16])<<32)|p[((end-1)/32)%16])>>shift);
        unsigned bits=(unsigned(state)*0xcbac1fedu & 0x8fff8fffu)^0x3b603b60u;
        float weight=__half2float(__hadd(__ushort_as_half(bits&65535),__ushort_as_half(bits>>16)));
        sum+=weight*x[i];
    }
    for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(!lane)y[row]=sum;
}
void exl3_packed_gemv(const uint8_t* packed,const float* x,float* y,int in,int out,void* stream) {
    packed_mv<<<(out+7)/8,256,0,(cudaStream_t)stream>>>((const uint32_t*)packed,x,y,in,out);
}
__global__ void reconstruct(const uint32_t* p,float* w,int in,int out) {
    int tile=blockIdx.x,it=tile%(in/16),ot=tile/(in/16),r=threadIdx.x/16,c=threadIdx.x%16;
    p+=tile*16;unsigned end=(offset(r,c)+257)*2,start=end-16,shift=31-((end-1)&31);
    uint16_t state=uint16_t(((uint64_t(p[(start/32)%16])<<32)|p[((end-1)/32)%16])>>shift);
    unsigned bits=(unsigned(state)*0xcbac1fedu & 0x8fff8fffu)^0x3b603b60u;
    half a=__ushort_as_half(bits&65535),b=__ushort_as_half(bits>>16);
    w[(ot*16+c)*in+it*16+r]=__half2float(__hadd(a,b));
}
__global__ void had(const float* x,float* y,const half* scale,int width,bool pre) {
    __shared__ float values[128];int i=blockIdx.x*128+threadIdx.x,j=i%width;
    values[threadIdx.x]=x[i]*(pre?__half2float(scale[j]):1.f);__syncthreads();
    for(int bit=1;bit<128;bit*=2) {
        float a=values[threadIdx.x],b=values[threadIdx.x^bit];__syncthreads();
        values[threadIdx.x]=(threadIdx.x&bit)?b-a:a+b;__syncthreads();
    }
    y[i]=values[threadIdx.x]*0.088388347648f*(pre?1.f:__half2float(scale[j]));
}
void exl3_reconstruct(const uint8_t* p,float* w,int in,int out,void* s) {
    reconstruct<<<(in/16)*(out/16),256,0,(cudaStream_t)s>>>((const uint32_t*)p,w,in,out);
}
void exl3_hadamard(const float* x,float* y,const uint16_t* scale,int width,int tokens,bool pre,void* s) {
    had<<<width/128*tokens,128,0,(cudaStream_t)s>>>(x,y,(const half*)scale,width,pre);
}
}
