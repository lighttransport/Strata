// QPN2 dataflow adapted from v100-skinny, revision
// 5b589c0dc81223e0ba65bcb3e755874723f8b515, kernels/skinny_kernels.cu.
// Copyright (c) 2026 v100-skinny contributors, MIT.
// The shift/rebias decoder derives from LMDeploy/TurboMind (OpenMMLab,
// Apache-2.0). Notices: third_party/licenses/v100-skinny.txt.
#include "strata/kernels/glm_fp4.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
namespace strata::kernels {
namespace {
__device__ __forceinline__ half2 fp8e4m3_to_half2(unsigned char b) {
    const unsigned short hb=((static_cast<unsigned short>(b)&0x80u)<<8)|((static_cast<unsigned short>(b)&0x7fu)<<7);
    const half h=__hmul(__ushort_as_half(hb),__ushort_as_half(0x5c00));
    return __halves2half2(h,h);
}
__device__ __forceinline__ void dequant8_tm(unsigned q,half2 scale,half2 out[4]) {
    constexpr unsigned S=0x80008000u,EM=0x0e000e00u;
    unsigned bits[4]={((q<<12)&S)|((q<<9)&EM),((q<<8)&S)|((q<<5)&EM),
                      ((q<<4)&S)|((q<<1)&EM),(q&S)|((q>>3)&EM)};
    #pragma unroll
    for(int i=0;i<4;++i)out[i]=__hmul2(*reinterpret_cast<const half2*>(&bits[i]),scale);
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define MMA_8N8K4(C, A0, A1, B0, B1) asm volatile("trap;")
#else
#define MMA_8N8K4(C, A0, A1, B0, B1)                                        \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))
#endif

template <int SPLITK, int NACC, bool NARROW=false>
__global__ void fp4_qpn(const uint8_t *__restrict__ bcodes,
                                  const uint8_t *__restrict__ bscales,
                                  const half *__restrict__ x,
                                  float *__restrict__ y, int N, int K, int M,
                                  float gscale) {
  __shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 256 : 1];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = NARROW ? blockIdx.x / 4 : blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4;
  const int part = NARROW ? warp * 4 + qp : warp;
  const int parts = SPLITK * (NARROW ? 4 : 1);
  const int g0 = part * G / parts, g1 = (part + 1) * G / parts;
  const int weight_lane = NARROW ? (lane & 19) + (blockIdx.x % 4) * 4 : lane;
  const uint2 *cb = reinterpret_cast<const uint2 *>(bcodes) +
                    (size_t)tile * G * 32 + weight_lane;
  const uint8_t *sb = bscales + (size_t)tile * G * 32 + weight_lane;

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[a][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g1; g++) {
    const uint2 q2 = __ldcs(cb + (size_t)g * 32);
    const half2 sc2 =
        __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
    half2 b[8];
    dequant8_tm(q2.x, sc2, b + 0);
    dequant8_tm(q2.y, sc2, b + 4);
    const unsigned *B = reinterpret_cast<const unsigned *>(b);
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (r < M) {
      const half *xrow = x + (size_t)r * K;
      a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
      a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
    }
    const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(c[0], A0[0], A0[1], B[0], B[1]);
    MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
    MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
    MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
  }

#pragma unroll
  for (int a = 1; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[0][i] += c[a][i];

  if (SPLITK == 1 && !NARROW) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (row < M)
        y[(size_t)row * N + (size_t)tile * 32 + qp * 8 + col] =
            c[0][i];
    }
    return;
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    cs[warp][row * 32 + qp * 8 + col] = c[0][i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < (NARROW ? 64 : 256); e += blockDim.x) {
    const int row = NARROW ? e / 8 : e / 32, col = NARROW ? e % 8 : e % 32;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < SPLITK; w++) {
      if constexpr(NARROW) {
#pragma unroll
        for(int q = 0; q < 4; ++q) v += cs[w][row * 32 + q * 8 + col];
      } else v += cs[w][e];
    }
    if (row < M)
      y[(size_t)row * N + (size_t)blockIdx.x * (NARROW ? 8 : 32) + col] = v;
  }
}

__device__ __forceinline__ float e4m3(unsigned b) {
    if((b&127)==127)return __int_as_float(0x7fffffff);
    return __half2float(__ushort_as_half(((b&128)<<8)|((b&127)<<7)))*256.f;
}
__device__ __forceinline__ float e2m1(unsigned q) {
    const unsigned m=q&7;
    const unsigned bits=(q&8)<<12 | (m ? (m==1 ? 0x3800u : (m+28u)<<9) : 0u);
    return __half2float(__ushort_as_half(bits));
}
__device__ __forceinline__ float weight_value(GlmPackedFormat format,const void* weights,const void* scales,
                                             float global,int row,int column,int K) {
    const auto* w=static_cast<const uint8_t*>(weights);
    const auto* s=static_cast<const uint8_t*>(scales);
    const size_t i=size_t(row)*K+column;
    size_t fp8_index=i,fp8_group=size_t(row)*(K/32)+column/32;
    if(format==GlmPackedFormat::Mxfp8Qpn||format==GlmPackedFormat::BlockFp8Qpn) {
        const int r=row&7,qp=(row&31)>>3,lane=(qp<<2)+(r&3)+((r&4)?16:0);
        fp8_group=(size_t(row/32)*(K/16)+column/16)*32+lane;
        fp8_index=fp8_group*16+column%16;
    }
    if(format==GlmPackedFormat::Nvfp4) {
        const int r=row&7,qp=(row&31)>>3,lane=(qp<<2)+(r&3)+((r&4)?16:0);
        const size_t group=(size_t(row/32)*(K/16)+column/16)*32+lane;
        const int k=column&15,p=(k&7)/2+(k&1)*4+(k/8)*8;
        const unsigned code=(w[group*8+p/2]>>((p&1)*4))&15;
        return (e2m1(code)*e4m3(s[group]))*global;
    }
    if(format==GlmPackedFormat::Mxfp8||format==GlmPackedFormat::Mxfp8Qpn) {
        const unsigned exponent=s[fp8_group];
        const float scale=__uint_as_float(exponent?exponent<<23:0x00400000u);
        return e4m3(w[fp8_index])*scale;
    }
    if(format==GlmPackedFormat::BlockFp8||format==GlmPackedFormat::BlockFp8Qpn)
        return e4m3(w[fp8_index])*static_cast<const float*>(scales)[size_t(row/128)*((K+127)/128)+column/128];
    if(format==GlmPackedFormat::Bf16)return __uint_as_float(unsigned(static_cast<const uint16_t*>(weights)[i])<<16);
    if(format==GlmPackedFormat::F16)return __half2float(static_cast<const half*>(weights)[i]);
    return static_cast<const float*>(weights)[i];
}
__global__ void prepack_kernel(const uint8_t* codes,const uint8_t* scales,uint8_t* out,uint8_t* os,int N,int K) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(N)*(K/16))return;
    const int lane=i%32,g=(i/32)%(K/16),tile=i/(32*(K/16));
    const int qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0),row=tile*32+qp*8+r;
    const uint8_t* src=codes+size_t(row)*(K/2)+g*8;
    constexpr int order[16]={0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15};
    uint2 packed=make_uint2(0,0);auto* dst=reinterpret_cast<uint8_t*>(&packed);
    #pragma unroll
    for(int j=0;j<8;++j) {
        const int a=order[2*j],b=order[2*j+1];
        dst[j]=((src[a/2]>>((a&1)*4))&15)|(((src[b/2]>>((b&1)*4))&15)<<4);
    }
    reinterpret_cast<uint2*>(out)[i]=packed;
    if(scales)os[i]=scales[size_t(row)*(K/16)+g];
}
__device__ __forceinline__ size_t packed_scale_index(int row,int col,int C) {
    const int r=row&7,qp=(row&31)>>3,lane=(qp<<2)+(r&3)+((r&4)?16:0);
    return (size_t(row/32)*C+col)*32+lane;
}
__global__ void fp8_prepack_kernel(const uint8_t* codes,const uint8_t* scales,uint8_t* out,uint8_t* os,int N,int K) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(N)*(K/16))return;
    const int lane=i%32,g=(i/32)%(K/16),tile=i/(32*(K/16));
    const int qp=(lane>>2)&3,r=(lane&3)+((lane&16)?4:0),row=tile*32+qp*8+r;
    reinterpret_cast<uint4*>(out)[i]=*reinterpret_cast<const uint4*>(codes+size_t(row)*K+g*16);
    if(scales)os[i]=scales[size_t(row)*(K/32)+g/2];
}
__global__ void csf32_kernel(const uint8_t* bases,const uint8_t* codes,const uint16_t* offsets,
                              const uint32_t* pages,const uint16_t* exceptions,uint8_t* out,int* invalid,int C) {
    const int tile=blockIdx.x,lane=threadIdx.x&31,part=threadIdx.x>>5;
    const int row=(tile/(C/4))*128+(part%4)*32+lane;
    const int col=(tile%(C/4))*4+(part/4)*2;
    const int block=tile*8+part,page=tile/32;
    const uint8_t code=codes[size_t(block)*32+lane];
    out[packed_scale_index(row,col,C)]=uint8_t(bases[row]+(code&15));
    out[packed_scale_index(row,col+1,C)]=uint8_t(bases[row]+(code>>4));
    __syncthreads();
    const size_t first=pages[page]/2+offsets[block+page],last=pages[page]/2+offsets[block+page+1];
    for(size_t i=first+lane;i<last;i+=32) {
        const unsigned value=exceptions[i],slot=value&255;
        out[packed_scale_index(row-lane+slot/2,col+slot%2,C)]=value>>8;
    }
    __syncthreads();
    if(out[packed_scale_index(row,col,C)]>=127 || out[packed_scale_index(row,col+1,C)]>=127)
        atomicExch(invalid,1);
}
__global__ void reference_kernel(GlmPackedFormat format,const void* w,const void* scales,float global,
                                  const half* x,float* y,int N,int K,int M) {
    const int row=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31,t=blockIdx.y;
    if(row>=N||t>=M)return;
    float sum=0;
    for(int k=lane;k<K;k+=32)sum=fmaf(__half2float(__float2half_rn(weight_value(format,w,scales,global,row,k,K))),
                                                 __half2float(x[size_t(t)*K+k]),sum);
    for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(!lane)y[size_t(t)*N+row]=sum;
}
__global__ void dequant_kernel(GlmPackedFormat format,const void* w,const void* scales,float global,half* y,int N,int K) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(N)*K)y[i]=__float2half_rn(weight_value(format,w,scales,global,i/K,i%K,K));
}
// Reconstruct the exact operands used by the skinny SM70 decoder, including
// both FP16 scale roundings. One lane expands sixteen weights, rather than
// re-reading the packed fragment independently for every output value.
__global__ void fp4_sm70_operands(const uint2* codes,const uint8_t* scales,float global,half* out,int N,int K) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(N)*(K/16))return;
    const int lane=i%32,g=(i/32)%(K/16),tile=i/(32*(K/16));
    const int row=tile*32+((lane>>2)&3)*8+(lane&3)+((lane&16)?4:0);
    const uint2 q=codes[i];
    const half2 scale=__hmul2(fp8e4m3_to_half2(scales[i]),__float2half2_rn(global*16384.f));
    alignas(16) half2 values[8];
    dequant8_tm(q.x,scale,values);dequant8_tm(q.y,scale,values+4);
    auto* target=reinterpret_cast<uint4*>(out+size_t(row)*K+g*16);
    target[0]=reinterpret_cast<const uint4*>(values)[0];
    target[1]=reinterpret_cast<const uint4*>(values)[1];
}
__device__ __forceinline__ half2 fp8_pair(unsigned q) {
    unsigned bits=((q&0x80u)<<8)|((q&0x7fu)<<7)|
                  ((q&0x8000u)<<16)|((q&0x7f00u)<<15);
    return *reinterpret_cast<const half2*>(&bits);
}
template<GlmPackedFormat FORMAT>
__device__ __forceinline__ bool fp8_operands(const void* weights,const void* scales,int n,int g,int K,half* out) {
    constexpr bool packed=FORMAT==GlmPackedFormat::Mxfp8Qpn||FORMAT==GlmPackedFormat::BlockFp8Qpn;
    constexpr bool mx=FORMAT==GlmPackedFormat::Mxfp8||FORMAT==GlmPackedFormat::Mxfp8Qpn;
    const size_t group=packed?packed_scale_index(n,g,K/16):size_t(n)*(K/32)+g/2;
    const size_t offset=packed?group*16:size_t(n)*K+g*16;
    const uint4 words=*reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(weights)+offset);
    const unsigned* q=reinterpret_cast<const unsigned*>(&words);
    unsigned nan=0;
    #pragma unroll
    for(int i=0;i<4;++i)nan|=((q[i]&0x7f7f7f7fu)+0x01010101u)&0x80808080u;
    if(nan)return false;
    half2 scale;
    float factor=0;
    if constexpr(mx) {
        const unsigned e=static_cast<const uint8_t*>(scales)[group];
        // E4M3's shift representation is multiplied by 256 * 2^(e-127).
        // This power of two is exact in FP16 throughout this normal range.
        // Extreme E8M0 scales retain the scalar FP32 reconstruction below.
        if(e<105||e>134)return false;
        scale=__half2half2(__ushort_as_half((e-104u)<<10));
    }else {
        const float s=static_cast<const float*>(scales)[size_t(n/128)*((K+127)/128)+g/8];
        if(!(s>=0.f&&s<=0x1.fffffep119f))return false;
        factor=s*256.f;
    }
    auto* pairs=reinterpret_cast<half2*>(out);
    #pragma unroll
    for(int i=0;i<4;++i) {
        #pragma unroll
        for(int j=0;j<2;++j) {
            const half2 raw=fp8_pair(q[i]>>(j*16));
            if constexpr(mx)pairs[i*2+j]=__hmul2(raw,scale);
            else {
                const float2 v=__half22float2(raw);
                pairs[i*2+j]=__floats2half2_rn(v.x*factor,v.y*factor);
            }
        }
    }
    return true;
}
// Same quadpair geometry for MXFP8, block FP8 and BF16. Group scaling is
// applied before each MMA operand conversion, rather than as a single matrix
// epilogue scale: MXFP8 varies along K and block FP8 varies along both axes.
// SPLITK is fixed across M=1..8 so verification does not change the K reduction.
template<GlmPackedFormat FORMAT,int SPLITK,int OUTER=1>
__global__ void scaled_qpn(const void* w,const void* scales,
                            const half* x,float* y,int N,int K,int M) {
    __shared__ float sums[SPLITK][256];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,qp=(lane>>2)&3;
    const int r=(lane&3)+((lane&16)?4:0),n=blockIdx.x*32+qp*8+r;
    const int groups=K/16,part=blockIdx.y*SPLITK+warp;
    const int first=groups*part/(SPLITK*OUTER),last=groups*(part+1)/(SPLITK*OUTER);
    float c[8]={};
    for(int g=first;g<last;++g) {
        alignas(16) half b[16];
        if constexpr(FORMAT==GlmPackedFormat::F16) {
            uint4 b0=make_uint4(0,0,0,0),b1=b0;
            if(n<N) {
                const half* row=static_cast<const half*>(w)+size_t(n)*K+g*16;
                b0=*reinterpret_cast<const uint4*>(row);b1=*reinterpret_cast<const uint4*>(row+8);
            }
            *reinterpret_cast<uint4*>(b)=b0;*reinterpret_cast<uint4*>(b+8)=b1;
        }else if constexpr(FORMAT==GlmPackedFormat::Mxfp8||FORMAT==GlmPackedFormat::BlockFp8||
                           FORMAT==GlmPackedFormat::Mxfp8Qpn||FORMAT==GlmPackedFormat::BlockFp8Qpn) {
            if(n>=N) {
                #pragma unroll
                for(int j=0;j<16;++j)b[j]=__float2half_rn(0.f);
            }else if(!fp8_operands<FORMAT>(w,scales,n,g,K,b)) {
                #pragma unroll
                for(int j=0;j<16;++j)b[j]=__float2half_rn(weight_value(FORMAT,w,scales,1.f,n,g*16+j,K));
            }
        }else {
            #pragma unroll
            for(int j=0;j<16;++j)b[j]=n<N?__float2half_rn(weight_value(FORMAT,w,scales,1.f,n,g*16+j,K)):__float2half_rn(0.f);
        }
        const unsigned* B=reinterpret_cast<const unsigned*>(b);
        uint4 a0=make_uint4(0,0,0,0),a1=a0;
        if(r<M) {a0=*reinterpret_cast<const uint4*>(x+size_t(r)*K+g*16);a1=*reinterpret_cast<const uint4*>(x+size_t(r)*K+g*16+8);}
        const unsigned* A0=reinterpret_cast<const unsigned*>(&a0);
        const unsigned* A1=reinterpret_cast<const unsigned*>(&a1);
        MMA_8N8K4(c,A0[0],A0[1],B[0],B[1]);
        MMA_8N8K4(c,A0[2],A0[3],B[2],B[3]);
        MMA_8N8K4(c,A1[0],A1[1],B[4],B[5]);
        MMA_8N8K4(c,A1[2],A1[3],B[6],B[7]);
    }
    #pragma unroll
    for(int i=0;i<8;++i) {
        const int row=(i&2)|((lane&16)?4:0)|(lane&1);
        const int col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2);
        sums[warp][row*32+qp*8+col]=c[i];
    }
    __syncthreads();
    for(int e=threadIdx.x;e<256;e+=blockDim.x) {
        float sum=0;
        #pragma unroll
        for(int i=0;i<SPLITK;++i)sum+=sums[i][e];
        if((e>>5)<M && blockIdx.x*32+(e&31)<N)
            y[size_t(blockIdx.y)*8*N+size_t(e>>5)*N+blockIdx.x*32+(e&31)]=sum;
    }
}
__global__ void reduce_partials(const float* partials,float* y,int N,int M,int parts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N*M)return;
    float sum=0;
    for(int part=0;part<parts;++part)sum+=partials[size_t(part)*8*N+i];
    y[i]=sum;
}
template<GlmPackedFormat FORMAT,int PARTS>
void small_gemm(const void* w,const void* scales,const half* x,float* y,float* partials,int N,int K,int M,cudaStream_t stream) {
    scaled_qpn<FORMAT,8,PARTS><<<dim3((N+31)/32,PARTS),256,0,stream>>>(w,scales,x,partials,N,K,M);
    reduce_partials<<<(N*M+255)/256,256,0,stream>>>(partials,y,N,M,PARTS);
}
template<GlmPackedFormat FORMAT>
void scaled_gemm(const void* w,const void* scales,const half* x,float* y,float* partials,int N,int K,int M,cudaStream_t stream) {
    if(partials && N<=8192 && K>=4096) {
        if(N<=32)small_gemm<FORMAT,64>(w,scales,x,y,partials,N,K,M,stream);
        else if(N<=64)small_gemm<FORMAT,32>(w,scales,x,y,partials,N,K,M,stream);
        else if(N<=128)small_gemm<FORMAT,16>(w,scales,x,y,partials,N,K,M,stream);
        else if(N<=1024)small_gemm<FORMAT,8>(w,scales,x,y,partials,N,K,M,stream);
        else if(N<=2048)small_gemm<FORMAT,4>(w,scales,x,y,partials,N,K,M,stream);
        else small_gemm<FORMAT,2>(w,scales,x,y,partials,N,K,M,stream);
    }else scaled_qpn<FORMAT,8><<<(N+31)/32,256,0,stream>>>(w,scales,x,y,N,K,M);
}
__global__ void route_gather(const float* x,const int* routes,half* out,int count,int K,int stride) {
    const int column=blockIdx.x*blockDim.x+threadIdx.x;
    if(column>=K)return;
    for(int row=blockIdx.y;row<count;row+=gridDim.y)
        out[size_t(row)*K+column]=__float2half_rn(x[size_t(routes[row]/stride)*K+column]);
}
__global__ void route_scatter(const float* x,const int* routes,const float* weights,float* sum,int count,int K,int stride) {
    const int column=blockIdx.x*blockDim.x+threadIdx.x;
    if(column>=K)return;
    for(int row=blockIdx.y;row<count;row+=gridDim.y) {
        const int route=routes[row];const size_t target=size_t(route/stride)*K+column;
        const float value=__fmul_rn(weights[route],x[size_t(row)*K+column]);
        sum[target]=__fadd_rn(sum[target],value);
    }
}
void valid(int N,int K,int M) {
    if(N<1||K<1||M<1||K%64)throw std::invalid_argument("GLM FP4: expected K%64=0 and positive N/M");
}
} // namespace
void glm_fp4_prepack(const uint8_t* w,const uint8_t* s,uint8_t* out,uint8_t* os,int N,int K,void* stream) {
    valid(N,K,1);if(N%32)throw std::invalid_argument("FP4 prepack requires N%32=0");const size_t count=size_t(N)*(K/16);
    prepack_kernel<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(w,s,out,os,N,K);
}
void glm_fp8_prepack(GlmPackedFormat format,const uint8_t* w,const void* scales,uint8_t* out,void* os,int N,int K,void* stream) {
    valid(N,K,1);
    if(N%32||(format!=GlmPackedFormat::Mxfp8&&format!=GlmPackedFormat::BlockFp8))
        throw std::invalid_argument("FP8 prepack requires N%32=0 and a native FP8 format");
    auto s=static_cast<cudaStream_t>(stream);const size_t count=size_t(N)*(K/16);
    const bool mx=format==GlmPackedFormat::Mxfp8;
    fp8_prepack_kernel<<<(count+255)/256,256,0,s>>>(w,mx?static_cast<const uint8_t*>(scales):nullptr,out,static_cast<uint8_t*>(os),N,K);
    if(!mx)cudaMemcpyAsync(os,scales,size_t((N+127)/128)*((K+127)/128)*4,cudaMemcpyDeviceToDevice,s);
}
void glm_fp4_csf32_scales(const uint8_t* bases,const uint8_t* codes,const uint16_t* offsets,
                          const uint32_t* pages,const uint16_t* exceptions,uint8_t* out,int* invalid,
                          int N,int K,void* stream) {
    valid(N,K,1);
    if(N%128)throw std::invalid_argument("CSF32 GPU scales require N%128=0");
    auto s=static_cast<cudaStream_t>(stream);
    cudaMemsetAsync(invalid,0,sizeof(int),s);
    csf32_kernel<<<(N/128)*(K/64),256,0,s>>>(bases,codes,offsets,pages,exceptions,out,invalid,K/16);
}
void glm_fp4_gemm(GlmPackedFormat format,const void* w,const void* scales,float global,
                  const uint16_t* x,float* y,int N,int K,int M,bool sm70,void* stream,float* partials) {
    valid(N,K,M);auto s=static_cast<cudaStream_t>(stream);const half* h=reinterpret_cast<const half*>(x);
    if(!sm70) {reference_kernel<<<dim3((N+7)/8,M),256,0,s>>>(format,w,scales,global,h,y,N,K,M);return;}
    if(M>8)throw std::invalid_argument("GLM FP4: skinny MMA supports M<=8");
    if(format==GlmPackedFormat::Nvfp4) {
        if(N%32)throw std::invalid_argument("FP4 MMA requires N%32=0");
        // Eight independent K partitions keep grouped narrow MoE shapes busy.
        // Same partition/reduction at every verify width.
        // Quadpairs split K within a warp here, so every quad must have the
        // same trip count at mma.sync. Other widths use warp-only splitting.
        if(N<=2048 && K%512==0)fp4_qpn<8,2,true><<<N/8,256,0,s>>>(static_cast<const uint8_t*>(w),static_cast<const uint8_t*>(scales),h,y,N,K,M,global);
        else fp4_qpn<8,2><<<N/32,256,0,s>>>(static_cast<const uint8_t*>(w),static_cast<const uint8_t*>(scales),h,y,N,K,M,global);
    }else {
        switch(format) {
        case GlmPackedFormat::Mxfp8:scaled_gemm<GlmPackedFormat::Mxfp8>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::BlockFp8:scaled_gemm<GlmPackedFormat::BlockFp8>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::Bf16:scaled_gemm<GlmPackedFormat::Bf16>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::F32:scaled_gemm<GlmPackedFormat::F32>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::F16:scaled_gemm<GlmPackedFormat::F16>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::Mxfp8Qpn:scaled_gemm<GlmPackedFormat::Mxfp8Qpn>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::BlockFp8Qpn:scaled_gemm<GlmPackedFormat::BlockFp8Qpn>(w,scales,h,y,partials,N,K,M,s);break;
        default:throw std::invalid_argument("GLM FP4: unknown packed format");
        }
    }
}
void glm_fp4_dequant_f16(GlmPackedFormat format,const void* w,const void* s,float global,
                         uint16_t* out,int N,int K,void* stream,bool sm70) {
    valid(N,K,1);const size_t count=size_t(N)*K;
    if(sm70&&format==GlmPackedFormat::Nvfp4) {
        if(N%32)throw std::invalid_argument("SM70 FP4 operands require N%32=0");
        fp4_sm70_operands<<<(count/16+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint2*>(w),static_cast<const uint8_t*>(s),global,reinterpret_cast<half*>(out),N,K);
        return;
    }
    if(format==GlmPackedFormat::F16) {
        if(w!=out)cudaMemcpyAsync(out,w,count*2,cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(stream));
        return;
    }
    dequant_kernel<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(format,w,s,global,reinterpret_cast<half*>(out),N,K);
}
void glm_fp4_route_gather(const float* x,const int* routes,uint16_t* out,int count,int K,int stride,void* stream) {
    if(count<0 || K<1 || stride<1)throw std::invalid_argument("GLM FP4: invalid route gather extent");
    if(count)route_gather<<<dim3((K+255)/256,std::min(count,65535)),256,0,static_cast<cudaStream_t>(stream)>>>(x,routes,reinterpret_cast<half*>(out),count,K,stride);
}
void glm_fp4_route_scatter(const float* x,const int* routes,const float* weights,float* sum,int count,int K,int stride,void* stream) {
    if(count<0 || K<1 || stride<1)throw std::invalid_argument("GLM FP4: invalid route scatter extent");
    if(count)route_scatter<<<dim3((K+255)/256,std::min(count,65535)),256,0,static_cast<cudaStream_t>(stream)>>>(x,routes,weights,sum,count,K,stride);
}
} // namespace strata::kernels
