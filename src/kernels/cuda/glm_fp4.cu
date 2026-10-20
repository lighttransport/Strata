// QPN2 dataflow adapted from v100-skinny, revision
// 5b589c0dc81223e0ba65bcb3e755874723f8b515, kernels/skinny_kernels.cu.
// Copyright (c) 2026 v100-skinny contributors, MIT.
// The shift/rebias decoder derives from LMDeploy/TurboMind (OpenMMLab,
// Apache-2.0). Notices: third_party/licenses/v100-skinny.txt.
#include "strata/kernels/glm_fp4.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
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

// Only batch/grid and operand selection differ from fp4_qpn. Every expert's
// MMA chains, K partitions and final FP32 reduction retain their original order.
template <int SPLITK, int NACC, bool NARROW, bool DOWN>
__global__ void fp4_qpn_grouped(const GlmFp4ExpertGroup* groups,
                                const half* input,float* output,float* up,int N,int K) {
  const auto* group=groups+blockIdx.y/(DOWN?1:2);
  const int projection=DOWN?2:int(blockIdx.y%2);
  const int M=group->count;
  const uint8_t* bcodes=group->weights[projection];
  const uint8_t* bscales=group->scales[projection];
  const float gscale=group->global[projection];
  const half* x=input+size_t(group->offset)*K;
  float* y=(projection==1?up:output)+size_t(group->offset)*N;
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

// Keep each projection's MMA chains and shared-memory reduction identical to
// fp4_qpn<8,2>. Only activation loads and the final pointwise epilogue are shared.
template<bool NARROW>
__global__ void fp4_gate_up(const uint8_t* gate,const uint8_t* gate_scales,float gate_global,
                            const uint8_t* up,const uint8_t* up_scales,float up_global,
                            const half* x,half* hidden,int N,int K,int M,float limit) {
    __shared__ float sums[2][8][256];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int tile=NARROW?blockIdx.x/4:blockIdx.x,qp=(lane>>2)&3;
    const int r=(lane&3)+((lane&16)?4:0),G=K/16;
    const int part=NARROW?warp*4+qp:warp,parts=NARROW?32:8;
    const int weight_lane=NARROW?(lane&19)+(blockIdx.x%4)*4:lane;
    const size_t offset=size_t(tile)*G*32+weight_lane;
    const uint2* codes[2]={reinterpret_cast<const uint2*>(gate)+offset,reinterpret_cast<const uint2*>(up)+offset};
    const uint8_t* scales[2]={gate_scales+offset,up_scales+offset};
    const half2 globals[2]={__float2half2_rn(gate_global*16384.f),__float2half2_rn(up_global*16384.f)};
    float c[2][2][8]={};
    #pragma unroll 4
    for(int g=part*G/parts;g<(part+1)*G/parts;++g) {
        uint4 a01=make_uint4(0,0,0,0),a23=a01;
        if(r<M) {
            a01=*reinterpret_cast<const uint4*>(x+size_t(r)*K+g*16);
            a23=*reinterpret_cast<const uint4*>(x+size_t(r)*K+g*16+8);
        }
        const unsigned* A0=reinterpret_cast<const unsigned*>(&a01);
        const unsigned* A1=reinterpret_cast<const unsigned*>(&a23);
        #pragma unroll
        for(int p=0;p<2;++p) {
            const uint2 q=__ldcs(codes[p]+size_t(g)*32);
            const half2 scale=__hmul2(fp8e4m3_to_half2(__ldg(scales[p]+size_t(g)*32)),globals[p]);
            half2 b[8];dequant8_tm(q.x,scale,b);dequant8_tm(q.y,scale,b+4);
            const unsigned* B=reinterpret_cast<const unsigned*>(b);
            MMA_8N8K4(c[p][0],A0[0],A0[1],B[0],B[1]);
            MMA_8N8K4(c[p][1],A0[2],A0[3],B[2],B[3]);
            MMA_8N8K4(c[p][0],A1[0],A1[1],B[4],B[5]);
            MMA_8N8K4(c[p][1],A1[2],A1[3],B[6],B[7]);
        }
    }
    #pragma unroll
    for(int p=0;p<2;++p) {
        #pragma unroll
        for(int i=0;i<8;++i) {
            c[p][0][i]+=c[p][1][i];
            const int row=(i&2)|((lane&16)?4:0)|(lane&1);
            const int col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2);
            sums[p][warp][row*32+qp*8+col]=c[p][0][i];
        }
    }
    __syncthreads();
    for(int e=threadIdx.x;e<(NARROW?64:256);e+=blockDim.x) {
        const int row=NARROW?e/8:e/32,col=NARROW?e%8:e%32;
        float result[2]={};
        #pragma unroll
        for(int p=0;p<2;++p) {
            #pragma unroll
            for(int w=0;w<8;++w) {
                if constexpr(NARROW) {
                    #pragma unroll
                    for(int q=0;q<4;++q)result[p]+=sums[p][w][row*32+q*8+col];
                }else result[p]+=sums[p][w][e];
            }
        }
        if(row<M) {
            const float a=limit>0?fminf(result[0],limit):result[0];
            const float b=limit>0?fminf(limit,fmaxf(-limit,result[1])):result[1];
            hidden[size_t(row)*N+size_t(blockIdx.x)*(NARROW?8:32)+col]=
                __float2half_rn(a*(1.f/(1.f+expf(-a)))*b);
        }
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
    if(format==GlmPackedFormat::Mxfp8Qpn||format==GlmPackedFormat::BlockFp8Qpn||format==GlmPackedFormat::F16Qpn) {
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
    if(format==GlmPackedFormat::F16Qpn)return __half2float(static_cast<const half*>(weights)[fp8_index]);
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
// Canonical adjacent nibbles become adjacent half lanes. Preserve both
// FP16 scale roundings and the final half multiply used by fp4_sm70_operands.
// Every thread writes one contiguous 32-byte row span; no packed-code copy.
template <bool PACKED_SCALES>
__global__ void fp4_raw_operands(const uint2* codes, const uint8_t* scales, float global,
                               half* out, int N, int K) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * (K / 16))
        return;
    const int row = i / (K / 16), group = i % (K / 16);
    const uint2 q = codes[i];
    const size_t si = PACKED_SCALES ? packed_scale_index(row, group, K / 16) : i;
    const half2 scale = __hmul2(fp8e4m3_to_half2(scales[si]), __float2half2_rn(global * 16384.f));
    alignas(16) half2 values[8];
    const unsigned words[2] = {q.x, q.y};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned b = (words[j / 4] >> ((j % 4) * 8)) & 255;
        const unsigned bits = ((b & 8) << 12) | ((b & 7) << 9) | ((b & 128) << 24) | ((b & 112) << 21);
        values[j] = __hmul2(*reinterpret_cast<const half2*>(&bits), scale);
    }
    auto* dst = reinterpret_cast<uint4*>(out + i * 16);
    dst[0] = reinterpret_cast<const uint4*>(values)[0];
    dst[1] = reinterpret_cast<const uint4*>(values)[1];
}
// One CSF32 tile covers 128 rows and four scale groups. Resolve each
// warp's exception list in shared memory, then emit coalesced FP16 row spans.
__global__ void fp4_csf32_operands(const uint2* weights, const uint8_t* bases, const uint8_t* codes,
                                 const uint16_t* offsets, const uint32_t* pages, const uint16_t* exceptions,
                                 float global, half* out, int* invalid, int K) {
    __shared__ uint8_t scales[128 * 4];
    __shared__ __align__(16) half tile_values[128][72];
    const int tid = threadIdx.x, tile = blockIdx.x, C = K / 16;
    const int row0 = (tile / (C / 4)) * 128, col0 = (tile % (C / 4)) * 4;
    if (tid < 256) {
        const int lane = tid & 31, part = tid >> 5, r = (part % 4) * 32 + lane, col = (part / 4) * 2;
        const int block = tile * 8 + part, page = tile / 32;
        const uint8_t code = codes[size_t(block) * 32 + lane];
        scales[r * 4 + col] = uint8_t(bases[row0 + r] + (code & 15));
        scales[r * 4 + col + 1] = uint8_t(bases[row0 + r] + (code >> 4));
        __syncwarp();
        const size_t first = pages[page] / 2 + offsets[block + page],
                     last = pages[page] / 2 + offsets[block + page + 1];
        for (size_t i = first + lane; i < last; i += 32) {
            const unsigned value = exceptions[i], slot = value & 255;
            if (slot >= 64) {
                atomicExch(invalid, 1);
                continue;
            }
            scales[((part % 4) * 32 + slot / 2) * 4 + col + slot % 2] = value >> 8;
        }
    }
    __syncthreads();
    const int r = tid / 4, group = tid % 4;
    const uint8_t code_scale = scales[tid];
    if (code_scale >= 127)
        atomicExch(invalid, 1);
    const half2 scale = __hmul2(fp8e4m3_to_half2(code_scale), __float2half2_rn(global * 16384.f));
    const uint2 q = weights[size_t(row0 + r) * (K / 16) + col0 + group];
    const unsigned words[2] = {q.x, q.y};
    alignas(16) half2 values[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned b = (words[j / 4] >> ((j % 4) * 8)) & 255;
        const unsigned bits = ((b & 8) << 12) | ((b & 7) << 9) | ((b & 128) << 24) | ((b & 112) << 21);
        values[j] = __hmul2(*reinterpret_cast<const half2*>(&bits), scale);
    }
    auto* dst = reinterpret_cast<uint4*>(&tile_values[r][group * 16]);
    dst[0] = reinterpret_cast<const uint4*>(values)[0];
    dst[1] = reinterpret_cast<const uint4*>(values)[1];
    __syncthreads();
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int row = tid / 8 + pass * 64, col = (tid % 8) * 8;
        *reinterpret_cast<uint4*>(out + size_t(row0 + row) * K + col0 * 16 + col) =
            *reinterpret_cast<const uint4*>(&tile_values[row][col]);
    }
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
// Each block reuses a reconstructed 64x64 weight tile across 32/64 tokens.
// Operand buffers alias the output tile only after every WMMA reader finishes.
// Transposed weight stores are contiguous across output rows. Both half leading
// dimensions are 72, with 32-byte-aligned WMMA tile origins.
template<int MT>
__global__ void fp4_prefill_mma(const uint2* codes,const uint8_t* scales,float global,
                               const half* x,float* y,int N,int K,int M) {
#if __CUDA_ARCH__ >= 700
    namespace wm=nvcuda::wmma;
    constexpr int NT=64,BK=64,LD=72,ACC=MT/32;
    extern __shared__ __align__(32) unsigned char storage[];
    auto* sw=reinterpret_cast<half*>(storage);
    auto* sx=sw+BK*LD;
    auto* output=reinterpret_cast<float*>(storage);
    const int tid=threadIdx.x,warp=tid/32;
    const int n0=blockIdx.x*NT,m0=blockIdx.y*MT;
    const int wn=(warp%4)*16,wm0=(warp/4)*(MT/2);
    wm::fragment<wm::accumulator,16,16,16,float> acc[ACC];
    #pragma unroll
    for(int a=0;a<ACC;++a)wm::fill_fragment(acc[a],0.f);
    const half2 gm=__float2half2_rn(global*16384.f);
    for(int k0=0;k0<K;k0+=BK) {
        if(tid<NT*(BK/16)) {
            const int r=tid%NT,g=tid/NT,row=n0+r;
            half2 values[8];
            if(row<N) {
                const int rr=row&7,lane=((row&31)>>3)*4+(rr&3)+((rr&4)?16:0);
                const size_t group=(size_t(row/32)*(K/16)+k0/16+g)*32+lane;
                const uint2 q=codes[group];
                const half2 scale=__hmul2(fp8e4m3_to_half2(scales[group]),gm);
                dequant8_tm(q.x,scale,values);dequant8_tm(q.y,scale,values+4);
            }else {
                #pragma unroll
                for(int j=0;j<8;++j)values[j]=__float2half2_rn(0.f);
            }
            #pragma unroll
            for(int j=0;j<8;++j) {
                sw[(g*16+2*j)*LD+r]=__low2half(values[j]);
                sw[(g*16+2*j+1)*LD+r]=__high2half(values[j]);
            }
        }
        for(int i=tid;i<MT*BK/8;i+=256) {
            const int r=i/(BK/8),c=(i%(BK/8))*8;
            const uint4 v=m0+r<M?*reinterpret_cast<const uint4*>(x+size_t(m0+r)*K+k0+c):make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(sx+r*LD+c)=v;
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<BK;kk+=16) {
            wm::fragment<wm::matrix_b,16,16,16,half,wm::row_major> b;
            wm::load_matrix_sync(b,sw+kk*LD+wn,LD);
            #pragma unroll
            for(int a=0;a<ACC;++a) {
                wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major> operand;
                wm::load_matrix_sync(operand,sx+(wm0+a*16)*LD+kk,LD);
                wm::mma_sync(acc[a],operand,b,acc[a]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int a=0;a<ACC;++a)wm::store_matrix_sync(output+(wm0+a*16)*NT+wn,acc[a],NT,wm::mem_row_major);
    __syncthreads();
    for(int i=tid;i<MT*NT;i+=256) {
        const int m=m0+i/NT,n=n0+i%NT;
        if(m<M&&n<N)y[size_t(m)*N+n]=output[i];
    }
#else
    asm volatile("trap;");
#endif
}
// Reuse each reconstructed weight across up to 128 tokens. Splitting K
// supplies enough blocks for the narrow 32-output tile on a V100.
__global__ void fp4_prefill_split(const uint2* codes,const uint8_t* scales,float global,
                                 const half* x,float* y,int N,int K,int M,int parts) {
#if __CUDA_ARCH__ >= 700
    namespace wm=nvcuda::wmma;
    constexpr int MT=128,NT=32,BK=64,LDW=40,LDX=72;
    extern __shared__ __align__(32) unsigned char storage[];
    auto* sw=reinterpret_cast<half*>(storage);
    auto* sx=sw+BK*LDW;
    auto* output=reinterpret_cast<float*>(storage);
    const int tid=threadIdx.x,warp=tid/32;
    const int n0=blockIdx.x*NT,m0=blockIdx.y*MT;
    const int wn=(warp%2)*16,wm0=(warp/2)*32;
    const int slice=((K/BK+parts-1)/parts)*BK;
    const int first=int(blockIdx.z)*slice,last=min(K,first+slice);
    wm::fragment<wm::accumulator,16,16,16,float> acc[2];
    wm::fill_fragment(acc[0],0.f);wm::fill_fragment(acc[1],0.f);
    const half2 gm=__float2half2_rn(global*16384.f);
    for(int k0=first;k0<last;k0+=BK) {
        if(tid<NT*(BK/16)) {
            const int r=tid%NT,g=tid/NT,row=n0+r;
            const int rr=row&7,lane=((row&31)>>3)*4+(rr&3)+((rr&4)?16:0);
            const size_t group=(size_t(row/32)*(K/16)+k0/16+g)*32+lane;
            const uint2 q=codes[group];
            const half2 scale=__hmul2(fp8e4m3_to_half2(scales[group]),gm);
            half2 values[8];dequant8_tm(q.x,scale,values);dequant8_tm(q.y,scale,values+4);
            #pragma unroll
            for(int j=0;j<8;++j) {
                sw[(g*16+2*j)*LDW+r]=__low2half(values[j]);
                sw[(g*16+2*j+1)*LDW+r]=__high2half(values[j]);
            }
        }
        for(int i=tid;i<MT*BK/8;i+=256) {
            const int r=i/(BK/8),c=(i%(BK/8))*8;
            const uint4 v=m0+r<M?*reinterpret_cast<const uint4*>(x+size_t(m0+r)*K+k0+c):make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(sx+r*LDX+c)=v;
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<BK;kk+=16) {
            wm::fragment<wm::matrix_b,16,16,16,half,wm::row_major> b;
            wm::load_matrix_sync(b,sw+kk*LDW+wn,LDW);
            #pragma unroll
            for(int a=0;a<2;++a) {
                wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major> operand;
                wm::load_matrix_sync(operand,sx+(wm0+a*16)*LDX+kk,LDX);
                wm::mma_sync(acc[a],operand,b,acc[a]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int a=0;a<2;++a)wm::store_matrix_sync(output+(wm0+a*16)*NT+wn,acc[a],NT,wm::mem_row_major);
    __syncthreads();
    const size_t base=size_t(blockIdx.z)*M*N;
    for(int i=tid;i<MT*NT;i+=256) {
        const int m=m0+i/NT,n=n0+i%NT;
        if(m<M)y[base+size_t(m)*N+n]=output[i];
    }
#else
    asm volatile("trap;");
#endif
}
__global__ void reduce_prefill_partials(const float* partials,float* y,int count,int parts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    float sum=0;
    for(int part=0;part<parts;++part)sum+=partials[size_t(part)*count+i];
    y[i]=sum;
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
template<GlmPackedFormat FORMAT>
__global__ void half_qpn_cache(const void* w,const void* scales,half* out,int N,int K) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(N)*(K/16))return;
    const int g=i%(K/16),n=i/(K/16);
    alignas(16) half values[16];
    if constexpr(FORMAT==GlmPackedFormat::F16) {
        const auto* src=reinterpret_cast<const uint4*>(static_cast<const half*>(w)+size_t(n)*K+g*16);
        auto* dst=reinterpret_cast<uint4*>(values);dst[0]=src[0];dst[1]=src[1];
    } else if(!fp8_operands<FORMAT>(w,scales,n,g,K,values)) {
        #pragma unroll
        for(int j=0;j<16;++j)values[j]=__float2half_rn(weight_value(FORMAT,w,scales,1.f,n,g*16+j,K));
    }
    auto* dst=reinterpret_cast<uint4*>(out+packed_scale_index(n,g,K/16)*16);
    const auto* src=reinterpret_cast<const uint4*>(values);dst[0]=src[0];dst[1]=src[1];
}
// Same quadpair geometry for MXFP8, block FP8 and BF16. Group scaling is
// applied before each MMA operand conversion, rather than as a single matrix
// epilogue scale: MXFP8 varies along K and block FP8 varies along both axes.
// SPLITK is fixed across M=1..8 so verification does not change the K reduction.
template<GlmPackedFormat FORMAT,int SPLITK,int OUTER=1,int NACC=1>
__device__ __forceinline__ void scaled_qpn_body(const void* w,const void* scales,
                            const half* x,float* y,int N,int K,int M,float (&sums)[SPLITK][256]) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,qp=(lane>>2)&3;
    const int r=(lane&3)+((lane&16)?4:0),n=blockIdx.x*32+qp*8+r;
    const int groups=K/16,part=blockIdx.y*SPLITK+warp;
    const int first=groups*part/(SPLITK*OUTER),last=groups*(part+1)/(SPLITK*OUTER);
    float c[NACC][8]={};
    for(int g=first;g<last;++g) {
        alignas(16) half b[16];
        if constexpr(FORMAT==GlmPackedFormat::F16Qpn) {
            uint4 b0=make_uint4(0,0,0,0),b1=b0;
            if(n<N) {
                const half* row=static_cast<const half*>(w)+packed_scale_index(n,g,K/16)*16;
                b0=*reinterpret_cast<const uint4*>(row);b1=*reinterpret_cast<const uint4*>(row+8);
            }
            *reinterpret_cast<uint4*>(b)=b0;*reinterpret_cast<uint4*>(b+8)=b1;
        }else if constexpr(FORMAT==GlmPackedFormat::F16) {
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
        MMA_8N8K4(c[0],A0[0],A0[1],B[0],B[1]);
        MMA_8N8K4(c[1%NACC],A0[2],A0[3],B[2],B[3]);
        MMA_8N8K4(c[2%NACC],A1[0],A1[1],B[4],B[5]);
        MMA_8N8K4(c[3%NACC],A1[2],A1[3],B[6],B[7]);
    }
    #pragma unroll
    for(int a=1;a<NACC;++a)
    #pragma unroll
    for(int i=0;i<8;++i)c[0][i]+=c[a][i];
    #pragma unroll
    for(int i=0;i<8;++i) {
        const int row=(i&2)|((lane&16)?4:0)|(lane&1);
        const int col=(i&1)|(((lane>>1)&1)<<1)|((i>>2)<<2);
        sums[warp][row*32+qp*8+col]=c[0][i];
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
template<GlmPackedFormat FORMAT,int SPLITK,int OUTER=1,int NACC=1>
__global__ void scaled_qpn(const void* w,const void* scales,
                            const half* x,float* y,int N,int K,int M) {
    __shared__ float sums[SPLITK][256];
    scaled_qpn_body<FORMAT,SPLITK,OUTER,NACC>(w,scales,x,y,N,K,M,sums);
}
// Independent matrices share one launch and input, without concatenating or
// copying resident weights. Each block keeps the ordinary eight-warp K tree.
__global__ void mxfp8_qpn_three(GlmFp8Projection a,GlmFp8Projection b,GlmFp8Projection c,
                               const half* x,int N,int K,int M) {
    __shared__ float sums[8][256];
    const auto p=blockIdx.z==0?a:blockIdx.z==1?b:c;
    scaled_qpn_body<GlmPackedFormat::Mxfp8Qpn,8>(p.weights,p.scales,x,p.output,N,K,M,sums);
}
__global__ void reduce_partials(const float* partials,float* y,int N,int M,int parts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N*M)return;
    float sum=0;
    for(int part=0;part<parts;++part)sum+=partials[size_t(part)*8*N+i];
    y[i]=sum;
}
template<GlmPackedFormat FORMAT,int PARTS,int NACC=1>
void small_gemm(const void* w,const void* scales,const half* x,float* y,float* partials,int N,int K,int M,cudaStream_t stream) {
    scaled_qpn<FORMAT,8,PARTS,NACC><<<dim3((N+31)/32,PARTS),256,0,stream>>>(w,scales,x,partials,N,K,M);
    reduce_partials<<<(N*M+255)/256,256,0,stream>>>(partials,y,N,M,PARTS);
}
template<GlmPackedFormat FORMAT,int NACC=1>
void scaled_gemm(const void* w,const void* scales,const half* x,float* y,float* partials,int N,int K,int M,cudaStream_t stream) {
    if(partials && N<=8192 && K>=4096) {
        if(N<=32)small_gemm<FORMAT,64,NACC>(w,scales,x,y,partials,N,K,M,stream);
        else if(N<=64)small_gemm<FORMAT,32,NACC>(w,scales,x,y,partials,N,K,M,stream);
        else if(N<=128)small_gemm<FORMAT,16,NACC>(w,scales,x,y,partials,N,K,M,stream);
        else if(N<=1024)small_gemm<FORMAT,8,NACC>(w,scales,x,y,partials,N,K,M,stream);
        else if(N<=2048)small_gemm<FORMAT,4,NACC>(w,scales,x,y,partials,N,K,M,stream);
        else small_gemm<FORMAT,2,NACC>(w,scales,x,y,partials,N,K,M,stream);
    }else scaled_qpn<FORMAT,8,1,NACC><<<(N+31)/32,256,0,stream>>>(w,scales,x,y,N,K,M);
}
template<GlmPackedFormat FORMAT>
void fp8_gemm(const void* w,const void* scales,const half* x,float* y,float* partials,int N,int K,int M,cudaStream_t stream,int accumulators) {
    if(accumulators==4)scaled_gemm<FORMAT,4>(w,scales,x,y,partials,N,K,M,stream);
    else if(accumulators==2)scaled_gemm<FORMAT,2>(w,scales,x,y,partials,N,K,M,stream);
    else scaled_gemm<FORMAT>(w,scales,x,y,partials,N,K,M,stream);
}
__global__ void route_gather(const float* x,const int* routes,half* out,int count,int K,int stride) {
    const int column=blockIdx.x*blockDim.x+threadIdx.x;
    if(column>=K)return;
    for(int row=blockIdx.y;row<count;row+=gridDim.y)
        out[size_t(row)*K+column]=__float2half_rn(x[size_t(routes[row]/stride)*K+column]);
}
// Resolve the one row for each expert/token in parallel, then retain the
// exact expert order in a single per-column accumulation. No atomics or tree.
__global__ void grouped_route_scatter(const GlmFp4ExpertGroup* __restrict__ groups,int count,
                                      const int* __restrict__ routes,const float* __restrict__ weights,
                                      const float* __restrict__ values,float* __restrict__ sum) {
    __shared__ int rows[64];
    __shared__ float coefficients[64];
    const int token=blockIdx.y,g=threadIdx.x;
    if(g<count) {
        const auto* group=groups+g;
        int selected=-1;float coefficient=0;
        for(int j=0;j<group->count;++j) {
            const int row=group->offset+j,route=routes[row];
            if(route/8==token){selected=row;coefficient=weights[route];break;}
        }
        rows[g]=selected;coefficients[g]=coefficient;
    }
    __syncthreads();
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    const size_t target=size_t(token)*4096+col;
    float value=sum[target];
    for(int j=0;j<count;++j) {
        const int row=rows[j];
        if(row>=0)value=__fadd_rn(value,__fmul_rn(coefficients[j],values[size_t(row)*4096+col]));
    }
    sum[target]=value;
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
                  const uint16_t* x,float* y,int N,int K,int M,bool sm70,void* stream,float* partials,int fp8_accumulators) {
    valid(N,K,M);auto s=static_cast<cudaStream_t>(stream);const half* h=reinterpret_cast<const half*>(x);
    if(fp8_accumulators!=1&&fp8_accumulators!=2&&fp8_accumulators!=4)
        throw std::invalid_argument("GLM FP4: FP8 accumulators must be 1, 2 or 4");
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
        case GlmPackedFormat::Mxfp8:fp8_gemm<GlmPackedFormat::Mxfp8>(w,scales,h,y,partials,N,K,M,s,fp8_accumulators);break;
        case GlmPackedFormat::BlockFp8:fp8_gemm<GlmPackedFormat::BlockFp8>(w,scales,h,y,partials,N,K,M,s,fp8_accumulators);break;
        case GlmPackedFormat::Bf16:scaled_gemm<GlmPackedFormat::Bf16>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::F32:scaled_gemm<GlmPackedFormat::F32>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::F16:scaled_gemm<GlmPackedFormat::F16>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::F16Qpn:scaled_gemm<GlmPackedFormat::F16Qpn>(w,scales,h,y,partials,N,K,M,s);break;
        case GlmPackedFormat::Mxfp8Qpn:fp8_gemm<GlmPackedFormat::Mxfp8Qpn>(w,scales,h,y,partials,N,K,M,s,fp8_accumulators);break;
        case GlmPackedFormat::BlockFp8Qpn:fp8_gemm<GlmPackedFormat::BlockFp8Qpn>(w,scales,h,y,partials,N,K,M,s,fp8_accumulators);break;
        default:throw std::invalid_argument("GLM FP4: unknown packed format");
        }
    }
}
void glm_mxfp8_gemm_three(GlmFp8Projection a,GlmFp8Projection b,GlmFp8Projection c,
                         const uint16_t* x,int N,int K,int M,void* stream) {
    valid(N,K,M);
    if(N%32||M>8||!x||!a.weights||!b.weights||!c.weights||!a.scales||!b.scales||!c.scales||
       !a.output||!b.output||!c.output)
        throw std::invalid_argument("GLM FP8: three products require packed MXFP8 and tokens1..8");
    mxfp8_qpn_three<<<dim3(N/32,1,3),256,0,static_cast<cudaStream_t>(stream)>>>(
        a,b,c,reinterpret_cast<const half*>(x),N,K,M);
}
void glm_fp4_expert_group_products(const GlmFp4ExpertGroup* groups,int count,bool down,
                                   const uint16_t* input,float* output,float* up,void* stream) {
    if(count<1||count>64||!groups||!input||!output||(!down&&!up))
        throw std::invalid_argument("GLM FP4: invalid resident expert batch");
    const auto s=static_cast<cudaStream_t>(stream);
    if(down)fp4_qpn_grouped<8,2,false,true><<<dim3(4096/32,count),256,0,s>>>(groups,reinterpret_cast<const half*>(input),output,nullptr,4096,2048);
    else fp4_qpn_grouped<8,2,true,false><<<dim3(2048/8,2*count),256,0,s>>>(groups,reinterpret_cast<const half*>(input),output,up,2048,4096);
}
void glm_fp4_gate_up_f16(const uint8_t* gate,const uint8_t* gate_scales,float gate_global,
                         const uint8_t* up,const uint8_t* up_scales,float up_global,
                         const uint16_t* x,uint16_t* hidden,int N,int K,int M,float limit,void* stream) {
    valid(N,K,M);
    if(N%32||M>8||!gate||!gate_scales||!up||!up_scales||!x||!hidden)
        throw std::invalid_argument("GLM FP4: fused gate/up requires aligned NVFP4 matrices and tokens1..8");
    auto s=static_cast<cudaStream_t>(stream);
    if(N<=2048&&K%512==0)fp4_gate_up<true><<<N/8,256,0,s>>>(gate,gate_scales,gate_global,up,up_scales,up_global,reinterpret_cast<const half*>(x),reinterpret_cast<half*>(hidden),N,K,M,limit);
    else fp4_gate_up<false><<<N/32,256,0,s>>>(gate,gate_scales,gate_global,up,up_scales,up_global,reinterpret_cast<const half*>(x),reinterpret_cast<half*>(hidden),N,K,M,limit);
}
void glm_fp4_csf32_dequant_f16(const uint8_t* weights, const uint8_t* bases, const uint8_t* codes,
                             const uint16_t* offsets, const uint32_t* pages, const uint16_t* exceptions,
                             float global, uint16_t* output, int* invalid, int N, int K, void* stream) {
    valid(N, K, 1);
    if (N % 128 || !weights || !bases || !codes || !offsets || !pages || !exceptions || !output || !invalid)
        throw std::invalid_argument("GLM FP4: fused CSF operands require aligned validated matrices");
    const auto s = static_cast<cudaStream_t>(stream);
    cudaMemsetAsync(invalid, 0, sizeof(int), s);
    fp4_csf32_operands<<<(N / 128) * (K / 64), 512, 0, s>>>(
        reinterpret_cast<const uint2*>(weights), bases, codes, offsets, pages, exceptions,
        global, reinterpret_cast<half*>(output), invalid, K);
}
void glm_fp4_raw_dequant_f16(const uint8_t* weights, const uint8_t* scales, float global,
                           uint16_t* output, int N, int K, bool packed_scales, void* stream) {
    valid(N, K, 1);
    if (N % 32 || !weights || !scales || !output)
        throw std::invalid_argument("GLM FP4: raw operands need aligned NVFP4 matrices");
    const size_t count = size_t(N) * (K / 16);
    const auto s = static_cast<cudaStream_t>(stream);
    if (packed_scales)
        fp4_raw_operands<true><<<(count + 255) / 256, 256, 0, s>>>(
            reinterpret_cast<const uint2*>(weights), scales, global, reinterpret_cast<half*>(output), N, K);
    else
        fp4_raw_operands<false><<<(count + 255) / 256, 256, 0, s>>>(
            reinterpret_cast<const uint2*>(weights), scales, global, reinterpret_cast<half*>(output), N, K);
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
void glm_fp4_prefill_mma(const uint8_t* weights,const uint8_t* scales,float global,
                        const uint16_t* x,float* y,int N,int K,int M,int tile,void* stream) {
    valid(N,K,M);
    if(N%32||K%32||M<=8||(tile!=32&&tile!=64))
        throw std::invalid_argument("GLM FP4: prefill MMA requires N/K multiples of32, M>8 and tile32/64");
    const auto s=static_cast<cudaStream_t>(stream);
    if(tile==32)fp4_prefill_mma<32><<<dim3((N+63)/64,(M+31)/32),256,13824,s>>>(
        reinterpret_cast<const uint2*>(weights),scales,global,reinterpret_cast<const half*>(x),y,N,K,M);
    else fp4_prefill_mma<64><<<dim3((N+63)/64,(M+63)/64),256,18432,s>>>(
        reinterpret_cast<const uint2*>(weights),scales,global,reinterpret_cast<const half*>(x),y,N,K,M);
}
void glm_fp4_prefill_split_mma(const uint8_t* weights,const uint8_t* scales,float global,
                              const uint16_t* x,float* y,int N,int K,int M,
                              int k_parts,float* scratch,void* stream) {
    valid(N,K,M);
    if(N%32||N>8192||K>8192||M<=8||M>512||
       (k_parts!=1&&k_parts!=2&&k_parts!=4)||!weights||!scales||!x||!y)
        throw std::invalid_argument("GLM FP4: split prefill requires aligned bounded matrices and k_parts1/2/4");
    const int parts=std::min(k_parts,K/64);
    if(parts>1&&!scratch)throw std::invalid_argument("GLM FP4: split prefill needs partial output scratch");
    const auto s=static_cast<cudaStream_t>(stream);
    fp4_prefill_split<<<dim3(N/32,(M+127)/128,parts),256,23552,s>>>(
        reinterpret_cast<const uint2*>(weights),scales,global,reinterpret_cast<const half*>(x),
        parts>1?scratch:y,N,K,M,parts);
    if(parts>1)reduce_prefill_partials<<<(N*M+255)/256,256,0,s>>>(scratch,y,N*M,parts);
}
void glm_fp8_cache_f16_qpn(GlmPackedFormat format,const void* w,const void* scales,uint16_t* out,int N,int K,void* stream) {
    valid(N,K,1);if(N%32)throw std::invalid_argument("half QPN requires N%32=0");
    const size_t count=size_t(N)*(K/16);auto s=static_cast<cudaStream_t>(stream);
    if(format==GlmPackedFormat::Mxfp8Qpn)half_qpn_cache<GlmPackedFormat::Mxfp8Qpn><<<(count+255)/256,256,0,s>>>(w,scales,reinterpret_cast<half*>(out),N,K);
    else if(format==GlmPackedFormat::Mxfp8)half_qpn_cache<GlmPackedFormat::Mxfp8><<<(count+255)/256,256,0,s>>>(w,scales,reinterpret_cast<half*>(out),N,K);
    else if(format==GlmPackedFormat::BlockFp8Qpn)half_qpn_cache<GlmPackedFormat::BlockFp8Qpn><<<(count+255)/256,256,0,s>>>(w,scales,reinterpret_cast<half*>(out),N,K);
    else if(format==GlmPackedFormat::BlockFp8)half_qpn_cache<GlmPackedFormat::BlockFp8><<<(count+255)/256,256,0,s>>>(w,scales,reinterpret_cast<half*>(out),N,K);
    else throw std::invalid_argument("unsupported half QPN source");
}
void glm_f16_prepack_qpn(const uint16_t* w,uint16_t* out,int N,int K,void* stream) {
    valid(N,K,1);
    if(N%32||!w||!out||w==out)throw std::invalid_argument("FP16 QPN requires aligned distinct matrices");
    const size_t count=size_t(N)*(K/16);
    half_qpn_cache<GlmPackedFormat::F16><<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
        w,nullptr,reinterpret_cast<half*>(out),N,K);
}
void glm_fp4_route_gather(const float* x,const int* routes,uint16_t* out,int count,int K,int stride,void* stream) {
    if(count<0 || K<1 || stride<1)throw std::invalid_argument("GLM FP4: invalid route gather extent");
    if(count)route_gather<<<dim3((K+255)/256,std::min(count,65535)),256,0,static_cast<cudaStream_t>(stream)>>>(x,routes,reinterpret_cast<half*>(out),count,K,stride);
}
void glm_fp4_expert_group_scatter(const GlmFp4ExpertGroup* groups,int count,
                                 const int* routes,const float* weights,const float* values,
                                 float* sum,int tokens,void* stream) {
    if(count<1||count>64||tokens<1||tokens>8||!groups||!routes||!weights||!values||!sum)
        throw std::invalid_argument("GLM FP4: invalid grouped ordered scatter");
    grouped_route_scatter<<<dim3(4096/256,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(groups,count,routes,weights,values,sum);
}
void glm_fp4_route_scatter(const float* x,const int* routes,const float* weights,float* sum,int count,int K,int stride,void* stream) {
    if(count<0 || K<1 || stride<1)throw std::invalid_argument("GLM FP4: invalid route scatter extent");
    if(count)route_scatter<<<dim3((K+255)/256,std::min(count,65535)),256,0,static_cast<cudaStream_t>(stream)>>>(x,routes,weights,sum,count,K,stride);
}
} // namespace strata::kernels
