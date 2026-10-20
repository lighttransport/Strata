#include "strata/kernels/cpu/fp4_local_w4a16.hpp"
#include "strata/host/glm_fp4.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define W4A16_AVX2 1
#endif
namespace strata::kernels::cpu {
namespace {
const std::array<float,127>& half_scales() {
    static const auto values=[] {std::array<float,127> a{};for(int i=0;i<127;++i)a[i]=strata_glm_e4m3(uint8_t(i))*.5f;return a;}();
    return values;
}
void scalar(const uint8_t* w,const uint8_t* s,const float* x,float global,int K,int first,int last,float* y,bool planar) {
    constexpr int mag[]={0,1,2,3,4,6,8,12};const auto& scale=half_scales();
    for(int r=first;r<last;++r) {
        float sum=0;
        for(int j=0;j<K;++j) {
            const size_t at=size_t(r)*(K/2)+(planar?(j/32)*16+j%16:j/2);
            const unsigned q=(w[at]>>(planar?(j%32)/16*4:j%2*4))&15;
            const float weight=float(q&8?-mag[q&7]:mag[q&7])*scale[s[size_t(r)*(K/16)+j/16]];
            sum=std::fma(weight,x[j],sum);
        }
        y[r]=sum*global;
    }
}
#ifdef W4A16_AVX2
template<bool Planar>
__attribute__((target("avx2,fma")))
void vector_rows(const uint8_t* w,const uint8_t* s,const float* x,float global,int K,int first,int last,float* y) {
    const auto& scale=half_scales();
    const auto lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const auto mask=_mm_set1_epi8(15);
    for(int r=first;r<last;++r) {
        auto acc0=_mm256_setzero_ps(),acc1=acc0;
        for(int block=0;block<K/32;++block) {
            const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+size_t(r)*(K/2)+block*16));
            const auto lo=_mm_and_si128(packed,mask),hi=_mm_and_si128(_mm_srli_epi16(packed,4),mask);
            const auto a=_mm_shuffle_epi8(lut,Planar?lo:_mm_unpacklo_epi8(lo,hi));
            const auto b=_mm_shuffle_epi8(lut,Planar?hi:_mm_unpackhi_epi8(lo,hi));
            const auto s0=_mm256_set1_ps(scale[s[size_t(r)*(K/16)+block*2]]);
            const auto s1=_mm256_set1_ps(scale[s[size_t(r)*(K/16)+block*2+1]]);
            const auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a)),s0);
            const auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a,8))),s0);
            const auto w2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)),s1);
            const auto w3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b,8))),s1);
            acc0=_mm256_fmadd_ps(w0,_mm256_loadu_ps(x+block*32),acc0);
            acc1=_mm256_fmadd_ps(w1,_mm256_loadu_ps(x+block*32+8),acc1);
            acc0=_mm256_fmadd_ps(w2,_mm256_loadu_ps(x+block*32+16),acc0);
            acc1=_mm256_fmadd_ps(w3,_mm256_loadu_ps(x+block*32+24),acc1);
        }
        const auto acc=_mm256_add_ps(acc0,acc1);
        auto v=_mm_add_ps(_mm256_castps256_ps128(acc),_mm256_extractf128_ps(acc,1));
        v=_mm_hadd_ps(v,v);v=_mm_hadd_ps(v,v);y[r]=_mm_cvtss_f32(v)*global;
    }
}
template<bool Planar,int Tokens=2>
__attribute__((target("avx2,fma")))
void vector_token_rows(const uint8_t* w,const uint8_t* s,const float* x,float global,int K,int N,int first,int last,float* y) {
    const auto& scale=half_scales();
    const auto lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const auto mask=_mm_set1_epi8(15);
    for(int r=first;r<last;++r) {
        __m256 acc0[Tokens]={};
        __m256 acc1[Tokens]={};
        for(int block=0;block<K/32;++block) {
            const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+size_t(r)*(K/2)+block*16));
            const auto lo=_mm_and_si128(packed,mask),hi=_mm_and_si128(_mm_srli_epi16(packed,4),mask);
            const auto a=_mm_shuffle_epi8(lut,Planar?lo:_mm_unpacklo_epi8(lo,hi));
            const auto b=_mm_shuffle_epi8(lut,Planar?hi:_mm_unpackhi_epi8(lo,hi));
            const auto s0=_mm256_set1_ps(scale[s[size_t(r)*(K/16)+block*2]]);
            const auto s1=_mm256_set1_ps(scale[s[size_t(r)*(K/16)+block*2+1]]);
            const auto w0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a)),s0);
            const auto w1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a,8))),s0);
            const auto w2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)),s1);
            const auto w3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b,8))),s1);
            for(int t=0;t<Tokens;++t) {
            acc0[t]=_mm256_fmadd_ps(w0,_mm256_loadu_ps(x+size_t(t)*K+block*32),acc0[t]);
            acc1[t]=_mm256_fmadd_ps(w1,_mm256_loadu_ps(x+size_t(t)*K+block*32+8),acc1[t]);
            acc0[t]=_mm256_fmadd_ps(w2,_mm256_loadu_ps(x+size_t(t)*K+block*32+16),acc0[t]);
            acc1[t]=_mm256_fmadd_ps(w3,_mm256_loadu_ps(x+size_t(t)*K+block*32+24),acc1[t]);
            }
        }
        for(int t=0;t<Tokens;++t) {
        const auto acc=_mm256_add_ps(acc0[t],acc1[t]);
        auto v=_mm_add_ps(_mm256_castps256_ps128(acc),_mm256_extractf128_ps(acc,1));
        v=_mm_hadd_ps(v,v);v=_mm_hadd_ps(v,v);y[size_t(t)*N+r]=_mm_cvtss_f32(v)*global;
        }
    }
}
#endif
}
void fp4_local_w4a16_rows(const uint8_t* w,const uint8_t* s,const float* x,float global,int K,int N,int first,int last,float* y,bool planar,bool simd) {
    if(!w||!s||!x||!y||K<32||K%32||K>16384||N<1||N>16384||first<0||last<first||last>N||!std::isfinite(global)||global<=0)
        throw std::invalid_argument("invalid W4A16 CPU extent");
#ifdef W4A16_AVX2
    if(simd&&__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")) {
        if(planar)vector_rows<true>(w,s,x,global,K,first,last,y);
        else vector_rows<false>(w,s,x,global,K,first,last,y);
        return;
    }
#else
    (void)simd;
#endif
    scalar(w,s,x,global,K,first,last,y,planar);
}
void fp4_local_w4a16_batch_rows(const uint8_t* w,const uint8_t* s,const float* x,float global,
        int K,int N,int tokens,int first,int last,float* y,bool planar,bool simd) {
    if(!w||!s||!x||!y||K<32||K%32||K>16384||N<1||N>16384||tokens<1||tokens>16||
       first<0||last<first||last>N||!std::isfinite(global)||global<=0)
        throw std::invalid_argument("invalid batched W4A16 CPU extent");
    int t=0;
#ifdef W4A16_AVX2
    if(simd&&__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")) {
        for(;t+3<tokens;t+=4) {
            if(planar)vector_token_rows<true,4>(w,s,x+size_t(t)*K,global,K,N,first,last,y+size_t(t)*N);
            else vector_token_rows<false,4>(w,s,x+size_t(t)*K,global,K,N,first,last,y+size_t(t)*N);
        }
        for(;t+1<tokens;t+=2) {
            if(planar)vector_token_rows<true>(w,s,x+size_t(t)*K,global,K,N,first,last,y+size_t(t)*N);
            else vector_token_rows<false>(w,s,x+size_t(t)*K,global,K,N,first,last,y+size_t(t)*N);
        }
    }
#endif
    for(;t<tokens;++t)fp4_local_w4a16_rows(w,s,x+size_t(t)*K,global,K,N,first,last,y+size_t(t)*N,planar,simd);
}

}
