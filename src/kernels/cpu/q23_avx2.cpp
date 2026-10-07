#include "strata/kernels/cpu/q23_avx2.hpp"
#include "strata/kernels/canon_expert.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace strata::kernels::cpu {
namespace {
float half(ggml_half h) { uint16_t u; std::memcpy(&u,&h,2);return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(u))); }
float sum(__m256 v) {
    __m128 a=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
    a=_mm_hadd_ps(a,a);return _mm_cvtss_f32(_mm_hadd_ps(a,a));
}
// Word 2k of lane 0 and word 2k+1 of lane 1 (of a vector whose two lanes hold the same eight 16-bit scales).
inline __m256i scale_mask(int k) {
    return _mm256_setr_epi8(4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,
                            4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3);
}
constexpr int kMaxBlocks=16;
// Optional software prefetch distance in bytes (STRATA_Q23_PREFETCH, default off); it does not change results.
int prefetch_distance() {
    static const int d=[]{const char* v=std::getenv("STRATA_Q23_PREFETCH");return v?std::max(0,std::atoi(v)):0;}();
    return d;
}
template<int TY,int NT,int H> inline void chunk(const __m256i& q0,const __m256i& q1,const __m256i& hm,const __m256i& sA,const __m256i& sB,
                                               const __m256i* masks,const int8_t* const* y,const __m256i* p4,int b,__m256i* sums) {
    const __m256i src=H<4?q0:q1;
    __m256i q=_mm256_and_si256(_mm256_srli_epi16(src,2*(H%4)),_mm256_set1_epi8(3));
    if constexpr(TY==11) {
        const __m256i hb=H>=2?_mm256_srli_epi16(hm,H-2):_mm256_slli_epi16(hm,2-H);
        q=_mm256_or_si256(q,_mm256_and_si256(hb,_mm256_set1_epi8(4)));
    }
    const __m256i sc=_mm256_shuffle_epi8(H<4?sA:sB,masks[H%4]);
    for(int t=0;t<NT;++t) {
        const __m256i v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[t]+32*H));
        __m256i p=_mm256_maddubs_epi16(q,v);
        if constexpr(TY==11)p=_mm256_sub_epi16(p,p4[(t*kMaxBlocks+b)*8+H]);
        sums[t]=_mm256_add_epi32(sums[t],_mm256_madd_epi16(p,sc));
    }
}
template<int TY,int NT,int... H> inline void chunks(std::integer_sequence<int,H...>,const __m256i& q0,const __m256i& q1,const __m256i& hm,
                                                    const __m256i& sA,const __m256i& sB,const __m256i* masks,const int8_t* const* y,
                                                    const __m256i* p4,int b,__m256i* sums) {
    (chunk<TY,NT,H>(q0,q1,hm,sA,sB,masks,y,p4,b,sums),...);
}
// Per 32-value chunk: unpack 2-bit (and Q3_K high-bit) codes once, broadcast the two 16-value scales with one byte
// shuffle, then one maddubs + madd per token. Q3_K codes are kept unsigned (q + 4) and the 4 * y term is subtracted
// in int16 before the scale, so every lane's integer sum and the float order match the previous kernel exactly.
template<int TY,int NT> void dot(const uint8_t* w,int blocks,const void* const* acts,float* result,const __m256i* p4,int pf) {
    constexpr size_t stride=TY==10?sizeof(block_q2_K):sizeof(block_q3_K);
    const __m256i masks[4]={scale_mask(0),scale_mask(1),scale_mask(2),scale_mask(3)};
    __m256 acc[NT];for(auto& a:acc)a=_mm256_setzero_ps();
    for(int b=0;b<blocks;++b) {
        const auto* raw=w+b*stride;
        if(pf) {
            _mm_prefetch(reinterpret_cast<const char*>(raw)+pf,_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(raw)+pf+64,_MM_HINT_T0);
        }
        const uint8_t* qs;float d,dm=0;__m128i lo,hi;__m256i mins{},hm{};
        if constexpr(TY==10) {
            const auto& x=*reinterpret_cast<const block_q2_K*>(raw);qs=x.qs;ggml_half h[2];std::memcpy(h,raw+80,4);d=half(h[0]);dm=half(h[1]);
            const auto sc=_mm_loadu_si128(reinterpret_cast<const __m128i*>(x.scales));
            const auto s8=_mm_and_si128(sc,_mm_set1_epi8(15));
            lo=_mm_cvtepu8_epi16(s8);hi=_mm_cvtepu8_epi16(_mm_srli_si128(s8,8));
            mins=_mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc,4),_mm_set1_epi8(15)));
        } else {
            const auto& x=*reinterpret_cast<const block_q3_K*>(raw);qs=x.qs;d=half(x.d);
            uint32_t a[4];std::memcpy(a,x.scales,12);const uint32_t t=a[2],m=0x0f0f0f0f,u=0x03030303;
            a[2]=((a[0]>>4)&m)|(((t>>4)&u)<<4);a[3]=((a[1]>>4)&m)|(((t>>6)&u)<<4);
            a[0]=(a[0]&m)|((t&u)<<4);a[1]=(a[1]&m)|(((t>>2)&u)<<4);
            // Built from registers: storing a[] and reloading 16 bytes defeats store forwarding.
            const auto s8=_mm_sub_epi8(_mm_setr_epi32(int(a[0]),int(a[1]),int(a[2]),int(a[3])),_mm_set1_epi8(32));
            lo=_mm_cvtepi8_epi16(s8);hi=_mm_cvtepi8_epi16(_mm_srli_si128(s8,8));
            hm=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.hmask));
        }
        // Lane inserts, not vperm2i128 (8 uops on Zen 1).
        const __m256i sA=_mm256_inserti128_si256(_mm256_castsi128_si256(lo),lo,1),sB=_mm256_inserti128_si256(_mm256_castsi128_si256(hi),hi,1);
        const __m256i q0=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs)),q1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs+32));
        const int8_t* y[NT];
        for(int t=0;t<NT;++t)y[t]=static_cast<const block_q8_K*>(acts[t])[b].qs;
        __m256i sums[NT];for(auto& s:sums)s=_mm256_setzero_si256();
        chunks<TY,NT>(std::make_integer_sequence<int,8>{},q0,q1,hm,sA,sB,masks,y,p4,b,sums);
        for(int t=0;t<NT;++t) {
            const auto& yb=static_cast<const block_q8_K*>(acts[t])[b];
            if constexpr(TY==10) {
                const auto offset=_mm256_madd_epi16(mins,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(yb.bsums)));
                acc[t]=_mm256_fmadd_ps(_mm256_set1_ps(-dm*yb.d),_mm256_cvtepi32_ps(offset),acc[t]);
            }
            acc[t]=_mm256_fmadd_ps(_mm256_set1_ps(d*yb.d),_mm256_cvtepi32_ps(sums[t]),acc[t]);
        }
    }
    for(int t=0;t<NT;++t)result[t]=sum(acc[t]);
}
template<int TY,int NT> void rows(const uint8_t* w,const uint8_t* up,size_t stride,int n,const void* const* act,float* const* out,int first,int last,float clamp,bool canon,float skip) {
    // Q3_K: 4 * (pair sums of activations) per 32-value chunk, so (q + 4) * y - 4 * y stays exact in int16.
    alignas(32) __m256i p4[TY==11?NT*kMaxBlocks*8:1];
    if constexpr(TY==11) {
        const __m256i four=_mm256_set1_epi8(4);
        for(int t=0;t<NT;++t)
            for(int b=0;b<n/256;++b)
                for(int h=0;h<8;++h)
                    p4[(t*kMaxBlocks+b)*8+h]=_mm256_maddubs_epi16(four,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const block_q8_K*>(act[t])[b].qs+32*h)));
    }
    const int pf=prefetch_distance();
    for(int r=first;r<last;++r) {
        float a[NT],u[NT];dot<TY,NT>(w+size_t(r)*stride,n/256,act,a,p4,pf);
        // Gate skipping (canonical, opt-in): when every token's unit is below the threshold its up row is not read.
        bool need_up=up!=nullptr;
        if(need_up&&canon&&skip>0) {need_up=false;for(int t=0;t<NT;++t){need_up=need_up||!canon_gate_skipped(a[t],clamp,skip);u[t]=0;}}
        if(need_up)dot<TY,NT>(up+size_t(r)*stride,n/256,act,u,p4,pf);
        for(int t=0;t<NT;++t) {
            if(up&&canon)out[t][r]=canon_swiglu(a[t],u[t],clamp,skip);
            else if(up) {float g=clamp>0?std::fmin(a[t],clamp):a[t],v=clamp>0?std::fmax(-clamp,std::fmin(u[t],clamp)):u[t];out[t][r]=g/(1.f+std::exp(-g))*v;}
            else out[t][r]=a[t];
        }
    }
}
template<int TY> void dispatch(const uint8_t* w,const uint8_t* up,size_t stride,int n,const void* const* act,int nt,float* const* out,int first,int last,float clamp,bool canon,float skip) {
#define CASE(N) case N:rows<TY,N>(w,up,stride,n,act,out,first,last,clamp,canon,skip);break
    switch(nt){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);default:throw std::invalid_argument("Q23: token width must be 1..8");}
#undef CASE
}
void run(int type,const uint8_t* w,const uint8_t* up,size_t stride,int n,const void* const* act,int nt,float* const* out,int first,int last,float clamp,bool canon,float skip) {
    if(n<=0||n%256||first<0||last<first)throw std::invalid_argument("Q23: invalid row geometry");
    if(type==11&&n/256>kMaxBlocks)throw std::invalid_argument("Q23: Q3_K rows longer than 4096 values");
    if(type==10)dispatch<10>(w,up,stride,n,act,nt,out,first,last,clamp,canon,skip);
    else if(type==11)dispatch<11>(w,up,stride,n,act,nt,out,first,last,clamp,canon,skip);
    else throw std::invalid_argument("Q23: unsupported weight type");
}
}
void q23_rows(int type,const uint8_t* w,size_t stride,int n,const void* const* act,int nt,float* const* out,int first,int last){run(type,w,nullptr,stride,n,act,nt,out,first,last,0,false,0);}
void q23_gu_rows(int type,const uint8_t* w,const uint8_t* up,size_t stride,int n,const void* const* act,int nt,float* const* out,int first,int last,float clamp,bool canon,float gate_skip){run(type,w,up,stride,n,act,nt,out,first,last,clamp,canon,gate_skip);}
}
