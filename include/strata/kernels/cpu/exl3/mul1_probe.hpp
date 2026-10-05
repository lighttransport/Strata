#pragma once
// Performance probe only. This codebook cannot execute an MCG-quantized model.
// Integer formulation follows ExLlamaV3 moe_mul1.cpp (MIT; third_party/exl3/LICENSE).
#include "strata/kernels/cpu/exl3/math.hpp"
namespace strata::cpu::exl3 {
inline unsigned mul1_bytesum(uint16_t s) {
    uint32_t p=uint32_t(s)*0x83dcd12du;
    return (p&255)+((p>>8)&255)+((p>>16)&255)+(p>>24);
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
template<unsigned R,unsigned H>
__attribute__((target("avx2"))) inline __m256i mul1_codes(__m256i even,__m256i odd,__m256i previous) {
    constexpr unsigned base=exl3_offsets[R*16+H*8],end=(base+257)*2;
    constexpr unsigned ai=((end-16)/32)%16,bi=((end-1)/32)%16,shift=31-((end-1)&31);
    const auto a=ai==15?previous:(ai==0?even:odd),b=bi==0?even:odd;
    return _mm256_and_si256(_mm256_or_si256(_mm256_slli_epi32(a,32-shift),_mm256_srli_epi32(b,shift)),_mm256_set1_epi32(65535));
}
template<unsigned R=0>
__attribute__((target("avx2"))) inline void mul1_rows(__m256i even,__m256i odd,__m256i previous,const int8_t* x,__m256i& lo,__m256i& hi) {
    const auto mult=_mm256_set1_epi32(int32_t(0x83dcd12du)),ones=_mm256_set1_epi8(1),xx=_mm256_set1_epi16(x[R]);
    auto a=_mm256_maddubs_epi16(_mm256_mullo_epi32(mul1_codes<R,0>(even,odd,previous),mult),ones);
    auto b=_mm256_maddubs_epi16(_mm256_mullo_epi32(mul1_codes<R,1>(even,odd,previous),mult),ones);
    lo=_mm256_add_epi32(lo,_mm256_madd_epi16(a,xx));hi=_mm256_add_epi32(hi,_mm256_madd_epi16(b,xx));
    if constexpr(R<15)mul1_rows<R+1>(even,odd,previous,x,lo,hi);
}
__attribute__((target("avx2"))) inline void mul1_tile_dot_avx2(const uint8_t* p,const int8_t* x,int32_t* y) {
    uint32_t w[16];for(unsigned i=0;i<16;++i)w[i]=word(p,i);
    const auto even=_mm256_setr_epi32(w[0],w[2],w[4],w[6],w[8],w[10],w[12],w[14]);
    const auto odd=_mm256_setr_epi32(w[1],w[3],w[5],w[7],w[9],w[11],w[13],w[15]);
    const auto previous=_mm256_permutevar8x32_epi32(odd,_mm256_setr_epi32(7,0,1,2,3,4,5,6));
    auto lo=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y)),hi=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y+8));
    mul1_rows(even,odd,previous,x,lo,hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y),lo);_mm256_storeu_si256(reinterpret_cast<__m256i*>(y+8),hi);
}
#endif
inline void mul1_tile_dot(const uint8_t* p,const int8_t* x,int32_t* y) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(__builtin_cpu_supports("avx2")){mul1_tile_dot_avx2(p,x,y);return;}
#endif
    for(unsigned r=0;r<16;++r)for(unsigned c=0;c<16;++c)y[c]+=int32_t(mul1_bytesum(state(p,exl3_offsets[r*16+c])))*x[r];
}
}
