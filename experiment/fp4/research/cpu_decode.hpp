#pragma once
#include "strata/kernels/cpu/exl3/math.hpp"
namespace research {
inline __m256i parity8(__m256i v) {
    v=_mm256_xor_si256(v,_mm256_srli_epi32(v,8));
    v=_mm256_xor_si256(v,_mm256_srli_epi32(v,4));
    v=_mm256_xor_si256(v,_mm256_srli_epi32(v,2));
    v=_mm256_xor_si256(v,_mm256_srli_epi32(v,1));
    return _mm256_and_si256(v,_mm256_set1_epi32(1));
}
inline __m256 xor_decode8(__m256i s) {
    auto h=_mm256_srli_epi32(s,2);
    auto bit=[&](int mask){return parity8(_mm256_and_si256(h,_mm256_set1_epi32(mask)));};
    auto p=_mm256_or_si256(bit(0x1089),_mm256_slli_epi32(bit(0x2112),1));
    auto q=_mm256_or_si256(bit(0x0a24),_mm256_slli_epi32(bit(0x2441),1));
    auto index=_mm256_or_si256(_mm256_slli_epi32(p,2),_mm256_xor_si256(q,_mm256_and_si256(s,_mm256_set1_epi32(3))));
    // Twice each palette value is an exact signed byte. PSHUFB decodes eight states.
    const auto lut=_mm256_broadcastsi128_si256(_mm_setr_epi8(-12,-3,0,4,-8,-2,1,6,-6,-1,2,8,-4,0,3,12));
    auto v=_mm256_shuffle_epi8(lut,index);
    v=_mm256_srai_epi32(_mm256_slli_epi32(v,24),24);
    return _mm256_mul_ps(_mm256_cvtepi32_ps(v),_mm256_set1_ps(.5f));
}
inline const std::array<float,65536>& mul1_lookup() {
    static const auto table=[] {std::array<float,65536> t{};for(unsigned s=0;s<65536;++s)t[s]=strata::cpu::exl3::codebook(s,1);return t;}();
    return table;
}
// Eight output columns per vector, with direct extraction from packed tile words.
inline void tile_dot_simd(const uint8_t* p,const float* x,float* y,int kind) {
    const float* book=kind==0?strata::cpu::exl3::lookup().data():kind==1?mul1_lookup().data():nullptr;
    auto lo=_mm256_loadu_ps(y),hi=_mm256_loadu_ps(y+8);
    uint32_t words[16];for(unsigned i=0;i<16;++i)words[i]=strata::cpu::exl3::word(p,i);
    auto even=_mm256_setr_epi32(words[0],words[2],words[4],words[6],words[8],words[10],words[12],words[14]);
    auto odd=_mm256_setr_epi32(words[1],words[3],words[5],words[7],words[9],words[11],words[13],words[15]);
    auto previous=_mm256_permutevar8x32_epi32(odd,_mm256_setr_epi32(7,0,1,2,3,4,5,6));
    for(unsigned r=0;r<16;++r)for(unsigned half=0;half<2;++half){
        unsigned base=strata::cpu::exl3::exl3_offsets[r*16+half*8],end=(base+257)*2;
        unsigned ai=((end-16)/32)%16,bi=((end-1)/32)%16,shift=31-((end-1)&31);
        auto a=ai==15?previous:ai==0?even:odd,b=bi==0?even:odd;
        auto s=_mm256_and_si256(_mm256_or_si256(_mm256_sllv_epi32(a,_mm256_set1_epi32(32-shift)),
                                               _mm256_srlv_epi32(b,_mm256_set1_epi32(shift))),_mm256_set1_epi32(65535));
        auto w=kind==2?xor_decode8(s):_mm256_i32gather_ps(book,s,4);
        if(half)hi=_mm256_fmadd_ps(_mm256_set1_ps(x[r]),w,hi);else lo=_mm256_fmadd_ps(_mm256_set1_ps(x[r]),w,lo);
    }
    _mm256_storeu_ps(y,lo);_mm256_storeu_ps(y+8,hi);
}
} // namespace research
