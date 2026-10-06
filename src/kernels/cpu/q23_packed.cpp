#include "strata/kernels/cpu/q23_avx2.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <array>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace strata::kernels::cpu {
namespace {
size_t stride(int type){if(type==10)return 84;if(type==11)return 110;throw std::invalid_argument("packed Q23 type");}
void scales3(const uint8_t* block,uint8_t* out) {
    uint32_t a[4];std::memcpy(a,block+96,12);const uint32_t t=a[2],m=0x0f0f0f0f,u=0x03030303;
    a[2]=((a[0]>>4)&m)|(((t>>4)&u)<<4);a[3]=((a[1]>>4)&m)|(((t>>6)&u)<<4);
    a[0]=(a[0]&m)|((t&u)<<4);a[1]=(a[1]&m)|(((t>>2)&u)<<4);std::memcpy(out,a,16);
}
void transform(int type,const uint8_t* src,uint8_t* dst,int n,int rows,bool inverse) {
    if(n<=0||n%256||rows<=0||rows%32)throw std::invalid_argument("packed Q23 requires complete 32x256 tiles");
    const int bits=type==10?2:3,meta=type==10?640:448,nb=n/256;const size_t bytes=stride(type);
    std::memset(dst,0,size_t(rows)*nb*bytes);
    for(int r=0;r<rows;r+=32)for(int b=0;b<nb;++b)for(int lane=0;lane<32;++lane) {
        const size_t ro=(size_t(r+lane)*nb+b)*bytes,to=(size_t(r/32)*nb+b)*32*bytes;
        const uint8_t* original=inverse?nullptr:src+ro;uint8_t* restored=inverse?dst+ro:nullptr;
        const uint8_t* tile=inverse?src+to:nullptr;uint8_t* packed=inverse?nullptr:dst+to;
        if(!inverse) {
            if(type==10) {for(int g=0;g<16;++g)packed[g*32+lane]=original[g];std::memcpy(packed+512+2*lane,original+80,2);std::memcpy(packed+576+2*lane,original+82,2);}
            else {
                uint8_t sc[16];scales3(original,sc);
                for(int g=0;g<16;++g){packed[(g%8)*32+lane]|=(sc[g]&15)<<(g/8*4);packed[256+(g%4)*32+lane]|=(sc[g]>>4)<<(g/4*2);}
                std::memcpy(packed+384+2*lane,original+108,2);
            }
        } else {
            if(type==10) {for(int g=0;g<16;++g)restored[g]=tile[g*32+lane];std::memcpy(restored+80,tile+512+2*lane,2);std::memcpy(restored+82,tile+576+2*lane,2);}
            else {
                uint8_t sc[16];for(int g=0;g<16;++g)sc[g]=((tile[(g%8)*32+lane]>>(g/8*4))&15)|(((tile[256+(g%4)*32+lane]>>(g/4*2))&3)<<4);
                for(int g=0;g<8;++g)restored[96+g]=(sc[g]&15)|((sc[g+8]&15)<<4);
                for(int g=0;g<4;++g)restored[104+g]=(sc[g]>>4)|((sc[g+4]>>4)<<2)|((sc[g+8]>>4)<<4)|((sc[g+12]>>4)<<6);
                std::memcpy(restored+108,tile+384+2*lane,2);
            }
        }
        for(int k=0;k<256;++k) {
            int q=0;
            if(!inverse) {
                q=(original[(type==10?16:32)+(k/128)*32+k%32]>>(2*((k%128)/32)))&3;
                if(type==11)q|=((original[k%32]>>(k/32))&1)<<2;
            }
            for(int bit=0;bit<bits;++bit) {
                const int offset=meta+(bit*64+k/4)*16+lane/2,shift=(lane%2)*4+k%4;
                if(inverse)q|=((tile[offset]>>shift)&1)<<bit;
                else packed[offset]|=((q>>bit)&1)<<shift;
            }
            if(inverse) {
                restored[(type==10?16:32)+(k/128)*32+k%32]|=(q&3)<<(2*((k%128)/32));
                if(type==11)restored[k%32]|=((q>>2)&1)<<(k/32);
            }
        }
    }
}
struct alignas(16) Lut {uint8_t lo[16],hi[16];};
__m256i lookup(const Lut& lut,__m128i idx) {
    auto a=_mm_shuffle_epi8(_mm_load_si128(reinterpret_cast<const __m128i*>(lut.lo)),idx);
    auto b=_mm_shuffle_epi8(_mm_load_si128(reinterpret_cast<const __m128i*>(lut.hi)),idx);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(a,b)),_mm_unpackhi_epi8(a,b),1);
}
__m256i direct(const int8_t* a,__m128i idx) {
    auto v=_mm256_cvtepu8_epi16(idx),sum=_mm256_setzero_si256();
    for(int k=0;k<4;++k) {
        auto mask=_mm256_cmpeq_epi16(_mm256_and_si256(v,_mm256_set1_epi16(1<<k)),_mm256_set1_epi16(1<<k));
        sum=_mm256_add_epi16(sum,_mm256_and_si256(mask,_mm256_set1_epi16(a[k])));
    }
    return sum;
}
__m256i scale32(__m128i x,bool sign){return sign?_mm256_cvtepi8_epi32(x):_mm256_cvtepu8_epi32(x);}
}
void q23_pack(int type,const uint8_t* src,uint8_t* dst,int n,int rows){transform(type,src,dst,n,rows,false);}
void q23_unpack(int type,const uint8_t* src,uint8_t* dst,int n,int rows){transform(type,src,dst,n,rows,true);}
void q23_packed_rows(int type,const uint8_t* weights,int n,const void* const* acts,int nt,float* const* out,int first,int last,bool use_lut) {
    if(n<=0||n%256||nt<1||nt>8||first<0||last<first)throw std::invalid_argument("packed Q23 geometry");
    const int nb=n/256,bits=type==10?2:3,meta=type==10?640:448;const size_t bytes=stride(type);
    // Thread-local capacity avoids allocation in the measured steady-state loop.
    thread_local std::vector<Lut> tables;
    if(use_lut) {
        tables.resize(size_t(nt)*nb*64);
        for(int t=0;t<nt;++t)for(int b=0;b<nb;++b)for(int g=0;g<64;++g) {
            const auto* a=static_cast<const block_q8_K*>(acts[t])[b].qs+g*4;auto& lut=tables[(t*nb+b)*64+g];
            for(int index=0;index<16;++index) {int value=0;for(int k=0;k<4;++k)if(index&(1<<k))value+=a[k];lut.lo[index]=uint8_t(value);lut.hi[index]=uint8_t(uint16_t(value)>>8);}
        }
    }
    for(int row=first/32*32;row<last;row+=32) {
        __m256 accum[8][4];for(int t=0;t<nt;++t)for(auto& a:accum[t])a=_mm256_setzero_ps();
        for(int b=0;b<nb;++b) {
            const auto* tile=weights+(size_t(row/32)*nb+b)*32*bytes;
            __m256i dots[8][4],mins[8][4];for(int t=0;t<nt;++t)for(int j=0;j<4;++j)dots[t][j]=mins[t][j]=_mm256_setzero_si256();
            for(int group=0;group<16;++group) {
                __m256i values[8][2];for(int t=0;t<nt;++t)values[t][0]=values[t][1]=_mm256_setzero_si256();
                for(int k=0;k<4;++k)for(int bit=0;bit<bits;++bit) {
                    auto index=_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile+meta+(bit*64+group*4+k)*16));
                    auto low=_mm_and_si128(index,_mm_set1_epi8(15)),high=_mm_and_si128(_mm_srli_epi16(index,4),_mm_set1_epi8(15));
                    __m128i idx[2]={_mm_unpacklo_epi8(low,high),_mm_unpackhi_epi8(low,high)};
                    for(int t=0;t<nt;++t)for(int half=0;half<2;++half) {
                        auto v=use_lut?lookup(tables[(t*nb+b)*64+group*4+k],idx[half]):direct(static_cast<const block_q8_K*>(acts[t])[b].qs+group*16+k*4,idx[half]);
                        values[t][half]=_mm256_add_epi16(values[t][half],_mm256_sll_epi16(v,_mm_cvtsi32_si128(bit)));
                    }
                }
                __m256i scale,min{};
                if(type==10) {const auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile+group*32));scale=_mm256_and_si256(v,_mm256_set1_epi8(15));min=_mm256_and_si256(_mm256_srli_epi16(v,4),_mm256_set1_epi8(15));}
                else {
                    const auto lo=_mm256_and_si256(_mm256_srl_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile+(group%8)*32)),_mm_cvtsi32_si128(group/8*4)),_mm256_set1_epi8(15));
                    const auto hi=_mm256_and_si256(_mm256_srl_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile+256+(group%4)*32)),_mm_cvtsi32_si128(group/4*2)),_mm256_set1_epi8(3));
                    scale=_mm256_sub_epi8(_mm256_or_si256(lo,_mm256_slli_epi16(hi,4)),_mm256_set1_epi8(32));
                }
                for(int t=0;t<nt;++t) {
                    const auto& a=static_cast<const block_q8_K*>(acts[t])[b];
                    if(type==11)for(auto& v:values[t])v=_mm256_sub_epi16(v,_mm256_set1_epi16(4*a.bsums[group]));
                    for(int j=0;j<4;++j) {
                        const auto s=j<2?_mm256_castsi256_si128(scale):_mm256_extracti128_si256(scale,1);
                        const auto v=j%2?_mm256_extracti128_si256(values[t][j/2],1):_mm256_castsi256_si128(values[t][j/2]);
                        dots[t][j]=_mm256_add_epi32(dots[t][j],_mm256_mullo_epi32(_mm256_cvtepi16_epi32(v),scale32(j%2?_mm_srli_si128(s,8):s,type==11)));
                        if(type==10) {auto m=j<2?_mm256_castsi256_si128(min):_mm256_extracti128_si256(min,1);mins[t][j]=_mm256_add_epi32(mins[t][j],_mm256_mullo_epi32(scale32(j%2?_mm_srli_si128(m,8):m,false),_mm256_set1_epi32(a.bsums[group])));}
                    }
                }
            }
            for(int j=0;j<4;++j) {
                const auto d=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile+(type==10?512:384)+16*j)));
                for(int t=0;t<nt;++t) {
                    auto ad=_mm256_set1_ps(static_cast<const block_q8_K*>(acts[t])[b].d);
                    if(type==10) {auto dm=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile+576+16*j)));accum[t][j]=_mm256_fnmadd_ps(_mm256_mul_ps(dm,ad),_mm256_cvtepi32_ps(mins[t][j]),accum[t][j]);}
                    accum[t][j]=_mm256_fmadd_ps(_mm256_mul_ps(d,ad),_mm256_cvtepi32_ps(dots[t][j]),accum[t][j]);
                }
            }
        }
        for(int t=0;t<nt;++t) {alignas(32) float values[32];for(int j=0;j<4;++j)_mm256_store_ps(values+j*8,accum[t][j]);for(int r=std::max(first,row);r<std::min(last,row+32);++r)out[t][r]=values[r-row];}
    }
}
}
