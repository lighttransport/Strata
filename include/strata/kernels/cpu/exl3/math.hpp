#pragma once
// EXL3 format equations adapted from Turboderp (2025), MIT: third_party/exl3/LICENSE.
// Reference: ExLlamaV3 c5d9c657966ffeeaa9353f0cc899f18629da4a13 and clair exl3_a64fx.c.
#include <cstdint>
#include <bit>
#include <cmath>
#include <array>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
namespace strata::cpu::exl3 {

inline float half_value(uint16_t bits) {
    unsigned sign=bits>>15, exp=(bits>>10)&31, frac=bits&1023;
    float v;
    if(exp==0) v=frac ? std::ldexp((float)frac,-24) : 0.0f;
    else if(exp==31) v=frac ? NAN : INFINITY;
    else v=std::ldexp(1.0f+(float)frac/1024.0f,(int)exp-15);
    return sign ? -v : v;
}
inline uint16_t half_round(float value) {
    uint32_t bits=std::bit_cast<uint32_t>(value);
    unsigned sign=(bits>>16)&0x8000u, exp=(bits>>23)&255u, frac=bits&0x7fffffu;
    if(exp==255)return (uint16_t)(sign|0x7c00u|(frac?0x200u:0));
    int e=(int)exp-127+15;
    if(e<=0) {
        if(e<-10)return (uint16_t)sign;
        frac|=0x800000u;
        unsigned shift=(unsigned)(14-e), h=frac>>shift;
        if((frac>>(shift-1)&1u) && ((frac&((1u<<(shift-1))-1)) || (h&1)))++h;
        return (uint16_t)(sign|h);
    }
    if(e>=31)return (uint16_t)(sign|0x7c00u);
    unsigned h=frac>>13;
    if((frac&0x1000u) && ((frac&0xfffu) || (h&1)))++h;
    if(h==0x400u){h=0;e++;if(e>=31)return (uint16_t)(sign|0x7c00u);}
    return (uint16_t)(sign|((unsigned)e<<10)|h);
}
inline float codebook(uint16_t state, unsigned cb) {
    uint32_t p;
    if (cb == 0) {
        p=((uint32_t)state*0xcbac1fedu & 0x8fff8fffu)^0x3b603b60u;
        return half_value(half_round(half_value((uint16_t)p)+half_value((uint16_t)(p>>16))));
    }
    if (cb == 1) {
        p=(uint32_t)state*0x83dcd12du;
        unsigned sum=(p&255)+((p>>8)&255)+((p>>16)&255)+(p>>24);
        return half_value(half_round(fmaf(half_value((uint16_t)(0x6400+sum)),
                                     half_value(0x1eee),half_value(0xc931))));
    }
    return NAN;
}
inline constexpr uint8_t exl3_offsets[256] = {
    0, 32, 64, 96, 128, 160, 192, 224, 4, 36, 68, 100, 132, 164, 196, 228,
    1, 33, 65, 97, 129, 161, 193, 225, 5, 37, 69, 101, 133, 165, 197, 229,
    8, 40, 72, 104, 136, 168, 200, 232, 12, 44, 76, 108, 140, 172, 204, 236,
    9, 41, 73, 105, 137, 169, 201, 233, 13, 45, 77, 109, 141, 173, 205, 237,
    16, 48, 80, 112, 144, 176, 208, 240, 20, 52, 84, 116, 148, 180, 212, 244,
    17, 49, 81, 113, 145, 177, 209, 241, 21, 53, 85, 117, 149, 181, 213, 245,
    24, 56, 88, 120, 152, 184, 216, 248, 28, 60, 92, 124, 156, 188, 220, 252,
    25, 57, 89, 121, 153, 185, 217, 249, 29, 61, 93, 125, 157, 189, 221, 253,
    2, 34, 66, 98, 130, 162, 194, 226, 6, 38, 70, 102, 134, 166, 198, 230,
    3, 35, 67, 99, 131, 163, 195, 227, 7, 39, 71, 103, 135, 167, 199, 231,
    10, 42, 74, 106, 138, 170, 202, 234, 14, 46, 78, 110, 142, 174, 206, 238,
    11, 43, 75, 107, 139, 171, 203, 235, 15, 47, 79, 111, 143, 175, 207, 239,
    18, 50, 82, 114, 146, 178, 210, 242, 22, 54, 86, 118, 150, 182, 214, 246,
    19, 51, 83, 115, 147, 179, 211, 243, 23, 55, 87, 119, 151, 183, 215, 247,
    26, 58, 90, 122, 154, 186, 218, 250, 30, 62, 94, 126, 158, 190, 222, 254,
    27, 59, 91, 123, 155, 187, 219, 251, 31, 63, 95, 127, 159, 191, 223, 255
};

inline uint32_t word(const uint8_t* p,unsigned i) {
    p+=4*i; return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
inline uint16_t state(const uint8_t* p,unsigned t) {
    unsigned end=(t+257)*2,start=end-16,shift=31-((end-1)&31);
    return uint16_t(((uint64_t(word(p,(start/32)%16))<<32)|word(p,((end-1)/32)%16))>>shift);
}
inline const std::array<float,65536>& lookup() {
    static const auto table=[] { std::array<float,65536> t{}; for(unsigned i=0;i<t.size();++i)t[i]=codebook(i,0); return t; }();
    return table;
}
inline void tile(const uint8_t* p,float* out) {
    const auto& t=lookup(); for(unsigned i=0;i<256;++i)out[i]=t[state(p,exl3_offsets[i])];
}
// Decode eight output columns together; preserve the scalar input accumulation order.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) inline void tile_dot_avx2(const uint8_t* p,const float* x,float* y) {
    const float* book=lookup().data();
    auto lo=_mm256_loadu_ps(y),hi=_mm256_loadu_ps(y+8);
    const auto bits=_mm256_set1_epi32(65535);
    uint32_t words[16];for(unsigned i=0;i<16;++i)words[i]=word(p,i);
    const auto even=_mm256_setr_epi32(words[0],words[2],words[4],words[6],words[8],words[10],words[12],words[14]);
    const auto odd=_mm256_setr_epi32(words[1],words[3],words[5],words[7],words[9],words[11],words[13],words[15]);
    const auto previous=_mm256_permutevar8x32_epi32(odd,_mm256_setr_epi32(7,0,1,2,3,4,5,6));
    for(unsigned r=0;r<16;++r)for(unsigned half=0;half<2;++half) {
        unsigned base=exl3_offsets[r*16+half*8],end=(base+257)*2;
        unsigned ai=((end-16)/32)%16,bi=((end-1)/32)%16;
        unsigned shift=31-((end-1)&31);
        auto a=ai==15?previous:(ai==0?even:odd);
        auto b=bi==0?even:odd;
        auto state=_mm256_and_si256(_mm256_or_si256(_mm256_sllv_epi32(a,_mm256_set1_epi32(32-shift)),_mm256_srlv_epi32(b,_mm256_set1_epi32(shift))),bits);
        auto w=_mm256_i32gather_ps(book,state,4);
        auto product=_mm256_mul_ps(_mm256_set1_ps(x[r]),w);
        if(half)hi=_mm256_add_ps(hi,product);else lo=_mm256_add_ps(lo,product);
    }
    _mm256_storeu_ps(y,lo);_mm256_storeu_ps(y+8,hi);
}
#endif
inline void tile_dot(const uint8_t* p,const float* x,float* y) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool vectorized=__builtin_cpu_supports("avx2") && !std::getenv("STRATA_EXL3_SCALAR");
    if(vectorized){tile_dot_avx2(p,x,y);return;}
#endif
    float w[256];tile(p,w);
    for(unsigned r=0;r<16;++r)for(unsigned c=0;c<16;++c)y[c]+=x[r]*w[r*16+c];
}
// BF16 dense decode can stream host weights directly instead of uploading the
// entire layer matrix for a single vector. Eight independent FP32 accumulators.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) inline float bf16_dot_avx2(const uint16_t* w,const float* x,size_t n) {
    auto acc=_mm256_setzero_ps();
    for(size_t i=0;i<n;i+=8) {
        auto bits=_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+i))),16);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_castsi256_ps(bits),_mm256_loadu_ps(x+i)));
    }
    float lanes[8];_mm256_storeu_ps(lanes,acc);float sum=0;for(float v:lanes)sum+=v;return sum;
}
#endif
inline float bf16_dot(const uint16_t* w,const float* x,size_t n) {
    if(n%8)throw std::runtime_error("BF16: input dimension must be divisible by eight");
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(__builtin_cpu_supports("avx2"))return bf16_dot_avx2(w,x,n);
#endif
    float lanes[8]={};for(size_t i=0;i<n;++i)lanes[i%8]+=std::bit_cast<float>(uint32_t(w[i])<<16)*x[i];
    float sum=0;for(float v:lanes)sum+=v;return sum;
}
// Orthonormal Walsh-Hadamard: the upstream had_r_128 wrapper applies 1/sqrt(128).
inline void hadamard(float* x,size_t n) {
    if(n%128)throw std::runtime_error("EXL3: dimensions must be divisible by 128");
    for(size_t base=0;base<n;base+=128)for(size_t step=1;step<128;step*=2)
        for(size_t i=0;i<128;i+=step*2)for(size_t j=0;j<step;++j) {
            float a=x[base+i+j],b=x[base+i+j+step]; x[base+i+j]=a+b;x[base+i+j+step]=a-b;
        }
    for(size_t i=0;i<n;++i)x[i]*=0.088388347648f;
}
// Source trellis order [input/16, output/16, 32 int16]. FP32 accumulation.
inline void linear(const uint8_t* trellis,const uint16_t* suh,const uint16_t* svh,
                   size_t in,size_t out,const float* x,float* y) {
    if(!in||!out||in%128||out%128)throw std::runtime_error("EXL3: invalid linear geometry");
    std::vector<float> h(in);for(size_t i=0;i<in;++i)h[i]=x[i]*half_value(suh[i]);hadamard(h.data(),in);
    std::fill(y,y+out,0.f);float w[256];
    for(size_t it=0;it<in/16;++it)for(size_t ot=0;ot<out/16;++ot) {
        tile(trellis+(it*(out/16)+ot)*64,w);
        for(size_t r=0;r<16;++r)for(size_t c=0;c<16;++c)y[ot*16+c]+=h[it*16+r]*w[r*16+c];
    }
    hadamard(y,out);for(size_t i=0;i<out;++i)y[i]*=half_value(svh[i]);
}
} // namespace strata::cpu::exl3
