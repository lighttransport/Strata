#include "strata/kernels/cpu/fp4_local_q8.hpp"
#include <cmath>
#include <bit>
#include <cstring>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <limits>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define LOCAL_AVX2 1
#endif
namespace strata::kernels::cpu {
namespace {
struct ScaleTable {
    float half[127];
    ScaleTable() {for(int s=0;s<127;++s)half[s]=(s>>3)?std::ldexp(1.f+(s&7)*.125f,(s>>3)-8):(s&7)/1024.f;}
};
const ScaleTable& table(){static const ScaleTable t;return t;}
constexpr int mag[8]={0,1,2,3,4,6,8,12};
void scalar(const uint8_t* w,const uint8_t* s,const int8_t* x,const float* xs,
            float global,int K,int N,int M,int first,int last,float* y) {
    const auto& st=table();
    for(int r=first;r<last;++r)for(int t=0;t<M;++t) {
        float sum=0;
        for(int j=0;j<K;j+=16) {
            int dot=0;
            for(int k=0;k<16;++k) {
                unsigned q=(w[size_t(r)*(K/2)+(j+k)/2]>>(((j+k)%2)*4))&15;
                dot+=(q&8?-mag[q&7]:mag[q&7])*int(x[size_t(t)*K+j+k]);
            }
            float factor=st.half[s[size_t(r)*(K/16)+j/16]]*xs[size_t(t)*(K/64)+j/64];
            sum=std::fma(float(dot),factor,sum);
        }
        y[size_t(t)*N+r]=sum*global;
    }
}
#ifdef LOCAL_AVX2
template<int M>
__attribute__((target("avx2,fma")))
void vector_rows(const uint8_t* w,const uint8_t* s,const int8_t* x,const float* xs,const int16_t* sums,
                 float global,int K,int N,int first,int last,float* y) {
    const auto& st=table();
    // Unsigned offset codebook avoids negating -128 activations. Pair sums
    // are bounded by 2*24*128=6144, so maddubs never saturates.
    const auto lut=_mm_setr_epi8(12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0);
    const auto mask=_mm_set1_epi8(15);const auto ones=_mm256_set1_epi16(1);
    for(int r=first;r<last;++r) {
        float acc[M]={};
        for(int j=0;j<K;j+=32) {
            const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+size_t(r)*(K/2)+j/2));
            const auto lo=_mm_and_si128(packed,mask),hi=_mm_and_si128(_mm_srli_epi16(packed,4),mask);
            const auto v0=_mm_shuffle_epi8(lut,_mm_unpacklo_epi8(lo,hi));
            const auto v1=_mm_shuffle_epi8(lut,_mm_unpackhi_epi8(lo,hi));
            const auto values=_mm256_inserti128_si256(_mm256_castsi128_si256(v0),v1,1);
            const float s0=st.half[s[size_t(r)*(K/16)+j/16]],s1=st.half[s[size_t(r)*(K/16)+j/16+1]];
            for(int t=0;t<M;++t) {
                auto dot=_mm256_madd_epi16(_mm256_maddubs_epi16(values,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+size_t(t)*K+j))),ones);
                dot=_mm256_hadd_epi32(dot,dot);dot=_mm256_hadd_epi32(dot,dot);
                const int d0=_mm_cvtsi128_si32(_mm256_castsi256_si128(dot))-12*int(sums[size_t(t)*(K/16)+j/16]);
                const int d1=_mm_cvtsi128_si32(_mm256_extracti128_si256(dot,1))-12*int(sums[size_t(t)*(K/16)+j/16+1]);
                const float as=xs[size_t(t)*(K/64)+j/64];
                acc[t]=std::fma(float(d0),s0*as,acc[t]);
                acc[t]=std::fma(float(d1),s1*as,acc[t]);
            }
        }
        for(int t=0;t<M;++t)y[size_t(t)*N+r]=acc[t]*global;
    }
}
// Opt-in tree reduction: preserve integer products, accumulate eight FP32
// lanes independently, then reduce once per row. This changes float order.
template<int M,int R=1,bool Planar=false>
__attribute__((target("avx2,fma")))
void vector_tree(const uint8_t* w,const uint8_t* s,const int8_t* x,const float* xs,
                 float global,int K,int N,int first,int last,float* y) {
    const auto& st=table();
    const auto lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const auto mask=_mm_set1_epi8(15);const auto ones=_mm256_set1_epi16(1);
    for(int base=first;base<last;base+=R) {
        __m256 acc[R][M];for(auto& row:acc)for(auto& v:row)v=_mm256_setzero_ps();
        for(int j=0;j<K;j+=32) {
            __m256i acts[M],magnitudes[M];float ascales[M];
            for(int t=0;t<M;++t) {
                acts[t]=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+size_t(t)*K+j));
                magnitudes[t]=_mm256_abs_epi8(acts[t]);ascales[t]=xs[size_t(t)*(K/64)+j/64];
            }
            for(int rr=0;rr<R&&base+rr<last;++rr) {
            const int r=base+rr;
            const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+size_t(r)*(K/2)+j/2));
            const auto lo=_mm_and_si128(packed,mask),hi=_mm_and_si128(_mm_srli_epi16(packed,4),mask);
            const auto v0=_mm_shuffle_epi8(lut,Planar?lo:_mm_unpacklo_epi8(lo,hi));
            const auto v1=_mm_shuffle_epi8(lut,Planar?hi:_mm_unpackhi_epi8(lo,hi));
            const auto values=_mm256_inserti128_si256(_mm256_castsi128_si256(v0),v1,1);
            const float s0=st.half[s[size_t(r)*(K/16)+j/16]],s1=st.half[s[size_t(r)*(K/16)+j/16+1]];
            for(int t=0;t<M;++t) {
                // Move activation signs to small weights, not weight signs to
                // activations: abs(-128) is legal as UNSIGNED 128 in maddubs.
                auto dot=_mm256_madd_epi16(_mm256_maddubs_epi16(magnitudes[t],_mm256_sign_epi8(values,acts[t])),ones);
                const float as=ascales[t];
                const auto factor=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(s0*as)),_mm_set1_ps(s1*as),1);
                acc[rr][t]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),factor,acc[rr][t]);
            }
        }
        }
        for(int rr=0;rr<R&&base+rr<last;++rr)for(int t=0;t<M;++t) {
            const int r=base+rr;
            auto v=_mm_add_ps(_mm256_castps256_ps128(acc[rr][t]),_mm256_extractf128_ps(acc[rr][t],1));
            v=_mm_hadd_ps(v,v);v=_mm_hadd_ps(v,v);
            y[size_t(t)*N+r]=_mm_cvtss_f32(v)*global;
        }
    }
}

// Exact half-scale cache, batch one. Convert two scale values at a time;
// multiplication and FMA order match the existing K32 tree loop.
template<int R>
__attribute__((target("avx2,fma,f16c")))
void vector_half(const uint8_t* w,const uint16_t* s,const int8_t* x,const float* xs,
                 float global,int K,int first,int last,float* y) {
    const auto lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const auto mask=_mm_set1_epi8(15),ones128=_mm_set1_epi16(1);
    const auto ones=_mm256_broadcastsi128_si256(ones128);
    for(int base=first;base<last;base+=R) {
        __m256 acc[R];for(auto& a:acc)a=_mm256_setzero_ps();
        for(int j=0;j<K;j+=32) {
            const auto act=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+j));
            const auto magnitude=_mm256_abs_epi8(act);
            const auto as=_mm_set1_ps(xs[j/64]);
            for(int rr=0;rr<R&&base+rr<last;++rr) {
                const int r=base+rr;
                const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+size_t(r)*(K/2)+j/2));
                const auto lo=_mm_and_si128(packed,mask),hi=_mm_and_si128(_mm_srli_epi16(packed,4),mask);
                const auto v0=_mm_shuffle_epi8(lut,_mm_unpacklo_epi8(lo,hi));
                const auto v1=_mm_shuffle_epi8(lut,_mm_unpackhi_epi8(lo,hi));
                const auto values=_mm256_inserti128_si256(_mm256_castsi128_si256(v0),v1,1);
                uint32_t pair;std::memcpy(&pair,s+size_t(r)*(K/16)+j/16,4);
                const auto scales=_mm_mul_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(int(pair))),as);
                const auto factor=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_shuffle_ps(scales,scales,0)),_mm_shuffle_ps(scales,scales,0x55),1);
                const auto dot=_mm256_madd_epi16(_mm256_maddubs_epi16(magnitude,_mm256_sign_epi8(values,act)),ones);
                acc[rr]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),factor,acc[rr]);
            }
        }
        for(int rr=0;rr<R&&base+rr<last;++rr) {
            auto v=_mm_add_ps(_mm256_castps256_ps128(acc[rr]),_mm256_extractf128_ps(acc[rr],1));
            v=_mm_hadd_ps(v,v);v=_mm_hadd_ps(v,v);y[base+rr]=_mm_cvtss_f32(v)*global;
        }
    }
}

// K64 variant keeps two K16 groups in each 256-bit lane pair, avoiding
// cross-lane nibble assembly. Batch one only; reduction order is opt-in.
template<int R,bool Planar=false>
__attribute__((target("avx2,fma")))
void vector_wide(const uint8_t* w,const uint8_t* s,const int8_t* x,const float* xs,
                 float global,int K,int N,int first,int last,float* y) {
    (void)N;
    const auto& st=table();
    const auto lut=_mm256_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12,
                                   0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const auto mask=_mm256_set1_epi8(15),ones=_mm256_set1_epi16(1);
    for(int base=first;base<last;base+=R) {
        // Eight live accumulators spill on the four-row specialization. Use
        // four instead; this is still the explicit experimental tree path.
        __m256 acc[R][R==4?1:2];for(auto& row:acc)for(auto& a:row)a=_mm256_setzero_ps();
        for(int j=0;j<K;j+=64) {
            const auto x0=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+j));
            const auto x1=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+j+32));
            const __m256i acts[2]={_mm256_permute2x128_si256(x0,x1,0x20),_mm256_permute2x128_si256(x0,x1,0x31)};
            const __m256i absx[2]={_mm256_abs_epi8(acts[0]),_mm256_abs_epi8(acts[1])};
            const float as=xs[j/64];
            for(int rr=0;rr<R&&base+rr<last;++rr) {
                const int r=base+rr;
                const auto packed=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+size_t(r)*(K/2)+j/2));
                const auto lo=_mm256_and_si256(packed,mask),hi=_mm256_and_si256(_mm256_srli_epi16(packed,4),mask);
                const __m256i vals[2]={_mm256_shuffle_epi8(lut,Planar?lo:_mm256_unpacklo_epi8(lo,hi)),_mm256_shuffle_epi8(lut,Planar?hi:_mm256_unpackhi_epi8(lo,hi))};
                for(int g=0;g<2;++g) {
                    const auto dot=_mm256_madd_epi16(_mm256_maddubs_epi16(absx[g],_mm256_sign_epi8(vals[g],acts[g])),ones);
                    const float s0=st.half[s[size_t(r)*(K/16)+j/16+g]]*as;
                    const float s1=st.half[s[size_t(r)*(K/16)+j/16+g+2]]*as;
                    const auto factor=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(s0)),_mm_set1_ps(s1),1);
                    acc[rr][R==4?0:g]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),factor,acc[rr][R==4?0:g]);
                }
            }
        }
        for(int rr=0;rr<R&&base+rr<last;++rr) {
            const auto a=R==4?acc[rr][0]:_mm256_add_ps(acc[rr][0],acc[rr][1]);
            auto v=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
            v=_mm_hadd_ps(v,v);v=_mm_hadd_ps(v,v);y[base+rr]=_mm_cvtss_f32(v)*global;
        }
    }
}

#endif
float quant_scale(float maximum) {
    return maximum?std::max(maximum/127.f,std::numeric_limits<float>::min()):1.f;
}
bool quant_scalar(const float* x,int8_t* out,float* scale,int16_t* sums) {
    float maximum=0;
    for(int i=0;i<64;++i){if(!std::isfinite(x[i]))return false;maximum=std::max(maximum,std::abs(x[i]));}
    const float d=quant_scale(maximum),inverse=1.f/d;
    for(int g=0;g<4;++g) {
        int sum=0;
        for(int k=0;k<16;++k){int q=std::clamp(int(std::nearbyint(x[g*16+k]*inverse)),-127,127);out[g*16+k]=int8_t(q);sum+=q;}
        sums[g]=int16_t(sum);
    }
    *scale=d;return true;
}
#ifdef LOCAL_AVX2
__attribute__((target("avx2,fma")))
bool quant_vector(const float* x,int8_t* out,float* scale,int16_t* sums) {
    auto maximum=_mm256_setzero_ps();
    const auto sign=_mm256_set1_ps(-0.f),largest=_mm256_set1_ps(std::numeric_limits<float>::max());
    for(int j=0;j<64;j+=8) {
        const auto a=_mm256_andnot_ps(sign,_mm256_loadu_ps(x+j));
        if(_mm256_movemask_ps(_mm256_cmp_ps(a,largest,_CMP_NLE_UQ)))return false;
        maximum=_mm256_max_ps(maximum,a);
    }
    auto m=_mm_max_ps(_mm256_castps256_ps128(maximum),_mm256_extractf128_ps(maximum,1));
    m=_mm_max_ps(m,_mm_movehl_ps(m,m));m=_mm_max_ss(m,_mm_shuffle_ps(m,m,1));
    const float d=quant_scale(_mm_cvtss_f32(m));const auto inverse=_mm256_set1_ps(1.f/d);
    for(int j=0;j<64;j+=8) {
        const auto rounded=_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j),inverse),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        const auto q=_mm256_cvttps_epi32(rounded);
        const auto words=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out+j),_mm_packs_epi16(words,_mm_setzero_si128()));
    }
    for(int g=0;g<4;++g) {
        auto sum=_mm_madd_epi16(_mm_maddubs_epi16(_mm_set1_epi8(1),_mm_loadu_si128(reinterpret_cast<const __m128i*>(out+16*g))),_mm_set1_epi16(1));
        sum=_mm_hadd_epi32(sum,sum);sum=_mm_hadd_epi32(sum,sum);sums[g]=int16_t(_mm_cvtsi128_si32(sum));
    }
    *scale=d;return true;
}
#endif

}
bool fp4_local_q8_avx2_available() {
#ifdef LOCAL_AVX2
    return __builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma");
#else
    return false;
#endif
}
bool fp4_local_quantize_q8_64(const float* x,int8_t* out,float* scale,int16_t* sums,bool simd) {
    if(!x||!out||!scale||!sums)return false;
#ifdef LOCAL_AVX2
    if(simd&&fp4_local_q8_avx2_available())return quant_vector(x,out,scale,sums);
#else
    (void)simd;
#endif
    return quant_scalar(x,out,scale,sums);
}

uint16_t fp4_local_half_scale(uint8_t scale) {
    if(scale>126)throw std::invalid_argument("invalid finite nonnegative E4M3 scale");
    if(!scale)return 0;
    const uint32_t bits=std::bit_cast<uint32_t>(table().half[scale]);
    return uint16_t((((bits>>23)-112)<<10)|((bits>>13)&1023));
}
bool fp4_local_q8_half_available() {
#ifdef LOCAL_AVX2
    return fp4_local_q8_avx2_available()&&__builtin_cpu_supports("f16c");
#else
    return false;
#endif
}
void fp4_local_q8_half_rows(const uint8_t* w,const uint16_t* s,const int8_t* x,const float* xs,
    float global,int K,int N,int first,int last,float* y,int tile) {
    if(!w||!s||!x||!xs||!y||K<64||K%64||K>16384||N<1||N>16384||first<0||last<first||last>N||!std::isfinite(global)||global<=0||(tile!=1&&tile!=4))
        throw std::invalid_argument("invalid FP16 scale-cache extent");
#ifdef LOCAL_AVX2
    if(fp4_local_q8_half_available()) {
        if(tile==4)vector_half<4>(w,s,x,xs,global,K,first,last,y);
        else vector_half<1>(w,s,x,xs,global,K,first,last,y);
        return;
    }
#endif
    throw std::runtime_error("FP16 scale-cache kernel requires AVX2, FMA and F16C");
}

void fp4_local_q8_planar_rows(const uint8_t* w,const uint8_t* s,const int8_t* x,const float* xs,
    float global,int K,int N,int M,int first,int last,float* y,int tile,bool wide) {
    if(!w||!s||!x||!xs||!y||K<64||K%64||K>16384||N<1||N>16384||M<1||M>8||first<0||last<first||last>N||!std::isfinite(global)||global<=0||(tile!=1&&tile!=4)||((tile==4||wide)&&M!=1))
        throw std::invalid_argument("invalid planar FP4 extent");
#ifdef LOCAL_AVX2
    if(fp4_local_q8_avx2_available()) {
        if(wide) {
            if(tile==4)vector_wide<4,true>(w,s,x,xs,global,K,N,first,last,y);
            else vector_wide<1,true>(w,s,x,xs,global,K,N,first,last,y);
            return;
        }
        if(tile==4){vector_tree<1,4,true>(w,s,x,xs,global,K,N,first,last,y);return;}
        switch(M) {
#define PLANAR(T) case T:vector_tree<T,1,true>(w,s,x,xs,global,K,N,first,last,y);return;
            PLANAR(1) PLANAR(2) PLANAR(3) PLANAR(4) PLANAR(5) PLANAR(6) PLANAR(7) PLANAR(8)
#undef PLANAR
        }
    }
#endif
    throw std::runtime_error("planar FP4 kernel requires AVX2 and FMA");
}

void fp4_local_q8_rows(const uint8_t* w,const uint8_t* s,const int8_t* x,const float* xs,const int16_t* sums,
                      float global,int K,int N,int M,int first,int last,float* y,bool simd,bool tree_reduce,int row_tile,bool wide_k) {
    if(!w||!s||!x||!xs||!sums||!y||K<64||K%64||K>16384||N<1||N>16384||M<1||M>8||first<0||last<first||last>N||!std::isfinite(global)||global<=0)
        throw std::invalid_argument("invalid local FP4 Q8 extent");
    if((row_tile!=1&&row_tile!=2&&row_tile!=4)||(row_tile!=1&&(!tree_reduce||M!=1)))
        throw std::invalid_argument("row tiling requires batch-one tree reduction and tile 1/2/4");
    if(wide_k&&(!tree_reduce||M!=1))throw std::invalid_argument("wide K requires batch-one tree reduction");
    // Checkpoint scale validation occurs once in glm_load_fp4, outside hot loops.
#ifdef LOCAL_AVX2
    if(simd&&fp4_local_q8_avx2_available()) {
        if(wide_k) {
            if(row_tile==4)vector_wide<4>(w,s,x,xs,global,K,N,first,last,y);
            else if(row_tile==2)vector_wide<2>(w,s,x,xs,global,K,N,first,last,y);
            else vector_wide<1>(w,s,x,xs,global,K,N,first,last,y);
            return;
        }
        if(row_tile==2){vector_tree<1,2>(w,s,x,xs,global,K,N,first,last,y);return;}
        if(row_tile==4){vector_tree<1,4>(w,s,x,xs,global,K,N,first,last,y);return;}
        switch(M) {
#define CASE(T) case T:if(tree_reduce)vector_tree<T>(w,s,x,xs,global,K,N,first,last,y);else vector_rows<T>(w,s,x,xs,sums,global,K,N,first,last,y);return;
            CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
        }
    }
#else
    (void)simd;(void)tree_reduce;
#endif
    scalar(w,s,x,xs,global,K,N,M,first,last,y);
}
}
