#include "strata/kernels/cpu/glm_fp4.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/host/glm_fp4.h"
#include <cmath>
#include <stdexcept>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
namespace strata::kernels::cpu {
namespace {
using strata::cpu::exl3::half_value;
using strata::cpu::exl3::half_round;
static void scale_table(float global,bool qpn,float* table) {
    const float g=qpn?half_value(half_round(global*16384.f)):global;
    for(int i=0;i<128;++i) {
        const float value=strata_fp4_e4m3(uint8_t(i))*g;
        table[i]=qpn?half_value(half_round(value))/16384.f:value;
    }
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
template<bool TABLE>
__attribute__((target("avx512f,avx512bw,fma,f16c")))
void rows512(const uint8_t* w,const uint8_t* scales,const float* table,const float* x,float* y,
              int K,int N,int M,int first,int last) {
    const __m512 lut=_mm512_setr_ps(0,.5f,1,1.5f,2,3,4,6,0,.5f,1,1.5f,2,3,4,6);
    const __m512i mask=_mm512_set1_epi32(7);
    const __m128i nibble=_mm_set1_epi8(15);
    for(int row=first;row<last;++row) {
        __m512 acc[8];for(int t=0;t<M;++t)acc[t]=_mm512_setzero_ps();
        const uint8_t* weights=w+size_t(row)*(K/2);
        const uint8_t* groups=scales+size_t(row)*(K/16);
        for(int j=0;j<K;j+=16) {
            const __m128i packed=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights+j/2));
            const __m128i q8=_mm_unpacklo_epi8(_mm_and_si128(packed,nibble),_mm_and_si128(_mm_srli_epi16(packed,4),nibble));
            const __m512i q=_mm512_cvtepu8_epi32(q8);
            const __m512 sign=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(q,_mm512_set1_epi32(8)),28));
            const __m512 values_lut=TABLE?_mm512_loadu_ps(table+size_t(groups[j/16])*8):lut;
            const __m512 mag=_mm512_permutexvar_ps(_mm512_and_si512(q,mask),values_lut);
            __m512 values=_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(mag),_mm512_castps_si512(sign)));
            if constexpr(!TABLE) {
                values=_mm512_mul_ps(values,_mm512_set1_ps(table[groups[j/16]]));
                values=_mm512_cvtph_ps(_mm512_cvtps_ph(values,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
            }
            for(int t=0;t<M;++t)acc[t]=_mm512_fmadd_ps(values,_mm512_loadu_ps(x+size_t(t)*K+j),acc[t]);
        }
        for(int t=0;t<M;++t)y[size_t(t)*N+row]=_mm512_reduce_add_ps(acc[t]);
    }
}
template<bool TABLE>
__attribute__((target("avx2,fma,f16c")))
void rows256(const uint8_t* w,const uint8_t* scales,const float* table,const float* x,float* y,
              int K,int N,int M,int first,int last) {
    const __m256 lut=_mm256_setr_ps(0,.5f,1,1.5f,2,3,4,6);
    for(int row=first;row<last;++row) {
        __m256 a[8],b[8];for(int t=0;t<M;++t)a[t]=b[t]=_mm256_setzero_ps();
        for(int j=0;j<K;j+=16) {
            const auto* src=w+size_t(row)*(K/2)+j/2;
            const __m128i packed=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
            const __m128i qs=_mm_unpacklo_epi8(_mm_and_si128(packed,_mm_set1_epi8(15)),_mm_and_si128(_mm_srli_epi16(packed,4),_mm_set1_epi8(15)));
            const unsigned scale=scales[size_t(row)*(K/16)+j/16];
            const __m256 values_lut=TABLE?_mm256_loadu_ps(table+size_t(scale)*8):lut;
            __m256 values[2];
            for(int half=0;half<2;++half) {
                const __m128i q8=half?_mm_srli_si128(qs,8):qs;
                const __m256i q=_mm256_cvtepu8_epi32(q8);
                const __m256i sign=_mm256_slli_epi32(_mm256_and_si256(q,_mm256_set1_epi32(8)),28);
                values[half]=_mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(_mm256_permutevar8x32_ps(values_lut,_mm256_and_si256(q,_mm256_set1_epi32(7)))),sign));
                if constexpr(!TABLE) {
                    values[half]=_mm256_mul_ps(values[half],_mm256_set1_ps(table[scale]));
                    values[half]=_mm256_cvtph_ps(_mm256_cvtps_ph(values[half],_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
                }
            }
            for(int t=0;t<M;++t) {
                a[t]=_mm256_fmadd_ps(values[0],_mm256_loadu_ps(x+size_t(t)*K+j),a[t]);
                b[t]=_mm256_fmadd_ps(values[1],_mm256_loadu_ps(x+size_t(t)*K+j+8),b[t]);
            }
        }
        for(int t=0;t<M;++t) {
            const __m256 sum=_mm256_add_ps(a[t],b[t]);
            __m128 v=_mm_add_ps(_mm256_castps256_ps128(sum),_mm256_extractf128_ps(sum,1));
            v=_mm_hadd_ps(v,v);v=_mm_hadd_ps(v,v);y[size_t(t)*N+row]=_mm_cvtss_f32(v);
        }
    }
}
#endif
template<bool TABLE>
void dispatch(const uint8_t* w,const uint8_t* s,const float* table,const float* x,float* y,
              int K,int N,int M,int first,int last) {
    if(K<16||K%16||N<1||M<1||M>8||first<0||last<first||last>N)
        throw std::invalid_argument("GLM FP4 CPU: invalid matrix extent");
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(cpu_isa_cap()>=3 && cpu_avx512bw_ok() && cpu_avx2_ok()) {
        rows512<TABLE>(w,s,table,x,y,K,N,M,first,last);return;
    }
    if(cpu_avx2_ok()) {
        rows256<TABLE>(w,s,table,x,y,K,N,M,first,last);return;
    }
#endif
    for(int r=first;r<last;++r)for(int t=0;t<M;++t) {
        float sum=0;
        for(int j=0;j<K;++j) {
            const unsigned q=(w[size_t(r)*(K/2)+j/2]>>((j&1)*4))&15;
            const unsigned scale=s[size_t(r)*(K/16)+j/16];
            const float magnitude=TABLE?table[size_t(scale)*8+(q&7)]:0.f;
            const float weight=TABLE?(q&8?-magnitude:magnitude):half_value(half_round(strata_fp4_e2m1(q)*table[scale]));
            sum=std::fma(weight,x[size_t(t)*K+j],sum);
        }
        y[size_t(t)*N+r]=sum;
    }
}
} // namespace
void glm_fp4_rows(const uint8_t* w,const uint8_t* s,float global,const float* x,float* y,
                  int K,int N,int M,int first,int last,bool qpn) {
    alignas(64) float table[128];scale_table(global,qpn,table);
    dispatch<false>(w,s,table,x,y,K,N,M,first,last);
}
void glm_fp4_weight_table(float global,bool qpn,float* table) {
    float scales[128];scale_table(global,qpn,scales);
    for(int s=0;s<128;++s)for(int q=0;q<8;++q)
        table[s*8+q]=half_value(half_round(strata_fp4_e2m1(q)*scales[s]));
    for(int i=128*8;i<glm_fp4_weight_table_floats;++i)table[i]=0.f;
}
void glm_fp4_rows_table(const uint8_t* w,const uint8_t* s,const float* table,const float* x,float* y,
                        int K,int N,int M,int first,int last) {
    dispatch<true>(w,s,table,x,y,K,N,M,first,last);
}
}
