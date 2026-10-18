#include "strata/kernels/cpu/glm_fp4.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/host/glm_fp4.h"
#include <algorithm>
#include <cmath>
#include <bit>
#include <limits>
#include <stdexcept>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
namespace strata::kernels::cpu {
namespace {
using strata::cpu::exl3::half_value;
using strata::cpu::exl3::half_round;
struct LogicalScales {
    const uint8_t* data;
    struct Row {const uint8_t* data;uint8_t at(int group)const {return data[group];}};
    Row row(int index,int groups)const {return {data+size_t(index)*groups};}
};
struct CsfScales {
    const GlmFp4Csf32Rows& data;
    struct Row {
        const uint8_t *codes,*exceptions;
        const uint64_t* masks;
        const uint32_t* offsets;
        uint8_t base;
        uint8_t raw(int group)const {
            const uint8_t code=codes[group/2];
            return uint8_t(base+((code>>((group%2)*4))&15));
        }
        uint8_t at(int group)const {
            const int word=group/64,bit=group%64;
            const uint64_t mask=masks[word],flag=uint64_t(1)<<bit;
            if(mask&flag)return exceptions[offsets[word]+std::popcount(mask&(flag-1))];
            return raw(group);
        }
    };
    Row row(int index,int groups)const {
        const size_t word=size_t(index)*((groups+63)/64);
        return {data.codes.data()+size_t(index)*(groups/2),
            data.exceptions.data(),data.masks.data()+word,data.offsets.data()+word,data.bases[index]};
    }
};
struct SparseScales {
    const GlmFp4Csf32SparseRows& data;
    struct Row {
        const uint8_t* codes;
        const uint16_t* end;
        mutable const uint16_t* next;
        uint8_t base;
        int quartet_stride,pair_stride;
        uint8_t at(int group)const {
            // Dot products visit groups in order. Repeated reads of the same
            // group retain its exception until the next group advances it.
            while(next!=end && (*next>>8)<group)++next;
            if(next!=end && (*next>>8)==group)return uint8_t(*next);
            const auto code=codes[(group/4)*quartet_stride+(group/2%2)*pair_stride];
            return uint8_t(base+((code>>((group%2)*4))&15));
        }
        uint32_t quartet(int group)const {
            // Q8 products visit complete quartets in increasing order. Decode
            // both packed bytes once and scan the sparse exception stream once.
            const auto* pair=codes+(group/4)*quartet_stride;
            const unsigned a=pair[0],b=pair[pair_stride];
            uint32_t values=uint32_t(uint8_t(base+(a&15))) |
                (uint32_t(uint8_t(base+(a>>4)))<<8) |
                (uint32_t(uint8_t(base+(b&15)))<<16) |
                (uint32_t(uint8_t(base+(b>>4)))<<24);
            while(next!=end && (*next>>8)<group)++next;
            while(next!=end && (*next>>8)<group+4) {
                const unsigned shift=unsigned((*next>>8)-group)*8;
                values=(values&~(uint32_t(255)<<shift)) | (uint32_t(uint8_t(*next))<<shift);
                ++next;
            }
            return values;
        }
    };
    Row row(int index,int groups)const {
        const auto* records=data.exceptions.data();
        // Avoid arithmetic on a null data() pointer for an exception-free matrix.
        const auto* begin=data.exceptions.empty()?nullptr:records+data.offsets[index];
        const auto* end=data.exceptions.empty()?nullptr:records+data.offsets[index+1];
        if(data.borrowed_codes) {
            const auto* codes=data.borrowed_codes+size_t(index/128)*(groups/4)*256+(index/32%4)*32+index%32;
            return {codes,end,begin,data.borrowed_bases[index],256,128};
        }
        return {data.codes.data()+size_t(index)*(groups/2),end,begin,data.bases[index],2,1};
    }
};
static void scale_table(float global,bool qpn,float* table) {
    const float g=qpn?half_value(half_round(global*16384.f)):global;
    for(int i=0;i<128;++i) {
        const float value=strata_fp4_e4m3(uint8_t(i))*g;
        table[i]=qpn?half_value(half_round(value))/16384.f:value;
    }
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,f16c")))
void round_f16_avx2(float* x,size_t count) {
    size_t i=0;
    for(;i+8<=count;i+=8) {
        const auto v=_mm256_loadu_ps(x+i);
        // The scalar codec canonicalizes NaNs. Retain that behavior as well.
        if(_mm256_movemask_ps(_mm256_cmp_ps(v,v,_CMP_UNORD_Q))) {
            for(int j=0;j<8;++j)x[i+j]=half_value(half_round(x[i+j]));
        }else _mm256_storeu_ps(x+i,_mm256_cvtph_ps(_mm256_cvtps_ph(v,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC)));
    }
    for(;i<count;++i)x[i]=half_value(half_round(x[i]));
}
__attribute__((target("avx512f,avx512bw")))
void quantize_q8_avx512(const float* x,GlmFp4Q8& out) {
    const auto abs_mask=_mm512_set1_epi32(0x7fffffff);
    for(size_t block=0;block<out.scales.size();++block) {
        __m512 values[4],magnitude=_mm512_setzero_ps();
        for(int i=0;i<4;++i) {
            values[i]=_mm512_loadu_ps(x+block*64+i*16);
            const auto a=_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(values[i]),abs_mask));
            if(_mm512_cmp_ps_mask(a,_mm512_set1_ps(INFINITY),_CMP_NLT_UQ))
                throw std::invalid_argument("GLM FP4 CPU: nonfinite Q8 activation");
            magnitude=_mm512_max_ps(magnitude,a);
        }
        const float maximum=_mm512_reduce_max_ps(magnitude);
        float scale=maximum/127.f;
        if(maximum>0 && scale==0)scale=maximum;
        out.scales[block]=scale;
        if(scale==0){std::fill_n(out.values.data()+block*64,64,int8_t(0));continue;}
        for(int i=0;i<4;++i) {
            const auto divided=_mm512_div_ps(values[i],_mm512_set1_ps(scale));
            const auto clipped=_mm512_min_ps(_mm512_set1_ps(127.f),_mm512_max_ps(_mm512_set1_ps(-127.f),divided));
            const auto packed=_mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(clipped));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out.values.data()+block*64+i*16),packed);
        }
    }
}
// Fix the verification width before entering the dot-product loop so the
// accumulators stay in registers. Each row keeps the same FMA and sum order.
template<bool TABLE, int M, class Scales>
__attribute__((target("avx512f,avx512bw,fma,f16c")))
void rows512(const uint8_t* w,const Scales& scales,const float* table,const float* x,float* y,
              int K,int N,int first,int last) {
    const __m512 lut=_mm512_setr_ps(0,.5f,1,1.5f,2,3,4,6,0,.5f,1,1.5f,2,3,4,6);
    const __m512i mask=_mm512_set1_epi32(7);
    const __m128i nibble=_mm_set1_epi8(15);
    for(int row=first;row<last;++row) {
        __m512 acc[8];for(int t=0;t<M;++t)acc[t]=_mm512_setzero_ps();
        const uint8_t* weights=w+size_t(row)*(K/2);
        const auto groups=scales.row(row,K/16);
        for(int j=0;j<K;j+=16) {
            const __m128i packed=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights+j/2));
            const __m128i q8=_mm_unpacklo_epi8(_mm_and_si128(packed,nibble),_mm_and_si128(_mm_srli_epi16(packed,4),nibble));
            const __m512i q=_mm512_cvtepu8_epi32(q8);
            const __m512 sign=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(q,_mm512_set1_epi32(8)),28));
            const __m512 values_lut=TABLE?_mm512_loadu_ps(table+size_t(groups.at(j/16))*8):lut;
            const __m512 mag=_mm512_permutexvar_ps(_mm512_and_si512(q,mask),values_lut);
            __m512 values=_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(mag),_mm512_castps_si512(sign)));
            if constexpr(!TABLE) {
                values=_mm512_mul_ps(values,_mm512_set1_ps(table[groups.at(j/16)]));
                values=_mm512_cvtph_ps(_mm512_cvtps_ph(values,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
            }
            for(int t=0;t<M;++t)acc[t]=_mm512_fmadd_ps(values,_mm512_loadu_ps(x+size_t(t)*K+j),acc[t]);
        }
        for(int t=0;t<M;++t)y[size_t(t)*N+row]=_mm512_reduce_add_ps(acc[t]);
    }
}
template<bool TABLE, int M, class Scales>
__attribute__((target("avx2,fma,f16c")))
void rows256(const uint8_t* w,const Scales& scales,const float* table,const float* x,float* y,
              int K,int N,int first,int last) {
    const __m256 lut=_mm256_setr_ps(0,.5f,1,1.5f,2,3,4,6);
    for(int row=first;row<last;++row) {
        const auto groups=scales.row(row,K/16);
        __m256 a[8],b[8];for(int t=0;t<M;++t)a[t]=b[t]=_mm256_setzero_ps();
        for(int j=0;j<K;j+=16) {
            const auto* src=w+size_t(row)*(K/2)+j/2;
            const __m128i packed=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
            const __m128i qs=_mm_unpacklo_epi8(_mm_and_si128(packed,_mm_set1_epi8(15)),_mm_and_si128(_mm_srli_epi16(packed,4),_mm_set1_epi8(15)));
            const unsigned scale=groups.at(j/16);
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
template<int M>
__attribute__((target("avx512f,avx512bw,avx512vnni,fma")))
void rows_q8_vnni(const uint8_t* w,const GlmFp4Csf32SparseRows& s,const float* table,
                   const GlmFp4Q8& x,float* y,int K,int N,int first,int last) {
    const __m128i lut128=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const __m512i lut=_mm512_broadcast_i32x4(lut128),zero=_mm512_setzero_si512();
    const SparseScales source{s};
    for(int row=first;row<last;++row) {
        const auto groups=source.row(row,K/16);
        const auto* weights=w+size_t(row)*(K/2);
        __m512 acc[M];for(int t=0;t<M;++t)acc[t]=_mm512_setzero_ps();
        for(int j=0;j<K;j+=64) {
            const __m512i packed=_mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights+j/2)));
            const __m512i nibble=_mm512_or_si512(_mm512_and_si512(packed,_mm512_set1_epi16(15)),
                _mm512_slli_epi16(_mm512_and_si512(packed,_mm512_set1_epi16(240)),4));
            const __m512i values=_mm512_shuffle_epi8(lut,nibble);
            const uint32_t scales=groups.quartet(j/16);
            const float a=table[size_t(scales&255)*8+1];
            const float b=table[size_t((scales>>8)&255)*8+1];
            const float c=table[size_t((scales>>16)&255)*8+1];
            const float d=table[size_t(scales>>24)*8+1];
            const __m512 scale=_mm512_setr_ps(a,a,a,a,b,b,b,b,c,c,c,c,d,d,d,d);
            for(int t=0;t<M;++t) {
                const __m512i act=_mm512_loadu_si512(x.values.data()+size_t(t)*K+j);
                const auto negative=_mm512_cmpgt_epi8_mask(zero,act);
                const __m512i signed_weights=_mm512_mask_sub_epi8(values,negative,zero,values);
                const __m512i dots=_mm512_dpbusd_epi32(zero,_mm512_abs_epi8(act),signed_weights);
                const __m512 factors=_mm512_mul_ps(scale,_mm512_set1_ps(x.scales[size_t(t)*(K/64)+j/64]));
                acc[t]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(dots),factors,acc[t]);
            }
        }
        for(int t=0;t<M;++t)y[size_t(t)*N+row]=_mm512_reduce_add_ps(acc[t]);
    }
}
#endif
template<bool TABLE, class Scales>
void dispatch(const uint8_t* w,const Scales& s,const float* table,const float* x,float* y,
              int K,int N,int M,int first,int last) {
    if(K<16||K%16||N<1||M<1||M>8||first<0||last<first||last>N)
        throw std::invalid_argument("GLM FP4 CPU: invalid matrix extent");
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(cpu_isa_cap()>=3 && cpu_avx512bw_ok() && cpu_avx2_ok()) {
        switch(M) {
            case 1: rows512<TABLE,1>(w,s,table,x,y,K,N,first,last);return;
            case 2: rows512<TABLE,2>(w,s,table,x,y,K,N,first,last);return;
            case 3: rows512<TABLE,3>(w,s,table,x,y,K,N,first,last);return;
            case 4: rows512<TABLE,4>(w,s,table,x,y,K,N,first,last);return;
            case 5: rows512<TABLE,5>(w,s,table,x,y,K,N,first,last);return;
            case 6: rows512<TABLE,6>(w,s,table,x,y,K,N,first,last);return;
            case 7: rows512<TABLE,7>(w,s,table,x,y,K,N,first,last);return;
            case 8: rows512<TABLE,8>(w,s,table,x,y,K,N,first,last);return;
        }
    }
    if(cpu_avx2_ok()) {
        switch(M) {
            case 1: rows256<TABLE,1>(w,s,table,x,y,K,N,first,last);return;
            case 2: rows256<TABLE,2>(w,s,table,x,y,K,N,first,last);return;
            case 3: rows256<TABLE,3>(w,s,table,x,y,K,N,first,last);return;
            case 4: rows256<TABLE,4>(w,s,table,x,y,K,N,first,last);return;
            case 5: rows256<TABLE,5>(w,s,table,x,y,K,N,first,last);return;
            case 6: rows256<TABLE,6>(w,s,table,x,y,K,N,first,last);return;
            case 7: rows256<TABLE,7>(w,s,table,x,y,K,N,first,last);return;
            case 8: rows256<TABLE,8>(w,s,table,x,y,K,N,first,last);return;
        }
    }
#endif
    for(int r=first;r<last;++r)for(int t=0;t<M;++t) {
        float sum=0;
        for(int j=0;j<K;++j) {
            const unsigned q=(w[size_t(r)*(K/2)+j/2]>>((j&1)*4))&15;
            const unsigned scale=s.row(r,K/16).at(j/16);
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
    dispatch<false>(w,LogicalScales{s},table,x,y,K,N,M,first,last);
}
void glm_fp4_weight_table(float global,bool qpn,float* table) {
    float scales[128];scale_table(global,qpn,scales);
    for(int s=0;s<128;++s)for(int q=0;q<8;++q)
        table[s*8+q]=half_value(half_round(strata_fp4_e2m1(q)*scales[s]));
    for(int i=128*8;i<glm_fp4_weight_table_floats;++i)table[i]=0.f;
}
void glm_fp4_rows_table(const uint8_t* w,const uint8_t* s,const float* table,const float* x,float* y,
                        int K,int N,int M,int first,int last) {
    dispatch<true>(w,LogicalScales{s},table,x,y,K,N,M,first,last);
}
GlmFp4Csf32Rows glm_fp4_csf32_rows(const uint8_t* logical,const uint8_t* bases,
                                   const uint8_t* codes,int rows,int groups) {
    if(!logical||!bases||!codes||rows<128||rows%128||groups<4||groups%4||
       uint64_t(rows)*groups>std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("GLM FP4 CPU: invalid CSF32 scale extent");
    GlmFp4Csf32Rows out;out.bases.assign(bases,bases+rows);out.rows=rows;out.groups=groups;
    out.codes.resize(size_t(rows)*groups/2);
    for(int row=0;row<rows;++row)for(int group=0;group<groups;group+=2) {
        const size_t index=(size_t(row/128)*(groups/4)+group/4)*256+
            (group%4/2)*128+(row/32%4)*32+row%32;
        out.codes[size_t(row)*(groups/2)+group/2]=codes[index];
    }
    const size_t words=(groups+63)/64;
    out.masks.resize(size_t(rows)*words);out.offsets.resize(out.masks.size());
    const CsfScales source{out};
    for(int row=0;row<rows;++row) {
        const auto decoded=source.row(row,groups);
        for(size_t word=0;word<words;++word) {
            const size_t index=size_t(row)*words+word;
            out.offsets[index]=out.exceptions.size();
            for(int bit=0;bit<64&&int(word*64)+bit<groups;++bit) {
                const int group=word*64+bit;
                const uint8_t value=logical[size_t(row)*groups+group];
                if(value!=decoded.raw(group)) {
                    out.masks[index]|=uint64_t(1)<<bit;out.exceptions.push_back(value);
                }
            }
        }
    }
    return out;
}
void glm_fp4_rows_csf32_table(const uint8_t* w,const GlmFp4Csf32Rows& s,const float* table,
                             const float* x,float* y,int K,int N,int M,int first,int last) {
    if(N!=s.rows||K/16!=s.groups)throw std::invalid_argument("GLM FP4 CPU: CSF32 scale shape mismatch");
    dispatch<true>(w,CsfScales{s},table,x,y,K,N,M,first,last);
}
GlmFp4Csf32SparseRows glm_fp4_csf32_sparse_rows(const uint8_t* logical,const uint8_t* bases,
                                              const uint8_t* codes,int rows,int groups,bool borrow) {
    if(!logical||!bases||!codes||rows<128||rows%128||groups<4||groups>256||groups%4||
       uint64_t(rows)*groups>std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("GLM FP4 CPU: invalid sparse CSF32 scale extent");
    GlmFp4Csf32SparseRows out;out.rows=rows;out.groups=groups;
    if(borrow){out.borrowed_bases=bases;out.borrowed_codes=codes;}
    else {out.bases.assign(bases,bases+rows);out.codes.resize(size_t(rows)*groups/2);}
    out.offsets.resize(size_t(rows)+1);
    for(int row=0;row<rows;++row) {
        auto* dst=borrow?nullptr:out.codes.data()+size_t(row)*(groups/2);
        if(dst)for(int group=0;group<groups;group+=2) {
            const size_t index=(size_t(row/128)*(groups/4)+group/4)*256+
                (group%4/2)*128+(row/32%4)*32+row%32;
            dst[group/2]=codes[index];
        }
        out.offsets[row]=out.exceptions.size();
        for(int group=0;group<groups;++group) {
            const auto value=logical[size_t(row)*groups+group];
            const size_t index=(size_t(row/128)*(groups/4)+group/4)*256+
                (group%4/2)*128+(row/32%4)*32+row%32;
            const auto raw=uint8_t(bases[row]+((codes[index]>>((group%2)*4))&15));
            if(value!=raw)out.exceptions.push_back(uint16_t((group<<8)|value));
        }
    }
    out.offsets[rows]=out.exceptions.size();
    // Capacity is charged to the host cache; do not retain geometric growth
    // slack when thousands of experts share a small memory allowance.
    std::vector<uint16_t>(out.exceptions).swap(out.exceptions);
    return out;
}
void glm_fp4_rows_csf32_sparse_table(const uint8_t* w,const GlmFp4Csf32SparseRows& s,const float* table,
                                    const float* x,float* y,int K,int N,int M,int first,int last) {
    if(N!=s.rows||K/16!=s.groups)throw std::invalid_argument("GLM FP4 CPU: sparse CSF32 scale shape mismatch");
    dispatch<true>(w,SparseScales{s},table,x,y,K,N,M,first,last);
}
GlmFp4Q8 glm_fp4_quantize_q8(const float* x,int columns,int tokens) {
    if(!x||columns<64||columns%64||tokens<1||tokens>8)
        throw std::invalid_argument("GLM FP4 CPU: invalid Q8 activation extent");
    GlmFp4Q8 out;out.columns=columns;out.tokens=tokens;
    out.values.resize(size_t(columns)*tokens);out.scales.resize(size_t(columns/64)*tokens);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(cpu_isa_cap()>=3&&cpu_avx512bw_ok()) {quantize_q8_avx512(x,out);return out;}
#endif
    for(size_t block=0;block<out.scales.size();++block) {
        float magnitude=0;
        for(int i=0;i<64;++i) {
            const auto value=x[block*64+i];
            if(!std::isfinite(value))throw std::invalid_argument("GLM FP4 CPU: nonfinite Q8 activation");
            magnitude=std::max(magnitude,std::abs(value));
        }
        float scale=magnitude/127.f;
        if(magnitude>0 && scale==0)scale=magnitude;
        out.scales[block]=scale;
        for(int i=0;i<64;++i)out.values[block*64+i]=scale==0?0:
            int8_t(std::clamp(std::nearbyint(x[block*64+i]/scale),-127.f,127.f));
    }
    return out;
}
void glm_fp4_round_f16(float* x,size_t count) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(cpu_isa_cap()>=2&&cpu_avx2_ok()){round_f16_avx2(x,count);return;}
#endif
    for(size_t i=0;i<count;++i)x[i]=half_value(half_round(x[i]));
}
bool glm_fp4_q8_vnni_available() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return cpu_isa_cap()>=3&&cpu_avx512bw_ok()&&cpu_avx2_ok()&&__builtin_cpu_supports("avx512vnni");
#else
    return false;
#endif
}
void glm_fp4_rows_csf32_q8(const uint8_t* w,const GlmFp4Csf32SparseRows& s,const float* table,
                          const GlmFp4Q8& x,float* y,int K,int N,int M,int first,int last) {
    if(K<64||K%64||N<1||M<1||M>8||first<0||last<first||last>N||s.rows!=N||s.groups!=K/16||
       x.columns!=K||x.tokens<M||x.values.size()<size_t(K)*M||x.scales.size()<size_t(K/64)*M)
        throw std::invalid_argument("GLM FP4 CPU: invalid Q8 product extent");
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(glm_fp4_q8_vnni_available()) {
        switch(M) {
            case 1:rows_q8_vnni<1>(w,s,table,x,y,K,N,first,last);return;
            case 2:rows_q8_vnni<2>(w,s,table,x,y,K,N,first,last);return;
            case 3:rows_q8_vnni<3>(w,s,table,x,y,K,N,first,last);return;
            case 4:rows_q8_vnni<4>(w,s,table,x,y,K,N,first,last);return;
            case 5:rows_q8_vnni<5>(w,s,table,x,y,K,N,first,last);return;
            case 6:rows_q8_vnni<6>(w,s,table,x,y,K,N,first,last);return;
            case 7:rows_q8_vnni<7>(w,s,table,x,y,K,N,first,last);return;
            case 8:rows_q8_vnni<8>(w,s,table,x,y,K,N,first,last);return;
        }
    }
#endif
    static constexpr int8_t magnitude[]={0,1,2,3,4,6,8,12};
    const SparseScales source{s};
    for(int row=first;row<last;++row)for(int t=0;t<M;++t) {
        const auto scales=source.row(row,K/16);float sum=0;
        for(int j=0;j<K;j+=16) {
            int dot=0;
            for(int i=0;i<16;++i) {
                const int q=(w[size_t(row)*(K/2)+(j+i)/2]>>(((j+i)%2)*4))&15;
                dot+=(q&8?-magnitude[q&7]:magnitude[q&7])*int(x.values[size_t(t)*K+j+i]);
            }
            const float factor=table[size_t(scales.at(j/16))*8+1]*x.scales[size_t(t)*(K/64)+j/64];
            sum=std::fma(float(dot),factor,sum);
        }
        y[size_t(t)*N+row]=sum;
    }
}

}
