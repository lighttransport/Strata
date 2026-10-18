#include "strata/kernels/cpu/glm_fp4.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/host/glm_fp4.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv){try {
    {
        std::vector<float> original(65536+137),rounded;
        for(unsigned i=0;i<65536;++i)original[i]=strata::cpu::exl3::half_value(uint16_t(i));
        for(size_t i=65536;i<original.size();++i)original[i]=std::sin(float(i))*70000.f;
        rounded=original;
        strata::kernels::cpu::glm_fp4_round_f16(rounded.data()+1,rounded.size()-2);
        for(size_t i=0;i<rounded.size();++i) {
            const float expected=(i==0||i+1==rounded.size())?original[i]:
                strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(original[i]));
            if(!(std::isnan(expected)&&std::isnan(rounded[i]))&&std::memcmp(&expected,&rounded[i],sizeof(float)))
                throw std::runtime_error("FP16 conversion differs from scalar codec");
        }
        for(size_t count=0;count<=129;++count) {
            auto values=original;strata::kernels::cpu::glm_fp4_round_f16(values.data()+1,count);
            for(size_t i=0;i<values.size();++i)if(i<1||i>count)
                if(std::memcmp(&values[i],&original[i],sizeof(float)))throw std::runtime_error("FP16 conversion wrote outside extent");
        }
    }
    constexpr int N=32,K=256;
    std::vector<uint8_t> weights(N*K/2),scales(N*K/16);
    for(size_t i=0;i<weights.size();++i)weights[i]=uint8_t(31*i+19);
    for(size_t i=0;i<scales.size();++i)scales[i]=uint8_t(8+i%100);
    std::vector<float> x(K*8),reference(N*8),actual(N*8),one;
    for(size_t i=0;i<x.size();++i)x[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.13f)*(i%107==0?8000.f:1.f)));
    constexpr float global=.00061f;
    for(bool qpn:{false,true}) {
        alignas(64) float table[strata::kernels::cpu::glm_fp4_weight_table_floats];
        strata::kernels::cpu::glm_fp4_weight_table(global,qpn,table);
        const float g=qpn?strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(global*16384.f)):global;
        for(int t=0;t<8;++t)for(int r=0;r<N;++r) {
            double sum=0;
            for(int j=0;j<K;++j) {
                float scale=strata_fp4_e4m3(scales[r*(K/16)+j/16])*g;
                if(qpn)scale=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(scale))/16384.f;
                const int q=(weights[r*(K/2)+j/2]>>((j&1)*4))&15;
                const float w=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(strata_fp4_e2m1(q)*scale));
                sum+=double(w)*x[t*K+j];
            }
            reference[t*N+r]=float(sum);
        }
        for(int M=1;M<=8;++M) {
            std::fill(actual.begin(),actual.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows(weights.data(),scales.data(),global,x.data(),actual.data(),K,N,M,0,13,qpn);
            strata::kernels::cpu::glm_fp4_rows_table(weights.data(),scales.data(),table,x.data(),actual.data(),K,N,M,13,N);
            double error=0,energy=0;
            for(int i=0;i<M*N;++i){error+=double(actual[i]-reference[i])*(actual[i]-reference[i]);energy+=double(reference[i])*reference[i];}
            const double rel=std::sqrt(error/std::max(energy,1e-30));
            if(!std::isfinite(rel)||rel>1e-5)throw std::runtime_error("CPU W4A16 parity failed");
            if(M==1)one.assign(actual.begin(),actual.begin()+N);
            else if(std::memcmp(one.data(),actual.data(),N*sizeof(float)))throw std::runtime_error("CPU W4A16 first row changes with verify width");
        }
    }
    std::cout<<"FP4 CPU W4A16 PASS (1..8 tokens, disjoint rows, outliers, fixed weight scales)\n";
    for(int groups:{16,128,256})for(int rows:{128,256}) {
        const int columns=groups*16;
        std::vector<uint8_t> bases(rows),codes(size_t(rows)*groups/2),logical(size_t(rows)*groups),w(size_t(rows)*columns/2);
        for(int i=0;i<rows;++i)bases[i]=30+i%20;
        for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*59+19);
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i*53+7);
        const size_t tiles=size_t(rows/128)*(groups/4),pages=(tiles+31)/32;
        std::vector<uint16_t> offsets(size_t(rows/32)*(groups/2)+pages);
        std::vector<uint32_t> page_bases(pages);
        if(strata_fp4_csf32_decode(rows,groups,bases.data(),bases.size(),codes.data(),codes.size(),
            offsets.data(),offsets.size(),page_bases.data(),pages,nullptr,0,logical.data(),logical.size()))
            throw std::runtime_error("CSF32 test fixture failed host validation");
        const auto clean=logical;
        auto empty_sparse=strata::kernels::cpu::glm_fp4_csf32_sparse_rows(logical.data(),bases.data(),codes.data(),rows,groups);
        if(!empty_sparse.exceptions.empty())throw std::runtime_error("exception-free sparse fixture has records");
        // Dense exception rows, empty rows and exceptions on mask boundaries.
        for(int row=0;row<rows;++row)for(int group=0;group<groups;++group)
            if(row%3==0||group==0||group==63||group==64||group==127)
                logical[size_t(row)*groups+group]=uint8_t(8+(row+group*7)%110);
        auto compact=strata::kernels::cpu::glm_fp4_csf32_rows(logical.data(),bases.data(),codes.data(),rows,groups);
        auto sparse=strata::kernels::cpu::glm_fp4_csf32_sparse_rows(logical.data(),bases.data(),codes.data(),rows,groups);
        const auto source_bases=bases,source_codes=codes;
        const auto borrowed=strata::kernels::cpu::glm_fp4_csf32_sparse_rows(logical.data(),source_bases.data(),source_codes.data(),rows,groups,true);
        if(borrowed.bytes()+size_t(rows)*(1+groups/2)!=sparse.bytes())throw std::runtime_error("borrowed CSF32 layout copied checkpoint bytes");
        std::fill(bases.begin(),bases.end(),0);std::fill(codes.begin(),codes.end(),0);
        std::vector<float> input(size_t(columns)*8),expected(size_t(rows)*8),got(expected.size()),first;
        for(size_t i=0;i<input.size();++i)input[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.37f)));
        alignas(64) float table[strata::kernels::cpu::glm_fp4_weight_table_floats];
        strata::kernels::cpu::glm_fp4_weight_table(global,true,table);
        for(int tokens=1;tokens<=8;++tokens) {
            std::fill(expected.begin(),expected.end(),-777.f);std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_table(w.data(),logical.data(),table,input.data(),expected.data(),columns,rows,tokens,13,rows-1);
            strata::kernels::cpu::glm_fp4_rows_csf32_table(w.data(),compact,table,input.data(),got.data(),columns,rows,tokens,13,rows-1);
            if(std::memcmp(expected.data(),got.data(),got.size()*4))throw std::runtime_error("CSF32 CPU rows differ from logical scale planes");
            std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_csf32_sparse_table(w.data(),sparse,table,input.data(),got.data(),columns,rows,tokens,13,rows-1);
            if(std::memcmp(expected.data(),got.data(),got.size()*4))throw std::runtime_error("sparse CSF32 CPU rows differ from logical scale planes");
            std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_csf32_sparse_table(w.data(),borrowed,table,input.data(),got.data(),columns,rows,tokens,13,rows-1);
            if(std::memcmp(expected.data(),got.data(),got.size()*4))throw std::runtime_error("borrowed CSF32 CPU rows differ from logical scale planes");
            if(tokens==1)first.assign(got.begin(),got.begin()+rows);
            else if(std::memcmp(first.data(),got.data(),rows*4))throw std::runtime_error("CSF32 CPU first row changes with verification width");
        }
        strata::kernels::cpu::glm_fp4_rows_table(w.data(),clean.data(),table,input.data(),expected.data(),columns,rows,4,0,rows);
        strata::kernels::cpu::glm_fp4_rows_csf32_sparse_table(w.data(),empty_sparse,table,input.data(),got.data(),columns,rows,4,0,rows);
        if(std::memcmp(expected.data(),got.data(),size_t(rows)*4*4))throw std::runtime_error("exception-free sparse CPU rows differ");
        auto qinput=input;
        qinput[0]=8192.f;qinput[63]=-4096.f;
        std::fill(qinput.begin()+128,qinput.begin()+192,0.f);
        const auto q8=strata::kernels::cpu::glm_fp4_quantize_q8(qinput.data(),columns,8);
        for(size_t i=0;i<qinput.size();++i) {
            const float scale=q8.scales[i/64];
            float maximum=0;
            for(size_t j=i/64*64;j<i/64*64+64;++j)maximum=std::max(maximum,std::abs(qinput[j]));
            float reference_scale=maximum/127.f;
            if(maximum>0&&reference_scale==0)reference_scale=maximum;
            const int reference=reference_scale==0?0:int(std::clamp(std::nearbyint(qinput[i]/reference_scale),-127.f,127.f));
            if(scale!=reference_scale||int(q8.values[i])!=reference)throw std::runtime_error("Q8 quantizer differs from scalar reference");
            if(std::abs(qinput[i]-float(q8.values[i])*scale)>scale*.5001f+1.e-6f)
                throw std::runtime_error("Q8 activation exceeds half-scale error bound");
        }
        if(q8.scales[2]!=0)throw std::runtime_error("Q8 zero block has nonzero scale");
        std::vector<float> qfirst;
        for(int tokens=1;tokens<=8;++tokens) {
            std::fill(got.begin(),got.end(),-777.f);
            strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),sparse,table,q8,got.data(),columns,rows,tokens,13,rows-1);
            auto borrowed_output=got;
            strata::kernels::cpu::glm_fp4_rows_csf32_q8(w.data(),borrowed,table,q8,borrowed_output.data(),columns,rows,tokens,13,rows-1);
            if(std::memcmp(got.data(),borrowed_output.data(),got.size()*4))throw std::runtime_error("borrowed Q8 rows differ from owning scales");
            double error=0,norm=0;
            for(int t=0;t<tokens;++t)for(int row=13;row<rows-1;++row) {
                double sum=0;
                for(int j=0;j<columns;++j) {
                    const auto packed=w[size_t(row)*(columns/2)+j/2];
                    const auto code=(packed>>((j%2)*4))&15;
                    const auto scale=logical[size_t(row)*groups+j/16];
                    const float factor=table[size_t(scale)*8+1]*q8.scales[size_t(t)*(columns/64)+j/64];
                    sum+=double(strata_fp4_e2m1(code)*2)*q8.values[size_t(t)*columns+j]*factor;
                }
                const double delta=got[size_t(t)*rows+row]-sum;
                error+=delta*delta;norm+=sum*sum;
            }
            if(std::sqrt(error/std::max(norm,1.e-30))>1.e-5)throw std::runtime_error("Q8 dot differs from decoded FP64 reference");
            for(int t=0;t<8;++t)for(int row=0;row<rows;++row)
                if((t>=tokens||row<13||row==rows-1)&&got[size_t(t)*rows+row]!=-777.f)
                    throw std::runtime_error("Q8 product wrote outside requested rows");
            if(tokens==1)qfirst.assign(got.begin(),got.begin()+rows);
            else if(std::memcmp(qfirst.data(),got.data(),rows*4))throw std::runtime_error("Q8 first row changes with verification width");
        }
        qinput[0]=std::numeric_limits<float>::infinity();
        bool rejected=false;try{strata::kernels::cpu::glm_fp4_quantize_q8(qinput.data(),columns,1);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Q8 accepted nonfinite activation");
    }
    std::cout<<"FP4 CPU CSF32 rows PASS (host codec, sparse/dense exceptions, mask boundaries, widths 1..8)\n";
    std::cout<<"FP4 CPU W4A8 PASS (FP64 dots, quantization error bounds, partial rows, fixed first row) mode="
             <<(strata::kernels::cpu::glm_fp4_q8_vnni_available()?"vnni":"scalar")<<'\n';
    if(argc==2 && std::string(argv[1])=="--bench") {
        constexpr int rows=2048,columns=4096,copies=16,repeats=64;
        const size_t wb=size_t(rows)*columns/2,sb=size_t(rows)*columns/16;
        std::vector<uint8_t> w(wb*copies),s(sb*copies);
        std::vector<float> input(columns*8),baseline(rows*8),output(rows*8);
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i*59+7);
        for(size_t i=0;i<s.size();++i)s[i]=uint8_t(24+i%100);
        for(size_t i=0;i<input.size();++i)input[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.17f)));
        alignas(64) float table[strata::kernels::cpu::glm_fp4_weight_table_floats];
        strata::kernels::cpu::glm_fp4_weight_table(global,true,table);
        for(int tokens:{1,4,8}) {
            strata::kernels::cpu::glm_fp4_rows(w.data(),s.data(),global,input.data(),baseline.data(),columns,rows,tokens,0,rows,true);
            strata::kernels::cpu::glm_fp4_rows_table(w.data(),s.data(),table,input.data(),output.data(),columns,rows,tokens,0,rows);
            if(std::memcmp(baseline.data(),output.data(),size_t(tokens)*rows*4))throw std::runtime_error("cached weight lookup changes CPU outputs");
            for(bool lookup:{false,true}) {
                auto start=std::chrono::steady_clock::now();
                for(int i=0;i<repeats;++i) {
                    const auto* codes=w.data()+(i%copies)*wb;const auto* scales=s.data()+(i%copies)*sb;
                    if(lookup)strata::kernels::cpu::glm_fp4_rows_table(codes,scales,table,input.data(),output.data(),columns,rows,tokens,0,rows);
                    else strata::kernels::cpu::glm_fp4_rows(codes,scales,global,input.data(),output.data(),columns,rows,tokens,0,rows,true);
                }
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
                std::cout<<"CPU_KERNEL_BENCH N="<<rows<<" K="<<columns<<" M="<<tokens<<" lookup="<<lookup
                         <<" us="<<us<<" packed_GB_s="<<(wb+sb)/(us*1000.)<<" working_set_bytes="<<copies*(wb+sb)<<" threads=1\n";
            }
        }
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
