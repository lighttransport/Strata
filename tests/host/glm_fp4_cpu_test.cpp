#include "strata/kernels/cpu/glm_fp4.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/host/glm_fp4.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv){try {
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
    for(int groups:{16,128,256}) {
        constexpr int rows=128;const int columns=groups*16;
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
        // Dense exception rows, empty rows and exceptions on mask boundaries.
        for(int row=0;row<rows;++row)for(int group=0;group<groups;++group)
            if(row%3==0||group==0||group==63||group==64||group==127)
                logical[size_t(row)*groups+group]=uint8_t(8+(row+group*7)%110);
        auto compact=strata::kernels::cpu::glm_fp4_csf32_rows(logical.data(),bases.data(),codes.data(),rows,groups);
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
            if(tokens==1)first.assign(got.begin(),got.begin()+rows);
            else if(std::memcmp(first.data(),got.data(),rows*4))throw std::runtime_error("CSF32 CPU first row changes with verification width");
        }
    }
    std::cout<<"FP4 CPU CSF32 rows PASS (host codec, sparse/dense exceptions, mask boundaries, widths 1..8)\n";
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
