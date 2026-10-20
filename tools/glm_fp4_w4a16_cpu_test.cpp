#include "strata/kernels/cpu/fp4_local_w4a16.hpp"
#include "strata/host/glm_fp4.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
float bf16(float x){auto v=std::bit_cast<uint32_t>(x);v+=0x7fff+((v>>16)&1);return std::bit_cast<float>(v&0xffff0000u);}
uint64_t output_digest=14695981039346656037ULL;
void check(int K,int N,int pattern,bool exhaustive=false) {
    std::mt19937 rng(53);std::vector<uint8_t>w(N*K/2),s(N*K/16),packed(w.size());
    for(auto& v:w)v=rng()%256;
    for(size_t i=0;i<s.size();++i)s[i]=uint8_t(i%127);
    if(exhaustive)for(int r=0;r<N;++r){
        for(int j=0;j<K;++j){const int q=j%16;auto& v=w[size_t(r)*K/2+j/2];v=uint8_t((v&~(15<<((j%2)*4)))|(q<<((j%2)*4)));}
        for(int j=0;j<K/16;++j)s[size_t(r)*K/16+j]=uint8_t(std::max(0,r-1)%127);
    }
    if(strata_glm_fp4_repack_k32(w.data(),packed.data(),w.size(),0))throw std::runtime_error("packing failed");
    std::vector<float>x(K);
    for(int j=0;j<K;++j){
        float v=(int(rng()%2049)-1024)/1024.f;
        if(pattern==1)v=1;
        if(pattern==2)v=j%2?8.f:-8.f;
        if(pattern==3)v=0;
        if(pattern==4)v=j%64?v*.001f:16.f;
        x[j]=bf16(v);
    }
    std::vector<float>y[4];for(auto& a:y)a.assign(N,-123.f);
    for(int mode=0;mode<4;++mode)
        strata::kernels::cpu::fp4_local_w4a16_rows(mode>=2?packed.data():w.data(),s.data(),x.data(),.375f,K,N,1,N-1,y[mode].data(),mode>=2,mode%2);
    if(y[0]!=y[2]||y[1]!=y[3])throw std::runtime_error("packing changed W4A16 output bits");
    for(int mode=0;mode<4;++mode) {
        for(float value:y[mode]){output_digest^=std::bit_cast<uint32_t>(value);output_digest*=1099511628211ULL;}
        if(y[mode].front()!=-123||y[mode].back()!=-123)throw std::runtime_error("row guard overwritten");
        for(int r=1;r<N-1;++r) {
            double ref=0,norm=0;
            for(int j=0;j<K;++j){size_t at=size_t(r)*K+j;const unsigned q=(w[at/2]>>((j%2)*4))&15;
                const double product=double(strata_glm_e2m1(uint8_t(q)))*strata_glm_e4m3(s[at/16])*x[j];ref+=product;norm+=std::abs(product);}
            ref*=.375;norm*=.375;
            const double steps=mode%2?K/16+6:K+2;
            const double eps=std::numeric_limits<float>::epsilon()/2;
            const double bound=2*steps*eps/(1-steps*eps)*norm+1e-7;
            if(!std::isfinite(y[mode][r])||std::abs(double(y[mode][r])-ref)>bound)throw std::runtime_error("FP64 rounding bound exceeded");
        }
    }
    for(int tokens:{1,2,3,4,7,16}) {
        std::vector<float> bx(size_t(tokens)*K),got(size_t(tokens)*N,-123.f),ref(got.size());
        for(int t=0;t<tokens;++t)for(int j=0;j<K;++j)bx[size_t(t)*K+j]=x[(j+17*t)%K]*(t%2?-1.f:1.f);
        for(int mode=0;mode<4;++mode) {
            const auto* codes=mode>=2?packed.data():w.data();
            std::fill(got.begin(),got.end(),-123.f);ref=got;
            for(int t=0;t<tokens;++t)strata::kernels::cpu::fp4_local_w4a16_rows(codes,s.data(),bx.data()+size_t(t)*K,.375f,K,N,1,N-1,ref.data()+size_t(t)*N,mode>=2,mode%2);
            strata::kernels::cpu::fp4_local_w4a16_batch_rows(codes,s.data(),bx.data(),.375f,K,N,tokens,1,N-1,got.data(),mode>=2,mode%2);
            if(std::memcmp(got.data(),ref.data(),got.size()*sizeof(float)))throw std::runtime_error("batch token/row parity mismatch");
        }
    }

}
void kernel_bench(int repeats,bool planar) {
    constexpr int K=4096,N=128;
    std::mt19937 rng(53);
    std::vector<uint8_t>w(N*K/2),s(N*K/16);
    std::vector<float>x(K),y(N),reference(N);
    for(auto& v:w)v=rng()%256;
    for(auto& v:s)v=uint8_t(40+rng()%40);
    for(auto& v:x)v=bf16((int(rng()%2049)-1024)/1024.f);
    auto run=[&](bool layout){strata::kernels::cpu::fp4_local_w4a16_rows(w.data(),s.data(),x.data(),.375f,K,N,0,N,y.data(),layout,true);};
    run(false);reference=y;
    if(planar&&strata_glm_fp4_repack_k32(w.data(),w.data(),w.size(),0))throw std::runtime_error("packing failed");
    run(planar);
    if(std::memcmp(y.data(),reference.data(),N*sizeof(float)))throw std::runtime_error("benchmark packed parity failed");
    const auto start=std::chrono::steady_clock::now();const auto cpu=std::clock();
    for(int i=0;i<repeats;++i)run(planar);
    const double cpu_ms=1000.*(std::clock()-cpu)/CLOCKS_PER_SEC;
    const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(std::memcmp(y.data(),reference.data(),N*sizeof(float)))throw std::runtime_error("benchmark output changed");
    std::cout<<"W4A16 synthetic cache-resident K="<<K<<" N="<<N<<" repeats="<<repeats<<" planar="<<planar
             <<" wall_ms="<<wall_ms<<" cpu_ms="<<cpu_ms<<" checksum="<<y[17]<<"\n";
}
void batch_bench(bool reuse,int tokens,int repeats,int copies=1) {
    if(tokens<1||tokens>16||repeats<1||repeats>1000000||copies<1||copies>1024)throw std::runtime_error("invalid batch benchmark bounds");
    constexpr int K=4096,N=128;std::mt19937 rng(53);
    std::vector<uint8_t>w(N*K/2),scales(N*K/16);
    for(auto& v:w)v=rng()%256;
    for(auto& v:scales)v=uint8_t(40+rng()%40);
    if(strata_glm_fp4_repack_k32(w.data(),w.data(),w.size(),0))throw std::runtime_error("packing failed");
    // Distinct physical weight copies exceed cache without changing the oracle.
    const size_t wb=w.size(),sb=scales.size();
    w.resize(wb*copies);scales.resize(sb*copies);
    for(int c=1;c<copies;++c) {
        std::memcpy(w.data()+size_t(c)*wb,w.data(),wb);
        std::memcpy(scales.data()+size_t(c)*sb,scales.data(),sb);
    }
    std::vector<float>x(size_t(tokens)*K),y(size_t(tokens)*N);
    for(auto& v:x)v=bf16((int(rng()%2049)-1024)/1024.f);
    auto run=[&](bool batched,int copy=0) {
        const auto* cw=w.data()+size_t(copy)*wb;const auto* cs=scales.data()+size_t(copy)*sb;
        if(batched)strata::kernels::cpu::fp4_local_w4a16_batch_rows(cw,cs,x.data(),.375f,K,N,tokens,0,N,y.data(),true,true);
        else for(int t=0;t<tokens;++t)strata::kernels::cpu::fp4_local_w4a16_rows(cw,cs,x.data()+size_t(t)*K,.375f,K,N,0,N,y.data()+size_t(t)*N,true,true);
    };
    run(false);const auto expected=y;run(true);
    if(std::memcmp(y.data(),expected.data(),y.size()*sizeof(float)))throw std::runtime_error("batch benchmark initial mismatch");
    const auto start=std::chrono::steady_clock::now();const auto cpu=std::clock();
    for(int i=0;i<repeats;++i)run(reuse,i%copies);
    const double cpu_ms=1000.*(std::clock()-cpu)/CLOCKS_PER_SEC;
    const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(std::memcmp(y.data(),expected.data(),y.size()*sizeof(float)))throw std::runtime_error("batch benchmark output changed");
    std::cout<<"scope=W4A16 tokens="<<tokens<<" reuse="<<reuse<<" repeats="<<repeats<<" weight_copies="<<copies<<" packed_bytes="<<(wb+sb)*copies<<" cpu_ms="<<cpu_ms<<" wall_ms="<<wall_ms<<"\n";
}
int main(int argc,char** argv){try {
    if((argc==5||argc==6)&&std::string(argv[1])=="--batch-bench") {
        const std::string mode=argv[2];if(mode!="separate"&&mode!="reuse")throw std::runtime_error("invalid batch mode");
        batch_bench(mode=="reuse",std::stoi(argv[3]),std::stoi(argv[4]),argc==6?std::stoi(argv[5]):1);return 0;
    }
    if(argc==4&&std::string(argv[1])=="--kernel-bench") {
        const int repeats=std::stoi(argv[3]);const std::string mode=argv[2];
        if((mode!="adjacent"&&mode!="planar")||repeats<1||repeats>1000000)throw std::runtime_error("invalid benchmark arguments");
        kernel_bench(repeats,mode=="planar");return 0;
    }
    if(argc!=1)throw std::runtime_error("usage: glm-fp4-w4a16-cpu-test [--kernel-bench adjacent|planar REPEATS | --batch-bench separate|reuse TOKENS REPEATS [WEIGHT_COPIES]]");
    for(int k:{32,64,128,2048,4096,16384})for(int pattern=0;pattern<5;++pattern)check(k,7,pattern);
    for(int pattern=0;pattern<5;++pattern)check(64,129,pattern,true);
    std::vector<uint8_t>w(32,0x11),s(4,56);std::vector<float>x(64,1),y(1,-123);
    strata::kernels::cpu::fp4_local_w4a16_rows(w.data(),s.data(),x.data(),.375f,64,1,0,1,y.data(),false,true);
    if(y[0]!=12.f)throw std::runtime_error("analytic result mismatch");
    y[0]=-123;bool rejected=false;
    try{strata::kernels::cpu::fp4_local_w4a16_rows(w.data(),s.data(),x.data(),1,16,1,0,1,y.data(),false,true);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected||y[0]!=-123)throw std::runtime_error("invalid extent wrote output");
    strata::kernels::cpu::fp4_local_w4a16_rows(w.data(),s.data(),x.data(),1,64,1,0,0,y.data(),false,true);
    if(y[0]!=-123)throw std::runtime_error("empty row range wrote output");
    for(int tokens:{0,17})for(bool simd:{false,true}) {
        bool bad=false;
        try{strata::kernels::cpu::fp4_local_w4a16_batch_rows(w.data(),s.data(),x.data(),1,64,1,tokens,0,1,y.data(),false,simd);}
        catch(const std::invalid_argument&){bad=true;}
        if(!bad||y[0]!=-123)throw std::runtime_error("invalid batch wrote output");
    }
    // Empty row ranges must leave every token's output untouched.
    std::vector<float> bx(3*64,1),by(3,-123);
    strata::kernels::cpu::fp4_local_w4a16_batch_rows(w.data(),s.data(),bx.data(),1,64,1,3,0,0,by.data(),true,true);
    for(float v:by)if(v!=-123)throw std::runtime_error("empty batch row range wrote output");
    std::cout<<"output_digest="<<output_digest<<"\n";
    std::cout<<"PASS: W4A16 scalar/SIMD FP64 bounds, packed-layout bit parity, finite code/scale pairs, outliers, row guards and extent checks\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
