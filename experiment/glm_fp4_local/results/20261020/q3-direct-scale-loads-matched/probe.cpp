#include "strata/kernels/cpu/q23_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <ctime>
#include <cstdlib>
namespace strata::kernels::cpu {
void baseline_q23_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int);
void q23_candidate_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int);
}
using namespace strata::kernels::cpu;
// CPU kernel screen; optional disjoint weight copies exceed cache capacity.
// Packing, replication, activation quantization and reference checks are untimed.
void bench(int ty,int nt,int repeats,const std::string& layout,int copies=1) {
    if(copies<1||copies>1024||(ty!=10&&ty!=11)||nt<1||nt>4||repeats<1||repeats>1000000||
       (layout!="row"&&layout!="direct"&&layout!="lut"))throw std::runtime_error("invalid benchmark arguments");
    ggml_cpu_init();std::mt19937 rng(731);
    constexpr int cols=4096,rows=128;
    const auto type=ggml_type(ty);
    const size_t stride=ggml_row_size(type,cols),astride=ggml_row_size(GGML_TYPE_Q8_K,cols);
    std::vector<float>w(rows*cols),decoded(w.size());std::vector<uint8_t>q(rows*stride),packed(q.size());
    for(auto& v:w)v=float(int(rng()%20001)-10000)*.001f;
    ggml_quantize_chunk(type,w.data(),q.data(),0,rows,cols,nullptr);
    ggml_get_type_traits(type)->to_float(q.data(),decoded.data(),decoded.size());
    q23_pack(ty,q.data(),packed.data(),cols,rows);
    std::vector<std::vector<uint8_t>> acts(nt,std::vector<uint8_t>(astride));
    std::vector<std::vector<float>> a(nt,std::vector<float>(cols)),y(nt,std::vector<float>(rows));
    const void* ap[4];float* op[4];
    for(int t=0;t<nt;++t) {
        for(auto& v:a[t])v=float(int(rng()%2001)-1000)*.003f;
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(a[t].data(),acts[t].data(),cols);
        if(ty==11) {
            auto* blocks=reinterpret_cast<block_q8_K*>(acts[t].data());
            for(int b=0;b<cols/256;++b) for(auto& q:blocks[b].qs) if(q==-128)q=-127;
            if(std::getenv("STRATA_PROBE_MIN128"))blocks[0].qs[0]=-128;
        }
        const auto* qa=reinterpret_cast<const block_q8_K*>(acts[t].data());
        for(int j=0;j<cols;++j)a[t][j]=qa[j/256].d*qa[j/256].qs[j%256];
        ap[t]=acts[t].data();op[t]=y[t].data();
    }
    const auto& source=layout=="row"?q:packed;
    std::vector<uint8_t> ring;
    if(copies>1) {
        ring.resize(source.size()*size_t(copies));
        for(int i=0;i<copies;++i)std::memcpy(ring.data()+size_t(i)*source.size(),source.data(),source.size());
        for(int i=0;i<copies;++i)if(std::memcmp(ring.data()+size_t(i)*source.size(),source.data(),source.size()))
            throw std::runtime_error("weight replication mismatch");
    }
    bool candidate=false;
    auto run=[&](int copy) {
        const auto* weights=copies==1?source.data():ring.data()+size_t(copy)*source.size();
        if(layout=="row")(candidate?q23_candidate_rows:baseline_q23_rows)(ty,weights,stride,cols,ap,nt,op,0,rows);
        else q23_packed_rows(ty,weights,cols,ap,nt,op,0,rows,layout=="lut");
    };
    run(0);
    for(int t=0;t<nt;++t)for(int r=0;r<rows;++r) {
        double ref=0,norm=0;
        for(int j=0;j<cols;++j){const double v=double(decoded[r*cols+j])*a[t][j];ref+=v;norm+=std::abs(v);}
        if(!std::isfinite(y[t][r])||std::abs(y[t][r]-ref)>1e-5*norm+1e-5)throw std::runtime_error("benchmark FP64 check failed");
    }
    const auto expected=y;
    // Warm every physical copy and verify dispatch into every ring slot.
    for(int i=0;i<copies;++i) {
        run(i);
        for(int t=0;t<nt;++t)if(std::memcmp(y[t].data(),expected[t].data(),rows*sizeof(float)))
            throw std::runtime_error("replicated weight output mismatch");
    }
    for(int round=0;round<12;++round)for(int order=0;order<2;++order) {
        candidate=bool(order^(round%2));
        run(0);
        const auto start=std::chrono::steady_clock::now();const auto cpu=std::clock();
        for(int i=0;i<repeats;++i)run((round*repeats+i)%copies);
        const double cpu_ms=1000.*(std::clock()-cpu)/CLOCKS_PER_SEC;
        const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        for(int t=0;t<nt;++t)if(std::memcmp(y[t].data(),expected[t].data(),rows*sizeof(float)))
            throw std::runtime_error("paired candidate output differs from baseline");
        std::cout<<"round="<<round<<" candidate="<<candidate<<" nt="<<nt<<" copies="<<copies
                 <<" calls="<<repeats<<" bytes="<<source.size()*size_t(copies)
                 <<" cpu_ms="<<cpu_ms<<" wall_ms="<<wall_ms<<"\n";
    }
}
int main(int argc,char** argv){try {
    if(argc!=3 && argc!=4)throw std::runtime_error("usage: paired_probe TOKENS COPIES [TYPE=10|11]");
    bench(argc==4?std::stoi(argv[3]):10,std::stoi(argv[1]),64,"row",std::stoi(argv[2]));
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
