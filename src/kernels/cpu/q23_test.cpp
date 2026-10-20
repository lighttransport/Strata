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
    auto run=[&](int copy) {
        const auto* weights=copies==1?source.data():ring.data()+size_t(copy)*source.size();
        if(layout=="row")q23_rows(ty,weights,stride,cols,ap,nt,op,0,rows);
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
    const auto start=std::chrono::steady_clock::now();const auto cpu=std::clock();
    for(int i=0;i<repeats;++i)run(i%copies);
    const double cpu_ms=1000.*(std::clock()-cpu)/CLOCKS_PER_SEC;
    const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    for(int t=0;t<nt;++t)if(std::memcmp(y[t].data(),expected[t].data(),rows*sizeof(float)))throw std::runtime_error("benchmark output changed");
    std::cout<<"scope="<<(copies==1?"synthetic_cache_resident":"synthetic_weight_ring")<<" ty="<<ty<<" layout="<<layout<<" nt="<<nt<<" repeats="<<repeats
             <<" copies="<<copies<<" working_set_bytes="<<source.size()*size_t(copies)
             <<" packed_bytes="<<q.size()<<" wall_ms="<<wall_ms<<" cpu_ms="<<cpu_ms
             <<" payload_GB_s="<<double(q.size())*repeats/wall_ms/1e6
             <<" gweight_token_s="<<double(rows)*cols*nt*repeats/cpu_ms/1e6<<" checksum="<<y[0][17]<<"\n";
}
int main(int argc,char** argv){try {
    if((argc==6||argc==7)&&std::string(argv[1])=="--bench") {
        bench(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),argv[5],argc==7?std::stoi(argv[6]):1);return 0;
    }
    const bool dump=argc==2&&std::string(argv[1])=="--dump";
    if(argc>1&&!dump)throw std::runtime_error("usage: q23_test [--dump | --bench TYPE(10|11) TOKENS(1..4) REPEATS row|direct|lut [WEIGHT_COPIES(1..1024)]]");
    ggml_cpu_init();std::mt19937 rng(731);int checked=0;
    for(int ty:{10,11})for(int cols:{256,2048,4096})for(int pattern=0;pattern<4;++pattern) {
        constexpr int rows=64,nt=4;const auto type=ggml_type(ty);
        const size_t stride=ggml_row_size(type,cols),astride=ggml_row_size(GGML_TYPE_Q8_K,cols);
        std::vector<float> w(rows*cols),decoded(w.size());std::vector<uint8_t> q(rows*stride);
        for(auto& v:w)v=pattern==0?0.f:float(int(rng()%20001)-10000)*(pattern==1?.00001f:.001f);
        ggml_quantize_chunk(type,w.data(),q.data(),0,rows,cols,nullptr);
        ggml_get_type_traits(type)->to_float(q.data(),decoded.data(),decoded.size());
        std::vector<uint8_t> packed(q.size()),roundtrip(q.size());
        q23_pack(ty,q.data(),packed.data(),cols,rows);q23_unpack(ty,packed.data(),roundtrip.data(),cols,rows);
        if(roundtrip!=q)throw std::runtime_error("packed Q23 changed GGML bytes");
        std::vector<std::vector<uint8_t>> acts(nt,std::vector<uint8_t>(astride,0));
        std::vector<std::vector<float>> a(nt,std::vector<float>(cols)),got(nt,std::vector<float>(rows,-9876));
        const void* ap[nt];float* op[nt];
        for(int t=0;t<nt;++t) {
            for(int k=0;k<cols;++k)a[t][k]=pattern==0?0.f:pattern==3?(k%2?1e3f:-1e3f):float(int(rng()%2001)-1000)*.003f;
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(a[t].data(),acts[t].data(),cols);
            const auto* qa=reinterpret_cast<const block_q8_K*>(acts[t].data());
            for(int k=0;k<cols;++k)a[t][k]=qa[k/256].d*qa[k/256].qs[k%256];
            ap[t]=acts[t].data();op[t]=got[t].data();
        }
        q23_rows(ty,q.data(),stride,cols,ap,nt,op,1,rows-1);
        if(dump)for(const auto& row:got)std::cout.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(float));
        for(int t=0;t<nt;++t)for(int r=1;r<rows-1;++r) {
            double ref=0,norm=0;for(int k=0;k<cols;++k){double v=double(decoded[r*cols+k])*a[t][k];ref+=v;norm+=std::abs(v);}
            if(!std::isfinite(got[t][r])||std::abs(got[t][r]-ref)>1e-5*norm+1e-5)throw std::runtime_error("Q23 scalar dot mismatch");
            ++checked;
        }
        for(int width=1;width<=nt;++width) {
            std::vector<std::vector<float>> out(width,std::vector<float>(rows,-9876));float* ptr[nt];for(int t=0;t<width;++t)ptr[t]=out[t].data();
            q23_rows(ty,q.data(),stride,cols,ap,width,ptr,1,rows-1);
            for(int t=0;t<width;++t)if(out[t]!=got[t])throw std::runtime_error("Q23 verification width changed output");
        }
        std::vector<std::vector<float>> packed_ref;
        for(bool lut:{false,true})for(int width=1;width<=nt;++width) {
            std::vector<std::vector<float>> output(width,std::vector<float>(rows,-9876));float* ptr[nt];for(int t=0;t<width;++t)ptr[t]=output[t].data();
            q23_packed_rows(ty,packed.data(),cols,ap,width,ptr,1,rows-1,lut);
            for(int t=0;t<width;++t)for(int r=1;r<rows-1;++r) {
                double ref=0,norm=0;for(int k=0;k<cols;++k){double v=double(decoded[r*cols+k])*a[t][k];ref+=v;norm+=std::abs(v);}
                if(!std::isfinite(output[t][r])||std::abs(output[t][r]-ref)>1e-5*norm+1e-5)throw std::runtime_error("packed Q23 scalar dot mismatch");
            }
            if(!lut && width==nt)packed_ref=output;
            if(lut)for(int t=0;t<width;++t)if(output[t]!=packed_ref[t])throw std::runtime_error("packed Q23 lookup/direct or width mismatch");
        }
    }
    std::cout<<"Q23 PASS scalar dots="<<checked<<" widths=1..4 partial rows, zero and extreme activations\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
