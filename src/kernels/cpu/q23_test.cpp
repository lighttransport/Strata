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
using namespace strata::kernels::cpu;
int main(){try {
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
