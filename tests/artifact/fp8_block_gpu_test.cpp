// FP8 release dequantization on the GPU (fp8_block_to_float) against strata/artifact/fp8_experts.hpp on the host:
// every one of the 256 E4M3 codes, random scales including tiny and huge ones, partial 128-blocks; bit for bit.
#include "strata/artifact/fp8_experts.hpp"
#include "strata/kernels/exl3.hpp"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"gpu: %s\n",cudaGetErrorString(e));std::exit(2);}}
int main() {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count){std::printf("no GPU\n");return 77;}
    std::mt19937 rng(20261017);int fails=0;
    for(auto [rows,cols]:{std::pair{2048,4096},std::pair{4096,2048},std::pair{300,200},std::pair{1,256}}) {
        const int srows=(rows+127)/128,scols=(cols+127)/128;
        std::vector<uint8_t> w(size_t(rows)*cols);for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i<256?i:rng());
        std::vector<float> s(size_t(srows)*scols);std::uniform_real_distribution<float> exp(-40.f,20.f);
        for(auto& v:s)v=std::ldexp(1.0f+float(rng()%1000)/1000.f,int(exp(rng)));
        std::vector<float> ref(w.size()),got(w.size());
        for(int r=0;r<rows;++r)for(int c=0;c<cols;++c)ref[size_t(r)*cols+c]=strata::artifact::fp8::e4m3(w[size_t(r)*cols+c])*s[(r/128)*scols+c/128];
        uint8_t* dw;float *ds,*dout;
        check(cudaMalloc(&dw,w.size()));check(cudaMalloc(&ds,s.size()*4));check(cudaMalloc(&dout,got.size()*4));
        check(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(ds,s.data(),s.size()*4,cudaMemcpyHostToDevice));
        strata::kernels::fp8_block_to_float(dw,ds,dout,rows,cols,nullptr);check(cudaDeviceSynchronize());
        check(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));
        size_t diff=0;for(size_t i=0;i<got.size();++i){uint32_t a,b;std::memcpy(&a,&got[i],4);std::memcpy(&b,&ref[i],4);
            const bool nan=std::isnan(got[i])&&std::isnan(ref[i]);diff+=!nan&&a!=b;}
        std::printf("%dx%d: %zu of %zu values differ  %s\n",rows,cols,diff,got.size(),diff?"FAIL":"ok");fails+=diff!=0;
        cudaFree(dw);cudaFree(ds);cudaFree(dout);
    }
    std::printf("fp8_block_gpu_test: %d failures\n",fails);return fails?1:0;
}
