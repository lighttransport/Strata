// CPU-only activation quantization timing; excludes expert projections and routing.
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::kernels::cpu;
int main(){try {
    if(!native_experts_available())return 77;
    ggml_cpu_init();
    constexpr int banks=32, calls=8192;
    for(bool fp4:{false,true})for(bool hidden:{false,true})for(int nt:{1,4}) {
        NativeFmt f;std::string error;
        if(!native_fmt(fp4?kNvfp4Type:10,fp4?kNvfp4Type:11,4096,2048,f,error))throw std::runtime_error(error);
        const int n=hidden?f.n_ff:f.n_embd;
        const size_t bytes=hidden?f.h_bytes:f.act_bytes;
        std::vector<float> x(size_t(banks)*nt*n);
        std::mt19937 rng(731);
        for(auto& v:x)v=float(int(rng()%2049)-1024)/1024;
        std::vector<uint8_t> expected(size_t(banks)*nt*bytes),out(nt*bytes);
        auto run=[&](int b) {
            for(int t=0;t<nt;++t) {
                const float* input=x.data()+(size_t(b)*nt+t)*n;
                if(hidden)native_quant_h(f,input,out.data()+t*bytes);
                else native_quant_act(f,input,out.data()+t*bytes);
            }
        };
        for(int b=0;b<banks;++b) {
            for(int t=0;t<nt;++t)ggml_get_type_traits_cpu(ggml_type(hidden?f.d_act:f.gu_act))->from_float(
                x.data()+(size_t(b)*nt+t)*n,expected.data()+(size_t(b)*nt+t)*bytes,n);
            run(b);
            if(std::memcmp(out.data(),expected.data()+size_t(b)*nt*bytes,out.size()))throw std::runtime_error("quantization differs from GGML");
        }
        std::vector<double> times;
        for(int rep=0;rep<7;++rep) {
            const auto start=std::chrono::steady_clock::now();
            for(int i=0;i<calls;++i)run(i%banks);
            times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/calls);
            if(std::memcmp(out.data(),expected.data()+size_t((calls-1)%banks)*nt*bytes,out.size()))throw std::runtime_error("timed quantization differs");
        }
        std::sort(times.begin(),times.end());
        std::cout<<"format="<<(fp4?"nvfp4":"q2k_q3k")<<" phase="<<(hidden?"hidden":"input")<<" tokens="<<nt
                 <<" values_per_token="<<n<<" banks="<<banks<<" calls_per_sample="<<calls<<" median_us="<<times[3]<<" samples_us=";
        for(double t:times)std::cout<<t<<",";
        std::cout<<" bit_exact=1\n";
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
