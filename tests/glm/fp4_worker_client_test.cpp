// Loopback against the real worker and generated finite, uniform NVFP4 tensors.
#include "strata/net/glm_fp4_worker.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include <cmath>
#include <iostream>
#include <cstring>
int main(int argc,char** argv) {try {
    if(argc!=2)return 2;
    strata::net::GlmFp4WorkerConnection remote(argv[1],std::string(64,'a'),std::string(64,'b'),false,false);
    auto bytes=remote.fetch(0,size_t(4096)*2048/2);
    for(auto value:bytes)if(value!=0x11)throw std::runtime_error("fetched weight bytes changed");
    const float hidden=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round((.5f/(1.f+std::exp(-.5f)))*.5f));
    const float expected=hidden*1024.f;
    for(unsigned tokens:{1u,2u,4u,8u,1u}) {
        const std::vector<float> input(size_t(tokens)*4096,1.f/4096);
        const auto result=remote.run(3,0,input);
        if(result.size()!=input.size())throw std::runtime_error("output shape mismatch");
        for(float value:result)if(std::memcmp(&value,&expected,sizeof(float)))
            throw std::runtime_error("W4A16 output differs from uniform-weight analytic reference");
    }
    std::cout<<"PASS real worker: tensor fetch, W4A16 widths 1/2/4/8, repeat and shutdown\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
