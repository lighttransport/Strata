#include "strata/host/q3_layout.h"
#include <vector>
#include <chrono>
#include <ctime>
#include <cstring>
#include <iostream>
#include <stdexcept>
extern "C" int candidate_expand(const uint8_t*,size_t,uint8_t*,size_t);
int main(){try {
    for(size_t blocks:{size_t(2048),size_t(32768),size_t(262144)}) {
        std::vector<uint8_t> source(blocks*110),reference(blocks*114),candidate(reference.size());
        uint32_t state=39179;for(auto& b:source){state=1664525u*state+1013904223u;b=uint8_t(state>>24);}
        if(strata_q3k_expand_scales(source.data(),source.size(),reference.data(),reference.size())||
           candidate_expand(source.data(),source.size(),candidate.data(),candidate.size())||candidate!=reference)
            throw std::runtime_error("conversion parity failed");
        const int calls=blocks==262144?4:32;
        for(int round=0;round<12;++round)for(int order=0;order<2;++order) {
            bool selected=bool(order^(round%2));auto fn=selected?candidate_expand:strata_q3k_expand_scales;
            auto& output=selected?candidate:reference;
            const auto wall=std::chrono::steady_clock::now();const auto cpu=std::clock();
            for(int i=0;i<calls;++i)if(fn(source.data(),source.size(),output.data(),output.size()))throw std::runtime_error("conversion failed");
            const double cpu_ms=1000.*(std::clock()-cpu)/CLOCKS_PER_SEC;
            const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count();
            if(candidate!=reference)throw std::runtime_error("timed output changed");
            std::cout<<"blocks="<<blocks<<" round="<<round<<" candidate="<<selected<<" calls="<<calls
                     <<" cpu_ms="<<cpu_ms<<" wall_ms="<<wall_ms<<" input_bytes="<<source.size()<<" output_bytes="<<output.size()<<'\n';
        }
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
