#include "strata/artifact/glm_exl3.hpp"
#include <iostream>
#include <chrono>
int main(int argc,char** argv) {
    try {
        if(argc!=2 && argc!=5)throw std::runtime_error("usage: strata-glm-exl3-check MODEL_DIR [LAYER EXPERT PROJECTION]");
        strata::artifact::GlmExl3 model(argv[1]);
        constexpr double gib=1073741824.;
        std::cout<<"shards="<<model.tensors.shards()<<" tensors="<<model.tensors.tensors().size()<<" payload_bytes="<<model.tensors.bytes()
                 <<"\nmain_experts_GiB="<<model.experts_bytes/gib<<" main_fixed_GiB="<<model.fixed_bytes/gib
                 <<" draft_GiB="<<model.draft_bytes/gib<<" vision_GiB="<<model.vision_bytes/gib<<'\n';
        if(argc==5) {
            auto m=model.load(std::stoul(argv[2]),std::stoul(argv[3]),argv[4]);std::vector<float>x(m.in),y(m.out);
            for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.01f);
            auto start=std::chrono::steady_clock::now();m.apply(x.data(),y.data());double sum=0;
            for(float f:y){if(!std::isfinite(f))throw std::runtime_error("nonfinite projection result");sum+=f;}
            std::cout<<"projection_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" checksum="<<sum<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
