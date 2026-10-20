#include "strata/artifact/glm_fp4.hpp"
#include <cstdio>
int main(int argc,char** argv) {
    try {
        if(argc<3)throw std::runtime_error("usage: glm_fp4_inspect DIRECTORY PROJECTION_PREFIX...");
        strata::artifact::Safetensors ts(argv[1]);
        for(int i=2;i<argc;++i) {
            auto m=strata::artifact::glm_load_fp4(ts,argv[i]);auto v=m.dequantize();double sum=0,maxabs=0;
            for(float x:v) {if(!std::isfinite(x))throw std::runtime_error("nonfinite weight");sum+=double(x)*x;maxabs=std::max(maxabs,double(std::abs(x)));}
            std::printf("%s rows=%zu cols=%zu global=%.9g bytes=%zu l2=%.9g max_abs=%.9g\n",argv[i],m.rows,m.cols,m.global,m.bytes(),std::sqrt(sum),maxabs);
        }
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
