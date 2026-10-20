// Bounded correctness and same-process A/B timings; no production defaults change.
#include "strata/kernels/fp4_local.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
using strata::kernels::Fp4LocalGpu;
static void test(const strata::artifact::GlmFp4Matrix& w,const std::string& name,bool bench) {
    Fp4LocalGpu gpu(w);
    for(int m:{1,7,16}) {
        std::vector<float> a(size_t(m)*w.cols);
        for(size_t i=0;i<a.size();++i)a[i]=float(int((i*37+i/w.cols*19)%257)-128)/128;
        auto base=gpu.run(a,m,true,false),tiled=gpu.run(a,m,true,true);
        if(base!=tiled)throw std::runtime_error("tiled and baseline outputs differ: "+name);
        std::printf("EQUAL %s m=%d n=%zu k=%zu\n",name.c_str(),m,w.rows,w.cols);
        if(bench) {
            std::vector<float> times[2];
            for(int repeat=0;repeat<10;++repeat) {
                for(int j=0;j<2;++j) {
                    int mode=j^(repeat&1);float ms;
                    gpu.run(a,m,false,mode,&ms);
                    if(repeat>=2)times[mode].push_back(ms);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            for(auto& t:times)std::sort(t.begin(),t.end());
            float b=(times[0][3]+times[0][4])/2,t=(times[1][3]+times[1][4])/2;
            std::printf("TIMING %s m=%d baseline_ms=%.6f tiled_ms=%.6f speedup=%.3f repeats=8\n",name.c_str(),m,b,t,b/t);
        }
    }
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("usage: glm-fp4-rdna4-test PARTIAL_MODEL_DIR");
        std::setvbuf(stdout,nullptr,_IOLBF,0);strata::kernels::fp4_local_device_info();
        // Asymmetric data, partial N tiles, K16 tail, every FP4 code and finite scale.
        for(size_t n:{size_t(17),size_t(32),size_t(47)})for(size_t k:{size_t(16),size_t(48)}) {
            strata::artifact::GlmFp4Matrix w;w.rows=n;w.cols=k;w.global=0.0001f;
            w.codes.resize(n*k/2);w.scales.resize(n*k/16);
            for(size_t i=0;i<w.codes.size();++i)w.codes[i]=uint8_t(i*73+19);
            for(size_t i=0;i<w.scales.size();++i)w.scales[i]=uint8_t(i%127);
            test(w,"synthetic",false);
        }
        strata::artifact::Safetensors ts(argv[1]);
        for(int layer:{3,4})for(const char* p:{"gate_proj","up_proj","down_proj"}) {
            std::string name="model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+p;
            test(strata::artifact::glm_load_fp4(ts,name),name,true);
        }
        std::puts("PASS RDNA4 tiled: 36 projection cases, bit-identical to baseline");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
