#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include "ggml.h"
#include <random>
#include <iostream>
#include <cmath>
using namespace strata::kernels::cpu;
int main(){try {
    ExpertPool pool(15,true);if(!pool.numa_rows_available())return 77;
    const auto cores=physical_cores(false);auto old=pin_current_thread(cores.front());
    for(int down_type:{11,23}) {
    NativeFmt f;std::string error;if(!native_fmt(10,down_type,256,512,f,error))throw std::runtime_error(error);f.swiglu_limit=10;
    constexpr int experts=2,nt=4;std::mt19937 rng(817);
    std::vector<uint8_t> g(experts*f.up_off),u(g.size()),d(experts*(f.bytes-f.down_off));
    std::vector<float> values(256*512);for(auto* dest:{&g,&u,&d})for(int e=0;e<experts;++e) {
        for(auto& v:values)v=float(int(rng()%2001)-1000)*.0001f;
        const bool down=dest==&d;ggml_quantize_chunk(down?ggml_type(down_type):GGML_TYPE_Q2_K,values.data(),dest->data()+size_t(e)*(down?f.bytes-f.down_off:f.up_off),0,down?256:512,down?512:256,nullptr);
    }
    std::vector<std::vector<uint8_t>> acts(nt,std::vector<uint8_t>(f.act_bytes));std::vector<float> input(256);
    for(int t=0;t<nt;++t){for(auto& v:input)v=float(int(rng()%2001)-1000)*.001f;native_quant_act(f,input.data(),acts[t].data());}
    std::vector<float> out(experts*nt*256),reference;
    for(bool huge:{false,true})for(int layout=0;layout<=2;++layout) {
        NumaTensor ng(g.data(),f.gu_row,512,experts,pool.numa_cores(),huge,layout?10:0,256),nu(u.data(),f.gu_row,512,experts,pool.numa_cores(),huge,layout?10:0,256),nd(d.data(),f.d_row,256,experts,pool.numa_cores(),huge,layout&&down_type==11?11:0,512);
        if(huge)for(int node=0;node<2;++node)if(reinterpret_cast<uintptr_t>(ng.shard(0,node))%(2*1024*1024))throw std::runtime_error("huge arena alignment mismatch");
        std::vector<uint8_t> copied(f.up_off);ng.copy(1,copied.data());if(std::memcmp(copied.data(),g.data()+f.up_off,f.up_off))throw std::runtime_error("owned tiled copy mismatch");
        f.q23_layout=layout;std::vector<ExpertJobMulti> jobs(experts);
        for(int e=0;e<experts;++e){auto& j=jobs[e];j.nt=nt;j.blob=g.data()+e*f.up_off;for(int node=0;node<2;++node)j.numa[node]={ng.shard(e,node),nu.shard(e,node),nd.shard(e,node)};for(int t=0;t<nt;++t){j.nact[t]=acts[t].data();j.out[t]=out.data()+(e*nt+t)*256;}}
        f.fuse_h_quant=false;pool.run_split_multi_native(f,jobs.data(),experts);reference=out;
        f.fuse_h_quant=true;
        for(int repeat=0;repeat<8;++repeat){pool.run_split_multi_native(f,jobs.data(),experts);if(out!=reference)throw std::runtime_error("fused hidden quantization changed outputs");}
    }
    std::vector<float> weights(8),result(4*256),expected(result.size());for(auto& v:weights)v=float(int(rng()%201)-100)*.01f;
    for(int t=0;t<4;++t)for(int r=0;r<256;++r)for(int k=0;k<2;++k)expected[t*256+r]+=weights[t*2+k]*out[(t*2+k)*256+r];
    pool.reduce_routed(out.data(),weights.data(),result.data(),4,2,256);
    for(size_t i=0;i<result.size();++i)if(std::abs(result[i]-expected[i])>1e-6f*(1+std::abs(expected[i])))throw std::runtime_error("routed reduction mismatch");
    }
    restore_thread_affinity(old);std::cout<<"Q23 pool PASS owned row/tile copies, fused quantization, fixed-order reduction\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
