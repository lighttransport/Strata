#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include "ggml.h"
#include <random>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <memory>
using namespace strata::kernels::cpu;
// Synthetic one-layer CPU service; NUMA-owned rows when available, otherwise complete blobs.
int bench(int nt,int workers,int repeats,bool flow=false,bool sweep=false,bool quant_sweep=false,int E=8,int banks=1) {
    if((nt!=1&&nt!=4)||workers<1||workers>31||repeats<1||repeats>20||E<1||E>8||banks<1||banks>16)
        throw std::runtime_error("invalid pool benchmark bounds");
    ExpertPool pool(workers,true);const bool owned=pool.numa_rows_available();
    if(quant_sweep&&!owned)throw std::runtime_error("fusion sweep requires NUMA-owned rows");
    const auto previous=pin_current_thread(planned_host_core());
    constexpr int H=4096,F=2048;
    NativeFmt f;std::string error;
    if(!native_fmt(10,11,H,F,f,error))throw std::runtime_error(error);
    f.swiglu_limit=10;
    std::mt19937 rng(817);
    std::vector<uint8_t> g(E*f.up_off),u(g.size()),d(E*(f.bytes-f.down_off));
    std::vector<float> values(H*F);
    for(auto* dest:{&g,&u,&d})for(int e=0;e<E;++e) {
        for(auto& v:values)v=float(int(rng()%2001)-1000)*.0001f;
        const bool down=dest==&d;
        ggml_quantize_chunk(down?GGML_TYPE_Q3_K:GGML_TYPE_Q2_K,values.data(),dest->data()+size_t(e)*(down?f.bytes-f.down_off:f.up_off),0,down?H:F,down?F:H,nullptr);
    }
    std::vector<std::unique_ptr<NumaTensor>> ng,nu,nd;
    std::vector<uint8_t> complete;
    if(owned) {
        for(int b=0;b<banks;++b) {
            ng.push_back(std::make_unique<NumaTensor>(g.data(),f.gu_row,F,E,pool.numa_cores(),false));
            nu.push_back(std::make_unique<NumaTensor>(u.data(),f.gu_row,F,E,pool.numa_cores(),false));
            nd.push_back(std::make_unique<NumaTensor>(d.data(),f.d_row,H,E,pool.numa_cores(),false));
        }
    } else {
        complete.resize(size_t(banks)*E*f.bytes);
        for(int b=0;b<banks;++b)for(int e=0;e<E;++e) {
            auto* blob=complete.data()+(size_t(b)*E+e)*f.bytes;
            std::memcpy(blob,g.data()+size_t(e)*f.up_off,f.up_off);
            std::memcpy(blob+f.up_off,u.data()+size_t(e)*f.up_off,f.up_off);
            std::memcpy(blob+f.down_off,d.data()+size_t(e)*(f.bytes-f.down_off),f.bytes-f.down_off);
        }
    }
    std::vector<std::vector<uint8_t>> acts(nt,std::vector<uint8_t>(f.act_bytes));
    std::vector<float> input(H),out(E*nt*H),sum(nt*H),weights(nt*E,1.f/E);
    for(int t=0;t<nt;++t) {
        for(auto& v:input)v=float(int(rng()%2001)-1000)*.001f;
        native_quant_act(f,input.data(),acts[t].data());
    }
    std::vector<ExpertJobMulti> jobs(E);
    for(int e=0;e<E;++e) {
        auto& j=jobs[e];j.nt=nt;j.blob=owned?g.data()+size_t(e)*f.up_off:complete.data()+size_t(e)*f.bytes;
        if(owned)for(int node=0;node<2;++node)j.numa[node]={ng[0]->shard(e,node),nu[0]->shard(e,node),nd[0]->shard(e,node)};
        for(int t=0;t<nt;++t){j.nact[t]=acts[t].data();j.out[t]=out.data()+(t*E+e)*H;}
    }
    auto split=[&] {
        pool.run_split_multi_native(f,jobs.data(),E);
        pool.reduce_routed(out.data(),weights.data(),sum.data(),nt,E,H);
    };
    split();const auto reference=out,reference_sum=sum;
    // Physically separate copies retain the same arithmetic reference. Each call,
    // including warmup, advances the bank; this is not a routing-trace replay.
    size_t next_bank=0;
    auto run=[&] {
        const size_t bank=next_bank++%banks;
        for(int e=0;e<E;++e) {
            if(owned)for(int node=0;node<2;++node)
                jobs[e].numa[node]={ng[bank]->shard(e,node),nu[bank]->shard(e,node),nd[bank]->shard(e,node)};
            else jobs[e].blob=complete.data()+(bank*E+e)*f.bytes;
        }
        if(flow)pool.run_layer_native(f,jobs.data(),E,out.data(),weights.data(),sum.data(),nt,E);
        else split();
    };
    run();
    if(out!=reference||sum!=reference_sum)throw std::runtime_error("pool mode differs from split reference");
    for(float v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite pool benchmark output");
    std::cout<<"scope=synthetic_cpu_expert_layer H="<<H<<" F="<<F<<" experts="<<E<<" tokens="<<nt
             <<" workers="<<workers<<" owned="<<owned<<" mode="<<(quant_sweep?"flow-quant-sweep":flow?"flow":"split")<<" packed_bytes="<<E*f.bytes<<" banks="<<banks<<" rotating_bytes="<<size_t(banks)*E*f.bytes<<" input_quantization=excluded\n";
    constexpr int task_counts[]={1,2,3,4,6,8};
    constexpr int quant_tasks[]={3,4,6};
    const int variants=quant_sweep?6:sweep?6:1;
    for(int r=0;r<repeats;++r)for(int order=0;order<variants;++order) {
        const int variant=(r+order)%variants;
        const int tasks=quant_sweep?quant_tasks[variant%3]:sweep?task_counts[variant]:0;
        if(quant_sweep)f.fuse_h_quant=variant>=3;
        if(sweep||quant_sweep) {
            pool.set_native_tasks_per_thread(tasks);
            run(); // Warm the selected schedule before timing; no worker is active at the setter.
        }
        const double gu=pool.ms_multi_gu,q=pool.ms_multi_q,down=pool.ms_multi_down;
        const auto start=std::chrono::steady_clock::now();run();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(std::memcmp(out.data(),reference.data(),out.size()*sizeof(float))||
           std::memcmp(sum.data(),reference_sum.data(),sum.size()*sizeof(float)))throw std::runtime_error("pool benchmark output changed");
        uint64_t hash=14695981039346656037ULL;
        for(const auto* v:{&out,&sum})for(size_t i=0;i<v->size()*sizeof(float);++i){hash^=reinterpret_cast<const uint8_t*>(v->data())[i];hash*=1099511628211ULL;}
        std::cout<<"round="<<r<<" ms="<<ms;
        if(sweep||quant_sweep)std::cout<<" tasks_per_thread="<<tasks;
        if(quant_sweep)std::cout<<" fuse_h_quant="<<f.fuse_h_quant;
        if(!flow)std::cout<<" gu_ms="<<pool.ms_multi_gu-gu<<" quant_ms="<<pool.ms_multi_q-q<<" down_ms="<<pool.ms_multi_down-down;
        std::cout<<" output_hash="<<hash<<"\n";
    }
    if(previous.valid)restore_thread_affinity(previous);
    return 0;
}
int main(int argc,char** argv){try {
    if((argc>=5&&argc<=8)&&std::string(argv[1])=="--bench") {
        const std::string mode=argc>=6?argv[5]:"split";
        if(mode!="split"&&mode!="flow"&&mode!="flow-sweep"&&mode!="flow-quant-sweep")throw std::runtime_error("invalid pool benchmark mode");
        return bench(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),mode!="split",mode=="flow-sweep",mode=="flow-quant-sweep",argc>=7?std::stoi(argv[6]):8,argc==8?std::stoi(argv[7]):1);
    }
    if(argc!=1)throw std::runtime_error("usage: q23_pool_test [--bench TOKENS(1|4) WORKERS REPEATS [split|flow|flow-sweep|flow-quant-sweep [EXPERTS(1..8) [BANKS(1..16)]]]]");
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
    std::vector<float> original_output;
    const NativeFmt original_format=f;
    for(bool huge:{false,true})for(int layout=0;layout<=(down_type==11?3:2);++layout) {
        f=original_format;
        const bool expanded=layout==3,tiled=layout==1||layout==2;
        if(down_type==11 && !tiled)f.canon=true;
        if(expanded) {f.q3_expanded_down=true;f.d_row=size_t(f.n_ff/256)*114;f.bytes=f.down_off+f.d_row*f.n_embd;}

        NumaTensor ng(g.data(),f.gu_row,512,experts,pool.numa_cores(),huge,tiled?10:0,256),nu(u.data(),f.gu_row,512,experts,pool.numa_cores(),huge,tiled?10:0,256),nd(d.data(),f.d_row,256,experts,pool.numa_cores(),huge,tiled&&down_type==11?11:0,512,expanded);
        if(expanded) {
            std::vector<uint8_t> copied_down(f.d_row*256),expected_down(copied_down.size());
            const size_t old_down=original_format.d_row*256;
            nd.copy(1,copied_down.data());
            if(strata_q3k_expand_scales(d.data()+old_down,old_down,expected_down.data(),expected_down.size()) || copied_down!=expected_down)
                throw std::runtime_error("direct NUMA Q3 expanded copy mismatch");
        }
        if(huge)for(int node=0;node<2;++node)if(reinterpret_cast<uintptr_t>(ng.shard(0,node))%(2*1024*1024))throw std::runtime_error("huge arena alignment mismatch");
        std::vector<uint8_t> copied(f.up_off);ng.copy(1,copied.data());if(std::memcmp(copied.data(),g.data()+f.up_off,f.up_off))throw std::runtime_error("owned tiled copy mismatch");
        f.q23_layout=tiled?layout:0;std::vector<ExpertJobMulti> jobs(experts);
        for(int e=0;e<experts;++e){auto& j=jobs[e];j.nt=nt;j.blob=g.data()+e*f.up_off;for(int node=0;node<2;++node)j.numa[node]={ng.shard(e,node),nu.shard(e,node),nd.shard(e,node)};for(int t=0;t<nt;++t){j.nact[t]=acts[t].data();j.out[t]=out.data()+(e*nt+t)*256;}}
        f.fuse_h_quant=false;pool.run_split_multi_native(f,jobs.data(),experts);reference=out;
        if(layout==0)original_output=out;
        if(expanded && out!=original_output)throw std::runtime_error("direct NUMA Q3 expansion changed outputs");
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
