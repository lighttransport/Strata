#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include "ggml.h"
#include "strata/host/q3_layout.h"
#include <random>
#include <iostream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cstring>
#include <memory>
using namespace strata::kernels::cpu;
namespace strata::kernels::cpu {void set_q3_compiler(bool);}
// Synthetic complete Q2/Q3 expert service; link the experimental compiler dispatch wrapper.
// The linked wrapper selects candidate rows; other calls retain the baseline.
int bench(int nt,int workers,int repeats,bool flow=false,bool sweep=false,bool quant_sweep=false,int E=8,int banks=1,int canonical=0,bool mixed=false,int calls=1) {
    if((nt!=1&&nt!=2&&nt!=4)||workers<1||workers>31||repeats<1||repeats>20||E<1||E>8||banks<1||banks>16)
        throw std::runtime_error("invalid pool benchmark bounds");
    if(calls<1||calls>64)throw std::runtime_error("calls must be 1..64");
    if(mixed&&(nt!=2||E!=7))throw std::runtime_error("mixed routes require two tokens and seven experts");
    ExpertPool pool(workers,true);const bool owned=pool.numa_rows_available();
    if(quant_sweep&&!owned)throw std::runtime_error("fusion sweep requires NUMA-owned rows");
    // Single-NUMA workers use the complete-blob branch below; two-node hosts
    // retain their node-owned shards. Both compiler variants share weight banks.
    const auto previous=pin_current_thread(planned_host_core());
    constexpr int H=4096,F=2048;
    NativeFmt f;std::string error;
    if(!native_fmt(10,11,H,F,f,error))throw std::runtime_error(error);
    f.swiglu_limit=10;f.canon=canonical!=0;f.gate_skip=canonical==2?.5f:0.f;
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
    set_q3_compiler(false);
    bool candidate=false;
    std::vector<std::vector<uint8_t>> acts(nt,std::vector<uint8_t>(f.act_bytes));
    std::vector<float> input(H),out(E*nt*H),sum(nt*H),weights(nt*E,1.f/E);
    for(int t=0;t<nt;++t) {
        for(auto& v:input)v=float(int(rng()%2001)-1000)*.001f;
        native_quant_act(f,input.data(),acts[t].data());
    }
    std::vector<ExpertJobMulti> jobs(E);
    for(int e=0;e<E;++e) {
        auto& j=jobs[e];j.nt=mixed&&e!=0?1:nt;j.blob=owned?g.data()+size_t(e)*f.up_off:complete.data()+size_t(e)*f.bytes;
        if(owned)for(int node=0;node<2;++node)j.numa[node]={ng[0]->shard(e,node),nu[0]->shard(e,node),nd[0]->shard(e,node)};
        // Token 0 routes to experts 0,1,2,3; token 1 to 0,4,5,6.
        // Unused dense result slots stay zero and carry zero reduction weight.
        if(mixed)for(int t=0;t<nt;++t)weights[t*E+e]=0.f;
        for(int t=0;t<j.nt;++t){
            const int token=mixed&&e!=0?int(e>=4):t;
            j.nact[t]=acts[token].data();j.out[t]=out.data()+(token*E+e)*H;
            if(mixed)weights[token*E+e]=.25f;
        }
    }
    auto split=[&] {
        pool.run_split_multi_native(f,jobs.data(),E);
        pool.reduce_routed(out.data(),weights.data(),sum.data(),nt,E,H,f.canon);
    };
    split();const auto reference=out,reference_sum=sum;
    // Both compilers share the same NUMA weight banks. Each call,
    // including warmup, advances the bank; this is not a routing-trace replay.
    size_t next_bank[2]={0,0};
    auto run=[&] {
        const size_t bank=next_bank[candidate]++%banks;
        for(int e=0;e<E;++e) {
            if(owned)for(int node=0;node<2;++node)
                jobs[e].numa[node]={ng[bank]->shard(e,node),nu[bank]->shard(e,node),nd[bank]->shard(e,node)};
            else jobs[e].blob=complete.data()+(bank*E+e)*f.bytes;
        }
        if(flow)pool.run_layer_native(f,jobs.data(),E,out.data(),weights.data(),sum.data(),nt,E);
        else split();
    };
    run();next_bank[0]=0;
    if(out!=reference||sum!=reference_sum)throw std::runtime_error("pool mode differs from split reference");
    for(float v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite pool benchmark output");
    std::cout<<"scope=synthetic_cpu_expert_layer H="<<H<<" F="<<F<<" experts="<<E<<" tokens="<<nt
             <<" mixed="<<mixed<<" canonical="<<f.canon<<" gate_skip="<<f.gate_skip<<" workers="<<workers<<" owned="<<owned<<" mode="<<(quant_sweep?"flow-quant-sweep":flow?"flow":"split")<<" packed_bytes="<<E*f.bytes<<" banks="<<banks<<" rotating_bytes="<<size_t(banks)*E*f.bytes<<" input_quantization=excluded\n";
    for(int r=0;r<repeats;++r)for(int order=0;order<2;++order) {
        candidate=bool(order^(r%2));set_q3_compiler(candidate);
        run(); // warm selected compiler outside timing; synchronous pool call
        const double gu=pool.ms_multi_gu,q=pool.ms_multi_q,down=pool.ms_multi_down;
        // CPU-clock overhead is amortized only in batch mode. It includes all
        // process threads, including pool spinning, not just useful arithmetic.
        const auto cpu_start=calls>1?std::clock():std::clock_t{};
        const auto start=std::chrono::steady_clock::now();
        for(int call=0;call<calls;++call)run();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/calls;
        const double cpu_ms=calls>1?1000.*(std::clock()-cpu_start)/CLOCKS_PER_SEC/calls:0;
        if(std::memcmp(out.data(),reference.data(),out.size()*sizeof(float))||
           std::memcmp(sum.data(),reference_sum.data(),sum.size()*sizeof(float)))throw std::runtime_error("pool benchmark output changed");
        uint64_t hash=14695981039346656037ULL;
        for(const auto* v:{&out,&sum})for(size_t i=0;i<v->size()*sizeof(float);++i){hash^=reinterpret_cast<const uint8_t*>(v->data())[i];hash*=1099511628211ULL;}
        std::cout<<"round="<<r<<" candidate="<<candidate<<" calls="<<calls<<" ms="<<ms;
        if(calls>1)std::cout<<" cpu_ms="<<cpu_ms;

        if(quant_sweep)std::cout<<" fuse_h_quant="<<f.fuse_h_quant;
        if(!flow)std::cout<<" gu_ms="<<(pool.ms_multi_gu-gu)/calls<<" quant_ms="<<(pool.ms_multi_q-q)/calls<<" down_ms="<<(pool.ms_multi_down-down)/calls;
        std::cout<<" output_hash="<<hash<<"\n";
    }
    if(previous.valid)restore_thread_affinity(previous);
    return 0;
}
int main(int argc,char** argv){try {
    if(argc<4||argc>7)throw std::runtime_error("usage: paired_pool TOKENS BANKS split|flow [CANONICAL(0|1|2)] [dense|mixed [CALLS(1..64)]]");
    const int canonical=argc>=5?std::stoi(argv[4]):0;
    const std::string pattern=argc>=6?argv[5]:"dense";
    const bool mixed=pattern=="mixed";
    if(pattern!="dense"&&!mixed)throw std::runtime_error("invalid route pattern");
    const int calls=argc==7?std::stoi(argv[6]):1;
    if(canonical<0||canonical>2)throw std::runtime_error("invalid canonical mode");
    if(!std::getenv("STRATA_Q23_AVX2")||std::string(std::getenv("STRATA_Q23_AVX2"))!="1")throw std::runtime_error("requires STRATA_Q23_AVX2=1");
    const std::string mode=argv[3];if(mode!="split"&&mode!="flow")throw std::runtime_error("invalid mode");
    return bench(std::stoi(argv[1]),15,20,mode=="flow",false,false,mixed?7:2,std::stoi(argv[2]),canonical,mixed,calls);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
