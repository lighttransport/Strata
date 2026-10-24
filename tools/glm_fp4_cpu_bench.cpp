// Rotating native-FP4 code working set; excludes activation quantization/model loading.
#include "strata/kernels/cpu/glm_fp4.hpp"
#include "strata/kernels/cpu/exl3/pool.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/host/glm_fp4.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace c=strata::kernels::cpu;
namespace {
// Copies used to prepare auxiliary operands can have different alignment or
// cache placement. Give every variant the same activation and scale storage.
struct SharedInputStorage {
    c::GlmFp4Q8& base; c::GlmFp4Q8& operands;
    SharedInputStorage(c::GlmFp4Q8& a,c::GlmFp4Q8& b):base(a),operands(b) {
        if(&base!=&operands){base.values.swap(operands.values);base.scales.swap(operands.scales);}
    }
    SharedInputStorage(const SharedInputStorage&)=delete;
    SharedInputStorage& operator=(const SharedInputStorage&)=delete;
    ~SharedInputStorage() {
        if(&base!=&operands){base.values.swap(operands.values);base.scales.swap(operands.scales);}
    }
};
}
// --layer-sim E M [POOL [LAYERS]]: decode-layer streaming. Each simulated layer reads E distinct experts (gate and
// up 2048x4096, down 4096x2048, own codes and CSF32 sparse scales) out of a POOL much larger than the last-level
// cache, with M token rows each, in the engine's two CPU schedules: all workers splitting every expert's rows
// (CPU_FUSED) and teams of 2/4/8 taking whole experts (CPU_TEAM_SIZE). Activation quantization of the hidden rows
// is included; SwiGLU and the final scatter-add are not. Reports bytes streamed per second. Page placement follows
// the process policy (run under numactl to choose it).
int layer_sim(int workers,int experts,int width,int pool_size,int layers) {
    if(experts<1||width<1||width>8||pool_size<experts||layers<1)
        throw std::invalid_argument("--layer-sim E M [POOL [LAYERS]]: 1<=M<=8, E<=POOL, LAYERS>=1");
    strata::cpu::exl3::Pool pool(workers,true,true);
    auto scales_for=[](int N,int K) {
        const int G=K/16;const size_t sb=size_t(N)*G;
        std::vector<uint8_t> bases(N),codes(sb/2),scales(sb);
        for(int i=0;i<N;++i)bases[i]=30+i%20;
        for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*59+19);
        const size_t tiles=size_t(N/128)*(G/4),pages=(tiles+31)/32;
        std::vector<uint16_t> offsets(size_t(N/32)*(G/2)+pages);std::vector<uint32_t> page_bases(pages);
        if(strata_fp4_csf32_decode(N,G,bases.data(),bases.size(),codes.data(),codes.size(),offsets.data(),offsets.size(),
                                   page_bases.data(),pages,nullptr,0,scales.data(),scales.size()))throw std::runtime_error("CSF32 fixture");
        for(size_t i=0;i<scales.size();i+=1021)scales[i]=uint8_t(8+i%110);
        return c::glm_fp4_csf32_sparse_rows(scales.data(),bases.data(),codes.data(),N,G,false);
    };
    struct Expert {std::vector<uint8_t> codes[3];c::GlmFp4Csf32SparseRows scales[3];size_t bytes=0;};
    const auto gate_scales=scales_for(2048,4096),down_scales=scales_for(4096,2048);
    std::vector<Expert> pool_experts(static_cast<size_t>(pool_size));
    for(int e=0;e<pool_size;++e) {
        auto& x=pool_experts[size_t(e)];
        for(int p=0;p<3;++p) {
            x.codes[p].resize(size_t(2048)*4096/2);
            for(size_t i=0;i<x.codes[p].size();++i)x.codes[p][i]=uint8_t(i*53+7+e*13+p);
            x.scales[p]=p<2?gate_scales:down_scales;
            x.bytes+=x.codes[p].size()+x.scales[p].bytes();
        }
    }
    alignas(64)float table[c::glm_fp4_weight_table_floats];c::glm_fp4_weight_table(.00061f,true,table);
    std::vector<float> x(size_t(4096)*width);
    for(size_t i=0;i<x.size();++i)x[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.37f)));
    const auto input=c::glm_fp4_quantize_q8(x.data(),4096,width);
    std::vector<float> gate(size_t(experts)*width*2048),up(gate.size()),hidden(gate.size()),result(size_t(experts)*width*4096);
    std::vector<c::GlmFp4Q8> qhidden(static_cast<size_t>(experts));
    std::vector<int> chosen(static_cast<size_t>(experts));
    auto choose=[&](int layer) {  // E distinct experts per layer, a different set each layer
        uint32_t rng=uint32_t(layer)*2654435761u+12345u;
        std::vector<uint8_t> used(static_cast<size_t>(pool_size),0);
        for(int j=0;j<experts;++j){int e;do{rng=rng*1664525u+1013904223u;e=int((rng>>8)%uint32_t(pool_size));}while(used[size_t(e)]);used[size_t(e)]=1;chosen[size_t(j)]=e;}
    };
    auto gate_up_range=[&](size_t rank,size_t count,size_t j) {
        const int first=int(2048*rank/count),last=int(2048*(rank+1)/count);
        const auto& e=pool_experts[size_t(chosen[j])];
        c::glm_fp4_rows_csf32_q8(e.codes[0].data(),e.scales[0],table,input,gate.data()+j*width*2048,4096,2048,width,first,last,1,width==1);
        c::glm_fp4_rows_csf32_q8(e.codes[1].data(),e.scales[1],table,input,up.data()+j*width*2048,4096,2048,width,first,last,1,width==1);
        for(int t=0;t<width;++t)for(int r=first;r<last;++r) {
            const size_t i=j*width*2048+size_t(t)*2048+r;const float g=gate[i];hidden[i]=(g/(1.f+std::exp(-g)))*up[i];
        }
    };
    auto quantize=[&](size_t j){qhidden[j]=c::glm_fp4_quantize_q8(hidden.data()+j*width*2048,2048,width);};
    auto down_range=[&](size_t rank,size_t count,size_t j) {
        const int first=int(4096*rank/count),last=int(4096*(rank+1)/count);
        const auto& e=pool_experts[size_t(chosen[j])];
        c::glm_fp4_rows_csf32_q8(e.codes[2].data(),e.scales[2],table,qhidden[j],result.data()+j*width*4096,2048,4096,width,first,last,1,width==1);
    };
    auto spin_barrier=[](std::atomic<size_t>& arrived,size_t target) {
        arrived.fetch_add(1,std::memory_order_acq_rel);
        while(arrived.load(std::memory_order_acquire)<target) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    };
    auto run_layer=[&](int team_size) {
        const size_t W=pool.size();
        if(!team_size) {
            std::atomic<size_t> arrived{0};
            pool.run([&](size_t rank) {
                for(size_t j=0;j<size_t(experts);++j)gate_up_range(rank,W,j);
                spin_barrier(arrived,W);
                for(size_t j=rank;j<size_t(experts);j+=W)quantize(j);
                spin_barrier(arrived,2*W);
                for(size_t j=0;j<size_t(experts);++j)down_range(rank,W,j);
            });
            return;
        }
        const size_t size=size_t(team_size),teams=(W+size-1)/size;
        struct alignas(64) Counter {std::atomic<size_t> value{0};};
        auto arrived=std::make_unique<Counter[]>(teams);
        pool.run([&](size_t rank) {
            const size_t team=rank%teams,member=rank/teams,count=(W-1-team)/teams+1;
            size_t phase=0;
            for(size_t j=team;j<size_t(experts);j+=teams) {
                gate_up_range(member,count,j);spin_barrier(arrived[team].value,count*++phase);
                if(member==0)quantize(j);
                spin_barrier(arrived[team].value,count*++phase);
                down_range(member,count,j);
            }
        });
    };
    std::cout<<"{\"kind\":\"layer_sim_configuration\",\"workers\":"<<workers<<",\"experts\":"<<experts<<",\"width\":"<<width
        <<",\"pool\":"<<pool_size<<",\"layers\":"<<layers<<",\"expert_bytes\":"<<pool_experts[0].bytes
        <<",\"working_set_bytes\":"<<pool_experts[0].bytes*size_t(pool_size)<<",\"vnni\":"<<c::glm_fp4_q8_vnni_available()<<"}\n";
    std::vector<float> reference;
    for(int team:{0,2,4,8}) {
        if(team>int(pool.size()))continue;
        choose(0);run_layer(team);
        if(reference.empty())reference=result;
        else if(std::memcmp(reference.data(),result.data(),result.size()*sizeof(float)))
            throw std::runtime_error("layer-sim: team schedule changes outputs");
        for(int sample=0;sample<3;++sample) {
            for(int l=0;l<std::min(layers,8);++l){choose(l+1000*sample);run_layer(team);}   // warm
            double bytes=0,us=0;
            for(int l=0;l<layers;++l) {
                choose(l+1+100000*(sample+1));
                for(int j=0;j<experts;++j)bytes+=double(pool_experts[size_t(chosen[size_t(j)])].bytes);
                const auto start=std::chrono::steady_clock::now();
                run_layer(team);
                us+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            }
            std::cout<<"{\"kind\":\"layer_sim\",\"schedule\":\""<<(team?"team"+std::to_string(team):std::string("fused"))
                <<"\",\"workers\":"<<workers<<",\"experts\":"<<experts<<",\"width\":"<<width<<",\"sample\":"<<sample
                <<",\"us_per_layer\":"<<us/layers<<",\"gb_per_s\":"<<bytes/us/1e3<<"}\n";
        }
    }
    return 0;
}
int main(int argc,char** argv) {try {
    if(argc>2&&std::string(argv[2])=="--layer-sim") {
        if(argc<5||argc>7)throw std::invalid_argument("usage: cpu-bench WORKERS --layer-sim E M [POOL [LAYERS]]");
        const int workers=std::stoi(argv[1]);
        if(workers<1||workers>72)throw std::invalid_argument("workers must be 1..72");
        return layer_sim(workers,std::stoi(argv[3]),std::stoi(argv[4]),argc>5?std::stoi(argv[5]):300,argc>6?std::stoi(argv[6]):84);
    }
    const int workers=argc>1?std::stoi(argv[1]):1;
    if(workers<1||workers>72)throw std::invalid_argument("workers must be 1..72");
    const int batch=argc>2?std::stoi(argv[2]):1;
    if(batch<1||batch>16||(batch&(batch-1)))throw std::invalid_argument("products per wake must be 1, 2, 4, 8 or 16");
    const bool all_widths=argc>3&&std::string(argv[3])=="--all-widths";
    if(argc>4||(argc>3&&!all_widths))throw std::invalid_argument("usage: cpu-bench [workers [products-per-wake [--all-widths]]]");
    strata::cpu::exl3::Pool pool(workers,true,true);
    if(batch>1)for(int sample=0;sample<3;++sample) {
        for(int i=0;i<16;++i)pool.run([](size_t){});
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<64;++i)pool.run([](size_t){});
        const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/64;
        std::cout<<"{\"kind\":\"pool_overhead\",\"workers\":"<<workers<<",\"sample\":"<<sample<<",\"us_per_wake\":"<<us<<"}\n";
    }
    const std::vector<int> widths=all_widths?std::vector<int>{1,2,3,4,5,6,7,8}:std::vector<int>{1,2,4,8};
    for(int N:{2048,4096}) {
        const int K=N==2048?4096:2048,G=K/16,copies=16,repeats=64;
        const size_t wb=size_t(N)*K/2,sb=size_t(N)*G;
        std::vector<uint8_t>w(wb*copies),bases(N),codes(sb/2),scales(sb);
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i*53+7);
        for(int i=0;i<N;++i)bases[i]=30+i%20;
        for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*59+19);
        const size_t tiles=size_t(N/128)*(G/4),pages=(tiles+31)/32;
        std::vector<uint16_t> offsets(size_t(N/32)*(G/2)+pages);std::vector<uint32_t> page_bases(pages);
        if(strata_fp4_csf32_decode(N,G,bases.data(),bases.size(),codes.data(),codes.size(),offsets.data(),offsets.size(),page_bases.data(),pages,nullptr,0,scales.data(),scales.size()))throw std::runtime_error("CSF32 fixture");
        for(size_t i=0;i<scales.size();i+=1021)scales[i]=uint8_t(8+i%110);
        auto sparse=c::glm_fp4_csf32_sparse_rows(scales.data(),bases.data(),codes.data(),N,G,true),cached=sparse;
        c::glm_fp4_cache_scale_quartets(cached);
        std::vector<float>x(K*8),reference(N*8),out(N*8);
        for(size_t i=0;i<x.size();++i)x[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.37f)));
        auto q=c::glm_fp4_quantize_q8(x.data(),K,8);auto prepared=q;c::glm_fp4_prepare_q8(prepared);
        auto biased=q;c::glm_fp4_prepare_q8_bias(biased);
        alignas(64)float table[c::glm_fp4_weight_table_floats];c::glm_fp4_weight_table(.00061f,true,table);
        struct Variant {const char* name;int tile;bool prepare,cache,bias=false,local=false,policy=false;};
        const Variant variants[]={{"baseline",1,false,false},{"prepared",1,true,false},{"tile2",2,false,false},
            {"tile4",4,false,false},{"tile2_prepared",2,true,false},{"quartets",1,false,true},
            {"biased",1,false,false,true},{"biased_tile2",2,false,false,true},
            {"biased_quartets",1,false,true,true},{"biased_quartets_tile2",2,false,true,true},
            {"local",1,false,false,false,true},{"local_tile2",2,false,false,false,true},
            {"biased_local",1,false,false,true,true},{"biased_local_tile2",2,false,false,true,true},
            {"policy",1,false,false,true,false,true}};
        constexpr int count=sizeof(variants)/sizeof(variants[0]);
        auto run=[&](const Variant& v,int width,int copy) {
            auto& operands=v.bias?biased:(v.prepare?prepared:q);
            SharedInputStorage shared(q,operands);
            pool.run([&](size_t rank) {
                for(int step=0;step<batch;++step)c::glm_fp4_rows_csf32_q8(w.data()+wb*((copy+step)%copies),v.cache?cached:sparse,table,operands,
                    out.data(),K,N,width,N*rank/pool.size(),N*(rank+1)/pool.size(),v.tile,v.local,v.policy);
            });
        };
        std::cout<<"{\"kind\":\"configuration\",\"rows\":"<<N<<",\"columns\":"<<K
            <<",\"workers\":"<<workers<<",\"products_per_wake\":"<<batch<<",\"shared_activation_storage\":true,\"activation_address_mod64\":"<<(reinterpret_cast<uintptr_t>(q.values.data())%64)
            <<",\"code_working_set_bytes\":"<<w.size()<<",\"vnni\":"<<c::glm_fp4_q8_vnni_available()<<"}\n";
        for(int width:widths) {
            run(variants[0],width,0);reference=out;
            for(const auto& v:variants) {
                run(v,width,0);
                if(std::memcmp(reference.data(),out.data(),size_t(width)*N*4))throw std::runtime_error("variant changes Q8 outputs");
            }
            for(int sample=0;sample<3;++sample)for(int order=0;order<count;++order) {
                const auto& v=variants[(order+sample)%count];
                for(int i=0;i<copies;i+=batch)run(v,width,i);
                const auto start=std::chrono::steady_clock::now();
                for(int i=0;i<repeats;i+=batch)run(v,width,i%copies);
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
                std::cout<<"{\"kind\":\"timing\",\"rows\":"<<N<<",\"columns\":"<<K<<",\"workers\":"<<workers
                    <<",\"width\":"<<width<<",\"variant\":\""<<v.name<<"\",\"sample\":"<<sample<<",\"us\":"<<us<<"}\n";
            }
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
