#include <algorithm>
#include "glm_fp4_service_lane.hpp"
#ifdef STRATA_FP4_NATIVE_CPU
#ifndef STRATA_FP4_CPU_ONLY
#error Native CPU tree requires STRATA_FP4_CPU_ONLY
#endif
#include "experiment/glm_fp4_local/native_cpu_expert.hpp"
#include "ggml-cpu.h"
static std::unique_ptr<strata::kernels::cpu::ExpertPool> native_pool;
static bool preserve_small=false;
static bool native_numa_compact=false;
static bool native_k32=false;
static bool native_workspace=false;
static bool worker_nt1_tasks=false;
static unsigned worker_nt1_calls=0,worker_other_calls=0;
static void select_worker_tasks(int tokens) {
    if(!worker_nt1_tasks)return;
    native_pool->set_native_tasks_per_thread(tokens==1?6:3);
    if(tokens==1)++worker_nt1_calls;else ++worker_other_calls;
}
#else
static constexpr bool native_workspace=false;
#endif
// Bounded three-node tree MoE test. No production decoder defaults change.
#ifdef STRATA_FP4_CPU_ONLY
#include "strata/kernels/cpu/fp4_local.hpp"
#include <memory>
#else
#include "strata/kernels/fp4_local.hpp"
#endif
#include "ucomm.h"
#include "glm_fp4_registered_staging.hpp"
static strata::experiment::RegisteredStaging registered_staging;
#include "strata/host/glm_fp4_protocol.h"
#include <fstream>
#include <array>
#include <cstdio>
#include <random>
#include <thread>
#include <chrono>
using Matrix=strata::artifact::GlmFp4Matrix;
static void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
static void ok(ucomm_status s){if(s!=UCOMM_OK)throw std::runtime_error(ucomm_strerror(s));}
static void recv_exact(ucomm_t* c,uint32_t tag,void* p,size_t n){size_t got=0;ok(ucomm_recv(c,1-ucomm_rank(c),tag,p,n,&got));require(got==n,"network message length mismatch");}
static void send(ucomm_t* c,uint32_t tag,const void* p,size_t n){ok(ucomm_send(c,1-ucomm_rank(c),tag,p,n));}
#include <future>
static float round_bf16(float x){uint32_t b;std::memcpy(&b,&x,4);b+=0x7fff+((b>>16)&1);b&=0xffff0000;std::memcpy(&x,&b,4);return x;}
#ifdef STRATA_FP4_CPU_ONLY
static int cpu_threads=4;
static bool cpu_paired=false;
static bool cpu_fused=false;
static bool cpu_reuse=false;
#endif
struct Expert {
#ifdef STRATA_FP4_CPU_ONLY
#ifdef STRATA_FP4_NATIVE_CPU
    strata::experiment::NativeFp4CpuExpert cpu;
    std::vector<uint8_t> activation_scratch;
    explicit Expert(const std::array<Matrix,3>& w,bool use_tiled=false):cpu(w,preserve_small,native_k32) {
        require(!use_tiled,"tiled is a GPU-only selector");
        if(native_numa_compact) {
            require(cpu.prepare_numa(*native_pool),"NUMA compact mode requires a two-node pool");
            cpu.release_original_blob();
            require(cpu.retained_blob_bytes()==0,"NUMA compact mode retained duplicate weights");
        }
    }
    std::vector<float> run(const std::vector<float>& input,int nt) {
        select_worker_tasks(nt);
        return cpu.run(*native_pool,input,nt,native_numa_compact);
    }
#else
    std::unique_ptr<strata::kernels::cpu::Fp4LocalCpuExpert> cpu;
    explicit Expert(const std::array<Matrix,3>& w,bool use_tiled=false) {
        require(!use_tiled,"tiled is a GPU-only selector");
        cpu=std::make_unique<strata::kernels::cpu::Fp4LocalCpuExpert>(w,cpu_threads);
    }
    std::vector<float> run(const std::vector<float>& input,int nt) {return (cpu_fused||cpu_reuse)?cpu->run_fused(input,nt,true,cpu_reuse):cpu->run(input,nt,cpu_paired,true);}
#endif
#else
    bool tiled;
    std::array<std::unique_ptr<strata::kernels::Fp4LocalGpu>,3> gpu;
    explicit Expert(const std::array<Matrix,3>& w,bool use_tiled=false):tiled(use_tiled){for(int i=0;i<3;++i)gpu[i]=std::make_unique<strata::kernels::Fp4LocalGpu>(w[i]);}
    std::vector<float> project(int p,const std::vector<float>& input,int nt) {return gpu[p]->run(input,nt,true,tiled);}
    std::vector<float> run(const std::vector<float>& input,int nt) {
        auto gate=project(0,input,nt),up=project(1,input,nt);
        for(size_t i=0;i<gate.size();++i) {
            float g=std::min(round_bf16(gate[i]),10.f),u=std::clamp(round_bf16(up[i]),-10.f,10.f);
            gate[i]=round_bf16(round_bf16(g/(1+std::exp(-g)))*u);
        }
        return project(2,gate,nt);
    }
#endif
    void run_into(const std::vector<float>& input,int nt,std::vector<float>& output) {
#ifdef STRATA_FP4_NATIVE_CPU
        if(native_workspace) {
            select_worker_tasks(nt);
            cpu.run_into(*native_pool,input,nt,output,activation_scratch,native_numa_compact);
            return;
        }
#endif
        output=run(input,nt);
    }
};
static double compare(const std::vector<float>& got,const std::vector<float>& ref) {
    require(got.size()==ref.size(),"comparison shape mismatch");double err=0,norm=0;
    for(size_t i=0;i<got.size();++i) {
        require(std::isfinite(got[i])&&std::isfinite(ref[i]),"nonfinite MoE result");
        double e=double(got[i])-ref[i];err+=e*e;norm+=double(ref[i])*ref[i];
    }
    double rel=std::sqrt(err/std::max(norm,1e-30));require(rel<0.005,"routed split/unsplit error exceeds 0.5 percent L2");return rel;
}
using Header=std::array<uint32_t,7>; // magic, version, op, layer, middle M, leaf M, sequence
static constexpr uint32_t magic=STRATA_GLM_FP4_WIRE_MAGIC,version=STRATA_GLM_FP4_WIRE_VERSION;
static ucomm_t* connect(const char* root,int rank,int port,bool udp) {
    require(port>=1&&port<=2,"IB port out of bounds");
    ucomm_config cfg;ucomm_config_default(&cfg);cfg.world=2;cfg.rank=rank;cfg.root_addr=root;
    cfg.ib_port=port;cfg.timeout_ms=60000;cfg.backend=udp?UCOMM_BACKEND_UDP:UCOMM_BACKEND_IB;
    ucomm_t* c=nullptr;ok(ucomm_init(&cfg,&c));
    std::printf("TRANSPORT %s edge_rank=%d ib_port=%d\n",ucomm_backend_name(c),rank,port);return c;
}
static void put_matrix(ucomm_t* c,const Matrix& w) {
    uint32_t shape[2]={uint32_t(w.rows),uint32_t(w.cols)};send(c,1,shape,sizeof(shape));send(c,2,&w.global,4);
    send(c,3,w.codes.data(),w.codes.size());send(c,4,w.scales.data(),w.scales.size());
}
static Matrix get_matrix(ucomm_t* c,int p) {
    uint32_t shape[2];recv_exact(c,1,shape,sizeof(shape));
    require(shape[0]==(p==2?4096u:2048u)&&shape[1]==(p==2?2048u:4096u),"matrix shape mismatch");
    Matrix w;w.rows=shape[0];w.cols=shape[1];recv_exact(c,2,&w.global,4);
    w.codes.resize(w.rows*w.cols/2);w.scales.resize(w.rows*w.cols/16);
    recv_exact(c,3,w.codes.data(),w.codes.size());recv_exact(c,4,w.scales.data(),w.scales.size());return w;
}
static std::vector<float> receive(ucomm_t* c,int m,uint32_t tag) {
    require(m>=0&&m<=16,"activation size out of bounds");std::vector<float> v(size_t(m)*4096);
    if(m) {
        if(auto* buffers=registered_staging.find(c)) {
            auto* data=buffers->rx(v.size()*sizeof(float));
            recv_exact(c,tag,data,v.size()*sizeof(float));
            std::memcpy(v.data(),data,v.size()*sizeof(float));
        }else recv_exact(c,tag,v.data(),v.size()*sizeof(float));
    }
    return v;
}
static void put(ucomm_t* c,const std::vector<float>& v,uint32_t tag) {
    if(v.empty())return;
    if(auto* buffers=registered_staging.find(c)) {
        auto* data=buffers->tx(v.size()*sizeof(float));
        std::memcpy(data,v.data(),v.size()*sizeof(float));
        send(c,tag,data,v.size()*sizeof(float));
    }else send(c,tag,v.data(),v.size()*sizeof(float));
}
static Header get_header(ucomm_t* c,uint32_t sequence,int role,bool direct) {
    Header h{};recv_exact(c,20,h.data(),sizeof(h));
    require(strata_glm_fp4_command_valid(h.data(),sequence,role,direct),"invalid command: protocol, sequence, bounds or destination");
    return h;
}
static void acknowledge(ucomm_t* c,uint32_t seq) {send(c,22,&seq,4);}
static void check_ack(ucomm_t* c,uint32_t seq) {uint32_t got;recv_exact(c,22,&got,4);require(got==seq,"stale response");}
int main(int argc,char** argv) {
    try {
#ifdef STRATA_FP4_NATIVE_CPU
        require(argc>=7&&argc<=18,"usage: tree-node ROLE UP_ROOT DOWN_ROOT MODEL UP_PORT DOWN_PORT [direct] [extended] [dump=FILE] [udp] [threads=N] [preserve-small] [persistent-dispatch (direct primary)] [numa-compact (primary)] [native-k32] [native-workspace] [registered-staging] [worker-nt1-tasks (four-participant workers)]");
#else
        require(argc>=7&&argc<=15,"usage: tree-node ROLE UP_ROOT DOWN_ROOT MODEL UP_PORT DOWN_PORT [tiled] [direct] [extended] [dump=FILE] [udp] [threads=N (CPU only)] [paired|fused|reuse (CPU only)] [registered-staging]");
#endif
        int role=std::stoi(argv[1]),up_port=std::stoi(argv[5]),down_port=std::stoi(argv[6]);
        require(role>=0&&role<=2,"invalid role");bool tiled=false,direct=false,extended=false,udp=false,persistent_dispatch=false,staging=false;
#ifdef STRATA_FP4_CPU_ONLY
        bool threads_set=false;
#endif
        std::string dump;
        for(int i=7;i<argc;++i) {
            std::string flag=argv[i];
            if(flag=="persistent-dispatch"&&!persistent_dispatch)persistent_dispatch=true;
            else if(flag=="registered-staging"&&!staging)staging=true;
            else if(flag=="tiled"&&!tiled)tiled=true;
            else if(flag=="direct"&&!direct)direct=true;
            else if(flag=="extended"&&!extended)extended=true;
            else if(flag=="udp"&&!udp)udp=true;
#ifdef STRATA_FP4_CPU_ONLY
#ifdef STRATA_FP4_NATIVE_CPU
            else if(flag=="worker-nt1-tasks"&&!worker_nt1_tasks)worker_nt1_tasks=true;
            else if(flag=="native-workspace"&&!native_workspace)native_workspace=true;
            else if(flag=="native-k32"&&!native_k32)native_k32=true;
            else if(flag=="numa-compact"&&!native_numa_compact)native_numa_compact=true;
            else if(flag=="preserve-small"&&!preserve_small)preserve_small=true;
#endif
            else if(flag=="paired"&&!cpu_paired)cpu_paired=true;
            else if(flag=="fused"&&!cpu_fused)cpu_fused=true;
            else if(flag=="reuse"&&!cpu_reuse)cpu_reuse=true;
            else if(flag.starts_with("threads=")&&!threads_set) {
                const auto value=flag.substr(8);size_t used=0;
                cpu_threads=std::stoi(value,&used);
                require(used==value.size()&&cpu_threads>=1&&cpu_threads<=32,"invalid CPU thread count");threads_set=true;
            }
#endif
            else if(flag.starts_with("dump=")&&dump.empty()&&role==0&&flag.size()>5)dump=flag.substr(5);
            else throw std::runtime_error("invalid or repeated mode selector");
        }
        require(!persistent_dispatch||(role==0&&direct),"persistent-dispatch requires direct-mode primary");
        uint32_t endian=1;require(*reinterpret_cast<uint8_t*>(&endian)==1,"little endian required");
        std::setvbuf(stdout,nullptr,_IOLBF,0);
#ifdef STRATA_FP4_CPU_ONLY
        require(!tiled,"tiled is a GPU-only selector");
        require(int(cpu_paired)+int(cpu_fused)+int(cpu_reuse)<=1,"paired, fused and reuse selectors are mutually exclusive");
#ifdef STRATA_FP4_NATIVE_CPU
        require(!cpu_paired&&!cpu_fused&&!cpu_reuse,"W4A16 mode selector in native CPU build");
        require(cpu_threads>=2,"native CPU tree requires threads=2..32");
        require(!native_numa_compact||role==0,"numa-compact is a primary-only selector");
        require(!worker_nt1_tasks||(role!=0&&cpu_threads==4),"worker-nt1-tasks requires a four-participant native worker");
        ggml_cpu_init();
        // One pool serves local and reference experts; network tasks never use it.
        native_pool=std::make_unique<strata::kernels::cpu::ExpertPool>(cpu_threads-1,true);
        require(!native_numa_compact||native_pool->numa_rows_available(),"numa-compact requires pinned workers on both NUMA nodes");
        if(worker_nt1_tasks)std::puts("CPU_TASK_POLICY worker_nt1=6 other=3");
        if(native_workspace)std::puts("CPU_WORKSPACE reuse=1");
        if(native_k32)std::puts("CPU_LAYOUT native_k32=1");
        if(native_numa_compact)std::puts("CPU_NUMA compact=1");
        std::printf("BACKEND CPU NATIVE_FP4_Q8 threads=%d preserve_small=%d no_GPU=1\n",cpu_threads,int(preserve_small));
#else
        std::printf("BACKEND CPU W4A16 threads=%d projection_verify=FP64 no_GPU=1\n",cpu_threads);
        if(cpu_paired)std::puts("CPU_GATE_UP paired=1");
        if(cpu_fused)std::puts("CPU_EXPERT fused=1");
        if(cpu_reuse)std::puts("CPU_EXPERT fused=1 reuse_weights=1");
#endif
#else
        strata::kernels::fp4_local_device_info();
#endif
        ucomm_t* up=role<2?connect(argv[2],role,up_port,udp):nullptr;
        ucomm_t* down=direct?(role!=1?connect(argv[3],role==0?0:1,down_port,udp):nullptr)
                            :(role>0?connect(argv[3],role-1,down_port,udp):nullptr);
        std::printf("TOPOLOGY %s role=%d\n",direct?"direct":"relay",role);
        // Verify matching protocol before any variable payload.
#ifdef STRATA_FP4_NATIVE_CPU
        uint32_t features=extended?STRATA_GLM_FP4_HELLO_EXTENDED:0;
        features|=STRATA_GLM_FP4_HELLO_NATIVE_Q8;
        if(preserve_small)features|=STRATA_GLM_FP4_HELLO_PRESERVE_SMALL;
        Header hello{magic,version,2,4096,2048,direct?4u:3u,features},peer{};
#else
        Header hello{magic,version,2,4096,2048,direct?4u:3u,extended?1u:0u},peer{};
#endif
        auto handshake=[&](ucomm_t* c){
            if(ucomm_rank(c)==0){send(c,21,hello.data(),sizeof(hello));recv_exact(c,21,peer.data(),sizeof(peer));}
            else{recv_exact(c,21,peer.data(),sizeof(peer));send(c,21,hello.data(),sizeof(hello));}
            require(peer==hello,"incompatible tree worker");
        };
        if(up)handshake(up);
        if(down)handshake(down);
        std::unique_ptr<strata::artifact::Safetensors> ts;if(role==0)ts=std::make_unique<strata::artifact::Safetensors>(argv[4]);
        if(staging) {
            constexpr size_t capacity=16*4096*sizeof(float);
            if(up)registered_staging.add(up,capacity);
            if(down)registered_staging.add(down,capacity);
            std::printf("REGISTERED_STAGING contexts=%zu bytes=%zu\n",registered_staging.size(),registered_staging.size()*2*capacity);
        }
        std::array<std::unique_ptr<Expert>,2> cache;
        std::array<std::unique_ptr<Expert>,6> reference;
        size_t bytes=0;const char* proj[]={"gate_proj","up_proj","down_proj"};
        for(int layer=0;layer<2;++layer)for(int owner=0;owner<3;++owner) {
            std::array<Matrix,3> w;
            for(int p=0;p<3;++p) {
                if(role==0) {
                    w[p]=strata::artifact::glm_load_fp4(*ts,"model.language_model.layers."+std::to_string(layer+3)+".mlp.experts."+std::to_string(owner)+"."+proj[p]);
                    require(w[p].rows==(p==2?4096u:2048u)&&w[p].cols==(p==2?2048u:4096u),"unexpected model dimensions");
                    if(owner)put_matrix(direct&&owner==2?down:up,w[p]);
                }else if(role==1&&owner&&(!direct||owner==1)) {
                    w[p]=get_matrix(up,p);if(owner==2)put_matrix(down,w[p]);
                }else if(role==2&&owner==2)w[p]=get_matrix(down,p);
            }
            if(role==owner){for(auto& m:w)bytes+=m.bytes();cache[layer]=std::make_unique<Expert>(w,tiled);}
            if(role==0)reference[layer*3+owner]=std::make_unique<Expert>(w);
        }
        if(down)ok(ucomm_barrier(down));
        if(up)ok(ucomm_barrier(up));
#ifdef STRATA_FP4_CPU_ONLY
        std::printf("READY role=%d experts=2 packed_host_bytes=%zu\n",role,bytes);
#else
        std::printf("READY role=%d experts=2 packed_device_bytes=%zu\n",role,bytes);
#endif
        std::unique_ptr<ServiceLane> first_lane,second_lane;
        if(persistent_dispatch){first_lane=std::make_unique<ServiceLane>();second_lane=std::make_unique<ServiceLane>();std::puts("DISPATCH persistent=1");}
        uint32_t seq=0;
        if(role!=0) {
            auto* parent=role==1?up:down;
            std::vector<float> retained_middle,retained_leaf;
            for(;;) {
                Header h=get_header(parent,++seq,role,direct);
                if(h[2]==0) {
                    if(role==1&&!direct){send(down,20,h.data(),sizeof(h));check_ack(down,seq);}
                    acknowledge(parent,seq);break;
                }
                auto middle=receive(parent,h[4],10),leaf=receive(parent,h[5],11);
                std::vector<float> middle_out,leaf_out;
                if(native_workspace) {
                    middle_out.swap(retained_middle);leaf_out.swap(retained_leaf);
                    // Idle routes must send empty outputs, never a previous request's data.
                    if(!h[4])middle_out.clear();
                    if(!h[5])leaf_out.clear();
                }
                if(role==2) {
                    require(h[4]==0,"leaf received middle work");
                    if(h[5])cache[h[3]]->run_into(leaf,h[5],leaf_out);
                }else if(direct) {
                    require(h[5]==0,"direct middle received leaf work");
                    if(h[4])cache[h[3]]->run_into(middle,h[4],middle_out);
                }else {
                    // The downstream context has exactly one owner during this task.
                    // Forward even idle commands to retain contiguous sequence checks.
                    auto remote=std::async(std::launch::async,[&,h]{
                        Header sub=h;sub[4]=0;send(down,20,sub.data(),sizeof(sub));put(down,leaf,11);
                        check_ack(down,h[6]);return receive(down,h[5],31);
                    });
                    if(h[4])cache[h[3]]->run_into(middle,h[4],middle_out);
                    leaf_out=remote.get();
                }
                acknowledge(parent,seq);put(parent,middle_out,30);put(parent,leaf_out,31);
                if(native_workspace) {middle_out.swap(retained_middle);leaf_out.swap(retained_leaf);}
            }
            std::printf("WORKER PASS role=%d commands=%u weights_reloaded=0\n",role,seq-1);
        }else {
            std::vector<float> retained_local,retained_reference;
            const int batches[]={1,7,16,3,2,15,16,7};std::array<std::vector<float>,8> prior;int cases=0;
            int scenarios=extended?8:4,repeats=extended?3:2;
            std::ofstream raw;
            if(!dump.empty()){raw.open(dump,std::ios::binary);require(bool(raw),"cannot open output dump");}
            for(int repeat=0;repeat<repeats;++repeat)for(int scenario=0;scenario<scenarios;++scenario) {
                int nt=batches[scenario];std::vector<float> input(size_t(nt)*4096);std::mt19937 rng(2026+scenario);
                for(float& x:input)x=round_bf16((int(rng()%513)-256)/256.f);
                if(scenario==4)std::fill(input.begin(),input.end(),0.f);
                if(scenario==5)for(size_t i=0;i<input.size();++i)input[i]=(i&1)?-3.f:3.f;
                for(int layer=0;layer<2;++layer) {
                    std::array<std::vector<float>,3> group,gains,out;
                    std::array<std::vector<int>,3> tokens;
                    for(int e=0;e<3;++e)for(int t=nt-1;t>=0;--t) {
                        int route=scenario==3?2:scenario==4?1:t%3;
                        if(scenario==7)continue; // all experts idle; residual-only layer
                        if(scenario==6){if(e==t%3)continue;} // varying top-2 selection
                        else if(scenario!=2&&scenario!=5&&route!=e)continue;
                        tokens[e].push_back(t);
                        gains[e].push_back(scenario==6?.5f:(scenario==2||scenario==5)?float(e+1)/6:1.f);
                        group[e].insert(group[e].end(),input.begin()+size_t(t)*4096,input.begin()+size_t(t+1)*4096);
                    }
                    Header h{magic,version,1,uint32_t(layer),uint32_t(tokens[1].size()),uint32_t(tokens[2].size()),++seq};
                    std::future<std::vector<float>> first,second;
                    if(direct) {
                        // One task owns each IB context; neither worker forwards data.
                        auto request=[&](ucomm_t* c,int owner) {
                            Header sub=h;sub[owner==1?5:4]=0;
                            send(c,20,sub.data(),sizeof(sub));put(c,group[owner],owner==1?10:11);
                            check_ack(c,h[6]);return receive(c,tokens[owner].size(),owner==1?30:31);
                        };
                        // Capture request by value so it remains alive until completion.
                        if(persistent_dispatch) {
                            first=first_lane->submit([=]{return request(up,1);});
                            second=second_lane->submit([=]{return request(down,2);});
                        }else {
                            first=std::async(std::launch::async,request,up,1);
                            second=std::async(std::launch::async,request,down,2);
                        }
                    }else {
                        send(up,20,h.data(),sizeof(h));put(up,group[1],10);put(up,group[2],11);
                    }
                    if(!tokens[0].empty()) {
                        if(native_workspace)out[0].swap(retained_local);
                        cache[layer]->run_into(group[0],tokens[0].size(),out[0]);
                    }
                    if(direct){out[1]=first.get();out[2]=second.get();}
                    else {check_ack(up,seq);out[1]=receive(up,h[4],30);out[2]=receive(up,h[5],31);}
                    std::vector<float> mixed(input.size()),expected(input.size());
                    for(int e=0;e<3;++e)if(!tokens[e].empty()) {
                        std::vector<float> ref;
                        if(native_workspace)ref.swap(retained_reference);
                        reference[layer*3+e]->run_into(group[e],tokens[e].size(),ref);
                        for(size_t t=0;t<tokens[e].size();++t)for(size_t j=0;j<4096;++j) {
                            size_t dst=size_t(tokens[e][t])*4096+j,src=t*4096+j;
                            mixed[dst]+=gains[e][t]*out[e][src];expected[dst]+=gains[e][t]*ref[src];
                        }
                        if(native_workspace)ref.swap(retained_reference);
                    }
                    if(native_workspace&&!tokens[0].empty())out[0].swap(retained_local);
                    double rel=compare(mixed,expected);++cases;
                    std::printf("TREE PASS repeat=%d scenario=%d layer=%d m=%d rel_l2=%.9g\n",repeat,scenario,layer+3,nt,rel);
                    for(size_t i=0;i<input.size();++i)input[i]=round_bf16(std::clamp(input[i]+mixed[i]*.125f,-4.f,4.f));
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
                uint64_t fingerprint=14695981039346656037ull;
                for(float x:input){uint32_t bits;std::memcpy(&bits,&x,4);for(int b=0;b<4;++b){fingerprint^=(bits>>(8*b))&255;fingerprint*=1099511628211ull;}}
                std::printf("OUTPUT repeat=%d scenario=%d fnv64=%016llx\n",repeat,scenario,(unsigned long long)fingerprint);
                if(repeat)require(input.size()==prior[scenario].size()&&std::memcmp(input.data(),prior[scenario].data(),input.size()*sizeof(float))==0,"repeat changed output bits");else prior[scenario]=input;
                if(raw.is_open()){raw.write(reinterpret_cast<const char*>(input.data()),input.size()*sizeof(float));require(bool(raw),"output dump write failed");}
            }
            Header stop{magic,version,0,0,0,0,++seq};send(up,20,stop.data(),sizeof(stop));check_ack(up,seq);
            if(direct){send(down,20,stop.data(),sizeof(stop));check_ack(down,seq);}
            std::printf("PRIMARY PASS cases=%d repeat_bit_exact=1\n",cases);
        }
#ifdef STRATA_FP4_NATIVE_CPU
        if(worker_nt1_tasks)std::printf("CPU_TASK_COUNTS nt1=%u other=%u\n",worker_nt1_calls,worker_other_calls);
#endif
        first_lane.reset();second_lane.reset();
        registered_staging.clear();
        if(down)ucomm_finalize(down);
        if(up)ucomm_finalize(up);
        std::puts(direct?"PASS three-node direct partial-weight MoE":"PASS three-node tree partial-weight MoE");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
