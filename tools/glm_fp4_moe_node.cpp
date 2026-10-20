// Persistent bounded two-node routed MoE test. No production decoder defaults change.
#include "strata/kernels/fp4_local.hpp"
#include "ucomm.h"
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
static Matrix partition(const Matrix& in,size_t begin,size_t end,bool down) {
    require(begin<end&&begin%16==0&&end%16==0&&end<=(down?in.cols:in.rows),"invalid expert split");
    Matrix out;out.rows=down?in.rows:end-begin;out.cols=down?end-begin:in.cols;out.global=in.global;
    out.codes.resize(out.rows*out.cols/2);out.scales.resize(out.rows*out.cols/16);
    for(size_t r=0;r<out.rows;++r) {
        size_t row=down?r:r+begin,col=down?begin:0;
        std::copy_n(in.codes.data()+row*in.cols/2+col/2,out.cols/2,out.codes.data()+r*out.cols/2);
        std::copy_n(in.scales.data()+row*in.cols/16+col/16,out.cols/16,out.scales.data()+r*out.cols/16);
    }
    return out;
}
static void send_matrix(ucomm_t* c,const Matrix& m) {
    uint32_t shape[2]={uint32_t(m.rows),uint32_t(m.cols)};send(c,1,shape,sizeof(shape));send(c,2,&m.global,4);
    send(c,3,m.codes.data(),m.codes.size());send(c,4,m.scales.data(),m.scales.size());
}
static Matrix recv_matrix(ucomm_t* c,int projection) {
    Matrix m;uint32_t shape[2];recv_exact(c,1,shape,sizeof(shape));
    require(shape[0]==(projection==2?4096u:640u)&&shape[1]==(projection==2?640u:4096u),"worker matrix shape mismatch");
    m.rows=shape[0];m.cols=shape[1];recv_exact(c,2,&m.global,4);
    m.codes.resize(m.rows*m.cols/2);m.scales.resize(m.rows*m.cols/16);
    recv_exact(c,3,m.codes.data(),m.codes.size());recv_exact(c,4,m.scales.data(),m.scales.size());return m;
}
static float round_bf16(float x){uint32_t b;std::memcpy(&b,&x,4);b+=0x7fff+((b>>16)&1);b&=0xffff0000;std::memcpy(&x,&b,4);return x;}
struct Expert {
    bool tiled;
    std::array<std::unique_ptr<strata::kernels::Fp4LocalGpu>,3> gpu;
    explicit Expert(const std::array<Matrix,3>& w,bool use_tiled=false):tiled(use_tiled){for(int i=0;i<3;++i)gpu[i]=std::make_unique<strata::kernels::Fp4LocalGpu>(w[i]);}
    std::vector<float> run(const std::vector<float>& input,int nt) {
        auto gate=gpu[0]->run(input,nt,true,tiled),up=gpu[1]->run(input,nt,true,tiled);
        for(size_t i=0;i<gate.size();++i) {
            float g=std::min(round_bf16(gate[i]),10.f),u=std::clamp(round_bf16(up[i]),-10.f,10.f);
            gate[i]=round_bf16(round_bf16(g/(1+std::exp(-g)))*u);
        }
        return gpu[2]->run(gate,nt,true,tiled);
    }
};
// Versioned bounded protocol; both hosts use little-endian IEEE float32.
static constexpr uint32_t magic=0x34465053,version=1;
using Command=std::array<uint32_t,6>; // magic, version, op, slot, tokens, sequence
static void command(ucomm_t* c,uint32_t op,uint32_t slot,uint32_t nt,uint32_t seq) {
    Command h{magic,version,op,slot,nt,seq};send(c,20,h.data(),sizeof(h));
}
static double compare(const std::vector<float>& got,const std::vector<float>& ref) {
    require(got.size()==ref.size(),"comparison shape mismatch");double err=0,norm=0;
    for(size_t i=0;i<got.size();++i) {
        require(std::isfinite(got[i])&&std::isfinite(ref[i]),"nonfinite MoE result");
        double e=double(got[i])-ref[i];err+=e*e;norm+=double(ref[i])*ref[i];
    }
    double rel=std::sqrt(err/std::max(norm,1e-30));require(rel<0.005,"routed split/unsplit error exceeds 0.5 percent L2");return rel;
}
int main(int argc,char** argv) {
    try {
        require(argc==4||argc==5,"usage: glm-fp4-moe-node RANK ROOT:PORT MODEL_DIR [tiled]");
        int rank=std::stoi(argv[1]);require(rank==0||rank==1,"rank must be 0 or 1");
        bool tiled=argc==5;require(!tiled||std::string(argv[4])=="tiled","unknown kernel variant");
        uint32_t endian=1;require(*reinterpret_cast<uint8_t*>(&endian)==1,"protocol requires little endian");
        std::setvbuf(stdout,nullptr,_IOLBF,0);strata::kernels::fp4_local_device_info();
        std::unique_ptr<strata::artifact::Safetensors> ts;
        if(!rank)ts=std::make_unique<strata::artifact::Safetensors>(argv[3]);
        ucomm_config cfg;ucomm_config_default(&cfg);cfg.rank=rank;cfg.world=2;cfg.root_addr=argv[2];
        cfg.timeout_ms=120000;cfg.backend=UCOMM_BACKEND_IB;
        ucomm_t* comm=nullptr;ok(ucomm_init(&cfg,&comm));
        std::printf("TRANSPORT %s rank=%d protocol=%u\n",ucomm_backend_name(comm),rank,version);
        Command hello{magic,version,4,4096,2048,1408},peer{};
        if(!rank){send(comm,21,hello.data(),sizeof(hello));recv_exact(comm,21,peer.data(),sizeof(peer));}
        else{recv_exact(comm,21,peer.data(),sizeof(peer));send(comm,21,hello.data(),sizeof(hello));}
        require(peer==hello,"protocol or split configuration mismatch");
        std::array<std::unique_ptr<Expert>,4> cached,baseline;
        const char* projections[]={"gate_proj","up_proj","down_proj"};size_t packed_bytes=0;
        for(int slot=0;slot<4;++slot) {
            std::array<Matrix,3> whole,local;
            for(int p=0;p<3;++p) {
                if(!rank) {
                    std::string prefix="model.language_model.layers."+std::to_string(3+slot/2)+".mlp.experts."+std::to_string(slot%2)+"."+projections[p];
                    whole[p]=strata::artifact::glm_load_fp4(*ts,prefix);
                    require(whole[p].rows==(p==2?4096u:2048u)&&whole[p].cols==(p==2?2048u:4096u),"unexpected GLM expert dimensions");
                    local[p]=partition(whole[p],0,1408,p==2);
                    auto remote=partition(whole[p],1408,2048,p==2);send_matrix(comm,remote);
                }else local[p]=recv_matrix(comm,p);
                packed_bytes+=local[p].bytes();
            }
            cached[slot]=std::make_unique<Expert>(local,tiled);
            if(!rank)baseline[slot]=std::make_unique<Expert>(whole);
        }
        ok(ucomm_barrier(comm));
        std::printf("READY rank=%d experts=4 packed_device_bytes=%zu weights_loaded_once=1\n",rank,packed_bytes);
        uint32_t sequence=0;
        if(rank) {
            for(;;) {
                Command h{};recv_exact(comm,20,h.data(),sizeof(h));
                require(h[0]==magic&&h[1]==version&&h[5]==++sequence,"invalid protocol/sequence");
                if(h[2]==0){require(h[3]==0&&h[4]==0,"invalid stop command");break;}
                require(h[2]==1&&h[3]<4&&h[4]>=1&&h[4]<=16,"invalid run command bounds");
                std::vector<float> input(size_t(h[4])*4096);recv_exact(comm,10,input.data(),input.size()*4);
                auto output=cached[h[3]]->run(input,h[4]);
                send(comm,22,&sequence,sizeof(sequence));send(comm,11,output.data(),output.size()*4);
            }
            std::printf("WORKER PASS run_commands=%u weights_reloaded=0\n",sequence-1);
        }else {
            std::array<std::vector<float>,4> previous;
            const int batches[]={1,7,16,3};int cases=0;
            for(int repeat=0;repeat<2;++repeat)for(int scenario=0;scenario<4;++scenario) {
                int nt=batches[scenario];std::vector<float> input(size_t(nt)*4096);
                std::mt19937 rng(8141+scenario);for(float& x:input)x=round_bf16((int(rng()%513)-256)/256.f);
                for(int layer=0;layer<2;++layer) {
                    std::vector<float> mixed(input.size()),reference(input.size());
                    for(int expert=0;expert<2;++expert) {
                        std::vector<int> tokens;std::vector<float> gains,group;
                        // Top-1 sparse/uneven routes and top-2 nonuniform normalized gains.
                        for(int t=nt-1;t>=0;--t) {
                            int route=scenario==3?1:(t%3==0?0:1);
                            if(scenario!=2&&route!=expert)continue;
                            tokens.push_back(t);gains.push_back(scenario==2?(expert==0?float(t+1)/18.f:1.f-float(t+1)/18.f):1.f);
                            group.insert(group.end(),input.begin()+size_t(t)*4096,input.begin()+size_t(t+1)*4096);
                        }
                        if(tokens.empty())continue;
                        int slot=layer*2+expert,m=int(tokens.size());
                        command(comm,1,slot,m,++sequence);send(comm,10,group.data(),group.size()*4);
                        // Worker and primary compute their FFN partitions concurrently.
                        auto local=cached[slot]->run(group,m);uint32_t response=0;
                        recv_exact(comm,22,&response,sizeof(response));require(response==sequence,"stale worker response");
                        std::vector<float> remote(local.size());recv_exact(comm,11,remote.data(),remote.size()*4);
                        auto expected=baseline[slot]->run(group,m);
                        for(size_t j=0;j<tokens.size();++j)for(size_t h=0;h<4096;++h) {
                            size_t dst=size_t(tokens[j])*4096+h,src=j*4096+h;
                            mixed[dst]+=gains[j]*(local[src]+remote[src]);reference[dst]+=gains[j]*expected[src];
                        }
                    }
                    double rel=compare(mixed,reference);
                    std::printf("ROUTED PASS repeat=%d scenario=%d layer=%d tokens=%d rel_l2=%.9g\n",repeat,scenario,layer+3,nt,rel);++cases;
                    // Synthetic residual keeps activations bounded; no claim of transformer parity.
                    for(size_t i=0;i<input.size();++i)input[i]=round_bf16(std::clamp(input[i]+mixed[i]*0.125f,-4.f,4.f));
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
                if(repeat)require(input==previous[scenario],"persistent worker repeat changed result");
                else previous[scenario]=input;
            }
            command(comm,0,0,0,++sequence);
            std::printf("PRIMARY PASS routed_layer_cases=%d run_commands=%u repeat_bit_exact=1\n",cases,sequence-1);
        }
        ucomm_finalize(comm);std::puts("PASS persistent two-node partial-weight MoE");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
