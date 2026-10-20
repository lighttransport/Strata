// Experimental two-node FP4 expert split. No production decoder defaults change.
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
int main(int argc,char** argv) {
    ucomm_t* comm=nullptr;
    try {
        require(argc>=4&&argc<=6,"usage: glm-fp4-node RANK ROOT:PORT MODEL_DIR [ib|udp] [tiled]");
        bool tiled=argc==6&&std::string(argv[5])=="tiled";
        require(argc!=6||tiled,"unknown kernel variant");
        int rank=std::stoi(argv[1]);require(rank==0||rank==1,"rank must be 0 or 1");
        std::setvbuf(stdout,nullptr,_IOLBF,0);strata::kernels::fp4_local_device_info();
        ucomm_config cfg;ucomm_config_default(&cfg);cfg.rank=rank;cfg.world=2;cfg.root_addr=argv[2];cfg.timeout_ms=120000;
        cfg.backend=argc>=5&&std::string(argv[4])=="udp"?UCOMM_BACKEND_UDP:UCOMM_BACKEND_IB;
        ok(ucomm_init(&cfg,&comm));std::printf("TRANSPORT %s rank=%d\n",ucomm_backend_name(comm),rank);
        std::unique_ptr<strata::artifact::Safetensors> ts;if(!rank)ts=std::make_unique<strata::artifact::Safetensors>(argv[3]);
        const char* names[]={"gate_proj","up_proj","down_proj"};
        for(int layer:{3,4}) {
            std::array<Matrix,3> whole,local;
            for(int i=0;i<3;++i) {
                if(!rank) {
                    whole[i]=strata::artifact::glm_load_fp4(*ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+names[i]);
                    local[i]=partition(whole[i],0,1408,i==2);auto remote=partition(whole[i],1408,2048,i==2);send_matrix(comm,remote);
                }else local[i]=recv_matrix(comm,i);
            }
            Expert expert(local,tiled);
            for(int nt:{1,16}) {
                std::vector<float> input(size_t(nt)*4096);
                if(!rank) {
                    std::mt19937 rng(1234+layer+nt);for(auto& x:input)x=round_bf16((int(rng()%513)-256)/256.f);
                    send(comm,10,input.data(),input.size()*4);
                }else recv_exact(comm,10,input.data(),input.size()*4);
                auto got=expert.run(input,nt);
                if(rank)send(comm,11,got.data(),got.size()*4);
                else {
                    std::vector<float> remote(got.size());recv_exact(comm,11,remote.data(),remote.size()*4);
                    Expert baseline(whole);auto expected=baseline.run(input,nt);double num=0,den=0,maxerr=0;
                    for(size_t i=0;i<got.size();++i) {got[i]+=remote[i];double e=double(got[i])-expected[i];num+=e*e;den+=double(expected[i])*expected[i];maxerr=std::max(maxerr,std::abs(e));}
                    double rel=std::sqrt(num/den);require(std::isfinite(rel)&&rel<0.005,"split/unsplit MLP error exceeds 0.5 percent L2");
                    std::printf("SPLIT PASS layer=%d expert=0 m=%d local_ff=1408 remote_ff=640 max_abs=%.9g rel_l2=%.9g\n",layer,nt,maxerr,rel);
                }
                ok(ucomm_barrier(comm));std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        }
        ucomm_finalize(comm);comm=nullptr;std::puts("PASS two layers, both GPUs, four split MLP cases");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
