#include "strata/artifact/expert_block_screen.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"
#include <cstring>
#include <iostream>
#include <random>
#include <unistd.h>
using namespace strata::kernels::cpu;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F call){bool rejected=false;try{call();}catch(const std::exception&){rejected=true;}require(rejected,"invalid input accepted");}
struct Transform:ExpertHiddenTransform {
    int keep=128,calls=0;std::array<float,2048> norm{};
    Transform(){norm.fill(1);}
    void apply(int layer,int expert,int token,float* h,int n)override {
        require(layer==7&&expert>=0&&expert<2&&token>=0&&token<4,"pool transform identity changed");++calls;
        expert_block_zero(h,n,expert_block_topk(expert_block_energy(h,norm.data(),n),keep));
    }
};
int main(){try {
    std::array<float,2048> hidden{},norm{};norm.fill(1);hidden.fill(-2);
    auto scores=expert_block_energy(hidden.data(),norm.data(),2048);auto mask=expert_block_topk(scores,64);
    for(int b=0;b<128;++b)require(mask.contains(b)==(b<64),"equal-energy tie order unstable");
    auto original=hidden;expert_block_zero(hidden.data(),2048,expert_block_topk(scores,128));require(hidden==original,"full mask changed activation bits");
    expert_block_zero(hidden.data(),2048,mask);for(int i=0;i<2048;++i)require(hidden[i]==(i<1024?-2.f:0.f),"masked channels incorrect");
    norm[0]=100;require(expert_block_energy(original.data(),norm.data(),2048)[0]>scores[0],"down norms not used");
    rejects([&]{expert_block_topk(scores,0);});rejects([&]{expert_block_topk(scores,129);});
    norm[0]=-1;rejects([&]{expert_block_energy(hidden.data(),norm.data(),2048);});norm[0]=1;
    hidden[0]=std::numeric_limits<float>::quiet_NaN();rejects([&]{expert_block_energy(hidden.data(),norm.data(),2048);});
    strata::artifact::ExpertBlockNorms bank{123,4,2,256,{0,0,0,1},std::vector<float>(4*2*2048,1)};
    auto path=std::filesystem::temp_directory_path()/("strata-block-norm-test-"+std::to_string(getpid()));
    bank.save(path);auto loaded=strata::artifact::ExpertBlockNorms::load(path,123,4,2,256);require(bank.values==loaded.values,"norm bank round trip changed values");
    rejects([&]{bank.save(path);});rejects([&]{strata::artifact::ExpertBlockNorms::load(path,124,4,2,256);});
    {std::fstream file(path,std::ios::in|std::ios::out|std::ios::binary);file.seekp(-1,std::ios::end);char c=1;file.write(&c,1);}
    rejects([&]{strata::artifact::ExpertBlockNorms::load(path,123,4,2,256);});std::filesystem::remove(path);
    NativeFmt f;std::string error;require(native_fmt(10,11,256,2048,f,error),"native geometry rejected");f.canon=true;f.swiglu_limit=10;f.observer_layer=7;
    std::vector<uint8_t> gate(2*f.up_off),up(gate.size()),down(2*(f.bytes-f.down_off));std::mt19937 rng(837);
    std::vector<float> weights(256*2048);
    for(auto* dest:{&gate,&up,&down})for(int e=0;e<2;++e){for(auto& v:weights)v=float(int(rng()%201)-100)*.001f;bool d=dest==&down;
        ggml_quantize_chunk(d?GGML_TYPE_Q3_K:GGML_TYPE_Q2_K,weights.data(),dest->data()+size_t(e)*(d?f.bytes-f.down_off:f.up_off),0,d?256:2048,d?2048:256,nullptr);}
    std::array<std::vector<uint8_t>,4> acts;std::array<float,256> input{};
    for(auto& a:acts){for(auto& v:input)v=float(int(rng()%201)-100)*.01f;a.resize(f.act_bytes);native_quant_act(f,input.data(),a.data());}
    ExpertPool pool(2,false);Transform transform;
    for(int keep:{128,64,32}) {
        transform.keep=keep;
        std::vector<float> reference(2*4*256);
        for(int e=0;e<2;++e)for(int t=0;t<4;++t){std::array<float,2048> h{};std::vector<uint8_t> q(f.h_bytes);const void* a=acts[t].data();float* hp=h.data();
            native_gu_rows(f,gate.data()+size_t(e)*f.up_off,&a,1,&hp,0,2048,up.data()+size_t(e)*f.up_off);transform.apply(7,e,t,h.data(),2048);native_quant_h(f,h.data(),q.data());
            const void* qa=q.data();float* out=reference.data()+(e*4+t)*256;native_down_rows(f,gate.data(),&qa,1,&out,0,256,down.data()+size_t(e)*(f.bytes-f.down_off));}
        for(int nt=1;nt<=4;++nt){std::vector<float> result(2*nt*256);std::array<ExpertJobMulti,2> jobs{};
            for(int e=0;e<2;++e){auto& j=jobs[e];j.expert_id=e;j.nt=nt;j.blob=gate.data()+size_t(e)*f.up_off;j.native_up=up.data()+size_t(e)*f.up_off;j.native_down=down.data()+size_t(e)*(f.bytes-f.down_off);
                for(int t=0;t<nt;++t){j.nact[t]=acts[t].data();j.out[t]=result.data()+(e*nt+t)*256;}}
            transform.calls=0;f.hidden_transform=&transform;pool.run_split_multi_native(f,jobs.data(),2);require(transform.calls==2*nt,"pool missed hidden transforms");
            for(int e=0;e<2;++e)for(int t=0;t<nt;++t)require(!std::memcmp(result.data()+(e*nt+t)*256,reference.data()+(e*4+t)*256,256*4),"masked output differs by token width");
            std::vector<float> route_weights(nt*2,.5f),sum(nt*256);std::vector<float> routed(nt*2*256);
            for(int t=0;t<nt;++t)for(int e=0;e<2;++e)std::memcpy(routed.data()+(t*2+e)*256,result.data()+(e*nt+t)*256,256*4);
            transform.calls=0;pool.run_layer_native(f,jobs.data(),2,routed.data(),route_weights.data(),sum.data(),nt,2);require(transform.calls==2*nt,"dataflow bypassed transform");
        }
    }
    std::cout<<"Expert block screen PASS masks, norms, corruption, canonical widths 1-4 and pool transforms\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
