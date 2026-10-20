#include "strata/kernels/cpu/native_task_policy.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace strata::kernels::cpu;
static void check(bool ok) {
    if (!ok) throw std::runtime_error("FP4 route task check failed");
}
struct Transform final : ExpertHiddenTransform {
    int calls = 0;
    void apply(int, int, int, float* values, int count) override {
        ++calls;
        for (int i = 0; i < count; ++i) values[i] *= .5f;
    }
};
struct Observer final : ExpertObserver {
    void hidden(int, int, const float*, int) override {}
};
int main() { try {
    if (!cpu_avx2_ok()) return 77;
    NativeFmt f; std::string error;
    check(native_fmt(kNvfp4Type,kNvfp4Type,4096,2048,f,error));
    f.swiglu_limit=10;
    std::vector<ExpertJobMulti> jobs(4);
    for (auto& j:jobs) j.nt=1;
    int decisions=0;
    auto policy=[&](bool enabled,int base,int fixed,int participants,bool owned,int count) {
        ++decisions;
        return detail::nvfp4_split_tasks(enabled,base,fixed,participants,owned,f,jobs.data(),count);
    };
    for (int nt:{1,2,3,4,8}) {
        for (auto& j:jobs) j.nt=nt;
        for (int n:{0,1,2,3,4}) {
            check(policy(true,3,0,4,false,n)==((n==3&&nt<=2)||(n==2&&nt==1)?6:3));
            check(policy(false,3,0,4,false,n)==3);
            check(policy(true,12,0,4,false,n)==12);
            check(policy(true,3,24,4,false,n)==3);
            check(policy(true,3,0,16,false,n)==3);
            check(policy(true,3,0,4,true,n)==3);
        }
    }
    for (auto& j:jobs) j.nt=1;
    jobs[1].nt=2;check(policy(true,3,0,4,false,3)==3);jobs[1].nt=1;
    check(detail::nvfp4_split_tasks(true,3,0,4,false,f,nullptr,3)==3);
    for (auto field:{&NativeFmt::gu_type,&NativeFmt::d_type}) {
        int saved=f.*field;f.*field=10;check(policy(true,3,0,4,false,3)==3);f.*field=saved;
    }
    f.n_embd=2048;check(policy(true,3,0,4,false,3)==3);f.n_embd=4096;
    f.n_ff=1024;check(policy(true,3,0,4,false,3)==3);f.n_ff=2048;
    f.nvfp4_k32=true;check(policy(true,3,0,4,false,3)==3);f.nvfp4_k32=false;
    f.canon=true;check(policy(true,3,0,4,false,3)==3);f.canon=false;
    f.fuse_h_quant=true;check(policy(true,3,0,4,false,3)==3);f.fuse_h_quant=false;
    f.gate_skip=.5f;check(policy(true,3,0,4,false,3)==3);f.gate_skip=0;
    f.swiglu_limit=0;check(policy(true,3,0,4,false,3)==3);f.swiglu_limit=10;
    Observer observer;f.observer=&observer;check(policy(true,3,0,4,false,3)==3);f.observer=nullptr;
    Transform transform;f.hidden_transform=&transform;check(policy(true,3,0,4,false,3)==6);
    // Coexistence: applying the FP4 policy after Q2/Q3 must retain its choice.
    NativeFmt q23;check(native_fmt(10,11,4096,2048,q23,error));q23.swiglu_limit=10;
    int q23_tasks=detail::q23_split_tasks(true,3,0,4,false,q23,jobs.data(),3);
    check(q23_tasks==12);
    check(detail::nvfp4_split_tasks(true,q23_tasks,0,4,false,q23,jobs.data(),3)==12);
    check(detail::q23_split_tasks(true,3,0,4,false,f,jobs.data(),3)==3);

    // Finite FP4 blocks and independent token inputs; shared weights are legal.
    std::vector<uint8_t> blob(f.bytes);
    for (size_t off:{size_t(0),f.up_off,f.down_off}) {
        size_t end=off==0?f.up_off:off==f.up_off?f.down_off:f.tail_off;
        for (size_t b=off;b<end;b+=36) {
            std::fill_n(blob.data()+b,4,uint8_t(32));
            for (int k=4;k<36;++k) blob[b+k]=uint8_t((b/36+k*19)%256);
        }
    }
    const float scales[4]={.125f,.125f,.125f,0.f};
    check(f.tail_off+sizeof(scales)<=blob.size());
    std::memcpy(blob.data()+f.tail_off,scales,sizeof(scales));
    std::vector<std::vector<uint8_t>> act(3,std::vector<uint8_t>(f.act_bytes));
    std::vector<float> x(4096);
    for (int t=0;t<3;++t) {
        for (int i=0;i<4096;++i) x[i]=float((i*13+t*17)%257-128)/128.f;
        native_quant_act(f,x.data(),act[t].data());
    }
    std::vector<float> output(4*3*4096);
    for (int e=0;e<4;++e) {
        jobs[e].expert_id=e;jobs[e].blob=blob.data();
        for (int t=0;t<3;++t) {
            jobs[e].nact[t]=act[t].data();
            jobs[e].out[t]=output.data()+(e*3+t)*4096;
        }
    }
    ExpertPool pool(3,true);pool.set_native_tasks_per_thread(3);pool.set_q23_route_tasks(true);
    int comparisons=0;
    for (bool transform_enabled:{false,true}) {
        f.hidden_transform=transform_enabled?&transform:nullptr;
        for (int nt:{1,2,3,2,1}) for (int count:{3,1,4,2,3,0}) {
            for (auto& j:jobs) j.nt=nt;
            auto run=[&](bool enabled) {
                pool.set_nvfp4_route_tasks(enabled);
                std::fill(output.begin(),output.end(),0.f);transform.calls=0;
                pool.run_split_multi_native(f,jobs.data(),count);
                check(transform.calls==(transform_enabled?count*nt:0));
                for(float v:output) check(std::isfinite(v));
                if(count) check(std::any_of(output.begin(),output.end(),[](float v){return v!=0.f;}));
            };
            run(false);const auto expected=output;run(true);
            check(!std::memcmp(output.data(),expected.data(),output.size()*sizeof(float)));
            ++comparisons;
        }
    }
    std::cout<<"PASS decisions="<<decisions<<" changing_route_comparisons="<<comparisons<<"\n";
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
