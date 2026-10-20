#include "strata/kernels/cpu/native_task_policy.hpp"
#include "ggml.h"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace strata::kernels::cpu;
static void check(bool ok) { if (!ok) throw std::runtime_error("Q2/Q3 route task check failed"); }
int main() { try {
    if (!cpu_avx2_ok()) return 77;
    NativeFmt f; std::string error;
    check(native_fmt(10,11,4096,2048,f,error)); f.swiglu_limit=10;
    std::vector<ExpertJobMulti> jobs(8); for(auto& j:jobs)j.nt=1;
    int decisions=0;
    auto policy=[&](bool enabled,int base,int fixed,int participants,bool owned,int n) {
        ++decisions; return detail::q23_split_tasks(enabled,base,fixed,participants,owned,f,jobs.data(),n);
    };
    for(int n:{0,1,2,3,4,5,6,7,8}) {
        check(policy(true,3,0,4,false,n)==((n==3||n==4||n==8)?12:3));
        check(policy(false,3,0,4,false,n)==3);
        check(policy(true,6,0,4,false,n)==6);
        check(policy(true,3,41,4,false,n)==3);
        check(policy(true,3,0,16,false,n)==3);
        check(policy(true,3,0,4,true,n)==3);
    }
    f.n_ff=1024;
    for(int n=0;n<=8;++n)check(policy(true,3,0,4,false,n)==(n==8?12:3));
    f.n_ff=1536;check(policy(true,3,0,4,false,8)==3);f.n_ff=2048;
    jobs[1].nt=2;check(policy(true,3,0,4,false,3)==3);jobs[1].nt=1;
    for(auto field:{&NativeFmt::gu_type,&NativeFmt::d_type}) {auto old=f.*field;f.*field=2;check(policy(true,3,0,4,false,3)==3);f.*field=old;}
    f.canon=true;check(policy(true,3,0,4,false,3)==3);f.canon=false;
    f.fuse_h_quant=true;check(policy(true,3,0,4,false,3)==3);f.fuse_h_quant=false;
    f.q3_expanded_down=true;check(policy(true,3,0,4,false,3)==3);f.q3_expanded_down=false;
    f.gate_skip=.5f;check(policy(true,3,0,4,false,3)==3);f.gate_skip=0;
    f.swiglu_limit=0;check(policy(true,3,0,4,false,3)==3);f.swiglu_limit=10;
    int comparisons=0;
    for(int width:{2048,1024}) {
    check(native_fmt(10,11,4096,width,f,error));f.swiglu_limit=10;
    std::vector<uint8_t> blob(f.bytes);std::vector<float> w(4096*width);
    for(size_t i=0;i<w.size();++i)w[i]=float(int(i*17%251)-125)*.0004f;
    ggml_quantize_chunk(GGML_TYPE_Q2_K,w.data(),blob.data(),0,width,4096,nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q2_K,w.data(),blob.data()+f.up_off,0,width,4096,nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q3_K,w.data(),blob.data()+f.down_off,0,4096,width,nullptr);
    std::vector<float> x(4096),out(8*4096),sum(4096),weights(8,.125f);
    for(int i=0;i<4096;++i)x[i]=std::sin(i*.031f);
    std::vector<uint8_t> act(f.act_bytes);native_quant_act(f,x.data(),act.data());
    for(int i=0;i<8;++i){jobs[i].expert_id=i;jobs[i].blob=blob.data();jobs[i].nact[0]=act.data();jobs[i].out[0]=out.data()+i*4096;}
    ExpertPool pool(3,true);pool.set_native_tasks_per_thread(3);
    for(int repetition=0;repetition<3;++repetition)for(int n:{3,1,8,2,4,5,3,0}) {
        auto run=[&](bool enabled){pool.set_q23_route_tasks(enabled);std::fill(out.begin(),out.end(),0.f);pool.run_split_multi_native(f,jobs.data(),n);pool.reduce_routed(out.data(),weights.data(),sum.data(),1,8,4096);};
        run(false);const auto expected=out,expected_sum=sum;run(true);
        check(!std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)));
        check(!std::memcmp(sum.data(),expected_sum.data(),sum.size()*sizeof(float)));++comparisons;
    }
    }
    std::cout<<"PASS decisions="<<decisions<<" changing_route_comparisons="<<comparisons<<"\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
