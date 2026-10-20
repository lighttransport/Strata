#include "experiment/glm_fp4_local/native_cpu_expert.hpp"
#include "ggml-cpu.h"
#include <chrono>
#include <iostream>
using namespace strata;
static void equal(const std::vector<float>& a,const std::vector<float>& b) {
 if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float)))throw std::runtime_error("NUMA output changed");
}
int main(int argc,char**argv){try {
 if(argc<2||argc>3||(argc==3&&std::string(argv[2])!="bench"))throw std::runtime_error("usage: numa-test PARTIAL_DIR [bench]");
 ggml_cpu_init();kernels::cpu::ExpertPool pool(15,true);
 if(!pool.numa_rows_available()){std::cout<<"SKIP requires two-node CPU pool\n";return 77;}
 const auto previous=kernels::cpu::pin_current_thread(kernels::cpu::planned_host_core());
 artifact::Safetensors ts(argv[1]);int checks=0;
 for(int layer:{3,4}) {
  std::array<artifact::GlmFp4Matrix,3>w;int p=0;
  for(const char* name:{"gate_proj","up_proj","down_proj"})w[p++]=artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+name);
  for(bool preserve:{false,true}) {
   experiment::NativeFp4CpuExpert expert(w,preserve);std::vector<float> x(4096,1.f);
   bool rejected=false;try{expert.run(pool,x,1,true);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)throw std::runtime_error("unprepared rows accepted");++checks;
   auto before=expert.run(pool,x,1);if(!expert.prepare_numa(pool)||!expert.prepare_numa(pool))throw std::runtime_error("NUMA preparation failed");equal(before,expert.run(pool,x,1));++checks;
   for(float amplitude:{1.f,.001f})for(int nt:{1,2,3,4,7,8,9,15,16}) {
    x.resize(size_t(nt)*4096);for(size_t i=0;i<x.size();++i)x[i]=amplitude*float(int(i*19%257)-128)/128.f;
    auto ref=expert.run(pool,x,nt);equal(ref,expert.run(pool,x,nt,true));equal(ref,expert.run(pool,x,nt,true));checks+=2;
   }
   kernels::cpu::ExpertPool small(1,true);
   if(small.numa_rows_available())throw std::runtime_error("expected single-node small pool");
   if(expert.prepare_numa(small))throw std::runtime_error("unsupported NUMA preparation accepted");
   rejected=false;try{expert.run(small,std::vector<float>(4096,1.f),1,true);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)throw std::runtime_error("incompatible pool accepted");++checks;
  }
  if(argc==3)for(int banks:{1,16}) {
   std::vector<std::unique_ptr<experiment::NativeFp4CpuExpert>> experts;
   for(int b=0;b<banks;++b){auto expert=std::make_unique<experiment::NativeFp4CpuExpert>(w,true);if(!expert->prepare_numa(pool))throw std::runtime_error("benchmark prepare failed");experts.push_back(std::move(expert));}
   for(int nt:{1,2,4,8,16}) {
    std::vector<float>x(size_t(nt)*4096);for(size_t i=0;i<x.size();++i)x[i]=float(int(i*19%257)-128)/128.f;
    const auto ref=experts[0]->run(pool,x,nt);size_t next[2]={};
    auto run=[&](bool owned){return experts[next[owned]++%banks]->run(pool,x,nt,owned);};
    for(int pair=0;pair<20;++pair)for(int order=0;order<2;++order) {
     bool owned=bool(order^(pair&1));equal(ref,run(owned));
     auto start=std::chrono::steady_clock::now();std::vector<float>out;
     for(int call=0;call<8;++call)out=run(owned);
     double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/8;
     equal(ref,out);++checks;
     std::cout<<"PAIR layer="<<layer<<" banks="<<banks<<" tokens="<<nt<<" pair="<<pair<<" owned="<<owned<<" ms="<<ms<<" calls=8 bit_exact=1\n";
    }
   }
  }
 }
 if(previous.valid)kernels::cpu::restore_thread_affinity(previous);
 std::cout<<"PASS checks="<<checks<<" no_gpu=1\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
