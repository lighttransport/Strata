#include "experiment/glm_fp4_local/native_cpu_expert.hpp"
#include "ggml-cpu.h"
#include <iostream>
using namespace strata;
static void equal(const std::vector<float>& a,const std::vector<float>& b) {
 if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float)))throw std::runtime_error("compact scales changed output");
}
int main(int argc,char**argv){try {
 if(argc<2||argc>3||(argc==3&&std::string(argv[2])!="native-k32"))throw std::runtime_error("usage: compact-test PARTIAL_DIR [native-k32]");
 const bool k32=argc==3;
 ggml_cpu_init();kernels::cpu::ExpertPool pool(15,true);
 if(!pool.numa_rows_available()){std::cout<<"SKIP requires two-node pool\n";return 77;}
 artifact::Safetensors ts(argv[1]);int comparisons=0,rejections=0;
 for(int layer:{3,4}) {
  std::array<artifact::GlmFp4Matrix,3>w;int p=0;
  for(const char* name:{"gate_proj","up_proj","down_proj"})w[p++]=artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+name);
  for(bool preserve:{false,true}) {
   experiment::NativeFp4CpuExpert reference(w,preserve),compact(w,preserve,k32);
   bool rejected=false;try{compact.release_original_blob();}catch(const std::invalid_argument&){rejected=true;}
   if(!rejected)throw std::runtime_error("unprepared release accepted");++rejections;
   if(!reference.prepare_numa(pool)||!compact.prepare_numa(pool))throw std::runtime_error("prepare failed");
   const size_t before=compact.retained_blob_bytes();if(!before)throw std::runtime_error("missing original storage");
   compact.release_original_blob();compact.release_original_blob();
   if(compact.retained_blob_bytes()!=0||!compact.prepare_numa(pool)||compact.retained_blob_bytes()!=0)throw std::runtime_error("original blob retained or recreated");
   rejected=false;try{compact.run(pool,std::vector<float>(4096,1.f),1);}catch(const std::invalid_argument&){rejected=true;}
   if(!rejected)throw std::runtime_error("released complete-blob path accepted");++rejections;
   std::vector<float> workspace;std::vector<uint8_t> scratch;
   for(float amplitude:{0.f,1.f,.001f,64.f})for(int nt:{1,2,3,4,7,8,9,15,16}) {
    std::vector<float>x(size_t(nt)*4096);for(size_t i=0;i<x.size();++i)x[i]=amplitude*float(int(i*19%257)-128)/128.f;
    auto ref=reference.run(pool,x,nt);equal(ref,reference.run(pool,x,nt,true));equal(ref,compact.run(pool,x,nt,true));equal(ref,compact.run(pool,x,nt,true));compact.run_into(pool,x,nt,workspace,scratch,true);equal(ref,workspace);comparisons+=4;
   }
   std::cout<<"STORAGE layer="<<layer<<" preserve="<<preserve<<" original_capacity="<<before<<" after_release="<<compact.retained_blob_bytes()<<" scale_storage=32\n";
  }
 }
 std::cout<<"PASS comparisons="<<comparisons<<" rejections="<<rejections<<" no_gpu=1\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
