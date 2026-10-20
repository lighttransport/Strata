#include "experiment/glm_fp4_local/native_cpu_expert.hpp"
#include "ggml-cpu.h"
#include <iostream>
using namespace strata;
int main(int argc,char**argv){try{
 if(argc<2||argc>3||(argc==3&&std::string(argv[2])!="preserve-small"))throw std::runtime_error("usage: adapter-test PARTIAL_DIR [preserve-small]");
 const bool preserve_small=argc==3;
 ggml_cpu_init();artifact::Safetensors ts(argv[1]);kernels::cpu::ExpertPool pool(3,true);int checks=0;
 for(int layer:{3,4}){
  std::array<artifact::GlmFp4Matrix,3>w;int p=0;
  for(const char* name:{"gate_proj","up_proj","down_proj"})w[p++]=artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+name);
  experiment::NativeFp4CpuExpert expert(w,preserve_small), k32(w,preserve_small,true);
  std::vector<float> workspace;std::vector<uint8_t> scratch;
  for(int nt:{1,2,3,4,7,8,9,15,16,1,8,2,15,3}){
   std::vector<float> input(size_t(nt)*4096);for(size_t i=0;i<input.size();++i)input[i]=(preserve_small?.001f:1.f)*float(int((i*19)%257)-128)/128;
   auto got=expert.run(pool,input,nt),packed=k32.run(pool,input,nt),repeat=expert.run(pool,input,nt);
   if(std::memcmp(got.data(),packed.data(),got.size()*sizeof(float)))throw std::runtime_error("K32 adapter mismatch");++checks;if(std::memcmp(got.data(),repeat.data(),got.size()*sizeof(float)))throw std::runtime_error("repeat mismatch");
   expert.run_into(pool,input,nt,workspace,scratch);
   if(workspace.size()!=got.size()||std::memcmp(workspace.data(),got.data(),got.size()*sizeof(float)))throw std::runtime_error("workspace output mismatch");++checks;
   for(int t=0;t<nt;++t){std::vector<float>x(input.begin()+t*4096,input.begin()+(t+1)*4096);auto ref=expert.run(pool,x,1);
    if(std::memcmp(ref.data(),got.data()+t*4096,4096*4))throw std::runtime_error("single-token reference mismatch");++checks;}
  }
  {std::vector<float> alias(4096,1.f);bool rejected=false;
   try{expert.run_into(pool,alias,1,alias,scratch);}catch(const std::invalid_argument&){rejected=true;}
   if(!rejected||alias!=std::vector<float>(4096,1.f))throw std::runtime_error("workspace alias accepted or modified");++checks;}
  for(int nt:{0,17}){bool rejected=false;try{expert.run_into(pool,{},nt,workspace,scratch);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)throw std::runtime_error("invalid extent accepted");++checks;}
  for(int nt:{1,8,9,16})for(uint32_t bits:{0x7f800000u,0xff800000u,0x7fc00001u,0xffc00001u,0x7f800001u,0xff800001u}) {
   std::vector<float> bad(size_t(nt)*4096);
   for(size_t pos:{size_t(0),bad.size()-1}) {
    bad[pos]=std::bit_cast<float>(bits);bool rejected=false;
    try{expert.run_into(pool,bad,nt,workspace,scratch);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("nonfinite accepted");++checks;bad[pos]=0.f;
   }
  }
 }
 std::cout<<"PASS checks="<<checks<<" repeated_and_single_token_bit_exact=1 native_q8=1 no_gpu=1\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
