#include "native_cpu_expert.hpp"
#include "ggml-cpu.h"
#include <iostream>
#include <limits>
using namespace strata;
using Expert=experiment::NativeFp4CpuExpert;
static void require(bool yes){if(!yes)throw std::runtime_error("batch assertion failed");}
template<size_t Count> static void check_mode(const char* path,bool numa,int tokens){
 auto run_batch=[&](auto& pool,const auto& experts,const auto& input,auto& output,auto& act,bool use_numa=false){Expert::run_batch_into(pool,experts,input,output,act,use_numa,tokens);};
 ggml_cpu_init();kernels::cpu::ExpertPool pool(numa?15:3,true);
 const auto affinity=numa?kernels::cpu::pin_current_thread(kernels::cpu::planned_host_core()):kernels::cpu::ThreadAffinity{};
 if(numa)require(affinity.valid&&pool.numa_rows_available());
 artifact::Safetensors ts(path);int comparisons=0,rejections=0;
 for(int layer:{3,4}){
  std::array<std::unique_ptr<Expert>,Count> owned;
  std::array<const Expert*,Count> experts;
  std::array<artifact::GlmFp4Matrix,3> last_weights;
  for(int e=0;e<int(Count);++e){int i=0;for(const char* n:{"gate_proj","up_proj","down_proj"})last_weights[i++]=artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(e)+"."+n);
   owned[e]=std::make_unique<Expert>(last_weights,true,false);experts[e]=owned[e].get();
  }
  std::array<std::vector<float>,Count> output;std::vector<uint8_t> act(17,71);std::vector<float> x(size_t(tokens)*4096,1.f);
  auto rejected=[&](auto fn){auto old=output;auto scratch=act;bool caught=false;try{fn();}catch(const std::invalid_argument&){caught=true;}require(caught&&output==old&&act==scratch);++rejections;};
  for(auto& v:output)v.assign(11,42.f);
  for(int bad_tokens:{0,3})rejected([&]{Expert::run_batch_into(pool,experts,x,output,act,false,bad_tokens);});
  auto bad=experts;bad[Count-1]=nullptr;rejected([&]{run_batch(pool,bad,x,output,act);});
  x.resize(size_t(tokens)*4096-1);rejected([&]{run_batch(pool,experts,x,output,act);});x.resize(size_t(tokens)*4096,1.f);
  x[17]=std::numeric_limits<float>::infinity();rejected([&]{run_batch(pool,experts,x,output,act);});x[17]=1;
  Expert mixed(last_weights,false,false);bad=experts;bad[Count-1]=&mixed;rejected([&]{run_batch(pool,bad,x,output,act);});
  Expert k32(last_weights,true,true);bad[Count-1]=&k32;rejected([&]{run_batch(pool,bad,x,output,act);});
  output[Count-1]=x;rejected([&]{run_batch(pool,experts,output[Count-1],output,act);});
  rejected([&]{run_batch(pool,experts,x,output,act,true);});
  // An all-disabled policy and repeated expert pointers are valid. Dense job
  // IDs still keep each output slot independent.
  std::array<const Expert*,Count> repeated;repeated.fill(&mixed);
  run_batch(pool,repeated,x,output,act);
  auto plain=mixed.run(pool,x,tokens);
  for(const auto& v:output){require(v.size()==plain.size()&&!std::memcmp(v.data(),plain.data(),plain.size()*sizeof(float)));++comparisons;}
  if(numa)for(auto& e:owned)require(e->prepare_numa(pool));
  for(float amplitude:{1.f,.001f,0.f,1.f}){
   for(int i=0;i<tokens*4096;++i)x[i]=amplitude*float((i*19%257)-128)/128.f*(i<4096?1.f:.001f);
   run_batch(pool,experts,x,output,act,numa);
   for(int e=0;e<int(Count);++e){auto expected=experts[e]->run(pool,x,tokens,numa);require(output[e].size()==expected.size()&&!std::memcmp(output[e].data(),expected.data(),expected.size()*sizeof(float)));++comparisons;}
  }
  if(numa){
   std::array<std::array<std::vector<float>,Count>,4> before;
   const float amplitudes[]={1.f,.001f,0.f,1.f};
   for(int a=0;a<4;++a){
    for(int i=0;i<tokens*4096;++i)x[i]=amplitudes[a]*float((i*19%257)-128)/128.f*(i<4096?1.f:.001f);
    run_batch(pool,experts,x,before[a],act,true);
   }
   owned[Count-1]->release_original_blob();rejected([&]{run_batch(pool,experts,x,output,act,true);});
   for(auto& e:owned){e->release_original_blob();require(e->retained_blob_bytes()==0);}
   rejected([&]{run_batch(pool,experts,x,output,act,false);});
   for(int a=0;a<4;++a){
    for(int i=0;i<tokens*4096;++i)x[i]=amplitudes[a]*float((i*19%257)-128)/128.f*(i<4096?1.f:.001f);
    run_batch(pool,experts,x,output,act,true);
    for(int e=0;e<int(Count);++e){auto expected=experts[e]->run(pool,x,tokens,true);require(output[e].size()==before[a][e].size()&&!std::memcmp(output[e].data(),before[a][e].data(),output[e].size()*sizeof(float)));require(output[e].size()==expected.size()&&!std::memcmp(output[e].data(),expected.data(),expected.size()*sizeof(float)));++comparisons;}
   }
  }
  // The same pool and buffers must survive NT2 -> NT1 -> NT2 transitions.
  if(tokens==2)for(int active_tokens:{2,1,2,1}){
   x.resize(size_t(active_tokens)*4096);
   for(size_t i=0;i<x.size();++i)x[i]=float(int(i*13%257)-128)/128.f*(i<4096?.001f:.000001f);
   Expert::run_batch_into(pool,experts,x,output,act,numa,active_tokens);
   for(size_t e=0;e<Count;++e){auto expected=experts[e]->run(pool,x,active_tokens,numa);require(output[e].size()==expected.size()&&!std::memcmp(output[e].data(),expected.data(),expected.size()*sizeof(float)));++comparisons;}
  }
 }
 // Different shifts, expert IDs and token slots must not overwrite one another.
 experiment::MultiExpertHiddenScale<3> scales;
 std::array<float,3> values={std::ldexp(1.f,-20),std::ldexp(1.f,-40),std::numeric_limits<float>::denorm_min()};
 const auto original=values;
 for(int e=0;e<3;++e)scales.apply(0,e,e,&values[e],1);
 for(int e=2;e>=0;--e){scales.restore(e,e,&values[e],1);require(values[e]==original[e]);}
 for(int id:{-1,3}){bool caught=false;float v=1;try{scales.apply(0,id,0,&v,1);}catch(const std::invalid_argument&){caught=true;}require(caught&&v==1);}
 std::cout<<"PASS experts="<<Count<<" tokens="<<tokens<<" comparisons="<<comparisons<<" rejected="<<rejections<<" independent_scales=3 numa="<<numa<<"\n";
 if(affinity.valid)kernels::cpu::restore_thread_affinity(affinity);
}
int main(int argc,char**argv){try{
 require(argc==3);require(std::string(argv[2])=="numa"||std::string(argv[2])=="complete");
 const bool numa=std::string(argv[2])=="numa";
 for(int tokens:{1,2}){check_mode<2>(argv[1],numa,tokens);check_mode<3>(argv[1],numa,tokens);}
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
