// Bounded NT1 3/3/2 partial-layer test with real experts 0..7. Not a full decoder.
#include "native_cpu_expert.hpp"
#include "glm_fp4_batch_transport.hpp"
#include "glm_fp4_service_lane.hpp"
#include "ggml-cpu.h"
#include <bit>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
using namespace strata;
using Expert=experiment::NativeFp4CpuExpert;
using Link=experiment::NativeBatchLink;
static void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
static int number(const char* s){size_t used=0;int n=std::stoi(s,&used);require(used==std::strlen(s),"invalid integer");return n;}
static float bf16(float x){auto b=std::bit_cast<uint32_t>(x);b+=0x7fff+((b>>16)&1);return std::bit_cast<float>(b&0xffff0000u);}
int main(int argc,char**argv){try{
 require(argc==8,"usage: batch-tree ROLE MODEL ROOT1 ROOT2 REPEATS POLICY DUMP");
 int role=number(argv[1]),repeats=number(argv[5]),policy=number(argv[6]);
 require(role>=0&&role<=2&&repeats>=2&&repeats<=128&&(policy==0||policy==1),"argument range");
 std::setvbuf(stdout,nullptr,_IOLBF,0);ggml_cpu_init();
 kernels::cpu::ExpertPool pool(role==0?15:3,true);pool.set_native_tasks_per_thread(3);pool.set_nvfp4_route_tasks(false);
 // Declare links before lanes: lanes join before transport contexts are destroyed.
 std::unique_ptr<Link> first,second;
 std::unique_ptr<ServiceLane> lane1,lane2;
 kernels::cpu::ThreadAffinity affinity;
 if(role==0){
  // Create IO threads with the original allowed mask, before pinning compute.
  lane1=std::make_unique<ServiceLane>();lane2=std::make_unique<ServiceLane>();
  require(pool.numa_rows_available(),"primary requires two NUMA nodes");affinity=kernels::cpu::pin_current_thread(kernels::cpu::planned_host_core());require(affinity.valid,"primary pin failed");}
 artifact::Safetensors ts(argv[2]);const int base=role==0?0:role==1?3:6,loaded=role==0?8:role==1?3:2;
 std::array<std::vector<std::unique_ptr<Expert>>,2> weights;
 for(int l=0;l<2;++l)for(int e=0;e<loaded;++e){
  std::array<artifact::GlmFp4Matrix,3>w;int p=0;
  for(const char* name:{"gate_proj","up_proj","down_proj"})w[p++]=artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(l+3)+".mlp.experts."+std::to_string(base+e)+"."+name);
  auto expert=std::make_unique<Expert>(w,true,false);
  if(role==0){require(expert->prepare_numa(pool),"NUMA prepare failed");expert->release_original_blob();require(expert->retained_blob_bytes()==0,"original blob retained");}
  weights[l].push_back(std::move(expert));
 }
 std::array<std::vector<float>,3> output;std::array<std::vector<float>,2> pair_output;std::vector<uint8_t> act;
 auto run=[&](int layer,int count,const std::array<int,3>& ids,const std::vector<float>& input){
  for(int i=0;i<count;++i)require(ids[i]>=0&&ids[i]<loaded,"local expert ID out of range");
  if(count==3){std::array<const Expert*,3> selected;for(int i=0;i<3;++i)selected[i]=weights[layer][ids[i]].get();Expert::run_three_into(pool,selected,input,output,act,role==0);}
  else if(count==2){std::array<const Expert*,2> selected;for(int i=0;i<2;++i)selected[i]=weights[layer][ids[i]].get();Expert::run_batch_into(pool,selected,input,pair_output,act,role==0);for(int i=0;i<2;++i)output[i].swap(pair_output[i]);}
  else if(count==1)weights[layer][ids[0]]->run_into(pool,input,1,output[0],act,role==0);
 };
 if(role){
  Link link(argv[role+2],true,role==1?1:2,false);uint32_t seq=0;int enabled=0,batches=0,idle=0;
  std::vector<float> input(4096);
  for(;;){auto h=link.receive_command(++seq);if(!h[2]){link.acknowledge(seq);break;}
   require(h[4]<=uint32_t(loaded),"too many local experts");std::array<int,3> ids{};for(unsigned i=0;i<h[4];++i)ids[i]=int(h[6+i]);
   bool on=policy&&((((seq-1)/20)^((seq-1)/40))&1);pool.set_nvfp4_route_tasks(on);
   if(h[4]){link.receive(11,input.data(),input.size()*4);run(h[3],h[4],ids,input);if(on&&(h[4]==2||h[4]==3))++enabled;if(h[4]>=2)++batches;}else ++idle;
   link.acknowledge(seq);for(unsigned i=0;i<h[4];++i)link.send(20+i,output[i].data(),output[i].size()*4);
  }
  std::cout<<"SERVER PASS role="<<role<<" requests="<<seq-1<<" policy_on_calls="<<enabled<<" batch_calls="<<batches<<" idle_calls="<<idle<<" loaded_experts_per_layer="<<loaded<<"\n";return 0;
 }
 first=std::make_unique<Link>(argv[3],false,1,false);second=std::make_unique<Link>(argv[4],false,2,false);
 std::array<std::vector<float>,10> prior;std::vector<float> ref;std::vector<uint8_t> ref_act;std::ofstream dump(argv[7],std::ios::binary);require(bool(dump),"dump open failed");uint32_t seq=0;int checks=0;
 auto remote=[&](Link& link,uint32_t sequence,int layer,int count,const std::array<int,3>& ids,const std::vector<float>& input){
  auto h=Link::command(sequence,layer,count,ids);link.send(10,h.data(),sizeof(h));if(count)link.send(11,input.data(),input.size()*4);link.check_ack(sequence);
  std::vector<float> result(size_t(count)*4096);for(int i=0;i<count;++i)link.receive(20+i,result.data()+i*4096,4096*4);return result;
 };
 for(int repeat=0;repeat<repeats;++repeat)for(int scenario=0;scenario<10;++scenario){
  std::array<int,3> counts={3,3,2};if(scenario>=2&&scenario<=4)counts[scenario-2]=0;if(scenario==5)counts={0,0,0};if(scenario==8)counts={1,1,1};if(scenario==9)counts={2,2,2};
  std::array<std::array<int,3>,3> ids={{{0,1,2},{0,1,2},{0,1,0}}};if(scenario==1)ids={{{2,0,1},{1,2,0},{1,0,0}}};
  std::vector<float> input(4096);std::mt19937 rng(2026+scenario);for(auto& x:input)x=scenario==7?0.f:bf16(float(int(rng()%513)-256)/256.f*(scenario==6?.001f:1.f));
  for(int layer=0;layer<2;++layer){++seq;auto start=std::chrono::steady_clock::now();
   std::future<std::vector<float>> one,two;std::vector<float> middle,leaf;
   try {
    one=lane1->submit([&]{return remote(*first,seq,layer,counts[1],ids[1],input);});
    two=lane2->submit([&]{return remote(*second,seq,layer,counts[2],ids[2],input);});
    run(layer,counts[0],ids[0],input);middle=one.get();leaf=two.get();
   } catch(...) {
    // Tasks borrow this iteration's input and routing state. Finish both before
    // any referenced buffer, transport or model can leave scope.
    if(one.valid())one.wait();if(two.valid())two.wait();throw;
   }
   std::vector<float> mixed(4096),expected(4096),next(4096);
   for(int owner=0;owner<3;++owner)for(int i=0;i<counts[owner];++i){int global=(owner==0?0:owner==1?3:6)+ids[owner][i];float gain=float(global+1)/36;
    const float* y=owner==0?output[i].data():owner==1?middle.data()+i*4096:leaf.data()+i*4096;
    for(int j=0;j<4096;++j)mixed[j]+=gain*y[j];
   }
   for(int j=0;j<4096;++j)next[j]=bf16(std::clamp(input[j]+mixed[j]*.125f,-4.f,4.f));
   double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   for(int owner=0;owner<3;++owner)for(int i=0;i<counts[owner];++i){int global=(owner==0?0:owner==1?3:6)+ids[owner][i];weights[layer][global]->run_into(pool,input,1,ref,ref_act,true);float gain=float(global+1)/36;
    const float* y=owner==0?output[i].data():owner==1?middle.data()+i*4096:leaf.data()+i*4096;require(!std::memcmp(y,ref.data(),4096*4),"expert differs from oracle");
    for(int j=0;j<4096;++j)expected[j]+=gain*ref[j];
   }
   require(!std::memcmp(mixed.data(),expected.data(),4096*4),"reduction differs");for(int j=0;j<4096;++j)expected[j]=bf16(std::clamp(input[j]+expected[j]*.125f,-4.f,4.f));require(!std::memcmp(next.data(),expected.data(),4096*4),"residual differs");input.swap(next);++checks;
   std::cout<<"LAYER repeat="<<repeat<<" scenario="<<scenario<<" layer="<<layer+3<<" service_ms="<<ms<<"\n";
  }
  if(repeat)require(input==prior[scenario],"repeat mismatch");else prior[scenario]=input;dump.write(reinterpret_cast<const char*>(input.data()),input.size()*4);require(bool(dump),"dump write failed");
 }
 auto stop=[&](Link& link){auto h=Link::command(seq+1,0,0,{},true);link.send(10,h.data(),sizeof(h));link.check_ack(seq+1);return std::vector<float>{};};lane1->submit([&]{return stop(*first);}).get();lane2->submit([&]{return stop(*second);}).get();
 kernels::cpu::restore_thread_affinity(affinity);std::cout<<"PRIMARY PASS layers="<<checks<<" real_experts=8 assignment=3/3/2 native_q8=1 no_gpu=1\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
