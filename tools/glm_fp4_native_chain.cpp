// CPU-only partial MoE chain. Explicit native Q8 arithmetic, not a full decoder.
#include "experiment/glm_fp4_local/native_cpu_expert.hpp"
#include "ggml-cpu.h"
#include "glm_fp4_batch_transport.hpp"
#include <bit>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
using namespace strata;
using Expert=experiment::NativeFp4CpuExpert;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static float bf16(float x){auto b=std::bit_cast<uint32_t>(x);b+=0x7fff+((b>>16)&1);return std::bit_cast<float>(b&0xffff0000u);}
int main(int argc,char**argv){try{
 require(argc>=3&&argc<=10,"usage: native-chain PARTIAL THREADS [batch-three] [batch-two] [numa-compact] [dump=PATH] [serve=ROOT|remote=ROOT] [ib-port=1|2] [udp]");
 size_t used=0;int threads=std::stoi(argv[2],&used);require(used==std::strlen(argv[2])&&threads>=2&&threads<=16,"threads must be 2..16");
 bool batch=false,batch_two=false,compact=false,server=false,udp=false,port_set=false;int port=1;std::string dump,endpoint;
 for(int i=3;i<argc;++i){std::string arg=argv[i];
  if(arg=="batch-three"&&!batch)batch=true;
  else if(arg=="batch-two"&&!batch_two)batch_two=true;
  else if(arg=="numa-compact"&&!compact)compact=true;
  else if(arg.starts_with("dump=")&&arg.size()>5&&dump.empty())dump=arg.substr(5);
  else if(arg.starts_with("serve=")&&arg.size()>6&&endpoint.empty()){endpoint=arg.substr(6);server=true;}
  else if(arg.starts_with("remote=")&&arg.size()>7&&endpoint.empty()){endpoint=arg.substr(7);}
  else if((arg=="ib-port=1"||arg=="ib-port=2")&&!port_set){port=arg.back()-'0';port_set=true;}
  else if(arg=="udp"&&!udp)udp=true;
  else throw std::invalid_argument("invalid or repeated option");
 }
 require(!endpoint.empty()||(!udp&&!port_set),"transport options require an endpoint");
 require(!server||dump.empty(),"server does not dump client activations");
 require(endpoint.empty()||server||!batch,"batch-three is selected by the server");
 require(endpoint.empty()||server||!batch_two,"batch-two is selected by the server");
 std::setvbuf(stdout,nullptr,_IOLBF,0);
 ggml_cpu_init();kernels::cpu::ExpertPool pool(threads-1,true);
 kernels::cpu::ThreadAffinity affinity;
 if(compact){require(threads==16&&pool.numa_rows_available(),"compact mode requires sixteen participants on two NUMA nodes");affinity=kernels::cpu::pin_current_thread(kernels::cpu::planned_host_core());require(affinity.valid,"host pin failed");}
 artifact::Safetensors ts(argv[1]);
 std::array<std::array<std::unique_ptr<Expert>,3>,2> weights;
 size_t retained=0;
 for(int layer=0;layer<2;++layer)for(int e=0;e<3;++e){
  std::array<artifact::GlmFp4Matrix,3>w;int p=0;
  for(const char* name:{"gate_proj","up_proj","down_proj"})w[p++]=artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer+3)+".mlp.experts."+std::to_string(e)+"."+name);
  auto& expert=weights[layer][e];expert=std::make_unique<Expert>(w,true,false);
  if(compact){require(expert->prepare_numa(pool),"NUMA preparation failed");expert->release_original_blob();require(expert->retained_blob_bytes()==0,"compact mode retained original weights");}
  retained+=expert->retained_blob_bytes();
 }
 std::unique_ptr<experiment::NativeBatchLink> link;
 if(!endpoint.empty())link=std::make_unique<experiment::NativeBatchLink>(endpoint,server,port,udp);
 std::array<std::vector<float>,2> two_outputs;
 int two_calls=0;
 auto run_two=[&](const std::array<const Expert*,3>& selected,const std::vector<float>& input,
                  std::array<std::vector<float>,3>& output,std::vector<uint8_t>& act){
  const std::array<const Expert*,2> pair={selected[0],selected[1]};
  Expert::run_batch_into(pool,pair,input,two_outputs,act,compact);
  for(int i=0;i<2;++i)output[i].swap(two_outputs[i]);
  ++two_calls;
 };
 uint32_t sequence=0;
 auto next_sequence=[&]{require(sequence!=UINT32_MAX,"batch sequence exhausted");return ++sequence;};
 if(server){
  std::vector<float> input(4096);std::array<std::vector<float>,3> output;std::vector<uint8_t> act;
  int batched=0,sequential=0,idle=0;
  for(;;){
   const auto h=link->receive_command(next_sequence());
   if(!h[2]){link->acknowledge(sequence);break;}
   const int count=int(h[4]);
   if(count){
    link->receive(11,input.data(),input.size()*sizeof(float));
    std::array<const Expert*,3> selected{};
    for(int i=0;i<count;++i)selected[i]=weights[h[3]][h[6+i]].get();
    if(batch&&count==3){Expert::run_three_into(pool,selected,input,output,act,compact);++batched;}
    else if(batch_two&&count==2){run_two(selected,input,output,act);++batched;}
    else{for(int i=0;i<count;++i)selected[i]->run_into(pool,input,1,output[i],act,compact);++sequential;}
   }else ++idle;
   link->acknowledge(sequence);
   for(int i=0;i<count;++i)link->send(20+i,output[i].data(),output[i].size()*sizeof(float));
  }
  if(batch_two)std::cout<<"TWO_EXPERT_BATCH calls="<<two_calls<<"\n";
  std::cout<<"SERVER PASS requests="<<sequence-1<<" batch_calls="<<batched<<" sequential_calls="<<sequential<<" idle_calls="<<idle<<" retained_original_bytes="<<retained<<" native_q8=1 no_gpu=1\n";
  if(affinity.valid)kernels::cpu::restore_thread_affinity(affinity);
  return 0;
 }
 std::ofstream raw;if(!dump.empty()){raw.open(dump,std::ios::binary);require(bool(raw),"cannot open dump");}
 int checks=0,batches=0,fallbacks=0,idles=0,remote_calls=0;
 std::array<std::vector<float>,8> prior;
 std::array<std::vector<float>,3> outputs,refs;
 std::vector<uint8_t> act,ref_act;
 for(int repeat=0;repeat<3;++repeat)for(int scenario=0;scenario<8;++scenario){
  std::array<int,3> ids={0,1,2};std::array<float,3> gains={1.f/6,2.f/6,3.f/6};
  int count=scenario==0?1:scenario==1?2:scenario==4?0:3;
  if(scenario==1){ids={2,0,1};gains={.25f,.75f,0};}
  if(scenario==3){ids={2,0,1};gains={.5f,.25f,.25f};}
  if(scenario==7)gains.fill(0);
  std::vector<float> input(4096);std::mt19937 rng(2026+scenario);
  for(float& x:input)x=scenario==5?0.f:bf16(float(int(rng()%513)-256)/256.f*(scenario==6?.001f:1.f));
  for(int layer=0;layer<2;++layer){
   std::array<const Expert*,3> selected;
   for(int i=0;i<3;++i)selected[i]=weights[layer][ids[i]].get();
   const auto start=std::chrono::steady_clock::now();
   if(link){
    const auto h=experiment::NativeBatchLink::command(next_sequence(),layer,count,ids);
    link->send(10,h.data(),sizeof(h));
    if(count)link->send(11,input.data(),input.size()*sizeof(float));
    link->check_ack(sequence);
    for(int i=0;i<count;++i){outputs[i].resize(4096);link->receive(20+i,outputs[i].data(),outputs[i].size()*sizeof(float));}
    ++remote_calls;
   }
   else if(batch&&count==3){Expert::run_three_into(pool,selected,input,outputs,act,compact);++batches;}
   else if(batch_two&&count==2){run_two(selected,input,outputs,act);++batches;}
   else if(count){for(int i=0;i<count;++i)selected[i]->run_into(pool,input,1,outputs[i],act,compact);++fallbacks;}
   else ++idles;
   std::vector<float> mixed(4096),expected(4096),next(4096);
   for(int i=0;i<count;++i)for(int j=0;j<4096;++j)mixed[j]+=gains[i]*outputs[i][j];
   for(int j=0;j<4096;++j)next[j]=bf16(std::clamp(input[j]+mixed[j]*.125f,-4.f,4.f));
   const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   for(int i=0;i<count;++i){selected[i]->run_into(pool,input,1,refs[i],ref_act,compact);for(int j=0;j<4096;++j)expected[j]+=gains[i]*refs[i][j];}
   require(!std::memcmp(mixed.data(),expected.data(),mixed.size()*sizeof(float)),"routed reduction differs from sequential reference");
   for(int j=0;j<4096;++j)expected[j]=bf16(std::clamp(input[j]+expected[j]*.125f,-4.f,4.f));
   require(!std::memcmp(next.data(),expected.data(),next.size()*sizeof(float)),"residual differs from sequential reference");
   input.swap(next);++checks;
   std::cout<<"LAYER repeat="<<repeat<<" scenario="<<scenario<<" layer="<<layer+3<<" experts="<<count<<" batch="<<((batch&&count==3)||(batch_two&&count==2))<<" service_ms="<<ms<<"\n";
  }
  if(repeat)require(input.size()==prior[scenario].size()&&!std::memcmp(input.data(),prior[scenario].data(),input.size()*sizeof(float)),"repeat changed output");else prior[scenario]=input;
  if(raw.is_open()){raw.write(reinterpret_cast<const char*>(input.data()),input.size()*sizeof(float));require(bool(raw),"output write failed");}
 }
 if(link){const auto h=experiment::NativeBatchLink::command(next_sequence(),0,0,{0,0,0},true);link->send(10,h.data(),sizeof(h));link->check_ack(sequence);std::cout<<"REMOTE calls="<<remote_calls<<"\n";}
 if(batch_two)std::cout<<"TWO_EXPERT_BATCH calls="<<two_calls<<"\n";
 std::cout<<"PASS layers="<<checks<<" batch_calls="<<batches<<" sequential_calls="<<fallbacks<<" idle_calls="<<idles<<" retained_original_bytes="<<retained<<" native_q8=1 no_gpu=1\n";
 if(affinity.valid)kernels::cpu::restore_thread_affinity(affinity);
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
