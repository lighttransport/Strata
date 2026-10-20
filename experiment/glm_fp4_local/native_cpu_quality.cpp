#include "experiment/glm_fp4_local/native_cpu_expert.hpp"
#include "strata/kernels/cpu/fp4_local.hpp"
#include "ggml-cpu.h"
#include <bit>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
using namespace strata;
static float bf16(float x){auto b=std::bit_cast<uint32_t>(x);b+=0x7fff+((b>>16)&1);return std::bit_cast<float>(b&0xffff0000u);}
int main(int argc,char**argv){try{
 if(argc<2||argc>4)throw std::runtime_error("usage: native-quality PARTIAL_DIR [quality-only] [preserve-small]");
 bool quality_only=false,preserve_small=false;
 for(int i=2;i<argc;++i){const std::string flag=argv[i];
  if(flag=="quality-only"&&!quality_only)quality_only=true;
  else if(flag=="preserve-small"&&!preserve_small)preserve_small=true;
  else throw std::runtime_error("invalid or repeated quality option");
 }
 ggml_cpu_init();artifact::Safetensors ts(argv[1]);kernels::cpu::ExpertPool pool(3,true);
 std::cout<<std::setprecision(10);
 for(int layer:{3,4}){
  std::array<artifact::GlmFp4Matrix,3>w;int p=0;
  for(const char* name:{"gate_proj","up_proj","down_proj"})w[p++]=artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+name);
  experiment::NativeFp4CpuExpert native(w,preserve_small), legacy(w);kernels::cpu::Fp4LocalCpuExpert reference(w,4);
  for(int nt:{1,2,4,8,16})for(int pattern=0;pattern<4;++pattern){
   if(pattern==3&&nt!=1)continue;
   const float amplitude=pattern==0?1.f:pattern==1?.001f:pattern==2?64.f:0.f;
   std::mt19937 rng(2026+layer);std::vector<float>input(size_t(nt)*4096);
   for(float&x:input)x=bf16(amplitude*float(int(rng()%513)-256)/256);
   // Full FP64 per-projection and fused/separate validation on one normal case.
   auto ref=reference.run_fused(input,nt,nt==1&&pattern==0,true);
   auto got=native.run(pool,input,nt);
   if(preserve_small&&pattern!=1){auto old=legacy.run(pool,input,nt);
    if(std::memcmp(got.data(),old.data(),got.size()*sizeof(float)))throw std::runtime_error("rescaling changed normal/large/zero output");
   }
   double err=0,norm=0,gnorm=0,dot=0,maxerr=0;
   for(size_t i=0;i<got.size();++i){if(!std::isfinite(got[i])||!std::isfinite(ref[i]))throw std::runtime_error("nonfinite quality output");double d=double(got[i])-ref[i];err+=d*d;norm+=double(ref[i])*ref[i];gnorm+=double(got[i])*got[i];dot+=double(got[i])*ref[i];maxerr=std::max(maxerr,std::abs(d));}
   std::cout<<"QUALITY layer="<<layer<<" tokens="<<nt<<" preserve_small="<<preserve_small<<" pattern="<<pattern<<" amplitude="<<amplitude<<" rel_l2="<<std::sqrt(err/std::max(norm,1e-30))<<" native_l2="<<std::sqrt(gnorm)<<" ref_l2="<<std::sqrt(norm)<<" max_abs="<<maxerr<<" cosine="<<(norm>0&&gnorm>0?dot/std::sqrt(norm*gnorm):norm==0&&gnorm==0?1:0)<<"\n";
   if(pattern!=0||quality_only)continue;
   for(int round=0;round<9;++round)for(int order=0;order<2;++order){
    bool candidate=bool(order^(round&1));auto run=[&]{return candidate?native.run(pool,input,nt):reference.run_fused(input,nt,false,true);};
    run();const auto start=std::chrono::steady_clock::now();auto out=run();const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();const auto&expected=candidate?got:ref;
    if(std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)))throw std::runtime_error("timed backend changed output");
    std::cout<<"TIMING layer="<<layer<<" tokens="<<nt<<" round="<<round<<" native="<<candidate<<" ms="<<ms<<" exact_repeat=1\n";
   }
  }
 }
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
