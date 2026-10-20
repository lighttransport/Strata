// CPU-only paired GU screen. Link a symbol-renamed experimental kernel beside baseline.
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <vector>
#include <random>
#include <cstring>
#include <chrono>
#include <ctime>
#include <iostream>
#include <stdexcept>
namespace strata::kernels::cpu {
void candidate_nvfp4_256_gu_rows_clamped(const uint8_t*,size_t,size_t,int,const void* const*,int,float* const*,int,int,float,float,float,const uint8_t*);
}
using namespace strata::kernels::cpu;
int main(int argc,char**argv){try {
 if(argc!=5)throw std::runtime_error("usage: probe TOKENS COPIES [raw|normal]");
 const std::string scale_pattern=argv[3];
 if(scale_pattern!="raw"&&scale_pattern!="normal")throw std::runtime_error("invalid scale pattern");
 int nt=std::stoi(argv[1]),copies=std::stoi(argv[2]);
 if((nt!=1&&nt!=2&&nt!=4)||copies<1||copies>64)throw std::runtime_error("invalid geometry");
 constexpr int K=4096,R=128,B=K/64;
 const size_t row=B*sizeof(block_nvfp4),part=R*row;
 std::mt19937 rng(731);
 std::vector<block_nvfp4> w(size_t(copies)*2*R*B);
 for(auto& b:w){for(auto& s:b.d)s=scale_pattern=="normal"?8+rng()%119:rng()%256;for(auto& q:b.qs)q=rng();}
 // Prepared format: collapse duplicate codes, preserving 0x7f zero and 0xff finite.
 auto normalized = w;
 for (auto& b:normalized) for(auto& scale:b.d) scale=scale==127?0:(scale&127);
 std::vector<block_q8_0> a(nt*K/32);
 for(auto& b:a){b.d=uint16_t(0x1000+rng()%0x3000);for(auto& q:b.qs)q=int(rng()%255)-127;}
 const std::string act_pattern=argv[4];
 if(act_pattern!="safe"&&act_pattern!="min128")throw std::runtime_error("invalid activation pattern");
 if(act_pattern=="min128")a[0].qs[0]=-128;
 std::vector<float> y(nt*R),expected(size_t(copies)*nt*R);
 const void* ap[4];float* yp[4];
 for(int t=0;t<nt;++t){ap[t]=a.data()+t*K/32;yp[t]=y.data()+t*R;}
 auto run=[&](bool candidate,int bank){
  auto f=candidate?candidate_nvfp4_256_gu_rows_clamped:nvfp4_256_gu_rows_clamped;
  f(reinterpret_cast<uint8_t*>(((candidate && scale_pattern=="raw")?normalized.data():w.data()))+size_t(bank)*2*part,row,part,K,ap,nt,yp,0,R,.001f,.001f,7.f,nullptr);
 };
 for(int b=0;b<copies;++b){run(false,b);std::memcpy(expected.data()+size_t(b)*nt*R,y.data(),y.size()*4);run(true,b);
  if(std::memcmp(expected.data()+size_t(b)*nt*R,y.data(),y.size()*4))throw std::runtime_error("initial parity failure");}
 constexpr int calls=64;
 for(int round=0;round<12;++round)for(int order=0;order<2;++order){
  bool candidate=bool(order^(round&1));run(candidate,0);
  const auto wall=std::chrono::steady_clock::now();const auto cpu=std::clock();
  for(int i=0;i<calls;++i)run(candidate,i%copies);
  double cpu_ms=1000.*(std::clock()-cpu)/CLOCKS_PER_SEC;
  double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count();
  if(std::memcmp(expected.data()+size_t((calls-1)%copies)*nt*R,y.data(),y.size()*4))throw std::runtime_error("timed parity failure");
  std::cout<<"round="<<round<<" candidate="<<candidate<<" nt="<<nt<<" copies="<<copies<<" act_pattern="<<act_pattern<<" scale_pattern="<<scale_pattern<<" calls="<<calls<<" prepared_weight_copy_bytes="<<normalized.size()*sizeof(block_nvfp4)<<" bytes="<<w.size()*sizeof(block_nvfp4)<<" cpu_ms="<<cpu_ms<<" wall_ms="<<wall_ms<<" exact=1\n";
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
