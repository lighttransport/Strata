#include "native_hidden_scale.hpp"
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <immintrin.h>
using strata::experiment::SmallHiddenScale;
int main(){try{
 const auto mxcsr=_mm_getcsr();
 if(mxcsr&0x8040)throw std::runtime_error("subnormal test requires FTZ/DAZ off");
 SmallHiddenScale scale;std::mt19937 rng(124);int cases=0,fallbacks=0;
 for(int exponent=-149;exponent<=0;++exponent)for(int trial=0;trial<32;++trial){
  std::array<float,32>v{};v[0]=std::ldexp(1.f,exponent);
  for(int j=1;j<32;++j)v[j]=std::ldexp(float(int(rng()%2049)-1024)/1024,exponent);
  v[1]=-0.f;auto original=v,expected=v;float peak=0;for(float x:v)peak=std::max(peak,std::abs(x));
  const int shift=peak&&peak<0.0078125f?-7-std::ilogb(peak):0;
  for(float&x:expected)x=std::ldexp(x,shift);
  int token=trial%8;scale.apply(0,0,token,v.data(),v.size());
  if(scale.shift(token)!=shift||std::memcmp(v.data(),expected.data(),sizeof(v)))throw std::runtime_error("scale mismatch");
  scale.restore(token,v.data(),v.size());
  if(std::memcmp(v.data(),original.data(),sizeof(v)))throw std::runtime_error("restore mismatch");
  fallbacks+=shift>126;++cases;
 }
 for(float peak:{0.f,-0.f,std::nextafter(0.0078125f,0.f),0.0078125f,std::nextafter(0.0078125f,1.f),1.f}){
  float v=peak;scale.apply(0,0,0,&v,1);scale.restore(0,&v,1);if(std::memcmp(&v,&peak,4))throw std::runtime_error("threshold mismatch");++cases;
 }
 float v=1.e-6f;scale.apply(0,0,7,&v,1);scale.reset();for(int t=0;t<8;++t)if(scale.shift(t))throw std::runtime_error("reset mismatch");
 for(float invalid:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}){
  bool rejected=false;try{scale.apply(0,0,0,&invalid,1);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)throw std::runtime_error("nonfinite accepted");++cases;
 }
 std::cout<<"PASS cases="<<cases<<" fallback_cases="<<fallbacks<<" bit_exact_roundtrip=1 mxcsr="<<mxcsr<<"\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
