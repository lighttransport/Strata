#include "strata/ds41/expert_lane.hpp"
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
int main() {
 std::mt19937 rng(9384);
 for(int h: {17,256}) for(int threads: {1,3}) for(bool mixed: {false,true}) for(int nt: {1,2,6,16}) for(int top: {3,6,16}) {
  int n=nt*top;
  std::vector<int32_t> ids(n,-1), original(n,0), a,b;
  std::vector<float> va,vb,w(n), y(n*h), z(n*h), ref(n*h);
  for(int i=0;i<n;++i) {
   w[i]=float(int(rng()%100)-50)/73;
   if(mixed && i%3==0) { ids[i]=0; continue; }
   auto& entries=(i%2?a:b); auto& values=(i%2?va:vb);
   entries.push_back(i);
   for(int j=0;j<h;++j) values.push_back(float(int(rng()%10000)-5000)/113);
  }
  strata::ds41::ExpertLane lane(h,256,threads,false);
  strata::ds41::LaneLayer layer;
  std::vector<float> weights(3*h*256,0.001f);
  layer.f.gu_type=layer.f.d_type=layer.f.gu_act=layer.f.d_act=0;
  layer.f.n_embd=h; layer.f.n_ff=256;
  layer.f.gu_row=h*sizeof(float); layer.f.d_row=256*sizeof(float);
  layer.f.act_bytes=h*sizeof(float); layer.f.h_bytes=256*sizeof(float);
  layer.gate=reinterpret_cast<const uint8_t*>(weights.data());
  layer.up=reinterpret_cast<const uint8_t*>(weights.data()+h*256);
  layer.down=reinterpret_cast<const uint8_t*>(weights.data()+2*h*256);
  std::fill(z.begin(),z.end(),0.25f);
  unsetenv("STRATA_DS41_MERGE_DIRECT");
  lane.begin(layer,z.data(),nt,ids.data(),w.data(),top,ref.data());
  lane.finish_with_external2(original.data(),a.data(),va.data(),int(a.size()),b.data(),vb.data(),int(b.size()),top);
  setenv("STRATA_DS41_MERGE_DIRECT","1",1);
  setenv("STRATA_DS41_MERGE_VECTOR","1",1);
  lane.begin(layer,z.data(),nt,ids.data(),w.data(),top,y.data());
  lane.finish_with_external2(original.data(),a.data(),va.data(),int(a.size()),b.data(),vb.data(),int(b.size()),top);
  if(std::memcmp(ref.data(),y.data(),nt*h*sizeof(float))) return 1;
 }
 std::puts("PASS direct merge bitwise: local/external routes, sync/async, 96 geometries, vector tails");
}
