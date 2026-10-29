// Standalone CPU diagnostic; includes the implementation to time the actual opt-in helper.
#include "../../../src/ds41/expert_lane.cpp"
#include <cstdio>
#include <cstring>
__attribute__((noinline)) void baseline(float* y,const float* const* p,const float* w,int n,int h) {
 for(int k=0;k<n;++k) {
  if(!k) for(int j=0;j<h;++j) y[j]=p[k][j]*w[k];
  else for(int j=0;j<h;++j) y[j]+=p[k][j]*w[k];
 }
}
int main() {
 constexpr int h=5120,n=6;
 std::vector<float> data(h*n),a(h),b(h);const float* p[n];float w[n];
 for(int k=0;k<n;++k) {p[k]=data.data()+k*h;w[k]=float(k-2)/7;}
 for(size_t i=0;i<data.size();++i) data[i]=float(int(i*193%1001)-500)/137;
 baseline(a.data(),p,w,n,h);strata::ds41::collect_vectors(b.data(),p,w,n,h);
 if(std::memcmp(a.data(),b.data(),h*sizeof(float))) return 1;
 volatile float checksum=0;
 for(int mode=0;mode<2;++mode) {
  auto t=std::chrono::steady_clock::now();
  for(int r=0;r<2000;++r) {
   w[0]=float(r%7)/11;
   if(mode) strata::ds41::collect_vectors(b.data(),p,w,n,h);
   else baseline(b.data(),p,w,n,h);
   checksum=checksum+b[size_t(r)%h];
  }
  std::printf("%s %.3f ms\n",mode?"registers":"route-loop",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
 }
 std::printf("bitwise PASS checksum=%g\n",double(checksum));
}
