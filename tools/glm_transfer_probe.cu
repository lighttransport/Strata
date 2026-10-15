// Read-only device/host bandwidth and kernel diagnostic; wall clock, not GPU events.
// nvcc -O3 tools/glm_transfer_probe.cu -o probe-cuda   (modes: h2d|d2h|kernel [seconds], bounce SIZE [iterations], register FD SIZE)
// hipcc -O3 -DSTRATA_PROBE_HIP tools/glm_transfer_probe.cu -o probe-hip
#ifdef STRATA_PROBE_HIP
#include <hip/hip_runtime.h>
#define API(x) hip##x
#else
#include <cuda_runtime.h>
#define API(x) cuda##x
#endif
#include <algorithm>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { auto err=(x); if(err!=API(Success)){std::fprintf(stderr,"%s: %s\n",#x,API(GetErrorString)(err));return 1;} } while(0)
__global__ void twice(const float* a,float* b,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)b[i]=2*a[i];}
int main(int argc,char**argv){
 std::string mode=argc>1?argv[1]:"h2d";double duration=argc>2?std::atof(argv[2]):.4;
 // bounce SIZE [iterations]: round-trip latency of one D2H copy followed by one H2D copy of SIZE bytes
 //   (the host bounce a residual or expert rows take between the two cards); wall clock p50/p90/p99.
 // register FD SIZE: mmap a shared memfd and pin it with HostRegister, then DMA into it and read it back;
 //   answers whether two processes can DMA into one shared mapping (exit 0) or not (exit 5).
 if(mode=="bounce"){
  size_t size=argc>2?std::strtoull(argv[2],nullptr,10):131072;int iterations=argc>3?std::atoi(argv[3]):2000;
  if(!size||size>(256ULL<<20)||iterations<10)return 2;
  float* host=(float*)std::aligned_alloc(4096,size);if(!host)return 3;std::memset(host,0,size);
  CK(API(SetDevice)(0));CK(API(HostRegister)(host,size,0));float* dev=nullptr;CK(API(Malloc)((void**)&dev,size));
  API(Stream_t) stream;CK(API(StreamCreate)(&stream));std::vector<double> us;us.reserve(iterations);
  for(int i=0;i<iterations+50;++i){
   auto start=std::chrono::steady_clock::now();
   CK(API(MemcpyAsync)(host,dev,size,API(MemcpyDeviceToHost),stream));CK(API(StreamSynchronize)(stream));
   CK(API(MemcpyAsync)(dev,host,size,API(MemcpyHostToDevice),stream));CK(API(StreamSynchronize)(stream));
   double t=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
   if(i>=50)us.push_back(t);
  }
  std::sort(us.begin(),us.end());auto q=[&](double f){return us[std::min(us.size()-1,size_t(f*us.size()))];};
  std::printf("{\"mode\":\"bounce\",\"bytes\":%zu,\"iterations\":%d,\"p50_us\":%.3f,\"p90_us\":%.3f,\"p99_us\":%.3f,\"min_us\":%.3f,\"GB_s_at_p50\":%.6f}\n",
   size,iterations,q(.5),q(.9),q(.99),us.front(),2.0*size/q(.5)/1e3);
  CK(API(StreamDestroy)(stream));CK(API(Free)(dev));CK(API(HostUnregister)(host));std::free(host);return 0;
 }
 if(mode=="register"){
  int fd=argc>2?std::atoi(argv[2]):-1;size_t size=argc>3?std::strtoull(argv[3],nullptr,10):(64ULL<<20);
  if(fd<0||!size)return 2;
  void* map=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);if(map==MAP_FAILED){std::perror("mmap");return 3;}
  CK(API(SetDevice)(0));
  auto err=API(HostRegister)(map,size,API(HostRegisterMapped));
  if(err!=API(Success)){std::printf("{\"mode\":\"register\",\"bytes\":%zu,\"registered\":false,\"error\":\"%s\"}\n",size,API(GetErrorString)(err));return 5;}
  float* dev=nullptr;CK(API(Malloc)((void**)&dev,size));std::vector<float> pattern(size/4);for(size_t i=0;i<pattern.size();++i)pattern[i]=float(i%977);
  CK(API(Memcpy)(dev,pattern.data(),size,API(MemcpyHostToDevice)));
  auto start=std::chrono::steady_clock::now();CK(API(Memcpy)(map,dev,size,API(MemcpyDeviceToHost)));
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
  bool ok=!std::memcmp(map,pattern.data(),size);
  std::printf("{\"mode\":\"register\",\"bytes\":%zu,\"registered\":true,\"d2h_into_shared_us\":%.3f,\"GB_s\":%.6f,\"readback_equal\":%s}\n",size,us,size/us/1e3,ok?"true":"false");
  CK(API(HostUnregister)(map));CK(API(Free)(dev));munmap(map,size);return ok?0:4;
 }
 if((mode!="h2d"&&mode!="d2h"&&mode!="kernel")||duration<=0)return 2;
 constexpr size_t bytes=64ULL<<20;float* host=(float*)std::aligned_alloc(4096,bytes);if(!host)return 3;
 std::fill(host,host+bytes/4,1.25f);CK(API(SetDevice)(0));CK(API(HostRegister)(host,bytes,0));
 float *a=nullptr,*b=nullptr;CK(API(Malloc)((void**)&a,bytes));CK(API(Malloc)((void**)&b,bytes));
 CK(API(Memcpy)(a,host,bytes,API(MemcpyHostToDevice)));
 twice<<<256,256>>>(a,b,bytes/4);CK(API(GetLastError)());CK(API(DeviceSynchronize)());
 float check=0;CK(API(Memcpy)(&check,b,sizeof(check),API(MemcpyDeviceToHost)));if(check!=2.5f)return 4;
 API(Stream_t) stream;CK(API(StreamCreate)(&stream));std::vector<double> rates;
 for(int trial=0;trial<6;++trial){auto start=std::chrono::steady_clock::now();size_t iterations=0;double seconds;
  do{
   if(mode=="kernel")twice<<<256,256,0,stream>>>(a,b,bytes/4);
   else CK(API(MemcpyAsync)(mode=="h2d"?(void*)a:(void*)host,mode=="h2d"?(void*)host:(void*)a,bytes,mode=="h2d"?API(MemcpyHostToDevice):API(MemcpyDeviceToHost),stream));
   CK(API(StreamSynchronize)(stream));++iterations;
   seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  }while(seconds<duration);
  double rate=bytes*(mode=="kernel"?2.:1.)*iterations/seconds/1e9;
  if(trial)rates.push_back(rate);
  std::printf("{\"mode\":\"%s\",\"trial\":%d,\"warmup\":%s,\"bytes_per_iteration\":%zu,\"iterations\":%zu,\"seconds\":%.9f,\"GB_s\":%.6f,\"kernel_check\":true}\n",mode.c_str(),trial,trial?"false":"true",bytes,iterations,seconds,rate);
 }
 CK(API(StreamDestroy)(stream));CK(API(Free)(b));CK(API(Free)(a));CK(API(HostUnregister)(host));std::free(host);return 0;
}
