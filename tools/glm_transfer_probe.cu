// Read-only device/host bandwidth and kernel diagnostic; wall clock, not GPU events.
// nvcc -O3 tools/glm_transfer_probe.cu -o probe-cuda
// hipcc -O3 -DSTRATA_PROBE_HIP tools/glm_transfer_probe.cu -o probe-hip
#ifdef STRATA_PROBE_HIP
#include <hip/hip_runtime.h>
#define API(x) hip##x
#else
#include <cuda_runtime.h>
#define API(x) cuda##x
#endif
#include <algorithm>
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
