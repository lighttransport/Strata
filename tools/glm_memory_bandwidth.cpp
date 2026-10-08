// Linux/OpenMP RAM microbenchmark. Rates count logical bytes, not memory-controller traffic.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>
using Clock=std::chrono::steady_clock;
volatile double consumed=0;
int main(int argc,char**argv) {
 if(argc!=3)return 2;
 int threads=std::stoi(argv[1]); bool huge=std::string(argv[2])=="thp";
 if(threads<1||threads>32)return 2;
 omp_set_dynamic(0);omp_set_num_threads(threads);
 constexpr size_t bytes=1ULL<<30,n=bytes/sizeof(double);
 double *a=nullptr,*b=nullptr,*c=nullptr;
 for(auto p:{&a,&b,&c}) {
  if(posix_memalign((void**)p,2<<20,bytes))return 3;
  if(madvise(*p,bytes,huge?MADV_HUGEPAGE:MADV_NOHUGEPAGE))return 4;
 }
 #pragma omp parallel for schedule(static)
 for(size_t i=0;i<n;++i){a[i]=1;b[i]=2;c[i]=0;}
 std::string line;size_t huge_kib=0;
 {std::ifstream f("/proc/self/smaps_rollup");while(std::getline(f,line))if(line.starts_with("AnonHugePages:"))huge_kib=std::stoull(line.substr(14));}
 for(int mode=0;mode<4;++mode) {
  std::vector<double> rates;
  for(int rep=0;rep<8;++rep) {
   double sum=0;auto start=Clock::now();
   #pragma omp parallel reduction(+:sum)
   {
    const size_t begin=n*omp_get_thread_num()/threads,end=n*(omp_get_thread_num()+1)/threads;
    if(mode==0) {
     __m256d s0=_mm256_setzero_pd(),s1=s0,s2=s0,s3=s0;
     size_t i=begin;
     for(;i+16<=end;i+=16){s0=_mm256_add_pd(s0,_mm256_loadu_pd(a+i));s1=_mm256_add_pd(s1,_mm256_loadu_pd(a+i+4));s2=_mm256_add_pd(s2,_mm256_loadu_pd(a+i+8));s3=_mm256_add_pd(s3,_mm256_loadu_pd(a+i+12));}
     alignas(32) double v[4];_mm256_store_pd(v,_mm256_add_pd(_mm256_add_pd(s0,s1),_mm256_add_pd(s2,s3)));
     sum=v[0]+v[1]+v[2]+v[3];for(;i<end;++i)sum+=a[i];
    } else if(mode==1)std::memcpy(c+begin,a+begin,(end-begin)*sizeof(double));
    else if(mode==2) {
     #pragma omp simd
     for(size_t i=begin;i<end;++i)c[i]=a[i]+3*b[i];
    } else {
     size_t i=begin;
     // Prefix aligns streaming stores even when thread ranges are not multiples of four.
     for(;i<end && i%4;++i)c[i]=a[i];
     for(;i+4<=end;i+=4)_mm256_stream_pd(c+i,_mm256_loadu_pd(a+i));
     for(;i<end;++i)c[i]=a[i];_mm_sfence();
    }
   }
   double seconds=std::chrono::duration<double>(Clock::now()-start).count();
   if(mode==0){consumed=sum;if(sum!=double(n))return 5;}
   else {for(size_t i=0;i<n;i+=4096)if(c[i]!=(mode==2?7.:1.))return 6;consumed=c[n-1];}
   if(rep)rates.push_back(bytes*(mode==0?1:mode==2?3:2)/seconds/1e9);
  }
  std::sort(rates.begin(),rates.end());
  std::cout<<"{\"threads\":"<<threads<<",\"pages\":\""<<(huge?"thp":"4k")<<"\",\"anon_huge_mib\":"<<huge_kib/1024.0
   <<",\"array_bytes\":"<<bytes<<",\"kernel\":\""<<std::vector<std::string>{"read_avx2","memcpy","triad","copy_nt"}[mode]
   <<"\",\"median_GB_s\":"<<rates[rates.size()/2]<<",\"best_GB_s\":"<<rates.back()<<",\"worst_GB_s\":"<<rates.front()<<"}\n";
 }
 for(auto p:{a,b,c})free(p);
}
