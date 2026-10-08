// Read-only Linux SSD probe: real expert spans, O_DIRECT, bounded worker buffers.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
struct Span { unsigned long long offset, bytes; };
using Job=std::array<Span,3>;
unsigned long long disk_bytes(){std::ifstream f("/proc/self/io");std::string k;unsigned long long v;while(f>>k>>v)if(k=="read_bytes:")return v;return 0;}
int main(int argc,char**argv) {
 try {
  if(argc!=4)throw std::runtime_error("usage: MODEL SPANS_TXT WORKERS");
  int workers=std::stoi(argv[3]);if(workers<1||workers>32)throw std::runtime_error("workers 1..32");
  std::ifstream input(argv[2]);if(!input)throw std::runtime_error("cannot open span list");
  std::vector<Job> jobs;Job job;size_t max_bytes=0;unsigned long long bytes=0;
  while(input>>job[0].offset>>job[0].bytes>>job[1].offset>>job[1].bytes>>job[2].offset>>job[2].bytes){
   for(auto&s:job){auto end=(s.offset+s.bytes+4095)/4096*4096;s.offset=s.offset/4096*4096;s.bytes=end-s.offset;
    if(!s.bytes||s.bytes>64*1024*1024)throw std::runtime_error("invalid expert span");
    max_bytes=std::max(max_bytes,(size_t)s.bytes);bytes+=s.bytes;}
   jobs.push_back(job);
  }
  if(!input.eof()||jobs.empty())throw std::runtime_error("invalid span list");
  int fd=open(argv[1],O_RDONLY|O_DIRECT|O_CLOEXEC);if(fd<0)throw std::runtime_error("O_DIRECT open failed");
  std::atomic<size_t> next{0};std::atomic<bool> failed{false};std::vector<double> latency(jobs.size());
  auto io_before=disk_bytes();std::vector<std::thread> threads;auto start=std::chrono::steady_clock::now();
  for(int w=0;w<workers;++w)threads.emplace_back([&]{
   void* buffer=nullptr;if(posix_memalign(&buffer,4096,max_bytes)){failed=true;return;}
   for(;;){auto i=next.fetch_add(1);if(i>=jobs.size()||failed)break;auto before=std::chrono::steady_clock::now();
    for(auto&s:jobs[i]){auto n=pread(fd,buffer,s.bytes,s.offset);if(n!=(ssize_t)s.bytes){failed=true;break;}}
    latency[i]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();}
   free(buffer);
  });
  for(auto&t:threads)t.join();close(fd);
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  if(failed)throw std::runtime_error("direct read failed or was short; no buffered fallback");
  std::sort(latency.begin(),latency.end());auto pct=[&](double p){return latency[(size_t)((latency.size()-1)*p)];};
  std::cout<<"{\"workers\":"<<workers<<",\"experts\":"<<jobs.size()<<",\"aligned_bytes\":"<<bytes
   <<",\"process_read_bytes\":"<<disk_bytes()-io_before<<",\"seconds\":"<<seconds<<",\"GiB_s\":"<<bytes/1073741824.0/seconds
   <<",\"expert_service_ms_p50\":"<<pct(.5)<<",\"expert_service_ms_p95\":"<<pct(.95)
   <<",\"expert_service_ms_p99\":"<<pct(.99)<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
