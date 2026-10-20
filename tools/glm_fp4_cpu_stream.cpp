// Linux AVX-512 read probe. Run under the same NUMA policy as the model.
// Build: g++ -O3 -std=c++20 -pthread tools/glm_fp4_cpu_stream.cpp -o cpu-stream
#include <immintrin.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
struct alignas(64) Result {uint64_t lanes[8]{};};
__attribute__((target("avx512f")))
static void read_stream(const unsigned char* data,size_t bytes,int passes,Result& result) {
    auto sum=_mm512_setzero_si512();
    for(int pass=0;pass<passes;++pass) {
        asm volatile("" ::: "memory");
        for(size_t i=0;i<bytes;i+=64)sum=_mm512_add_epi64(sum,_mm512_load_si512(data+i));
    }
    _mm512_store_si512(result.lanes,sum);
}
int main(int argc,char** argv) {try {
    if(!__builtin_cpu_supports("avx512f"))return 77;
    const int workers=argc>1?std::stoi(argv[1]):32;
    constexpr size_t bytes=size_t(4)*1024*1024*1024;
    constexpr int passes=15,samples=7,warmups=2;
    cpu_set_t allowed;CPU_ZERO(&allowed);
    if(sched_getaffinity(0,sizeof(allowed),&allowed))throw std::runtime_error("cannot read CPU affinity");
    std::vector<int> cpus;
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&allowed))cpus.push_back(cpu);
    if(workers<1||size_t(workers)>=cpus.size()||bytes%(size_t(workers)*64))
        throw std::invalid_argument("workers need equal 64-byte slices and one free coordinator CPU");
    cpus.erase(cpus.begin());cpus.resize(workers);
    std::unique_ptr<unsigned char,decltype(&std::free)> data(static_cast<unsigned char*>(std::aligned_alloc(64,bytes)),&std::free);
    if(!data)throw std::bad_alloc();
    std::memset(data.get(),0x5a,bytes); // Fault in every page before timing.
    std::barrier sync(workers+1);std::atomic<bool> pinned{true};
    std::vector<Result> results(workers);std::vector<std::thread> threads;
    for(int rank=0;rank<workers;++rank)threads.emplace_back([&,rank] {
        cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpus[rank],&set);
        if(sched_setaffinity(0,sizeof(set),&set))pinned.store(false);
        for(int sample=0;sample<samples;++sample) {
            sync.arrive_and_wait();
            read_stream(data.get()+bytes/workers*rank,bytes/workers,passes,results[rank]);
            sync.arrive_and_wait();
        }
    });
    std::vector<double> rates;
    for(int sample=0;sample<samples;++sample) {
        const auto begin=std::chrono::steady_clock::now();sync.arrive_and_wait();sync.arrive_and_wait();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        const uint64_t expected=uint64_t(0x5a5a5a5a5a5a5a5aULL)*(bytes/workers/64)*passes;
        for(const auto& result:results)for(auto lane:result.lanes)
            if(lane!=expected)pinned.store(false);
        const double rate=double(bytes)*passes/seconds/1e9;
        if(sample>=warmups)rates.push_back(rate);
        std::cout<<"{\"sample\":"<<sample<<",\"warmup\":"<<(sample<warmups?"true":"false")
                 <<",\"seconds\":"<<seconds<<",\"read_GB_s\":"<<rate<<"}\n";
    }
    for(auto& thread:threads)thread.join();
    if(!pinned.load())throw std::runtime_error("CPU affinity or read checksum failed");
    std::sort(rates.begin(),rates.end());
    std::cout<<"{\"workers\":"<<workers<<",\"buffer_bytes\":"<<bytes<<",\"passes_per_sample\":"<<passes
             <<",\"median_read_GB_s\":"<<rates[rates.size()/2]<<",\"cpus\":[";
    for(size_t i=0;i<cpus.size();++i)std::cout<<(i?",":"")<<cpus[i];
    std::cout<<"]}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
