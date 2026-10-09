// Expert-sized HIP H2D transfers, alone and alongside a DRAM reader.
// This is a bandwidth contention probe, not a quantized-expert compute benchmark.
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <vector>
static void check(hipError_t e) { if(e!=hipSuccess){std::cerr<<hipGetErrorString(e)<<'\n';std::exit(1);} }
int main() {
    constexpr size_t bytes=512ull<<20, slice=9ull<<20;
    void *host=nullptr,*device=nullptr;
    check(hipHostMalloc(&host,bytes)); check(hipMalloc(&device,slice));
    std::memset(host,53,bytes);
    std::vector<uint64_t> data(bytes/8,17);
    for(int workers:{0,8,12}) for(int trial=0;trial<3;++trial) {
        std::atomic<bool> stop{false},start{false};
        std::vector<std::thread> pool;
        std::vector<uint64_t> counts(workers),checksums(workers);
        for(int w=0;w<workers;++w) pool.emplace_back([&,w]{
            cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(w,&mask);
            if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask))std::abort();
            const size_t a=data.size()*w/workers,b=data.size()*(w+1)/workers;
            while(!start.load(std::memory_order_acquire)) std::this_thread::yield();
            uint64_t sum=0;
            while(!stop.load(std::memory_order_relaxed)) {
                for(size_t i=a;i<b;++i) sum+=data[i];
                counts[w]+=(b-a)*8;
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
            checksums[w]=sum;
        });
        auto begin=std::chrono::steady_clock::now();start.store(true,std::memory_order_release);
        uint64_t transferred=0;
        do {
            size_t offset=(transferred/slice % (bytes/slice))*slice;
            check(hipMemcpy(device,static_cast<char*>(host)+offset,slice,hipMemcpyHostToDevice));
            transferred+=slice;
        } while(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<2.0);
        stop.store(true);for(auto& t:pool)t.join();
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        uint64_t read=0,sum=0;for(int w=0;w<workers;++w){read+=counts[w];sum+=checksums[w];}
        std::vector<unsigned char> back(slice);check(hipMemcpy(back.data(),device,slice,hipMemcpyDeviceToHost));
        for(auto b:back)if(b!=53)std::abort();
        std::cout<<"{\"workers\":"<<workers<<",\"trial\":"<<trial<<",\"seconds\":"<<elapsed
            <<",\"h2d_GB_s\":"<<transferred/elapsed/1e9<<",\"cpu_read_GB_s\":"<<read/elapsed/1e9
            <<",\"checksum\":"<<sum<<",\"copy_exact\":true}\n"<<std::flush;
    }
    check(hipFree(device));check(hipHostFree(host));
}
