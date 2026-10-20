// CPU-only NVFP4 row streaming, with pinned first touch and a same-buffer read control.
// Build with AVX2/FMA/F16C and link strata_kernels_cpu, strata_host, GGML CPU/base.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <ctime>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <fstream>
#include <sstream>
#include <map>
#include <cerrno>
#include <filesystem>
#if defined(__linux__)
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <unistd.h>
#endif
using namespace strata::kernels::cpu;
namespace {
constexpr int K=4096, N=256, REPS=9;
constexpr size_t BLOCKS=size_t(N)*K/64, BYTES=BLOCKS*sizeof(block_nvfp4);
uint64_t read_bytes(const uint8_t* p,size_t bytes) {
    __m256i a=_mm256_setzero_si256(),b=a,c=a,d=a;
    size_t i=0;
    for(;i+128<=bytes;i+=128) {
        a=_mm256_add_epi64(a,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i)));
        b=_mm256_add_epi64(b,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i+32)));
        c=_mm256_add_epi64(c,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i+64)));
        d=_mm256_add_epi64(d,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i+96)));
    }
    alignas(32) uint64_t v[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v),_mm256_add_epi64(_mm256_add_epi64(a,b),_mm256_add_epi64(c,d)));
    uint64_t sum=v[0]+v[1]+v[2]+v[3];
    for(;i<bytes;i+=8){uint64_t x;std::memcpy(&x,p+i,8);sum+=x;}
    return sum;
}
#if defined(__linux__)
int cpu_node(int cpu) {
    const auto path=std::filesystem::path("/sys/devices/system/cpu")/("cpu"+std::to_string(cpu));
    for(const auto& entry:std::filesystem::directory_iterator(path)) {
        const auto name=entry.path().filename().string();
        if(name.starts_with("node")&&name.size()>4)return std::stoi(name.substr(4));
    }
    return -1;
}
#endif
struct Worker {
    std::unique_ptr<block_nvfp4[]> weights;
    std::vector<block_q8_0> q;
    std::vector<float> out,reference;
    uint64_t checksum=0,hash=0;
};
}
int main(int argc,char**argv){try {
    if(argc<3||argc>5)throw std::runtime_error("usage: glm-fp4-stream-bench THREADS COPIES_PER_THREAD [physical|logical [default|huge|nohuge|huge-bind|nohuge-bind]]");
    const int threads=std::stoi(argv[1]),copies=std::stoi(argv[2]);
    auto cores=physical_cores(false);
    const std::string placement=argc>=4?argv[3]:"physical";
    const std::string pages=argc==5?argv[4]:"default";
    if(pages!="default"&&pages!="huge"&&pages!="nohuge"&&pages!="huge-bind"&&pages!="nohuge-bind")throw std::runtime_error("invalid page advice");
    if(placement!="physical"&&placement!="logical")throw std::runtime_error("invalid placement");
    if(placement=="logical") {
#if defined(__linux__)
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if(sched_getaffinity(0,sizeof(allowed),&allowed))throw std::runtime_error("cannot query allowed CPUs");
        for(int cpu=0;cpu<CPU_SETSIZE;++cpu)
            if(CPU_ISSET(cpu,&allowed)&&std::find(cores.begin(),cores.end(),cpu)==cores.end())cores.push_back(cpu);
#else
        throw std::runtime_error("logical placement currently requires Linux");
#endif
    }
    if(threads<1||threads>int(cores.size())||copies<1||copies>128)throw std::runtime_error("invalid threads or copies (1..128)");
    if(!cpu_avx2_ok())return 77;
    ggml_cpu_init();
    std::vector<Worker> data(threads);
    // Default-initialized POD: the pinned worker, not this allocating thread, writes pages first.
    for(auto& w:data){w.weights.reset(new block_nvfp4[BLOCKS*copies]);w.q.resize(K/32);w.out.resize(size_t(N)*copies);w.reference.resize(w.out.size());}
    size_t advised_bytes=0;
    if(pages!="default") {
#if defined(__linux__)
        const long page_size=sysconf(_SC_PAGESIZE);
        if(page_size<=0)throw std::runtime_error("cannot query page size");
        for(int id=0;id<threads;++id) {
            auto& w=data[id];
            const uintptr_t base=reinterpret_cast<uintptr_t>(w.weights.get());
            const uintptr_t begin=(base+page_size-1)/page_size*page_size;
            const uintptr_t end=(base+BYTES*copies)/page_size*page_size;
            if(end>begin) {
                if(pages.ends_with("-bind")) {
                    const int node=cpu_node(cores[id]);
                    if(node<0||node>=int(8*sizeof(unsigned long)))throw std::runtime_error("unsupported CPU NUMA node");
                    const unsigned long mask=1ul<<node;
                    if(syscall(SYS_mbind,reinterpret_cast<void*>(begin),end-begin,MPOL_BIND,&mask,8*sizeof(mask),0))
                        throw std::runtime_error("weight buffer NUMA bind failed");
                }
                if(madvise(reinterpret_cast<void*>(begin),end-begin,pages.starts_with("huge")?MADV_HUGEPAGE:MADV_NOHUGEPAGE))
                    throw std::runtime_error("weight buffer page advice failed");
                advised_bytes+=end-begin;
            }
        }
#else
        throw std::runtime_error("page advice requires Linux");
#endif
    }
    std::barrier sync(threads+1);
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for(int id=0;id<threads;++id)workers.emplace_back([&,id]{
        auto old=pin_current_thread(cores[id]);
        if(!old.valid)failed.store(true);
        auto& w=data[id];uint64_t rng=731+id;
        auto next=[&]{rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;};
        for(size_t i=0;i<BLOCKS*copies;++i){auto& b=w.weights[i];for(auto& s:b.d)s=32+next()%48;for(auto& q:b.qs)q=next()%256;}
        for(auto& b:w.q){b.d=0x2400;for(auto& q:b.qs)q=int(next()%255)-127;}
        const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_NVFP4)->vec_dot;
        for(size_t r=0;r<w.reference.size();++r)dot(K,&w.reference[r],0,w.weights.get()+r*(K/64),0,w.q.data(),0,1);
        const auto* bytes=reinterpret_cast<const uint8_t*>(w.weights.get());
        w.checksum=read_bytes(bytes,BYTES*copies);
        w.hash=14695981039346656037ULL;
        for(size_t i=0;i<w.reference.size()*sizeof(float);++i){w.hash^=reinterpret_cast<const uint8_t*>(w.reference.data())[i];w.hash*=1099511628211ULL;}
        sync.arrive_and_wait();
        for(int rep=0;rep<=REPS;++rep)for(int order=0;order<2;++order){
            const int mode=(rep+order)%2;
            sync.arrive_and_wait();
            uint64_t sum=0;
            if(mode==0)sum=read_bytes(bytes,BYTES*copies);
            else for(int c=0;c<copies;++c){const void* ap[]={w.q.data()};float* op[]={w.out.data()+size_t(c)*N};nvfp4_256_rows(bytes+size_t(c)*BYTES,K/64*sizeof(block_nvfp4),K,ap,1,op,0,N);}
            sync.arrive_and_wait();
            // Separate rendezvous keeps verification out of this and the next interval.
            if(mode==0 ? sum!=w.checksum : std::memcmp(w.out.data(),w.reference.data(),w.out.size()*sizeof(float))!=0)failed.store(true);
            sync.arrive_and_wait();
        }
        if(old.valid)restore_thread_affinity(old);
    });
    sync.arrive_and_wait();
    std::vector<double> ms[2], cpu_ms[2], active_cores[2];
    for(int rep=0;rep<=REPS;++rep)for(int order=0;order<2;++order){
        const int mode=(rep+order)%2;
        const auto cpu_start=std::clock();
        const auto start=std::chrono::steady_clock::now();sync.arrive_and_wait();sync.arrive_and_wait();
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const double cpu_elapsed=1000.0*(std::clock()-cpu_start)/CLOCKS_PER_SEC;
        if(rep){ms[mode].push_back(elapsed);cpu_ms[mode].push_back(cpu_elapsed);active_cores[mode].push_back(cpu_elapsed/elapsed);}
        sync.arrive_and_wait();
    }
    for(auto& t:workers)t.join();
    if(failed.load())throw std::runtime_error("affinity or exact parity failed");
    long huge_kib=-1;
#if defined(__linux__)
    std::ifstream smaps("/proc/self/smaps_rollup");std::string line;
    while(std::getline(smaps,line))if(line.starts_with("AnonHugePages:")) {
        std::istringstream fields(line);std::string label;fields>>label>>huge_kib;break;
    }
#endif
#if defined(__linux__)
    std::map<int,size_t> node_pages;
    std::ifstream numa_maps("/proc/self/numa_maps");
    while(std::getline(numa_maps,line)) {
        if(line.find(" anon=")==std::string::npos)continue;
        std::istringstream fields(line);std::string field;
        while(fields>>field)if(field.size()>2&&field[0]=='N'&&field.find('=')!=std::string::npos) {
            const auto eq=field.find('=');
            node_pages[std::stoi(field.substr(1,eq-1))]+=std::stoull(field.substr(eq+1));
        }
    }
    // Query only: nodes=nullptr and flags=0 never migrate pages. Outside timing.
    const long query_page=sysconf(_SC_PAGESIZE);
    if(query_page>0)for(int id=0;id<threads;++id) {
        const uintptr_t base=reinterpret_cast<uintptr_t>(data[id].weights.get());
        const uintptr_t begin=(base+query_page-1)/query_page*query_page;
        std::vector<void*> addresses;
        for(uintptr_t p=begin;p<base+BYTES*copies;p+=2*1024*1024)addresses.push_back(reinterpret_cast<void*>(p));
        std::vector<int> status(addresses.size(),-1);
        const int expected=cpu_node(cores[id]);
        const long result=syscall(SYS_move_pages,0,addresses.size(),addresses.data(),nullptr,status.data(),0);
        const int query_error=result<0?errno:0;
        size_t local=0,remote=0,unknown=0;
        for(int node:status) {
            if(query_error||node<0||expected<0)++unknown;
            else if(node==expected)++local;
            else ++remote;
        }
        std::cout<<"residency_worker="<<id<<" expected_node="<<expected<<" samples="<<addresses.size()
                 <<" local="<<local<<" remote="<<remote<<" unknown="<<unknown<<" query_errno="<<query_error<<"\n";
    }
    std::cout<<"process_anon_numa_pages=";
    for(const auto& [node,count]:node_pages)std::cout<<node<<":"<<count<<",";
    std::cout<<"\n";
#endif
    std::cout<<"page_advice="<<pages<<" advised_bytes="<<advised_bytes<<" process_anon_huge_kib="<<huge_kib<<"\n";
    uint64_t hash=0;for(const auto& w:data)hash^=w.hash;
    std::cout<<"placement="<<placement<<" cpus=";
    for(int id=0;id<threads;++id)std::cout<<cores[id]<<",";
    std::cout<<"\n";
    // Preserve pairing before sorting the legacy summaries. Read and dot use
    // the same bytes, but their ratio is a payload-rate comparison, not a
    // measurement of DRAM traffic or theoretical bandwidth utilization.
    for(int rep=0;rep<REPS;++rep) {
        std::cout<<"paired_rep="<<rep+1<<" first_mode="<<((rep+1)%2?"nvfp4":"read")
                 <<" read_ms="<<ms[0][rep]<<" nvfp4_ms="<<ms[1][rep]
                 <<" read_cpu_ms="<<cpu_ms[0][rep]<<" nvfp4_cpu_ms="<<cpu_ms[1][rep]
                 <<" dot_to_read_payload_ratio="<<ms[0][rep]/ms[1][rep]<<"\n";
    }
    for(int mode=0;mode<2;++mode){
        std::sort(ms[mode].begin(),ms[mode].end());
        std::sort(cpu_ms[mode].begin(),cpu_ms[mode].end());
        std::sort(active_cores[mode].begin(),active_cores[mode].end());
        std::cout<<"mode="<<(mode?"nvfp4":"read")<<" threads="<<threads<<" copies="<<copies<<" packed_bytes="<<BYTES*copies*threads<<" median_ms="<<ms[mode][REPS/2]<<" payload_GB_s="<<BYTES*copies*threads/ms[mode][REPS/2]/1e6<<" cpu_median_ms="<<cpu_ms[mode][REPS/2]<<" active_cores_median="<<active_cores[mode][REPS/2]<<" hash="<<hash<<" samples_ms=";
        for(double v:ms[mode])std::cout<<v<<",";std::cout<<"\n";
    }
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
