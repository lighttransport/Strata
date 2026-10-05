#pragma once
#include "strata/artifact/glm_exl3.hpp"
#include <condition_variable>
#include <thread>
#include <functional>
#include <mutex>
#include <exception>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#endif
namespace strata::cpu::exl3 {
// Output partitions are whole Hadamard blocks. Each pinned worker allocates and
// first-touches its own packed rows; source buffers are released after packing.
inline bool huge_pages=false;
struct MemoryChunk {
    uint8_t* data=nullptr;size_t bytes=0;
    explicit MemoryChunk(size_t n):bytes(n) {
#ifdef __linux__
        data=(uint8_t*)mmap(nullptr,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(data==MAP_FAILED){data=nullptr;throw std::bad_alloc();}
        if(madvise(data,n,huge_pages?MADV_HUGEPAGE:MADV_NOHUGEPAGE)) {munmap(data,n);data=nullptr;throw std::runtime_error("EXL3: cannot set weight page advice");}
#else
        if(huge_pages)throw std::runtime_error("EXL3: huge page advice currently requires Linux");
        data=(uint8_t*)std::malloc(n);if(!data)throw std::bad_alloc();
#endif
    }
    ~MemoryChunk(){if(!data)return;
#ifdef __linux__
        munmap(data,bytes);
#else
        std::free(data);
#endif
    }
};
struct RowStorage {
    std::shared_ptr<MemoryChunk> owner;uint8_t* pointer=nullptr;size_t bytes=0;
    uint8_t* data(){return pointer;}const uint8_t* data()const{return pointer;}
    size_t size()const{return bytes;}bool empty()const{return !bytes;}
};
struct RowArena {
    std::vector<std::shared_ptr<MemoryChunk>> chunks;size_t used=0;
    RowStorage allocate(size_t bytes) {
        if(!bytes)return {};
        if(chunks.empty()||bytes>chunks.back()->bytes-used){chunks.push_back(std::make_shared<MemoryChunk>(std::max<size_t>(256*1024*1024,bytes)));used=0;}
        auto owner=chunks.back();RowStorage result{owner,owner->data+used,bytes};used+=bytes;return result;
    }
};
class Pool {
    std::vector<std::thread> workers;
    std::vector<RowArena> arenas;
    std::mutex mutex;std::condition_variable wake,done;
    std::function<void(size_t)> task;size_t epoch=0,pending=0;bool stop=false;std::exception_ptr error;
public:
    explicit Pool(size_t count,bool pin) {
        if(!count||count>256)throw std::runtime_error("EXL3: invalid worker count");
        arenas.resize(count);
        std::vector<int> cpus;
#ifdef __linux__
        cpu_set_t allowed;CPU_ZERO(&allowed);if(sched_getaffinity(0,sizeof(allowed),&allowed))throw std::runtime_error("EXL3: cannot read CPU affinity");
        std::set<std::string> seen;
        for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&allowed)) {
            std::ifstream f("/sys/devices/system/cpu/cpu"+std::to_string(c)+"/topology/thread_siblings_list");std::string key;std::getline(f,key);
            if(key.empty()||seen.insert(key).second)cpus.push_back(c);
        }
#endif
        std::cerr<<"EXL3 workers="<<count<<" placement="<<(pin?"pinned-first-touch":"first-touch")<<" weight_pages="<<(huge_pages?"THP-advice":"4k")<<" cpus=";
        if(pin&&!cpus.empty())for(size_t i=0;i<count;++i)std::cerr<<(i?",":"")<<cpus[(i+1)%cpus.size()];
        std::cerr<<'\n';
        for(size_t i=0;i<count;++i)workers.emplace_back([&,i,pin,cpus] {
#ifdef __linux__
            bool affinity_ok=true;
            if(pin&&!cpus.empty()) {cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpus[(i+1)%cpus.size()],&set);affinity_ok=pthread_setaffinity_np(pthread_self(),sizeof(set),&set)==0;}
#endif
            size_t seen_epoch=0;
            std::unique_lock lock(mutex);
            while(true) {
                wake.wait(lock,[&]{return stop||epoch!=seen_epoch;});if(stop)return;seen_epoch=epoch;auto job=task;lock.unlock();
                try{
#ifdef __linux__
                    if(!affinity_ok)throw std::runtime_error("EXL3: worker CPU pinning failed");
#endif
                    job(i);}catch(...){std::lock_guard guard(mutex);if(!error)error=std::current_exception();}
                lock.lock();if(!--pending)done.notify_one();
            }
        });
    }
    ~Pool() {{std::lock_guard lock(mutex);stop=true;}wake.notify_all();for(auto& w:workers)w.join();}
    Pool(const Pool&)=delete;
    size_t size()const{return workers.size();}
    void run(std::function<void(size_t)> job) {
        std::unique_lock lock(mutex);task=std::move(job);error=nullptr;pending=workers.size();++epoch;wake.notify_all();done.wait(lock,[&]{return !pending;});task={};if(error)std::rethrow_exception(error);
    }
    struct Rows {size_t first=0,last=0;RowStorage packed;};
    struct Matrix {size_t in=0,out=0;std::vector<uint16_t>suh,svh;std::vector<Rows> rows;};
    Matrix load(const artifact::GlmExl3& model,unsigned layer,unsigned expert,const std::string& projection) {
        auto source=model.load(layer,expert,projection);Matrix m;m.in=source.in;m.out=source.out;m.suh=std::move(source.suh);m.svh=std::move(source.svh);m.rows.resize(size());
        run([&](size_t rank) {
            auto& r=m.rows[rank];r.first=(m.out/128*rank/size())*128;r.last=(m.out/128*(rank+1)/size())*128;
            r.packed=arenas[rank].allocate((r.last-r.first)*m.in/4);
            for(size_t ot=r.first/16;ot<r.last/16;++ot)for(size_t it=0;it<m.in/16;++it)
                std::memcpy(r.packed.data()+((ot-r.first/16)*(m.in/16)+it)*64,source.trellis.data()+(it*(m.out/16)+ot)*64,64);
        });return m;
    }
    void apply(const Matrix& m,const float* x,float* y) {
        std::vector<float> h(m.in);for(size_t i=0;i<m.in;++i)h[i]=x[i]*half_value(m.suh[i]);hadamard(h.data(),m.in);
        run([&](size_t rank) {
            const auto& r=m.rows.at(rank);std::fill(y+r.first,y+r.last,0.f);
            for(size_t ot=r.first/16;ot<r.last/16;++ot)for(size_t it=0;it<m.in/16;++it) {
                tile_dot(r.packed.data()+((ot-r.first/16)*(m.in/16)+it)*64,h.data()+it*16,y+ot*16);
            }
            hadamard(y+r.first,r.last-r.first);for(size_t i=r.first;i<r.last;++i)y[i]*=half_value(m.svh[i]);
        });
    }
};
}
