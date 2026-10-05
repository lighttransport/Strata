#pragma once
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <array>
#ifdef __linux__
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#endif
namespace strata::kernels::cpu {
// Original packed rows, split evenly by output row. No numeric conversion.
class NumaTensor {
    std::array<uint8_t*,2> data_{};
    size_t mapped_=0;
public:
    size_t row_bytes=0,half_rows=0,experts=0,bytes=0;
    NumaTensor(const uint8_t* source,size_t row,size_t rows,size_t count,std::array<int,2> cpus)
        : row_bytes(row),half_rows(rows/2),experts(count) {
        if(!source || !row || !rows || rows%2 || !count || row>SIZE_MAX/rows || count>SIZE_MAX/(row*rows))
            throw std::invalid_argument("NUMA packed tensor: invalid geometry");
        bytes=row*rows*count;
#ifdef __linux__
        const size_t page=sysconf(_SC_PAGESIZE);mapped_=(bytes/2+page-1)/page*page;
        try {
            for(int node=0;node<2;++node) {
                if(cpus[node]<0)throw std::runtime_error("NUMA packed tensor: missing node worker");
                auto* p=mmap(nullptr,mapped_,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
                if(p==MAP_FAILED)throw std::runtime_error("NUMA packed tensor: mmap failed");
                data_[node]=static_cast<uint8_t*>(p);unsigned long mask=1ul<<node;
                if(syscall(SYS_mbind,p,mapped_,MPOL_BIND,&mask,8*sizeof(mask),0))
                    throw std::runtime_error("NUMA packed tensor: mbind failed");
                madvise(p,mapped_,MADV_NOHUGEPAGE);
            }
            std::array<std::exception_ptr,2> errors;std::array<std::jthread,2> threads;
            // Each bound node copies and first-touches only its own arena.
            for(int node=0;node<2;++node)threads[node]=std::jthread([&,node] {
                try {
                    cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpus[node],&set);
                    if(pthread_setaffinity_np(pthread_self(),sizeof(set),&set))throw std::runtime_error("NUMA packed tensor: affinity failed");
                    for(size_t e=0;e<count;++e)std::memcpy(data_[node]+e*half_rows*row,
                        source+(e*rows+node*half_rows)*row,half_rows*row);
                }catch(...){errors[node]=std::current_exception();}
            });
            for(auto& t:threads)t.join();
            for(auto& e:errors)if(e)std::rethrow_exception(e);
        }catch(...){release();throw;}
#else
        (void)cpus;throw std::runtime_error("NUMA packed tensors require Linux");
#endif
    }
    ~NumaTensor(){release();}
    NumaTensor(const NumaTensor&)=delete;NumaTensor& operator=(const NumaTensor&)=delete;
    const uint8_t* shard(size_t expert,int node)const {
        if(expert>=experts || node<0 || node>1)throw std::out_of_range("NUMA packed tensor shard");
        return data_[node]+expert*half_rows*row_bytes;
    }
    void copy(size_t expert,void* dest)const {
        for(int node=0;node<2;++node)std::memcpy(static_cast<uint8_t*>(dest)+node*half_rows*row_bytes,
            shard(expert,node),half_rows*row_bytes);
    }
private:
    void release()noexcept {
#ifdef __linux__
        for(auto& p:data_)if(p){munmap(p,mapped_);p=nullptr;}
#endif
    }
};
}
