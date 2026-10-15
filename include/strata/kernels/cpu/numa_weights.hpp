#pragma once
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <array>
#include <vector>
#include "strata/kernels/cpu/q23_avx2.hpp"
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
    size_t mapped_=0,stride_=0;
    int packed_type_=0,columns_=0;
    bool view_=false;
public:
    size_t row_bytes=0,half_rows=0,experts=0,bytes=0,unmoved_pages=0;
    struct View {};
    // Shards inside the source mapping instead of copies: each half's whole pages move to its node and stay
    // bound there, so node-local reads need no anonymous memory. Bytes and row order equal the copies'.
    // lazy: leave the mapping untouched. Pages are read when a worker first uses them and land on that worker's
    // node, so a host with less memory than the tensor keeps only what the page cache retains.
    NumaTensor(View,const uint8_t* source,size_t row,size_t rows,size_t count,bool lazy=false)
        : stride_(row*rows),view_(true),row_bytes(row),half_rows(rows/2),experts(count) {
        if(!source || !row || !rows || rows%2 || !count || row>SIZE_MAX/rows || count>SIZE_MAX/(row*rows))
            throw std::invalid_argument("NUMA packed tensor view: invalid geometry");
        bytes=row*rows*count;
        data_[0]=const_cast<uint8_t*>(source);data_[1]=const_cast<uint8_t*>(source)+half_rows*row;
        if(!lazy)relocate_view_pages();
    }
    // A lazy view can be placed after streamed prefill has populated it,
    // without re-encoding, copying or checksumming the expert weights.
    void relocate_view_pages() {
        if(!view_)throw std::invalid_argument("NUMA relocation requires mapped tensor rows");
        unmoved_pages=0;
#ifdef __linux__
        // move_pages instead of mbind: binding each half would split the mapping into one VMA per half-expert,
        // more than the default vm.max_map_count on a full model.
        const uintptr_t page=sysconf(_SC_PAGESIZE);
        std::array<size_t,2> failed{};std::array<std::jthread,2> threads;
        std::array<std::vector<void*>,2> retry;
        for(int node=0;node<2;++node)threads[node]=std::jthread([&,node] {
            constexpr size_t batch=16384;
            std::vector<void*> pages;std::vector<int> nodes,status;pages.reserve(batch);
            auto flush=[&] {
                if(pages.empty())return;
                nodes.assign(pages.size(),node);status.assign(pages.size(),-1);
                if(syscall(SYS_move_pages,0,pages.size(),pages.data(),nodes.data(),status.data(),MPOL_MF_MOVE)<0)
                    retry[node].insert(retry[node].end(),pages.begin(),pages.end());
                else for(size_t i=0;i<pages.size();++i)if(status[i]!=node)retry[node].push_back(pages[i]);
                pages.clear();
            };
            for(size_t e=0;e<experts;++e) {
                const auto begin=reinterpret_cast<uintptr_t>(data_[node]+e*stride_),end=begin+half_rows*row_bytes;
#ifdef MADV_POPULATE_READ
                // move_pages moves only mapped pages: map this half first (a pack's checksum pass already did).
                if(end>(begin+page-1)/page*page)
                    madvise(reinterpret_cast<void*>((begin+page-1)/page*page),end-(begin+page-1)/page*page,MADV_POPULATE_READ);
#endif
                for(uintptr_t p=(begin+page-1)/page*page;p+page<=end;p+=page) {
                    pages.push_back(reinterpret_cast<void*>(p));
                    if(pages.size()==batch)flush();
                }
            }
            flush();
        });
        for(auto& t:threads)t.join();
        // A full node refuses pages until the other half's pages leave it: retry while a pass makes progress.
        for(int pass=0;pass<4 && !retry[0].empty()+!retry[1].empty();++pass) {
            const size_t before=retry[0].size()+retry[1].size();
            for(int node=0;node<2;++node) {
                std::vector<void*> left;
                for(size_t at=0;at<retry[node].size();at+=16384) {
                    const size_t n=std::min<size_t>(16384,retry[node].size()-at);
                    std::vector<int> nodes(n,node),status(n,-1);
                    if(syscall(SYS_move_pages,0,n,retry[node].data()+at,nodes.data(),status.data(),MPOL_MF_MOVE)<0)
                        left.insert(left.end(),retry[node].begin()+at,retry[node].begin()+at+n);
                    else for(size_t i=0;i<n;++i)if(status[i]!=node)left.push_back(retry[node][at+i]);
                }
                retry[node].swap(left);
            }
            if(retry[0].size()+retry[1].size()==before)break;
        }
        failed={retry[0].size(),retry[1].size()};
        unmoved_pages=failed[0]+failed[1];
#endif
    }
    NumaTensor(const uint8_t* source,size_t row,size_t rows,size_t count,std::array<int,2> cpus,bool huge=false,int packed_type=0,int columns=0)
        : packed_type_(packed_type),columns_(columns),row_bytes(row),half_rows(rows/2),experts(count) {
        if(!source || !row || !rows || rows%2 || !count || row>SIZE_MAX/rows || count>SIZE_MAX/(row*rows))
            throw std::invalid_argument("NUMA packed tensor: invalid geometry");
        bytes=row*rows*count;
#ifndef STRATA_NATIVE_EXPERTS
        if(packed_type_)throw std::invalid_argument("Q23 tiles require native expert support");
#endif
#ifdef __linux__
        const size_t page=huge?2*1024*1024:size_t(sysconf(_SC_PAGESIZE));mapped_=(bytes/2+page-1)/page*page;
        try {
            for(int node=0;node<2;++node) {
                if(cpus[node]<0)throw std::runtime_error("NUMA packed tensor: missing node worker");
                const size_t extra=huge?page:0;
                auto* p=mmap(nullptr,mapped_+extra,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
                if(p==MAP_FAILED)throw std::runtime_error("NUMA packed tensor: mmap failed");
                if(huge) {
                    auto raw=reinterpret_cast<uintptr_t>(p),aligned=(raw+page-1)/page*page;
                    const size_t prefix=aligned-raw,suffix=extra-prefix;
                    if(prefix)munmap(p,prefix);
                    if(suffix)munmap(reinterpret_cast<void*>(aligned+mapped_),suffix);
                    p=reinterpret_cast<void*>(aligned);
                }
                data_[node]=static_cast<uint8_t*>(p);unsigned long mask=1ul<<node;
                if(syscall(SYS_mbind,p,mapped_,MPOL_BIND,&mask,8*sizeof(mask),0))
                    throw std::runtime_error("NUMA packed tensor: mbind failed");
                madvise(p,mapped_,huge?MADV_HUGEPAGE:MADV_NOHUGEPAGE);
            }
            std::array<std::exception_ptr,2> errors;std::array<std::jthread,2> threads;
            // Each bound node copies and first-touches only its own arena.
            for(int node=0;node<2;++node)threads[node]=std::jthread([&,node] {
                try {
                    cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpus[node],&set);
                    if(pthread_setaffinity_np(pthread_self(),sizeof(set),&set))throw std::runtime_error("NUMA packed tensor: affinity failed");
                    for(size_t e=0;e<count;++e) {
                        auto* dest=data_[node]+e*half_rows*row;const auto* src=source+(e*rows+node*half_rows)*row;
#ifdef STRATA_NATIVE_EXPERTS
                        if(packed_type_)q23_pack(packed_type_,src,dest,columns_,half_rows);
                        else
#endif
                        std::memcpy(dest,src,half_rows*row);
                    }
                }catch(...){errors[node]=std::current_exception();}
            });
            for(auto& t:threads)t.join();
            for(auto& e:errors)if(e)std::rethrow_exception(e);
        }catch(...){release();throw;}
#else
        (void)cpus;throw std::runtime_error("NUMA packed tensors require Linux");
#endif
    }
    ~NumaTensor(){if(!view_)release();}
    NumaTensor(const NumaTensor&)=delete;NumaTensor& operator=(const NumaTensor&)=delete;
    const uint8_t* shard(size_t expert,int node)const {
        if(expert>=experts || node<0 || node>1)throw std::out_of_range("NUMA packed tensor shard");
        return data_[node]+expert*(view_?stride_:half_rows*row_bytes);
    }
    void copy(size_t expert,void* dest)const {
        for(int node=0;node<2;++node) {
            auto* out=static_cast<uint8_t*>(dest)+node*half_rows*row_bytes;
#ifdef STRATA_NATIVE_EXPERTS
            if(packed_type_)q23_unpack(packed_type_,shard(expert,node),out,columns_,half_rows);
            else
#endif
            std::memcpy(out,shard(expert,node),half_rows*row_bytes);
        }
    }
private:
    void release()noexcept {
#ifdef __linux__
        for(auto& p:data_)if(p){munmap(p,mapped_);p=nullptr;}
#endif
    }
};
}
