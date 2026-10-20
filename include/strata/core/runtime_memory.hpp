#pragma once
#include "strata/core/expert_source.hpp"
#include "strata/host/glm_env.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core {
inline uint64_t glm_fp4_cpu_scale_cache_bytes() {
    const auto mib=strata_env_value_ll("STRATA_GLM_FP4_CPU_SCALE_CACHE_MIB",2048);
    if(mib<128 || mib>10240)throw std::invalid_argument("GLM FP4: CPU scale cache must be 128..10240 MiB");
    return uint64_t(mib)*1024*1024;
}
inline uint64_t glm_fp4_host_reserve_bytes() {
    return std::max<uint64_t>(6ULL*1024*1024*1024,glm_fp4_cpu_scale_cache_bytes()+4ULL*1024*1024*1024);
}
// The Slurm request remains an upper bound when the job's memory controller
// is nested or not mounted in this process's namespace.
inline uint64_t host_memory_allowance_bytes() {
    uint64_t limit=UINT64_MAX;
#ifdef __linux__
    limit=uint64_t(sysconf(_SC_PHYS_PAGES))*uint64_t(sysconf(_SC_PAGESIZE));
    detail::HostMemory memory;
    detail::host_available_memory(memory);
    limit=std::min(limit,memory.cgroup_limit);
    if(const char* text=std::getenv("SLURM_MEM_PER_NODE"))try {
        size_t used=0;const uint64_t mib=std::stoull(text,&used);
        if(text[used]=='\0'&&mib&&mib<=UINT64_MAX/(1024*1024))limit=std::min(limit,mib*1024*1024);
    }catch(const std::exception&) {}
#endif
    return limit;
}
// The page cache can cause Linux to swap idle worker stacks even when model
// weights fit. Opt-in locking is bounded and excludes model/large arenas.
inline void lock_small_runtime_mappings() {
    constexpr size_t MiB=1024*1024;
    const char* enabled=std::getenv("STRATA_GLM_LOCK_RUNTIME");
    if(!enabled||std::string(enabled)=="0")return;
    if(std::string(enabled)!="1")throw std::invalid_argument("STRATA_GLM_LOCK_RUNTIME must be 0 or 1");
#ifdef __linux__
    struct Region {uintptr_t begin=0,end=0;std::string permissions;size_t rss=0;};
    std::ifstream maps("/proc/self/smaps");std::string line;Region current;
    std::vector<Region> candidates;size_t bytes=0;
    while(std::getline(maps,line)) {
        unsigned long begin,end;char permissions[5];
        if(std::sscanf(line.c_str(),"%lx-%lx %4s",&begin,&end,permissions)==3)current={begin,end,permissions,0};
        else if(line.starts_with("Rss:"))current.rss=std::stoull(line.substr(4))*1024;
        else if(line.starts_with("VmFlags:")&&current.permissions=="rw-p"&&current.rss&&current.end-current.begin<=256*MiB
                &&line.find(" lo ")==std::string::npos&&line.find(" io ")==std::string::npos&&line.find(" pf ")==std::string::npos) {
            candidates.push_back(current);bytes+=current.end-current.begin;
        }
    }
    if(bytes>2*1024*MiB)throw std::runtime_error("runtime locking exceeds the 2 GiB per-pass bound");
    for(const auto& region:candidates)if(mlock(reinterpret_cast<void*>(region.begin),region.end-region.begin))
        throw std::runtime_error("cannot lock runtime mappings within the existing memlock allowance");
    if(bytes)std::cerr<<"RUNTIME_LOCK added_MiB="<<bytes/double(MiB)<<'\n';
#else
    throw std::runtime_error("runtime mapping lock requires Linux");
#endif
}

}
