#pragma once
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <utility>

namespace strata::kernels::cpu {
// Caller holds the cache mutex. Loading happens outside that mutex, so another
// job may already have published this key. Reuse its entry before any eviction
// or accounting change. Expert owns bytes and its iterator into this LRU.
template<class Expert>
std::shared_ptr<Expert> glm_publish_cpu_expert(
    std::map<std::pair<int,int>,std::shared_ptr<Expert>>& cache,
    std::list<std::pair<int,int>>& lru,size_t& bytes,size_t limit,
    const std::pair<int,int>& key,std::shared_ptr<Expert> candidate) {
    if(auto found=cache.find(key);found!=cache.end()) {
        lru.splice(lru.begin(),lru,found->second->order);
        return found->second;
    }
    while(bytes+candidate->bytes>limit&&!cache.empty()) {
        auto victim=cache.find(lru.back());
        bytes-=victim->second->bytes;cache.erase(victim);lru.pop_back();
    }
    lru.push_front(key);
    try {cache.emplace(key,candidate);}
    catch(...) {lru.pop_front();throw;}
    candidate->order=lru.begin();bytes+=candidate->bytes;
    return candidate;
}
}
