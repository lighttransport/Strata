#include "strata/kernels/cpu/glm_expert_cache.hpp"
#include <array>
#include <barrier>
#include <mutex>
#include <stdexcept>
#include <thread>

using Key=std::pair<int,int>;
struct Expert {size_t bytes=16;std::list<Key>::iterator order;};
using Ptr=std::shared_ptr<Expert>;
static void require(bool condition) {
    if(!condition)throw std::runtime_error("CPU expert cache invariant failed");
}
int main() {
    for(int repeat=0;repeat<64;++repeat) {
        std::map<Key,Ptr> cache;std::list<Key> lru;size_t bytes=0;
        std::mutex mutex;std::barrier loaded(2);
        std::array<Ptr,2> result;
        std::array<std::weak_ptr<Expert>,2> candidates;
        auto miss=[&](int rank) {
            auto candidate=std::make_shared<Expert>();candidates[rank]=candidate;
            // Both jobs have finished loading the same cold expert before
            // either may publish it, as in a wide decode's split CPU jobs.
            loaded.arrive_and_wait();
            std::lock_guard lock(mutex);
            result[rank]=strata::kernels::cpu::glm_publish_cpu_expert(cache,lru,bytes,64,{3,7},std::move(candidate));
        };
        std::thread first(miss,0),second(miss,1);first.join();second.join();
        require(cache.size()==1&&lru.size()==1&&bytes==16);
        require(result[0]==result[1]&&result[0]==cache.at({3,7}));
        require(candidates[0].expired()!=candidates[1].expired());
        auto publish=[&](int id) {
            std::lock_guard lock(mutex);
            return strata::kernels::cpu::glm_publish_cpu_expert(cache,lru,bytes,64,{3,id},std::make_shared<Expert>());
        };
        // Fill, touch the winning entry, then evict repeatedly. The losing
        // publication must not leave a stale key or evict unrelated entries.
        publish(8);publish(9);publish(10);
        require(publish(7)==result[0]&&bytes==64&&cache.size()==4&&lru.size()==4);
        publish(11);require(!cache.count({3,8})&&cache.count({3,7}));
        for(int id=12;id<32;++id) {
            publish(id);size_t actual=0;
            require(cache.size()==lru.size()&&cache.size()==4);
            for(auto it=lru.begin();it!=lru.end();++it) {
                require(cache.count(*it)==1&&cache.at(*it)->order==it);
                actual+=cache.at(*it)->bytes;
            }
            require(bytes==actual&&bytes==64);
        }
        // A caller may retain weights after their cache entry is evicted.
        require(result[0]->bytes==16);
    }
}
