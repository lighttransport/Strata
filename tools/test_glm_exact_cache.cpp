// CPU-only Linux integration test; temporary fixture, never touches model files.
#include "strata/core/exact_expert_cache.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
using Cache=strata::core::ExactExpertCache;
int main() {
#ifdef __linux__
    char name[]="/tmp/strata-exact-test-XXXXXX";
    int fd=mkstemp(name); assert(fd>=0);
    std::vector<unsigned char> bytes(256*1024);
    for(size_t i=0;i<bytes.size();++i)bytes[i]=(i*37+i/4096)%251;
    assert(write(fd,bytes.data(),bytes.size())==(ssize_t)bytes.size()); close(fd);
    auto source=[&](size_t base) { Cache::Source s; for(int i=0;i<3;++i)s[i]={name,base+i*8192+32,5000}; return s; };
    try {
        Cache c(3*8192*2);
        auto first=c.acquire({{0,source(0)}});
        for(int i=0;i<3;++i)assert(!memcmp(first[0]->data(i),bytes.data()+i*8192+32,5000));
        auto hit=c.acquire({{0,source(0)}}); assert(c.stats.hits==1 && c.stats.misses==1);
        assert(first[0].get()==hit[0].get());
        auto second=c.acquire({{1,source(32768)}});
        bool pinned=false; try {c.acquire({{2,source(65536)}});}catch(const std::runtime_error&){pinned=true;}
        assert(pinned); // No dangling job pointer, even on capacity failure.
        assert(c.resident_bytes()==3*8192*2);
        // Clearing the lookup after an error must not hide still-live leases.
        bool retry_pinned=false; try {c.acquire({{3,source(98304)}});}catch(const std::runtime_error&){retry_pinned=true;}
        assert(retry_pinned && c.resident_bytes()==3*8192*2);
        assert(!memcmp(first[0]->data(0),bytes.data()+32,5000));
        first.clear();hit.clear();second.clear();
        auto a=c.acquire({{0,source(0)}});a.clear();
        auto b=c.acquire({{1,source(32768)}});b.clear();
        auto d=c.acquire({{2,source(65536)}});d.clear();
        assert(c.stats.evictions==1 && c.resident_bytes()<=3*8192*2);
        auto duplicate=c.acquire({{2,source(65536)},{2,source(65536)}});
        assert(duplicate[0].get()==duplicate[1].get()); duplicate.clear();
        Cache failure(1024*1024,4,Cache::Policy::lfu,8);
        bool short_read=false;
        try { failure.acquire({{9,source(bytes.size())}}); } catch(const std::runtime_error&) {short_read=true;}
        assert(short_read && failure.resident_bytes()==0);
        auto recovered=failure.acquire({{9,source(0)}});
        assert(!memcmp(recovered[0]->data(0),bytes.data()+32,5000));

        // Hits can be computed while misses are being filled; ready is only
        // published after wait(), including when the I/O already completed.
        Cache asynchronous(3*8192*3);
        auto warm=asynchronous.acquire({{0,source(0)}}); warm.clear();
        auto batch=asynchronous.begin({{0,source(0)},{1,source(32768)},{1,source(32768)}});
        assert(batch[0]->ready && !batch[1]->ready && batch[1]==batch[2]);
        assert(!memcmp(batch[0]->data(0),bytes.data()+32,5000));
        asynchronous.wait();
        assert(batch[1]->ready);
        for (int i=0;i<3;++i) assert(!memcmp(batch[1]->data(i),bytes.data()+32768+i*8192+32,5000));
        batch.clear();
        auto bad_batch=asynchronous.begin({{4,source(bytes.size())}});
        assert(!bad_batch[0]->ready);
        bool async_failure=false;
        try { asynchronous.wait(); } catch(const std::runtime_error&) {async_failure=true;}
        assert(async_failure && !bad_batch[0]->ready);
        bad_batch.clear();
        auto good_batch=asynchronous.acquire({{4,source(0)}});
        assert(good_batch[0]->ready);

        // LRU keeps the more recent entry; LFU keeps the repeatedly used one.
        for (auto policy:{Cache::Policy::lru,Cache::Policy::lfu}) {
            Cache replacement(3*8192*2,2,policy,8);
            for (int i=0;i<8;++i) replacement.acquire({{0,source(0)}});
            replacement.acquire({{1,source(32768)}});
            replacement.acquire({{2,source(65536)}});
            auto probe=replacement.begin({{0,source(0)}});
            assert(probe[0]->ready==(policy==Cache::Policy::lfu));
            replacement.wait(); probe.clear();
        }
        Cache seeded(3*8192*2,2,Cache::Policy::lfu,8);
        seeded.acquire({{0,source(0)}});
        seeded.seed_heat(0,16);
        seeded.acquire({{1,source(32768)}});
        seeded.acquire({{2,source(65536)}});
        auto kept=seeded.begin({{0,source(0)}}); assert(kept[0]->ready); seeded.wait(); kept.clear();
        // A prior is evictable: a held cold entry still wins lifetime protection.
        auto held=seeded.acquire({{1,source(32768)}});
        auto another=seeded.acquire({{3,source(98304)}});
        assert(!memcmp(held[0]->data(0),bytes.data()+32768+32,5000));
        held.clear(); another.clear();
        // Compare indexed replacement with a small independent serial model,
        // across decay boundaries, changing priors and temporarily held leases.
        for (auto policy:{Cache::Policy::lru,Cache::Policy::lfu}) {
            Cache indexed(3*8192*3,2,policy,4);
            struct Ref { double heat=0; uint64_t tick=0; bool resident=false; } ref[8];
            uint64_t accesses[2]={}, tick=0;
            int held_key=-1;
            std::vector<std::shared_ptr<Cache::Entry>> lease;
            for (int i=0;i<2000;++i) {
                if (i%23==11) { lease.clear(); held_key=-1; }
                if (i%97==0) {
                    const int k=(i/97)%8;
                    ref[k].heat=double(i%19);
                    indexed.seed_heat(k,ref[k].heat);
                }
                const int k=i%7 ? (i*17+i/13)%3 : 3+i%5;
                const bool expected=ref[k].resident;
                ref[k].heat+=1;
                if (++accesses[k/4]%240==0)
                    for (int j=k/4*4;j<k/4*4+4;++j) ref[j].heat*=0.97;
                ref[k].tick=++tick;
                if (!expected) {
                    int count=0; for (const auto& r:ref) count+=r.resident;
                    if (count==3) {
                        int victim=-1;
                        for (int j=0;j<8;++j) if (ref[j].resident && j!=held_key) {
                            if (victim<0 || (policy==Cache::Policy::lfu && ref[j].heat<ref[victim].heat) ||
                                ((policy==Cache::Policy::lru || ref[j].heat==ref[victim].heat) && ref[j].tick<ref[victim].tick)) victim=j;
                        }
                        assert(victim>=0); ref[victim].resident=false;
                    }
                    ref[k].resident=true;
                }
                auto probe=indexed.begin({{k,source(size_t(k)*32768)}});
                assert(probe[0]->ready==expected);
                indexed.wait();
                if (i%23==0) { lease=probe; held_key=k; }
            }
        }
        {
            Cache abandoned(3*8192*2);
            abandoned.begin({{0,source(0)},{1,source(32768)}});
        } // destruction drains readers even if the caller discarded all leases
        unlink(name);
        std::cout<<"exact cache: bytes, leases, async hits/misses, LRU/LFU, seeded heat and read-failure recovery passed\n";
    } catch(...) {unlink(name);throw;}
#endif
}
