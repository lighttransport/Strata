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
        Cache failure(1024*1024);
        bool short_read=false;
        try { failure.acquire({{9,source(bytes.size())}}); } catch(const std::runtime_error&) {short_read=true;}
        assert(short_read && failure.resident_bytes()==0);
        auto recovered=failure.acquire({{9,source(0)}});
        assert(!memcmp(recovered[0]->data(0),bytes.data()+32,5000));
        unlink(name);
        std::cout<<"exact cache: bytes, hits, leases, eviction, duplicate IDs, short-read recovery passed\n";
    } catch(...) {unlink(name);throw;}
#endif
}
