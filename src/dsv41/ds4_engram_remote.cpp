#include "ds4_engram_remote.hpp"
#include "ds4_engram_io.hpp"
#include "ucomm.h"
#include "engram_wire.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
namespace strata::ds4 {
struct EngramRemote::Impl {
    ucomm_t* c=nullptr;
    uint64_t size=0;
    bool healthy=true;
    uint64_t calls=0, rows=0, transferred=0;
    double wait_ms=0;
    std::vector<eg_request> requests;
    std::vector<uint8_t> bytes;
    ucomm_mr_t* region=nullptr;
    ~Impl() {
        if(!c) return;
        // A broken context must not enter a collective teardown or release in-flight buffers.
        if(!healthy) {std::fprintf(stderr,"fatal: remote Engram connection failed\n");std::_Exit(1);}
        std::fprintf(stderr,"remote Engram stats: calls=%llu rows=%llu bytes=%llu wait_ms=%.3f\n",(unsigned long long)calls,(unsigned long long)rows,(unsigned long long)transferred,wait_ms);
        if(region)ucomm_mr_dereg(region);
        if(ucomm_send(c,0,EG_REQUEST,nullptr,0)!=UCOMM_OK)std::_Exit(1);
        ucomm_finalize(c);
    }
    bool fetch() {
        if(!healthy)return false;
        size_t total=0;
        for(auto r:requests)total+=r.length;
        eg_status status{};size_t got=0;
        const auto start=std::chrono::steady_clock::now();
        healthy=ucomm_send(c,0,EG_REQUEST,requests.data(),requests.size()*sizeof(eg_request))==UCOMM_OK &&
          ucomm_recv(c,0,EG_STATUS,&status,sizeof status,&got)==UCOMM_OK && got==sizeof status && !status.code && status.bytes==total &&
          ucomm_recv(c,0,EG_DATA,bytes.data(),total,&got)==UCOMM_OK && got==total;
        wait_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(healthy){++calls;rows+=requests.size();transferred+=total;}
        return healthy;
    }
};
EngramRemote::EngramRemote():impl_(new Impl){}
EngramRemote::~EngramRemote()=default;
bool EngramRemote::connect(const std::string& address,uint64_t size,const std::string& sha,std::string& error) {
    auto& p=*impl_;ucomm_config cfg;ucomm_config_default(&cfg);
    cfg.rank=1;cfg.world=2;cfg.root_addr=address.c_str();
    cfg.backend=std::getenv("UCOMM_BACKEND") && std::string(std::getenv("UCOMM_BACKEND"))=="udp"?UCOMM_BACKEND_UDP:UCOMM_BACKEND_IB;
    if(cfg.timeout_ms<=0)cfg.timeout_ms=60000;
    auto s=ucomm_init(&cfg,&p.c);
    if(s!=UCOMM_OK){error=ucomm_strerror(s);return false;}
    eg_hello h{};size_t got=0;
    if(ucomm_recv(p.c,0,EG_HELLO,&h,sizeof h,&got)!=UCOMM_OK || got!=sizeof h){p.healthy=false;error="Engram handshake failed";return false;}
    char hex[65];for(int i=0;i<32;i++)std::snprintf(hex+i*2,3,"%02x",h.sha256[i]);
    if(h.magic!=EG_MAGIC || h.version!=1 || h.size!=size || sha!=hex){error="Engram source identity mismatch";return false;}
    p.size=size;p.bytes.resize(EG_MAX_BYTES);p.requests.reserve(EG_MAX_ENTRIES);
    if(ucomm_mr_reg(p.c,p.bytes.data(),p.bytes.size(),&p.region)!=UCOMM_OK)p.region=nullptr;
    std::fprintf(stderr,"remote Engram connected backend=%s size=%llu\n",ucomm_backend_name(p.c),(unsigned long long)size);
    return true;
}
bool EngramRemote::read(EgRead* rows,size_t count) {
    auto& p=*impl_;
    if(!count) {
        p.requests.assign(1,eg_request{0,4,0});
        return p.fetch();
    }
    for(size_t begin=0;begin<count;) {
        size_t end=begin,total=0;p.requests.clear();
        while(end<count && end-begin<EG_MAX_ENTRIES) {
            auto& r=rows[end];
            if(r.mx8 || r.n2 || !r.n || r.n>EG_MAX_ROW || r.off>p.size || r.n>p.size-r.off)return false;
            if(r.n>EG_MAX_BYTES-total)break;
            total+=r.n;p.requests.push_back({r.off,r.n,0});++end;
        }
        if(!p.fetch())return false;
        size_t at=0;
        for(size_t i=begin;i<end;++i){std::memcpy(rows[i].dst,p.bytes.data()+at,rows[i].n);at+=rows[i].n;}
        begin=end;
    }
    return true;
}
}
