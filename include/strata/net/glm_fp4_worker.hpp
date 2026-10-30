#pragma once
#include "strata/host/glm_fp4_worker.h"
#include "ucomm.h"
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::net {
// ucomm retains requests on timeout; their DMA buffers must remain alive.
// This dedicated process cannot recover that context. Exit before unwinding
// any activation/weight buffers; the launcher owns staging and peer cleanup.
[[noreturn]] inline void glm_fp4_transport_failure(ucomm_status status) {
    std::fprintf(stderr,"FP4 transport: %s; terminating with in-flight buffers intact\n",ucomm_strerror(status));
    std::fflush(stderr);std::_Exit(1);
}

// An idle service may outlive one operation timeout. Keep the same posted
// receive alive across timeouts; ucomm_wait retains it until completion.
inline void glm_fp4_receive_idle(ucomm_t* c,int peer,unsigned tag,void* data,size_t bytes,
                                 unsigned idle_seconds=3600) {
    ucomm_req_t* request=nullptr;
    auto status=ucomm_irecv(c,peer,tag,data,bytes,&request);
    if(status!=UCOMM_OK)glm_fp4_transport_failure(status);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(idle_seconds);
    size_t got=0;
    do {status=ucomm_wait(request,&got);}
    while(status==UCOMM_ERR_TIMEOUT&&std::chrono::steady_clock::now()<deadline);
    if(status!=UCOMM_OK)glm_fp4_transport_failure(status);
    if(got!=bytes)throw std::runtime_error("FP4 worker truncated idle request");
}

// One serialized owner per context. Compute and fetch use separate instances.
class GlmFp4WorkerConnection {
    ucomm_t* comm_=nullptr;
    uint32_t sequence_=1;
    bool healthy_=false;
    void checked(ucomm_status status) {
        if(status!=UCOMM_OK)glm_fp4_transport_failure(status);
    }
    void receive(unsigned tag,void* data,size_t bytes) {
        size_t got=0;checked(ucomm_recv(comm_,1,tag,data,bytes,&got));
        if(got!=bytes){healthy_=false;throw std::runtime_error("FP4 remote: truncated response");}
    }
    void request(unsigned op,unsigned layer,unsigned expert,unsigned tokens,unsigned tensor) {
        if(!healthy_||sequence_==UINT32_MAX)throw std::runtime_error("FP4 remote connection is failed or exhausted");
        uint32_t header[8]={STRATA_GLM_FP4_WORKER_MAGIC,STRATA_GLM_FP4_WORKER_VERSION,op,sequence_++,layer,expert,tokens,tensor};
        checked(ucomm_send(comm_,1,22,header,sizeof header));
    }
public:
    GlmFp4WorkerConnection(const std::string& address,const std::string& layout,const std::string& index,bool qpn,bool fetch) {
        if(layout.size()!=64||index.size()!=64)throw std::invalid_argument("FP4 remote invalid fingerprint");
        ucomm_config cfg;ucomm_config_default(&cfg);cfg.rank=0;cfg.world=2;cfg.root_addr=address.c_str();
        if(cfg.timeout_ms<=0)cfg.timeout_ms=60000;
        checked(ucomm_init(&cfg,&comm_));healthy_=true;
        std::array<char,132> hello{},peer{};
        std::memcpy(hello.data(),layout.data(),64);std::memcpy(hello.data()+64,index.data(),64);
        hello[128]=char(qpn);hello[129]=char(fetch);hello[130]=STRATA_GLM_FP4_WORKER_VERSION;
        checked(ucomm_send(comm_,1,20,hello.data(),hello.size()));receive(21,peer.data(),peer.size());
        if(peer!=hello){healthy_=false;throw std::runtime_error("FP4 remote handshake mismatch");}
    }
    ~GlmFp4WorkerConnection() {
        if(healthy_)try {request(0,0,0,0,0);ucomm_finalize(comm_);}catch(...){}
        // Failed contexts cannot safely enter ucomm's draining finalizer.
    }
    GlmFp4WorkerConnection(const GlmFp4WorkerConnection&)=delete;
    GlmFp4WorkerConnection& operator=(const GlmFp4WorkerConnection&)=delete;
    std::vector<float> run(unsigned layer,unsigned expert,const std::vector<float>& input) {
        if(input.empty()||input.size()%4096||input.size()>8*4096)throw std::invalid_argument("FP4 remote invalid activation size");
        request(1,layer,expert,input.size()/4096,0);
        checked(ucomm_send(comm_,1,23,input.data(),input.size()*sizeof(float)));
        std::vector<float> result(input.size());receive(24,result.data(),result.size()*sizeof(float));return result;
    }
    std::vector<uint8_t> fetch(unsigned tensor,size_t bytes) {
        if(bytes>32*1024*1024)throw std::invalid_argument("FP4 remote oversized tensor");
        request(2,0,0,0,tensor);std::vector<uint8_t> result(bytes);receive(24,result.data(),bytes);return result;
    }
};
}
