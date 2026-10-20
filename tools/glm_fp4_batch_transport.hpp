#pragma once
#include "glm_fp4_registered_staging.hpp"
#include "strata/host/glm_fp4_protocol.h"
#include <array>
#include <bit>
#include <cstring>
#include <cstdio>
#include <string>
#include <limits>
#include <memory>
#include <stdexcept>

namespace strata::experiment {
// One synchronous owner per link. Registrations outlive all sends/receives and
// are destroyed before the ucomm context. No model weights cross this link.
class NativeBatchLink {
    std::unique_ptr<ucomm_t,decltype(&ucomm_finalize)> context_{nullptr,ucomm_finalize};
    RegisteredStaging staging_;
    static void ok(ucomm_status status){if(status!=UCOMM_OK)throw std::runtime_error(ucomm_strerror(status));}
public:
    using Header=std::array<uint32_t,9>;
    NativeBatchLink(const std::string& root,bool server,int port,bool udp) {
        static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);
        if(std::endian::native!=std::endian::little)throw std::runtime_error("batch protocol requires little endian");
        ucomm_config cfg;ucomm_config_default(&cfg);cfg.world=2;cfg.rank=server?0:1;
        cfg.root_addr=root.c_str();cfg.ib_port=port;cfg.timeout_ms=10000;
        cfg.backend=udp?UCOMM_BACKEND_UDP:UCOMM_BACKEND_IB;
        ucomm_t* raw=nullptr;ok(ucomm_init(&cfg,&raw));context_.reset(raw);
        staging_.add(raw,STRATA_GLM_FP4_BATCH_WIDTH*sizeof(float));
        const std::array<uint32_t,4> hello={STRATA_GLM_FP4_BATCH_MAGIC,STRATA_GLM_FP4_BATCH_VERSION,STRATA_GLM_FP4_BATCH_WIDTH,STRATA_GLM_FP4_BATCH_MAX_EXPERTS};
        std::array<uint32_t,4> peer{};
        if(server){send(1,hello.data(),sizeof(hello));receive(1,peer.data(),sizeof(peer));}
        else{receive(1,peer.data(),sizeof(peer));send(1,hello.data(),sizeof(hello));}
        if(peer!=hello)throw std::runtime_error("incompatible native batch peer");
        std::printf("TRANSPORT %s registered_staging=1 role=%s\n",ucomm_backend_name(raw),server?"server":"client");
    }
    void send(uint32_t tag,const void* data,size_t bytes) {
        auto* buffer=staging_.find(context_.get())->tx(bytes);std::memcpy(buffer,data,bytes);
        ok(ucomm_send(context_.get(),1-ucomm_rank(context_.get()),tag,buffer,bytes));
    }
    void receive(uint32_t tag,void* data,size_t bytes) {
        auto* buffer=staging_.find(context_.get())->rx(bytes);size_t got=0;
        ok(ucomm_recv(context_.get(),1-ucomm_rank(context_.get()),tag,buffer,bytes,&got));
        if(got!=bytes)throw std::runtime_error("batch payload length mismatch");
        std::memcpy(data,buffer,bytes);
    }
    static Header command(uint32_t sequence,uint32_t layer,uint32_t count,const std::array<int,3>& ids,bool stop=false) {
        Header h={STRATA_GLM_FP4_BATCH_MAGIC,STRATA_GLM_FP4_BATCH_VERSION,stop?0u:1u,layer,count,sequence,0,0,0};
        if(count>3)throw std::invalid_argument("batch count out of bounds");
        for(uint32_t i=0;i<count;++i)h[6+i]=uint32_t(ids[i]);
        if(!strata_glm_fp4_batch_command_valid(h.data(),sequence))throw std::invalid_argument("invalid outgoing batch command");
        return h;
    }
    Header receive_command(uint32_t expected_sequence) {
        Header h{};receive(10,h.data(),sizeof(h));
        if(!strata_glm_fp4_batch_command_valid(h.data(),expected_sequence))throw std::runtime_error("invalid incoming batch command");
        return h;
    }
    void acknowledge(uint32_t sequence){send(12,&sequence,sizeof(sequence));}
    void check_ack(uint32_t sequence){uint32_t got=0;receive(12,&got,sizeof(got));if(got!=sequence)throw std::runtime_error("invalid batch acknowledgement");}
};
}
