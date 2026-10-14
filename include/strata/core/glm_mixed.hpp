#pragma once
#include "strata/host/glm_mixed.h"
#include "strata/host/glm_env.h"
#ifdef __linux__
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <poll.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <chrono>
#include <array>
#include <algorithm>
#include <iostream>
#include <sstream>
namespace strata::core {
inline void mixed_packet(int fd, strata_mixed_message &packet, bool send_value, int timeout_ms=120000) {
    if (!send_value) {
        pollfd p{fd,POLLIN,0};int rc;
        do {rc=poll(&p,1,timeout_ms);} while (rc<0 && errno==EINTR);
        if (rc!=1 || !(p.revents&POLLIN)) throw std::runtime_error("mixed worker timeout or disconnect");
    }
    ssize_t n;
    do {n=send_value ? send(fd,&packet,sizeof packet,MSG_NOSIGNAL) : recv(fd,&packet,sizeof packet,MSG_TRUNC);}
    while (n<0 && errno==EINTR);
    if (n!=sizeof packet) throw std::runtime_error("mixed worker disconnected or invalid packet size");
}
inline void mixed_signal(int fd,uint64_t &value,bool send_value,int timeout_ms=120000) {
    strata_mixed_message packet{value,0,0};mixed_packet(fd,packet,send_value,timeout_ms);
    if (packet.tag || packet.kind) throw std::runtime_error("mixed worker unexpected packet type");
    value=packet.sequence;
}
class MixedMapping {
public:
    strata_mixed_buffer *buffer = nullptr;
    int socket = -1;
    MixedMapping(int memory, int channel): socket(channel) {
        struct stat s{};
        if (fstat(memory, &s) || s.st_size != sizeof(strata_mixed_buffer))
            throw std::runtime_error("mixed buffer size mismatch");
        void *p = mmap(nullptr, sizeof(strata_mixed_buffer), PROT_READ | PROT_WRITE, MAP_SHARED, memory, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mixed buffer mmap failed");
        buffer = static_cast<strata_mixed_buffer *>(p);
    }
    ~MixedMapping() { if (buffer) munmap(buffer, sizeof *buffer); }
    MixedMapping(const MixedMapping &) = delete;
};
class MixedClient: public MixedMapping {
    uint64_t next = 0, decode_rows=0, prefill_rows=0;
    double wait_ms=0, decode_wait_ms=0, ownership_wait_ms=0, prefill_wait_ms=0;
    uint64_t decode_return_bytes=0;
    bool configured[64]{};
    uint8_t ownership[64][288]{};
    std::array<uint8_t,STRATA_MIXED_ROUTES> requested{}, seen{};
    uint32_t chunk_index=0, delivered=0;
public:
    MixedClient(int memory, int channel): MixedMapping(memory, channel) {
        uint64_t ready = 0; mixed_signal(socket, ready, false);
        if (ready != STRATA_MIXED_MAGIC) throw std::runtime_error("mixed worker handshake failed");
    }
    ~MixedClient() {
        buffer->reserved=1;
        uint64_t stop=0;
        try { mixed_signal(socket,stop,true); } catch (...) {}
        std::ostringstream report;
        report << "MIXED_CLIENT requests=" << next << " decode_rows=" << decode_rows
                  << " prefill_rows=" << prefill_rows << " blocked_wait_ms=" << wait_ms
                  << " decode_wait_ms=" << decode_wait_ms << " ownership_wait_ms=" << ownership_wait_ms
                  << " prefill_wait_ms=" << prefill_wait_ms << " decode_return_bytes=" << decode_return_bytes << '\n';
        std::cerr << report.str();
    }
    void configure(unsigned layer, unsigned experts, const unsigned long long *primary, const uint64_t *prompt=nullptr,
                   bool refresh=false, const float *scores=nullptr) {
        if (layer >= 64 || experts > 288) throw std::runtime_error("mixed ownership geometry");
        uint8_t owned[288]{};
        for (unsigned e=0; e<experts; ++e) owned[e] = primary && primary[e];
        if (configured[layer] && !refresh) {
            if (!std::memcmp(owned,ownership[layer],sizeof owned)) return;
            if (strata_env_truthy_one("STRATA_GLM_MIXED_STABLE_OWNERSHIP")) {
                bool overlap=false;
                for (unsigned e=0;e<experts;++e) overlap|=owned[e] && buffer->worker_owned[layer*288+e];
                if (!overlap) {std::memcpy(ownership[layer],owned,sizeof owned);return;}
            }
        }
        buffer->layer=layer; buffer->nt=1; buffer->routes=0; buffer->prefill=2;
        buffer->canon=0; buffer->swiglu_limit=0; buffer->gate_skip=0;
        buffer->chunk_tag=refresh ? 1 : 0;
        std::memcpy(buffer->primary_owned, owned, sizeof owned);
        if (strata_env_truthy_one("STRATA_GLM_MIXED_PROMPT_PRIOR") || strata_env_truthy_one("STRATA_GLM_MIXED_ADAPT_CACHE")) {
            for (unsigned e=0;e<experts;++e) buffer->route_weight[e]=scores ? scores[e] : prompt ? float(prompt[e]) : 0.f;
            if (!strata_mixed_prompt_valid(buffer,experts)) throw std::runtime_error("mixed prompt counts invalid");
        }
        submit(); wait();
        std::memcpy(ownership[layer], owned, sizeof owned); configured[layer]=true;
    }
    void submit() {
        buffer->magic = STRATA_MIXED_MAGIC; buffer->version = STRATA_MIXED_VERSION;
        buffer->sequence = ++next; buffer->generation = next;
        if (buffer->prefill==1 || buffer->prefill==STRATA_MIXED_PREFILL_PARTIAL) {
            // Prefill can evict any decode resident. Reload the ownership set
            // once before decode even if primary ownership did not change.
            std::fill(std::begin(configured),std::end(configured),false);
            requested.fill(0);seen.fill(0);chunk_index=0;delivered=0;
            const bool partial=buffer->prefill==STRATA_MIXED_PREFILL_PARTIAL;
            for (uint32_t i=0;i<(partial ? buffer->nt : buffer->routes);++i) {
                const int d=partial ? int(i) : buffer->destination[i];
                if (d<0 || d>=STRATA_MIXED_ROUTES || requested[d]) throw std::runtime_error("mixed prefill request destinations");
                requested[d]=1;
            }
        }
        uint64_t seq = next; mixed_signal(socket, seq, true);
    }
    void wait() {
        const auto start=std::chrono::steady_clock::now();
        uint64_t seq = 0; mixed_signal(socket, seq, false);
        if (seq != next || buffer->sequence != next || buffer->generation != next)
            throw std::runtime_error("mixed worker stale reply");
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        wait_ms+=elapsed;
        if (buffer->prefill==0 || buffer->prefill==STRATA_MIXED_DECODE_PARTIAL) {
            decode_rows+=buffer->routes; decode_wait_ms+=elapsed;
            decode_return_bytes+=uint64_t(buffer->prefill==0 ? buffer->routes : buffer->nt)*STRATA_MIXED_HIDDEN*sizeof(float);
        } else if (buffer->prefill==2) ownership_wait_ms+=elapsed;
        if (buffer->prefill==1) prefill_rows+=buffer->routes;
    }
    bool wait_chunk() {
        const auto start=std::chrono::steady_clock::now();
        strata_mixed_message packet{};mixed_packet(socket,packet,false);
        if (packet.sequence!=next || packet.kind!=STRATA_MIXED_CHUNK || packet.tag!=buffer->chunk_tag ||
            buffer->sequence!=next || buffer->generation!=next ||
            (packet.tag&~STRATA_MIXED_FINAL)!=++chunk_index ||
            !strata_mixed_chunk_valid(buffer,requested.data(),seen.data()))
            throw std::runtime_error("mixed prefill stale or invalid chunk");
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        wait_ms+=elapsed; prefill_wait_ms+=elapsed;
        delivered+=buffer->chunk_rows;prefill_rows+=buffer->chunk_rows;
        if (buffer->chunk_tag&STRATA_MIXED_FINAL) {
            if (delivered!=(buffer->prefill==STRATA_MIXED_PREFILL_PARTIAL ? buffer->nt : buffer->routes) || buffer->chunk_rows) throw std::runtime_error("mixed prefill incomplete rows");
            return false;
        }
        return true;
    }
    void acknowledge_chunk() { strata_mixed_message packet{next,chunk_index,STRATA_MIXED_ACK};mixed_packet(socket,packet,true); }
    void finish_partial(float *sum) {
        wait();
        for (uint32_t i=0;i<buffer->nt*STRATA_MIXED_HIDDEN;++i) sum[i]+=buffer->output[i];
    }
    void finish(float *rows) {
        wait();
        for (uint32_t i = 0; i < buffer->routes; ++i)
            std::memcpy(rows + size_t(buffer->destination[i]) * STRATA_MIXED_HIDDEN,
                        buffer->output + size_t(i) * STRATA_MIXED_HIDDEN, STRATA_MIXED_HIDDEN * sizeof(float));
    }
};
}
#endif
