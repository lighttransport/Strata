#pragma once
// Layer-split transport (see include/strata/host/glm_stage.h). Linux only, like the mixed worker.
#include "strata/host/glm_stage.h"
#include "strata/core/glm_mixed.hpp"
#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <chrono>
namespace strata::core {
class StageMapping {
public:
    strata_stage_buffer *buffer = nullptr;
    int socket = -1;
    StageMapping(int memory, int channel): socket(channel) {
        struct stat s{};
        if (fstat(memory, &s) || size_t(s.st_size) < sizeof(strata_stage_buffer))
            throw std::runtime_error("stage buffer size mismatch");
        void *p = mmap(nullptr, sizeof(strata_stage_buffer), PROT_READ | PROT_WRITE, MAP_SHARED, memory, 0);
        if (p == MAP_FAILED) throw std::runtime_error("stage buffer mmap failed");
        buffer = static_cast<strata_stage_buffer *>(p);
    }
    ~StageMapping() { if (buffer) munmap(buffer, sizeof *buffer); }
    StageMapping(const StageMapping &) = delete;
    StageMapping &operator=(const StageMapping &) = delete;
};
// Primary side: one blocking call per command.
class StageClient: public StageMapping {
    uint64_t next = 0;
public:
    double wait_ms = 0; uint64_t calls = 0;
    StageClient(int memory, int channel): StageMapping(memory, channel) {
        uint64_t ready = 0; mixed_signal(socket, ready, false, 900000);
        if (ready != STRATA_STAGE_MAGIC) throw std::runtime_error("stage handshake failed");
    }
    ~StageClient() {
        std::fprintf(stderr, "STAGE_CLIENT calls=%llu wait_ms=%.1f\n", (unsigned long long)calls, wait_ms);
        try { auto &b = prepare(STRATA_STAGE_STOP); (void)b; send(); } catch (...) {}
    }
    strata_stage_buffer &prepare(uint32_t command) {
        auto &b = *buffer;
        b.magic = STRATA_STAGE_MAGIC; b.version = STRATA_STAGE_VERSION; b.command = command; b.status = 0;
        b.nt = 0; b.prefill = 0; b.reserved0 = b.reserved1 = 0; b.position = 0; b.count = 0; b.id = 0; b.result = 0;
        b.error[0] = 0;
        return b;
    }
    void send() { b_sequence(); uint64_t seq = next; mixed_signal(socket, seq, true); }
    // Sends the prepared request and waits for its reply; throws the stage's error message.
    strata_stage_buffer &call(int timeout_ms = 900000) {
        const auto start = std::chrono::steady_clock::now();
        send();
        uint64_t seq = 0; mixed_signal(socket, seq, false, timeout_ms);
        if (seq != next || buffer->sequence != next) throw std::runtime_error("stage stale reply");
        wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); ++calls;
        if (buffer->status) throw std::runtime_error(std::string("stage: ") + buffer->error);
        return *buffer;
    }
private:
    void b_sequence() { buffer->sequence = ++next; }
};
}
#endif
