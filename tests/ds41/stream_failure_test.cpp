// Exercise the real worker stream with a deterministic transport: no GPU, model or network.
// Build with -ffunction-sections -fdata-sections -Wl,--gc-sections so unused worker-main
// dependencies are discarded. tools/test_ds41_stream.py drives both scenarios with a timeout.
#define main unused_ds41_worker_main
#include "../../src/program/ds41_cpu_worker.cpp"
#undef main

#include <cassert>

namespace {
bool fail_send = false;
int received = 0, sent = 0;
constexpr int count = 128;
}
extern "C" {
ucomm_status ucomm_init(const ucomm_config*, ucomm_t** out) {
    *out = reinterpret_cast<ucomm_t*>(uintptr_t(1)); return UCOMM_OK;
}
ucomm_status ucomm_mr_reg(ucomm_t*, void*, size_t, ucomm_mr_t** out) {
    *out = reinterpret_cast<ucomm_mr_t*>(uintptr_t(1)); return UCOMM_OK;
}
void ucomm_mr_dereg(ucomm_mr_t*) {}
void ucomm_finalize(ucomm_t*) { assert(!fail_send); }
const char* ucomm_strerror(ucomm_status) { return "injected send failure"; }
ucomm_status ucomm_recv(ucomm_t*, int, uint32_t tag, void* dst, size_t cap, size_t* got) {
    if (tag == 60) {
        const int32_t header[4] = {received++ ? -1 : 0, count, 0, 0};
        assert(cap == sizeof header);
        std::memcpy(dst, header, sizeof header);
    } else {
        assert(tag == 61 && cap == count * sizeof(int32_t));
        for (int i = 0; i < count; ++i) static_cast<int32_t*>(dst)[i] = i;
    }
    if (got) *got = cap;
    return UCOMM_OK;
}
ucomm_status ucomm_send(ucomm_t*, int, uint32_t tag, const void* data, size_t size) {
    assert(tag == 62 && size == 4096);
    // Let the copiers fill the two-slot ring and block with more work pending.
    if (fail_send) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); return UCOMM_ERR_IO; }
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) assert(bytes[i] == uint8_t(sent));
    ++sent;
    return UCOMM_OK;
}
}
int main(int argc, char**) {
    fail_send = argc > 1;
    setenv("STRATA_DS41_STREAM_RING", "2", 1);
    setenv("STRATA_DS41_STREAM_COPIERS", "4", 1);
    std::vector<uint8_t> bytes(size_t(count) * 4096);
    std::vector<Layer> layers(1);
    auto& layer = layers[0];
    layer.f.bytes = layer.slot_bytes = 4096;
    layer.base = bytes.data();
    for (int i = 0; i < count; ++i) {
        layer.local.push_back(i);
        std::memset(bytes.data() + size_t(i) * 4096, i, 4096);
    }
    StreamServer stream("192.0.2.1:29574", ucomm_config{}, layers, -1);
    stream.join();
    assert(sent == count);
}
