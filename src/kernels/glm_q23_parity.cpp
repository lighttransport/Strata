// glm_q23_parity - the canonical expert arithmetic (canon_expert.hpp), CPU against GPU, bit for bit.
//
//     glm_q23_parity [--bench]
//
// Each case builds experts from random Q2_K/Q3_K bytes with finite fp16 scales, routes one to four tokens through
// them in groups, and compares every output row of glm_q23_expert_grouped with the CPU pool's kernels in canonical
// mode (canon_quant_q8k, q23_gu_rows, q23_rows). Inputs include a zero block, a clamped SwiGLU and both down
// formats. --bench also times the GPU kernels on expert-sized groups.
#include "strata/kernels/canon_expert.hpp"
#include "strata/kernels/cpu/q23_avx2.hpp"
#include "strata/kernels/glm_q23.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace cpu = strata::kernels::cpu;

namespace {
void ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
uint16_t half_of(float value) {   // exact for the powers of two times small mantissas used here
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const int exponent = int((bits >> 23) & 255) - 127 + 15;
    return uint16_t(((bits >> 31) << 15) | (exponent << 10) | ((bits >> 13) & 1023));
}
// Random block bytes with finite scales: Q2_K keeps d and dmin in its last four bytes, Q3_K d in its last two.
void fill_rows(std::vector<uint8_t>& w, int type, std::mt19937& rng) {
    for (auto& b : w) b = uint8_t(rng());
    const size_t block = type == 10 ? 84 : 110;
    std::uniform_real_distribution<float> scale(1.f / 4096, 1.f / 256);
    for (size_t at = 0; at + block <= w.size(); at += block) {
        const uint16_t d = half_of(scale(rng) * (rng() & 1 ? 1.f : -1.f));
        if (type == 10) {
            const uint16_t dm = half_of(scale(rng));
            std::memcpy(&w[at + 80], &d, 2);
            std::memcpy(&w[at + 82], &dm, 2);
        } else std::memcpy(&w[at + 108], &d, 2);
    }
}
struct Case { int gu, down, hidden, ff, experts, tokens; float limit; float skip = 0; };

template <class T> struct DeviceBuffer {
    T* p = nullptr;
    explicit DeviceBuffer(size_t n) { ok(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { cudaFree(p); }
    void put(const void* host, size_t n) { ok(cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
};

bool run(const Case& c, bool bench, std::mt19937& rng) {
    auto L = k::native_expert_layout(c.gu, c.down, c.hidden, c.ff);
    L.swiglu_limit = c.limit;
    L.gate_skip = c.skip;
    if (!k::glm_q23_supported(L)) { std::printf("unsupported layout\n"); return false; }
    // Every token routes to every expert, so groups have `tokens` entries; route j = t * experts + e.
    const int routes = c.experts * c.tokens;
    std::vector<std::vector<uint8_t>> blobs(c.experts, std::vector<uint8_t>(L.bytes));
    for (auto& blob : blobs) {
        std::vector<uint8_t> gate(L.up_off), up(L.up_off), down(L.bytes - L.down_off);
        fill_rows(gate, c.gu, rng); fill_rows(up, c.gu, rng); fill_rows(down, c.down, rng);
        std::memcpy(blob.data(), gate.data(), gate.size());
        std::memcpy(blob.data() + L.up_off, up.data(), up.size());
        std::memcpy(blob.data() + L.down_off, down.data(), down.size());
    }
    std::vector<float> x((size_t)c.tokens * c.hidden);
    std::normal_distribution<float> normal(0.f, 1.5f);
    for (auto& v : x) v = normal(rng);
    std::fill_n(x.begin() + 256, 256, 0.f);                 // a zero Q8_K block
    x[700] = 40.f;                                          // a dominant value: most of its block rounds to zero
    // CPU reference.
    const size_t act_bytes = (size_t)c.hidden / 256 * 292, h_bytes = (size_t)c.ff / 256 * 292;
    std::vector<std::vector<uint8_t>> xq(c.tokens, std::vector<uint8_t>(act_bytes));
    for (int t = 0; t < c.tokens; ++t) cpu::canon_quant_q8k(x.data() + (size_t)t * c.hidden, xq[t].data(), c.hidden);
    std::vector<float> expected((size_t)routes * c.hidden);
    for (int e = 0; e < c.experts; ++e) {
        std::vector<std::vector<float>> h(c.tokens, std::vector<float>(c.ff));
        std::vector<std::vector<uint8_t>> hq(c.tokens, std::vector<uint8_t>(h_bytes));
        std::vector<const void*> act(c.tokens), hact(c.tokens);
        std::vector<float*> hp(c.tokens), out(c.tokens);
        for (int t = 0; t < c.tokens; ++t) {
            act[t] = xq[t].data(); hp[t] = h[t].data(); hact[t] = hq[t].data();
            out[t] = expected.data() + (size_t)(t * c.experts + e) * c.hidden;
        }
        cpu::q23_gu_rows(c.gu, blobs[e].data(), blobs[e].data() + L.up_off, L.gu_row, c.hidden, act.data(), c.tokens,
                         hp.data(), 0, c.ff, c.limit, true, c.skip);
        for (int t = 0; t < c.tokens; ++t) cpu::canon_quant_q8k(h[t].data(), hq[t].data(), c.ff);
        cpu::q23_rows(c.down, blobs[e].data() + L.down_off, L.d_row, c.ff, hact.data(), c.tokens, out.data(), 0, c.hidden);
        // Width invariance: each token alone gives the rows it gets in the group.
        for (int t = 0; t < c.tokens; ++t) {
            std::vector<float> h1(c.ff), out1(c.hidden);
            std::vector<uint8_t> hq1(h_bytes);
            const void* a1 = xq[t].data();
            float* hp1 = h1.data();
            cpu::q23_gu_rows(c.gu, blobs[e].data(), blobs[e].data() + L.up_off, L.gu_row, c.hidden, &a1, 1, &hp1, 0, c.ff, c.limit, true, c.skip);
            cpu::canon_quant_q8k(h1.data(), hq1.data(), c.ff);
            const void* ha1 = hq1.data();
            float* op1 = out1.data();
            cpu::q23_rows(c.down, blobs[e].data() + L.down_off, L.d_row, c.ff, &ha1, 1, &op1, 0, c.hidden);
            if (std::memcmp(out1.data(), out[t], (size_t)c.hidden * 4)) {
                std::printf("  CPU rows of token %d differ between one-token and %d-token calls\n", t, c.tokens);
                return false;
            }
        }
    }
    // GPU.
    DeviceBuffer<uint8_t> weights((size_t)c.experts * L.bytes);
    for (int e = 0; e < c.experts; ++e)
        ok(cudaMemcpy(weights.p + (size_t)e * L.bytes, blobs[e].data(), L.bytes, cudaMemcpyHostToDevice), "weights");
    std::vector<unsigned long long> ptr(c.experts);
    std::vector<int32_t> start(c.experts + 1), dst(routes), tok(routes), count{c.experts};
    for (int e = 0; e < c.experts; ++e) {
        ptr[e] = (unsigned long long)(weights.p + (size_t)e * L.bytes);
        start[e] = e * c.tokens;
        for (int t = 0; t < c.tokens; ++t) { dst[e * c.tokens + t] = t * c.experts + e; tok[e * c.tokens + t] = t; }
    }
    start[c.experts] = routes;
    DeviceBuffer<unsigned long long> dptr(c.experts);
    DeviceBuffer<int32_t> dstart(c.experts + 1), ddst(routes), dtok(routes), dcount(1);
    DeviceBuffer<float> dx(x.size()), dout(expected.size());
    DeviceBuffer<uint8_t> scratch(k::glm_q23_scratch_bytes(L, c.tokens, routes));
    dptr.put(ptr.data(), ptr.size()); dstart.put(start.data(), start.size()); ddst.put(dst.data(), dst.size());
    dtok.put(tok.data(), tok.size()); dcount.put(count.data(), 1); dx.put(x.data(), x.size());
    ok(cudaMemset(dout.p, 0xff, expected.size() * 4), "memset");
    auto launch = [&] {
        k::glm_q23_expert_grouped(L, dptr.p, dstart.p, dcount.p, ddst.p, dtok.p, c.experts, routes, dx.p, c.tokens,
                                  scratch.p, dout.p, nullptr);
    };
    launch();
    ok(cudaDeviceSynchronize(), "kernels");
    std::vector<float> actual(expected.size());
    ok(cudaMemcpy(actual.data(), dout.p, actual.size() * 4, cudaMemcpyDeviceToHost), "download");
    size_t differing = 0, nonfinite = 0;
    double largest = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i])) ++nonfinite;
        if (std::memcmp(&expected[i], &actual[i], 4)) {
            if (!differing)
                std::printf("  first difference at route %zu row %zu: cpu %.9g gpu %.9g\n", i / c.hidden, i % c.hidden,
                            expected[i], actual[i]);
            ++differing;
        }
        largest = std::max(largest, (double)std::fabs(expected[i]));
    }
    std::printf("gu=%d down=%d %dx%d experts=%d tokens=%d limit=%g skip=%g: %zu of %zu outputs differ, %zu non-finite, max |y| %.4g\n",
                c.gu, c.down, c.hidden, c.ff, c.experts, c.tokens, c.limit, c.skip, differing, expected.size(), nonfinite, largest);
    if (bench) {
        const int rounds = 50;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < rounds; ++i) launch();
        ok(cudaDeviceSynchronize(), "bench");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / rounds;
        std::printf("  bench: %.3f ms per call, %.1f GB/s of expert bytes (%.1f GB/s x tokens)\n", ms,
                    c.experts * double(L.bytes) / ms / 1e6, routes * double(L.bytes) / ms / 1e6);
    }
    return !differing && !nonfinite && largest > 0;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::mt19937 rng(20261007);
    bool pass = true;
    for (const Case& c : {Case{10, 11, 4096, 2048, 2, 1, 10.f}, Case{10, 11, 4096, 2048, 3, 3, 10.f},
                          Case{10, 11, 4096, 2048, 2, 4, 0.f}, Case{10, 10, 4096, 2048, 2, 2, 10.f},
                          Case{11, 11, 4096, 2048, 1, 3, 10.f}, Case{10, 11, 1024, 512, 5, 6, 10.f},
                          Case{10, 11, 4096, 2048, 2, 3, 10.f, 0.25f}, Case{10, 10, 4096, 2048, 2, 2, 10.f, 2.f}})
        pass = run(c, false, rng) && pass;
    if (bench)
        for (const Case& c : {Case{10, 11, 4096, 2048, 8, 1, 10.f}, Case{10, 11, 4096, 2048, 8, 3, 10.f},
                              Case{10, 11, 4096, 2048, 24, 1, 10.f}, Case{10, 11, 4096, 2048, 1, 1, 10.f}})
            pass = run(c, true, rng) && pass;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
