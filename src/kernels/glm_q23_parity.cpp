// glm_q23_parity - the canonical expert arithmetic (canon_expert.hpp), CPU against GPU, bit for bit.
//
//     glm_q23_parity [--bench]
//
// Each case builds experts from random Q2_K/Q3_K bytes with finite fp16 scales, routes one to eight tokens through
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
#include <fstream>
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
void fill_rows(std::vector<uint8_t>& w, int type, std::mt19937& rng, bool tiny = false) {
    for (auto& b : w) b = uint8_t(rng());
    const size_t block = type == 10 ? 84 : 110;
    std::uniform_real_distribution<float> scale(1.f / 4096, 1.f / 256);
    for (size_t at = 0; at + block <= w.size(); at += block) {
        const uint16_t d = tiny ? uint16_t((rng() % 1024) | ((rng() & 1) << 15)) : half_of(scale(rng) * (rng() & 1 ? 1.f : -1.f));
        if (type == 10) {
            const uint16_t dm = tiny ? uint16_t(rng() % 1024) : half_of(scale(rng));
            std::memcpy(&w[at + 80], &d, 2);
            std::memcpy(&w[at + 82], &dm, 2);
        } else std::memcpy(&w[at + 108], &d, 2);
    }
}
struct Case { int gu, down, hidden, ff, experts, tokens; float limit; float skip = 0; int ragged = 0; bool tiny = false; bool rounding_boundary = false; };

template <class T> struct DeviceBuffer {
    T* p = nullptr;
    explicit DeviceBuffer(size_t n) { ok(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { cudaFree(p); }
    void put(const void* host, size_t n) { ok(cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
};

bool run(const Case& c, bool bench, std::mt19937& rng, std::istream* fixture = nullptr) {
    auto L = k::native_expert_layout(c.gu, c.down, c.hidden, c.ff);
    L.swiglu_limit = c.limit;
    L.gate_skip = c.skip;
    if (!k::glm_q23_supported(L)) { std::printf("unsupported layout\n"); return false; }
    // Every token routes to every expert, so groups have `tokens` entries; route j = t * experts + e.
    const int routes = c.experts * c.tokens;
    std::vector<std::vector<uint8_t>> blobs(c.experts, std::vector<uint8_t>(L.bytes));
    for (auto& blob : blobs) {
        std::vector<uint8_t> gate(L.up_off), up(L.up_off), down(L.bytes - L.down_off);
        fill_rows(gate, c.gu, rng, c.tiny); fill_rows(up, c.gu, rng, c.tiny); fill_rows(down, c.down, rng, c.tiny);
        std::memcpy(blob.data(), gate.data(), gate.size());
        std::memcpy(blob.data() + L.up_off, up.data(), up.size());
        std::memcpy(blob.data() + L.down_off, down.data(), down.size());
    }
    std::vector<float> x((size_t)c.tokens * c.hidden);
    std::normal_distribution<float> normal(0.f, 1.5f);
    for (auto& v : x) v = normal(rng);
    std::fill_n(x.begin() + 256, 256, 0.f);                 // a zero Q8_K block
    x[700] = 40.f;                                          // a dominant value: most of its block rounds to zero
    if (fixture) {
        fixture->read((char *)blobs[0].data(), L.bytes);
        fixture->read((char *)x.data(), x.size() * sizeof(float));
        if (!*fixture) { std::fprintf(stderr, "truncated fixture\n"); return false; }
    }
    if (c.rounding_boundary) {
        // A separately rounded multiply lands on 1.5; a contracted FMA lands just below it.
        // HIP __fmul_rn/__fadd_rn wrappers alone do not prevent contraction.
        std::fill_n(x.begin(), 256, 0.f);
        const uint32_t bits[]{0x3f8ccccd, 0xbc54dced};
        std::memcpy(x.data(), bits, sizeof(bits));
    }
    // CPU reference.
    const size_t act_bytes = (size_t)c.hidden / 256 * 292, h_bytes = (size_t)c.ff / 256 * 292;
    std::vector<std::vector<uint8_t>> xq(c.tokens, std::vector<uint8_t>(act_bytes));
    for (int t = 0; t < c.tokens; ++t) cpu::canon_quant_q8k(x.data() + (size_t)t * c.hidden, xq[t].data(), c.hidden);
    std::vector<float> expected((size_t)routes * c.hidden), fixture_h;
    std::vector<uint8_t> fixture_hq;
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
        if (fixture) { fixture_h = h[0]; fixture_hq = hq[0]; }
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
    int entries = 0;
    for (int e = 0; e < c.experts; ++e) {
        ptr[e] = (unsigned long long)(weights.p + (size_t)e * L.bytes);
        start[e] = entries;
        const int width = c.ragged ? 1 + e % std::min(c.tokens, c.ragged) : c.tokens;
        for (int t = 0; t < width; ++t) { dst[entries] = t * c.experts + e; tok[entries++] = t; }
        for (int t = width; t < c.tokens; ++t)
            std::fill_n(expected.data() + (size_t)(t * c.experts + e) * c.hidden, c.hidden, 0.f);
    }
    start[c.experts] = entries;
    DeviceBuffer<unsigned long long> dptr(c.experts);
    DeviceBuffer<int32_t> dstart(c.experts + 1), ddst(routes), dtok(routes), dcount(1);
    DeviceBuffer<float> dx(x.size()), dout(expected.size());
    DeviceBuffer<uint8_t> scratch(k::glm_q23_scratch_bytes(L, c.tokens, routes));
    dptr.put(ptr.data(), ptr.size()); dstart.put(start.data(), start.size()); ddst.put(dst.data(), dst.size());
    dtok.put(tok.data(), tok.size()); dcount.put(count.data(), 1); dx.put(x.data(), x.size());
    ok(cudaMemset(dout.p, c.ragged ? 0 : 0xff, expected.size() * 4), "memset");
    auto launch = [&] {
        k::glm_q23_expert_grouped(L, dptr.p, dstart.p, dcount.p, ddst.p, dtok.p, c.experts, routes, dx.p, c.tokens,
                                  scratch.p, dout.p, nullptr);
    };
    launch();
    ok(cudaDeviceSynchronize(), "kernels");
    std::vector<float> actual(expected.size());
    ok(cudaMemcpy(actual.data(), dout.p, actual.size() * 4, cudaMemcpyDeviceToHost), "download");
    bool input_quant_equal = true;
    if (c.rounding_boundary) {
        std::vector<uint8_t> actual_xq(act_bytes);
        ok(cudaMemcpy(actual_xq.data(), scratch.p, act_bytes, cudaMemcpyDeviceToHost), "input quant download");
        input_quant_equal = actual_xq == xq[0];
        std::printf("rounding_boundary input_quant_equal=%d\n", int(input_quant_equal));
    }
    if (fixture) {
        std::vector<uint8_t> bytes(k::glm_q23_scratch_bytes(L, c.tokens, routes));
        ok(cudaMemcpy(bytes.data(), scratch.p, bytes.size(), cudaMemcpyDeviceToHost), "scratch download");
        const size_t ho = (act_bytes + 255) & ~size_t(255);
        const size_t hqo = ho + ((size_t(c.ff) * 4 + 255) & ~size_t(255));
        auto compare = [&](const char *name, const void *a, const void *b, size_t n) {
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i) if (((const uint8_t*)a)[i] != ((const uint8_t*)b)[i]) {
                if (diff < 8) std::printf("%s byte=%zu cpu=%u gpu=%u\n", name, i, ((const uint8_t*)a)[i], ((const uint8_t*)b)[i]);
                ++diff;
            }
            std::printf("%s differing_bytes=%zu\n", name, diff);
        };
        compare("xq", xq[0].data(), bytes.data(), act_bytes);
        compare("hidden", fixture_h.data(), bytes.data() + ho, c.ff * 4);
        compare("hq", fixture_hq.data(), bytes.data() + hqo, h_bytes);
    }
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
    std::printf("gu=%d down=%d %dx%d experts=%d tokens=%d limit=%g skip=%g ragged=%d: %zu of %zu outputs differ, %zu non-finite, max |y| %.4g\n",
                c.gu, c.down, c.hidden, c.ff, c.experts, c.tokens, c.limit, c.skip, int(c.ragged), differing, expected.size(), nonfinite, largest);
    if (bench) {
        const int rounds = 50;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < rounds; ++i) launch();
        ok(cudaDeviceSynchronize(), "bench");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / rounds;
        std::printf("  bench: %.3f ms per call, %.1f GB/s of expert bytes (%.1f GB/s x tokens)\n", ms,
                    c.experts * double(L.bytes) / ms / 1e6, entries * double(L.bytes) / ms / 1e6);
    }
    return input_quant_equal && !differing && !nonfinite && largest > 0;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::mt19937 rng(20261007);
    bool pass = true;
    if (argc == 3 && std::string(argv[1]) == "--fixture") {
        std::ifstream fixture(argv[2], std::ios::binary);
        int32_t h[4]{}; float limit = 0;
        fixture.read((char *)h, sizeof(h)); fixture.read((char *)&limit, sizeof(limit));
        if (!fixture || h[2] != 4096 || h[3] != 2048) return 2;
        pass = run(Case{h[0], h[1], h[2], h[3], 1, 1, limit}, false, rng, &fixture);
        std::printf("%s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    for (const Case& c : {Case{10, 11, 4096, 2048, 2, 1, 10.f}, Case{10, 11, 4096, 2048, 3, 2, 10.f},
                          Case{10, 11, 4096, 2048, 3, 3, 10.f},
                          Case{10, 11, 4096, 2048, 2, 4, 0.f}, Case{10, 10, 4096, 2048, 2, 2, 10.f},
                          Case{11, 11, 4096, 2048, 1, 3, 10.f}, Case{10, 11, 1024, 512, 5, 6, 10.f},
                          Case{10, 11, 4096, 2048, 2, 5, 10.f}, Case{10, 11, 4096, 2048, 2, 7, 10.f},
                          Case{10, 11, 4096, 2048, 3, 8, 10.f}, Case{10, 11, 4096, 2048, 8, 8, 10.f, 0.f, 8}, Case{10, 10, 4096, 2048, 2, 8, 10.f},
                          Case{11, 11, 4096, 2048, 1, 8, 10.f}, Case{10, 11, 4096, 2048, 2, 8, 10.f, 0.25f}, Case{10, 10, 4096, 2048, 2, 2, 10.f, 2.f},
                          Case{10, 11, 4096, 2048, 2, 1, 10.f, 0.f, 0, true},
                          Case{10, 11, 4096, 2048, 2, 4, 10.f, 0.f, 0, true},
                          Case{10, 11, 4096, 2048, 1, 1, 10.f, 0.f, 0, false, true}})
        pass = run(c, false, rng) && pass;
    if (bench)
        for (const Case& c : {Case{10, 11, 4096, 2048, 8, 1, 10.f}, Case{10, 11, 4096, 2048, 8, 2, 10.f},
                              Case{10, 11, 4096, 2048, 24, 2, 10.f}, Case{10, 11, 4096, 2048, 8, 3, 10.f},
                              Case{10, 11, 4096, 2048, 8, 8, 10.f}, Case{10, 11, 4096, 2048, 24, 8, 10.f},
                              Case{10, 11, 4096, 2048, 24, 8, 10.f, 0.f, 8},
                              Case{10, 11, 4096, 2048, 24, 8, 10.f, 0.f, 2},
                              Case{10, 11, 4096, 2048, 24, 1, 10.f}, Case{10, 11, 4096, 2048, 1, 1, 10.f}})
            pass = run(c, true, rng) && pass;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
