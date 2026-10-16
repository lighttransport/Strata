// src/kernels/cpu/iq1s_expert_parity.cpp - Muse Spark Open's routed experts (IQ1_S gate/up/down, 4096 -> 4096 -> 4096)
// through the engine's dispatch (native_gu_rows / native_down_rows) with NativeFmt::iq1s_rows, against ggml-cpu.
// No GPU, no model: random blocks (every grid index, scale and delta sign occurs) and random activations quantized the
// way the engine quantizes them.
//
//     iq1s_expert_parity
//         gate/up and down rows, 1..8 tokens: each token's rows bit for bit the same alone and in a group (#152), and
//         against ggml-cpu's IQ1_S vec_dot (the default dispatch, iq1s_rows off) rel <= 1e-5.
//     iq1s_expert_parity --bench [--threads 15] [--mb 1024] [--nt 1,2,4] [--reps 5]
//         whole experts (gate/up, the hidden's Q8_K, down) over --mb MB of expert blobs on T threads, each thread its
//         own experts: GB/s of expert bytes and ms per expert, the kernel and ggml-cpu's.  The decode-time CPU lane.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t kLatent = 4096, kFF = 4096;   // moe_latent_size, expert_feed_forward_length
constexpr int kMaxT = 8;
constexpr int kIq1s = 19;

uint64_t mix(uint64_t& s) {   // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// random bytes, then a sane fp16 scale at the front of every 50-byte block
void fill_expert(uint8_t* p, size_t bytes, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) {
        const uint64_t v = mix(seed);
        std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i));
    }
    const size_t bs = ggml_type_size((ggml_type) kIq1s);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-2f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(p + o, &h, 2);
    }
}

struct Acts {
    std::vector<std::vector<uint8_t>> x, h;
    const void* xp[kMaxT];
    const void* hp[kMaxT];
    Acts(const cpu::NativeFmt& f, unsigned seed)
        : x(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes)), h(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes)) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> xv((size_t) kLatent), hv((size_t) kFF);
        for (int t = 0; t < kMaxT; ++t) {
            for (auto& v : xv) v = nd(rng);
            for (auto& v : hv) v = 0.05f * nd(rng);
            cpu::native_quant_act(f, xv.data(), x[(size_t) t].data());
            cpu::native_quant_h(f, hv.data(), h[(size_t) t].data());
            xp[t] = x[(size_t) t].data();
            hp[t] = h[(size_t) t].data();
        }
    }
};

double rel(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

size_t rows_differ(const float* a, const float* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    return d;
}

bool formats(cpu::NativeFmt& kernel, cpu::NativeFmt& ggml) {
    std::string err;
    if (!cpu::native_fmt(kIq1s, kIq1s, kLatent, kFF, kernel, err) || !cpu::native_fmt(kIq1s, kIq1s, kLatent, kFF, ggml, err)) {
        std::printf("iq1s_expert_parity: %s\n", err.c_str());
        return false;
    }
    kernel.iq1s_rows = true;
    return true;
}

int check() {
    cpu::NativeFmt f, g;
    if (!formats(f, g)) return 1;
    std::vector<uint8_t> blob(f.bytes);
    uint64_t seed = 0x1a1500ull;
    fill_expert(blob.data(), blob.size(), seed);
    const Acts a(f, 19u);
    int failures = 0;
    // gate/up (ff rows) and down (latent rows), each token alone first: the rows every width must reproduce
    struct Phase { const char* name; size_t rows; bool down; };
    for (const Phase& p : {Phase{"gate/up", (size_t) kFF, false}, Phase{"down", (size_t) kLatent, true}}) {
        const size_t R = p.rows;
        std::vector<float> one(kMaxT * R), ref(kMaxT * R), grp(kMaxT * R);
        for (int t = 0; t < kMaxT; ++t) {
            float* o = &one[t * R];
            float* r = &ref[t * R];
            if (p.down) {
                cpu::native_down_rows(f, blob.data(), &a.hp[t], 1, &o, 0, (int) R);
                cpu::native_down_rows(g, blob.data(), &a.hp[t], 1, &r, 0, (int) R);
            } else {
                cpu::native_gu_rows(f, blob.data(), &a.xp[t], 1, &o, 0, (int) R);
                cpu::native_gu_rows(g, blob.data(), &a.xp[t], 1, &r, 0, (int) R);
            }
        }
        const double r_ggml = rel(one.data(), ref.data(), one.size());
        std::printf("  IQ1_S %-8s vs ggml-cpu: rel %.2e\n", p.name, r_ggml);
        if (!(r_ggml <= 1e-5)) { std::printf("  IQ1_S %-8s MISMATCH against ggml\n", p.name); ++failures; }
        size_t differ = 0;
        for (int nt = 1; nt <= kMaxT; ++nt) {
            float* op[kMaxT];
            for (int t = 0; t < nt; ++t) op[t] = &grp[t * R];
            if (p.down) cpu::native_down_rows(f, blob.data(), a.hp, nt, op, 0, (int) R);
            else cpu::native_gu_rows(f, blob.data(), a.xp, nt, op, 0, (int) R);
            for (int t = 0; t < nt; ++t) differ += rows_differ(op[t], &one[t * R], R);
        }
        std::printf("  IQ1_S %-8s 1..8 tokens: %zu rows differ from the one-token rows\n", p.name, differ);
        if (differ) ++failures;
    }
    return failures;
}

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    for (size_t from = 0;;) {
        const size_t comma = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, comma == std::string::npos ? std::string::npos : comma - from).c_str()));
        if (comma == std::string::npos) break;
        from = comma + 1;
    }
    return v;
}

int bench(int threads, int mb, const std::vector<int>& nts, int reps) {
    cpu::NativeFmt f, g;
    if (!formats(f, g)) return 1;
    const size_t E = std::max<size_t>((size_t) threads * 2, ((size_t) mb << 20) / f.bytes);
    std::vector<uint8_t> blobs(E * f.bytes);
    {   // first touch on the threads that read them (NUMA)
        std::vector<std::thread> th;
        for (int i = 0; i < threads; ++i)
            th.emplace_back([&, i] {
                uint64_t seed = 0xbe1500ull + (uint64_t) i;
                for (size_t e = E * i / threads; e < E * (i + 1) / threads; ++e)
                    fill_expert(blobs.data() + e * f.bytes, f.bytes, seed);
            });
        for (auto& t : th) t.join();
    }
    const Acts a(f, 5u);
    std::printf("  %zu experts of %.2f MB (%.0f MB), %d threads\n", E, f.bytes / 1e6, E * f.bytes / 1e6, threads);
    for (int nt : nts) {
        for (const cpu::NativeFmt* fp : {&f, &g}) {
            const cpu::NativeFmt& fm = *fp;
            auto run = [&] {
                std::vector<std::thread> th;
                for (int i = 0; i < threads; ++i)
                    th.emplace_back([&, i] {
                        std::vector<float> ffb((size_t) kMaxT * kFF), outb((size_t) kMaxT * kLatent);
                        std::vector<std::vector<uint8_t>> hq(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
                        float* ff[kMaxT];
                        float* out[kMaxT];
                        const void* hp[kMaxT];
                        for (int t = 0; t < kMaxT; ++t) {
                            ff[t] = &ffb[(size_t) t * kFF];
                            out[t] = &outb[(size_t) t * kLatent];
                            hp[t] = hq[(size_t) t].data();
                        }
                        for (size_t e = E * i / threads; e < E * (i + 1) / threads; ++e) {
                            const uint8_t* blob = blobs.data() + e * fm.bytes;
                            cpu::native_gu_rows(fm, blob, a.xp, nt, ff, 0, (int) kFF);
                            for (int t = 0; t < nt; ++t) cpu::native_quant_h(fm, ff[t], hq[(size_t) t].data());
                            cpu::native_down_rows(fm, blob, hp, nt, out, 0, (int) kLatent);
                        }
                    });
                for (auto& t : th) t.join();
            };
            run();
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                run();
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            const double bytes = (double) E * fm.bytes;
            std::printf("  %-6s nt %d: %6.2f GB/s, %6.1f GMAC/s, %.3f ms per expert per thread\n",
                        fp == &f ? "kernel" : "ggml", nt, bytes / best / 1e9,
                        (double) E * 3 * kLatent * kFF * nt / best / 1e9, best * 1e3 * threads / (double) E);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool do_bench = false;
    int threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2 - 1), mb = 1024, reps = 5;
    std::vector<int> nts = {1, 2, 4};
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const bool more = i + 1 < argc;
        if (s == "--bench") do_bench = true;
        else if (s == "--threads" && more) threads = std::max(1, std::atoi(argv[++i]));
        else if (s == "--mb" && more) mb = std::atoi(argv[++i]);
        else if (s == "--nt" && more) nts = ints(argv[++i]);
        else if (s == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else {
            std::fprintf(stderr, "usage: iq1s_expert_parity | --bench [--threads N] [--mb 1024] [--nt 1,2,4] [--reps 5]\n");
            return 2;
        }
    }
    ggml_cpu_init();
    std::printf("iq1s_expert_parity: %s (avx2 %d)\n", cpu::cpu_name().c_str(), (int) cpu::cpu_avx2_ok());
    if (!cpu::cpu_avx2_ok()) {
        std::printf("iq1s_expert_parity: no AVX2 here, nothing to check (ggml-cpu's dot runs IQ1_S)\n");
        return 0;
    }
    if (do_bench) return bench(threads, mb, nts, reps);
    const int failures = check();
    std::printf("iq1s_expert_parity: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
