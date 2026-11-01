// src/minimax/m3_cpu_bench.cpp - the MiniMax CPU expert lane on its own: real expert geometry (6144 -> 3072 -> 6144,
// IQ2_XS gate/up, IQ3_XXS or IQ4_XS down), random weights with sane block scales, a pool of distinct experts so nothing
// stays in cache.  Sweeps the worker count and the row-chunk sizes and prints ms per layer-sized round and the
// effective GB/s of expert bytes.  No GPU, no model files.
//
//   m3_cpu_bench [--experts 96] [--rounds 200] [--per-round 3] [--threads 1,8,16,35,70] [--gu 16,32,64]
//                [--dn 48,96,192] [--down-type 18|23] [--touch main|workers]
#include "par.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace C = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static std::vector<int> ints(const std::string& s) {
    std::vector<int> r;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        r.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return r;
}

int main(int argc, char** argv) {
    int experts = 96, rounds = 200, per_round = 3, down_type = 18;
    std::vector<int> threads = {1, 8, 16, 35, 70}, gu = {16, 32, 64}, dn = {48, 96, 192};
    std::string touch = "main", kernel = "ggml";   // ggml: native_gu/down_rows; iq512 / iq256: the repo's multi-token kernels
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--experts") experts = std::atoi(v.c_str());
        else if (a == "--rounds") rounds = std::atoi(v.c_str());
        else if (a == "--per-round") per_round = std::atoi(v.c_str());
        else if (a == "--threads") threads = ints(v);
        else if (a == "--gu") gu = ints(v);
        else if (a == "--dn") dn = ints(v);
        else if (a == "--down-type") down_type = std::atoi(v.c_str());
        else if (a == "--touch") touch = v;
        else if (a == "--kernel") kernel = v;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    const int N = 6144, FF = 3072;
    C::NativeFmt f;
    std::string err;
    if (!C::native_fmt(17, down_type, N, FF, f, err)) { std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); return 1; }
    f.swiglu_alpha = 1.702f;
    f.swiglu_limit = 7.f;
    const size_t blob = f.bytes, total = (size_t) experts * blob;
    uint8_t* pool = (uint8_t*) mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(pool, total, MADV_HUGEPAGE);
    auto fill = [&](size_t first, size_t last, unsigned seed) {   // blobs [first, last)
        std::mt19937_64 rng(seed);
        for (size_t e = first; e < last; ++e) {
            uint8_t* b = pool + e * blob;
            for (size_t o = 0; o < blob; o += 8) { uint64_t r = rng(); std::memcpy(b + o, &r, std::min<size_t>(8, blob - o)); }
            auto scales = [&](size_t off, size_t row, size_t rows, size_t blk_bytes) {   // fp16 d ~ 0.01 at each block start
                for (size_t r = 0; r < rows; ++r)
                    for (size_t o = 0; o + blk_bytes <= row; o += blk_bytes) { b[off + r * row + o] = 0x1f; b[off + r * row + o + 1] = 0x21; }
            };
            scales(0, f.gu_row, FF, 74);
            scales(f.up_off, f.gu_row, FF, 74);
            scales(f.down_off, f.d_row, N, down_type == 18 ? 98 : 136);
        }
    };
    const auto tf = Clock::now();
    if (touch == "workers") {   // first touch spread over many threads, as the pack load does
        Par p(15);
        p.run(experts, [&](int e) { fill((size_t) e, (size_t) e + 1, 100 + e); });
    } else fill(0, (size_t) experts, 1);
    std::printf("pool %d experts x %.2f MB = %.2f GiB, filled by %s in %.1f s; down type %d\n", experts, blob / 1e6,
                total / 1073741824.0, touch.c_str(), ms_since(tf) / 1000.0, down_type);

    std::mt19937 rg(7);
    std::vector<float> x(N);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : x) v = nd(rg);
    std::vector<uint8_t> act(C::kNativeActBytes), hq((size_t) per_round * C::kNativeHBytes);
    std::vector<float> ffb((size_t) per_round * FF), outb((size_t) per_round * N);
    C::native_quant_act(f, x.data(), act.data());

    std::printf("%7s %6s %6s %10s %10s %10s %9s\n", "threads", "rowsGU", "rowsDN", "gate/up ms", "down ms", "round ms", "GB/s");
    for (int T : threads) {
        Par par(T - 1);
        for (int rgu : gu)
            for (int rdn : dn) {
                double t_gu = 0, t_dn = 0;
                for (int r = -10; r < rounds; ++r) {   // 10 warm-up rounds
                    size_t pick[8];
                    for (int j = 0; j < per_round; ++j) pick[j] = (size_t) ((r + 11) * 7 + j * 13) % (size_t) experts;
                    const void* actp[1] = {act.data()};
                    auto t0 = Clock::now();
                    const int gc = (FF + rgu - 1) / rgu, dc = (N + rdn - 1) / rdn;
                    par.run(per_round * gc, [&](int i) {
                        const int j = i / gc, c = i % gc;
                        float* ffp[1] = {ffb.data() + (size_t) j * FF};
                        const int r0 = c * rgu, r1 = std::min(FF, (c + 1) * rgu);
                        if (kernel == "ggml") {
                            C::native_gu_rows(f, pool + pick[j] * blob, actp, 1, ffp, r0, r1);
                        } else {
                            // raw gate and up dots with the repo's multi-token kernel, then swigluoai
                            thread_local float upb[4096];
                            float* upp[1] = {upb};
                            const uint8_t* bl = pool + pick[j] * blob;
                            if (kernel == "iq512") {
                                C::iq512_rows(f.gu_type, bl, f.gu_row, N, actp, 1, ffp, r0, r1);
                                C::iq512_rows(f.gu_type, bl + f.up_off, f.gu_row, N, actp, 1, upp, r0, r1);
                            } else {
                                C::iq256_rows(f.gu_type, bl, f.gu_row, N, actp, 1, ffp, r0, r1);
                                C::iq256_rows(f.gu_type, bl + f.up_off, f.gu_row, N, actp, 1, upp, r0, r1);
                            }
                            for (int r = r0; r < r1; ++r) {
                                float g = std::fmin(ffp[0][r], 7.f), u = std::fmax(-7.f, std::fmin(upb[r], 7.f));
                                ffp[0][r] = g / (1.f + std::exp(-1.702f * g)) * (1.f + u);
                            }
                        }
                    });
                    auto t1 = Clock::now();
                    for (int j = 0; j < per_round; ++j) C::native_quant_h(f, ffb.data() + (size_t) j * FF, hq.data() + (size_t) j * C::kNativeHBytes);
                    auto t2 = Clock::now();
                    par.run(per_round * dc, [&](int i) {
                        const int j = i / dc, c = i % dc;
                        const void* hp[1] = {hq.data() + (size_t) j * C::kNativeHBytes};
                        float* op[1] = {outb.data() + (size_t) j * N};
                        const int r0 = c * rdn, r1 = std::min(N, (c + 1) * rdn);
                        const uint8_t* dw = pool + pick[j] * blob + f.down_off;
                        if (kernel == "iq512" && C::iq512_supported(f.d_type)) C::iq512_rows(f.d_type, dw, f.d_row, FF, hp, 1, op, r0, r1);
                        else if (kernel != "ggml" && C::iq256_supported(f.d_type)) C::iq256_rows(f.d_type, dw, f.d_row, FF, hp, 1, op, r0, r1);
                        else C::native_down_rows(f, pool + pick[j] * blob, hp, 1, op, r0, r1);
                    });
                    auto t3 = Clock::now();
                    if (r >= 0) {
                        t_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
                        t_dn += std::chrono::duration<double, std::milli>(t3 - t2).count();
                    }
                }
                const double gu_ms = t_gu / rounds, dn_ms = t_dn / rounds;
                std::printf("%7d %6d %6d %10.3f %10.3f %10.3f %9.1f\n", T, rgu, rdn, gu_ms, dn_ms, gu_ms + dn_ms,
                            per_round * (double) blob / 1e6 / (gu_ms + dn_ms));
                std::fflush(stdout);
            }
    }
    return 0;
}
