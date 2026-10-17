// tools/a64fx_probe/q23_sdot_bench.cpp - the SVE SDOT Q2_K/Q3_K kernel (q23_sdot_sve.c) against ggml's vec_dot on A64FX.
//
//   q23_sdot_bench [--threads 48] [--mb 2048] [--reps 3] [--rows 2048 --cols 4096] [--tokens 1,2,3,4] [--types q2_k,q3_k]
//                  [--kernels ggml,sdot] [--check]
//
// --check: for 1..4 tokens, 64 random rows, the SDOT result against ggml's vec_dot on the same Q8_K activation
// (vector relative error; the integer sums are identical, only float order differs) -> PASS at <= 1e-5.
// Timing: every thread streams its own rows (larger than any cache) through the kernel; ggml handles several tokens
// as one vec_dot per token, the SDOT kernel decodes a block once for all tokens.
#include "ggml.h"
#include "ggml-cpu.h"
#include "q23_sdot_sve.h"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

uint64_t mix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void fill_rows(uint8_t* p, size_t bytes, ggml_type t, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) { const uint64_t v = mix(seed); std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i)); }
    const size_t bs = ggml_type_size(t);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-2f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d), h2 = ggml_fp32_to_fp16(d * 0.5f);
        if (t == GGML_TYPE_Q2_K) { std::memcpy(p + o + 80, &h, 2); std::memcpy(p + o + 82, &h2, 2); }
        if (t == GGML_TYPE_Q3_K) std::memcpy(p + o + 108, &h, 2);
    }
}

struct Act { std::vector<uint8_t> q8k; std::vector<q23s_act> s; };

Act make_act(int cols, unsigned seed) {
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> f(cols); for (auto& v : f) v = nd(rng);
    Act a; a.q8k.resize(ggml_row_size(GGML_TYPE_Q8_K, cols));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(f.data(), a.q8k.data(), cols);
    a.s.resize(cols / 256); q23s_prepare(a.q8k.data(), cols, a.s.data());
    return a;
}

using SdotFn = void (*)(int, const void*, const q23s_act* const*, int, float*);
SdotFn sdot_for(ggml_type t) { return t == GGML_TYPE_Q2_K ? q2k_sdot_row : q3k_sdot_row; }

int check(const std::vector<ggml_type>& types, int cols) {
    int failures = 0;
    for (ggml_type t : types) {
        const auto* tr = ggml_get_type_traits_cpu(t);
        const size_t rb = ggml_row_size(t, cols); const int rows = 64;
        std::vector<uint8_t> m(rb * rows); uint64_t seed = 0xc0ffee + t; fill_rows(m.data(), m.size(), t, seed);
        std::vector<Act> acts; for (int k = 0; k < 4; ++k) acts.push_back(make_act(cols, 101u + k));
        for (int nt = 1; nt <= 4; ++nt) {
            const q23s_act* ap[4]; for (int k = 0; k < 4; ++k) ap[k] = acts[k].s.data();
            double num = 0, den = 0;
            for (int r = 0; r < rows; ++r) {
                float got[4]; sdot_for(t)(cols, m.data() + (size_t) r * rb, ap, nt, got);
                for (int k = 0; k < nt; ++k) {
                    float ref = 0; tr->vec_dot(cols, &ref, 0, m.data() + (size_t) r * rb, 0, acts[k].q8k.data(), 0, 1);
                    num += (double) (got[k] - ref) * (got[k] - ref); den += (double) ref * ref;
                }
            }
            const double rel = std::sqrt(num / std::max(den, 1e-30)); const bool ok = rel <= 1e-5;
            std::printf("  %-5s tokens=%d sdot vs ggml vec_dot: rel %.2e %s\n", ggml_type_name(t), nt, rel, ok ? "PASS" : "FAIL");
            failures += !ok;
        }
    }
    return failures;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out; size_t p = 0;
    while (p <= s.size()) { size_t e = s.find(',', p); if (e == std::string::npos) e = s.size(); if (e > p) out.push_back(s.substr(p, e - p)); p = e + 1; }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    int threads = omp_get_max_threads(), mb = 2048, reps = 3, rows = 2048, cols = 4096; bool do_check = false;
    std::string tokens_arg = "1,2,3,4", types_arg = "q2_k,q3_k", kernels_arg = "ggml,sdot";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--threads") threads = std::stoi(next()); else if (a == "--mb") mb = std::stoi(next()); else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--rows") rows = std::stoi(next()); else if (a == "--cols") cols = std::stoi(next()); else if (a == "--tokens") tokens_arg = next();
        else if (a == "--types") types_arg = next(); else if (a == "--kernels") kernels_arg = next(); else if (a == "--check") do_check = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    ggml_cpu_init();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (q23s_vl_bytes() != 64) { std::printf("A64FX_SDOT FAIL: SVE vector length %d bytes, kernel needs 64\n", q23s_vl_bytes()); return 1; }
    std::vector<ggml_type> types;
    for (auto& s : split(types_arg)) { if (s == "q2_k") types.push_back(GGML_TYPE_Q2_K); else if (s == "q3_k") types.push_back(GGML_TYPE_Q3_K); else { std::fprintf(stderr, "type %s\n", s.c_str()); return 2; } }
    std::printf("q23_sdot_bench: %d threads, rows %d x cols %d, %d MB of rows per run, %d reps\n", threads, rows, cols, mb, reps);
    if (do_check && check(types, cols)) return 1;
    omp_set_num_threads(threads);
    for (ggml_type t : types) {
        const auto* tr = ggml_get_type_traits_cpu(t);
        const size_t rb = ggml_row_size(t, cols), mat = rb * rows;
        const size_t mats = std::max<size_t>(1, (size_t) mb * 1024 * 1024 / threads / mat);
        std::vector<std::vector<uint8_t>> mem(threads);
        #pragma omp parallel num_threads(threads)
        { const int tid = omp_get_thread_num(); mem[tid].resize(mats * mat); uint64_t seed = 0x5eed + tid + 1000 * t; fill_rows(mem[tid].data(), mem[tid].size(), t, seed); }
        std::vector<Act> acts; for (int k = 0; k < 4; ++k) acts.push_back(make_act(cols, 11u + k));
        const q23s_act* ap[4]; for (int k = 0; k < 4; ++k) ap[k] = acts[k].s.data();
        for (auto& kern : split(kernels_arg)) for (auto& ts : split(tokens_arg)) {
            const int nt = std::stoi(ts); const bool sd = kern == "sdot";
            std::vector<float> sink(threads); double best = 1e30;
            for (int rep = 0; rep < reps; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                #pragma omp parallel num_threads(threads)
                {
                    const int tid = omp_get_thread_num(); float acc = 0, o[4];
                    for (size_t m = 0; m < mats; ++m) {
                        const uint8_t* base = mem[tid].data() + m * mat;
                        for (int r = 0; r < rows; ++r) {
                            const uint8_t* row = base + (size_t) r * rb;
                            if (sd) { sdot_for(t)(cols, row, ap, nt, o); for (int k = 0; k < nt; ++k) acc += o[k]; }
                            else for (int k = 0; k < nt; ++k) { float v; tr->vec_dot(cols, &v, 0, row, 0, acts[k].q8k.data(), 0, 1); acc += v; }
                        }
                    }
                    sink[tid] = acc;
                }
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            const double bytes = (double) threads * mats * mat, macs = bytes / rb * cols * nt;
            std::printf("A64FX_SDOT kernel=%s type=%s threads=%d rows=%d cols=%d tokens=%d GBps=%.1f GMACs=%.1f GBps_per_thread=%.2f\n",
                        kern.c_str(), ggml_type_name(t), threads, rows, cols, nt, bytes / best / 1e9, macs / best / 1e9, bytes / best / 1e9 / threads);
        }
    }
    return 0;
}
