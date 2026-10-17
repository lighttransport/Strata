// tools/a64fx_probe/q23_sve_bench.cpp - ggml's quantized dot kernels on A64FX, GLM-5.3-Flash expert shapes.
//
// The A64FX simulator (tools/sim/a64fx.py) has one unmeasured constant: the Q2_K/Q3_K (and IQ2/IQ1) expert dot
// rate per core on SVE. This bench streams synthetic expert rows (random blocks with sane scales, larger than any
// cache) through ggml-cpu's vec_dot for each format, on T threads, each thread its own rows, and reports GB/s of
// expert bytes and GMAC/s; `--check` compares every kernel against a double-precision dequantize-and-dot.
//
//   q23_sve_bench [--threads 48] [--mb 1024] [--reps 3] [--rows 2048 --cols 4096] [--types q2_k,q3_k,q4_k,iq2_xxs,iq2_s,iq1_s,q8_0]
//                 [--tokens 1] [--check]
//
// GLM-5.3-Flash: gate/up rows 2048 x 4096 inputs, down 4096 x 2048. DeepSeek V4.1 Flash: 2304 x 5120 / 5120 x 2304.
// Build (A64FX, Fujitsu clang mode; ggml from a llama.cpp checkout built with the a64fx recipe):
//   FCC -Nclang -O3 -fopenmp -march=armv8.2-a+sve -std=c++17 -I$LLAMA/ggml/include q23_sve_bench.cpp \
//       -L$BUILD/bin -lggml-cpu -lggml-base -o q23_sve_bench
#include "ggml.h"
#include "ggml-cpu.h"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

const std::map<std::string, ggml_type> kTypes = {
    {"q2_k", GGML_TYPE_Q2_K}, {"q3_k", GGML_TYPE_Q3_K}, {"q4_k", GGML_TYPE_Q4_K}, {"q5_k", GGML_TYPE_Q5_K}, {"q6_k", GGML_TYPE_Q6_K},
    {"q8_0", GGML_TYPE_Q8_0}, {"q4_0", GGML_TYPE_Q4_0}, {"iq2_xxs", GGML_TYPE_IQ2_XXS}, {"iq2_xs", GGML_TYPE_IQ2_XS},
    {"iq2_s", GGML_TYPE_IQ2_S}, {"iq3_xxs", GGML_TYPE_IQ3_XXS}, {"iq1_s", GGML_TYPE_IQ1_S}, {"iq1_m", GGML_TYPE_IQ1_M},
};

// Random rows: random bytes, then a sane fp16 scale where the block starts with one (K-quants and IQ types do; Q8_0 and
// Q4_0 too). IQ1_M's scale lives elsewhere and random bytes are still a valid block.
void fill_rows(uint8_t* p, size_t bytes, ggml_type t, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) { const uint64_t v = mix(seed); std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i)); }
    const size_t bs = ggml_type_size(t);
    if (t == GGML_TYPE_IQ1_M) return;
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-2f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(p + o, &h, 2);
        if (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K) {
            // these carry a second fp16 (dmin / d for the K scales) right after the first in some layouts: keep it sane too
            const ggml_fp16_t h2 = ggml_fp32_to_fp16(d * 0.5f);
            if (t == GGML_TYPE_Q2_K) std::memcpy(p + o + 80, &h, 2), std::memcpy(p + o + 82, &h2, 2);
            if (t == GGML_TYPE_Q3_K) std::memcpy(p + o + 108, &h, 2);
            if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K) std::memcpy(p + o + 2, &h2, 2);
        }
    }
}

struct Act { std::vector<uint8_t> q; std::vector<float> f; };

Act make_act(ggml_type t, int cols, unsigned seed) {
    const auto* traits = ggml_get_type_traits_cpu(t);
    const ggml_type vt = traits->vec_dot_type;
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    Act a; a.f.resize(cols); for (auto& v : a.f) v = nd(rng);
    a.q.resize(ggml_row_size(vt, cols));
    ggml_get_type_traits_cpu(vt)->from_float(a.f.data(), a.q.data(), cols);
    return a;
}

// The quantized activation as floats. Q8_K (the K-quant and IQ dot's activation type) has no to_float in ggml's type
// traits: decode its blocks by hand (float d, int8 qs[256], int16 bsums[16] = 292 bytes).
void act_to_float(ggml_type vt, const uint8_t* q, int cols, float* x) {
    if (vt == GGML_TYPE_Q8_K) {
        for (int b = 0; b < cols / 256; ++b) {
            const uint8_t* blk = q + (size_t) b * 292; float d; std::memcpy(&d, blk, 4);
            for (int i = 0; i < 256; ++i) x[256 * b + i] = d * (float) (int8_t) blk[4 + i];
        }
    } else ggml_get_type_traits(vt)->to_float(q, x, cols);
}

double ref_dot(ggml_type t, const uint8_t* row, int cols, const Act& a) {
    std::vector<float> w(cols), x(cols);
    ggml_get_type_traits(t)->to_float(row, w.data(), cols);
    const ggml_type vt = ggml_get_type_traits_cpu(t)->vec_dot_type;
    act_to_float(vt, a.q.data(), cols, x.data());   // the dot sees the quantized activation
    double s = 0; for (int i = 0; i < cols; ++i) s += (double) w[i] * x[i];
    return s;
}

int check(const std::vector<ggml_type>& types, int rows, int cols) {
    int failures = 0;
    for (ggml_type t : types) {
        const auto* traits = ggml_get_type_traits_cpu(t);
        if (!traits->vec_dot) { std::printf("  %-8s no vec_dot\n", ggml_type_name(t)); continue; }
        const size_t rb = ggml_row_size(t, cols);
        std::vector<uint8_t> m(rb * std::min(rows, 64)); uint64_t seed = 0x1a1500ull + t; fill_rows(m.data(), m.size(), t, seed);
        const Act a = make_act(t, cols, 7u);
        double num = 0, den = 0;
        for (int r = 0; r < std::min(rows, 64); ++r) {
            float got = 0; traits->vec_dot(cols, &got, 0, m.data() + (size_t) r * rb, 0, a.q.data(), 0, 1);
            const double ref = ref_dot(t, m.data() + (size_t) r * rb, cols, a);
            num += (got - ref) * (got - ref); den += ref * ref;
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const bool ok = rel <= 1e-4;
        std::printf("  %-8s vec_dot vs double dequant-dot: rel %.2e %s\n", ggml_type_name(t), rel, ok ? "PASS" : "FAIL");
        failures += !ok;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    int threads = omp_get_max_threads(), mb = 1024, reps = 3, rows = 2048, cols = 4096, tokens = 1; bool do_check = false;
    std::string types_arg = "q2_k,q3_k,q4_k,iq2_xxs,iq2_s,iq1_s,q8_0";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--threads") threads = std::stoi(next()); else if (a == "--mb") mb = std::stoi(next()); else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--rows") rows = std::stoi(next()); else if (a == "--cols") cols = std::stoi(next()); else if (a == "--types") types_arg = next();
        else if (a == "--tokens") tokens = std::stoi(next()); else if (a == "--check") do_check = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::vector<ggml_type> types;
    for (size_t p = 0; p < types_arg.size();) { size_t e = types_arg.find(',', p); if (e == std::string::npos) e = types_arg.size();
        auto it = kTypes.find(types_arg.substr(p, e - p)); if (it == kTypes.end()) { std::fprintf(stderr, "unknown type %s\n", types_arg.substr(p, e - p).c_str()); return 2; }
        types.push_back(it->second); p = e + 1; }
    ggml_cpu_init();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("q23_sve_bench: %s, %d threads, rows %d x cols %d, %d MB of rows per run, %d reps, %d token(s)\n",
                ggml_cpu_has_sve() ? "SVE" : (ggml_cpu_has_neon() ? "NEON" : "scalar"), threads, rows, cols, mb, reps, tokens);
    if (ggml_cpu_has_sve()) std::printf("  sve vector length %d bits\n", ggml_cpu_get_sve_cnt() * 8);
    if (do_check) { const int f = check(types, rows, cols); if (f) return 1; }
    omp_set_num_threads(threads);
    for (ggml_type t : types) {
        const auto* traits = ggml_get_type_traits_cpu(t);
        if (!traits->vec_dot) continue;
        const size_t rb = ggml_row_size(t, cols), mat = rb * rows;
        const size_t mats_per_thread = std::max<size_t>(1, (size_t) mb * 1024 * 1024 / threads / mat);
        std::vector<std::vector<uint8_t>> mem(threads);
        #pragma omp parallel num_threads(threads)
        { const int tid = omp_get_thread_num(); mem[tid].resize(mats_per_thread * mat); uint64_t seed = 0x5eed + tid + 1000 * t; fill_rows(mem[tid].data(), mem[tid].size(), t, seed); }
        std::vector<Act> acts; for (int k = 0; k < tokens; ++k) acts.push_back(make_act(t, cols, 11u + k));
        std::vector<float> sink(threads);
        double best = 1e30;
        for (int rep = 0; rep < reps; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            #pragma omp parallel num_threads(threads)
            {
                const int tid = omp_get_thread_num(); float acc = 0;
                for (size_t m = 0; m < mats_per_thread; ++m) {
                    const uint8_t* base = mem[tid].data() + m * mat;
                    for (int r = 0; r < rows; ++r) for (int k = 0; k < tokens; ++k) { float v; traits->vec_dot(cols, &v, 0, base + (size_t) r * rb, 0, acts[k].q.data(), 0, 1); acc += v; }
                }
                sink[tid] = acc;
            }
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        const double bytes = (double) threads * mats_per_thread * mat, macs = bytes / rb * cols * tokens;
        std::printf("A64FX_DOT type=%s threads=%d rows=%d cols=%d tokens=%d bytes_per_row=%zu GBps=%.1f GMACs=%.1f GBps_per_thread=%.2f ms_per_matrix=%.3f\n",
                    ggml_type_name(t), threads, rows, cols, tokens, rb, bytes / best / 1e9, macs / best / 1e9, bytes / best / 1e9 / threads,
                    best / (threads * mats_per_thread) * 1e3);
    }
    return 0;
}
