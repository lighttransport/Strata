// src/kernels/cpu/iq1s_w1a8_bench.cpp - the IQ1_S W1A8 row kernel against a double reference and against the
// engine's current decode (Fmt32<19>, muse worktree), on DeepSeek V4.1 Flash's expert shapes. No model, no GPU.
//
//     iq1s_w1a8_bench                     parity: random IQ1_S rows (every grid index, scale and delta sign) x random
//                                         activations; kernel vs double reference rel <= 1e-5, baseline vs reference too
//     iq1s_w1a8_bench --bench [--threads 15] [--mb 1024] [--reps 3] [--rows 2304 --cols 5120]
//                                         GMAC/s and GB/s of expert bytes at widths 1/2/4 for both kernels, each thread
//                                         streaming its own matrices (larger than any cache: the decode-time lane)
#include "iq1s_w1a8_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace k = strata::kernels::cpu::iq1s_w1a8;

namespace {

uint64_t mix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// random IQ1_S rows: random qs/qh (every grid index, scale and delta sign occur), a sane fp16 d
void fill_rows(uint8_t* p, size_t bytes, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) { const uint64_t v = mix(seed); std::memcpy(p + i, &v, std::min<size_t>(8, bytes - i)); }
    for (size_t o = 0; o + k::kBlockBytes <= bytes; o += k::kBlockBytes) {
        const float d = (0.5f + (float) (mix(seed) >> 40) * (1.f / 16777216.f)) * 1e-2f;
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(p + o, &h, 2);
    }
}

std::vector<k::Act> acts(int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(cols);
    for (auto& v : x) v = nd(rng);
    std::vector<k::Act> a(cols / k::kBlock);
    for (size_t i = 0; i < a.size(); ++i) k::quantize_act(x.data() + i * k::kBlock, a[i]);
    return a;
}

double reference(const uint8_t* row, int cols, const k::Act* a) {
    std::vector<float> w(cols);
    ggml_get_type_traits(GGML_TYPE_IQ1_S)->to_float(row, w.data(), cols);
    double s = 0;
    for (int i = 0; i < cols; ++i) s += (double) w[i] * (double) a[i / k::kBlock].d * (double) a[i / k::kBlock].q[i % k::kBlock];
    return s;
}

int parity(int rows, int cols) {
    const k::Table tab(iq1s_grid);
    const int nb = cols / k::kBlock;
    std::vector<uint8_t> m((size_t) rows * nb * k::kBlockBytes);
    uint64_t seed = 0x1a1500ull;
    fill_rows(m.data(), m.size(), seed);
    std::vector<std::vector<k::Act>> a;
    for (int t = 0; t < 4; ++t) a.push_back(acts(cols, 19u + t));
    const k::Act* ap[4] = {a[0].data(), a[1].data(), a[2].data(), a[3].data()};
    // the integer dots are exact; the float accumulation order is not ggml's, so compare over the whole output vector
    // (sqrt(sum diff^2 / sum ref^2), as iq1s_expert_parity does) rather than per row where a near-zero dot inflates it
    double num = 0, den = 0, num_w = 0, num_b = 0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = m.data() + (size_t) r * nb * k::kBlockBytes;
        float one[4], four[4], base[4];
        for (int t = 0; t < 4; ++t) k::row_dot<1>(tab, row, nb, &ap[t], &one[t]);
        k::row_dot<4>(tab, row, nb, ap, four);
        k::row_dot_baseline<4>(iq1s_grid, row, nb, ap, base);
        for (int t = 0; t < 4; ++t) {
            const double ref = reference(row, cols, ap[t]);
            num += (one[t] - ref) * (one[t] - ref); den += ref * ref;
            num_w += (double) (four[t] - one[t]) * (four[t] - one[t]);
            num_b += (base[t] - ref) * (base[t] - ref);
        }
    }
    const double worst = std::sqrt(num / den), worst_w = std::sqrt(num_w / den), worst_b = std::sqrt(num_b / den);
    std::printf("iq1s_w1a8 parity on %d x %d: kernel vs double ref rel %.2e, width-4 vs width-1 rel %.2e, baseline (Fmt32<19>) vs ref rel %.2e\n",
                rows, cols, worst, worst_w, worst_b);
    const bool ok = worst <= 1e-5 && worst_w <= 1e-6 && worst_b <= 1e-5;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

template <class F>
double timed(int threads, F&& fn) {
    std::vector<std::thread> ts;
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t) ts.emplace_back([&, t] { fn(t); });
    for (auto& t : ts) t.join();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int bench(int threads, int mb, int reps, int rows, int cols) {
    const k::Table tab(iq1s_grid);
    const int nb = cols / k::kBlock;
    const size_t row_bytes = (size_t) nb * k::kBlockBytes, mat_bytes = row_bytes * rows;
    const size_t mats_per_thread = std::max<size_t>(1, (size_t) mb * 1024 * 1024 / threads / mat_bytes);
    std::printf("iq1s_w1a8 bench: %d threads x %zu matrices of %d x %d (%.1f MB each, %.0f MB total), %d reps\n", threads, mats_per_thread, rows, cols,
                mat_bytes / 1e6, threads * mats_per_thread * mat_bytes / 1e6, reps);
    std::vector<std::vector<uint8_t>> mem(threads);
    timed(threads, [&](int t) { mem[t].resize(mats_per_thread * mat_bytes); uint64_t seed = 0x5eed + t; fill_rows(mem[t].data(), mem[t].size(), seed); });
    std::vector<std::vector<k::Act>> a;
    for (int t = 0; t < 4; ++t) a.push_back(acts(cols, 7u + t));
    const k::Act* ap[4] = {a[0].data(), a[1].data(), a[2].data(), a[3].data()};
    std::vector<float> sink(threads);
    for (int which = 0; which < 2; ++which) {
        const char* name = which ? "baseline Fmt32<19>" : "W1A8";
        for (int nt : {1, 2, 4}) {
            double best = 1e30;
            for (int rep = 0; rep < reps; ++rep) {
                const double s = timed(threads, [&](int t) {
                    float acc = 0, res[4];
                    for (size_t m = 0; m < mats_per_thread; ++m) {
                        const uint8_t* mat = mem[t].data() + m * mat_bytes;
                        for (int r = 0; r < rows; ++r) {
                            const uint8_t* row = mat + (size_t) r * row_bytes;
                            if (which == 0) { if (nt == 1) k::row_dot<1>(tab, row, nb, ap, res); else if (nt == 2) k::row_dot<2>(tab, row, nb, ap, res); else k::row_dot<4>(tab, row, nb, ap, res); }
                            else { if (nt == 1) k::row_dot_baseline<1>(iq1s_grid, row, nb, ap, res); else if (nt == 2) k::row_dot_baseline<2>(iq1s_grid, row, nb, ap, res); else k::row_dot_baseline<4>(iq1s_grid, row, nb, ap, res); }
                            acc += res[0];
                        }
                    }
                    sink[t] = acc;
                });
                best = std::min(best, s);
            }
            const double bytes = (double) threads * mats_per_thread * mat_bytes, macs = (double) threads * mats_per_thread * rows * cols * nt;
            std::printf("  %-20s width %d: %7.2f GB/s of expert bytes, %7.1f GMAC/s, %.3f ms per %dx%d matrix\n", name, nt, bytes / best / 1e9, macs / best / 1e9,
                        best / (threads * mats_per_thread) * 1e3, rows, cols);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int threads = 15, mb = 1024, reps = 3, rows = 2304, cols = 5120; bool run_bench = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::atoi(argv[++i]); };
        if (a == "--bench") run_bench = true; else if (a == "--threads") threads = next(); else if (a == "--mb") mb = next(); else if (a == "--reps") reps = next();
        else if (a == "--rows") rows = next(); else if (a == "--cols") cols = next(); else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (cols % k::kBlock) { std::fprintf(stderr, "cols must be a multiple of 256\n"); return 2; }
    if (!run_bench) return parity(std::min(rows, 64), cols);
    return bench(threads, mb, reps, rows, cols);
}
