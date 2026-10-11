// Register-resident AVX2/FMA throughput and dependent scalar FMA latency.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <vector>
using Clock = std::chrono::steady_clock;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const int threads = std::stoi(argv[1]);
    if (threads < 1 || threads > omp_get_num_procs()) return 2;
    omp_set_dynamic(0); omp_set_num_threads(threads);
    constexpr int iterations = 1000000;
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<double> times;
        bool valid = true;
        for (int rep = 0; rep < 8; ++rep) {
            auto start = Clock::now();
            int bad = 0, actual = 0;
            #pragma omp parallel reduction(+:bad)
            {
                #pragma omp single
                actual = omp_get_num_threads();
                if (mode == 0) {
                    // Exact integer sequence prevents a tolerance from hiding corruption.
                    __m128 x = _mm_set_ss(0), one = _mm_set_ss(1);
                    for (int i = 0; i < iterations; ++i) x = _mm_fmadd_ss(x, one, one);
                    if (_mm_cvtss_f32(x) != float(iterations)) ++bad;
                } else {
                    __m256 x[8]; for (auto& v : x) v = _mm256_setzero_ps();
                    const __m256 one = _mm256_set1_ps(1);
                    for (int i = 0; i < iterations; ++i) {
                        for (auto& v : x) v = _mm256_fmadd_ps(v, one, one);
                        // Force all accumulators to remain independent and observable.
                        asm volatile("" : "+x"(x[0]), "+x"(x[1]), "+x"(x[2]), "+x"(x[3]),
                                          "+x"(x[4]), "+x"(x[5]), "+x"(x[6]), "+x"(x[7]));
                    }
                    alignas(32) float values[8];
                    for (auto v : x) {
                        _mm256_store_ps(values, v);
                        for (float value : values) if (value != float(iterations)) ++bad;
                    }
                }
            }
            valid &= bad == 0 && actual == threads;
            if (rep) times.push_back(std::chrono::duration<double>(Clock::now()-start).count());
        }
        std::sort(times.begin(), times.end());
        const double ops = double(iterations) * threads * (mode ? 128 : 2);
        std::cout << "{\"kernel\":\"" << (mode ? "avx2_fma_throughput" : "dependent_fma")
                  << "\",\"threads\":" << threads << ",\"iterations\":" << iterations
                  << ",\"median_seconds\":" << times[3] << ",\"best_seconds\":" << times.front()
                  << ",\"worst_seconds\":" << times.back() << ",\"GFLOP_s\":" << ops/times[3]/1e9
                  << ",\"timed_repeats\":7,\"warmups\":1"
                  << (mode ? "" : ",\"chain_ns_per_fma\":" + std::to_string(times[3]*1e9/iterations))
                  << ",\"verified\":" << (valid ? "true" : "false") << "}\n";
        if (!valid) return 3;
    }
}
