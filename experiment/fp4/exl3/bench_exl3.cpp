// Synthetic K2/MCG EXL3 packed projection, using Strata's production kernels.
#include "strata/kernels/cpu/exl3/math.hpp"
#include "../common/report.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <omp.h>
#ifdef EXL3_GPU
#include "strata/kernels/exl3.hpp"
#include <cuda_runtime.h>
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
};
#endif

int main(int argc, char** argv) { try {
    bool smoke = true;
    int threads = 2;
    std::string csv_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--benchmark") smoke = false;
        else if ((arg == "--threads" || arg == "--csv") && i + 1 < argc) {
            if (arg == "--threads") threads = std::stoi(argv[++i]);
            else csv_path = argv[++i];
        } else throw std::runtime_error("usage: bench_exl3 [--benchmark] [--threads N] [--csv FILE]");
    }
    if (threads < 1 || threads > 256) throw std::runtime_error("invalid thread count");
    if (smoke && !csv_path.empty()) throw std::runtime_error("CSV requires --benchmark");
    omp_set_num_threads(threads);
    const int sets = smoke ? 1 : 64, repeats = smoke ? 1 : 20;
    std::cout << (smoke ? "SMOKE ONLY; timings are not performance evidence\n" : "BENCHMARK; machine must be idle\n")
              << "format=EXL3_K2_MCG weights=synthetic activation=fp32 transforms=included sets=" << sets << '\n';
    fp4::Csv csv(csv_path);
    for (bool down : {false, true}) {
        const int in = smoke ? 128 : (down ? 2048 : 4096);
        const int out = smoke ? 128 : (down ? 4096 : 2048);
        const size_t bytes = size_t(in) * out / 4;
        std::vector<uint8_t> packed(bytes * sets);
        std::mt19937 rng(down ? 43 : 42);
        for (auto& v : packed) v = uint8_t(rng());
        std::vector<uint16_t> si(in, 0x3c00), so(out, 0x3c00);
        std::vector<float> x(in), h(in), y(out);
        for (int i = 0; i < in; ++i) x[i] = std::sin(float(i) * .017f);
        const char* device = "cpu";
#ifdef EXL3_GPU
        device = "gpu";
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU " << prop.name << " sm_" << prop.major << prop.minor << '\n';
        Device<uint8_t> dp(packed.size());
        Device<uint16_t> dsi(in), dso(out);
        Device<float> dx(in), dh(in), dy(out);
        check(cudaMemcpy(dp.p, packed.data(), packed.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dsi.p, si.data(), in * 2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dso.p, so.data(), out * 2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx.p, x.data(), in * 4, cudaMemcpyHostToDevice));
        auto run = [&](int set) {
            strata::kernels::exl3_hadamard(dx.p, dh.p, dsi.p, in, 1, true, nullptr);
            strata::kernels::exl3_packed_gemv(dp.p + bytes * set, dh.p, dy.p, in, out, nullptr);
            strata::kernels::exl3_hadamard(dy.p, dy.p, dso.p, out, 1, false, nullptr);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
        };
#else
        // Output-tile-major packed layout, identical to the CUDA kernel.
        auto run = [&](int set) {
            for (int i = 0; i < in; ++i) h[i] = x[i] * strata::cpu::exl3::half_value(si[i]);
            strata::cpu::exl3::hadamard(h.data(), in);
            std::fill(y.begin(), y.end(), 0.f);
#pragma omp parallel for schedule(static)
            for (int ot = 0; ot < out / 16; ++ot)
                for (int it = 0; it < in / 16; ++it)
                    strata::cpu::exl3::tile_dot(packed.data() + bytes * set + (size_t(ot) * (in / 16) + it) * 64,
                                              h.data() + it * 16, y.data() + ot * 16);
            strata::cpu::exl3::hadamard(y.data(), out);
            for (int i = 0; i < out; ++i) y[i] *= strata::cpu::exl3::half_value(so[i]);
        };
#endif
        run(0);
        std::vector<double> times;
        for (int r = 0; r < repeats; ++r) {
            double start = fp4::now_s();
            for (int set = 0; set < sets; ++set) run(set);
            times.push_back((fp4::now_s() - start) * 1e6 / sets);
        }
#ifdef EXL3_GPU
        check(cudaMemcpy(y.data(), dy.p, out * 4, cudaMemcpyDeviceToHost));
#endif
        double checksum = 0;
        for (float v : y) {
            if (!std::isfinite(v)) throw std::runtime_error("nonfinite EXL3 output");
            checksum += v;
        }
        auto stats = fp4::stats_of(times);
        std::cout << device << " projection=" << (down ? "down" : "gate_or_up")
                  << " in=" << in << " out=" << out << " packed_bytes=" << bytes
                  << " us_med=" << stats.med << " checksum=" << checksum << '\n';
        // No numerical reference: mark error columns unavailable, not zero.
        if (!smoke) csv.row(device, "exl3_k2_mcg_packed", down ? "down" : "gate_or_up", "synthetic", 1,
                           std::string(device) == "cpu" ? "t" + std::to_string(threads) : "sm86",
                           stats, bytes, 0, 2.0 * in * out, NAN, NAN, 0);
    }
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
