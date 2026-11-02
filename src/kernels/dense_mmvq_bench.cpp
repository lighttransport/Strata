// dense_mmvq_bench - time native_mmvq on the dense projections of MiniMax M3.1 Flash (Q5_K / Q6_K / Q8_0 / Q4_K shapes) for 1..8
// activation columns, the way the engine calls it (one launch of up to 8 columns).  Random weights and activations: the
// timings do not depend on the values.  Prints microseconds per call and the weight bytes read per second.
//   dense_mmvq_bench [--iters 300] [--types 13,14] [--only q,kv,o,shg,shd,ffn,head]
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace K = strata::kernels;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

static double bytes_per_weight(int type) {
    switch (type) {
        case 12: return 144.0 / 256;   // Q4_K
        case 13: return 176.0 / 256;   // Q5_K
        case 14: return 210.0 / 256;   // Q6_K
        case 8: return 34.0 / 32;      // Q8_0
        default: return 1.0;
    }
}

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::puts("SKIP: no CUDA/HIP device"); return 77; }
    int iters = 300;
    std::string only;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--only")) only = argv[i + 1];
    }
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    struct Shape { const char* name; int type, n_in, n_out; };
    const Shape shapes[] = {
        {"q", 13, 6144, 8192},   {"kv", 14, 6144, 512},   {"o", 13, 8192, 6144},     {"shg", 13, 6144, 3072},
        {"shd", 14, 3072, 6144}, {"ffn", 13, 6144, 12288}, {"ffd", 14, 12288, 6144}, {"head", 12, 6144, 200064},
        {"q8q", 8, 6144, 8192},  {"q8g", 8, 6144, 12288},
    };
    std::printf("%-5s %-4s %6s %7s | us per call for 1..8 columns | GB/s at 1, 4, 8 columns\n", "name", "type", "n_in", "n_out");
    for (const Shape& s : shapes) {
        if (!only.empty() && only.find(s.name) == std::string::npos) continue;
        const size_t wbytes = (size_t) ((double) s.n_in * s.n_out * bytes_per_weight(s.type));
        void* w = nullptr;
        ck(cudaMalloc(&w, wbytes + 4096), "w");
        ck(cudaMemset(w, 0x21, wbytes + 4096), "w");
        float* x = nullptr;
        ck(cudaMalloc(&x, (size_t) s.n_in * 8 * 4), "x");
        std::vector<float> hx((size_t) s.n_in * 8);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((i * 2654435761u) % 2001) / 1000.f - 1.f;
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        void* q8 = nullptr;
        ck(cudaMalloc(&q8, K::native_q8_1_bytes(s.n_in * 8, 1)), "q8");
        K::native_quantize_q8_1(x, q8, s.n_in * 8, 1, st);
        float* y = nullptr;
        ck(cudaMalloc(&y, (size_t) s.n_out * 8 * 4), "y");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        double us[9] = {};
        for (int nc = 1; nc <= 8; ++nc) {
            for (int i = 0; i < 20; ++i) K::native_mmvq(s.type, w, q8, y, s.n_in, s.n_out, nc, st);
            ck(cudaDeviceSynchronize(), "warm");
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) K::native_mmvq(s.type, w, q8, y, s.n_in, s.n_out, nc, st);
            cudaEventRecord(e1, st);
            ck(cudaEventSynchronize(e1), "run");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            us[nc] = ms * 1000.0 / iters;
        }
        std::printf("%-5s %-4d %6d %7d |", s.name, s.type, s.n_in, s.n_out);
        for (int nc = 1; nc <= 8; ++nc) std::printf(" %7.1f", us[nc]);
        std::printf(" | %5.0f %5.0f %5.0f\n", wbytes / us[1] * 1e-3, wbytes / us[4] * 1e-3, wbytes / us[8] * 1e-3);
        std::fflush(stdout);
        cudaFree(w); cudaFree(x); cudaFree(q8); cudaFree(y);
    }
    return 0;
}
