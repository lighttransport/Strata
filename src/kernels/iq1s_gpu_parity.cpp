// src/kernels/iq1s_gpu_parity.cpp - the GPU's IQ1_S mat-vec (iq_mmvq type 19, the dot the grouped expert kernels use)
// against ggml-cpu's IQ1_S dequantizer and the GPU's own q8_1 activations, in double: what is left is float summation
// order. Muse Spark's expert shapes (4096 wide), 1..4 columns, random grid codes and fp16 scales. A ctest; needs a GPU.
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { std::printf("FAIL %s: %s\n", w, cudaGetErrorString(e)); std::exit(1); } }

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { std::printf("no GPU\n"); return 77; }
    const int n_in = 4096, n_out = 1024, ncols = 4;
    std::mt19937 rng(19);
    const size_t row = k::iq_row_bytes(GGML_TYPE_IQ1_S, n_in);
    std::vector<uint8_t> w((size_t) n_out * row);
    for (auto& v : w) v = (uint8_t) rng();
    for (size_t o = 0; o < w.size(); o += 50) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.005f + 0.02f * (rng() % 1000) / 1000.f);
        std::memcpy(&w[o], &d, 2);
    }
    std::vector<float> x((size_t) ncols * n_in);
    std::normal_distribution<float> nd;
    for (auto& v : x) v = nd(rng);
    // CPU reference weights: ggml-cpu's dequantizer (independent of the GPU code)
    std::vector<float> wf((size_t) n_out * n_in);
    ggml_get_type_traits(GGML_TYPE_IQ1_S)->to_float(w.data(), wf.data(), (int64_t) wf.size());
    void *dw, *dq;
    float *dx, *dy;
    const size_t q8_bytes = (size_t) ncols * (n_in / 32) * 36;
    ck(cudaMalloc(&dw, w.size()), "malloc"); ck(cudaMalloc(&dq, q8_bytes), "malloc");
    ck(cudaMalloc(&dx, x.size() * 4), "malloc"); ck(cudaMalloc(&dy, (size_t) ncols * n_out * 4), "malloc");
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "upload");
    k::quantize_q8_1_rows(dx, ncols, n_in, dq, nullptr);
    // the activations as the GPU quantized them: block_q8_1 = {half2 ds, int8 qs[32]}
    std::vector<uint8_t> q8(q8_bytes);
    ck(cudaMemcpy(q8.data(), dq, q8_bytes, cudaMemcpyDeviceToHost), "download");
    int fails = 0;
    for (int nc = 1; nc <= ncols; ++nc) {
        k::iq_mmvq(GGML_TYPE_IQ1_S, dw, dq, dy, n_in, n_out, nc, nullptr);
        ck(cudaDeviceSynchronize(), "mmvq");
        std::vector<float> y((size_t) nc * n_out);
        ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "download");
        double err = 0, scale = 0;
        for (int c = 0; c < nc; ++c) {
            std::vector<double> a(n_in);
            for (int b = 0; b < n_in / 32; ++b) {
                const uint8_t* blk = q8.data() + ((size_t) c * (n_in / 32) + b) * 36;
                ggml_fp16_t dh;
                std::memcpy(&dh, blk, 2);
                const double d = ggml_fp16_to_fp32(dh);
                for (int i = 0; i < 32; ++i) a[(size_t) b * 32 + i] = d * (int8_t) blk[4 + i];
            }
            for (int r = 0; r < n_out; ++r) {
                double ref = 0;
                for (int i = 0; i < n_in; ++i) ref += (double) wf[(size_t) r * n_in + i] * a[(size_t) i];
                err = std::max(err, std::fabs(y[(size_t) c * n_out + r] - ref));
                scale = std::max(scale, std::fabs(ref));
            }
        }
        const bool ok = err / scale < 1e-5;
        std::printf("IQ1_S mmvq %d x %d, %d column(s): max error %.2e of max %.3g (rel %.2e)  %s\n", n_out, n_in, nc, err, scale,
                    err / scale, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    std::printf("iq1s_gpu_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
