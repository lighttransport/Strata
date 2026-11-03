// tc_mmvq_test - native_tc_mmvq (weights-once tensor-core matvec) against native_mmvq (q8_1 dp4a) on MiniMax dense shapes, and its time against 1..64
// columns.  Random but well-formed Q4_K / Q5_K / Q6_K blocks.  Prints max |difference| relative to the largest |reference| value and microseconds a call.
//   tc_mmvq_test [--iters 100] [--only q,kv,o,shg,shd,head]
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_tc_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace K = strata::kernels;
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); } }

static int block_bytes(int t) { return t == 12 ? 144 : t == 13 ? 176 : 210; }

static void fill_blocks(std::vector<uint8_t>& w, int type, size_t nblocks) {
    unsigned s = 12345u;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
    const int bb = block_bytes(type);
    for (size_t b = 0; b < nblocks; ++b) {
        uint8_t* p = w.data() + b * bb;
        for (int i = 0; i < bb; ++i) p[i] = (uint8_t) (rnd() & 255);
        const __half d = __float2half(0.0004f * (1 + rnd() % 4)), m = __float2half(0.0003f * (1 + rnd() % 4));
        if (type == 14) { std::memcpy(p + 208, &d, 2); for (int i = 0; i < 16; ++i) p[192 + i] = (uint8_t) (int8_t) ((int) (rnd() % 41) - 20); }
        else { std::memcpy(p, &d, 2); std::memcpy(p + 2, &m, 2); }
    }
}

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) { std::puts("SKIP: no CUDA device"); return 77; }
    int iters = 100;
    std::string only;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--only")) only = argv[i + 1];
    }
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    struct Shape { const char* name; int type, n_in, n_out; };
    const Shape shapes[] = {{"q", 13, 6144, 8192}, {"kv", 14, 6144, 512}, {"o", 13, 8192, 6144}, {"shg", 13, 6144, 3072},
                            {"shd", 14, 3072, 6144}, {"head", 12, 6144, 200064}};
    int bad = 0;
    for (const Shape& sh : shapes) {
        if (!only.empty() && only.find(sh.name) == std::string::npos) continue;
        const size_t nb = (size_t) sh.n_out * (sh.n_in / 256);
        std::vector<uint8_t> hw(nb * block_bytes(sh.type) + 256, 0);
        fill_blocks(hw, sh.type, nb);
        void* w = nullptr;
        ck(cudaMalloc(&w, hw.size()), "w");
        ck(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "w");
        const int maxc = 64;
        std::vector<float> hx((size_t) maxc * sh.n_in);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) (((i * 2654435761u) >> 7) % 2001) / 1000.f - 1.f;
        float *x = nullptr, *yref = nullptr, *ytc = nullptr;
        void *x16 = nullptr, *q8 = nullptr;
        ck(cudaMalloc(&x, hx.size() * 4), "x");
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        ck(cudaMalloc(&x16, hx.size() * 2), "x16");
        ck(cudaMemset(x16, 0, hx.size() * 2), "x16");
        K::native_f32_to_f16(x, x16, (int64_t) hx.size(), st);
        ck(cudaMalloc(&q8, K::native_q8_1_bytes(sh.n_in * maxc, 1)), "q8");
        ck(cudaMalloc(&yref, (size_t) maxc * sh.n_out * 4), "yref");
        ck(cudaMalloc(&ytc, (size_t) maxc * sh.n_out * 4), "ytc");
        K::native_quantize_q8_1(x, q8, sh.n_in * maxc, 1, st);   // the columns back to back
        std::printf("%-5s type %d %6d x %6d | cols: max rel err vs the q8_1 kernel, tc us (q8_1 kernel us)\n", sh.name, sh.type, sh.n_in, sh.n_out);
        const double qb = (double) K::native_q8_1_bytes(sh.n_in, 1);
        for (int cols : {1, 4, 8, 9, 16, 24, 32, 48, 64}) {
            // reference: the q8_1 kernel, 8 columns a launch
            for (int c0 = 0; c0 < cols; c0 += 8) {
                const int nc = std::min(8, cols - c0);
                K::native_mmvq(sh.type, w, (const char*) q8 + (size_t) (c0 * qb), yref + (size_t) c0 * sh.n_out, sh.n_in, sh.n_out, nc, st);
            }
            K::native_tc_mmvq(sh.type, w, x16, ytc, sh.n_in, sh.n_out, cols, st);
            ck(cudaStreamSynchronize(st), "run");
            std::vector<float> a((size_t) cols * sh.n_out), b((size_t) cols * sh.n_out);
            ck(cudaMemcpy(a.data(), yref, a.size() * 4, cudaMemcpyDeviceToHost), "a");
            ck(cudaMemcpy(b.data(), ytc, b.size() * 4, cudaMemcpyDeviceToHost), "b");
            double mx = 0, md = 0, rms = 0;
            for (size_t i = 0; i < a.size(); ++i) { mx = std::max(mx, (double) std::fabs(a[i])); md = std::max(md, (double) std::fabs(a[i] - b[i])); rms += (double) a[i] * a[i]; }
            rms = std::sqrt(rms / (double) a.size());
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i) K::native_tc_mmvq(sh.type, w, x16, ytc, sh.n_in, sh.n_out, cols, st);
            cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            const double tc_us = ms * 1000.0 / iters;
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i)
                for (int c0 = 0; c0 < cols; c0 += 8) {
                    const int nc = std::min(8, cols - c0);
                    K::native_mmvq(sh.type, w, (const char*) q8 + (size_t) (c0 * qb), yref + (size_t) c0 * sh.n_out, sh.n_in, sh.n_out, nc, st);
                }
            cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1);
            const double q_us = ms * 1000.0 / iters;
            const double rel = md / std::max(1e-30, mx);
            if (rel > 0.03) ++bad;
            std::printf("  %2d cols: max err %.2e of max %.2e (%.2f of rms) | tc %7.1f us (%6.1f GB/s)  q8_1 %7.1f us\n", cols, md, mx, md / std::max(1e-30, rms),
                        tc_us, (double) nb * block_bytes(sh.type) / tc_us / 1e3, q_us);
        }
        cudaFree(w); cudaFree(x); cudaFree(x16); cudaFree(q8); cudaFree(yref); cudaFree(ytc);
    }
    std::puts(bad ? "FAIL: a tensor-core result is far from the q8_1 kernel" : "OK");
    return bad ? 1 : 0;
}
