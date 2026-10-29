// Q6_K vs Q4_K vs Q5_K native_mmvq timing at the verify batch widths on the decoder's dense shapes (random weights, timing only).
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
int main() {
    struct T { const char* n; int type, bytes; } types[] = {{"Q6_K", 14, 210}, {"Q5_K", 13, 176}, {"Q4_K", 12, 144}};
    struct S { int n_in, n_out; } shapes[] = {{5120, 8192}, {4096, 5120}, {5120, 2304}};
    cudaStream_t s; cudaStreamCreate(&s);
    for (auto sh : shapes) for (int cols : {1, 3, 4}) for (auto t : types) {
        size_t wb = size_t(sh.n_in / 256) * t.bytes * sh.n_out, xb = size_t(sh.n_in / 32) * 36 * cols * 2;
        void *w, *x; float* y;
        cudaMalloc(&w, wb); cudaMalloc(&x, xb); cudaMalloc(&y, size_t(sh.n_out) * cols * 4);
        cudaMemset(w, 0x11, wb); cudaMemset(x, 0, xb);
        for (int i = 0; i < 20; ++i) strata::kernels::native_mmvq(t.type, w, x, y, sh.n_in, sh.n_out, cols, s);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a, s);
        for (int i = 0; i < 200; ++i) strata::kernels::native_mmvq(t.type, w, x, y, sh.n_in, sh.n_out, cols, s);
        cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
        printf("%dx%d cols=%d %s %.1f us  %.0f GB/s\n", sh.n_in, sh.n_out, cols, t.n, ms * 1e3 / 200, wb / (ms / 200 * 1e6));
        cudaFree(w); cudaFree(x); cudaFree(y);
    }
}
