// attn_bench: the batched prefill attention of the decoder (tiled FP32 kernel) against the tensor-core kernel (attn_mma_kernel): error and time, FP32 and FP16
// compressed caches. Exits 1 when the relative L2 error exceeds 2e-3 (FP16 rounding of Q, K/V and P gives ~3e-4) or the output is not finite.
// Built by including the kernel file, so the anonymous-namespace kernels are in this translation unit:
//   nvcc -O3 -std=c++20 -arch=sm_80 (or sm_120) -I<src>/include -I<src>/third_party/ggml attn_bench.cu -o attn_bench
//   ./attn_bench [TOKENS (256)] [SELECTED_ROWS (512)]
#include "../../../src/kernels/cuda/ds41.cu"
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::kernels;

static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); } }

int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 256, heads = 64, dim = 512, ring = 136, C = 4096, nsel = argc > 2 ? std::atoi(argv[2]) : 512, wrows = 128;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(T) * heads * dim), win(size_t(ring) * dim), comp(size_t(C) * dim), sinks(heads);
    for (auto& x : q) x = 0.35f * nd(rng);
    for (auto& x : win) x = 0.5f * nd(rng);
    for (auto& x : comp) x = 0.5f * nd(rng);
    for (auto& x : sinks) x = 0.5f * nd(rng);
    std::vector<int> sel(size_t(T) * 512, 0);
    std::vector<int4> info(T);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < nsel; ++i) sel[size_t(t) * 512 + i] = int((size_t(t) * 7 + size_t(i) * 7 + (rng() % 5)) % C);   // neighbouring tokens select similar rows
        info[t] = make_int4(t % ring, wrows, nsel, 0);
    }
    std::vector<uint16_t> comp16(comp.size());
    for (size_t i = 0; i < comp.size(); ++i) comp16[i] = __half_as_ushort(__float2half(comp[i]));
    float *dq, *dwin, *dcomp, *dsinks, *dout, *dout2;
    uint16_t* dcomp16;
    int *dsel;
    int4* dinfo;
    ck(cudaMalloc(&dq, q.size() * 4), "q"); ck(cudaMalloc(&dwin, win.size() * 4), "w"); ck(cudaMalloc(&dcomp, comp.size() * 4), "c");
    ck(cudaMalloc(&dcomp16, comp16.size() * 2), "c16"); ck(cudaMalloc(&dsinks, heads * 4), "s"); ck(cudaMalloc(&dsel, sel.size() * 4), "sel");
    ck(cudaMalloc(&dinfo, T * sizeof(int4)), "info"); ck(cudaMalloc(&dout, q.size() * 4), "o"); ck(cudaMalloc(&dout2, q.size() * 4), "o2");
    ck(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q"); ck(cudaMemcpy(dwin, win.data(), win.size() * 4, cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(dcomp, comp.data(), comp.size() * 4, cudaMemcpyHostToDevice), "c"); ck(cudaMemcpy(dcomp16, comp16.data(), comp16.size() * 2, cudaMemcpyHostToDevice), "c16");
    ck(cudaMemcpy(dsinks, sinks.data(), heads * 4, cudaMemcpyHostToDevice), "s"); ck(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice), "sel");
    ck(cudaMemcpy(dinfo, info.data(), T * sizeof(int4), cudaMemcpyHostToDevice), "info");
    const float scale = 1.0f / std::sqrt(float(dim));
    const size_t tiled_shared = size_t(kTG * kTileDim + kTR * kDCP + kTR * kTG) * sizeof(float);
    cudaFuncSetAttribute(attn_tiled_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tiled_shared));
    cudaFuncSetAttribute(attn_tiled_kernel<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tiled_shared));
    cudaFuncSetAttribute(attn_mma_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kMmaShared));
    cudaFuncSetAttribute(attn_mma_kernel<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kMmaShared));
    {
        cudaFuncAttributes fa{};
        cudaFuncGetAttributes(&fa, attn_mma_kernel<float>);
        int blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, attn_mma_kernel<float>, 256, kMmaShared);
        std::printf("attn_mma_kernel<float>: %d regs, %zu B local, static smem %zu B + dynamic %zu B, %d blocks/SM\n", fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes, kMmaShared, blocks);
        cudaFuncGetAttributes(&fa, attn_tiled_kernel<float>);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, attn_tiled_kernel<float>, 256, tiled_shared);
        std::printf("attn_tiled_kernel<float>: %d regs, %d blocks/SM\n", fa.numRegs, blocks);
        cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0);
        std::printf("%s: %d SMs, smem/SM %zu KB, L2 %d MB\n", pr.name, pr.multiProcessorCount, pr.sharedMemPerMultiprocessor / 1024, pr.l2CacheSize >> 20);
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time_ms = [&](auto&& launch) {
        for (int i = 0; i < 3; ++i) launch();
        cudaEventRecord(e0);
        const int reps = 10;
        for (int i = 0; i < reps; ++i) launch();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / reps;
    };
    int failed = 0;
    for (int half = 0; half < 2; ++half) {
        auto tiled = [&] {
            if (half) attn_tiled_kernel<uint16_t><<<dim3(T, heads / kTG), 256, tiled_shared>>>(dq, heads, dwin, ring, dinfo, dcomp16, dim, dsel, 512, dsinks, scale, dout);
            else attn_tiled_kernel<float><<<dim3(T, heads / kTG), 256, tiled_shared>>>(dq, heads, dwin, ring, dinfo, dcomp, dim, dsel, 512, dsinks, scale, dout);
        };
        auto mma = [&] {
            if (half) attn_mma_kernel<uint16_t><<<dim3(T, heads / kMG), 256, kMmaShared>>>(dq, heads, dwin, ring, dinfo, dcomp16, dim, dsel, 512, dsinks, scale, dout2);
            else attn_mma_kernel<float><<<dim3(T, heads / kMG), 256, kMmaShared>>>(dq, heads, dwin, ring, dinfo, dcomp, dim, dsel, 512, dsinks, scale, dout2);
        };
        const float a = time_ms(tiled), b = time_ms(mma);
        ck(cudaGetLastError(), "kernel");
        std::vector<float> o1(q.size()), o2(q.size());
        cudaMemcpy(o1.data(), dout, o1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o2.data(), dout2, o2.size() * 4, cudaMemcpyDeviceToHost);
        double num = 0, den = 0, mx = 0;
        for (size_t i = 0; i < o1.size(); ++i) { const double d = o1[i] - o2[i]; num += d * d; den += double(o1[i]) * o1[i]; mx = std::max(mx, std::fabs(d)); }
        const double rel = std::sqrt(num / den);
        const bool ok_run = std::isfinite(rel) && rel < 2e-3;
        failed += !ok_run;
        std::printf("comp=%s T=%d nsel=%d: tiled %.3f ms (%.1f us/token)  mma %.3f ms (%.1f us/token)  speedup %.2fx   rel L2 err %.3e  max abs err %.3e  %s\n", half ? "fp16" : "fp32", T, nsel,
                    a, a * 1e3 / T, b, b * 1e3 / T, a / b, rel, mx, ok_run ? "PASS" : "FAIL");
    }
    return failed ? 1 : 0;
}
