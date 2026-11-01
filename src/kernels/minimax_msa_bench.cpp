// minimax_msa_bench - time the three MSA decode kernels (block scores, block selection, sparse attention) at the M3.1
// geometry: 1 query, 4 index / kv heads x 128, 64 query heads, 128-token blocks, top-16, for a given number of cached
// positions.  Prints microseconds per kernel per call (CUDA events, 200 calls after warm-up).
//   minimax_msa_bench [n_pos ...]      default 270 2048 7000
#include "strata/kernels/minimax_msa.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(1); }
}

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::puts("SKIP: no CUDA/HIP device"); return 77; }
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {270, 2048, 7000};
    const int IH = 4, D = 128, QH = 64, BS = 128, TOPK_MAX = 16, LOCAL = 1;
    for (int n_pos : sizes) {
        const int nblk = (n_pos + BS - 1) / BS, topk = nblk < TOPK_MAX ? nblk : TOPK_MAX;
        std::vector<float> hq((size_t) IH * D), hk((size_t) n_pos * D), hqa((size_t) QH * D), hka((size_t) n_pos * IH * D), hv((size_t) n_pos * IH * D);
        unsigned seed = 7;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float) ((seed >> 8) & 0xffff) / 32768.0f - 1.0f; };
        for (auto* v : {&hq, &hk, &hqa, &hka, &hv}) for (float& x : *v) x = rnd();
        std::vector<int32_t> kpos(n_pos);
        for (int i = 0; i < n_pos; ++i) kpos[i] = i;
        const int32_t qpos = n_pos - 1;
        float *dq, *dk, *dqa, *dka, *dv, *ds, *dout;
        int32_t *dkp, *dqp, *dids;
        const size_t sstride = nblk, istride = topk;
        ck(cudaMalloc(&dq, hq.size() * 4), "m"); ck(cudaMalloc(&dk, hk.size() * 4), "m"); ck(cudaMalloc(&dqa, hqa.size() * 4), "m");
        ck(cudaMalloc(&dka, hka.size() * 4), "m"); ck(cudaMalloc(&dv, hv.size() * 4), "m"); ck(cudaMalloc(&ds, IH * sstride * 4), "m");
        ck(cudaMalloc(&dout, (size_t) QH * D * 4), "m"); ck(cudaMalloc(&dkp, n_pos * 4), "m"); ck(cudaMalloc(&dqp, 4), "m");
        ck(cudaMalloc(&dids, IH * istride * 4), "m");
        ck(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "c"); ck(cudaMemcpy(dk, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(dqa, hqa.data(), hqa.size() * 4, cudaMemcpyHostToDevice), "c"); ck(cudaMemcpy(dka, hka.data(), hka.size() * 4, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(dv, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "c"); ck(cudaMemcpy(dkp, kpos.data(), n_pos * 4, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(dqp, &qpos, 4, cudaMemcpyHostToDevice), "c");
        cudaEvent_t e0, e1, e2, e3;
        for (auto* e : {&e0, &e1, &e2, &e3}) ck(cudaEventCreate(e), "e");
        float t_score = 0, t_select = 0, t_attn = 0;
        const int reps = 200;
        for (int r = -20; r < reps; ++r) {
            ck(cudaEventRecord(e0), "r");
            strata::kernels::minimax_msa_block_scores(dq, dk, dqp, dkp, 1, IH, D, n_pos, BS, nblk, ds, sstride, nullptr);
            ck(cudaEventRecord(e1), "r");
            strata::kernels::minimax_msa_select_blocks(ds, dqp, 1, IH, nblk, BS, topk, LOCAL, sstride, dids, istride, nullptr);
            ck(cudaEventRecord(e2), "r");
            strata::kernels::minimax_msa_attention_f32(dqa, dka, dv, dqp, dkp, dids, 1, QH, IH, D, n_pos, BS, topk, istride, dout, nullptr);
            ck(cudaEventRecord(e3), "r");
            ck(cudaEventSynchronize(e3), "s");
            float a, b, c;
            ck(cudaEventElapsedTime(&a, e0, e1), "t"); ck(cudaEventElapsedTime(&b, e1, e2), "t"); ck(cudaEventElapsedTime(&c, e2, e3), "t");
            if (r >= 0) { t_score += a; t_select += b; t_attn += c; }
        }
        std::printf("n_pos %5d (%2d blocks, top-%2d): scores %7.1f us   select %7.1f us   attention %7.1f us   total %7.1f us\n", n_pos,
                    nblk, topk, 1000 * t_score / reps, 1000 * t_select / reps, 1000 * t_attn / reps, 1000 * (t_score + t_select + t_attn) / reps);
        cudaFree(dq); cudaFree(dk); cudaFree(dqa); cudaFree(dka); cudaFree(dv); cudaFree(ds); cudaFree(dout); cudaFree(dkp); cudaFree(dqp); cudaFree(dids);
    }
    return 0;
}
