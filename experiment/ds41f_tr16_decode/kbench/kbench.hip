// kbench: the canonical IQ dot (reference = the decoder's kernel) against a candidate (cand.cuh), bitwise, timed
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/dp4a.hpp"
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
constexpr int kCanonRows = 8;
__global__ void ref_kernel(int type, const uint8_t* const* wptr, const int* gstart, const int* eact,
                                      const block_q8_K* act, float* out, size_t part_out, int rows, int nb,
                                      size_t row_bytes, size_t part_stride) {
    __shared__ int sums[kCanonRows][20 * 8];
    __shared__ float scl[kCanonRows][20];   // a row's block scales (weight d / 8 x activation d), in parallel
    __shared__ float lanes[kCanonRows][8];
    __shared__ int xs[20 * 64];      // the entry's Q8_K values (nb blocks x 256 int8), staged once for the block's rows
    __shared__ float xd[20];         // and its block scales
    const int w = threadIdx.x / 32, t = threadIdx.x % 32, g = blockIdx.y, part = blockIdx.z;
    const int row = blockIdx.x * kCanonRows + w;
    const bool live = row < rows;
    const uint8_t* wr = wptr[g] + size_t(part) * part_stride + size_t(live ? row : 0) * row_bytes;
    float* o = out + size_t(part) * part_out;
    const int stride = type == 19 ? 50 : 66;
    for (int e = gstart[g]; e < gstart[g + 1]; ++e) {
        const block_q8_K* a = act + size_t(eact[e]) * nb;
        for (int i = threadIdx.x; i < nb * 64; i += blockDim.x)
            xs[i] = reinterpret_cast<const int*>(a[i / 64].qs)[i % 64];
        if (threadIdx.x < nb) xd[threadIdx.x] = a[threadIdx.x].d;
        __syncthreads();
        if (live)
#pragma unroll 2
            for (int cell = t; cell < nb * 8; cell += 32) {
                const int block = cell / 8, lane = cell % 8, gi = lane / 2;
                const uint8_t* p = wr + size_t(block) * stride;
                const int* xq = xs + block * 64;
                int sum = 0;
                if (type == 19) {
                    // every index first (independent loads in flight), then the grid, then the dots
                    unsigned hh[8], ix[8], gr[8];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) hh[ib] = unsigned(p[34 + 2*ib]) | (unsigned(p[35 + 2*ib]) << 8);
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) ix[ib] = p[2 + 4*ib + gi] | (((hh[ib] >> (3*gi)) & 7) << 8);
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) gr[ib] = iq1s_grid_gpu[ix[ib]];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) {
                        // 4-way integer dots: w = 8 (g - 1) + delta, so sum w x = 8 sum g x - 8 sum x + delta sum x
                        const int xw = xq[8*ib + lane];
                        const int g4 = int((gr[ib] >> (4 * (lane % 2))) & 0x0F0F0F0Fu);
                        const int dg = STRATA_DP4A(g4, xw, 0), dx = STRATA_DP4A(0x01010101, xw, 0);
                        sum += (8 * (dg - dx) + ((hh[ib] >> 15) ? -dx : dx)) * (2 * int((hh[ib] >> 12) & 7) + 1);
                    }
                } else {
                    unsigned hw[8], qi[8];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) {
                        const uint8_t* q = p + 2 + 8*ib;
                        hw[ib] = unsigned(q[4]) | (unsigned(q[5]) << 8) | (unsigned(q[6]) << 16) | (unsigned(q[7]) << 24);
                        qi[ib] = q[gi];
                    }
                    uint64_t gv[8];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) gv[ib] = iq2xxs_grid[qi[ib]];
#pragma unroll
                    for (int ib = 0; ib < 8; ++ib) {
                        // w = +-grid byte (8..43): the sign folded per byte, (g ^ 0xff) + 1 = -g carries out of no byte
                        const int xw = xq[8*ib + lane];
                        unsigned sg = (hw[ib] >> (7*gi)) & 127; sg |= (__popc(sg) & 1) << 7;
                        const unsigned s4 = (sg >> (4 * (lane % 2))) & 15u;
                        const unsigned tt = (s4 * 0x00204081u) & 0x01010101u, m = tt * 0xffu;
                        const unsigned grid = unsigned(gv[ib] >> (32 * (lane % 2)));
                        sum += STRATA_DP4A(int((grid ^ m) + tt), xw, 0) * (2 * int(hw[ib] >> 28) + 1);
                    }
                }
                sums[w][cell] = sum;
                if (lane == 0) {
                    const unsigned short bits = unsigned(p[0]) | (unsigned(p[1]) << 8);
                    const __half hd = *reinterpret_cast<const __half*>(&bits);
                    scl[w][block] = __fmul_rn(__fmul_rn(__half2float(hd), 0.125f), xd[block]);
                }
            }
        __syncthreads();
        if (live && t < 8) {
            float acc = 0.f;
            for (int b = 0; b < nb; ++b) acc = fmaf(scl[w][b], float(sums[w][8*b + t]), acc);
            lanes[w][t] = acc;
        }
        __syncthreads();
        if (live && t == 0) {
            const float* L = lanes[w];
            const float p0 = __fadd_rn(L[0],L[4]), p1 = __fadd_rn(L[1],L[5]);
            const float p2 = __fadd_rn(L[2],L[6]), p3 = __fadd_rn(L[3],L[7]);
            o[size_t(e) * rows + row] = __fadd_rn(__fadd_rn(p0,p1), __fadd_rn(p2,p3));
        }
        __syncthreads();
    }
}


#include "cand.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
int run(int type, int rows, int nb, int parts, int G, int per, int T, int iters) {
    const int stride = type == 19 ? 50 : 66;
    const size_t row_bytes = size_t(nb) * stride, mat = size_t(rows) * row_bytes, expert = mat * parts;
    const int E = G * per;
    std::mt19937 rng(1234);
    std::vector<uint8_t> w(expert * G);
    for (auto& b : w) b = uint8_t(rng());
    for (size_t r = 0; r < size_t(rows) * parts * G; ++r)
        for (int b = 0; b < nb; ++b) { uint16_t d = 0x2000 + (rng() & 0x0fff); std::memcpy(&w[r * row_bytes + size_t(b) * stride], &d, 2); }
    std::vector<block_q8_K> act(size_t(T) * nb);
    for (auto& a : act) { a.d = 0.01f + (rng() % 100) * 1e-4f; for (auto& q : a.qs) q = int8_t(int(rng() % 255) - 127); for (auto& s : a.bsums) s = 0; }
    std::vector<uint64_t> wp(G); std::vector<int> gs(G + 1), ea(E);
    uint8_t* dw; CK(cudaMalloc(&dw, w.size())); CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    for (int g = 0; g < G; ++g) { wp[g] = uint64_t(uintptr_t(dw + size_t(g) * expert)); gs[g] = g * per; for (int k = 0; k < per; ++k) ea[g * per + k] = (g + k) % T; }
    gs[G] = E;
    uint64_t* dwp; int *dgs, *dea; block_q8_K* dact; float *o1, *o2;
    CK(cudaMalloc(&dwp, G * 8)); CK(cudaMalloc(&dgs, (G + 1) * 4)); CK(cudaMalloc(&dea, E * 4)); CK(cudaMalloc(&dact, act.size() * sizeof(block_q8_K)));
    const size_t outn = size_t(parts) * E * rows;
    CK(cudaMalloc(&o1, outn * 4)); CK(cudaMalloc(&o2, outn * 4));
    CK(cudaMemcpy(dwp, wp.data(), G * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dgs, gs.data(), (G + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dea, ea.data(), E * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dact, act.data(), act.size() * sizeof(block_q8_K), cudaMemcpyHostToDevice));
    auto ref = [&] { ref_kernel<<<dim3((rows + kCanonRows - 1) / kCanonRows, G, parts), 32 * kCanonRows>>>(type, (const uint8_t* const*)dwp, dgs, dea, dact, o1, size_t(E) * rows, rows, nb, row_bytes, mat); };
    auto cand = [&] { cand_launch(type, (const uint8_t* const*)dwp, dgs, dea, dact, o2, size_t(E) * rows, rows, nb, row_bytes, mat, G, parts, 0); };
    cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
    float tr = 0, tc = 0;
    for (int pass = 0; pass < 2; ++pass) {
        ref(); cand(); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a0)); for (int i = 0; i < iters; ++i) ref(); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); CK(cudaEventElapsedTime(&tr, a0, a1));
        CK(cudaEventRecord(a0)); for (int i = 0; i < iters; ++i) cand(); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); CK(cudaEventElapsedTime(&tc, a0, a1));
    }
    std::vector<float> h1(outn), h2(outn);
    CK(cudaMemcpy(h1.data(), o1, outn * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2.data(), o2, outn * 4, cudaMemcpyDeviceToHost));
    size_t diff = 0; for (size_t i = 0; i < outn; ++i) diff += std::memcmp(&h1[i], &h2[i], 4) != 0;
    const double bytes = double(expert) * G;
    printf("type=%d rows=%d nb=%d parts=%d G=%d per=%d: ref %.1f us (%.0f GB/s)  cand %.1f us (%.0f GB/s)  differing=%zu/%zu\n", type, rows, nb, parts, G, per,
           tr * 1e3 / iters, bytes / (tr * 1e-3 / iters) / 1e9, tc * 1e3 / iters, bytes / (tc * 1e-3 / iters) / 1e9, diff, outn);
    cudaFree(dw); cudaFree(dwp); cudaFree(dgs); cudaFree(dea); cudaFree(dact); cudaFree(o1); cudaFree(o2);
    return diff != 0;
}
int main() {
    int bad = 0;
    bad += run(19, 2304, 20, 2, 10, 2, 6, 50);   // gate/up, verify-like
    bad += run(19, 5120, 9, 1, 10, 2, 6, 50);    // down
    bad += run(19, 2304, 20, 2, 3, 1, 1, 50);    // gate/up, single token
    bad += run(16, 2304, 20, 2, 10, 2, 6, 50);   // IQ2_XXS gate/up
    bad += run(16, 5120, 9, 1, 10, 3, 6, 50);    // IQ2_XXS down, 3 entries (odd chunk)
    bad += run(19, 5120, 9, 1, 7, 5, 8, 20);     // 5 entries per group
    return bad;
}
