// src/prefill/mmq_iq1m_parity.cpp - the IQ1_M MMQ tile (third_party/ggml/patches/apply_mmq_iq1m.py, Q6_K's tile and
// dot product) against two references on random IQ1_M blocks:
//   exact: the dequantized weights times the Q8_1-rounded activations MMQ itself multiplies (per 32 values: d = max|x|
//          / 127, q = rint(x / d)), summed in double - the kernel's integer dot is exact, so this agrees to float
//          accumulation order (~1e-6 relative); a tile-mapping or scale error shows as percent-level disagreement;
//   float: the same against the unrounded activations (the usual ~0.3-1% of an 8-bit activation path).
//
//     mmq_iq1m_parity [rows ...]        (default 1 7 64 200 activation rows; gate/up [4608 x 5120] and down [5120 x 2304])
//
// Exit status 1 when any product's exact-reference error exceeds 1e-4.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// random IQ1_M blocks: any qs / qh / low scale bits decode; the fp16 block scale (the four high nibbles of the
// scale words) drawn from sane values so no block is inf / nan / subnormal
std::vector<uint8_t> random_iq1m(int64_t rows, int64_t cols, std::mt19937& rng) {
    const int64_t blocks = rows * cols / 256;
    std::vector<uint8_t> w(size_t(blocks) * 56);
    std::uniform_int_distribution<int> byte(0, 255);
    static const uint16_t ds[] = {0x2E66, 0x3266, 0x3800, 0x2C00, 0x3555, 0x2A3D};   // 0.1 .. 0.67
    for (int64_t b = 0; b < blocks; ++b) {
        uint8_t* blk = w.data() + size_t(b) * 56;
        for (int i = 0; i < 56; ++i) blk[i] = uint8_t(byte(rng));
        const uint16_t d = ds[b % 6];
        uint16_t sc[4];
        std::memcpy(sc, blk + 48, 8);
        sc[0] = uint16_t((sc[0] & 0x0FFF) | ((d & 0x000F) << 12));
        sc[1] = uint16_t((sc[1] & 0x0FFF) | ((d & 0x00F0) << 8));
        sc[2] = uint16_t((sc[2] & 0x0FFF) | ((d & 0x0F00) << 4));
        sc[3] = uint16_t((sc[3] & 0x0FFF) | (d & 0xF000));
        std::memcpy(blk + 48, sc, 8);
    }
    return w;
}

int product(int64_t rows, int64_t cols, int T, const char* what, std::mt19937& rng) {
    const auto w = random_iq1m(rows, cols, rng);
    const size_t wbytes = mmq::matrix_bytes(GGML_TYPE_IQ1_M, rows, cols);
    if (wbytes != w.size()) { std::fprintf(stderr, "matrix_bytes %zu vs %zu\n", wbytes, w.size()); return 1; }
    std::vector<float> wf(size_t(rows) * cols);
    ggml_get_type_traits(GGML_TYPE_IQ1_M)->to_float(w.data(), wf.data(), rows * cols);
    std::vector<float> x(size_t(T) * cols), xq(size_t(T) * cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    for (int t = 0; t < T; ++t) {
        const float m = mag(rng);
        for (int64_t k = 0; k < cols; ++k) x[size_t(t) * cols + k] = nd(rng) * m;
        for (int64_t k0 = 0; k0 < cols; k0 += 32) {   // MMQ's Q8_1 rounding per 32
            float amax = 0;
            for (int k = 0; k < 32; ++k) amax = std::fmax(amax, std::fabs(x[size_t(t) * cols + k0 + k]));
            const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
            for (int k = 0; k < 32; ++k) xq[size_t(t) * cols + k0 + k] = d * std::rint(x[size_t(t) * cols + k0 + k] * id);
        }
    }
    std::vector<double> ref(size_t(T) * rows), refq(size_t(T) * rows);
    for (int t = 0; t < T; ++t)
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0, sq = 0;
            const float* a = x.data() + size_t(t) * cols;
            const float* aq = xq.data() + size_t(t) * cols;
            const float* b = wf.data() + size_t(r) * cols;
            for (int64_t k = 0; k < cols; ++k) { s += double(a[k]) * b[k]; sq += double(aq[k]) * b[k]; }
            ref[size_t(t) * rows + r] = s;
            refq[size_t(t) * rows + r] = sq;
        }

    uint8_t* dw = nullptr; float* dx = nullptr; void* dq = nullptr; float* dd = nullptr;
    int32_t* dids = nullptr; int32_t* dbounds = nullptr;
    ck(cudaMalloc(&dw, wbytes + 4096), "w");
    ck(cudaMemset(dw, 0, wbytes + 4096), "w");
    ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * sizeof(float)), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, cols)), "xq");
    ck(cudaMalloc(&dd, size_t(T) * rows * sizeof(float)), "dst");
    ck(cudaMalloc(&dids, (T + 128) * sizeof(int32_t)), "ids");
    ck(cudaMemset(dids, 0, (T + 128) * sizeof(int32_t)), "ids");
    ck(cudaMalloc(&dbounds, 2 * sizeof(int32_t)), "bounds");
    const int32_t bounds[2] = {0, T};
    ck(cudaMemcpy(dbounds, bounds, sizeof bounds, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, GGML_TYPE_IQ1_M, cols, cols, T, nullptr);
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw; p.type = GGML_TYPE_IQ1_M; p.w_rows = rows; p.w_cols = cols; p.expert_bytes = wbytes; p.n = 1;
    p.xq = dq; p.bounds = dbounds; p.ids = dids; p.total_rows = T; p.max_rows = T; p.dst = dd; p.ld_dst = rows;
    p.dst_rows = T;
    ctx.run(p, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> got(size_t(T) * rows);
    ck(cudaMemcpy(got.data(), dd, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "dst");
    cudaFree(dw); cudaFree(dx); cudaFree(dq); cudaFree(dd); cudaFree(dids); cudaFree(dbounds);

    double num = 0, numq = 0, den = 0, worstq = 0;
    for (int t = 0; t < T; ++t) {
        double rmax = 0;
        for (int64_t r = 0; r < rows; ++r) rmax = std::fmax(rmax, std::fabs(refq[size_t(t) * rows + r]));
        for (int64_t r = 0; r < rows; ++r) {
            const double g = got[size_t(t) * rows + r];
            const double e = g - ref[size_t(t) * rows + r], eq = g - refq[size_t(t) * rows + r];
            num += e * e;
            numq += eq * eq;
            den += refq[size_t(t) * rows + r] * refq[size_t(t) * rows + r];
            worstq = std::fmax(worstq, std::fabs(eq) / (rmax > 0 ? rmax : 1));
        }
    }
    const double rel = std::sqrt(num / den), relq = std::sqrt(numq / den);
    std::printf("%-8s T %4d  [%lld x %lld]  vs exact %.2e (worst %.2e of the row max)  vs float %.3f%%%s\n", what, T,
                (long long) rows, (long long) cols, relq, worstq, 100 * rel, relq > 1e-4 ? "  <-- WRONG" : "");
    return relq > 1e-4 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (!mmq::supported(GGML_TYPE_IQ1_M)) { std::fprintf(stderr, "IQ1_M is not in this build's MMQ\n"); return 1; }
    std::vector<int> Ts = {1, 7, 64, 200};
    if (argc > 1) { Ts.clear(); for (int i = 1; i < argc; ++i) Ts.push_back(std::atoi(argv[i])); }
    std::mt19937 rng(7);
    int bad = 0;
    for (int T : Ts) {
        bad += product(4608, 5120, T, "gate/up", rng);
        bad += product(5120, 2304, T, "down", rng);
    }
    std::printf(bad ? "FAIL\n" : "OK\n");
    return bad ? 1 : 0;
}
