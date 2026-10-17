// llvq Tetra decode: the trio-order word decoder against Tetra::decode, and the GEMV / dequant kernels (CPU scalar,
// AVX2, and the GPU when built with one) against a double-precision reference on random words, with timing.
//   llvq_tetra_decode_test [rows cols]
#include "strata/llvq/tetra_decode.hpp"
#ifdef STRATA_LLVQ_GPU
#include "strata/kernels/llvq_tetra.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace strata;
using namespace strata::llvq;

namespace {
double secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
struct Record {
    std::vector<uint8_t> bytes;
    TetraMatrix m;
};
// A record of random words (every 47-bit label is a word), gains, row scales and tail.
Record random_record(uint32_t rows, uint32_t cols, uint64_t seed) {
    SplitMix64 rng(seed);
    const uint32_t nb = cols / 24, tail = cols - 24 * nb, stride = (nb * 6 + 7) / 8 * 8;
    Record r;
    auto put = [&](const void* d, size_t n) { const auto* b = static_cast<const uint8_t*>(d); r.bytes.insert(r.bytes.end(), b, b + n); };
    put(&rows, 4); put(&cols, 4);
    double c[2] = {0.8 + 0.1 * rng.next_f64(), 1.3 + 0.2 * rng.next_f64()};
    put(c, 16);
    for (uint32_t i = 0; i < rows; ++i) { float s = float(0.01 + 0.02 * rng.next_f64()); put(&s, 4); }
    std::vector<uint8_t> row(stride);
    for (uint32_t i = 0; i < rows; ++i) {
        std::fill(row.begin(), row.end(), 0);
        for (uint32_t b = 0; b < nb; ++b) {
            uint64_t w = rng.next() & ((uint64_t(1) << 48) - 1);
            if (rng.next() % 64 == 0) w = rng.next() & 1 ? 0 : uint64_t(1) << 47;  // zero blocks, both gains
            std::memcpy(&row[6 * b], &w, 6);
        }
        put(row.data(), stride);
    }
    for (uint32_t i = 0; i < rows * tail; ++i) {
        float f = float(rng.next_gaussian() * 0.02);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        uint16_t h = uint16_t(u >> 16);
        put(&h, 2);
    }
    r.m.parse(r.bytes.data());
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    const uint32_t rows = argc > 2 ? uint32_t(std::atoi(argv[1])) : 2048, cols = argc > 2 ? uint32_t(std::atoi(argv[2])) : 4096;
    Tetra tetra;
    const DecodeTables T = DecodeTables::build(tetra);
    int fails = 0;
    {
        SplitMix64 rng(7);
        for (int i = 0; i < 100000; ++i) {
            uint64_t w = rng.next() & LABEL_MASK;
            auto nat = tetra.decode(w);
            int8_t y[DIM];
            T.decode_trio(w, y);
            for (int t = 0; t < DIM; ++t)
                if (nat[T.order[t]] != y[t]) { ++fails; break; }
        }
        std::printf("%s trio decode vs Tetra::decode on 100000 words\n", fails ? "FAIL" : "ok");
    }
    Record rec = random_record(rows, cols, 99);
    const TetraMatrix& m = rec.m;
    Rotation rot(cols, 0x11a7e5eedull);
    const int NT = 8;
    std::vector<float> x(size_t(NT) * cols), xp(size_t(NT) * cols);
    SplitMix64 rng(3);
    for (auto& v : x) v = float(rng.next_gaussian());
    for (int k = 0; k < NT; ++k) prepare_activation(T, rot, &x[size_t(k) * cols], cols, &xp[size_t(k) * cols]);
    // Reference: dequantized rows (natural rotated order) against the rotated activation, in double.
    std::vector<double> ref(size_t(rows) * NT);
    {
        std::vector<float> wrow(cols);
        std::vector<double> xr(size_t(NT) * cols);
        for (int k = 0; k < NT; ++k) {
            std::vector<double> v(x.begin() + size_t(k) * cols, x.begin() + size_t(k + 1) * cols);
            rot.apply(v.data());
            std::copy(v.begin(), v.end(), xr.begin() + size_t(k) * cols);
        }
        for (uint32_t r = 0; r < rows; ++r) {
            dequant_row(T, m, r, wrow.data());
            for (int k = 0; k < NT; ++k) {
                double s = 0;
                for (uint32_t c = 0; c < cols; ++c) s += double(wrow[c]) * xr[size_t(k) * cols + c];
                ref[size_t(r) * NT + k] = s;
            }
        }
    }
    auto check = [&](const char* what, const std::vector<float>& y, int nt) {
        double num = 0, den = 0;
        for (uint32_t r = 0; r < rows; ++r)
            for (int k = 0; k < nt; ++k) {
                double d = double(y[size_t(r) * nt + k]) - ref[size_t(r) * NT + k];
                num += d * d;
                den += ref[size_t(r) * NT + k] * ref[size_t(r) * NT + k];
            }
        double rel = std::sqrt(num / den);
        bool ok = rel < 1e-5;
        fails += !ok;
        std::printf("%s %s: relative error %.2e\n", ok ? "ok" : "FAIL", what, rel);
    };
    const double mbytes = double(m.stride) * rows / 1e6;
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    for (int nt : {1, 8}) {
        std::vector<float> y(size_t(rows) * nt);
        std::vector<float> xs(xp.begin(), xp.begin() + size_t(nt) * cols);
        gemv_scalar(T, m, xs.data(), nt, 0, rows, y.data());
        check(nt == 1 ? "cpu scalar gemv, 1 token" : "cpu scalar gemv, 8 tokens", y, nt);
#if defined(__AVX2__) && defined(__FMA__)
        std::fill(y.begin(), y.end(), 0.f);
        gemv_avx2(T, m, xs.data(), nt, 0, rows, y.data());
        check(nt == 1 ? "cpu avx2 gemv, 1 token" : "cpu avx2 gemv, 8 tokens", y, nt);
        const int reps = 50;
        auto t0 = std::chrono::steady_clock::now();
        {
            std::vector<std::thread> pool;
            const uint32_t per = (rows + threads - 1) / threads;
            for (unsigned t = 0; t < threads; ++t)
                pool.emplace_back([&, t] {
                    for (int it = 0; it < reps; ++it)
                        gemv_avx2(T, m, xs.data(), nt, std::min(rows, t * per), std::min(rows, (t + 1) * per), y.data());
                });
            for (auto& p : pool) p.join();
        }
        double s = secs(t0) / reps;
        std::printf("   avx2 %ux%u, %d token(s), %u threads: %.1f us, %.1f GB/s of words\n", rows, cols, nt, threads, s * 1e6, mbytes / 1e3 / s);
#endif
    }
#ifdef STRATA_LLVQ_GPU
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        auto* d = kernels::llvq_tetra_decode_create(&T.pre[0][0], &T.suf[0][0], &T.br_c[0][0], &T.br_s[0][0], T.rows, &T.vals[0][0], T.order);
        uint8_t* dw; float *dsc, *dx, *dxp, *dy, *dsg; uint16_t* dt; __half* dh;
        cudaMalloc((void**)&dw, size_t(m.stride) * rows);
        cudaMalloc((void**)&dsc, rows * 4);
        cudaMalloc((void**)&dt, size_t(rows) * m.tail * 2 + 2);
        cudaMalloc((void**)&dx, x.size() * 4);
        cudaMalloc((void**)&dxp, x.size() * 4);
        cudaMalloc((void**)&dy, size_t(rows) * NT * 4);
        cudaMalloc((void**)&dsg, cols * 4);
        cudaMalloc((void**)&dh, size_t(rows) * cols * 2);
        cudaMemcpy(dw, m.words, size_t(m.stride) * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(dsc, m.scales, rows * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dt, m.tail_bf16, size_t(rows) * m.tail * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        std::vector<float> sg(rot.signs().begin(), rot.signs().end());
        cudaMemcpy(dsg, sg.data(), cols * 4, cudaMemcpyHostToDevice);
        kernels::LlvqTetraMatrixDev md{dw, dsc, dt, float(m.centroids[0]), float(m.centroids[1]), m.rows, m.cols, m.stride};
        kernels::llvq_tetra_prepare(d, dsg, dx, NT, int(cols), dxp, nullptr);
        std::vector<float> gxp(x.size());
        cudaMemcpy(gxp.data(), dxp, x.size() * 4, cudaMemcpyDeviceToHost);
        double num = 0, den = 0;
        for (size_t i = 0; i < x.size(); ++i) { num += double(gxp[i] - xp[i]) * (gxp[i] - xp[i]); den += double(xp[i]) * xp[i]; }
        bool okp = std::sqrt(num / den) < 1e-5;
        fails += !okp;
        std::printf("%s gpu prepare (rotation + trio order) vs cpu: relative error %.2e\n", okp ? "ok" : "FAIL", std::sqrt(num / den));
        cudaMemcpy(dxp, xp.data(), x.size() * 4, cudaMemcpyHostToDevice);
        for (int nt : {1, 8}) {
            std::vector<float> y(size_t(rows) * nt);
            kernels::llvq_tetra_gemv(d, md, dxp, nt, dy, nullptr);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            check(nt == 1 ? "gpu gemv, 1 token" : "gpu gemv, 8 tokens", y, nt);
            cudaDeviceSynchronize();
            const int reps = 200;
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < reps; ++it) kernels::llvq_tetra_gemv(d, md, dxp, nt, dy, nullptr);
            cudaDeviceSynchronize();
            double s = secs(t0) / reps;
            std::printf("   gpu %ux%u, %d token(s): %.1f us, %.1f GB/s of words\n", rows, cols, nt, s * 1e6, mbytes / 1e3 / s);
        }
        kernels::llvq_tetra_dequant_f16(d, md, dh, nullptr);
        std::vector<__half> h(size_t(rows) * cols);
        cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<float> wrow(cols);
        double maxe = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            dequant_row(T, m, r, wrow.data());
            for (uint32_t c = 0; c < cols; ++c) maxe = std::max(maxe, double(std::fabs(__half2float(h[size_t(r) * cols + c]) - wrow[c]) / (std::fabs(wrow[c]) + 1e-3)));
        }
        bool okd = maxe < 2e-3;
        fails += !okd;
        std::printf("%s gpu dequant f16 vs cpu: max relative error %.2e\n", okd ? "ok" : "FAIL", maxe);
        kernels::llvq_tetra_decode_destroy(d);
    } else {
        std::printf("no GPU: skipped the GPU kernels\n");
    }
#endif
    return fails ? 1 : 0;
}
