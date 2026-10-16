// GPU Tetra encoder (CUDA or HIP): f64 words and t bit-identical to llvq's on llvq_dump's blocks (golden hash) and
// to the CPU encoder on 20,000 more; the f32 fallback's agreement and its loss of t are reported. With a reference
// file from llvq_dump, every block of that file is checked too, with timing.
//   llvq_tetra_gpu_test [reference.bin [ctas]]
#include "blocks.hpp"
#include "strata/kernels/llvq_tetra.hpp"
#include "strata/llvq/tetra_encoder.hpp"
#include "strata/llvq/tetra_view.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <thread>
#include <vector>

using namespace strata;

static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "gpu: %s\n", cudaGetErrorString(e)); std::exit(2); }
}

struct Run {
    std::vector<uint64_t> w;
    std::vector<double> t;
    double seconds = 0;
};

static Run gpu_run(kernels::LlvqTetraGpu* g, const std::vector<double>& x, bool f32) {
    const size_t n = x.size() / 24;
    Run r;
    r.w.resize(n);
    r.t.resize(n);
    double* dx = nullptr;
    float* dxf = nullptr;
    uint64_t* dw;
    double* dt;
    ck(cudaMalloc((void**)&dw, n * 8));
    ck(cudaMalloc((void**)&dt, n * 8));
    if (f32) {
        std::vector<float> xf(x.begin(), x.end());
        ck(cudaMalloc((void**)&dxf, xf.size() * 4));
        ck(cudaMemcpy(dxf, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice));
    } else {
        ck(cudaMalloc((void**)&dx, x.size() * 8));
        ck(cudaMemcpy(dx, x.data(), x.size() * 8, cudaMemcpyHostToDevice));
    }
    ck(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    if (f32) kernels::llvq_tetra_encode_f32(g, dxf, int64_t(n), dw, dt, nullptr);
    else kernels::llvq_tetra_encode_f64(g, dx, int64_t(n), dw, dt, nullptr);
    ck(cudaDeviceSynchronize());
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ck(cudaMemcpy(r.w.data(), dw, n * 8, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(r.t.data(), dt, n * 8, cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dxf); cudaFree(dw); cudaFree(dt);
    return r;
}

// Mismatches against (wref, tref); for differing words also the relative loss of t (the encoder maximizes t).
static size_t compare(const char* what, const Run& r, const std::vector<uint64_t>& wref, const std::vector<double>& tref) {
    size_t wbad = 0, tbad = 0, worse = 0;
    double loss = 0, worst = 0;
    for (size_t i = 0; i < r.w.size(); ++i) {
        wbad += r.w[i] != wref[i];
        tbad += std::memcmp(&r.t[i], &tref[i], 8) != 0;
        if (r.w[i] != wref[i] && tref[i] > 0 && std::isfinite(r.t[i]) && std::isfinite(tref[i])) {
            double l = (tref[i] - r.t[i]) / tref[i];
            worse += l > 1e-6;
            loss += l;
            worst = std::max(worst, l);
        }
    }
    std::printf("%s: %zu blocks, word mismatches %zu, t mismatches %zu, %.0f blocks/s", what, r.w.size(), wbad, tbad,
                r.w.size() / r.seconds);
    if (wbad) std::printf("; differing words worse by >1e-6 of t: %zu, mean loss %.3g, worst %.3g", worse, loss / r.w.size(), worst);
    std::printf("\n");
    return wbad + tbad;
}

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { std::printf("no GPU\n"); return 77; }
    llvq::Tetra tetra;
    llvq::core::TetraTablesView view;
    std::vector<int16_t> pos5;
    llvq::make_view(tetra, view, pos5);
    auto* g = kernels::llvq_tetra_gpu_create(&view, pos5.data());
    if (argc > 2) kernels::llvq_tetra_gpu_set_ctas(g, std::atoi(argv[2]));
    int status = 0;
    {
        auto x = llvq::test::dump_blocks(llvq::test::GOLDEN_N, llvq::test::GOLDEN_SEED);
        Run r = gpu_run(g, x, false);
        llvq::test::Fnv h;
        for (size_t i = 0; i < r.w.size(); ++i) { h.bytes(&r.w[i], 8); h.bytes(&r.t[i], 8); }
        const bool ok = h.h == llvq::test::GOLDEN_FNV;
        std::printf("%s f64: %zu blocks %s llvq (golden hash %016llx)\n", ok ? "ok" : "FAIL", r.w.size(),
                    ok ? "bit-identical to" : "differ from", (unsigned long long)h.h);
        status |= !ok;
    }
    {
        const size_t n = 20000;
        auto x = llvq::test::dump_blocks(n, 0xc0ffee);
        std::vector<uint64_t> w(n);
        std::vector<double> t(n);
        llvq::Encoder enc(tetra);
        std::vector<std::thread> pool;
        const unsigned th = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned k = 0; k < th; ++k)
            pool.emplace_back([&, k] {
                llvq::Scratch sc;
                for (size_t i = k; i < n; i += th) {
                    auto c = enc.encode(&x[24 * i], sc);
                    w[i] = c.word;
                    t[i] = c.t;
                }
            });
        for (auto& p : pool) p.join();
        status |= compare("vs CPU encoder, f64", gpu_run(g, x, false), w, t) != 0;
        compare("vs CPU encoder, f32 (format-compatible, not bit-exact)", gpu_run(g, x, true), w, t);
    }
    if (argc > 1) {
        std::ifstream f(argv[1], std::ios::binary);
        std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
        const size_t n = raw.size() / 208;
        std::vector<double> x(n * 24), tref(n);
        std::vector<uint64_t> wref(n);
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(&x[i * 24], raw.data() + i * 208, 192);
            std::memcpy(&wref[i], raw.data() + i * 208 + 192, 8);
            std::memcpy(&tref[i], raw.data() + i * 208 + 200, 8);
        }
        status |= compare("reference file, f64", gpu_run(g, x, false), wref, tref) != 0;
        compare("reference file, f32", gpu_run(g, x, true), wref, tref);
    }
    kernels::llvq_tetra_gpu_destroy(g);
    return status;
}
