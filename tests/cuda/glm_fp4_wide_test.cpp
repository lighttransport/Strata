// glm_fp4_gemm and glm_mxfp8_gemm_three at 9..32 tokens against the same tokens as 8-token calls: bitwise equal,
// rows past the tokens untouched.
#include "strata/kernels/glm_fp4.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace k = strata::kernels;
static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void *p = nullptr; size_t bytes;
    explicit Buffer(size_t n) : bytes(n) { ck(cudaMalloc(&p, n)); }
    ~Buffer() { cudaFree(p); }
    template <class T> T *as() const { return static_cast<T *>(p); }
    template <class T> void fill(const std::vector<T> &v) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<uint8_t> read() const { std::vector<uint8_t> v(bytes); ck(cudaMemcpy(v.data(), p, bytes, cudaMemcpyDeviceToHost)); return v; }
};
static uint64_t seed = 99;
static uint32_t next() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(seed >> 33); }
static std::vector<uint16_t> halves(size_t n) {   // finite FP16 in about [2^-3, 2^2)
    std::vector<uint16_t> v(n);
    for (auto &h : v) { const uint32_t r = next(); h = uint16_t((r & 0x8000) | ((12 + r % 5) << 10) | (r >> 6 & 0x3ff)); }
    return v;
}
static std::vector<uint8_t> fp8(size_t n) {   // E4M3 codes without NaN
    std::vector<uint8_t> v(n);
    for (auto &c : v) { const uint32_t r = next(); c = uint8_t((r % 120) | (r & 0x100 ? 0x80 : 0)); }
    return v;
}
static int failures = 0, cases = 0;
static void same(const std::string &what, const Buffer &a, const Buffer &b) {
    ++cases;
    if (a.read() != b.read()) { std::cerr << "FAIL " << what << '\n'; ++failures; }
}
struct Weights { k::GlmPackedFormat format; std::unique_ptr<Buffer> w, s; };
static Weights make(k::GlmPackedFormat format, int N, int K) {
    Weights m{format, nullptr, nullptr};
    const size_t count = size_t(N) * K;
    if (format == k::GlmPackedFormat::F16) {
        m.w = std::make_unique<Buffer>(count * 2); m.w->fill(halves(count));
    } else if (format == k::GlmPackedFormat::BlockFp8) {
        m.w = std::make_unique<Buffer>(count); m.w->fill(fp8(count));
        std::vector<float> scales(size_t((N + 127) / 128) * ((K + 127) / 128));
        for (auto &s : scales) s = 0.01f + float(next() % 100) * 1e-3f;
        m.s = std::make_unique<Buffer>(scales.size() * 4); m.s->fill(scales);
    } else {   // MXFP8 prepacked to the QPN layout
        Buffer raw(count), raw_scales(count / 32);
        raw.fill(fp8(count));
        std::vector<uint8_t> e(count / 32);
        for (auto &x : e) x = uint8_t(116 + next() % 10);
        raw_scales.fill(e);
        m.w = std::make_unique<Buffer>(count); m.s = std::make_unique<Buffer>(count / 16);
        k::glm_fp8_prepack(k::GlmPackedFormat::Mxfp8, raw.as<uint8_t>(), raw_scales.p, m.w->as<uint8_t>(), m.s->p, N, K, nullptr);
        ck(cudaDeviceSynchronize());
    }
    return m;
}
int main() {
    int devices = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    ck(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) return 77;
    try {
        cudaStream_t s; ck(cudaStreamCreate(&s));
        Buffer partials(k::glm_fp4_partial_bytes(8192, 32));
        // (N, K, partials): K-split small products, the 8192-row split, and the plain grid.
        const std::array<std::tuple<int, int, bool>, 5> shapes{{{32, 4096, true}, {512, 4096, true}, {2048, 4096, true},
                                                                {8192, 4096, true}, {1024, 2048, false}}};
        for (auto format : {k::GlmPackedFormat::Mxfp8Qpn, k::GlmPackedFormat::BlockFp8, k::GlmPackedFormat::F16})
        for (auto [N, K, split] : shapes) {
            const auto m = make(format, N, K);
            Buffer x(size_t(32) * K * 2); x.fill(halves(size_t(32) * K));
            for (int M : {9, 13, 16, 17, 24, 32}) for (int acc : {1, 2}) {
                if (acc != 1 && format == k::GlmPackedFormat::F16) continue;
                Buffer wide(size_t(32) * N * 4), chunks(size_t(32) * N * 4);
                ck(cudaMemset(wide.p, 0x5a, wide.bytes)); ck(cudaMemset(chunks.p, 0x5a, chunks.bytes));
                float *scratch = split ? partials.as<float>() : nullptr;
                k::glm_fp4_gemm(format, m.w->p, m.s ? m.s->p : nullptr, 1, x.as<uint16_t>(), wide.as<float>(), N, K, M, true, s, scratch, acc);
                for (int first = 0; first < M; first += 8)
                    k::glm_fp4_gemm(format, m.w->p, m.s ? m.s->p : nullptr, 1, x.as<uint16_t>() + size_t(first) * K,
                                    chunks.as<float>() + size_t(first) * N, N, K, std::min(8, M - first), true, s, scratch, acc);
                ck(cudaStreamSynchronize(s));
                same("gemm format=" + std::to_string(int(format)) + " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                     " M=" + std::to_string(M) + " accumulators=" + std::to_string(acc), wide, chunks);
            }
        }
        {   // three MXFP8 products in one launch
            const int N = 4096, K = 4096;
            std::array<Weights, 3> m{make(k::GlmPackedFormat::Mxfp8Qpn, N, K), make(k::GlmPackedFormat::Mxfp8Qpn, N, K),
                                     make(k::GlmPackedFormat::Mxfp8Qpn, N, K)};
            Buffer x(size_t(32) * K * 2); x.fill(halves(size_t(32) * K));
            for (int M : {9, 16, 20, 32}) {
                Buffer wide(size_t(3) * 32 * N * 4), chunks(size_t(3) * 32 * N * 4);
                ck(cudaMemset(wide.p, 0x5a, wide.bytes)); ck(cudaMemset(chunks.p, 0x5a, chunks.bytes));
                auto p = [&](const Buffer &out, int i, int first) {
                    return k::GlmFp8Projection{m[i].w->p, m[i].s->p, out.as<float>() + size_t(i) * 32 * N + size_t(first) * N};
                };
                k::glm_mxfp8_gemm_three(p(wide, 0, 0), p(wide, 1, 0), p(wide, 2, 0), x.as<uint16_t>(), N, K, M, s);
                for (int first = 0; first < M; first += 8)
                    k::glm_mxfp8_gemm_three(p(chunks, 0, first), p(chunks, 1, first), p(chunks, 2, first),
                                            x.as<uint16_t>() + size_t(first) * K, N, K, std::min(8, M - first), s);
                ck(cudaStreamSynchronize(s));
                same("three M=" + std::to_string(M), wide, chunks);
            }
        }
        cudaStreamDestroy(s);
    } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
    if (failures) { std::cerr << failures << " of " << cases << " cases differ\n"; return 1; }
    std::cout << "glm fp4 wide products: " << cases << " cases bitwise equal to 8-token calls\n";
    return 0;
}
