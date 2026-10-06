// Q8_0 decode GEMVs against an FP64 reference at GLM's MLA absorb and mHC projection shapes.
#include "strata/kernels/glm_q8.hpp"
#include "strata/kernels/glm.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Q8 {
    std::vector<uint8_t> bytes;
    std::vector<float> values;
};
Q8 make(size_t count, std::mt19937 &rng) {
    Q8 q;
    q.bytes.resize(count / 32 * 34);
    q.values.resize(count);
    std::uniform_real_distribution<float> d(1e-4f, 2e-2f);
    std::uniform_int_distribution<int> v(-127, 127);
    for (size_t b = 0; b < count / 32; ++b) {
        const __half h = __float2half(d(rng));
        std::memcpy(&q.bytes[b * 34], &h, 2);
        const float s = __half2float(h);
        for (int i = 0; i < 32; ++i) {
            const int8_t x = (int8_t)v(rng);
            q.bytes[b * 34 + 2 + i] = (uint8_t)x;
            q.values[b * 32 + i] = s * x;
        }
    }
    return q;
}
template <class T> T *upload(const std::vector<T> &x) {
    T *p = nullptr;
    ck(cudaMalloc(&p, x.size() * sizeof(T)));
    ck(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
std::vector<float> download(const float *p, size_t n) {
    std::vector<float> x(n);
    ck(cudaMemcpy(x.data(), p, n * 4, cudaMemcpyDeviceToHost));
    return x;
}
int failures = 0;
void compare(const char *what, const std::vector<float> &got, const std::vector<double> &want,
             const std::vector<double> &magnitude) {
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::abs(got[i] - want[i]) / (magnitude[i] + 1e-30));
    const bool pass = worst < 2e-6;
    if (!pass) ++failures;
    std::cout << (pass ? "PASS " : "FAIL ") << what << " worst_relative_to_abs_sum=" << worst << '\n';
}
int main() {
    try {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> act(-3.f, 3.f);
        cudaStream_t s;
        ck(cudaStreamCreate(&s));
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a));
        ck(cudaEventCreate(&b));
        struct Shape { const char *name; int heads, in, out; };
        for (const Shape shape : {Shape{"attn_k_b", 64, 256, 512}, Shape{"attn_v_b", 64, 512, 256}}) {
            const size_t rows = (size_t)shape.heads * shape.out;
            auto w = make(rows * shape.in, rng);
            auto *dw = upload(w.bytes);
            for (int tokens : {1, 2, 3, 4, 6, 8}) {
                const int ld_x = shape.heads * shape.in + 32, ld_y = (int)rows + 64;
                std::vector<float> x((size_t)tokens * ld_x);
                for (auto &v : x) v = act(rng);
                auto *dx = upload(x);
                float *dy = nullptr;
                ck(cudaMalloc(&dy, (size_t)tokens * ld_y * 4));
                ck(cudaMemset(dy, 0, (size_t)tokens * ld_y * 4));
                k::glm_q8_heads_gemv(dw, dx, dy, shape.heads, shape.in, shape.out, tokens, ld_x, ld_y, s);
                ck(cudaEventRecord(a, s));
                for (int r = 0; r < 20; ++r)
                    k::glm_q8_heads_gemv(dw, dx, dy, shape.heads, shape.in, shape.out, tokens, ld_x, ld_y, s);
                ck(cudaEventRecord(b, s));
                ck(cudaEventSynchronize(b));
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, a, b));
                auto all = download(dy, (size_t)tokens * ld_y);
                std::vector<float> got;
                std::vector<double> want, magnitude;
                for (int t = 0; t < tokens; ++t)
                    for (size_t r = 0; r < rows; ++r) {
                        double sum = 0, abs = 0;
                        const size_t h = r / shape.out;
                        for (int i = 0; i < shape.in; ++i) {
                            const double p = (double)w.values[r * shape.in + i] * x[(size_t)t * ld_x + h * shape.in + i];
                            sum += p; abs += std::abs(p);
                        }
                        got.push_back(all[(size_t)t * ld_y + r]); want.push_back(sum); magnitude.push_back(abs);
                    }
                std::string what = std::string(shape.name) + " tokens=" + std::to_string(tokens) +
                                   " us=" + std::to_string(1000 * ms / 20);
                compare(what.c_str(), got, want, magnitude);
                cudaFree(dx); cudaFree(dy);
            }
            cudaFree(dw);
        }
        {
            const int rows = 24, in = 16384;
            auto w = make((size_t)rows * in, rng);
            auto *dw = upload(w.bytes);
            float *scratch = nullptr;
            ck(cudaMalloc(&scratch, (size_t)rows * 8 * 32 * 4));
            for (int tokens : {1, 2, 3, 4, 8}) {
                std::vector<float> x((size_t)tokens * in);
                for (auto &v : x) v = act(rng);
                auto *dx = upload(x);
                float *dy = nullptr;
                ck(cudaMalloc(&dy, (size_t)tokens * rows * 4));
                k::glm_q8_rows_gemv(dw, dx, dy, scratch, rows, in, tokens, s);
                ck(cudaEventRecord(a, s));
                for (int r = 0; r < 20; ++r) k::glm_q8_rows_gemv(dw, dx, dy, scratch, rows, in, tokens, s);
                ck(cudaEventRecord(b, s));
                ck(cudaEventSynchronize(b));
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, a, b));
                auto got = download(dy, (size_t)tokens * rows);
                std::vector<double> want, magnitude;
                for (int t = 0; t < tokens; ++t)
                    for (int r = 0; r < rows; ++r) {
                        double sum = 0, abs = 0;
                        for (int i = 0; i < in; ++i) {
                            const double p = (double)w.values[(size_t)r * in + i] * x[(size_t)t * in + i];
                            sum += p; abs += std::abs(p);
                        }
                        want.push_back(sum); magnitude.push_back(abs);
                    }
                std::string what = "hc_fn tokens=" + std::to_string(tokens) + " us=" + std::to_string(1000 * ms / 20);
                compare(what.c_str(), got, want, magnitude);
                cudaFree(dx); cudaFree(dy);
            }
            float *dq = nullptr;
            ck(cudaMalloc(&dq, (size_t)rows * in * 4));
            k::glm_q8_dequant(dw, dq, (long long)rows * in, s);
            auto values = download(dq, (size_t)rows * in);
            const bool exact = values == w.values;
            if (!exact) ++failures;
            std::cout << (exact ? "PASS" : "FAIL") << " dequant exact\n";
            cudaFree(dq); cudaFree(scratch); cudaFree(dw);
        }
        {
            const int rows = 288, in = 4096;
            std::vector<float> w((size_t)rows * in);
            for (auto &v : w) v = act(rng) * 0.01f;
            auto *dw = upload(w);
            for (int tokens : {1, 2, 3, 4, 8}) {
                std::vector<float> x((size_t)tokens * in);
                for (auto &v : x) v = act(rng);
                auto *dx = upload(x);
                float *dy = nullptr;
                ck(cudaMalloc(&dy, (size_t)tokens * rows * 4));
                k::glm_f32_rows_gemv(dw, dx, dy, rows, in, tokens, s);
                ck(cudaEventRecord(a, s));
                for (int r = 0; r < 20; ++r) k::glm_f32_rows_gemv(dw, dx, dy, rows, in, tokens, s);
                ck(cudaEventRecord(b, s));
                ck(cudaEventSynchronize(b));
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, a, b));
                auto got = download(dy, (size_t)tokens * rows);
                std::vector<double> want, magnitude;
                for (int t = 0; t < tokens; ++t)
                    for (int r = 0; r < rows; ++r) {
                        double sum = 0, abs = 0;
                        for (int i = 0; i < in; ++i) {
                            const double p = (double)w[(size_t)r * in + i] * x[(size_t)t * in + i];
                            sum += p; abs += std::abs(p);
                        }
                        want.push_back(sum); magnitude.push_back(abs);
                    }
                std::string what = "router_f32 tokens=" + std::to_string(tokens) + " us=" + std::to_string(1000 * ms / 20);
                compare(what.c_str(), got, want, magnitude);
                cudaFree(dx); cudaFree(dy);
            }
            cudaFree(dw);
        }
        {
            // Token-batched router and mHC launches against per-token calls, bitwise.
            const int tokens = 4, n = 4096, experts = 288, k8 = 8;
            std::vector<float> streams((size_t)tokens * 4 * n), proj(tokens * 24), base(24), scale(3), y((size_t)tokens * n),
                logits((size_t)tokens * experts), bias(experts);
            for (auto *v : {&streams, &proj, &base, &scale, &y, &logits, &bias}) for (auto &x : *v) x = act(rng) * 0.3f;
            auto *ds = upload(streams), *dp = upload(proj), *db = upload(base), *dsc = upload(scale), *dy = upload(y),
                 *dl = upload(logits), *dbias = upload(bias);
            float *c1, *c2, *x1, *x2, *o1, *o2, *w1, *w2;
            int *i1, *i2;
            for (float **p : {&c1, &c2}) ck(cudaMalloc(p, tokens * 24 * 4));
            for (float **p : {&x1, &x2}) ck(cudaMalloc(p, (size_t)tokens * n * 4));
            for (float **p : {&o1, &o2}) ck(cudaMalloc(p, (size_t)tokens * 4 * n * 4));
            for (float **p : {&w1, &w2}) ck(cudaMalloc(p, tokens * k8 * 4));
            for (int **p : {&i1, &i2}) ck(cudaMalloc(p, tokens * k8 * 4));
            for (int t = 0; t < tokens; ++t) {
                k::glm_mhc_read(ds + (size_t)t * 4 * n, dp + t * 24, db, dsc, c1 + t * 24, x1 + (size_t)t * n, n, 20, 1e-6f, s);
                k::glm_mhc_write(ds + (size_t)t * 4 * n, c1 + t * 24, dy + (size_t)t * n, o1 + (size_t)t * 4 * n, n, s);
                k::glm_router(dl + t * experts, dbias, i1 + t * k8, w1 + t * k8, experts, k8, 2.5f, s);
            }
            k::glm_mhc_read_tokens(ds, dp, db, dsc, c2, x2, n, 20, 1e-6f, tokens, s);
            k::glm_mhc_write_tokens(ds, c2, dy, o2, n, tokens, s);
            k::glm_router_tokens(dl, dbias, i2, w2, experts, k8, 2.5f, tokens, s);
            ck(cudaStreamSynchronize(s));
            auto same = [&](const void *a, const void *b2, size_t bytes) {
                std::vector<char> ha(bytes), hb(bytes);
                ck(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(hb.data(), b2, bytes, cudaMemcpyDeviceToHost));
                return ha == hb;
            };
            const bool ok = same(c1, c2, tokens * 24 * 4) && same(x1, x2, (size_t)tokens * n * 4) &&
                            same(o1, o2, (size_t)tokens * 4 * n * 4) && same(i1, i2, tokens * k8 * 4) && same(w1, w2, tokens * k8 * 4);
            if (!ok) ++failures;
            std::cout << (ok ? "PASS" : "FAIL") << " token-batched router/mHC bitwise equal to per-token calls\n";
        }
        std::cout << (failures ? "FAIL" : "PASS") << " glm_q8_test\n";
        return failures != 0;
    } catch (const std::exception &e) {
        std::cerr << "glm_q8_test: " << e.what() << '\n';
        return 1;
    }
}
