#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Buffer {
    T *p = nullptr;
    size_t n;
    Buffer(size_t n) : n(n) {
        ck(cudaMalloc(&p, n * sizeof(T)));
        ck(cudaMemset(p, 0, n * sizeof(T)));
    }
    Buffer(const std::vector<T> &v) : Buffer(v.size()) { put(v); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<T> &v) {
        if (v.size() != n)
            throw std::runtime_error("size");
        ck(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
void near(const std::vector<float> &a, const std::vector<float> &b, float tolerance = 1e-4) {
    if (a.size() != b.size())
        throw std::runtime_error("size");
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::abs(a[i] - b[i]) / (1 + std::abs(b[i])) > tolerance)
            throw std::runtime_error("scaled parity error at " + std::to_string(i));
}
int main() {
    try {
        std::mt19937 rng(53);
        std::uniform_real_distribution<float> rf(-1, 1);
        auto random = [&](int n) {
            std::vector<float> v(n);
            for (auto &x : v)
                x = rf(rng);
            return v;
        };
        for (int B : {1, 3, 4, 5, 9}) {
            int n = 128;
            Buffer<float> x(random(n * B)), w(random(n * 4)), h(random(n * 3)), seq(h.get()), a(n * B),
                b(n * B);
            k::glm_conv_batch(x.p, w.p, h.p, a.p, n, 4, B, nullptr);
            for (int t = 0; t < B; ++t)
                k::glm_conv(x.p + t * n, w.p, seq.p, b.p + t * n, n, 4, nullptr);
            near(a.get(), b.get(), 0);
            near(h.get(), seq.get(), 0);
            int H = 64;
            Buffer<float> r(random(4 * H * B)), p(random(24 * B)), base(random(24)),
                scale(std::vector<float>{.1, .2, .3}), c(24 * B), cs(24 * B), y(random(H * B)), read(H * B),
                reads(H * B), write(4 * H * B), writes(4 * H * B);
            k::glm_mhc_read_batch(r.p, p.p, base.p, scale.p, c.p, read.p, H, 20, 1e-6, B, nullptr);
            k::glm_mhc_write_batch(r.p, c.p, y.p, write.p, H, B, nullptr);
            for (int t = 0; t < B; ++t) {
                k::glm_mhc_read(r.p + t * 4 * H, p.p + t * 24, base.p, scale.p, cs.p + t * 24,
                                reads.p + t * H, H, 20, 1e-6, nullptr);
                k::glm_mhc_write(r.p + t * 4 * H, cs.p + t * 24, y.p + t * H, writes.p + t * 4 * H, H,
                                 nullptr);
            }
            near(c.get(), cs.get(), 0);
            near(read.get(), reads.get(), 0);
            near(write.get(), writes.get(), 0);
            int heads = 2, dim = 128;
            Buffer<float> raw(random(B * heads * dim)), bias(random(heads * dim)),
                neg_a(std::vector<float>(heads, -.8)), decay(B * heads * dim), decays(B * heads * dim),
                norm(random(dim)), out(B * heads * dim), outs(B * heads * dim);
            k::glm_kda_gate_batch(raw.p, bias.p, neg_a.p, decay.p, heads, dim, -5, B, nullptr);
            k::glm_kda_output_batch(raw.p, raw.p, norm.p, out.p, heads, dim, 1e-5, B, nullptr);
            for (int t = 0; t < B; ++t) {
                k::glm_kda_gate(raw.p + t * heads * dim, bias.p, neg_a.p, decays.p + t * heads * dim, heads,
                                dim, -5, nullptr);
                k::glm_kda_output(raw.p + t * heads * dim, raw.p + t * heads * dim, norm.p,
                                  outs.p + t * heads * dim, heads, dim, 1e-5, nullptr);
            }
            near(decay.get(), decays.get(), 0);
            near(out.get(), outs.get(), 0);
        }
        for (int B : {1, 9, 64, 129}) {
            int H = 2, D = 128, N = H * D;
            Buffer<float> state(random(H * D * D)), seq(state.get()), q(random(B * N)), key(random(B * N)),
                v(random(B * N)), decay(std::vector<float>(B * N, -.02f)), beta(random(B * H)), out(B * N),
                ref(B * N);
            for (int t = 0; t < B; t += 64)
                k::glm_kda_chunk(state.p, q.p + t * N, key.p + t * N, v.p + t * N, decay.p + t * N,
                                 beta.p + t * H, out.p + t * N, H, D, std::min(64, B - t), nullptr);
            for (int t = 0; t < B; ++t)
                k::glm_kda_step(seq.p, q.p + t * N, key.p + t * N, v.p + t * N, decay.p + t * N,
                                beta.p + t * H, ref.p + t * N, H, D, nullptr);
            near(state.get(), seq.get(), 1e-6);
            near(out.get(), ref.get(), 1e-6);
        }
        for (int pos : {0, 1, 2, 3, 4, 4093}) {
            int B = 9, dim = 32, pool = 4, total = (pos + B) / pool;
            Buffer<float> keys(random(B * dim)), gates(random(B * dim)), ape(random(pool * dim)),
                pk(random(pool * dim)), pg(random(pool * dim)), sk(pk.get()), sg(pg.get()),
                pooled(std::max(1, total) * dim), expected(std::max(1, total) * dim);
            k::glm_index_prepare(keys.p, gates.p, ape.p, pk.p, pg.p, pooled.p, pos, B, pool, dim, nullptr);
            for (int t = 0; t < B; ++t) {
                ck(cudaMemcpy(sk.p + ((pos + t) % pool) * dim, keys.p + t * dim, dim * 4,
                              cudaMemcpyDeviceToDevice));
                ck(cudaMemcpy(sg.p + ((pos + t) % pool) * dim, gates.p + t * dim, dim * 4,
                              cudaMemcpyDeviceToDevice));
                if ((pos + t + 1) % pool == 0)
                    k::glm_index_pool(sk.p, sg.p, ape.p, expected.p + ((pos + t + 1) / pool - 1) * dim, pool,
                                      dim, nullptr);
            }
            near(pk.get(), sk.get(), 0);
            near(pg.get(), sg.get(), 0);
            near(pooled.get(), expected.get(), 0);
        }
        for (bool ties : {false, true}) {
            int B = 9, pos = 4093, pools = (pos + B) / 4, stride = 2052;
            auto hs = ties ? std::vector<float>(B * pools, 0) : random(B * pools);
            Buffer<float> scores(hs);
            Buffer<int> ids(B * stride), counts(B), ref(B * stride), rc(B);
            k::glm_index_select_batch(scores.p, ids.p, counts.p, pools, pos, B, 4, 2048, stride, nullptr);
            for (int t = 0; t < B; ++t)
                k::glm_index_select(scores.p + t * pools, ref.p + t * stride, rc.p + t, pos + t + 1, 4, 2048,
                                    nullptr);
            auto a = ids.get(), b = ref.get(), ns = counts.get();
            if (ns != rc.get())
                throw std::runtime_error("selection counts");
            for (int t = 0; t < B; ++t)
                for (int i = 0; i < ns[t]; ++i)
                    if (a[t * stride + i] != b[t * stride + i])
                        throw std::runtime_error("selection order");
        }
        {
            int B = 9, E = 288, K = 8;
            Buffer<float> logits(random(B * E)), bias(random(E)), weights(B * K), seqw(B * K);
            Buffer<int> ids(B * K), seqids(B * K), bounds(E + 1), dest(B * K), source(B * K), cursor(E);
            k::glm_route_batch(logits.p, bias.p, ids.p, weights.p, E, K, 2.5, B, nullptr);
            for (int t = 0; t < B; ++t)
                k::glm_router(logits.p + t * E, bias.p, seqids.p + t * K, seqw.p + t * K, E, K, 2.5, nullptr);
            if (ids.get() != seqids.get())
                throw std::runtime_error("routing IDs");
            near(weights.get(), seqw.get(), 0);
            k::glm_group_routes(ids.p, bounds.p, dest.p, source.p, cursor.p, E, K, B, nullptr);
            auto hb = bounds.get(), hd = dest.get(), hs = source.get(), hi = ids.get();
            std::vector<int> seen(B * K);
            for (int e = 0; e < E; ++e)
                for (int j = hb[e]; j < hb[e + 1]; ++j) {
                    if (hi[hd[j]] != e || hs[j] != hd[j] / K || seen[hd[j]]++)
                        throw std::runtime_error("CSR routing");
                }
            for (int v : seen)
                if (v != 1)
                    throw std::runtime_error("missing route");
        }
        {
            int B = 3, H = 2, L = 32, C = 2052, N = 512;
            Buffer<float> cache(random(N * L)), q(random(B * H * L)), gathered(B * C * L), scores(B * H * C),
                out(B * H * L), ref(B * H * L);
            std::vector<int> ids(B * C, 0), counts{3, 127, 512};
            for (int t = 0; t < B; ++t)
                for (int i = 0; i < counts[t]; ++i)
                    ids[t * C + i] = (i * 7) % N;
            Buffer<int> di(ids), dc(counts);
            k::glm_mla_gather(cache.p, di.p, dc.p, gathered.p, B, C, L, nullptr);
            cublasHandle_t h;
            cublasCreate(&h);
            cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
            float one = 1, zero = 0;
            auto cb = [](cublasStatus_t e) {
                if (e != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("blas");
            };
            cb(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, C, H, L, &one, gathered.p, L, C * L,
                                         q.p, L, H * L, &zero, scores.p, C, H * C, B));
            k::glm_mla_softmax(scores.p, dc.p, H, C, B, 1.f / 16, nullptr);
            cb(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, L, H, C, &one, gathered.p, L, C * L,
                                         scores.p, C, H * C, &zero, out.p, L, H * L, B));
            for (int t = 0; t < B; ++t)
                k::glm_mla(q.p + t * H * L, cache.p, di.p + t * C, dc.p + t, ref.p + t * H * L, H, L,
                           1.f / 16, nullptr);
            near(out.get(), ref.get());
            cublasDestroy(h);
        }
        ck(cudaDeviceSynchronize());
        std::cout << "GLM prefill batch, boundary, CSR, tie and sparse attention parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
