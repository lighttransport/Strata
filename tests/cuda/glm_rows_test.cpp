// glm_rows.hpp kernels against their one-row calls: every output and every state must be bitwise equal.
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include "strata/kernels/glm_rows.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace k = strata::kernels;
static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    float *p = nullptr; size_t n;
    explicit Buffer(size_t count) : n(count) { ck(cudaMalloc(&p, count * 4)); }
    ~Buffer() { cudaFree(p); }
    void fill(const std::vector<float> &v) { ck(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice)); }
    std::vector<float> read() const { std::vector<float> v(n); ck(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost)); return v; }
};
static uint64_t seed = 12345;
static std::vector<float> random(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto &x : v) { seed = seed * 6364136223846793005ull + 1442695040888963407ull; x = lo + (hi - lo) * float(seed >> 40) / float(1 << 24); }
    return v;
}
static int failures = 0;
static void same(const std::string &what, const Buffer &a, const Buffer &b) {
    const auto x = a.read(), y = b.read();
    if (std::memcmp(x.data(), y.data(), x.size() * 4)) { std::cerr << "FAIL " << what << '\n'; ++failures; }
}
// Per-row states: `rows` buffers for the batched call and `rows` copies for the one-row calls.
struct States {
    std::vector<std::unique_ptr<Buffer>> a, b;
    States(int rows, size_t n, float lo, float hi) {
        for (int r = 0; r < rows; ++r) {
            auto v = random(n, lo, hi);
            a.push_back(std::make_unique<Buffer>(n)); b.push_back(std::make_unique<Buffer>(n));
            a.back()->fill(v); b.back()->fill(v);
        }
    }
    k::GlmRowPointers table() const { k::GlmRowPointers t{}; for (size_t r = 0; r < a.size(); ++r) t.p[r] = a[r]->p; return t; }
    void check(const std::string &what) const { for (size_t r = 0; r < a.size(); ++r) same(what + " row " + std::to_string(r), *a[r], *b[r]); }
};
static void run(int rows, cudaStream_t s) {
    const int heads = 64, dim = 128, n = heads * dim, kernel = 4, latent = 512, pool = 4, idim = 128, capacity = 64;
    const std::string tag = " rows=" + std::to_string(rows);
    Buffer x(size_t(rows) * n), w(size_t(n) * kernel), ya(size_t(rows) * n), yb(size_t(rows) * n);
    x.fill(random(x.n, -2, 2)); w.fill(random(w.n, -1, 1));
    {   // convolution
        States history(rows, size_t(n) * (kernel - 1), -2, 2);
        k::glm_conv_rows(x.p, w.p, history.table(), ya.p, n, kernel, rows, s);
        for (int r = 0; r < rows; ++r) k::glm_conv(x.p + size_t(r) * n, w.p, history.b[r]->p, yb.p + size_t(r) * n, n, kernel, s);
        ck(cudaStreamSynchronize(s));
        same("conv" + tag, ya, yb); history.check("conv history" + tag);
    }
    {   // gate
        Buffer bias(n), a(heads); bias.fill(random(n, -1, 1)); a.fill(random(heads, -3, -0.1f));
        k::glm_kda_gate_rows(x.p, bias.p, a.p, ya.p, heads, dim, -5.f, rows, s);
        for (int r = 0; r < rows; ++r) k::glm_kda_gate(x.p + size_t(r) * n, bias.p, a.p, yb.p + size_t(r) * n, heads, dim, -5.f, s);
        ck(cudaStreamSynchronize(s));
        same("gate" + tag, ya, yb);
    }
    for (int columns : {32, 64, 128}) {   // recurrence
        Buffer q(size_t(rows) * n), key(size_t(rows) * n), v(size_t(rows) * n), g(size_t(rows) * n), beta(size_t(rows) * heads);
        q.fill(random(q.n, -1, 1)); key.fill(random(key.n, -1, 1)); v.fill(random(v.n, -1, 1));
        g.fill(random(g.n, -0.5f, -0.01f)); beta.fill(random(beta.n, -2, 2));
        States state(rows, size_t(heads) * dim * dim, -0.5f, 0.5f);
        k::glm_kda_chunk_rows(state.table(), q.p, key.p, v.p, g.p, beta.p, ya.p, heads, dim, rows, s, columns);
        for (int r = 0; r < rows; ++r)
            k::glm_kda_chunk(state.b[r]->p, q.p + size_t(r) * n, key.p + size_t(r) * n, v.p + size_t(r) * n,
                             g.p + size_t(r) * n, beta.p + size_t(r) * heads, yb.p + size_t(r) * n, heads, dim, 1, s, columns, 1);
        ck(cudaStreamSynchronize(s));
        same("kda" + tag + " columns=" + std::to_string(columns), ya, yb);
        state.check("kda state" + tag + " columns=" + std::to_string(columns));
    }
    k::GlmRowPositions position{};
    for (int r = 0; r < rows; ++r) position.p[r] = (r * 7 + 3) % (capacity - 1);   // every pool phase occurs
    {   // latent cache rows
        Buffer latent_rows(size_t(rows) * latent); latent_rows.fill(random(latent_rows.n, -1, 1));
        States cache(rows, size_t(capacity) * latent, -1, 1);
        k::glm_cache_rows_each(latent_rows.p, cache.table(), position, latent, rows, s);
        for (int r = 0; r < rows; ++r)
            ck(cudaMemcpyAsync(cache.b[r]->p + size_t(position.p[r]) * latent, latent_rows.p + size_t(r) * latent,
                               size_t(latent) * 4, cudaMemcpyDeviceToDevice, s));
        ck(cudaStreamSynchronize(s));
        cache.check("cache" + tag);
    }
    {   // indexer keys
        Buffer keys(size_t(rows) * idim), gates(size_t(rows) * idim), ape(size_t(pool) * idim);
        keys.fill(random(keys.n, -1, 1)); gates.fill(random(gates.n, -2, 2)); ape.fill(random(ape.n, -1, 1));
        States pk(rows, size_t(pool) * idim, -1, 1), pg(rows, size_t(pool) * idim, -2, 2), pooled(rows, size_t(capacity / pool) * idim, -1, 1);
        k::glm_index_prepare_rows(keys.p, gates.p, ape.p, pk.table(), pg.table(), pooled.table(), position, rows, pool, idim, s);
        for (int r = 0; r < rows; ++r)
            k::glm_index_prepare(keys.p + size_t(r) * idim, gates.p + size_t(r) * idim, ape.p, pk.b[r]->p, pg.b[r]->p,
                                 pooled.b[r]->p, position.p[r], 1, pool, idim, s);
        ck(cudaStreamSynchronize(s));
        pk.check("pending keys" + tag); pg.check("pending gates" + tag); pooled.check("pooled" + tag);
    }
    {   // index selection: positions past and before 2048 keys, so some rows select and some keep every key
        const int pools = 1024, stride = 2052;
        k::GlmRowPositions at{};
        for (int r = 0; r < rows; ++r) at.p[r] = 1500 + r * 97;
        Buffer scores(size_t(rows) * pools); scores.fill(random(scores.n, -4, 4));
        int *ids = nullptr, *counts = nullptr;
        ck(cudaMalloc(&ids, size_t(2) * rows * stride * 4)); ck(cudaMalloc(&counts, size_t(2) * rows * 4));
        ck(cudaMemset(ids, 0, size_t(2) * rows * stride * 4)); ck(cudaMemset(counts, 0, size_t(2) * rows * 4));
        k::glm_index_select_rows(scores.p, ids, counts, pools, at, rows, 4, 2048, stride, s);
        for (int r = 0; r < rows; ++r)
            k::glm_index_select_batch(scores.p + size_t(r) * pools, ids + size_t(rows + r) * stride, counts + rows + r,
                                      pools, at.p[r], 1, 4, 2048, stride, s);
        std::vector<int> hi(size_t(2) * rows * stride), hc(size_t(2) * rows);
        ck(cudaMemcpy(hi.data(), ids, hi.size() * 4, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(hc.data(), counts, hc.size() * 4, cudaMemcpyDeviceToHost));
        cudaFree(ids); cudaFree(counts);
        if (std::memcmp(hi.data(), hi.data() + size_t(rows) * stride, size_t(rows) * stride * 4) ||
            std::memcmp(hc.data(), hc.data() + rows, size_t(rows) * 4)) { std::cerr << "FAIL selection" << tag << '\n'; ++failures; }
    }
    for (auto [in, out] : {std::pair{256, 512}, std::pair{512, 256}}) {   // MLA absorption: rows=N against rows=1
        const int h = 8, ldx = h * in, ldy = h * out;
        Buffer wt(size_t(h) * in * out), xs(size_t(rows) * ldx), ys(size_t(rows) * ldy), one(size_t(rows) * ldy);
        wt.fill(random(wt.n, -1, 1)); xs.fill(random(xs.n, -1, 1));
        k::glm_mla_absorb_rows(wt.p, xs.p, ys.p, in, out, h, rows, ldx, ldy, s);
        for (int r = 0; r < rows; ++r)
            k::glm_mla_absorb_rows(wt.p, xs.p + size_t(r) * ldx, one.p + size_t(r) * ldy, in, out, h, 1, ldx, ldy, s);
        ck(cudaStreamSynchronize(s));
        same("absorb" + tag + " in=" + std::to_string(in), ys, one);
        const auto W = wt.read(), X = xs.read(), Y = ys.read();
        double worst = 0;
        for (int r = 0; r < rows; ++r) for (int hh = 0; hh < h; ++hh) for (int o = 0; o < out; ++o) {
            double sum = 0;
            for (int i = 0; i < in; ++i) sum += double(W[(size_t(hh) * out + o) * in + i]) * X[size_t(r) * ldx + size_t(hh) * in + i];
            worst = std::max(worst, std::abs(sum - Y[size_t(r) * ldy + size_t(hh) * out + o]));
        }
        if (worst > 1e-3) { std::cerr << "FAIL absorb reference" << tag << " error " << worst << '\n'; ++failures; }
    }
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t s = nullptr;
    try {
        ck(cudaStreamCreate(&s));
        for (int rows : {1, 5, 32}) run(rows, s);
    } catch (const std::exception &e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
    cudaStreamDestroy(s);
    if (failures) return 1;
    std::cout << "glm rows kernels: bitwise equal to one-row calls (rows 1, 5, 32)\n";
    return 0;
}
