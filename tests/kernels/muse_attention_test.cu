// tests/kernels/muse_attention_test.cu - Muse Spark decode attention (kernels/muse.hpp muse_attention), one block per
// head and split-K, against a double-precision reference on synthetic caches longer than one split, on full and
// ring layers; and a graph captured at one position replayed at others (the position is read from the device).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/muse.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
constexpr int QH = 128, KVH = 16, HD = 64, W = KVH * HD;

void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// out[h] over the n keys in slots 0 .. n-1 (the order does not matter), K and V as stored (f16)
std::vector<float> reference(const std::vector<float>& q, const std::vector<uint16_t>& kc, const std::vector<uint16_t>& vc,
                             int n, float scale) {
    std::vector<float> out((size_t) QH * HD);
    std::vector<double> s((size_t) n);
    for (int h = 0; h < QH; ++h) {
        const int kvh = h / (QH / KVH);
        double m = -INFINITY;
        for (int j = 0; j < n; ++j) {
            double d = 0;
            for (int i = 0; i < HD; ++i) d += (double) q[(size_t) h * HD + i] * k::f32_from_f16(kc[(size_t) j * W + kvh * HD + i]);
            s[(size_t) j] = d * scale;
            m = std::max(m, s[(size_t) j]);
        }
        double l = 0;
        std::vector<double> acc(HD, 0.0);
        for (int j = 0; j < n; ++j) {
            const double e = std::exp(s[(size_t) j] - m);
            l += e;
            for (int i = 0; i < HD; ++i) acc[(size_t) i] += e * k::f32_from_f16(vc[(size_t) j * W + kvh * HD + i]);
        }
        for (int i = 0; i < HD; ++i) out[(size_t) h * HD + i] = (float) (acc[(size_t) i] / l);
    }
    return out;
}

double max_err(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0, scale = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, (double) std::fabs(a[i] - b[i]));
        scale = std::max(scale, (double) std::fabs(b[i]));
    }
    return e / scale;
}
}  // namespace

int main() {
    const int slots = 40000;
    std::mt19937 rng(20261016);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q((size_t) QH * HD);
    for (float& v : q) v = nd(rng);
    std::vector<uint16_t> kc((size_t) slots * W), vc((size_t) slots * W);
    for (auto& v : kc) v = k::f16_from_f32(nd(rng));
    for (auto& v : vc) v = k::f16_from_f32(nd(rng));
    const float scale = 1.f / 8.f;

    cudaStream_t st;
    gpu(cudaStreamCreate(&st), "stream");
    float *dq, *dout, *dpart;
    uint16_t *dk, *dv;
    int32_t* dpos;
    const int max_splits = 64;
    gpu(cudaMalloc(&dq, q.size() * 4), "malloc");
    gpu(cudaMalloc(&dout, q.size() * 4), "malloc");
    gpu(cudaMalloc(&dpart, k::muse_attention_scratch_floats(QH, HD, max_splits) * 4), "malloc");
    gpu(cudaMalloc(&dk, kc.size() * 2), "malloc");
    gpu(cudaMalloc(&dv, vc.size() * 2), "malloc");
    gpu(cudaMalloc(&dpos, 4), "malloc");
    gpu(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "upload");
    gpu(cudaMemcpy(dk, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice), "upload");
    gpu(cudaMemcpy(dv, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice), "upload");

    std::vector<float> out(q.size());
    auto run = [&](int pos, int ring, int splits) {
        gpu(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice), "pos");
        k::muse_attention(dq, dk, dv, dout, QH, KVH, HD, dpos, ring, scale, st, splits, dpart);
        gpu(cudaStreamSynchronize(st), "attention");
        gpu(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost), "download");
        return out;
    };
    int failures = 0;
    const double tol = 2e-5;
    struct Case { int pos, ring; };
    for (const Case c : {Case{0, 0}, Case{10, 0}, Case{299, 0}, Case{4999, 0}, Case{slots - 1, 0}, Case{100, 2048},
                         Case{4999, 2048}}) {
        const int n = c.ring ? std::min(c.pos + 1, c.ring) : c.pos + 1;
        const auto ref = reference(q, kc, vc, n, scale);
        for (int splits : {1, 7, 16, 64}) {
            const double e = max_err(run(c.pos, c.ring, splits), ref);
            const bool ok = e <= tol;
            failures += !ok;
            std::printf("pos %5d ring %4d (%5d keys) splits %2d: max err %.2e  %s\n", c.pos, c.ring, n, splits, e,
                        ok ? "ok" : "FAIL");
        }
    }

    // a graph captured at one position, replayed at others, equals the direct call bit for bit
    for (int splits : {1, 16}) {
        int pos0 = 11;
        gpu(cudaMemcpy(dpos, &pos0, 4, cudaMemcpyHostToDevice), "pos");
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        gpu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        k::muse_attention(dq, dk, dv, dout, QH, KVH, HD, dpos, 0, scale, st, splits, dpart);
        gpu(cudaStreamEndCapture(st, &graph), "end capture");
        gpu(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        for (int pos : {11, 700, 9000}) {
            const std::vector<float> direct = run(pos, 0, splits);
            gpu(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice), "pos");
            gpu(cudaGraphLaunch(exec, st), "launch");
            gpu(cudaStreamSynchronize(st), "replay");
            std::vector<float> replay(q.size());
            gpu(cudaMemcpy(replay.data(), dout, replay.size() * 4, cudaMemcpyDeviceToHost), "download");
            const bool ok = replay == direct;
            failures += !ok;
            std::printf("graph replay, splits %2d, pos %5d: %s\n", splits, pos, ok ? "identical" : "FAIL");
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }
    std::printf("muse_attention_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
