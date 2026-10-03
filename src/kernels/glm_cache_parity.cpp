// Real mixed-quantization experts: shared prefill/decode storage must preserve
// every MMQ and native grouped product bit for bit, including padded strides.
#include "ggml.h"
#include "strata/core/model.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/gemm.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

namespace k = strata::kernels;
namespace mmq = strata::prefill::mmq;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void *p = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(n) { ck(cudaMalloc(&p, n)); ck(cudaMemset(p, 0, n)); }
    ~Buffer() { cudaFree(p); }
    void put(const void *source) { ck(cudaMemcpy(p, source, bytes, cudaMemcpyHostToDevice)); }
    float *f() { return (float *)p; }
    int *i() { return (int *)p; }
};
void identical(Buffer &a, Buffer &b) {
    std::vector<float> x(a.bytes / 4), y(b.bytes / 4);
    ck(cudaMemcpy(x.data(), a.p, a.bytes, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(y.data(), b.p, b.bytes, cudaMemcpyDeviceToHost));
    if (x != y) throw std::runtime_error("shared expert layout changed a product");
}
int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "usage: glm_cache_parity <model shard>\n"; return 2; }
    try {
        strata::core::ModelArtifact model(argv[1]);
        std::set<std::pair<int, int>> tested;
        cudaStream_t stream;
        ck(cudaStreamCreate(&stream));
        Buffer workspace(128 * 1024 * 1024);
        mmq::Context context(workspace.p, workspace.bytes);
        {
            constexpr int B = 3, T = 13, N = 64, K = 256;
            std::mt19937 rng(753);
            std::uniform_real_distribution<float> random(-1, 1);
            std::vector<ggml_fp16_t> x(B * T * K), w(B * N * K);
            for (auto &v : x) v = ggml_fp32_to_fp16(random(rng));
            for (auto &v : w) v = ggml_fp32_to_fp16(random(rng));
            Buffer dx(x.size() * 2), dw(w.size() * 2), result(B * T * N * 4), scratch(65536 * 2);
            dx.put(x.data()); dw.put(w.data());
            strata::prefill::Gemm gemm;
            std::string error;
            if (!gemm.init_external(stream, (uint16_t *)scratch.p, scratch.bytes / 2, workspace.p, workspace.bytes, error))
                throw std::runtime_error(error);
            gemm.f16_batched((uint16_t *)dx.p, (uint16_t *)dw.p, result.f(), T, N, K, B);
            std::vector<float> actual(B * T * N);
            ck(cudaMemcpy(actual.data(), result.p, result.bytes, cudaMemcpyDeviceToHost));
            for (int b = 0; b < B; ++b) for (int t = 0; t < T; ++t) for (int n = 0; n < N; ++n) {
                double sum = 0, magnitude = 0;
                for (int k = 0; k < K; ++k) {
                    const double term = double(ggml_fp16_to_fp32(x[(b * T + t) * K + k])) *
                        ggml_fp16_to_fp32(w[(b * N + n) * K + k]);
                    sum += term; magnitude += std::abs(term);
                }
                const float value = actual[(b * T + t) * N + n];
                if (!std::isfinite(value) || std::abs(value - sum) > 2e-6 * magnitude)
                    throw std::runtime_error("FP16 batched GEMM differs from independent double oracle");
            }
            std::cout << "FP16 batched GEMM double oracle passed\n";
        }
        for (size_t l = 0; l < model.descriptor().layers.size(); ++l) {
            const auto &layer = model.descriptor().layers[l];
            if (layer.ffn != strata::core::FfnKind::Moe) continue;
            const auto prefix = "blk." + std::to_string(l) + ".";
            const auto &g = model.at(prefix + "ffn_gate_exps.weight");
            const auto &u = model.at(prefix + "ffn_up_exps.weight");
            const auto &d = model.at(prefix + "ffn_down_exps.weight");
            if (!tested.emplace(g.tensor->type, d.tensor->type).second) continue;
            const int H = model.descriptor().hidden, F = layer.intermediate, E = 16, B = 32;
            const size_t gh = g.bytes / 288, db = d.bytes / 288, blob = 2 * gh + db;
            const size_t align = std::lcm(size_t(16), std::lcm(ggml_type_size((ggml_type)g.tensor->type),
                                                             ggml_type_size((ggml_type)d.tensor->type)));
            const size_t stride = (blob + align - 1) / align * align;
            std::vector<uint8_t> old(E * blob + 16384), shared(E * stride + 16384), native(E * blob);
            for (int e = 0; e < E; ++e) {
                std::memcpy(old.data() + e * 2 * gh, g.data() + e * gh, gh);
                std::memcpy(old.data() + e * 2 * gh + gh, u.data() + e * gh, gh);
                std::memcpy(old.data() + E * 2 * gh + e * db, d.data() + e * db, db);
                std::memcpy(shared.data() + e * stride, old.data() + e * 2 * gh, 2 * gh);
                std::memcpy(shared.data() + e * stride + 2 * gh, old.data() + E * 2 * gh + e * db, db);
                std::memcpy(native.data() + e * blob, shared.data() + e * stride, blob);
            }
            Buffer legacy(old.size()), packed(shared.size()), contiguous(native.size());
            legacy.put(old.data()); packed.put(shared.data()); contiguous.put(native.data());
            std::mt19937 rng(53);
            std::normal_distribution<float> normal;
            std::vector<float> x(B * H);
            for (auto &v : x) v = normal(rng);
            Buffer activation(x.size() * 4); activation.put(x.data());
            std::vector<int> bounds(E + 1);
            for (int e = 0; e < E; ++e) bounds[e + 1] = bounds[e] + e % 4;
            const int rows = bounds.back();
            // Pinned MMQ tiles read the route-ID tail speculatively, as in the
            // production route buffers. Keep its 128-entry zero guard.
            std::vector<int> source(rows + 128), destination(rows + 128);
            for (int j = 0; j < rows; ++j) { source[j] = (j * 7) % B; destination[j] = rows - 1 - j; }
            Buffer starts(bounds.size() * 4), src(source.size() * 4), dst(destination.size() * 4), identity((rows + 128) * 4);
            starts.put(bounds.data()); src.put(source.data()); dst.put(destination.data());
            mmq::iota(identity.i(), rows + 128, stream);
            Buffer qx(mmq::q8_bytes(rows, H)), qh(mmq::q8_bytes(rows, F));
            Buffer ga(rows * 2 * F * 4), gb(ga.bytes), ha(rows * F * 4), hb(ha.bytes);
            Buffer da(rows * H * 4), dp(da.bytes);
            mmq::quantize(activation.f(), src.i(), qx.p, g.tensor->type, H, H, rows, stream);
            for (int variant = 0; variant < 2; ++variant) {
                mmq::Product product{variant ? packed.p : legacy.p, (int)g.tensor->type, 2 * F, H,
                    variant ? stride : 2 * gh, E, qx.p, starts.i(), identity.i(), rows, 3,
                    variant ? gb.f() : ga.f(), 2 * F};
                context.run(product, stream);
                mmq::swiglu(product.dst, variant ? hb.f() : ha.f(), rows, F, false, stream, layer.swiglu_limit);
            }
            ck(cudaStreamSynchronize(stream)); identical(ga, gb); identical(ha, hb);
            mmq::quantize(ha.f(), nullptr, qh.p, d.tensor->type, F, F, rows, stream);
            for (int variant = 0; variant < 2; ++variant) {
                mmq::Product product{variant ? (char *)packed.p + 2 * gh : (char *)legacy.p + E * 2 * gh,
                    (int)d.tensor->type, H, F, variant ? stride : db, E, qh.p, starts.i(), dst.i(),
                    rows, 3, variant ? dp.f() : da.f(), H};
                context.run(product, stream);
            }
            ck(cudaStreamSynchronize(stream)); identical(da, dp);
            auto layout = k::native_expert_layout(g.tensor->type, d.tensor->type, H, F);
            layout.swiglu_limit = layer.swiglu_limit;
            for (auto &token : source) token %= 8;
            src.put(source.data());
            Buffer nq(k::native_q8_1_bytes(H, 8)), scratch(k::native_expert_scratch_bytes(rows, F));
            Buffer ptr(E * 8), count(4), na(da.bytes), nb(da.bytes);
            count.put(&E);
            k::native_quantize_q8_1(activation.f(), nq.p, H, 8, stream);
            for (int variant = 0; variant < 2; ++variant) {
                std::vector<unsigned long long> addresses(E);
                for (int e = 0; e < E; ++e) addresses[e] = (unsigned long long)(variant ? packed.p : contiguous.p)
                    + e * (variant ? stride : blob);
                ptr.put(addresses.data());
                k::native_expert_grouped(layout, (const unsigned long long *)ptr.p, starts.i(), count.i(),
                    dst.i(), src.i(), E, rows, nq.p, scratch.p, variant ? nb.f() : na.f(), stream);
                ck(cudaStreamSynchronize(stream));
            }
            identical(na, nb);
            std::cout << "layer=" << l << " gu_type=" << g.tensor->type << " down_type=" << d.tensor->type
                      << " padding=" << stride - blob << " MMQ/native products identical\n";
        }
        ck(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
