#include "ggml.h"
#include "strata/core/model.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <set>

void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Allocation {
    void *p = nullptr;
    explicit Allocation(size_t bytes) { check(cudaMalloc(&p, bytes)); }
    ~Allocation() { cudaFree(p); }
    Allocation(const Allocation &) = delete;
};
struct Stream {
    cudaStream_t p = nullptr;
    Stream() { check(cudaStreamCreate(&p)); }
    ~Stream() { cudaStreamDestroy(p); }
};
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: glm_quant_parity <GLM shard.gguf>\n";
        return 2;
    }
    try {
        strata::core::ModelArtifact artifact(argv[1]);
        std::set<int> pending = {10, 11, 21, 22, 23};
        Stream stream;
        std::mt19937 rng(53);
        std::normal_distribution<float> nd;
        for (const auto &[name, t] : artifact.tensors()) {
            const int type = t.tensor->type;
            if (!pending.count(type) || !name.ends_with("_exps.weight"))
                continue;
            const int cols = t.tensor->shape[0], rows = 8;
            const size_t bytes = strata::kernels::iq_row_bytes(type, cols) * rows;
            const auto *traits = ggml_get_type_traits((ggml_type)type);
            std::vector<float> ref((size_t)rows * cols), got(ref.size());
            for (int r = 0; r < rows; ++r)
                traits->to_float(t.data() + r * bytes / rows, ref.data() + r * cols, cols);
            Allocation w(bytes), out(ref.size() * 4);
            check(cudaMemcpy(w.p, t.data(), bytes, cudaMemcpyHostToDevice));
            strata::kernels::iq_dequant_f32(type, w.p, ref.size(), (float *)out.p, stream.p);
            check(cudaStreamSynchronize(stream.p));
            check(cudaMemcpy(got.data(), out.p, got.size() * 4, cudaMemcpyDeviceToHost));
            double maxerr = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                if (!std::isfinite(got[i]))
                    throw std::runtime_error("non-finite dequant");
                maxerr = std::max(maxerr, std::abs((double)got[i] - ref[i]) / (1 + std::abs(ref[i])));
            }
            if (maxerr > 1e-7)
                throw std::runtime_error("dequant mismatch for type " + std::to_string(type));
            std::vector<float> x(cols * 2);
            for (auto &v : x)
                v = nd(rng);
            Allocation dx(x.size() * 4), qx(cols / 32 * 36 * 2), y(rows * 4 * 2);
            check(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            strata::kernels::quantize_q8_1_rows((float *)dx.p, 2, cols, qx.p, stream.p);
            strata::kernels::native_mmvq(type, w.p, qx.p, (float *)y.p, cols, rows, 2, stream.p);
            check(cudaStreamSynchronize(stream.p));
            std::vector<uint8_t> packed(cols / 32 * 36 * 2);
            check(cudaMemcpy(packed.data(), qx.p, packed.size(), cudaMemcpyDeviceToHost));
            std::vector<float> outputs(rows * 2);
            check(cudaMemcpy(outputs.data(), y.p, outputs.size() * 4, cudaMemcpyDeviceToHost));
            double error = 0, denom = 0;
            for (int c = 0; c < 2; ++c)
                for (int r = 0; r < rows; ++r) {
                    double expected = 0;
                    for (int i = 0; i < cols; ++i) {
                        const auto *block = packed.data() + ((size_t)c * cols / 32 + i / 32) * 36;
                        uint16_t half;
                        std::memcpy(&half, block, 2);
                        const float scale = ggml_fp16_to_fp32(half);
                        expected += (double)ref[(size_t)r * cols + i] * scale * (int8_t)block[4 + i % 32];
                    }
                    if (!std::isfinite(outputs[c * rows + r]))
                        throw std::runtime_error("non-finite dot");
                    error += std::abs(outputs[c * rows + r] - expected);
                    denom += std::abs(expected);
                }
            const double relative = error / (denom + 1e-30);
            std::cout << "type=" << type << " dequant max=" << maxerr << " same-Q8 dot rel=" << relative
                      << '\n';
            // The pinned IQ2_S CUDA dot truncates its integer scaled sum / 4.
            // This is a small additional error beyond activation quantization.
            const double tolerance = type == 22 ? 1e-4 : 1e-5;
            if (relative > tolerance)
                throw std::runtime_error("same-activation dot mismatch for type " + std::to_string(type));
            pending.erase(type);
        }
        if (!pending.empty())
            throw std::runtime_error("fixture is missing required GLM quantizations");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
