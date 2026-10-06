#include "ggml.h"
#include "strata/core/model.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
// Compare every output bit, row offsets, CTA tails, and both allocation guards.
// The independent GGML float oracle remains required on the real model blocks.
void check_coalesced(int type, const uint8_t* blocks, int cols, int available_rows, cudaStream_t stream, bool benchmark) {
    const char* original = std::getenv("STRATA_GLM_DEQUANT_COALESCED");
    struct Restore {
        bool present;
        std::string value;
        ~Restore() { if (present) setenv("STRATA_GLM_DEQUANT_COALESCED", value.c_str(), 1);
                     else unsetenv("STRATA_GLM_DEQUANT_COALESCED"); }
    } restore{original != nullptr, original ? original : ""};
    const size_t row_bytes = ggml_row_size((ggml_type)type, cols);
    Allocation input(row_bytes * 260);
    check(cudaMemcpy(input.p, blocks, row_bytes * 260, cudaMemcpyHostToDevice));
    const auto* traits = ggml_get_type_traits((ggml_type)type);
    for (int rows : {1, 3, 257}) {
        for (int format = 0; format < 3; ++format) {
            const size_t bytes = (size_t)rows * cols * (format == 2 ? 4 : 2);
            Allocation output(bytes + 128);
            std::vector<uint8_t> baseline(bytes + 128), candidate(bytes + 128);
            for (int mode = 0; mode < 2; ++mode) {
                setenv("STRATA_GLM_DEQUANT_COALESCED", mode ? "1" : "0", 1);
                check(cudaMemsetAsync(output.p, 0xa5, bytes + 128, stream));
                void* values = (uint8_t*)output.p + 64;
                if (format == 0) strata::kernels::dequant_f16(type, input.p, 3, rows, cols, (uint16_t*)values, stream);
                if (format == 1) strata::kernels::dequant_bf16(type, input.p, 3, rows, cols, (uint16_t*)values, stream);
                if (format == 2) strata::kernels::dequant_f32(type, input.p, 3, rows, cols, (float*)values, stream);
                check(cudaStreamSynchronize(stream));
                auto& actual = mode ? candidate : baseline;
                check(cudaMemcpy(actual.data(), output.p, actual.size(), cudaMemcpyDeviceToHost));
                if (!std::all_of(actual.begin(), actual.begin() + 64, [](uint8_t b){return b == 0xa5;}) ||
                    !std::all_of(actual.end() - 64, actual.end(), [](uint8_t b){return b == 0xa5;}))
                    throw std::runtime_error("coalesced dequant guard corruption");
            }
            if (baseline != candidate) throw std::runtime_error("coalesced dequant bits differ");
            if (format == 2) {
                std::vector<float> ref(cols);
                for (int r = 0; r < rows; ++r) {
                    traits->to_float(blocks + (r + 3) * row_bytes, ref.data(), cols);
                    for (int c = 0; c < cols; ++c) {
                        float got;
                        std::memcpy(&got, candidate.data() + 64 + ((size_t)r * cols + c) * 4, 4);
                        if (!std::isfinite(got) || std::abs((double)got - ref[c]) / (1 + std::abs(ref[c])) > 1e-7)
                            throw std::runtime_error("coalesced dequant independent oracle mismatch");
                    }
                }
            }
        }
    }
    std::cout << "COALESCED type=" << type << " formats=3 row0=3 rows=1,3,257 bits=identical guards=unchanged oracle=passed\n";
    if (benchmark) {
        const int rows = std::min(4096, available_rows);
        Allocation weights(row_bytes * rows), output((size_t)rows * cols * 2);
        check(cudaMemcpy(weights.p, blocks, row_bytes * rows, cudaMemcpyHostToDevice));
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        for (int mode = 0; mode < 2; ++mode) {
            setenv("STRATA_GLM_DEQUANT_COALESCED", mode ? "1" : "0", 1);
            std::vector<float> times;
            for (int trial = 0; trial < 8; ++trial) {
                check(cudaEventRecord(begin, stream));
                for (int i = 0; i < 8; ++i)
                    strata::kernels::dequant_f16(type, weights.p, 0, rows, cols, (uint16_t*)output.p, stream);
                check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
                float ms;
                check(cudaEventElapsedTime(&ms, begin, end));
                if (trial) times.push_back(ms / 8);
            }
            std::sort(times.begin(), times.end());
            std::cout << "COALESCED_BENCH type=" << type << " rows=" << rows << " cols=" << cols
                      << " mode=" << mode << " median_ms=" << times[3] << '\n';
        }
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    }
}

// Test native IQ stores independently of the generic dequant dispatcher. Odd
// block counts cover inactive warps in every candidate launch geometry.
void check_iq_coalesced(int type, const uint8_t* blocks, int64_t available, cudaStream_t stream, bool benchmark) {
    const char* original = std::getenv("STRATA_GLM_IQ_COALESCED");
    struct Restore {
        bool present;
        std::string value;
        ~Restore() { if (present) setenv("STRATA_GLM_IQ_COALESCED", value.c_str(), 1);
                     else unsetenv("STRATA_GLM_IQ_COALESCED"); }
    } restore{original != nullptr, original ? original : ""};
    const size_t block_bytes = strata::kernels::iq_row_bytes(type, 256);
    const auto* traits = ggml_get_type_traits((ggml_type)type);
    Allocation input(block_bytes * 20);
    check(cudaMemcpy(input.p, blocks, block_bytes * 20, cudaMemcpyHostToDevice));
    for (int count : {1, 3, 5, 9, 17}) {
        const int values = count * 256;
        std::vector<float> reference(values);
        traits->to_float(blocks + 3 * block_bytes, reference.data(), values);
        for (int format = 0; format < 2; ++format) {
            const size_t bytes = values * (format ? 4 : 2);
            Allocation output(bytes + 128);
            std::vector<uint8_t> baseline(bytes + 128), actual(bytes + 128);
            for (int mode : {0, 1, 2, 4, 8, 16}) {
                setenv("STRATA_GLM_IQ_COALESCED", std::to_string(mode).c_str(), 1);
                check(cudaMemsetAsync(output.p, 0xa5, bytes + 128, stream));
                const void* source = (const uint8_t*)input.p + 3 * block_bytes;
                void* destination = (uint8_t*)output.p + 64;
                if (format) strata::kernels::iq_dequant_f32(type, source, values, (float*)destination, stream);
                else strata::kernels::iq_dequant_f16(type, source, values, (uint16_t*)destination, stream);
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(actual.data(), output.p, actual.size(), cudaMemcpyDeviceToHost));
                if (!std::all_of(actual.begin(), actual.begin() + 64, [](uint8_t b){return b == 0xa5;}) ||
                    !std::all_of(actual.end() - 64, actual.end(), [](uint8_t b){return b == 0xa5;}))
                    throw std::runtime_error("IQ coalesced guard corruption");
                if (!mode) baseline = actual;
                else if (actual != baseline) throw std::runtime_error("IQ coalesced bits differ for type " + std::to_string(type));
                if (format) for (int i = 0; i < values; ++i) {
                    float got;
                    std::memcpy(&got, actual.data() + 64 + i * 4, 4);
                    if (!std::isfinite(got) || std::abs((double)got - reference[i]) / (1 + std::abs(reference[i])) > 1e-7)
                        throw std::runtime_error("IQ coalesced independent oracle mismatch");
                }
            }
        }
    }
    std::cout << "IQ_COALESCED type=" << type << " formats=2 block0=3 blocks=1,3,5,9,17 modes=0,1,2,4,8,16 bits=identical guards=unchanged oracle=passed\n";
    if (!benchmark) return;
    const int64_t values = std::min<int64_t>(4096 * 4096, available) / 256 * 256;
    Allocation weights(strata::kernels::iq_row_bytes(type, values)), output(values * 2);
    check(cudaMemcpy(weights.p, blocks, strata::kernels::iq_row_bytes(type, values), cudaMemcpyHostToDevice));
    std::vector<uint8_t> baseline(values * 2), actual(values * 2);
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    for (int mode : {0, 1, 2, 4, 8, 16}) {
        setenv("STRATA_GLM_IQ_COALESCED", std::to_string(mode).c_str(), 1);
        strata::kernels::iq_dequant_f16(type, weights.p, values, (uint16_t*)output.p, stream);
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(), output.p, actual.size(), cudaMemcpyDeviceToHost));
        if (!mode) baseline = actual;
        else if (actual != baseline) throw std::runtime_error("IQ coalesced benchmark bits differ");
        std::vector<float> times;
        for (int trial = 0; trial < 8; ++trial) {
            check(cudaEventRecord(begin, stream));
            for (int i = 0; i < 8; ++i)
                strata::kernels::iq_dequant_f16(type, weights.p, values, (uint16_t*)output.p, stream);
            check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
            float ms;
            check(cudaEventElapsedTime(&ms, begin, end));
            if (trial) times.push_back(ms / 8);
        }
        std::sort(times.begin(), times.end());
        std::cout << "IQ_COALESCED_BENCH type=" << type << " values=" << values
                  << " mode=" << mode << " median_ms=" << times[3] << '\n';
    }
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
}

int main(int argc, char **argv) {
    const bool dense = argc == 3 && std::strcmp(argv[2], "--dense") == 0;
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--benchmark") != 0 && !dense)) {
        std::cerr << "usage: glm_quant_parity <GLM shard.gguf> [--benchmark | --dense]\n";
        return 2;
    }
    try {
        strata::core::ModelArtifact artifact(argv[1]);
        if (dense) {
            // Every non-expert matrix the native GEMV serves: 64 real rows, 1 and 4 Q8_1 columns, against the
            // dequantized rows dotted with the same Q8_1 activations.
            Stream stream;
            std::mt19937 rng(7);
            std::normal_distribution<float> nd;
            int checked = 0, failed = 0;
            for (const auto &[name, t] : artifact.tensors()) {
                const int type = t.tensor->type;
                if (name.ends_with("_exps.weight") || t.tensor->shape.size() != 2 ||
                    !strata::kernels::native_mmvq_supported(type) || type == 0)
                    continue;
                const int cols = t.tensor->shape[0], rows = std::min<int>(64, t.tensor->shape[1]);
                if (cols % 32) continue;
                const size_t row_bytes = ggml_row_size((ggml_type)type, cols), bytes = row_bytes * rows;
                const auto *traits = ggml_get_type_traits((ggml_type)type);
                std::vector<float> ref((size_t)rows * cols);
                for (int r = 0; r < rows; ++r) traits->to_float(t.data() + r * row_bytes, ref.data() + (size_t)r * cols, cols);
                for (int nc : {1, 4}) {
                    std::vector<float> x((size_t)cols * nc);
                    for (auto &v : x) v = nd(rng);
                    Allocation w(bytes), dx(x.size() * 4), qx((size_t)cols / 32 * 36 * nc), y((size_t)rows * 4 * nc);
                    check(cudaMemcpy(w.p, t.data(), bytes, cudaMemcpyHostToDevice));
                    check(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
                    strata::kernels::quantize_q8_1_rows((float *)dx.p, nc, cols, qx.p, stream.p);
                    strata::kernels::native_mmvq(type, w.p, qx.p, (float *)y.p, cols, rows, nc, stream.p);
                    check(cudaStreamSynchronize(stream.p));
                    std::vector<uint8_t> packed((size_t)cols / 32 * 36 * nc);
                    check(cudaMemcpy(packed.data(), qx.p, packed.size(), cudaMemcpyDeviceToHost));
                    std::vector<float> out((size_t)rows * nc);
                    check(cudaMemcpy(out.data(), y.p, out.size() * 4, cudaMemcpyDeviceToHost));
                    double error = 0, denom = 0;
                    for (int c = 0; c < nc; ++c)
                        for (int r = 0; r < rows; ++r) {
                            double expected = 0;
                            for (int i = 0; i < cols; ++i) {
                                const auto *block = packed.data() + ((size_t)c * cols / 32 + i / 32) * 36;
                                uint16_t half;
                                std::memcpy(&half, block, 2);
                                expected += (double)ref[(size_t)r * cols + i] * ggml_fp16_to_fp32(half) * (int8_t)block[4 + i % 32];
                            }
                            error += std::abs(out[(size_t)c * rows + r] - expected);
                            denom += std::abs(expected);
                        }
                    const double relative = error / (denom + 1e-30);
                    ++checked;
                    if (!(relative < 1e-4)) {
                        ++failed;
                        std::cout << "DENSE_MISMATCH " << name << " type=" << type << " cols=" << cols << " ncols=" << nc
                                  << " rel=" << relative << '\n';
                    }
                }
            }
            std::cout << "DENSE checked=" << checked << " failed=" << failed << '\n';
            return failed ? 1 : 0;
        }
        // Q2 and Q4 artifacts use different native expert formats. Validate
        // every IQ/K format actually present rather than requiring Q4 types
        // that do not occur in the user's Q2 shards.
        std::set<int> pending;
        for (const auto &[name, t] : artifact.tensors())
            if (name.ends_with("_exps.weight") && strata::kernels::iq_supported(t.tensor->type) &&
                strata::kernels::native_mmvq_supported(t.tensor->type))
                pending.insert(t.tensor->type);
        if (pending.empty())
            throw std::runtime_error("fixture contains no supported native IQ/K expert formats");
        Stream stream;
        std::set<int> iq;
        for (const auto &[name, t] : artifact.tensors())
            if (strata::kernels::iq_supported(t.tensor->type)) iq.insert(t.tensor->type);
        for (const auto &[name, t] : artifact.tensors()) {
            const int type = t.tensor->type;
            if (!iq.count(type)) continue;
            int64_t available = 1;
            for (auto n : t.tensor->shape) available *= n;
            if (available < 20 * 256) continue;
            check_iq_coalesced(type, t.data(), available, stream.p, argc == 3);
            iq.erase(type);
        }
        if (!iq.empty()) throw std::runtime_error("no sufficiently large fixture for IQ dequant format");
        std::set<int> generic;
        for (const auto &[name, t] : artifact.tensors())
            if (strata::kernels::dequant_bf16_supported(t.tensor->type)) generic.insert(t.tensor->type);
        for (const auto &[name, t] : artifact.tensors()) {
            const int type = t.tensor->type;
            if (!generic.count(type) || t.tensor->shape.size() < 2 || t.tensor->shape[1] < 260) continue;
            check_coalesced(type, t.data(), t.tensor->shape[0], t.tensor->shape[1], stream.p, argc == 3);
            generic.erase(type);
        }
        if (!generic.empty()) throw std::runtime_error("no sufficiently large fixture for generic dequant format");
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
            // The existing pinned IQ2_XS dot performs integer /2 and /4,
            // and IQ2_S performs integer /4. Preserve those truncations;
            // the dequantized float oracle has this additional small error.
            const double tolerance = (type == 17 || type == 22) ? 1e-4 : 1e-5;
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
