// Synthetic GLM gate/up rows: clamp, width independence, split mappings and
// opt-in dispatch. No GPU or model is required.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_even_signs.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/host/glm_env.h"
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

struct Interval { double low, high; };

static double silu(double gate) { return gate / (1.0 + std::exp(-gate)); }

// Bound the floating-point dot error before applying SwiGLU. A relative
// tolerance on the final output alone rejects valid reductions when large
// terms cancel in the gate. The reference uses ggml-dequantized weights,
// the actual Q8_K activations and double accumulation, independently of either
// SIMD dot implementation. 32 float epsilons cover the 16 block accumulations,
// their scalar products, dequantization rounding and horizontal reduction.
static Interval reference_interval(double gate, double up, double gate_norm, double up_norm, double limit) {
    const double epsilon = std::numeric_limits<float>::epsilon();
    const double eg = 32 * epsilon * (1 + gate_norm), eu = 32 * epsilon * (1 + up_norm);
    double gl = gate - eg, gh = gate + eg, ul = up - eu, uh = up + eu;
    if (limit > 0) {
        gl = std::min(gl, limit); gh = std::min(gh, limit);
        ul = std::clamp(ul, -limit, limit); uh = std::clamp(uh, -limit, limit);
    }
    double sl = std::min(silu(gl), silu(gh)), sh = std::max(silu(gl), silu(gh));
    constexpr double minimum_at = -1.2784645427610738;
    if (gl <= minimum_at && gh >= minimum_at) sl = std::min(sl, silu(minimum_at));
    const double values[] = {sl * ul, sl * uh, sh * ul, sh * uh};
    const auto range = std::minmax_element(std::begin(values), std::end(values));
    const double roundoff = 8 * epsilon * (1 + std::max(std::fabs(*range.first), std::fabs(*range.second)));
    return {*range.first - roundoff, *range.second + roundoff};
}

int main(int argc, char** argv) {
    for (int position = 0; position < 8; ++position) for (int index = 0; index < 128; ++index) {
        const uint64_t expected = uint64_t(ksigns_iq2xs[index]) << (8 * position);
        const uint64_t packed = uint64_t(index) << (8 * position);
        const uint32_t word = uint32_t(index) << (7 * (position % 4));
        if (cpu::detail::iq_even_signs_packed(packed) != expected ||
            cpu::detail::iq_even_signs_words(0xf0000000u | (position < 4 ? word : 0),
                                           0xa0000000u | (position >= 4 ? word : 0)) != expected) {
            std::puts("packed sign table mismatch"); return 1;
        }
    }
    std::mt19937 sign_random(1704);
    for (int trial = 0; trial < 1024; ++trial) {
        uint64_t indices = 0, expected = 0;
        uint32_t first = 0xf0000000u, second = 0xa0000000u;
        for (int position = 0; position < 8; ++position) {
            const auto index = sign_random() & 127u;
            indices |= uint64_t(index) << (8 * position);
            expected |= uint64_t(ksigns_iq2xs[index]) << (8 * position);
            (position < 4 ? first : second) |= uint32_t(index) << (7 * (position % 4));
        }
        if (cpu::detail::iq_even_signs_packed(indices) != expected ||
            cpu::detail::iq_even_signs_words(first, second) != expected) {
            std::puts("combined sign table mismatch"); return 1;
        }
    }
    if (argc == 2 && std::strcmp(argv[1], "--signs-only") == 0) {
        std::puts("iq_even_signs: all table comparisons passed"); return 0;
    }
    if (argc != 1) return 2;
    const auto positive = reference_interval(100, 100, 100, 100, 10);
    const auto negative = reference_interval(-100, 100, 100, 100, 10);
    const double unclamped = silu(100) * 100, symmetric_gate_clamp = silu(-10) * 10;
    if ((unclamped >= positive.low && unclamped <= positive.high) ||
        (symmetric_gate_clamp >= negative.low && symmetric_gate_clamp <= negative.high)) {
        std::puts("reference failed to reject missing or symmetric gate clamp"); return 1;
    }
    if (!cpu::cpu_avx512bw_ok()) { std::puts("SKIP: AVX-512 unavailable"); return 77; }
    constexpr int H = 4096, F = 2048, NT = 8, ROWS = 32, FIRST = 3, LAST = 29;
    constexpr float guard = 12345.f;
    int failures = 0;
    std::mt19937 random(1703);
    std::uniform_real_distribution<float> input(-64.f, 64.f);
    for (int type : {16, 17, 18, 21, 22, 23}) {
        cpu::NativeFmt fmt;
        std::string error;
        if (!cpu::native_fmt(type, GGML_TYPE_Q8_0, H, F, fmt, error)) {
            std::printf("format %d: %s\n", type, error.c_str()); return 1;
        }
        std::vector<uint8_t> blob(2 * fmt.up_off + 4096);
        for (auto& byte : blob) byte = uint8_t(random());
        const size_t block = ggml_type_size(ggml_type(type));
        for (size_t offset = 0; offset < 2 * fmt.up_off; offset += block) {
            const auto scale = ggml_fp32_to_fp16(0.007f);
            std::memcpy(blob.data() + offset, &scale, sizeof(scale));
        }
        std::vector<uint8_t> up(blob.begin() + fmt.up_off, blob.end());
        std::array<std::vector<uint8_t>, NT> quantized;
        const void* acts[NT];
        std::vector<float> x(H);
        for (int t = 0; t < NT; ++t) {
            for (float& value : x) value = input(random);
            quantized[t].resize(cpu::kNativeActBytes);
            cpu::native_quant_act(fmt, x.data(), quantized[t].data());
            acts[t] = quantized[t].data();
        }
        if (cpu::iq512_supported(type)) {
            std::array<std::array<double, ROWS>, NT> gates{}, ups{}, gate_norms{}, up_norms{};
            const auto* traits = ggml_get_type_traits(ggml_type(type));
            std::vector<float> gate_weights(H), up_weights(H);
            for (int row = FIRST; row < LAST; ++row) {
                traits->to_float(blob.data() + row * fmt.gu_row, gate_weights.data(), H);
                traits->to_float(up.data() + row * fmt.gu_row, up_weights.data(), H);
                for (int t = 0; t < NT; ++t) {
                    const auto* q = static_cast<const block_q8_K*>(acts[t]);
                    for (int i = 0; i < H; ++i) {
                        const double a = double(q[i / QK_K].d) * q[i / QK_K].qs[i % QK_K];
                        const double g = double(gate_weights[i]) * a, u = double(up_weights[i]) * a;
                        gates[t][row] += g; ups[t][row] += u;
                        gate_norms[t][row] += std::fabs(g); up_norms[t][row] += std::fabs(u);
                    }
                }
            }
            for (float limit : {0.f, 0.001f, 1.f, 10.f}) {
                std::array<std::vector<float>, NT> single;
                for (int t = 0; t < NT; ++t) {
                    single[t].assign(ROWS, guard);
                    float* row = single[t].data();
                    cpu::iq512_gu_rows_clamped(type, blob.data(), fmt.gu_row, fmt.up_off, H,
                                               acts + t, 1, &row, FIRST, LAST, limit);
                    std::vector<float> reference(ROWS, guard); float* ref = reference.data();
                    cpu::iq256_gu_rows_clamped(type, blob.data(), fmt.gu_row, fmt.up_off, H,
                                               acts + t, 1, &ref, FIRST, LAST, limit);
                    for (int r = 0; r < ROWS; ++r) {
                        if (r < FIRST || r >= LAST) {
                            if (single[t][r] != guard || reference[r] != guard) ++failures;
                            continue;
                        }
                        const auto bounds = reference_interval(gates[t][r], ups[t][r], gate_norms[t][r], up_norms[t][r], limit);
                        for (float got : {single[t][r], reference[r]}) {
                            if (!std::isfinite(got) || got < bounds.low || got > bounds.high) {
                                std::printf("dequantized reference mismatch type=%d token=%d row=%d limit=%g value=%g bounds=[%g,%g]\n",
                                            type, t, r, limit, got, bounds.low, bounds.high);
                                ++failures;
                            }
                        }
                    }
                }
                for (int nt = 1; nt <= NT; ++nt) for (bool separate : {false, true}) {
                    std::array<std::vector<float>, NT> actual, expected;
                    float* out[NT];
                    for (int t = 0; t < NT; ++t) {
                        actual[t].assign(ROWS, guard); expected[t].assign(ROWS, guard);
                        if (t < nt) for (int row = FIRST; row < LAST; ++row) expected[t][row] = single[t][row];
                        out[t] = actual[t].data();
                    }
                    cpu::iq512_gu_rows_clamped(type, blob.data(), fmt.gu_row, fmt.up_off, H,
                                               acts, nt, out, FIRST, LAST, limit,
                                               separate ? up.data() : nullptr);
                    for (int t = 0; t < NT; ++t) {
                        if (std::memcmp(actual[t].data(), expected[t].data(), ROWS * sizeof(float))) {
                            std::printf("clamp mismatch type=%d nt=%d token=%d limit=%g separate=%d\n",
                                        type, nt, t, limit, separate); ++failures;
                        }
                    }
                }
            }
        }
        fmt.swiglu_limit = 10.f;
        const bool vector512 = strata_env_truthy_one("STRATA_GLM_IQ512_CLAMP") &&
                               cpu::cpu_isa_cap() >= 3 && !strata_env_is_set("STRATA_NO_IQ512") && cpu::iq512_supported(type);
        for (int nt = 1; nt <= NT; ++nt) {
            std::array<std::vector<float>, NT> actual, expected;
            float *ap[NT], *ep[NT];
            for (int t = 0; t < NT; ++t) {
                actual[t].assign(ROWS, guard); expected[t].assign(ROWS, guard);
                ap[t] = actual[t].data(); ep[t] = expected[t].data();
            }
            cpu::native_gu_rows(fmt, blob.data(), acts, nt, ap, FIRST, LAST, up.data());
            if (vector512)
                cpu::iq512_gu_rows_clamped(type, blob.data(), fmt.gu_row, fmt.up_off, H, acts, nt,
                                           ep, FIRST, LAST, fmt.swiglu_limit, up.data());
            else
                cpu::iq256_gu_rows_clamped(type, blob.data(), fmt.gu_row, fmt.up_off, H, acts, nt,
                                           ep, FIRST, LAST, fmt.swiglu_limit, up.data());
            for (int t = 0; t < NT; ++t) if (std::memcmp(actual[t].data(), expected[t].data(), ROWS * sizeof(float))) {
                std::printf("dispatch mismatch type=%d nt=%d token=%d vector512=%d\n", type, nt, t, vector512);
                ++failures;
            }
        }
    }
    std::printf("iq_avx512_clamp_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
