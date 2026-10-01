#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main() {
#if defined(_WIN32)
    _putenv_s("STRATA_NATIVE_NUMA_LOCAL", "1");
#else
    setenv("STRATA_NATIVE_NUMA_LOCAL", "1", 1);
#endif
    cpu::NativeFmt fmt;
    std::string error;
    if (!cpu::native_fmt(17, 18, 256, 256, fmt, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    fmt.swiglu_limit = 10;
    std::mt19937 rng(53);
    std::vector<std::vector<uint8_t>> weights(16, std::vector<uint8_t>(fmt.bytes));
    for (auto& blob : weights) {
        for (auto& byte : blob) byte = (uint8_t)rng();
        for (size_t offset : {size_t(0), fmt.up_off, fmt.down_off}) {
            const size_t stride = offset == fmt.down_off ? 98 : 74;
            for (int row = 0; row < 256; ++row) {
                const uint16_t scale = 0x2400; // finite FP16 scale for every synthetic block
                std::memcpy(blob.data() + offset + row * stride, &scale, sizeof scale);
            }
        }
    }
    std::vector<float> x(3 * 256);
    for (float& value : x) value = float(int(rng() % 201) - 100) / 100;
    std::vector<std::vector<uint8_t>> act(3, std::vector<uint8_t>(fmt.act_bytes));
    for (int t = 0; t < 3; ++t) cpu::native_quant_act(fmt, x.data() + t * 256, act[t].data());
    for (bool host_works : {false, true}) {
        cpu::ExpertPool pool(4, false, host_works);
        for (int nt : {1, 2, 3}) {
            for (int count : {1, 2, 7, 16}) {
                std::vector<float> expected(count * nt * 256), actual(expected.size());
                std::vector<cpu::ExpertJobMulti> jobs(count);
                for (int e = 0; e < count; ++e) {
                    std::vector<float> ff(nt * 256);
                    std::vector<std::vector<uint8_t>> hq(nt, std::vector<uint8_t>(fmt.h_bytes));
                    const void* inputs[3] = {};
                    const void* hidden[3] = {};
                    float* intermediate[3] = {};
                    float* outputs[3] = {};
                    jobs[e].blob = weights[e].data();
                    jobs[e].nt = nt;
                    for (int t = 0; t < nt; ++t) {
                        inputs[t] = act[t].data();
                        intermediate[t] = ff.data() + t * 256;
                        hidden[t] = hq[t].data();
                        outputs[t] = expected.data() + (e * nt + t) * 256;
                        jobs[e].nact[t] = inputs[t];
                        jobs[e].out[t] = actual.data() + (e * nt + t) * 256;
                    }
                    cpu::native_gu_rows(fmt, weights[e].data(), inputs, nt, intermediate, 0, 256);
                    for (int t = 0; t < nt; ++t) cpu::native_quant_h(fmt, intermediate[t], hq[t].data());
                    cpu::native_down_rows(fmt, weights[e].data(), hidden, nt, outputs, 0, 256);
                }
                for (float value : expected) if (!std::isfinite(value)) return 2;
                for (int repetition = 0; repetition < 16; ++repetition) {
                    std::fill(actual.begin(), actual.end(), std::numeric_limits<float>::quiet_NaN());
                    pool.run_split_multi_native(fmt, jobs.data(), count);
                    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float))) {
                        std::fprintf(stderr, "native pool mismatch host=%d nt=%d count=%d repetition=%d\n",
                                     host_works, nt, count, repetition);
                        return 3;
                    }
                }
            }
        }
    }
    std::puts("native_pool_test: 384 batches bitwise equal, finite outputs, no untouched rows");
}
