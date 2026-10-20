#pragma once
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace strata::experiment {
// Experimental per-token protection for whole hidden vectors whose Q8_0
// half scales would be subnormal or zero. Does not protect input quantization
// or every tiny block within a vector with a larger peak.
class SmallHiddenScale final : public kernels::cpu::ExpertHiddenTransform {
    std::array<int, kernels::cpu::MAXT> shifts_{};
    static void validate(int token, const float* values, int count) {
        if (token < 0 || token >= kernels::cpu::MAXT || count < 0 || (!values && count))
            throw std::invalid_argument("invalid hidden rescale extent");
    }
public:
    void reset() { shifts_.fill(0); }
    int shift(int token) const {
        validate(token, nullptr, 0);
        return shifts_[token];
    }
    void apply(int, int, int token, float* values, int count) override {
        validate(token, values, count);
        float peak = 0;
        for (int i = 0; i < count; ++i) {
            if (!std::isfinite(values[i]))
                throw std::invalid_argument("nonfinite hidden rescale input");
            peak = std::max(peak, std::abs(values[i]));
        }
        shifts_[token] = 0;
        if (peak == 0 || peak >= 0.0078125f) return;
        int exponent = 0;
        std::frexp(peak, &exponent);
        const int power = -6 - exponent;
        shifts_[token] = power;
        if (power <= 126) {
            const float factor = std::ldexp(1.f, power);
            for (int i = 0; i < count; ++i) values[i] *= factor;
        } else {
            // Constructing a float multiplier would overflow for these shifts.
            for (int i = 0; i < count; ++i) values[i] = std::ldexp(values[i], power);
        }
    }
    void restore(int token, float* values, int count) const {
        validate(token, values, count);
        const int power = shifts_[token];
        if (!power) return;
        if (power <= 126) {
            const float factor = std::ldexp(1.f, -power);
            for (int i = 0; i < count; ++i) values[i] *= factor;
        } else {
            for (int i = 0; i < count; ++i) values[i] = std::ldexp(values[i], -power);
        }
    }
};

// Dense local expert IDs keep scale shifts independent within one pool batch.
template <size_t Experts>
class MultiExpertHiddenScale final : public kernels::cpu::ExpertHiddenTransform {
    static_assert(Experts > 0);
    std::array<SmallHiddenScale, Experts> scales_{};
    static size_t index(int expert) {
        if (expert < 0 || size_t(expert) >= Experts)
            throw std::invalid_argument("invalid hidden-rescale expert ID");
        return size_t(expert);
    }
public:
    void apply(int layer, int expert, int token, float* values, int count) override {
        scales_[index(expert)].apply(layer, expert, token, values, count);
    }
    void restore(int expert, int token, float* values, int count) const {
        scales_[index(expert)].restore(token, values, count);
    }
};
}
