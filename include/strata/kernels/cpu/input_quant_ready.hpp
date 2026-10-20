#pragma once
#include <cstdint>
#include <stdexcept>

namespace strata::kernels::cpu {

inline bool defer_input_quantization(bool requested, bool remote_tp, bool resident_check) {
    return requested && !remote_tp && !resident_check;
}

// Per-layer readiness only; caller owns buffers and serializes preparation.
// Mark a token ready after successful quantization, so exceptions cannot
// publish an unfinished buffer. Never retain this object across layers.
class InputQuantReady {
    uint64_t ready_ = 0;
    int tokens_;
public:
    explicit InputQuantReady(int tokens) : tokens_(tokens) {
        if (tokens < 1 || tokens > 64) throw std::invalid_argument("input quantization token count");
    }
    template<class Quantize>
    void ensure(int token, Quantize&& quantize) {
        if (token < 0 || token >= tokens_) throw std::out_of_range("input quantization token");
        const uint64_t bit = uint64_t{1} << token;
        if (!(ready_ & bit)) {
            quantize(token);
            ready_ |= bit;
        }
    }
};
}
