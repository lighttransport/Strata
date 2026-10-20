#pragma once
#include "strata/artifact/glm_fp4.hpp"
#include <array>

namespace strata::kernels::cpu {
// Experimental host-resident projection for partial-model execution. Owns packed
// weights, rounds finite inputs to BF16, and applies the global scale last.
// Arithmetic is W4A16 with FP32 accumulation, not GPU WMMA bit emulation.
class Fp4LocalCpu {
    friend class Fp4LocalCpuExpert;
    size_t rows_, columns_;
    float global_;
    std::vector<uint8_t> codes_, scales_;
    int padded_columns_, threads_;
public:
    explicit Fp4LocalCpu(const artifact::GlmFp4Matrix& matrix,int threads=1);
    // kernel_ms excludes input rounding, allocation and optional FP64 verification.
    std::vector<float> run(const std::vector<float>& input,int tokens,
                          bool verify=true,float* kernel_ms=nullptr) const;
    // Two projections with matching input width and thread count. Reuses input
    // rounding and one parallel dispatch; scales and output row counts may differ.
    std::array<std::vector<float>,2> run_pair(const Fp4LocalCpu& other,
        const std::vector<float>& input,int tokens,bool verify=true) const;
};
// Complete experimental W4A16 expert with the partial runner's BF16/clamp-10
// SiLU semantics. Pairing is explicit; the default uses separate projections.
class Fp4LocalCpuExpert {
    Fp4LocalCpu gate_, up_, down_;
public:
    explicit Fp4LocalCpuExpert(const std::array<artifact::GlmFp4Matrix,3>& weights,int threads=1);
    std::vector<float> run(const std::vector<float>& input,int tokens,
                          bool paired=false,bool verify=true) const;
    // reuse_weights groups up to four tokens to share decoded weights.
    // Opt-in persistent team: gate/up and SiLU row jobs, then down projection.
    std::vector<float> run_fused(const std::vector<float>& input,int tokens,bool verify=true,bool reuse_weights=false) const;
};
}
