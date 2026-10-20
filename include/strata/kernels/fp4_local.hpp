#pragma once
#include "strata/artifact/glm_fp4.hpp"
#include <memory>
namespace strata::kernels {
// Bounded experimental W4A16 primitive shared by HIP and CUDA node tests.
class Fp4LocalGpu {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit Fp4LocalGpu(const artifact::GlmFp4Matrix& matrix);
    ~Fp4LocalGpu();
    Fp4LocalGpu(const Fp4LocalGpu&)=delete;
    std::vector<float> run(const std::vector<float>& input,int tokens,bool verify=true,
                           bool rdna4_tiled=false,float* kernel_ms=nullptr);
};
void fp4_local_device_info();
}
