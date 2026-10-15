// Short batches retain FP32 products even when the optional FP16 paths are enabled.
#include "strata/prefill/gemm.hpp"
#ifdef STRATA_PREFILL_MMQ
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/glm_prefill.hpp"
#endif
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Device {
    void *p = nullptr;
    explicit Device(size_t bytes) { check(cudaMalloc(&p, bytes)); }
    ~Device() { (void)cudaFree(p); }
};
#ifdef STRATA_PREFILL_MMQ
void run_swiglu() {
    for (int ff:{1,257,2048}) for (bool interleaved:{false,true}) for (float limit:{0.f,7.f}) {
        const int rows=17,count=rows*ff;
        std::vector<float> gu(2*count);
        for (int i=0;i<2*count;++i) gu[i]=float(i%71-35)*.37f;
        gu[0]=-0.f;
        if (gu.size()>4) {gu[1]=1e-35f;gu[2]=-100.f;gu[3]=100.f;}
        std::vector<uint16_t> result(count+32,0x5a5a),reference(count);
        Device input(gu.size()*4),hidden(count*4),expected(count*2),actual(result.size()*2);
        check(cudaMemcpy(input.p,gu.data(),gu.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(actual.p,result.data(),result.size()*2,cudaMemcpyHostToDevice));
        strata::prefill::mmq::swiglu(static_cast<float *>(input.p),static_cast<float *>(hidden.p),rows,ff,interleaved,nullptr,limit);
        strata::kernels::glm_f16(static_cast<float *>(hidden.p),static_cast<uint16_t *>(expected.p),count,nullptr);
        strata::prefill::mmq::swiglu_f16(static_cast<float *>(input.p),static_cast<uint16_t *>(actual.p)+16,
                                        rows,ff,interleaved,nullptr,limit);
        check(cudaMemcpy(reference.data(),expected.p,count*2,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(result.data(),actual.p,result.size()*2,cudaMemcpyDeviceToHost));
        if (std::memcmp(reference.data(),result.data()+16,count*2))
            throw std::runtime_error("fused SwiGLU/FP16 differs from separate kernels");
        for (int i=0;i<16;++i)
            if (result[i]!=0x5a5a || result[16+count+i]!=0x5a5a)
                throw std::runtime_error("fused SwiGLU/FP16 overwrote guard region");
    }
    std::printf("fused SwiGLU/FP16 bitwise parity PASS\n");
}
#endif
void run(strata::prefill::Gemm &gemm, int mode, int n, bool lossy) {
    const int t = 17, k = 64, batches = mode ? 2 : 1, stride = mode ? n : n + 7;
    std::vector<__half> x(batches * t * k), w(batches * n * k);
    for (size_t i = 0; i < x.size(); ++i) x[i] = __float2half(float(int(i % 29) - 14) * .0137f);
    for (size_t i = 0; i < w.size(); ++i) w[i] = __float2half(float(int(i % 31) - 15) * .0113f);
    const size_t count = batches * t * stride;
    std::vector<float> result(count + 32, -777.25f);
    Device dx(x.size() * 2), dw(w.size() * 2), dy(result.size() * 4);
    check(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw.p, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dy.p, result.data(), result.size() * 4, cudaMemcpyHostToDevice));
    auto *out = static_cast<float *>(dy.p) + 16;
    auto *a = static_cast<uint16_t *>(dx.p), *b = static_cast<uint16_t *>(dw.p);
    strata::prefill::Gemm::lossy_allowed = lossy;
    if (mode == 0) gemm.f16(a, b, out, t, n, k, stride);
    if (mode == 1) gemm.f16_batched(a, b, out, t, n, k, batches);
    if (mode == 2) gemm.f16_batched_nn(b, a, out, n, t, k, batches);
    check(cudaMemcpy(result.data(), dy.p, result.size() * 4, cudaMemcpyDeviceToHost));
    int non_half = 0;
    for (int batch = 0; batch < batches; ++batch)
        for (int row = 0; row < t; ++row)
            for (int col = 0; col < stride; ++col) {
                const float actual = result[16 + (batch * t + row) * stride + col];
                if (col >= n) {
                    if (actual != -777.25f) throw std::runtime_error("FP16 widening overwrote row padding");
                    continue;
                }
                double reference = 0;
                for (int j = 0; j < k; ++j) {
                    const int wi = mode == 2 ? (batch * k + j) * n + col : (batch * n + col) * k + j;
                    reference += double(__half2float(x[(batch * t + row) * k + j])) * __half2float(w[wi]);
                }
                const double tolerance = lossy ? .003 : .000002;
                if (!std::isfinite(actual) || std::abs(actual - reference) > tolerance * (1 + std::abs(reference)))
                    throw std::runtime_error("prefill product differs from precision contract");
                non_half += actual != __half2float(__float2half(actual));
            }
    for (int i = 0; i < 16; ++i)
        if (result[i] != -777.25f || result[16 + count + i] != -777.25f)
            throw std::runtime_error("prefill product overwrote guard region");
    if (!lossy && !non_half) throw std::runtime_error("short batch unexpectedly rounded every output to FP16");
    std::printf("FP16 precision mode=%d n=%d lossy=%d PASS\n", mode, n, int(lossy));
}
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
#ifdef STRATA_PREFILL_MMQ
        run_swiglu();
#endif
        strata::prefill::Gemm gemm;
        std::string error;
        if (!gemm.init(nullptr, 256, error)) throw std::runtime_error(error);
        for (int mode : {0, 1, 2})
            for (int n : {37, 257, 513})
                for (bool lossy : {false, true, false}) run(gemm, mode, n, lossy);
        strata::prefill::Gemm::lossy_allowed = true;
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
