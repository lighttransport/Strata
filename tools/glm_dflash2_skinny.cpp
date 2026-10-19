// Opt-in PyTorch adapter for Strata's SM70 half-QPN skinny kernel.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include "strata/kernels/glm_fp4.hpp"
namespace k=strata::kernels;
static void matrix(const at::Tensor& w) {
    TORCH_CHECK(w.is_cuda()&&w.scalar_type()==at::kHalf&&w.dim()==2&&w.is_contiguous(),"skinny weight must be contiguous CUDA FP16");
    TORCH_CHECK(w.size(0)>0&&w.size(0)<=20480&&w.size(0)%32==0&&
                w.size(1)>0&&w.size(1)<=20480&&w.size(1)%64==0,"skinny weight shape unsupported");
}
static at::Tensor pack(const at::Tensor& w) {
    matrix(w);const c10::cuda::CUDAGuard guard(w.device());
    auto out=at::empty_like(w);
    k::glm_f16_prepack_qpn(static_cast<const uint16_t*>(w.data_ptr()),static_cast<uint16_t*>(out.data_ptr()),
        w.size(0),w.size(1),c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
static at::Tensor linear(const at::Tensor& x,const at::Tensor& w) {
    matrix(w);
    TORCH_CHECK(x.device()==w.device()&&x.scalar_type()==at::kHalf&&x.dim()==2&&x.is_contiguous()&&
                x.size(0)>=1&&x.size(0)<=8&&x.size(1)==w.size(1),"skinny input must be1..8 FP16 rows on the weight device");
    const c10::cuda::CUDAGuard guard(x.device());
    const int n=w.size(0),width=w.size(1),rows=x.size(0);
    auto out=at::empty({rows,n},x.options().dtype(at::kFloat));
    // Kernel geometry selects at most64 outer partitions. This buffer is local
    // to the call and allocated/freed on the caller's current Torch stream.
    at::Tensor partial;
    float* scratch=nullptr;
    if(n<=8192&&width>=4096) {
        partial=at::empty({64,8,n},out.options());scratch=partial.data_ptr<float>();
    }
    k::glm_fp4_gemm(k::GlmPackedFormat::F16Qpn,w.data_ptr(),nullptr,1,
        static_cast<const uint16_t*>(x.data_ptr()),out.data_ptr<float>(),n,width,rows,true,
        c10::cuda::getCurrentCUDAStream().stream(),scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
TORCH_LIBRARY(strata_dflash2_sm70,m) {
    m.def("pack(Tensor weight) -> Tensor",&pack);
    m.def("linear(Tensor input, Tensor packed_weight) -> Tensor",&linear);
}
