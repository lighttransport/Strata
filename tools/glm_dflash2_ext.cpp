// CUDA-only PyTorch evaluation helper: half operands, FP32 accumulation/output.
// V100 tensor cores support this combination; BF16 instructions are not needed.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cublas_v2.h>

static at::Tensor linear_f32(const at::Tensor& x,const at::Tensor& w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.device()==w.device(),"DFlash2 GEMM needs one CUDA device");
    TORCH_CHECK(x.scalar_type()==at::kHalf && w.scalar_type()==at::kHalf,"DFlash2 GEMM needs FP16 operands");
    TORCH_CHECK(x.dim()==2 && w.dim()==2 && x.is_contiguous() && w.is_contiguous(),"DFlash2 GEMM needs contiguous matrices");
    TORCH_CHECK(x.size(1)==w.size(1) && x.size(0)>0 && x.size(0)<=2048 &&
        x.size(1)>0 && x.size(1)<=20480 && w.size(0)>0 && w.size(0)<=20480,"DFlash2 GEMM bounds");
    const c10::cuda::CUDAGuard guard(x.device());
    auto y=at::empty({x.size(0),w.size(0)},x.options().dtype(at::kFloat));
    const float alpha=1,beta=0;
    const int n=w.size(0),m=x.size(0),k=x.size(1);
    const auto status=cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(),CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,
        &alpha,w.data_ptr(),CUDA_R_16F,k,x.data_ptr(),CUDA_R_16F,k,&beta,y.data_ptr(),CUDA_R_32F,n,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"DFlash2 cublasGemmEx failed: ",int(status));
    return y;
}
TORCH_LIBRARY(strata_dflash2,m) {
    m.def("linear_f32(Tensor x, Tensor w) -> Tensor",&linear_f32);
}
