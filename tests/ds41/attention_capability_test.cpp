// Link against the CUDA ds41 object and cudart. The runtime calls below are mocked:
// this tests dispatch/rejection on machines without a GPU, not device arithmetic.
#include "strata/kernels/ds41.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

namespace {
int major = 8, binary = 80, ptx = 80, launches = 0, attributes = 0;
bool configure_fails = false;
}
extern "C" {
cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr, int) { *value = major; return cudaSuccess; }
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* value, const void*) {
    *value = {}; value->binaryVersion = binary; value->ptxVersion = ptx; return cudaSuccess;
}
cudaError_t cudaFuncSetAttribute(const void*, cudaFuncAttribute, int) {
    ++attributes; return configure_fails ? cudaErrorInvalidValue : cudaSuccess;
}
unsigned __cudaPushCallConfiguration(dim3, dim3, size_t, cudaStream_t) { return 0; }
cudaError_t __cudaPopCallConfiguration(dim3* grid, dim3* block, size_t* shared, void* stream) {
    *grid = dim3(1); *block = dim3(1); *shared = 0;
    *static_cast<cudaStream_t*>(stream) = nullptr;
    return cudaSuccess;
}
#if CUDART_VERSION >= 13000
cudaError_t __cudaGetKernel(cudaKernel_t* handle, const void*) { *handle = nullptr; return cudaSuccess; }
cudaError_t __cudaLaunchKernel(cudaKernel_t, dim3, dim3, void**, size_t, cudaStream_t) {
    ++launches; return cudaSuccess;
}
#endif
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t) {
    ++launches; return cudaSuccess;
}
}
static bool run() {
    try {
        strata::kernels::ds41_attention_batched(nullptr, 1, 64, 512, nullptr, 136, nullptr,
                                               nullptr, 512, nullptr, 512, nullptr, 1.0f,
                                               nullptr, nullptr);
        return true;
    } catch (const std::runtime_error&) { return false; }
}
int main(int argc, char**) {
    if (argc > 1) {
        unsetenv("STRATA_DS41_ATTN_MMA");
        assert(run() && launches == 1);  // default still dispatches the tiled path
        return 0;
    }
    setenv("STRATA_DS41_ATTN_MMA", "1", 1);
    major = 7;
    assert(!run() && launches == 0 && attributes == 0);
    major = 8; binary = ptx = 75;
    assert(!run() && launches == 0);  // an older image on a newer GPU is also unsafe
    binary = 80; ptx = 75;
    assert(!run() && launches == 0);
    ptx = 80; configure_fails = true;
    assert(!run() && launches == 0);
    configure_fails = false;
    assert(run() && launches == 1);
    major = 12; binary = ptx = 120;
    assert(run() && launches == 2);
}
