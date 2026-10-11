// Standalone CUDA/HIP pinned transfers, streaming copy and library FP32 GEMM.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#define API(x) hip##x
#define BLAS(x) hipblas##x
#define BLAS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define BLAS_N HIPBLAS_OP_N
#define BACKEND "hip"
using DeviceProperties = hipDeviceProp_t;
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define API(x) cuda##x
#define BLAS(x) cublas##x
#define BLAS_SUCCESS CUBLAS_STATUS_SUCCESS
#define BLAS_N CUBLAS_OP_N
#define BACKEND "cuda"
using DeviceProperties = cudaDeviceProp;
#endif
void check(API(Error_t) status) {
    if (status != API(Success)) throw std::runtime_error(API(GetErrorString)(status));
}
void blas_check(BLAS(Status_t) status) {
    if (status != BLAS_SUCCESS) throw std::runtime_error("BLAS failed: " + std::to_string(int(status)));
}
__global__ void copy_kernel(const uint4* src, uint4* dst, size_t n) {
    for (size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x; i < n;
         i += size_t(gridDim.x)*blockDim.x) dst[i] = src[i];
}
struct Buffer {
    void* p = nullptr; bool host;
    Buffer(size_t bytes, bool pinned = false) : host(pinned) {
        if (host) check(API(HostAlloc)(&p, bytes, 0)); else check(API(Malloc)(&p, bytes));
    }
    ~Buffer() { if (p) { if (host) (void)API(FreeHost)(p); else (void)API(Free)(p); } }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
};
struct Events {
    API(Event_t) begin{}, end{};
    Events() { check(API(EventCreate)(&begin)); check(API(EventCreate)(&end)); }
    ~Events() { (void)API(EventDestroy)(begin); (void)API(EventDestroy)(end); }
};
template<class F> void measure(const char* name, double work, const char* unit, int repeats, F op) {
    Events e; std::vector<double> rates, ms;
    for (int r = 0; r <= repeats; ++r) {
        check(API(EventRecord)(e.begin));
        for (int i = 0; i < 4; ++i) op();
        check(API(EventRecord)(e.end)); check(API(EventSynchronize)(e.end));
        float elapsed; check(API(EventElapsedTime)(&elapsed, e.begin, e.end));
        if (!(elapsed > 0)) throw std::runtime_error("Invalid elapsed time");
        if (r) { ms.push_back(elapsed/4.); rates.push_back(work*4/(elapsed/1000.)/1e9); }
    }
    std::sort(rates.begin(), rates.end()); std::sort(ms.begin(), ms.end());
    std::cout << "{\"kernel\":\"" << name << "\",\"unit\":\"" << unit
              << "\",\"median\":" << rates[rates.size()/2] << ",\"best\":" << rates.back()
              << ",\"worst\":" << rates.front() << ",\"median_ms\":" << ms[ms.size()/2]
              << ",\"repeats\":" << repeats << ",\"operations_per_repeat\":4}" << std::endl;
}
int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    try {
        if (argc != 5) throw std::runtime_error("usage: gpu_check device buffer-MiB gemm-N repeats");
        int device=std::stoi(argv[1]), mib=std::stoi(argv[2]), n=std::stoi(argv[3]), reps=std::stoi(argv[4]);
        if (device<0 || mib<16 || mib>1024 || n<256 || n>4096 || n%256 || reps<3 || reps>31)
            throw std::runtime_error("Argument outside bounded range");
        check(API(SetDevice)(device)); DeviceProperties prop{}; check(API(GetDeviceProperties)(&prop,device));
        size_t free=0,total=0; check(API(MemGetInfo)(&free,&total));
        const size_t bytes=size_t(mib)<<20, matrix=size_t(n)*n*sizeof(float);
        if (free < 2*bytes+3*matrix+256+(256ull<<20)) throw std::runtime_error("Insufficient free VRAM with 256 MiB reserve");
        std::cout << "{\"backend\":\"" << BACKEND << "\",\"device\":" << device
                  << ",\"name\":" << std::quoted(prop.name) << ",\"buffer_bytes\":" << bytes
                  << ",\"gemm_n\":" << n << ",\"free_vram_bytes\":" << free << "}" << std::endl;
        Buffer host(bytes,true), back(bytes,true), a(bytes+128), b(bytes+128);
        auto* h=static_cast<unsigned char*>(host.p);
        for (size_t i=0;i<bytes;++i) h[i]=static_cast<unsigned char>((i*17+(i>>12))&255);
        // Guard both ends of the custom kernel's output allocation.
        check(API(Memset)(b.p,0xa5,bytes+128));
        void* output=static_cast<char*>(b.p)+64;
        measure("pcie_pinned_h2d",double(bytes),"GB/s",reps,[&]{check(API(MemcpyAsync)(a.p,host.p,bytes,API(MemcpyHostToDevice)));});
        check(API(Memcpy)(back.p,a.p,bytes,API(MemcpyDeviceToHost)));
        if (std::memcmp(host.p,back.p,bytes)) throw std::runtime_error("H2D byte mismatch");
        measure("pcie_pinned_d2h",double(bytes),"GB/s",reps,[&]{check(API(MemcpyAsync)(back.p,a.p,bytes,API(MemcpyDeviceToHost)));});
        if (std::memcmp(host.p,back.p,bytes)) throw std::runtime_error("D2H byte mismatch");
        measure("gpu_stream_copy",double(bytes)*2,"GB/s",reps,[&]{
            copy_kernel<<<4096,256>>>(static_cast<uint4*>(a.p),static_cast<uint4*>(output),bytes/sizeof(uint4));
            check(API(GetLastError)());
        });
        check(API(Memcpy)(back.p,output,bytes,API(MemcpyDeviceToHost)));
        if (std::memcmp(host.p,back.p,bytes)) throw std::runtime_error("GPU copy byte mismatch");
        unsigned char guard[64];
        for (size_t offset : {size_t(0),bytes+64}) {
            check(API(Memcpy)(guard,static_cast<char*>(b.p)+offset,64,API(MemcpyDeviceToHost)));
            for (auto value:guard) if (value!=0xa5) throw std::runtime_error("GPU copy guard changed");
        }
        std::cout << "{\"transfer_and_copy_verified\":true}" << std::endl;
        Buffer x(matrix), w(matrix), y(matrix+128);
        std::vector<float> values(size_t(n)*n,1.f/64);
        check(API(Memcpy)(x.p,values.data(),matrix,API(MemcpyHostToDevice)));
        check(API(Memcpy)(w.p,values.data(),matrix,API(MemcpyHostToDevice)));
        check(API(Memset)(y.p,0xa5,matrix+128));
        BLAS(Handle_t) handle{}; blas_check(BLAS(Create)(&handle));
#ifndef __HIPCC__
        // Explicit full FP32; do not silently count TF32 as FP32 compute.
        blas_check(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
#endif
        const float alpha=1,beta=0;
        measure("gpu_sgemm_fp32",2.*n*n*n,"GFLOP/s",reps,[&]{
            blas_check(BLAS(Sgemm)(handle,BLAS_N,BLAS_N,n,n,n,&alpha,
                static_cast<float*>(x.p),n,static_cast<float*>(w.p),n,&beta,
                reinterpret_cast<float*>(static_cast<char*>(y.p)+64),n));
        });
        blas_check(BLAS(Destroy)(handle));
        check(API(Memcpy)(values.data(),static_cast<char*>(y.p)+64,matrix,API(MemcpyDeviceToHost)));
        for (float value:values) if (value!=float(n)/4096) throw std::runtime_error("SGEMM result mismatch");
        for (size_t offset:{size_t(0),matrix+64}) {
            check(API(Memcpy)(guard,static_cast<char*>(y.p)+offset,64,API(MemcpyDeviceToHost)));
            for (auto value:guard) if (value!=0xa5) throw std::runtime_error("SGEMM guard changed");
        }
        std::cout << "{\"gemm_verified\":true,\"passed\":true}" << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
}
