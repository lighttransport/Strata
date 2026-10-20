#include "strata/kernels/fp4_local.hpp"
#include <algorithm>
#include <cstdio>
#include <limits>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetErrorString hipGetErrorString
#else
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#endif
namespace strata::kernels {
#define CHECK(x) do {auto status=(x);if(status!=0)throw std::runtime_error(cudaGetErrorString(status));}while(0)
template<class T> struct Mem {
    T* p=nullptr;size_t count;
    explicit Mem(size_t n):count(n){CHECK(cudaMalloc((void**)&p,n*sizeof(T)));}
    ~Mem(){if(p)(void)cudaFree(p);}
    Mem(const Mem&)=delete;
    void put(const T* src){CHECK(cudaMemcpy(p,src,count*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get()const {std::vector<T> v(count);CHECK(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
__host__ __device__ static uint16_t bf16(float x) {uint32_t b;__builtin_memcpy(&b,&x,4);b+=0x7fff+((b>>16)&1);return uint16_t(b>>16);}
__host__ __device__ static float unbf16(uint16_t b) {uint32_t v=uint32_t(b)<<16;float x;__builtin_memcpy(&x,&v,4);return x;}
__device__ static float e4(uint8_t q) {unsigned e=q>>3,m=q&7;return e?ldexpf(1.f+m*.125f,int(e)-7):m/512.f;}
__device__ static float weight(const uint8_t* w,const uint8_t* s,size_t i) {
    unsigned q=(w[i/2]>>((i%2)*4))&15,m=q&7,e=m>>1;
    float x=e?float((2+(m&1))<<(e-1))*.5f:float(m)*.5f;
    return (q&8?-x:x)*e4(s[i/16]);
}
__global__ static void gemm(const float* a,const uint8_t* w,const uint8_t* s,float* out,int m,int n,int k,float alpha) {
    int lane=threadIdx.x,row0=blockIdx.y*16,col0=blockIdx.x*16;
#if defined(__HIPCC__)
    using B8=unsigned short __attribute__((ext_vector_type(8)));
    using F8=float __attribute__((ext_vector_type(8)));
    int idx=lane%16,h=lane/16;F8 acc{};
    for(int kk=0;kk<k;kk+=16) {
        B8 av{},bv{};
        for(int i=0;i<8;++i) {
            if(row0+idx<m)av[i]=bf16(a[size_t(row0+idx)*k+kk+h*8+i]);
            if(col0+idx<n)bv[i]=bf16(weight(w,s,size_t(col0+idx)*k+kk+h*8+i));
        }
        acc=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av,bv,acc);
    }
    for(int i=0;i<8;++i)if(row0+h*8+i<m&&col0+idx<n)out[size_t(row0+h*8+i)*n+col0+idx]=acc[i]*alpha;
#else
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 sa[256],sb[256];
    __shared__ __align__(32) float sc[256];
    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> wf;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;wmma::fill_fragment(cf,0.f);
    for(int kk=0;kk<k;kk+=16) {
        for(int i=lane;i<256;i+=32) {
            int r=i/16,c=i%16;
            sa[i]=__float2bfloat16(row0+r<m?a[size_t(row0+r)*k+kk+c]:0);
            sb[i]=__float2bfloat16(col0+r<n?weight(w,s,size_t(col0+r)*k+kk+c):0);
        }
        __syncwarp();wmma::load_matrix_sync(af,sa,16);wmma::load_matrix_sync(wf,sb,16);
        wmma::mma_sync(cf,af,wf,cf);__syncwarp();
    }
    wmma::store_matrix_sync(sc,cf,16,wmma::mem_row_major);__syncwarp();
    for(int i=lane;i<256;i+=32)if(row0+i/16<m&&col0+i%16<n)out[size_t(row0+i/16)*n+col0+i%16]=sc[i]*alpha;
#endif
}
#if defined(__HIPCC__)
// Design adapted from JohnTDI-cpu/rdna4-wmma-guide (CC BY 4.0).
// Independent NVFP4 implementation: BF16, K16 E4M3 scales, K32 loop, 1x2
// register tiles for bounded M<=16. No MXFP4 E8M0 scale substitution.
__global__ static void gemm_tiled(const float* a,const uint8_t* w,const uint8_t* s,
                                float* out,int m,int n,int k,float alpha) {
    using B8=unsigned short __attribute__((ext_vector_type(8)));
    using F8=float __attribute__((ext_vector_type(8)));
    __shared__ float lut[16];
    int lane=threadIdx.x,idx=lane%16,h=lane/16,col0=blockIdx.x*32;
    if(lane<16) {
        int mag=lane&7,e=mag>>1;
        float v=e?float((2+(mag&1))<<(e-1))*.5f:float(mag)*.5f;
        lut[lane]=(lane&8)?-v:v;
    }
    __syncthreads();
    F8 acc[2]{};
    for(int kb=0;kb<k;kb+=32) {
        #pragma unroll
        for(int step=0;step<2;++step) {
            int kk=kb+step*16;if(kk>=k)break;
            B8 av{};
            if(idx<m) {
                // hipMalloc base and K16 alignment permit two 128-bit loads.
                const float4* ap=reinterpret_cast<const float4*>(a+size_t(idx)*k+kk+h*8);
                float4 lo=ap[0],hi=ap[1];
                av[0]=bf16(lo.x);av[1]=bf16(lo.y);av[2]=bf16(lo.z);av[3]=bf16(lo.w);
                av[4]=bf16(hi.x);av[5]=bf16(hi.y);av[6]=bf16(hi.z);av[7]=bf16(hi.w);
            }
            #pragma unroll
            for(int tile=0;tile<2;++tile) {
                int col=col0+tile*16+idx;B8 bv{};
                if(col<n) {
                    size_t wi=size_t(col)*k+kk+h*8;
                    uint32_t packed=*reinterpret_cast<const uint32_t*>(w+wi/2);
                    float scale=e4(s[(size_t(col)*k+kk)/16]);
                    #pragma unroll
                    for(int j=0;j<8;++j)bv[j]=bf16(lut[(packed>>(j*4))&15]*scale);
                }
                acc[tile]=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av,bv,acc[tile]);
            }
        }
    }
    #pragma unroll
    for(int tile=0;tile<2;++tile)
        for(int j=0;j<8;++j) {
            int row=h*8+j,col=col0+tile*16+idx;
            if(row<m&&col<n)out[size_t(row)*n+col]=acc[tile][j]*alpha;
        }
}
#endif
__global__ static void reference(const float* a,const uint8_t* w,const uint8_t* s,double* out,double* absout,int m,int n,int k,float alpha) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(m)*n)return;
    double total=0,ab=0;
    for(int j=0;j<k;++j) {double p=double(unbf16(bf16(a[(i/n)*k+j])))*double(weight(w,s,(i%n)*k+j));total+=p;ab+=fabs(p);}
    out[i]=total*alpha;absout[i]=ab*fabs(double(alpha));
}
struct Fp4LocalGpu::Impl {
    artifact::GlmFp4Matrix host;Mem<uint8_t> w,s;
    explicit Impl(const artifact::GlmFp4Matrix& m):host(m),w(m.codes.size()),s(m.scales.size()){w.put(m.codes.data());s.put(m.scales.data());}
};
Fp4LocalGpu::Fp4LocalGpu(const artifact::GlmFp4Matrix& m) {
    if(!m.rows||!m.cols||m.cols%16||m.rows>16384||m.cols>16384||m.codes.size()!=m.rows*m.cols/2||m.scales.size()!=m.rows*m.cols/16||!std::isfinite(m.global)||m.global<=0)
        throw std::runtime_error("FP4 GPU: unsupported matrix");
    for(auto s:m.scales)if(s>=127)throw std::runtime_error("FP4 GPU: invalid scale");
    size_t free,total;CHECK(cudaMemGetInfo(&free,&total));
    if(free<m.bytes()+256*1024*1024+2ull*1024*1024*1024)throw std::runtime_error("FP4 GPU: insufficient free memory including 2 GiB reserve");
    impl_=std::make_unique<Impl>(m);
}
Fp4LocalGpu::~Fp4LocalGpu()=default;
std::vector<float> Fp4LocalGpu::run(const std::vector<float>& input,int tokens,bool verify,bool rdna4_tiled,float* kernel_ms) {
    const auto& h=impl_->host;
    if(tokens<1||tokens>16||input.size()!=size_t(tokens)*h.cols)throw std::runtime_error("FP4 GPU: token/input bounds");
    for(float x:input)if(!std::isfinite(x))throw std::runtime_error("FP4 GPU: nonfinite input");
    Mem<float> a(input.size()),y(size_t(tokens)*h.rows);a.put(input.data());
#if !defined(__HIPCC__)
    if(rdna4_tiled)throw std::runtime_error("RDNA4 tiled kernel requires HIP");
#endif
    cudaEvent_t start{},stop{};
    if(kernel_ms){CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));CHECK(cudaEventRecord(start));}
#if defined(__HIPCC__)
    if(rdna4_tiled)gemm_tiled<<<(h.rows+31)/32,32>>>(a.p,impl_->w.p,impl_->s.p,y.p,tokens,h.rows,h.cols,h.global);
    else
#endif
    gemm<<<dim3((h.rows+15)/16,(tokens+15)/16),32>>>(a.p,impl_->w.p,impl_->s.p,y.p,tokens,h.rows,h.cols,h.global);
    CHECK(cudaGetLastError());
    if(kernel_ms){CHECK(cudaEventRecord(stop));CHECK(cudaEventSynchronize(stop));CHECK(cudaEventElapsedTime(kernel_ms,start,stop));CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));}
    CHECK(cudaDeviceSynchronize());auto got=y.get();
    if(verify) {
        Mem<double> ref(got.size()),ab(got.size());
        reference<<<(got.size()+127)/128,128>>>(a.p,impl_->w.p,impl_->s.p,ref.p,ab.p,tokens,h.rows,h.cols,h.global);
        CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());auto expected=ref.get(),absref=ab.get();
        double u=std::numeric_limits<float>::epsilon()/2,gamma=((h.cols+1)*u)/(1-(h.cols+1)*u),maxerr=0,num=0,den=0;
        for(size_t i=0;i<got.size();++i) {
            double e=std::abs(double(got[i])-expected[i]);
            if(!std::isfinite(got[i])||e>gamma*absref[i])throw std::runtime_error("FP4 GPU: FP64 bound failed");
            maxerr=std::max(maxerr,e);num+=e*e;den+=expected[i]*expected[i];
        }
        for(size_t sample=0;sample<32;++sample) {
            size_t i=sample*(got.size()-1)/31;double sum=0;
            for(size_t j=0;j<h.cols;++j) {
                size_t wi=(i%h.rows)*h.cols+j;float local=strata_glm_e2m1((h.codes[wi/2]>>((wi%2)*4))&15)*strata_glm_e4m3(h.scales[wi/16]);
                sum+=double(unbf16(bf16(input[(i/h.rows)*h.cols+j])))*local;
            }
            if(sum*h.global!=expected[i])throw std::runtime_error("FP4 GPU: independent CPU reference mismatch");
        }
        std::printf("PROJECTION PASS m=%d n=%zu k=%zu max_abs=%.9g rel_l2=%.9g\n",tokens,h.rows,h.cols,maxerr,den?std::sqrt(num/den):0);
    }
    return got;
}
void fp4_local_device_info() {cudaDeviceProp p{};CHECK(cudaGetDeviceProperties(&p,0));std::printf("GPU %s\n",p.name);}
}
