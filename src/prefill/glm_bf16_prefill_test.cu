// BF16 expert GEMM: batched/tail bounds and FP32 output vs a CPU dot and library GEMM.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { ck(cudaMalloc(&p, bytes)); }
    ~Device() { (void)cudaFree(p); }
};
uint16_t encode(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 64u);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float decode(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

void conversions() {
    using namespace strata::kernels;
    std::vector<float> x{0.f, -0.f, 1.f, 1.00390625f, 1.01171875f, 1e10f, -1e10f,
                         std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    Device dx(x.size()*4), dy(x.size()*2);
    ck(cudaMemcpy(dx.p, x.data(), x.size()*4, cudaMemcpyHostToDevice));
    glm_bf16((float*)dx.p, (uint16_t*)dy.p, x.size(), nullptr);
    std::vector<uint16_t> y(x.size());
    ck(cudaMemcpy(y.data(), dy.p, y.size()*2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < y.size(); ++i) if (y[i] != encode(x[i])) throw std::runtime_error("BF16 cast mismatch");
    std::mt19937 rng(712);
    for (int type : {10, 11, 16, 17, 22, 42}) {
        const size_t block_bytes = strata::kernels::iq_row_bytes(type, 256);
        std::vector<uint8_t> blocks(5 * block_bytes);
        for (auto& v : blocks) v = uint8_t(rng());
        // Finite FP16 block scales, including every Q2_0 64-value sub-block.
        const size_t step = type == 42 ? 18 : block_bytes;
        for (size_t i = 0; i < blocks.size(); i += step) { blocks[i]=0x55; blocks[i+1]=0x35; }
        if (type == 10 || type == 11)
            for (size_t i=0;i<blocks.size();i+=block_bytes) {
                blocks[i+block_bytes-2]=0x55;blocks[i+block_bytes-1]=0x35;
                if(type==10){blocks[i+block_bytes-4]=0x55;blocks[i+block_bytes-3]=0x35;}
            }
        Device q(blocks.size()), f(1024*4), b(1024*2);
        ck(cudaMemcpy(q.p, blocks.data(), blocks.size(), cudaMemcpyHostToDevice));
        iq_dequant_f32(type, (char*)q.p + block_bytes, 1024, (float*)f.p, nullptr);
        dequant_expert_bf16(type, q.p, 1, 4, 256, (uint16_t*)b.p, nullptr);
        std::vector<float> reference(1024); std::vector<uint16_t> actual(1024);
        ck(cudaMemcpy(reference.data(), f.p, 4096, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(actual.data(), b.p, 2048, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 1024; ++i)
            if (actual[i] != encode(reference[i])) throw std::runtime_error("direct BF16 dequant mismatch");
    }
    // Mask selects experts 0 and 2; padding and nonzero row offset must remain correct.
    std::vector<int> bounds{0,3,3,5}, source{2,0,1,1,0};
    Device db(bounds.size()*4), ds(source.size()*4), gx(3*32*4), gy(2*3*32*2);
    std::vector<float> input(3*32); for (size_t i=0;i<input.size();++i) input[i]=(float(i)-47)*1.01f;
    ck(cudaMemcpy(db.p,bounds.data(),bounds.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(ds.p,source.data(),source.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(gx.p,input.data(),input.size()*4,cudaMemcpyHostToDevice));
    glm_gather_expert_bf16((float*)gx.p,(uint16_t*)gy.p,(int*)db.p,(int*)ds.p,0,2,1,3,32,nullptr,5);
    std::vector<uint16_t> gathered(2*3*32);
    ck(cudaMemcpy(gathered.data(),gy.p,gathered.size()*2,cudaMemcpyDeviceToHost));
    for(int g=0;g<2;++g)for(int r=0;r<3;++r)for(int c=0;c<32;++c){
        const int expert=g?2:0, row=r+1;
        const uint16_t expected=row<bounds[expert+1]-bounds[expert]?encode(input[source[bounds[expert]+row]*32+c]):0;
        if(gathered[(g*3+r)*32+c]!=expected)throw std::runtime_error("BF16 masked gather mismatch");
    }
    std::puts("BF16 casts, masked gather and direct quantized dequant: PASS");
}

void gemm_case(int T,int N,int K,int B,bool wmma,bool bench) {
    std::mt19937 rng(91);std::uniform_real_distribution<float> dist(-.125f,.125f);
    std::vector<uint16_t> x(size_t(B)*T*K),w(size_t(B)*N*K);
    for(auto& v:x)v=encode(dist(rng));for(auto& v:w)v=encode(dist(rng));
    const size_t count=size_t(B)*T*N;
    std::vector<float> y(count+32,123456.f),ref(count);
    Device dx(x.size()*2),dw(w.size()*2),dy(y.size()*4),dr(count*4);
    ck(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dw.p,w.data(),w.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dy.p,y.data(),y.size()*4,cudaMemcpyHostToDevice));
    strata::prefill::Gemm gemm;std::string err;if(!gemm.init(nullptr,256,err))throw std::runtime_error(err);
    gemm.bf16_batched((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)dr.p,T,N,K,B);
    gemm.bf16_batched((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)dy.p+16,T,N,K,B,wmma);
    ck(cudaMemcpy(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(ref.data(),dr.p,count*4,cudaMemcpyDeviceToHost));
    float max_error=0;
    for(size_t i=0;i<count;++i){
        const float e=std::fabs(y[16+i]-ref[i]);max_error=std::max(max_error,e);
        if(!std::isfinite(y[16+i])||e>2e-4f*(1+std::fabs(ref[i])))throw std::runtime_error("WMMA/library mismatch");
    }
    for(int i=0;i<16;++i)if(y[i]!=123456.f||y[16+count+i]!=123456.f)throw std::runtime_error("GEMM output overrun");
    for(int batch=0;batch<B;++batch)for(int t:{0,T/2,T-1})for(int n:{0,N/2,N-1}){
        double dot=0;for(int k=0;k<K;++k)dot+=double(decode(x[(size_t(batch)*T+t)*K+k]))*decode(w[(size_t(batch)*N+n)*K+k]);
        if(std::fabs(y[16+(size_t(batch)*T+t)*N+n]-dot)>2e-4*(1+std::fabs(dot)))throw std::runtime_error("GEMM CPU oracle mismatch");
    }
    std::printf("BF16 %s T=%d N=%d K=%d B=%d max_error=%g PASS\n",wmma?"wmma":"blas",T,N,K,B,max_error);
    if(bench){
        for(bool custom:{false,true}){
            if(custom&&!wmma)continue;
            for(int i=0;i<3;++i)gemm.bf16_batched((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)dy.p+16,T,N,K,B,custom);
            cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a));
            for(int i=0;i<20;++i)gemm.bf16_batched((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)dy.p+16,T,N,K,B,custom);
            ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
            std::printf("BENCH backend=%s T=%d N=%d K=%d B=%d ms=%g TFLOPS=%g\n",custom?"wmma":"blas",T,N,K,B,ms/20,2.*T*N*K*B/(ms/20)/1e9);
            ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
        }
    }
}
int main(int argc,char**argv){try{
    bool wmma=false,bench=false;for(int i=1;i<argc;++i){if(std::string(argv[i])=="--wmma")wmma=true;else if(std::string(argv[i])=="--bench")bench=true;else return 2;}
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
#if defined(STRATA_USE_HIP)
    if(wmma){cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));
        if(std::strncmp(p.gcnArchName,"gfx1200",7)&&std::strncmp(p.gcnArchName,"gfx1201",7))return 77;}
#endif
    conversions();
    for(int T:{1,17,65,129})gemm_case(T,137,64,3,wmma,false);
    gemm_case(13,139,37,2,wmma,false);
    gemm_case(23,257,6144,2,wmma,false);
    if(bench){gemm_case(128,4096,4096,2,wmma,true);gemm_case(256,4096,2048,2,wmma,true);}
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
