// EXL3 K2 GPU GEMV that decodes each tile once (exl3_packed_gemv_tiled) against the per-row packed_mv and a double
// CPU reference on random tiles, 1/2/4 tokens; timing at GLM-5.3-Flash expert shapes. CUDA and HIP.
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"gpu: %s\n",cudaGetErrorString(e));std::exit(2);}}
int main() {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count){std::printf("no GPU\n");return 77;}
    std::mt19937 rng(7);int fails=0;
    for(auto [in,out]:{std::pair{4096,2048},std::pair{2048,4096},std::pair{256,128}}) {
        std::vector<uint8_t> packed(size_t(in)*out/4);for(auto& v:packed)v=uint8_t(rng());
        std::vector<float> x(4*size_t(in));for(auto& v:x)v=float(int(rng()%2001)-1000)*.001f;
        // Reference: decoded tiles (packed rows [ot][it]) in double.
        std::vector<double> ref(4*size_t(out),0.0);float w[256];
        for(int ot=0;ot<out/16;++ot)for(int it=0;it<in/16;++it) {
            strata::cpu::exl3::tile(packed.data()+(size_t(ot)*(in/16)+it)*64,w);
            for(int k=0;k<4;++k)for(int r=0;r<16;++r)for(int c=0;c<16;++c)ref[size_t(k)*out+ot*16+c]+=double(x[size_t(k)*in+it*16+r])*w[r*16+c];
        }
        uint8_t* dp;float *dx,*dy;
        check(cudaMalloc(&dp,packed.size()));check(cudaMalloc(&dx,x.size()*4));check(cudaMalloc(&dy,4*size_t(out)*4));
        check(cudaMemcpy(dp,packed.data(),packed.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        auto rel=[&](const std::vector<float>& y,int nt){double n=0,d=0;for(int k=0;k<nt;++k)for(int i=0;i<out;++i){double e=y[size_t(k)*out+i]-ref[size_t(k)*out+i];n+=e*e;d+=ref[size_t(k)*out+i]*ref[size_t(k)*out+i];}return std::sqrt(n/d);};
        auto time=[&](auto&& f){f();check(cudaDeviceSynchronize());const int reps=200;auto t0=std::chrono::steady_clock::now();for(int i=0;i<reps;++i)f();check(cudaDeviceSynchronize());return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/reps;};
        std::vector<float> y(4*size_t(out));
        strata::kernels::exl3_packed_gemv(dp,dx,dy,in,out,nullptr);check(cudaMemcpy(y.data(),dy,out*4,cudaMemcpyDeviceToHost));
        const double r0=rel(y,1);
        const double t0=time([&]{strata::kernels::exl3_packed_gemv(dp,dx,dy,in,out,nullptr);});
        std::printf("%dx%d packed_mv: relative error %.2e, %.1f us, %.1f GB/s\n",in,out,r0,t0*1e6,packed.size()/t0/1e9);
        for(int nt:{1,2,4}) {
            strata::kernels::exl3_packed_gemv_tiled(dp,dx,dy,in,out,nt,nullptr);check(cudaMemcpy(y.data(),dy,size_t(nt)*out*4,cudaMemcpyDeviceToHost));
            const double r=rel(y,nt);const bool ok=r<1e-5;fails+=!ok;
            const double t=time([&]{strata::kernels::exl3_packed_gemv_tiled(dp,dx,dy,in,out,nt,nullptr);});
            std::printf("%s %dx%d tiled, %d token(s): relative error %.2e, %.1f us, %.1f GB/s of tiles\n",ok?"ok":"FAIL",in,out,nt,r,t*1e6,packed.size()/t/1e9);
        }
        cudaFree(dp);cudaFree(dx);cudaFree(dy);
    }
    return fails?1:0;
}
