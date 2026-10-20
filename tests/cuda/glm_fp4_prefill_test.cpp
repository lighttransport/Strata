#include "strata/kernels/glm_fp4.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace k=strata::kernels;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void blas_check(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS failure: "+std::to_string(int(e)));}
struct Buffer {void* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n));}~Buffer(){cudaFree(p);}};
struct Blas {cublasHandle_t h{};Blas(){blas_check(cublasCreate(&h));}~Blas(){cublasDestroy(h);}};
struct Variant {int tile,parts;};
static constexpr Variant variants[]={{0,1},{32,1},{64,1},{128,1},{128,2},{128,4}};
static constexpr int variant_count=6;
static void run(int N,int K,int M,bool outliers,bool bench,Blas& blas) {
    struct Stream {cudaStream_t s{};cublasHandle_t b;explicit Stream(cublasHandle_t h):b(h){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));blas_check(cublasSetStream(b,s));}~Stream(){cublasSetStream(b,nullptr);cudaStreamDestroy(s);}} stream(blas.h);
    constexpr int copies=3;
    const size_t wb=size_t(N)*K/2,sb=size_t(N)*K/16;
    std::vector<uint8_t> codes(wb),scales(sb);
    for(size_t i=0;i<wb;++i)codes[i]=uint8_t(i*53+17);
    for(size_t i=0;i<sb;++i)scales[i]=uint8_t(24+i%80);
    Buffer raw(wb),rs(sb),packed(wb*copies),ps(sb*copies),half(size_t(N)*K*2),x(size_t(M)*K*2),y(size_t(M)*N*4),partials(size_t(M)*N*4*4);
    check(cudaMemcpyAsync(raw.p,codes.data(),wb,cudaMemcpyHostToDevice,stream.s));
    check(cudaMemcpyAsync(rs.p,scales.data(),sb,cudaMemcpyHostToDevice,stream.s));
    const float global=.00061f;
    for(int c=0;c<copies;++c)k::glm_fp4_prepack(static_cast<uint8_t*>(raw.p),static_cast<uint8_t*>(rs.p),
        static_cast<uint8_t*>(packed.p)+wb*c,static_cast<uint8_t*>(ps.p)+sb*c,N,K,stream.s);
    std::vector<uint16_t> input(size_t(M)*K);
    for(size_t i=0;i<input.size();++i) {
        float v=std::sin(float(i)*.37f)*2;
        if(outliers&&i%901==0)v=(i&1)?-8192.f:8192.f;
        input[i]=strata::cpu::exl3::half_round(v);
    }
    check(cudaMemcpyAsync(x.p,input.data(),input.size()*2,cudaMemcpyHostToDevice,stream.s));
    if(K>=128) {
        bool rejected=false;
        try {
            k::glm_fp4_prefill_split_mma(static_cast<uint8_t*>(packed.p),static_cast<uint8_t*>(ps.p),global,
                static_cast<uint16_t*>(x.p),static_cast<float*>(y.p),N,K,M,2,nullptr,stream.s);
        }catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("split prefill accepted missing scratch");
    }
    auto execute=[&](Variant v,int copy) {
        const auto* w=static_cast<uint8_t*>(packed.p)+wb*copy;
        const auto* s=static_cast<uint8_t*>(ps.p)+sb*copy;
        if(v.tile==128)k::glm_fp4_prefill_split_mma(w,s,global,static_cast<uint16_t*>(x.p),static_cast<float*>(y.p),
            N,K,M,v.parts,static_cast<float*>(partials.p),stream.s);
        else if(v.tile)k::glm_fp4_prefill_mma(w,s,global,static_cast<uint16_t*>(x.p),static_cast<float*>(y.p),N,K,M,v.tile,stream.s);
        else {
            k::glm_fp4_dequant_f16(k::GlmPackedFormat::Nvfp4,w,s,global,static_cast<uint16_t*>(half.p),N,K,stream.s,true);
            const float one=1,zero=0;
            blas_check(cublasGemmEx(blas.h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&one,half.p,CUDA_R_16F,K,
                x.p,CUDA_R_16F,K,&zero,y.p,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        check(cudaGetLastError());
    };
    execute(variants[0],0);check(cudaStreamSynchronize(stream.s));
    std::vector<float> expected(size_t(M)*N),got(expected.size());
    check(cudaMemcpy(expected.data(),y.p,expected.size()*4,cudaMemcpyDeviceToHost));
    // Independent scalar dot products also cover the small boundary fixtures.
    if(N<=96&&K<=256) {
        std::vector<uint16_t> decoded(size_t(N)*K);
        check(cudaMemcpy(decoded.data(),half.p,decoded.size()*2,cudaMemcpyDeviceToHost));
        for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
            double sum=0,absolute=0;
            for(int j=0;j<K;++j) {
                const double term=double(strata::cpu::exl3::half_value(decoded[size_t(n)*K+j]))*
                    strata::cpu::exl3::half_value(input[size_t(m)*K+j]);sum+=term;absolute+=std::abs(term);
            }
            if(std::abs(expected[size_t(m)*N+n]-sum)>2e-6*absolute+1e-6)
                throw std::runtime_error("prefill baseline differs from scalar reference");
        }
    }
    for(int variant=1;variant<variant_count;++variant) {
        const auto v=variants[variant];
        execute(v,0);check(cudaStreamSynchronize(stream.s));check(cudaMemcpy(got.data(),y.p,got.size()*4,cudaMemcpyDeviceToHost));
        double error=0,norm=0,maximum=0;
        for(size_t i=0;i<got.size();++i) {
            if(!std::isfinite(got[i]))throw std::runtime_error("nonfinite prefill MMA result");
            const double diff=double(got[i])-expected[i];error+=diff*diff;norm+=double(expected[i])*expected[i];maximum=std::max(maximum,std::abs(diff));
        }
        const double relative=std::sqrt(error/std::max(norm,1e-30));
        std::cout<<"{\"kind\":\"parity\",\"rows\":"<<N<<",\"columns\":"<<K<<",\"tokens\":"<<M
            <<",\"outliers\":"<<outliers<<",\"tile\":"<<v.tile<<",\"k_parts\":"<<v.parts
            <<",\"relative_l2\":"<<relative<<",\"max_abs\":"<<maximum<<"}"<<std::endl;
        if(relative>5e-5||maximum>5e-4*(1+std::sqrt(norm/got.size()))) {
            // Large cuBLAS shapes may select a different reduction. Preserve
            // independent scalar probes before rejecting the comparison.
            std::vector<uint16_t> decoded(size_t(N)*K);
            check(cudaMemcpy(decoded.data(),half.p,decoded.size()*2,cudaMemcpyDeviceToHost));
            double baseline_error=0,candidate_error=0;
            for(int a=0;a<8;++a)for(int b=0;b<8;++b) {
                const int m=a*(M-1)/7,n=b*(N-1)/7;double sum=0,absolute=0;
                for(int j=0;j<K;++j) {
                    const double term=double(strata::cpu::exl3::half_value(decoded[size_t(n)*K+j]))*
                        strata::cpu::exl3::half_value(input[size_t(m)*K+j]);sum+=term;absolute+=std::abs(term);
                }
                baseline_error=std::max(baseline_error,std::abs(expected[size_t(m)*N+n]-sum)/(absolute+1e-30));
                candidate_error=std::max(candidate_error,std::abs(got[size_t(m)*N+n]-sum)/(absolute+1e-30));
            }
            std::cout<<"{\"kind\":\"scalar_probes\",\"count\":64,\"baseline_max_error_over_sumabs\":"
                <<baseline_error<<",\"candidate_max_error_over_sumabs\":"<<candidate_error<<"}"<<std::endl;
            throw std::runtime_error("prefill MMA differs from native rounded-operand reference");
        }
    }
    if(!bench)return;
    cudaGraphExec_t graphs[variant_count]{};
    for(int v=0;v<variant_count;++v) {
        for(int i=0;i<6;++i)execute(variants[v],i%copies);
        check(cudaDeviceSynchronize());
        cudaGraph_t graph{};check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeGlobal));
        for(int c=0;c<copies;++c)execute(variants[v],c);
        check(cudaStreamEndCapture(stream.s,&graph));check(cudaGraphInstantiate(&graphs[v],graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));
    }
    cudaEvent_t start{},stop{};check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    constexpr int repeats=64;
    for(int sample=0;sample<variant_count;++sample)for(int order=0;order<variant_count;++order) {
        const int v=(order+sample)%variant_count;
        for(int i=0;i<8;++i)check(cudaGraphLaunch(graphs[v],stream.s));
        check(cudaEventRecord(start,stream.s));
        for(int i=0;i<repeats;++i)check(cudaGraphLaunch(graphs[v],stream.s));
        check(cudaEventRecord(stop,stream.s));check(cudaEventSynchronize(stop));float ms=0;check(cudaEventElapsedTime(&ms,start,stop));
        std::cout<<"{\"kind\":\"timing\",\"rows\":"<<N<<",\"columns\":"<<K<<",\"tokens\":"<<M
            <<",\"tile\":"<<variants[v].tile<<",\"k_parts\":"<<variants[v].parts
            <<",\"sample\":"<<sample<<",\"us\":"<<ms*1000/(repeats*copies)
            <<",\"rotating_code_bytes\":"<<(wb+sb)*copies<<"}\n";
    }
    for(auto graph:graphs)check(cudaGraphExecDestroy(graph));
    cudaEventDestroy(start);cudaEventDestroy(stop);
}
int main(int argc,char** argv){
    cudaDeviceProp props{};if(cudaGetDeviceProperties(&props,0)!=cudaSuccess||props.major<7)return 77;
    try {
        Blas blas;
        const bool bench=argc==2&&std::string(argv[1])=="--bench";
        if(argc==2&&std::string(argv[1])=="--diagnose")run(2048,4096,256,false,false,blas);
        else if(argc==2&&std::string(argv[1])=="--large")
            for(int N:{2048,4096})for(int M:{16,128,512})for(bool outliers:{false,true})
                run(N,N==2048?4096:2048,M,outliers,false,blas);
        else if(bench)for(int N:{2048,4096})for(int M:{16,32,64,128,256,512})run(N,N==2048?4096:2048,M,false,true,blas);
        else for(int N:{32,96})for(int K:{64,128,192,256,320})for(int M:{9,17,31,32,33,63,64,65,127,128,129})
            for(bool outliers:{false,true})run(N,K,M,outliers,false,blas);
        std::cout<<"prefill MMA PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
