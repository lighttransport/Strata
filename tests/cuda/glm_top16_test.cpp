// Independent CPU candidate oracle, masks, nonfinite/tie semantics, guards,
// nondefault streams and changed-input graph replays. No model required.
#include "strata/kernels/glm.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace k=strata::kernels;
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
    void* p=nullptr;
    explicit Buffer(size_t bytes){check(cudaMalloc(&p,bytes));}
    ~Buffer(){cudaFree(p);}
    Buffer(const Buffer&)=delete;
};
int main(int argc,char** argv){
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    try {
        const bool smoke=argc==2 && std::strcmp(argv[1],"--smoke")==0;
        if(argc>1 && !smoke)throw std::invalid_argument("expected optional --smoke");
        cudaStream_t stream;check(cudaStreamCreate(&stream));
        std::mt19937 random(3541);int cases=0,replays=0;
        for(int n:{1,15,16,17,255,256,257,4097,65537,154880})for(int rows:{1,3,7}) {
            if(smoke && (n<17 || n>257 || rows!=3))continue;
            const size_t input_bytes=size_t(n)*rows*4,scratch_bytes=k::glm_top16_scratch_bytes(n,rows);
            const size_t result_bytes=size_t(rows)*16*sizeof(k::GlmLogitCandidate);
            Buffer input(input_bytes+64),mask(n),scratch(scratch_bytes+64),result(result_bytes+64);
            auto* x=reinterpret_cast<float*>(static_cast<char*>(input.p)+32);
            auto* work=static_cast<char*>(scratch.p)+32;
            auto* output=reinterpret_cast<k::GlmLogitCandidate*>(static_cast<char*>(result.p)+32);
            std::vector<float> values(size_t(n)*rows);std::vector<uint8_t> allowed(n);
            auto upload=[&](int pattern){
                for(auto& v:values)v=float(int(random()%129)-64);
                for(int i=0;i<n;++i)allowed[i]=pattern==6?0:pattern==5?i<9:i%7!=0;
                if(pattern==1)std::fill(values.begin(),values.end(),-42.f);
                if(pattern==2)for(size_t i=0;i<values.size();++i)values[i]=i%2?-0.f:0.f;
                if(pattern==3)for(size_t i=0;i<values.size();++i)values[i]=
                    float(int(i%9)-4)*std::numeric_limits<float>::denorm_min();
                if(pattern==4)for(size_t i=0;i<values.size();++i) {
                    uint32_t bits=random();
                    if((bits&0x7f800000u)==0x7f800000u)bits^=0x00800000u;
                    values[i]=std::bit_cast<float>(bits);
                }
                if(pattern==7)for(size_t i=0;i<values.size();++i)values[i]=
                    i%3==0?std::numeric_limits<float>::quiet_NaN():
                    i%3==1?std::numeric_limits<float>::infinity():-std::numeric_limits<float>::infinity();
                check(cudaMemset(input.p,0x5a,input_bytes+64));
                check(cudaMemcpy(x,values.data(),input_bytes,cudaMemcpyHostToDevice));
                check(cudaMemcpy(mask.p,allowed.data(),n,cudaMemcpyHostToDevice));
                check(cudaMemset(scratch.p,0x5a,scratch_bytes+64));
                check(cudaMemset(result.p,0x5a,result_bytes+64));
            };
            auto compare=[&](bool masked){
                std::vector<k::GlmLogitCandidate> got(size_t(rows)*16);
                check(cudaMemcpy(got.data(),output,result_bytes,cudaMemcpyDeviceToHost));
                for(int row=0;row<rows;++row) {
                    std::vector<int> indices;
                    for(int id=0;id<n;++id)if((!masked || allowed[id]) && std::isfinite(values[size_t(row)*n+id]))indices.push_back(id);
                    std::stable_sort(indices.begin(),indices.end(),[&](int a,int b){return values[size_t(row)*n+a]>values[size_t(row)*n+b];});
                    for(int at=0;at<16;++at) {
                        const int id=at<int(indices.size())?indices[at]:-1;
                        const auto& actual=got[row*16+at];
                        if(actual.token!=id || (id>=0 && std::bit_cast<uint32_t>(actual.logit)!=std::bit_cast<uint32_t>(values[size_t(row)*n+id])) ||
                           (id<0 && actual.logit!=-std::numeric_limits<float>::infinity()))
                            throw std::runtime_error("candidate differs from independent CPU stable sort");
                    }
                }
                for(auto [buffer,bytes]:{std::pair{input.p,input_bytes},std::pair{scratch.p,scratch_bytes},std::pair{result.p,result_bytes}}) {
                    unsigned char first[32],last[32];check(cudaMemcpy(first,buffer,32,cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(last,static_cast<char*>(buffer)+bytes+32,32,cudaMemcpyDeviceToHost));
                    for(int i=0;i<32;++i)if(first[i]!=0x5a || last[i]!=0x5a)throw std::runtime_error("buffer guard overwritten");
                }
                std::vector<float> copy(values.size());check(cudaMemcpy(copy.data(),x,input_bytes,cudaMemcpyDeviceToHost));
                if(std::memcmp(copy.data(),values.data(),input_bytes))throw std::runtime_error("input modified");
            };
            for(int pattern=0;pattern<8;++pattern) {
                upload(pattern);const bool masked=pattern!=4;
                k::glm_top16(x,masked?static_cast<uint8_t*>(mask.p):nullptr,n,rows,work,output,stream);
                check(cudaStreamSynchronize(stream));compare(masked);++cases;
            }
            if(n==154880 && rows==7) {
                cudaGraph_t graph;cudaGraphExec_t executable;
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                k::glm_top16(x,static_cast<uint8_t*>(mask.p),n,rows,work,output,stream);
                check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
                for(int pattern:{0,2,5}) {
                    upload(pattern);check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));compare(true);++replays;
                }
                check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
                upload(0);cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
                for(int i=0;i<2;++i)k::glm_top16(x,static_cast<uint8_t*>(mask.p),n,rows,work,output,stream);
                check(cudaEventRecord(start,stream));
                for(int i=0;i<20;++i)k::glm_top16(x,static_cast<uint8_t*>(mask.p),n,rows,work,output,stream);
                check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float milliseconds;
                check(cudaEventElapsedTime(&milliseconds,start,end));
                std::cout<<"top16 GPU-only microbenchmark vocabulary="<<n<<" rows="<<rows<<" mean_ms="<<milliseconds/20<<'\n';
                check(cudaEventDestroy(start));check(cudaEventDestroy(end));
            }
        }
        for(auto [n,rows]:{std::pair{0,1},std::pair{-1,1},std::pair{16,0},std::pair{16,9}}) {
            bool rejected=false;try{k::glm_top16_scratch_bytes(n,rows);}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("invalid dimensions accepted");
        }
        bool rejected=false;try{k::glm_top16(nullptr,nullptr,16,1,nullptr,nullptr,stream);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("null buffers accepted");
        check(cudaStreamDestroy(stream));std::cout<<"PASS top16 cases="<<cases<<" graph_replays="<<replays<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
