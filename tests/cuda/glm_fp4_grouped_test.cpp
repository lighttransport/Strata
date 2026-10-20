#include "strata/kernels/glm_fp4.hpp"
#include "strata/kernels/glm.hpp"
#include "strata/kernels/glm_prefill.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace k=strata::kernels;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
    void* p=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n){ck(cudaMalloc(&p,n));}
    ~Buffer(){cudaFree(p);}
};
struct Plane {
    Buffer b;size_t valid;
    explicit Plane(size_t bytes):b(bytes+128),valid(bytes){}
    template<class T>T* ptr(){return reinterpret_cast<T*>(static_cast<char*>(b.p)+64);}
    void reset(cudaStream_t s){ck(cudaMemsetAsync(b.p,0x5a,b.bytes,s));}
    std::vector<unsigned char> read(){
        std::vector<unsigned char> bytes(b.bytes);ck(cudaMemcpy(bytes.data(),b.p,b.bytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<64;++i)if(bytes[i]!=0x5a||bytes[64+valid+i]!=0x5a)throw std::runtime_error("guard overwritten");
        return bytes;
    }
};
int main(int argc,char** argv){
    cudaStream_t s=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    try {
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
        cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));if(prop.major<7)return 77;
        const bool bench=argc>1&&std::string(argv[1])=="--bench";
        ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
        constexpr size_t elements=size_t(2048)*4096,codes_bytes=elements/2,scales_bytes=elements/16;
        Buffer codes(8*3*codes_bytes),scales(8*3*scales_bytes),descriptors(64*sizeof(k::GlmFp4ExpertGroup)),
            ids(64*sizeof(int)),input(8*4096*4),coefficients(64*4),seed(8*4096*4);
        std::vector<uint8_t> bytes(8*3*codes_bytes);
        for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(i*53+17+i/codes_bytes*7);
        ck(cudaMemcpyAsync(codes.p,bytes.data(),bytes.size(),cudaMemcpyHostToDevice,s));
        bytes.resize(8*3*scales_bytes);
        for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(24+(i*7+i/scales_bytes*3)%64);
        ck(cudaMemcpyAsync(scales.p,bytes.data(),bytes.size(),cudaMemcpyHostToDevice,s));
        ck(cudaStreamSynchronize(s));
        int cases=0,graphs=0;
        std::vector<std::pair<int,int>> shapes;
        for(int g:{1,2,4,8})for(int w:{1,2,3,4,8})shapes.push_back({g,w});
        shapes.push_back({16,4});shapes.push_back({64,1});
        for(const auto [groups,width]:shapes){
            // Mixed route counts, with distinct tokens per expert and shared
            // tokens across experts. Unused storage stays guarded by sentinels.
            std::vector<k::GlmFp4ExpertGroup> desc(groups);std::vector<int> routes;
            for(int g=0;g<groups;++g){
                auto& d=desc[g];d.count=(width==8||g%2==0)?width:std::max(1,width-1);d.offset=int(routes.size());
                for(int p=0;p<3;++p){
                    d.weights[p]=static_cast<uint8_t*>(codes.p)+(3*(g%8)+p)*codes_bytes;
                    d.scales[p]=static_cast<uint8_t*>(scales.p)+(3*(g%8)+p)*scales_bytes;
                    d.global[p]=.00019f*(1+((g+p)%3));
                }
                for(int t=0;t<d.count;++t)routes.push_back(((t+g)%8)*8+g%8);
            }
            const int rows=int(routes.size());
            Plane hx(64*4096*2),gate(64*2048*4),up(64*2048*4),hidden(64*2048*2),result(64*4096*4),sum(8*4096*4);
            std::vector<Plane*> planes={&hx,&gate,&up,&hidden,&result,&sum};
            ck(cudaMemcpyAsync(descriptors.p,desc.data(),desc.size()*sizeof(desc[0]),cudaMemcpyHostToDevice,s));
            ck(cudaMemcpyAsync(ids.p,routes.data(),rows*4,cudaMemcpyHostToDevice,s));
            auto reset=[&]{for(auto p:planes)p->reset(s);};
            auto run=[&](bool grouped,bool fused=false){
                ck(cudaMemcpyAsync(sum.ptr<float>(),seed.p,8*4096*4,cudaMemcpyDeviceToDevice,s));
                if(grouped){
                    k::glm_fp4_route_gather(static_cast<float*>(input.p),static_cast<int*>(ids.p),hx.ptr<uint16_t>(),rows,4096,8,s);
                    k::glm_fp4_expert_group_products(static_cast<k::GlmFp4ExpertGroup*>(descriptors.p),groups,false,hx.ptr<uint16_t>(),gate.ptr<float>(),up.ptr<float>(),s);
                    k::glm_swiglu(gate.ptr<float>(),up.ptr<float>(),gate.ptr<float>(),rows*2048,7.f,s);
                    k::glm_f16(gate.ptr<float>(),hidden.ptr<uint16_t>(),rows*2048,s);
                    k::glm_fp4_expert_group_products(static_cast<k::GlmFp4ExpertGroup*>(descriptors.p),groups,true,hidden.ptr<uint16_t>(),result.ptr<float>(),nullptr,s);
                }
                if(grouped&&fused)k::glm_fp4_expert_group_scatter(static_cast<k::GlmFp4ExpertGroup*>(descriptors.p),groups,
                    static_cast<int*>(ids.p),static_cast<float*>(coefficients.p),result.ptr<float>(),sum.ptr<float>(),8,s);
                for(int g=0;g<groups;++g){
                    const auto& d=desc[g];const auto o=d.offset;auto dr=static_cast<int*>(ids.p)+o;
                    if(!grouped){
                        k::glm_fp4_route_gather(static_cast<float*>(input.p),dr,hx.ptr<uint16_t>()+o*4096,d.count,4096,8,s);
                        for(int p=0;p<2;++p)k::glm_fp4_gemm(k::GlmPackedFormat::Nvfp4,d.weights[p],d.scales[p],d.global[p],
                            hx.ptr<uint16_t>()+o*4096,(p?up.ptr<float>():gate.ptr<float>())+o*2048,2048,4096,d.count,true,s);
                        k::glm_swiglu(gate.ptr<float>()+o*2048,up.ptr<float>()+o*2048,gate.ptr<float>()+o*2048,d.count*2048,7.f,s);
                        k::glm_f16(gate.ptr<float>()+o*2048,hidden.ptr<uint16_t>()+o*2048,d.count*2048,s);
                        k::glm_fp4_gemm(k::GlmPackedFormat::Nvfp4,d.weights[2],d.scales[2],d.global[2],hidden.ptr<uint16_t>()+o*2048,
                            result.ptr<float>()+o*4096,4096,2048,d.count,true,s);
                    }
                    if(!grouped||!fused)k::glm_fp4_route_scatter(result.ptr<float>()+o*4096,dr,static_cast<float*>(coefficients.p),sum.ptr<float>(),d.count,4096,8,s);
                }
                ck(cudaGetLastError());
            };
            auto read=[&]{ck(cudaStreamSynchronize(s));std::vector<std::vector<unsigned char>> all;for(auto p:planes)all.push_back(p->read());for(auto p:{&gate,&up,&result,&sum}){auto bytes=p->read();for(size_t i=64;i<64+p->valid;i+=4){float f;std::memcpy(&f,bytes.data()+i,4);if(!std::isfinite(f))throw std::runtime_error("nonfinite intermediate");}}return all;};
            std::vector<float> x(8*4096),weights(64),seeds(8*4096);
            auto change=[&](int pass){
                for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.13f+pass*.41f)*float(int(i%7)-3);
                for(size_t i=0;i<weights.size();++i)weights[i]=i%5?float(int(i%7)-3)*.0625f:0.f;
                for(size_t i=0;i<seeds.size();++i)seeds[i]=width==1?0.f:std::sin(float(i)*.11f+pass*.31f);
                ck(cudaMemcpyAsync(seed.p,seeds.data(),seeds.size()*4,cudaMemcpyHostToDevice,s));
                ck(cudaMemcpyAsync(input.p,x.data(),x.size()*4,cudaMemcpyHostToDevice,s));
                ck(cudaMemcpyAsync(coefficients.p,weights.data(),weights.size()*4,cudaMemcpyHostToDevice,s));
            };
            for(int pass=0;pass<2;++pass){
                change(pass);reset();run(false);const auto expected=read();
                for(bool fused:{false,true}){reset();run(true,fused);if(read()!=expected)throw std::runtime_error("grouped ordered scatter changed bits");++cases;}
            }
            for(bool fused:{false,true}){
                ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));run(true,fused);ck(cudaStreamEndCapture(s,&graph));
                ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                for(int pass=2;pass<4;++pass){
                    change(pass);reset();run(false);const auto expected=read();reset();ck(cudaGraphLaunch(exec,s));
                    if(read()!=expected)throw std::runtime_error("grouped ordered scatter graph changed bits");++graphs;
                }
                ck(cudaGraphExecDestroy(exec));exec=nullptr;ck(cudaGraphDestroy(graph));graph=nullptr;
            }
            if(bench&&groups>1&&groups<=8&&width<=4){
                for(int i=0;i<16;++i){run(true,false);run(true,true);}ck(cudaStreamSynchronize(s));
                for(int sample=0;sample<5;++sample)for(int at=0;at<2;++at){
                    const bool active=bool(at^(sample&1));cudaEvent_t first,last;ck(cudaEventCreate(&first));ck(cudaEventCreate(&last));
                    ck(cudaEventRecord(first,s));for(int i=0;i<16;++i)run(true,active);ck(cudaEventRecord(last,s));ck(cudaEventSynchronize(last));
                    float ms;ck(cudaEventElapsedTime(&ms,first,last));ck(cudaEventDestroy(first));ck(cudaEventDestroy(last));
                    std::cout<<"GROUPED_BENCH groups="<<groups<<" width="<<width<<" active="<<active<<" sample="<<sample<<" us="<<ms*1000/16<<'\n';
                }
            }
        }
        for(int n:{0,65}){bool rejected=false;try{k::glm_fp4_expert_group_products(nullptr,n,false,nullptr,nullptr,nullptr,s);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)throw std::runtime_error("invalid group count accepted");}
        ck(cudaStreamDestroy(s));s=nullptr;
        std::cout<<"RESIDENT_GROUPED passed=1 bitwise_cases="<<cases<<" graph_cases="<<graphs<<" guards=unchanged sm="<<prop.major<<prop.minor<<'\n';return 0;
    }catch(const std::exception& e){
        if(s)cudaStreamSynchronize(s);if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);if(s)cudaStreamDestroy(s);
        std::cerr<<e.what()<<'\n';return 1;
    }
}
