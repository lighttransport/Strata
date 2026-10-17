// Diagnostic CPU-versus-PCIe lower bound on the real full-Q23 expert formats.
// Rotate through 64 distinct model experts. This is not a decode speed result.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml-cpu.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <map>
#include <cmath>

namespace cpu=strata::kernels::cpu;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
static double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}
int main(int argc,char**argv)try{
    if(argc!=4)throw std::invalid_argument("calibrate model.gguf workers device");
    const int workers=std::stoi(argv[2]),device=std::stoi(argv[3]);
    ggml_cpu_init();check(cudaSetDevice(device));
    strata::GgufFile file(argv[1]);
    std::map<std::string,const strata::TensorInfo*> tensors;
    for(const auto& t:file.tensors())tensors[t.name]=&t;
    std::map<std::pair<int,int>,int> formats;
    for(int layer=0;layer<256;++layer){
        auto prefix="blk."+std::to_string(layer)+".ffn_";
        if(tensors.count(prefix+"gate_exps.weight") && tensors.count(prefix+"down_exps.weight"))
            formats.emplace(std::make_pair(tensors.at(prefix+"gate_exps.weight")->type,
                                          tensors.at(prefix+"down_exps.weight")->type),layer);
    }
    if(formats.empty())throw std::runtime_error("no expert formats");
    auto topology=cpu::detect_cpu_topology(true);
    auto affinity=cpu::pin_current_thread(topology.host_core);
    {
    cpu::ExpertPool pool(workers,true,true,cpu::PoolAffinity::All,0,&topology);
    for(auto [types,layer]:formats){
        auto prefix="blk."+std::to_string(layer)+".ffn_";
        const auto& gate=*tensors.at(prefix+"gate_exps.weight");
        const auto& up=*tensors.at(prefix+"up_exps.weight");
        const auto& down=*tensors.at(prefix+"down_exps.weight");
        if(gate.shape.size()!=3 || gate.shape!=up.shape || gate.type!=up.type || gate.shape[2]<64)
            throw std::runtime_error("unsupported calibration tensor geometry");
        const int H=gate.shape[0],F=gate.shape[1];
        cpu::NativeFmt f;std::string error;
        if(!cpu::native_fmt(types.first,types.second,H,F,f,error))throw std::runtime_error(error);
        f.swiglu_limit=10; f.canon=(types.first==10 && types.second==11);
        std::vector<uint8_t> blobs(64*f.bytes);
        for(int e=0;e<64;++e){
            const int id=(e*37+5)%gate.shape[2];auto* b=blobs.data()+e*f.bytes;
            std::memcpy(b,file.tensor_data(gate)+id*f.up_off,f.up_off);
            std::memcpy(b+f.up_off,file.tensor_data(up)+id*f.up_off,f.up_off);
            std::memcpy(b+f.down_off,file.tensor_data(down)+id*(f.bytes-f.down_off),f.bytes-f.down_off);
        }
        check(cudaHostRegister(blobs.data(),blobs.size(),cudaHostRegisterDefault));
        void* destination=nullptr;check(cudaMalloc(&destination,8*f.bytes));
        cudaStream_t stream;check(cudaStreamCreate(&stream));
        std::vector<float> activation(H);for(int i=0;i<H;++i)activation[i]=std::sin(i*.017f);
        std::vector<uint8_t> quant(f.act_bytes);cpu::native_quant_act(f,activation.data(),quant.data());
        std::vector<float> output(8*4*H);
        std::cout<<"{\"layer\":"<<layer<<",\"gate_type\":"<<types.first<<",\"down_type\":"<<types.second
                 <<",\"hidden\":"<<H<<",\"intermediate\":"<<F<<",\"working_set_bytes\":"<<blobs.size()
                 <<",\"workers\":"<<workers<<",\"host_participates\":true,\"device\":"<<device<<"}"<<std::endl;
        for(int nt:{1,2,4})for(int groups=1;groups<=8;++groups){
            std::vector<double> cpu_times,copy_times;
            for(int rep=0;rep<4;++rep){
                const int batches=64;auto begin=Clock::now();
                for(int batch=0;batch<batches;++batch){
                    cpu::ExpertJobMulti jobs[8];
                    for(int g=0;g<groups;++g){
                        const int expert=(batch*groups+g+rep*13)%64;
                        jobs[g].blob=blobs.data()+expert*f.bytes;jobs[g].nt=nt;
                        for(int t=0;t<nt;++t){jobs[g].nact[t]=quant.data();jobs[g].out[t]=output.data()+(g*4+t)*H;}
                    }
                    pool.run_split_multi_native(f,jobs,groups);
                }
                double cpu_time=ms(begin)/batches;
                begin=Clock::now();
                for(int batch=0;batch<batches;++batch){
                    for(int g=0;g<groups;++g){
                        const int expert=(batch*groups+g+rep*13)%64;
                        const auto* host=blobs.data()+expert*f.bytes;auto* gpu=static_cast<char*>(destination)+g*f.bytes;
                        check(cudaMemcpyAsync(gpu,host,f.up_off,cudaMemcpyHostToDevice,stream));
                        check(cudaMemcpyAsync(gpu+f.up_off,host+f.up_off,f.up_off,cudaMemcpyHostToDevice,stream));
                        check(cudaMemcpyAsync(gpu+f.down_off,host+f.down_off,f.bytes-f.down_off,cudaMemcpyHostToDevice,stream));
                    }
                    check(cudaStreamSynchronize(stream));
                }
                if(rep){cpu_times.push_back(cpu_time);copy_times.push_back(ms(begin)/batches);}
            }
            for(float value:output)if(!std::isfinite(value))throw std::runtime_error("nonfinite CPU calibration output");
            std::cout<<"{\"layer\":"<<layer<<",\"tokens\":"<<nt<<",\"experts\":"<<groups
                     <<",\"cpu_median_ms\":"<<median(cpu_times)<<",\"pcie_median_ms\":"<<median(copy_times)
                     <<",\"gpu_compute_included\":false,\"repetitions\":3}"<<std::endl;
        }
        check(cudaStreamDestroy(stream));check(cudaFree(destination));check(cudaHostUnregister(blobs.data()));
    }
    }
    cpu::restore_thread_affinity(affinity);
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
