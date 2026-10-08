// Column squared norms for the optimistic expert-block quality screen. No GPU.
#include "strata/artifact/expert_block_screen.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
int main(int argc,char** argv){try {
    if(argc<4||argc>5)throw std::runtime_error("usage: strata-glm-block-norms MODEL Q23_PACK OUTPUT [threads=4]");
    int threads=argc==5?std::stoi(argv[4]):4;if(threads<1||threads>16)throw std::invalid_argument("threads must be 1..16");
    const std::filesystem::path output=argv[3];if(std::filesystem::exists(output)||std::filesystem::exists(output.string()+".partial"))throw std::runtime_error("refusing to overwrite block norms");
    strata::core::ModelArtifact model(argv[1]);model.overlay_experts(argv[2]);const auto& m=model.descriptor();
    if(m.architecture!="glm5next"||m.layers.size()!=45||m.hidden!=4096)throw std::runtime_error("GLM 5.3 Flash required");
    strata::artifact::ExpertBlockNorms norms;norms.identity=model.source_fingerprint();norms.layers=int(m.layers.size());norms.experts=int(m.experts);norms.hidden=int(m.hidden);
    norms.eligible.resize(norms.layers);norms.values.resize(size_t(norms.layers)*norms.experts*2048);ggml_cpu_init();
    for(int l=3;l<norms.layers;++l) {
        const auto p="blk."+std::to_string(l)+".ffn_";const auto& g=model.at(p+"gate_exps.weight");const auto& u=model.at(p+"up_exps.weight");const auto& d=model.at(p+"down_exps.weight");
        if(g.tensor->shape!=std::vector<uint64_t>{4096,2048,uint64_t(m.experts)}||u.tensor->shape!=g.tensor->shape||d.tensor->shape!=std::vector<uint64_t>{2048,4096,uint64_t(m.experts)})throw std::runtime_error("block norm projection geometry mismatch");
        norms.eligible[l]=(g.tensor->type==10||g.tensor->type==11)&&u.tensor->type==g.tensor->type&&(d.tensor->type==10||d.tensor->type==11);
        if(!norms.eligible[l])continue;
        const auto* traits=ggml_get_type_traits(ggml_type(d.tensor->type));if(!traits||!traits->to_float)throw std::runtime_error("down projection has no dequantizer");
        const size_t row_bytes=d.bytes/(m.experts*m.hidden),slice=row_bytes*m.hidden;std::atomic<int> cursor{0};std::exception_ptr failure;std::mutex lock;
        std::vector<std::jthread> workers;
        for(int w=0;w<threads;++w)workers.emplace_back([&] {try {
            std::array<float,2048> row{};std::array<double,2048> sum{};
            for(int e;(e=cursor.fetch_add(1))<m.experts;) {
                sum.fill(0);const auto* source=d.data()+size_t(e)*slice;
                for(int r=0;r<m.hidden;++r){traits->to_float(source+r*row_bytes,row.data(),2048);for(int c=0;c<2048;++c){if(!std::isfinite(row[c]))throw std::runtime_error("nonfinite down weight");sum[c]+=double(row[c])*row[c];}}
                auto* dest=norms.values.data()+(size_t(l)*m.experts+e)*2048;for(int c=0;c<2048;++c)dest[c]=float(sum[c]);
            }
        }catch(...){std::lock_guard guard(lock);if(!failure)failure=std::current_exception();cursor.store(int(m.experts));}});
        workers.clear();if(failure)std::rethrow_exception(failure);d.file->discard_tensor_pages(*d.tensor,d.bytes);
        std::cerr<<"BLOCK_NORMS layer="<<l<<" experts="<<m.experts<<'\n';
    }
    norms.save(output);std::cout<<"{\"source_identity\":\""<<strata::artifact::block_identity(norms.identity)<<"\",\"layers\":"<<norms.layers<<",\"experts\":"<<norms.experts<<",\"block_channels\":16}\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
