// Replay real routed weight shapes; activations are deterministic synthetic data.
#include "strata/core/model.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/numa_weights.hpp"
#include <set>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace strata::kernels::cpu;
int main(int argc,char** argv){try {
    if(argc<3)throw std::runtime_error("usage: strata-glm-q2-kernel-bench MODEL ROUTES [mmap|numa] [nt=1] [workers=15] [rounds=5]");
    if(argc>3 && std::string(argv[3])!="numa" && std::string(argv[3])!="mmap")throw std::runtime_error("invalid placement policy");
    bool owned=argc>3 && std::string(argv[3])=="numa";int nt=argc>4?std::stoi(argv[4]):1,workers=argc>5?std::stoi(argv[5]):15,rounds=argc>6?std::stoi(argv[6]):5;
    if(nt<1||nt>4||workers<1||rounds<1||rounds>20)throw std::runtime_error("invalid benchmark bounds");
    strata::core::ModelArtifact model(argv[1]);
    const auto& descriptor=model.descriptor();
    if(descriptor.architecture!="glm5next" || descriptor.hidden!=4096 || descriptor.experts!=288 || descriptor.top_k!=8 || descriptor.layers.size()!=45)
        throw std::runtime_error("benchmark requires the GLM5.3Flash 4096/288/top8 geometry");
    ExpertPool pool(workers,true);
    if(owned&&!pool.numa_rows_available())throw std::runtime_error("two-node pinned pool unavailable");
    std::map<int,std::map<int,std::vector<int>>> routes;std::ifstream in(argv[2]);std::string line;
    while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int pos,layer,e;if(!(row>>pos>>layer))continue;std::vector<int> ids;while(row>>e)ids.push_back(e);
        if(layer>=3 && layer<45 && (ids.size()!=8 || std::any_of(ids.begin(),ids.end(),[&](int id){return id<0 || id>=descriptor.experts;}) || std::set<int>(ids.begin(),ids.end()).size()!=ids.size()))throw std::runtime_error("invalid top8 routing row");
        routes[pos][layer]=ids;}
    if(routes.size()<size_t(nt))throw std::runtime_error("insufficient routing positions");
    auto host=physical_cores(false);ThreadAffinity previous;if(!host.empty())previous=pin_current_thread(host[0]);
    struct Layer {NativeFmt f;std::vector<ExpertJobMulti> jobs;std::array<std::unique_ptr<NumaTensor>,3> owned;std::vector<std::vector<uint8_t>> acts;std::vector<float> out;};
    std::vector<Layer> layers;size_t bytes=0;
    for(int l=3;l<int(model.descriptor().layers.size());++l) {
        Layer x;std::string error;std::string p="blk."+std::to_string(l)+".ffn_";
        const auto& G=model.at(p+"gate_exps.weight");const auto& U=model.at(p+"up_exps.weight");const auto& D=model.at(p+"down_exps.weight");
        if(G.tensor->shape!=std::vector<uint64_t>{4096,2048,288} || U.tensor->shape!=G.tensor->shape || D.tensor->shape!=std::vector<uint64_t>{2048,4096,288})
            throw std::runtime_error("benchmark requires 2048-row GLM5.3Flash experts");
        if(!native_fmt(G.tensor->type,D.tensor->type,4096,2048,x.f,error))throw std::runtime_error(error);
        x.f.swiglu_limit=10;
        std::map<int,int> selected;auto row=routes.begin();for(int t=0;t<nt;++t,++row)for(int e:row->second.at(l))selected.emplace(e,selected.size());
        // Only selected expert slices are copied; each node stores local half rows.
        std::array<std::vector<uint8_t>,3> source;
        const strata::core::ArtifactTensor* tensors[]={&G,&U,&D};
        for(int part=0;part<3;++part) {
            size_t slice=tensors[part]->bytes/model.descriptor().experts;if(!owned)continue;
            source[part].resize(selected.size()*slice);
            for(auto [e,index]:selected)std::memcpy(source[part].data()+size_t(index)*slice,tensors[part]->data()+size_t(e)*slice,slice);
            if(owned)x.owned[part]=std::make_unique<NumaTensor>(source[part].data(),part==2?x.f.d_row:x.f.gu_row,part==2?4096:2048,selected.size(),pool.numa_cores());
        }
        x.acts.resize(nt);std::vector<float> a(4096);
        for(int t=0;t<nt;++t){for(int i=0;i<4096;++i)a[i]=std::sin(float(i+t*19)*.017f)*.1f;x.acts[t].resize(x.f.act_bytes);native_quant_act(x.f,a.data(),x.acts[t].data());}
        x.jobs.resize(selected.size());x.out.resize(size_t(nt)*8*4096);
        for(auto [e,index]:selected) {auto& j=x.jobs[index];j.blob=G.data()+size_t(e)*x.f.up_off;j.native_up=U.data()+size_t(e)*x.f.up_off;j.native_down=D.data()+size_t(e)*(x.f.bytes-x.f.down_off);
            if(owned)for(int node=0;node<2;++node)j.numa[node]={x.owned[0]->shard(index,node),x.owned[1]->shard(index,node),x.owned[2]->shard(index,node)};}
        row=routes.begin();for(int t=0;t<nt;++t,++row){int k=0;for(int e:row->second.at(l)){auto& j=x.jobs[selected.at(e)];int n=j.nt++;j.nact[n]=x.acts[t].data();j.out[n]=x.out.data()+size_t(t*8+k++)*4096;}}
        bytes+=x.jobs.size()*x.f.bytes;layers.push_back(std::move(x));
    }
    std::cout<<"policy="<<(owned?"numa":"mmap")<<" nt="<<nt<<" workers="<<workers<<" packed_bytes="<<bytes<<" scope=real_weight_routing_synthetic_activations_expert_only\n";
    std::vector<std::vector<float>> references;
    for(int round=-1;round<rounds;++round) {
        double ms=0,checksum=0;uint64_t output_hash=14695981039346656037ULL;
        for(auto& x:layers) {auto start=std::chrono::steady_clock::now();pool.run_split_multi_native(x.f,x.jobs.data(),x.jobs.size());ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();for(float v:x.out){if(!std::isfinite(v))throw std::runtime_error("nonfinite output");checksum+=v;const auto* bits=reinterpret_cast<const uint8_t*>(&v);for(size_t k=0;k<sizeof(v);++k){output_hash^=bits[k];output_hash*=1099511628211ULL;}}}
                if(round<0){for(auto& x:layers)references.push_back(x.out);continue;}
        for(size_t l=0;l<layers.size();++l)if(layers[l].out!=references[l])throw std::runtime_error("unstable repeated outputs");
        std::cout<<"round="<<round<<" ms="<<ms<<" packed_GB_s="<<bytes/ms/1e6<<" expert_only_tok_s="<<nt*1000/ms<<" output_hash="<<output_hash<<" checksum="<<checksum<<'\n';
    }
    if(previous.valid)restore_thread_affinity(previous);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
