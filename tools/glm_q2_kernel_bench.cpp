// Replay real routed weights with captured activations, or synthetic inputs for CSV traces.
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
#include "ggml.h"
#include <atomic>
#include <thread>
#include "strata/kernels/cpu/q23_avx2.hpp"
using namespace strata::kernels::cpu;
int main(int argc,char** argv){try {
    if(argc<3)throw std::runtime_error("usage: strata-glm-q2-kernel-bench MODEL ROUTES [mmap|numa] [nt=1] [workers=15] [rounds=5] [native|q2|q23] [first-layer=3] [last-layer=44] [row|direct|lut] [skip-positions=0]");
    if(argc>3 && std::string(argv[3])!="numa" && std::string(argv[3])!="mmap")throw std::runtime_error("invalid placement policy");
    bool owned=argc>3 && std::string(argv[3])=="numa";int nt=argc>4?std::stoi(argv[4]):1,workers=argc>5?std::stoi(argv[5]):15,rounds=argc>6?std::stoi(argv[6]):5;
    if(nt<1||nt>MAXT||workers<1||rounds<1||rounds>20)throw std::runtime_error("invalid benchmark bounds");
    const std::string conversion=argc>7?argv[7]:"native";
    if(conversion!="native" && conversion!="q2" && conversion!="q23")throw std::runtime_error("invalid conversion");
    const int first=argc>8?std::stoi(argv[8]):3,last=argc>9?std::stoi(argv[9]):44;
    if(first<3||last>44||first>last)throw std::runtime_error("invalid layer range");
    const std::string layout=argc>10?argv[10]:"row";
    if(layout!="row" && layout!="direct" && layout!="lut")throw std::runtime_error("invalid layout");
    const int skip=argc>11?std::stoi(argv[11]):0;
    if(skip<0)throw std::runtime_error("negative routing position offset");
    strata::core::ModelArtifact model(argv[1]);
    const auto& descriptor=model.descriptor();
    if(descriptor.architecture!="glm5next" || descriptor.hidden!=4096 || (descriptor.experts!=288 && descriptor.experts!=144) || descriptor.top_k!=8 || descriptor.layers.size()!=45)
        throw std::runtime_error("benchmark requires the GLM5.3Flash 4096/(144 or 288)/top8 geometry");
    const bool flow = std::getenv("STRATA_GLM_LAYER_FLOW") && std::string(std::getenv("STRATA_GLM_LAYER_FLOW")) == "1";
    const bool canonical = std::getenv("STRATA_GLM_CANON") && std::string(std::getenv("STRATA_GLM_CANON")) == "1";
    ExpertPool pool(workers,true);
    if(owned&&!pool.numa_rows_available())throw std::runtime_error("two-node pinned pool unavailable");
    std::map<int,std::map<int,std::vector<int>>> routes;std::map<int,std::map<int,std::vector<float>>> captured;
    std::ifstream in(argv[2],std::ios::binary);std::string line;uint32_t magic=0;in.read(reinterpret_cast<char*>(&magic),4);in.clear();in.seekg(0);
    if(magic==0x31435247) {
        while(in.peek()!=std::ifstream::traits_type::eof()) {
            uint32_t h[6];if(!in.read(reinterpret_cast<char*>(h),sizeof h)||h[0]!=magic||h[1]<3||h[1]>44||h[3]<1||h[3]>8||h[4]!=4096||h[5]!=8)throw std::runtime_error("invalid activation trace record");
            std::vector<int> ids(h[3]*8);std::vector<float> a(h[3]*4096);
            if(!in.read(reinterpret_cast<char*>(ids.data()),ids.size()*4)||!in.read(reinterpret_cast<char*>(a.data()),a.size()*4))throw std::runtime_error("truncated activation trace");
            for(uint32_t t=0;t<h[3];++t)if(!routes[h[2]+t].contains(h[1])) {
                auto begin=ids.begin()+t*8;std::vector<int> selected(begin,begin+8);
                if(std::set<int>(selected.begin(),selected.end()).size()!=8||std::any_of(selected.begin(),selected.end(),[&](int e){return e<0||e>=descriptor.experts;}))throw std::runtime_error("invalid captured experts");
                routes[h[2]+t][h[1]]=std::move(selected);captured[h[2]+t][h[1]]=std::vector<float>(a.begin()+t*4096,a.begin()+(t+1)*4096);
            }
        }
    } else while(std::getline(in,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int pos,layer,e;if(!(row>>pos>>layer))continue;std::vector<int> ids;while(row>>e)ids.push_back(e);
        if(layer>=3 && layer<45 && (ids.size()!=8 || std::any_of(ids.begin(),ids.end(),[&](int id){return id<0 || id>=descriptor.experts;}) || std::set<int>(ids.begin(),ids.end()).size()!=ids.size()))throw std::runtime_error("invalid top8 routing row");
        routes[pos][layer]=ids;}
    if(routes.size()<size_t(skip)+nt)throw std::runtime_error("insufficient routing positions");
    for(int i=0;i<skip;++i)routes.erase(routes.begin());
    auto host=physical_cores(false);ThreadAffinity previous;if(!host.empty())previous=pin_current_thread(host[0]);
    struct Layer {NativeFmt f;std::vector<ExpertJobMulti> jobs;std::array<std::unique_ptr<NumaTensor>,3> owned;std::array<std::vector<uint8_t>,3> packed;std::vector<std::vector<uint8_t>> acts;std::vector<float> out;};
    std::vector<Layer> layers;size_t bytes=0;
    for(int l=first;l<=last;++l) {
        Layer x;std::string error;std::string p="blk."+std::to_string(l)+".ffn_";
        const auto& G=model.at(p+"gate_exps.weight");const auto& U=model.at(p+"up_exps.weight");const auto& D=model.at(p+"down_exps.weight");
        if(G.tensor->shape!=std::vector<uint64_t>{4096,2048,(uint64_t)descriptor.experts} || U.tensor->shape!=G.tensor->shape || D.tensor->shape!=std::vector<uint64_t>{2048,4096,(uint64_t)descriptor.experts})
            throw std::runtime_error("benchmark requires 2048-row GLM5.3Flash experts");
        int gt=G.tensor->type,dt=D.tensor->type;
        if(conversion!="native" && gt==17)gt=10;
        if(conversion=="q23" && dt==18)dt=11;
        if(!native_fmt(gt,dt,4096,2048,x.f,error))throw std::runtime_error(error);
        x.f.swiglu_limit=descriptor.layers[l].swiglu_limit;
        x.f.canon=canonical;
        if(canonical && !((gt==10||gt==11)&&(dt==10||dt==11)))throw std::runtime_error("canonical benchmark requires Q2_K/Q3_K");
        x.f.q23_layout=layout=="row"?0:layout=="direct"?1:2;
        std::map<int,int> selected;auto row=routes.begin();for(int t=0;t<nt;++t,++row)for(int e:row->second.at(l))selected.emplace(e,selected.size());
        // Only selected expert slices are copied; each node stores local half rows.
        std::array<std::vector<uint8_t>,3> source;
        const strata::core::ArtifactTensor* tensors[]={&G,&U,&D};
        for(int part=0;part<3;++part) {
            const int target=part==2?dt:gt,original=tensors[part]->tensor->type,columns=part==2?2048:4096,rows=part==2?4096:2048;
            const size_t original_slice=tensors[part]->bytes/model.descriptor().experts,slice=(part==2?x.f.d_row:x.f.gu_row)*rows;
            if(target!=original) {
                x.packed[part].resize(selected.size()*slice);
                std::vector<std::pair<int,int>> ids(selected.begin(),selected.end());std::atomic<size_t> cursor{0};
                std::vector<std::jthread> converters;
                for(int w=0;w<std::min<int>(8,ids.size());++w)converters.emplace_back([&,w] {
                    if(!host.empty())pin_current_thread(host[w%host.size()]);
                    std::vector<float> weights(size_t(columns)*rows);
                    for(size_t j;(j=cursor.fetch_add(1))<ids.size();) {
                        auto [e,index]=ids[j];ggml_get_type_traits(ggml_type(original))->to_float(tensors[part]->data()+size_t(e)*original_slice,weights.data(),weights.size());
                        ggml_quantize_chunk(ggml_type(target),weights.data(),x.packed[part].data()+size_t(index)*slice,0,rows,columns,nullptr);
                    }
                });
                for(auto& t:converters)t.join();
                if(x.f.q23_layout) {
                    std::vector<uint8_t> tiled(x.packed[part].size());
                    for(size_t e=0;e<selected.size();++e)q23_pack(target,x.packed[part].data()+e*slice,tiled.data()+e*slice,columns,rows);
                    x.packed[part]=std::move(tiled);
                }
            }
            if(owned) {
                if(target==original) {source[part].resize(selected.size()*slice);for(auto [e,index]:selected)std::memcpy(source[part].data()+size_t(index)*slice,tensors[part]->data()+size_t(e)*slice,slice);}
                x.owned[part]=std::make_unique<NumaTensor>(target==original?source[part].data():x.packed[part].data(),part==2?x.f.d_row:x.f.gu_row,rows,selected.size(),pool.numa_cores(),std::getenv("STRATA_BENCH_HUGE")!=nullptr);
            }
        }
        x.acts.resize(nt);std::vector<float> a(4096);
        row=routes.begin();for(int t=0;t<nt;++t,++row){if(captured.empty()){for(int i=0;i<4096;++i)a[i]=std::sin(float(i+t*19)*.017f)*.1f;}else a=captured.at(row->first).at(l);x.acts[t].resize(x.f.act_bytes);native_quant_act(x.f,a.data(),x.acts[t].data());}
        x.jobs.resize(selected.size());x.out.resize(size_t(nt)*8*4096);
        for(auto [e,index]:selected) {auto& j=x.jobs[index];j.blob=G.data()+size_t(e)*x.f.up_off;j.native_up=U.data()+size_t(e)*x.f.up_off;j.native_down=D.data()+size_t(e)*(x.f.bytes-x.f.down_off);
            if(!x.packed[0].empty())j.blob=x.packed[0].data()+size_t(index)*x.f.up_off;
            if(!x.packed[1].empty())j.native_up=x.packed[1].data()+size_t(index)*x.f.up_off;
            if(!x.packed[2].empty())j.native_down=x.packed[2].data()+size_t(index)*(x.f.bytes-x.f.down_off);
            if(owned)for(int node=0;node<2;++node)j.numa[node]={x.owned[0]->shard(index,node),x.owned[1]->shard(index,node),x.owned[2]->shard(index,node)};}
        row=routes.begin();for(int t=0;t<nt;++t,++row){int k=0;for(int e:row->second.at(l)){auto& j=x.jobs[selected.at(e)];int n=j.nt++;j.nact[n]=x.acts[t].data();j.out[n]=x.out.data()+size_t(t*8+k++)*4096;}}
        bytes+=x.jobs.size()*x.f.bytes;layers.push_back(std::move(x));std::cerr<<"prepared_layer="<<l<<'\n';
    }
    std::cout<<"policy="<<(owned?"numa":"mmap")<<" conversion="<<conversion<<" layout="<<layout<<" nt="<<nt<<" workers="<<workers<<" skip_positions="<<skip<<" first_position="<<routes.begin()->first<<" packed_bytes="<<bytes<<" scope=real_weight_expert_only activations="<<(captured.empty()?"synthetic":"captured")<<'\n';
    std::cout<<"flow="<<flow<<" canon="<<canonical<<" reduction_weights=uniform\n";
    std::vector<float> route_weights(nt*8,.125f), routed_output(nt*4096);
    std::vector<std::vector<float>> references;
    for(int round=-1;round<rounds;++round) {
        double ms=0,checksum=0;uint64_t output_hash=14695981039346656037ULL;
        for(auto& x:layers) {auto start=std::chrono::steady_clock::now();if(flow)pool.run_layer_native(x.f,x.jobs.data(),x.jobs.size(),x.out.data(),route_weights.data(),routed_output.data(),nt,8);else pool.run_split_multi_native(x.f,x.jobs.data(),x.jobs.size());ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();for(float v:x.out){if(!std::isfinite(v))throw std::runtime_error("nonfinite output");checksum+=v;const auto* bits=reinterpret_cast<const uint8_t*>(&v);for(size_t k=0;k<sizeof(v);++k){output_hash^=bits[k];output_hash*=1099511628211ULL;}}}
                if(round<0){for(auto& x:layers)references.push_back(x.out);continue;}
        for(size_t l=0;l<layers.size();++l)if(layers[l].out!=references[l])throw std::runtime_error("unstable repeated outputs");
        std::cout<<"round="<<round<<" ms="<<ms<<" packed_GB_s="<<bytes/ms/1e6<<" expert_only_tok_s="<<nt*1000/ms<<" output_hash="<<output_hash<<" checksum="<<checksum<<'\n';
    }
    if(previous.valid)restore_thread_affinity(previous);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
