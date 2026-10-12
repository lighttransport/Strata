// Prune routed experts from an assembled GLM-5.3 GGUF (strata-glm-assemble output) without requantizing.
//
//   strata-glm-prune-experts INPUT.gguf PLAN.txt OUTPUT.gguf
//
// PLAN.txt: first line "experts <old> <new>", then one line per routed layer "<layer> <kept expert ids...>"
// (ascending). Every tensor of a planned layer whose last dimension is the old expert count (gate/up/down
// experts, the router weight ffn_gate_inp and its bias exp_probs_b) keeps only the planned experts' slices, in
// plan order. glm5next.expert_count becomes <new>; the strata.expert_pack.hash.* checksums of the sliced
// projections are recomputed over the new payloads; retained (original-format) projections stay retained.
// Two extra keys record provenance: strata.prune.source_experts (U32) and strata.prune.plan_hash (U64).
// The result is reopened through strata::core::ModelArtifact, which applies the assembled-expert rules.
#include "strata/core/model.hpp"
#include <iostream>
#include <map>
#include <sstream>

namespace {
template<class T> void put(std::ostream& out,T v) { out.write(reinterpret_cast<const char*>(&v),sizeof v); }
void str(std::ostream& out,const std::string& s) { put<uint64_t>(out,s.size());out.write(s.data(),s.size()); }
constexpr uint64_t kFnvBasis=14695981039346656037ULL;
uint64_t fnv(uint64_t h,const uint8_t* p,size_t n){for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;}

struct Plan { uint32_t old_experts=0,new_experts=0;std::map<int,std::vector<uint32_t>> keep;uint64_t hash=kFnvBasis; };
Plan read_plan(const std::filesystem::path& path) {
    std::ifstream in(path);if(!in)throw std::runtime_error("cannot read plan "+path.string());
    Plan p;std::string word;
    std::stringstream whole;whole<<in.rdbuf();const std::string text=whole.str();p.hash=fnv(p.hash,reinterpret_cast<const uint8_t*>(text.data()),text.size());
    std::istringstream lines(text);std::string line;
    if(!std::getline(lines,line))throw std::runtime_error("empty plan");
    {std::istringstream h(line);h>>word>>p.old_experts>>p.new_experts;if(word!="experts"||!p.old_experts||!p.new_experts||p.new_experts>=p.old_experts)throw std::runtime_error("plan header must be: experts <old> <new>");}
    while(std::getline(lines,line)) {
        if(line.empty())continue;
        std::istringstream l(line);int layer;l>>layer;std::vector<uint32_t> ids;uint32_t e;while(l>>e)ids.push_back(e);
        if(ids.size()!=p.new_experts)throw std::runtime_error("layer "+std::to_string(layer)+" keeps "+std::to_string(ids.size())+" experts, header says "+std::to_string(p.new_experts));
        for(size_t i=0;i<ids.size();++i)if(ids[i]>=p.old_experts||(i&&ids[i]<=ids[i-1]))throw std::runtime_error("plan ids must be ascending and < old expert count");
        if(!p.keep.emplace(layer,std::move(ids)).second)throw std::runtime_error("duplicate plan layer");
    }
    if(p.keep.empty())throw std::runtime_error("plan has no layers");
    return p;
}
int layer_of(const std::string& name){if(name.rfind("blk.",0))return -1;return std::stoi(name.substr(4,name.find('.',4)-4));}
}

int main(int argc,char** argv) { try {
    if(argc!=4)throw std::runtime_error("usage: strata-glm-prune-experts INPUT.gguf PLAN.txt OUTPUT.gguf");
    const std::filesystem::path input=argv[1],output=argv[3],partial=output.string()+".partial";
    if(std::filesystem::exists(output)||std::filesystem::exists(partial))throw std::runtime_error("refusing to overwrite output or partial");
    const Plan plan=read_plan(argv[2]);
    strata::GgufFile file(input.string());
    const auto* assembled=file.get("strata.expert_pack.assembled");
    if(!assembled||!assembled->u)throw std::runtime_error("input must be an assembled GLM GGUF (strata-glm-assemble output)");
    const auto* ec=file.get("glm5next.expert_count");
    if(!ec||ec->u!=plan.old_experts)throw std::runtime_error("plan's old expert count does not match glm5next.expert_count");
    { strata::core::ModelArtifact check(input); }  // the input itself must satisfy the loader's rules

    // New tensor table.
    struct Out {const strata::TensorInfo* t;std::vector<uint64_t> shape;uint64_t bytes,src_bytes,offset=0;bool sliced=false;const std::vector<uint32_t>* keep=nullptr;};
    std::vector<Out> outs;
    const uint64_t alignment=file.get("general.alignment")?file.get("general.alignment")->u:32;
    auto align=[&](uint64_t n){return (n+alignment-1)/alignment*alignment;};
    std::map<int,int> sliced_per_layer;
    for(const auto& t:file.tensors()) {
        Out o{&t,t.shape,strata::tensor_payload_bytes(t),strata::tensor_payload_bytes(t)};
        if(!o.bytes)throw std::runtime_error("unsized tensor "+t.name);
        const int layer=layer_of(t.name);const auto it=plan.keep.find(layer);
        const bool expert_dim=!t.shape.empty()&&t.shape.back()==plan.old_experts;
        for(size_t i=0;i+1<t.shape.size();++i)if(t.shape[i]==plan.old_experts&&expert_dim==false)
            std::cerr<<"note: "<<t.name<<" has an inner dimension equal to the expert count; left unchanged\n";
        if(expert_dim) {
            const bool known=t.name.ends_with(".ffn_gate_exps.weight")||t.name.ends_with(".ffn_up_exps.weight")||t.name.ends_with(".ffn_down_exps.weight")||
                             t.name.ends_with(".ffn_gate_inp.weight")||t.name.ends_with(".exp_probs_b.bias");
            if(!known)throw std::runtime_error("unexpected tensor with an expert dimension: "+t.name);
            if(it==plan.keep.end())throw std::runtime_error("routed layer missing from the plan: "+t.name);
            if(o.bytes%plan.old_experts)throw std::runtime_error("expert slices are not byte-aligned: "+t.name);
            o.sliced=true;o.keep=&it->second;o.shape.back()=plan.new_experts;o.bytes=o.bytes/plan.old_experts*plan.new_experts;++sliced_per_layer[layer];
        }
        outs.push_back(o);
    }
    for(const auto& [layer,ids]:plan.keep)if(sliced_per_layer[layer]!=5)
        throw std::runtime_error("layer "+std::to_string(layer)+" has "+std::to_string(sliced_per_layer[layer])+" expert tensors, expected 5");

    // Header: KV copied verbatim except expert_count (rewritten) and the checksums of sliced tensors (patched later).
    std::vector<uint8_t> raw(file.data_start());
    {std::ifstream in(input,std::ios::binary);in.exceptions(std::ios::failbit|std::ios::badbit);in.read(reinterpret_cast<char*>(raw.data()),raw.size());}
    strata::Cursor c(raw.data(),raw.size());c.read<uint32_t>();c.read<uint32_t>();c.read<uint64_t>();const uint64_t kv=c.read<uint64_t>();
    std::ostringstream header(std::ios::binary);
    put<uint32_t>(header,0x46554747);put<uint32_t>(header,3);put<uint64_t>(header,outs.size());put<uint64_t>(header,kv+2);
    std::map<std::string,size_t> hash_slot;  // tensor name -> byte offset of its U64 checksum value in the header
    for(uint64_t i=0;i<kv;++i) {
        const size_t start=c.pos();const std::string key=c.str();const auto type=static_cast<strata::MetaType>(c.read<uint32_t>());
        const size_t value_start=c.pos();strata::read_value(c,type);
        if(key=="glm5next.expert_count") {
            str(header,key);put<uint32_t>(header,uint32_t(type));
            if(type==strata::MetaType::U32)put<uint32_t>(header,plan.new_experts);else if(type==strata::MetaType::U64)put<uint64_t>(header,plan.new_experts);
            else throw std::runtime_error("unexpected expert_count type");
            continue;
        }
        if(key.rfind("strata.expert_pack.hash.",0)==0) {
            const std::string name=key.substr(24);
            const auto o=std::find_if(outs.begin(),outs.end(),[&](const Out& x){return x.t->name==name;});
            if(o!=outs.end()&&o->sliced) {
                if(type!=strata::MetaType::U64)throw std::runtime_error("unexpected checksum type");
                str(header,key);put<uint32_t>(header,uint32_t(type));hash_slot[name]=size_t(header.tellp());put<uint64_t>(header,0);
                continue;
            }
        }
        (void)value_start;
        header.write(reinterpret_cast<const char*>(raw.data()+start),c.pos()-start);
    }
    str(header,"strata.prune.source_experts");put<uint32_t>(header,4);put<uint32_t>(header,plan.old_experts);
    str(header,"strata.prune.plan_hash");put<uint32_t>(header,10);put<uint64_t>(header,plan.hash);
    uint64_t offset=0;
    for(auto& o:outs) {
        str(header,o.t->name);put<uint32_t>(header,o.shape.size());for(auto d:o.shape)put<uint64_t>(header,d);
        put<uint32_t>(header,o.t->type);put<uint64_t>(header,offset);o.offset=offset;offset=align(offset+o.bytes);
    }
    auto bytes=header.str();bytes.resize(align(bytes.size()),0);
    if(std::filesystem::space(output.parent_path().empty()?".":output.parent_path()).available<offset+bytes.size()+(uint64_t(2)<<30))
        throw std::runtime_error("insufficient disk space including 2 GiB reserve");
    std::cerr<<"PRUNE plan layers="<<plan.keep.size()<<" experts "<<plan.old_experts<<"->"<<plan.new_experts<<" output_bytes="<<offset+bytes.size()<<'\n';

    std::fstream out(partial,std::ios::binary|std::ios::in|std::ios::out|std::ios::trunc);out.exceptions(std::ios::failbit|std::ios::badbit);
    out.write(bytes.data(),bytes.size());
    const std::string zeros(alignment,0);size_t done=0;std::map<std::string,uint64_t> new_hash;
    for(const auto& o:outs) {
        const uint8_t* data=file.tensor_data(*o.t);
        uint64_t h=kFnvBasis;
        auto emit=[&](const uint8_t* p,size_t n){for(size_t at=0;at<n;){const size_t k=std::min<size_t>(8<<20,n-at);if(o.sliced)h=fnv(h,p+at,k);out.write(reinterpret_cast<const char*>(p+at),k);at+=k;}};
        if(o.sliced){const size_t slice=o.src_bytes/plan.old_experts;for(uint32_t e:*o.keep)emit(data+size_t(e)*slice,slice);}
        else emit(data,o.bytes);
        out.write(zeros.data(),align(o.bytes)-o.bytes);
        if(hash_slot.count(o.t->name))new_hash[o.t->name]=h;
        file.discard_tensor_pages(*o.t,o.src_bytes);
        if(++done%100==0)std::cerr<<"pruned tensors="<<done<<"/"<<outs.size()<<'\n';
    }
    for(const auto& [name,pos]:hash_slot){out.seekp(pos);put<uint64_t>(out,new_hash.at(name));}
    out.close();
    { strata::core::ModelArtifact check(partial.string()); }
    std::filesystem::rename(partial,output);
    std::cout<<"PRUNED tensors="<<outs.size()<<" rehashed="<<hash_slot.size()<<" bytes="<<std::filesystem::file_size(output)<<'\n';
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
