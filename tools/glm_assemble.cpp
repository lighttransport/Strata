// Assemble an immutable source GGUF and its validated expert overlay without requantization.
// Build: c++ -O2 -std=c++20 -Iinclude tools/glm_assemble.cpp -o strata-glm-assemble
#include "strata/core/model.hpp"
#include <iostream>
#include <sstream>

namespace {
template<class T> void put(std::ostream& out,T v) { out.write(reinterpret_cast<const char*>(&v),sizeof v); }
void str(std::ostream& out,const std::string& s) { put<uint64_t>(out,s.size());out.write(s.data(),s.size()); }
struct Metadata { uint64_t count=0;std::string bytes; };
Metadata metadata(const std::filesystem::path& path) {
    strata::GgufFile file(path.string());
    std::vector<uint8_t> header(file.data_start());
    std::ifstream input(path,std::ios::binary);input.exceptions(std::ios::failbit|std::ios::badbit);
    input.read(reinterpret_cast<char*>(header.data()),header.size());
    strata::Cursor c(header.data(),header.size());c.read<uint32_t>();c.read<uint32_t>();c.read<uint64_t>();
    Metadata result;result.count=c.read<uint64_t>();
    for(uint64_t i=0;i<result.count;++i) { c.str();strata::read_value(c,static_cast<strata::MetaType>(c.read<uint32_t>())); }
    result.bytes.assign(reinterpret_cast<const char*>(header.data()+24),c.pos()-24);return result;
}
}
int main(int argc,char** argv) { try {
    if(argc!=4)throw std::runtime_error("usage: strata-glm-assemble MODEL PACK OUTPUT");
    const std::filesystem::path source=argv[1],pack=argv[2],output=argv[3],partial=output.string()+".partial";
    if(std::filesystem::exists(output)||std::filesystem::exists(partial))throw std::runtime_error("refusing to overwrite output or partial");
    strata::GgufFile base(source.string());
    if(base.get("split.count")||base.get("strata.expert_pack.version"))throw std::runtime_error("assembly requires an unsplit original GGUF");
    strata::core::ModelArtifact model(source);
    const auto fingerprint=model.source_fingerprint();model.overlay_experts(pack);
    const auto source_meta=metadata(source),pack_meta=metadata(pack);
    strata::GgufFile pack_header(pack.string());
    for(const auto& [key,value]:pack_header.metadata())
        if(base.get(key))throw std::runtime_error("duplicate metadata key: "+key);
    std::ostringstream header(std::ios::binary);
    put<uint32_t>(header,0x46554747);put<uint32_t>(header,3);put<uint64_t>(header,model.tensors().size());
    put<uint64_t>(header,source_meta.count+pack_meta.count+1);
    header<<source_meta.bytes<<pack_meta.bytes;
    str(header,"strata.expert_pack.assembled");put<uint32_t>(header,7);put<uint8_t>(header,1);
    const uint64_t alignment=base.get("general.alignment")?base.get("general.alignment")->u:32;
    auto align=[&](uint64_t n){return (n+alignment-1)/alignment*alignment;};
    uint64_t offset=0;
    for(const auto& [name,t]:model.tensors()) {
        str(header,name);put<uint32_t>(header,t.tensor->shape.size());
        for(auto d:t.tensor->shape)put<uint64_t>(header,d);
        put<uint32_t>(header,t.tensor->type);put<uint64_t>(header,offset);offset=align(offset+t.bytes);
    }
    auto bytes=header.str();bytes.resize(align(bytes.size()),0);
    if(std::filesystem::space(output.parent_path().empty()?".":output.parent_path()).available<offset+bytes.size()+(uint64_t(2)<<30))
        throw std::runtime_error("insufficient disk space including 2 GiB reserve");
    std::ofstream out(partial,std::ios::binary);out.exceptions(std::ios::failbit|std::ios::badbit);out.write(bytes.data(),bytes.size());
    const std::string zeros(alignment,0);size_t copied=0,checked=0;
    for(const auto& [name,t]:model.tensors()) {
        uint64_t hash=14695981039346656037ULL;
        const auto* expected=t.file->get("strata.expert_pack.hash."+name);
        const auto* data=t.data();
        for(size_t at=0;at<t.bytes;) {
            const size_t n=std::min<size_t>(8<<20,t.bytes-at);
            if(expected)for(size_t i=0;i<n;++i){hash^=data[at+i];hash*=1099511628211ULL;}
            out.write(reinterpret_cast<const char*>(data+at),n);at+=n;
        }
        if(expected) {if(hash!=expected->u)throw std::runtime_error("payload checksum mismatch: "+name);++checked;}
        out.write(zeros.data(),align(t.bytes)-t.bytes);t.file->discard_tensor_pages(*t.tensor,t.bytes);
        if(++copied%100==0)std::cerr<<"assembled tensors="<<copied<<'\n';
    }
    out.close();
    if(strata::core::ModelArtifact(source).source_fingerprint()!=fingerprint)throw std::runtime_error("source changed during assembly");
    strata::core::ModelArtifact assembled(partial);
    if(assembled.tensors().size()!=model.tensors().size()||assembled.census().total!=model.census().total)
        throw std::runtime_error("assembled tensor census mismatch");
    std::filesystem::rename(partial,output);
    std::cout<<"ASSEMBLED tensors="<<copied<<" verified_experts="<<checked<<" bytes="<<std::filesystem::file_size(output)
             <<" source_fingerprint="<<fingerprint<<'\n';return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
