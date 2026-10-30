#include "strata/artifact/safetensors.hpp"
#include <iostream>
int main(int argc,char** argv) {try {
    if(argc<2||argc>3)return 2;
    strata::artifact::Safetensors tensors(argv[1],true,false,argc==3);
    const auto physical=tensors.bytes(),files=tensors.file_bytes();
    tensors.import_remote_catalogs(argv[1]);
    size_t remote=0,local=0;
    for(const auto& [name,t]:tensors.tensors()) {
        if(!t.remote){
            const auto* data=tensors.data(t);
            if(tensors.gpu_fixed(t)&&t.bytes){const auto first=data[0];tensors.drop(t);if(tensors.data(t)[0]!=first)throw std::runtime_error("GPU-fixed source changed after release");}
            ++local;continue;
        }
        ++remote;
        bool refused=false;try {tensors.data(t);}catch(const std::runtime_error&){refused=true;}
        if(!refused)throw std::runtime_error("remote tensor read without provider succeeded");
        tensors.drop(t);
    }
    uint64_t expected_resident=files;
    if(argc==3) {
        const auto report=strata::artifact::read_json((std::filesystem::path(argv[1])/"partition.json").string());
        if(const auto* names=report.find("gpu_fixed_shards"))for(const auto& name:names->array)
            expected_resident-=std::filesystem::file_size(std::filesystem::path(argv[1])/name.string());
    }
    if(tensors.resident_file_bytes()!=expected_resident)throw std::runtime_error("GPU-fixed source was incorrectly charged to host RAM");
    if(!remote||!local||physical!=tensors.bytes()||files!=tensors.file_bytes())
        throw std::runtime_error("catalog changed physical byte accounting");
    std::cout<<"PASS catalog: explicit remote reads, local mapping and physical admission bytes\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
