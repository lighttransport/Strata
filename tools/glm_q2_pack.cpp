// Stream a converted routed-expert GGUF sidecar. Original files are read-only.
#include "strata/core/model.hpp"
#include "strata/core/runtime_memory.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <sstream>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
template<class T> void write(std::ostream& out,T value){out.write(reinterpret_cast<const char*>(&value),sizeof value);}
void string(std::ostream& out,const std::string& value){write<uint64_t>(out,value.size());out.write(value.data(),value.size());}
struct Entry {std::string name;const strata::core::ArtifactTensor* source;uint32_t type;uint64_t offset,bytes,hash=0;double error=0,norm=0;};
uint64_t hash_bytes(uint64_t hash,const uint8_t* p,size_t n){for(size_t i=0;i<n;++i){hash^=p[i];hash*=1099511628211ULL;}return hash;}
std::string header(const std::vector<Entry>& entries,uint64_t source,bool calibrated) {
    std::ostringstream out(std::ios::binary);write<uint32_t>(out,0x46554747);write<uint32_t>(out,3);write<uint64_t>(out,entries.size());write<uint64_t>(out,entries.size()+3);
    string(out,"strata.expert_pack.version");write<uint32_t>(out,4);write<uint32_t>(out,1);
    string(out,"strata.expert_pack.source");write<uint32_t>(out,10);write<uint64_t>(out,source);
    string(out,"strata.expert_pack.calibrated");write<uint32_t>(out,7);write<uint8_t>(out,calibrated);
    for(const auto& e:entries){string(out,"strata.expert_pack.hash."+e.name);write<uint32_t>(out,10);write<uint64_t>(out,e.hash);}
    for(const auto& e:entries){string(out,e.name);write<uint32_t>(out,3);for(auto d:e.source->tensor->shape)write<uint64_t>(out,d);write<uint32_t>(out,e.type);write<uint64_t>(out,e.offset);}
    auto result=out.str();result.resize((result.size()+31)/32*32,0);return result;
}
}
int main(int argc,char** argv){try {
    if(argc>=2&&std::string(argv[1])=="--verify") {
        if(argc!=4)throw std::runtime_error("usage: strata-glm-q2-pack --verify MODEL PACK");
        strata::core::ModelArtifact model(argv[2]);model.overlay_experts(argv[3]);size_t checked=0;
        strata::core::lock_small_runtime_mappings();
        for(const auto& [name,t]:model.tensors())if(const auto* expected=t.file->get("strata.expert_pack.hash."+name)) {
            if(hash_bytes(14695981039346656037ULL,t.data(),t.bytes)!=expected->u)throw std::runtime_error("payload checksum mismatch: "+name);
            ++checked;t.file->discard_tensor_pages(*t.tensor,t.bytes);
        }
        std::cout<<"PACK verified projections="<<checked<<'\n';return 0;
    }
    if(argc<3)throw std::runtime_error("usage: strata-glm-q2-pack MODEL OUTPUT [--calibration=DIR | --uncalibrated] [--first-layer=3] [--last-layer=44] [--threads=8] [--down=q3|q2] [--reuse=PACK] [--all-layers]");
    // --down=q2 stores down projections as Q2_K (84 instead of 110 bytes per 256 weights). --reuse=PACK copies
    // every projection that an existing pack of the same source already holds in the requested type.
    // --all-layers also converts the layers the UD quantization kept in a higher format (IQ3_XXS gate/up, IQ4_XS down).
    std::string calibration,reuse;bool uncalibrated=false,all_layers=false;int first=3,last=44,threads=8;uint32_t down_type=11;
    for(int i=3;i<argc;++i){std::string arg=argv[i];if(arg=="--all-layers")all_layers=true;else if(arg=="--down=q2")down_type=10;else if(arg=="--down=q3")down_type=11;else if(arg.starts_with("--reuse="))reuse=arg.substr(8);else if(arg.starts_with("--calibration="))calibration=arg.substr(14);else if(arg=="--uncalibrated")uncalibrated=true;else if(arg.starts_with("--first-layer="))first=std::stoi(arg.substr(14));else if(arg.starts_with("--last-layer="))last=std::stoi(arg.substr(13));else if(arg.starts_with("--threads="))threads=std::stoi(arg.substr(10));else throw std::runtime_error("unknown argument "+arg);}
    if(first<3||last>44||first>last||threads<1||threads>16||uncalibrated==!calibration.empty())throw std::runtime_error("invalid conversion settings; choose calibration or explicit uncalibrated, threads 1..16");
    strata::core::ModelArtifact model(argv[1]);const uint64_t experts=model.descriptor().experts;
    if(model.is_exl3()||model.descriptor().architecture!="glm5next"||model.descriptor().hidden!=4096||experts>288||experts%16)throw std::runtime_error("GLM model required");
    const std::filesystem::path output(argv[2]),partial=output.string()+".partial";
    if(std::filesystem::exists(output)||std::filesystem::exists(partial))throw std::runtime_error("refusing to overwrite existing pack or partial file");
    std::vector<Entry> entries;uint64_t offset=0;
    for(int l=first;l<=last;++l)for(const char* part:{"gate","up","down"}) {
        const auto name="blk."+std::to_string(l)+".ffn_"+part+"_exps.weight";const auto& t=model.at(name);
        const bool down=std::string(part)=="down";const uint32_t type=down?down_type:10;
        // IQ2_XS gate/up and IQ3_XXS down (UD-Q2_K_XL), or K-quant sources (Q4_K/Q5_K/Q6_K, e.g. a REAP-50 Q4_K_M).
        const bool kquant=t.tensor->type==12||t.tensor->type==13||t.tensor->type==14;
        const bool higher=all_layers&&t.tensor->type==(down?23u:18u);   // IQ4_XS down, IQ3_XXS gate/up
        const bool assembled_down=down&&down_type==10&&t.tensor->type==11&&model.has_expert_pack();
        if(t.tensor->type!=(down?18u:17u)&&!kquant&&!higher&&!assembled_down)continue;
        if(t.tensor->shape!=std::vector<uint64_t>{down?2048u:4096u,down?4096u:2048u,experts})throw std::runtime_error("unsupported expert geometry: "+name);
        const uint64_t bytes=t.tensor->elements()/256*(type==10?84:110);entries.push_back({name,&t,type,offset,bytes});offset+=bytes;
    }
    if(entries.empty())throw std::runtime_error("no eligible projections");
    if(std::filesystem::space(output.parent_path().empty()?".":output.parent_path()).available<offset+(uint64_t(2)<<30))throw std::runtime_error("insufficient disk capacity for pack plus 2 GiB reserve");
    const auto fingerprint=model.source_fingerprint();auto h=header(entries,fingerprint,!calibration.empty());
    std::ofstream out(partial,std::ios::binary|std::ios::trunc);out.exceptions(std::ios::failbit|std::ios::badbit);out.write(h.data(),h.size());
    ggml_cpu_init();
    strata::core::lock_small_runtime_mappings();
    std::unique_ptr<strata::core::ModelArtifact> previous;
    if(!reuse.empty()) {
        // The overlay checks that the old pack was made from this source.
        previous=std::make_unique<strata::core::ModelArtifact>(argv[1]);previous->overlay_experts(reuse);
    }
    // Written pages leave the page cache as they are flushed: a pack is larger than the memory a loaded model leaves free.
    auto drop_written=[&] {
#ifdef __linux__
        const int fd=::open(partial.c_str(),O_RDONLY);
        if(fd>=0){::fdatasync(fd);::posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);::close(fd);}
#endif
    };
    for(auto& e:entries) {
        const auto& t=*e.source;const int cols=t.tensor->shape[0],rows=t.tensor->shape[1],experts=t.tensor->shape[2];
        const size_t slice=e.bytes/experts,source_slice=t.bytes/experts;
        if(previous) {
            const auto& old=previous->at(e.name);
            if(old.tensor->type==e.type&&old.bytes==e.bytes&&old.file!=t.file) {
                out.write(reinterpret_cast<const char*>(old.data()),e.bytes);e.hash=hash_bytes(14695981039346656037ULL,old.data(),e.bytes);
                const auto* expected=old.file->get("strata.expert_pack.hash."+e.name);
                if(!expected||expected->u!=e.hash)throw std::runtime_error("reused projection fails its checksum: "+e.name);
                out.flush();old.file->discard_tensor_pages(*old.tensor,old.bytes);drop_written();
                std::cerr<<"PACK tensor="<<e.name<<" bytes="<<e.bytes<<" reused\n";continue;
            }
        }
        std::vector<float> importance;
        if(!calibration.empty()) {
            const auto l=e.name.substr(0,e.name.find(".ffn_"));const auto p=std::filesystem::path(calibration)/(l+(e.name.find(".ffn_down_")==std::string::npos?".gu.f32":".down.f32"));
            importance.resize(size_t(cols)*experts);std::ifstream in(p,std::ios::binary);
            if(!in.read(reinterpret_cast<char*>(importance.data()),importance.size()*4)||in.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("invalid calibration tensor "+p.string());
            for(float v:importance)if(!std::isfinite(v)||v<0)throw std::runtime_error("nonfinite or negative importance");
        }
        // One float matrix per worker, plus one decoded row: <513 MiB at 16 workers.
        e.hash=14695981039346656037ULL;
        for(int base=0;base<experts;base+=threads) {
            const int count=std::min(threads,experts-base);std::vector<std::vector<uint8_t>> packed(count,std::vector<uint8_t>(slice));
            std::vector<std::jthread> workers;std::vector<std::exception_ptr> errors(count);std::vector<double> err(count),norm(count);
            for(int j=0;j<count;++j)workers.emplace_back([&,j] {try {
                const int expert=base+j;std::vector<float> weights(size_t(cols)*rows),decoded(cols);
                ggml_get_type_traits(ggml_type(t.tensor->type))->to_float(t.data()+size_t(expert)*source_slice,weights.data(),weights.size());
                for(float value:weights)if(!std::isfinite(value))throw std::runtime_error("nonfinite source expert weight");
                const float* im=importance.empty()?nullptr:importance.data()+size_t(expert)*cols;
                const auto written=ggml_quantize_chunk(ggml_type(e.type),weights.data(),packed[j].data(),0,rows,cols,im);if(written!=slice)throw std::runtime_error("quantizer byte count mismatch");
                for(int r=0;r<rows;++r) {
                    ggml_get_type_traits(ggml_type(e.type))->to_float(packed[j].data()+size_t(r)*slice/rows,decoded.data(),cols);
                    for(int k=0;k<cols;++k){const double w=weights[size_t(r)*cols+k],scale=im?im[k]:1.,delta=double(decoded[k])-w;err[j]+=scale*delta*delta;norm[j]+=scale*w*w;}
                }
            }catch(...){errors[j]=std::current_exception();}});
            for(auto& worker:workers)worker.join();
            for(auto error:errors)if(error)std::rethrow_exception(error);
            // First batch materializes worker arenas and retained runtime pages.
            if(base==0)strata::core::lock_small_runtime_mappings();
            for(int j=0;j<count;++j){out.write(reinterpret_cast<const char*>(packed[j].data()),slice);e.hash=hash_bytes(e.hash,packed[j].data(),slice);e.error+=err[j];e.norm+=norm[j];}
        }
        out.flush();t.file->discard_tensor_pages(*t.tensor,t.bytes);drop_written();
        std::cerr<<"PACK tensor="<<e.name<<" bytes="<<e.bytes<<" normalized_mse="<<std::setprecision(9)<<e.error/std::max(e.norm,1e-30)<<'\n';
    }
    if(model.source_fingerprint()!=fingerprint)throw std::runtime_error("source identity changed during conversion");
    h=header(entries,fingerprint,!calibration.empty());out.seekp(0);out.write(h.data(),h.size());out.close();
    std::filesystem::rename(partial,output);
    std::ofstream report(output.string()+".quality.json");report.exceptions(std::ios::failbit|std::ios::badbit);report<<"{\"source_fingerprint\":\""<<fingerprint<<"\",\"calibrated\":"<<(!calibration.empty()?"true":"false")<<",\"projections\":[";
    for(size_t i=0;i<entries.size();++i){const auto& e=entries[i];if(i)report<<',';report<<"{\"name\":\""<<e.name<<"\",\"bytes\":"<<e.bytes<<",\"normalized_mse\":"<<std::setprecision(12)<<e.error/std::max(e.norm,1e-30)<<'}';}report<<"]}\n";
    std::cerr<<"PACK complete path="<<output<<" bytes="<<offset+h.size()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
