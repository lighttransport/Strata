// Stream a converted routed-expert GGUF sidecar. Original files are read-only.
#include "strata/core/model.hpp"
#include "strata/artifact/fp8_experts.hpp"
#include "strata/core/runtime_memory.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <map>
#include <tuple>
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
std::string header(const std::vector<Entry>& entries,uint64_t source,bool calibrated,bool fp8) {
    std::ostringstream out(std::ios::binary);write<uint32_t>(out,0x46554747);write<uint32_t>(out,3);write<uint64_t>(out,entries.size());write<uint64_t>(out,entries.size()+3+fp8);
    string(out,"strata.expert_pack.version");write<uint32_t>(out,4);write<uint32_t>(out,1);
    string(out,"strata.expert_pack.source");write<uint32_t>(out,10);write<uint64_t>(out,source);
    string(out,"strata.expert_pack.calibrated");write<uint32_t>(out,7);write<uint8_t>(out,calibrated);
    // Converted from the FP8 release rather than requantized from the GGUF's own experts: any expert type.
    if(fp8){string(out,"strata.expert_pack.fp8");write<uint32_t>(out,7);write<uint8_t>(out,1);}
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
    if(argc<3)throw std::runtime_error("usage: strata-glm-q2-pack MODEL OUTPUT [--calibration=DIR | --uncalibrated] [--first-layer=3] [--last-layer=44] [--threads=8] [--down=q3|q2] [--reuse=PACK] [--all-layers] [--fp8=DIR --gate-type=T --down-type=T] [--resume]");
    // --fp8=DIR converts every layer in range from the FP8 release (strata/artifact/fp8_experts.hpp) instead of the
    // GGUF's own experts; --gate-type (gate and up) and --down-type pick any of q2_k, q3_k, q4_k, iq2_xxs, iq2_xs,
    // iq2_s, iq3_xxs, iq3_s, iq4_xs (default q2_k and the --down choice).
    // --down=q2 stores down projections as Q2_K (84 instead of 110 bytes per 256 weights). --reuse=PACK copies
    // every projection that an existing pack of the same source already holds in the requested type.
    // --all-layers also converts the layers the UD quantization kept in a higher format (IQ3_XXS gate/up, IQ4_XS down).
    std::string calibration,reuse,fp8_dir;bool uncalibrated=false,all_layers=false,resume=false;int first=3,last=44,threads=8;uint32_t down_type=11,gate_type=10;
    auto type_of=[](const std::string& name)->uint32_t {
        static const std::map<std::string,uint32_t> types={{"q2_k",10},{"q3_k",11},{"q4_k",12},{"iq2_xxs",16},{"iq2_xs",17},{"iq3_xxs",18},{"iq3_s",21},{"iq2_s",22},{"iq4_xs",23}};
        auto t=types.find(name);if(t==types.end())throw std::runtime_error("unknown expert type "+name);return t->second;
    };
    for(int i=3;i<argc;++i){std::string arg=argv[i];if(arg=="--all-layers")all_layers=true;else if(arg=="--resume")resume=true;else if(arg.starts_with("--fp8="))fp8_dir=arg.substr(6);else if(arg.starts_with("--gate-type="))gate_type=type_of(arg.substr(12));else if(arg.starts_with("--down-type="))down_type=type_of(arg.substr(12));else if(arg=="--down=q2")down_type=10;else if(arg=="--down=q3")down_type=11;else if(arg.starts_with("--reuse="))reuse=arg.substr(8);else if(arg.starts_with("--calibration="))calibration=arg.substr(14);else if(arg=="--uncalibrated")uncalibrated=true;else if(arg.starts_with("--first-layer="))first=std::stoi(arg.substr(14));else if(arg.starts_with("--last-layer="))last=std::stoi(arg.substr(13));else if(arg.starts_with("--threads="))threads=std::stoi(arg.substr(10));else throw std::runtime_error("unknown argument "+arg);}
    if(first<3||last>44||first>last||threads<1||threads>16||uncalibrated==!calibration.empty())throw std::runtime_error("invalid conversion settings; choose calibration or explicit uncalibrated, threads 1..16");
    strata::core::ModelArtifact model(argv[1]);const uint64_t experts=model.descriptor().experts;
    if(model.is_exl3()||model.descriptor().architecture!="glm5next"||model.descriptor().hidden!=4096||experts>288||experts%16)throw std::runtime_error("GLM model required");
    const std::filesystem::path output(argv[2]),partial=output.string()+".partial";
    if(std::filesystem::exists(output))throw std::runtime_error("refusing to overwrite existing pack");
    // --resume continues a .partial file whose .partial.progress lists the projections already written (same plan:
    // source, settings and layout); without it a partial file is refused as before.
    const std::filesystem::path progress_path=partial.string()+".progress";
    if(std::filesystem::exists(partial)&&!resume)throw std::runtime_error("refusing to overwrite existing partial file (--resume continues it)");
    if(fp8_dir.empty()&&(gate_type!=10||(down_type!=10&&down_type!=11)))throw std::runtime_error("--gate-type/--down-type other than Q2_K/Q3_K need --fp8=DIR");
    strata::artifact::fp8::Index fp8;if(!fp8_dir.empty())fp8=strata::artifact::fp8::scan(fp8_dir);
    std::vector<Entry> entries;uint64_t offset=0;
    for(int l=first;l<=last;++l)for(const char* part:{"gate","up","down"}) {
        const auto name="blk."+std::to_string(l)+".ffn_"+part+"_exps.weight";const auto& t=model.at(name);
        const bool down=std::string(part)=="down";const uint32_t type=down?down_type:gate_type;
        // IQ2_XS gate/up and IQ3_XXS down (UD-Q2_K_XL), or K-quant sources (Q4_K/Q5_K/Q6_K, e.g. a REAP-50 Q4_K_M).
        const bool kquant=t.tensor->type==12||t.tensor->type==13||t.tensor->type==14;
        const bool higher=all_layers&&t.tensor->type==(down?23u:18u);   // IQ4_XS down, IQ3_XXS gate/up
        const bool assembled_down=down&&down_type==10&&t.tensor->type==11&&model.has_expert_pack();
        if(fp8_dir.empty()&&t.tensor->type!=(down?18u:17u)&&!kquant&&!higher&&!assembled_down)continue;
        if(!fp8_dir.empty())for(uint64_t ex=0;ex<experts;++ex)if(!strata::artifact::fp8::has_expert(fp8,l,int(ex),std::string(part)+"_proj"))
            throw std::runtime_error("FP8 release lacks "+strata::artifact::fp8::expert_name(l,int(ex),std::string(part)+"_proj"));
        if(t.tensor->shape!=std::vector<uint64_t>{down?2048u:4096u,down?4096u:2048u,experts})throw std::runtime_error("unsupported expert geometry: "+name);
        const uint64_t bytes=ggml_row_size(ggml_type(type),t.tensor->shape[0])*t.tensor->shape[1]*t.tensor->shape[2];
        entries.push_back({name,&t,type,offset,bytes});offset+=(bytes+31)/32*32;
    }
    if(entries.empty())throw std::runtime_error("no eligible projections");
    if(std::filesystem::space(output.parent_path().empty()?".":output.parent_path()).available<offset+(uint64_t(2)<<30))throw std::runtime_error("insufficient disk capacity for pack plus 2 GiB reserve");
    const auto fingerprint=model.source_fingerprint();auto h=header(entries,fingerprint,!calibration.empty(),!fp8_dir.empty());
    // The plan: everything that fixes the bytes of the pack. A resumed run must have the same one.
    uint64_t plan=hash_bytes(14695981039346656037ULL,reinterpret_cast<const uint8_t*>(&fingerprint),8);
    {
        std::string key=calibration+"|"+fp8_dir+"|"+std::to_string(h.size());
        for(const auto& e:entries)key+="|"+e.name+":"+std::to_string(e.type)+":"+std::to_string(e.offset)+":"+std::to_string(e.bytes);
        plan=hash_bytes(plan,reinterpret_cast<const uint8_t*>(key.data()),key.size());
    }
    std::map<std::string,std::tuple<uint64_t,double,double>> completed;
    const bool resuming=resume&&std::filesystem::exists(partial);
    if(resuming) {
        std::ifstream pin(progress_path);std::string word;uint64_t their=0;
        if(!(pin>>word>>std::hex>>their>>std::dec)||word!="plan"||their!=plan)throw std::runtime_error("partial pack was made with another plan; remove it or match the options");
        std::string name;uint64_t hash;double err,nrm;
        while(pin>>name>>std::hex>>hash>>std::dec>>err>>nrm)completed[name]={hash,err,nrm};
        uint64_t need=h.size();
        for(const auto& e:entries)if(completed.count(e.name))need=std::max<uint64_t>(need,h.size()+e.offset+e.bytes);
        if(std::filesystem::file_size(partial)<need)throw std::runtime_error("partial pack is shorter than its progress record");
        std::cerr<<"PACK resuming: "<<completed.size()<<" of "<<entries.size()<<" projections already written\n";
    } else {
        std::ofstream pout(progress_path,std::ios::trunc);pout<<"plan "<<std::hex<<plan<<std::dec<<'\n';
    }
    std::ofstream out(partial,resuming?(std::ios::binary|std::ios::in|std::ios::out):(std::ios::binary|std::ios::trunc));
    out.exceptions(std::ios::failbit|std::ios::badbit);out.write(h.data(),h.size());
    // One line per finished projection, synced after its payload is.
    auto checkpoint=[&](const Entry& e) {
        std::ofstream pout(progress_path,std::ios::app);pout.exceptions(std::ios::failbit|std::ios::badbit);
        pout<<e.name<<' '<<std::hex<<e.hash<<std::dec<<' '<<std::setprecision(17)<<e.error<<' '<<e.norm<<'\n';pout.close();
#ifdef __linux__
        const int fd=::open(progress_path.c_str(),O_RDONLY);if(fd>=0){::fsync(fd);::close(fd);}
#endif
    };
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
        if(auto c=completed.find(e.name);c!=completed.end()) {
            std::tie(e.hash,e.error,e.norm)=c->second;std::cerr<<"PACK tensor="<<e.name<<" already written\n";continue;
        }
        out.seekp(std::streamoff(h.size()+e.offset));
        if(previous) {
            const auto& old=previous->at(e.name);
            if(old.tensor->type==e.type&&old.bytes==e.bytes&&old.file!=t.file) {
                out.write(reinterpret_cast<const char*>(old.data()),e.bytes);e.hash=hash_bytes(14695981039346656037ULL,old.data(),e.bytes);
                const auto* expected=old.file->get("strata.expert_pack.hash."+e.name);
                if(!expected||expected->u!=e.hash)throw std::runtime_error("reused projection fails its checksum: "+e.name);
                out.flush();old.file->discard_tensor_pages(*old.tensor,old.bytes);drop_written();checkpoint(e);
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
                if(fp8_dir.empty())ggml_get_type_traits(ggml_type(t.tensor->type))->to_float(t.data()+size_t(expert)*source_slice,weights.data(),weights.size());
                else {
                    const auto l=std::stoi(e.name.substr(4,e.name.find(".ffn_")-4));
                    const std::string part=e.name.find("_gate_")!=std::string::npos?"gate_proj":e.name.find("_up_")!=std::string::npos?"up_proj":"down_proj";
                    size_t r=0,c=0;auto w=strata::artifact::fp8::load_expert(fp8,l,expert,part,r,c);
                    if(r!=size_t(rows)||c!=size_t(cols))throw std::runtime_error("FP8 expert shape differs from the GGUF: "+e.name);
                    weights=std::move(w);
                }
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
        if(e.bytes%32){static const char zeros[32]={};out.write(zeros,32-e.bytes%32);}
        out.flush();t.file->discard_tensor_pages(*t.tensor,t.bytes);drop_written();checkpoint(e);
        std::cerr<<"PACK tensor="<<e.name<<" bytes="<<e.bytes<<" normalized_mse="<<std::setprecision(9)<<e.error/std::max(e.norm,1e-30)<<'\n';
    }
    if(model.source_fingerprint()!=fingerprint)throw std::runtime_error("source identity changed during conversion");
    h=header(entries,fingerprint,!calibration.empty(),!fp8_dir.empty());out.seekp(0);out.write(h.data(),h.size());out.close();
    std::filesystem::rename(partial,output);std::filesystem::remove(progress_path);
    std::ofstream report(output.string()+".quality.json");report.exceptions(std::ios::failbit|std::ios::badbit);report<<"{\"source_fingerprint\":\""<<fingerprint<<"\",\"calibrated\":"<<(!calibration.empty()?"true":"false")<<",\"projections\":[";
    for(size_t i=0;i<entries.size();++i){const auto& e=entries[i];if(i)report<<',';report<<"{\"name\":\""<<e.name<<"\",\"bytes\":"<<e.bytes<<",\"normalized_mse\":"<<std::setprecision(12)<<e.error/std::max(e.norm,1e-30)<<'}';}report<<"]}\n";
    std::cerr<<"PACK complete path="<<output<<" bytes="<<offset+h.size()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
