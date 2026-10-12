// Stream a converted routed-expert GGUF sidecar for MiMo-V2 (GGUF architecture "mimo2"), the MiMo counterpart
// of strata-glm-q2-pack. The original GGUF is read-only.
//
// Same sidecar format as the GLM pack: GGUF v3 holding only blk.L.ffn_{gate,up,down}_exps.weight, Q2_K gate/up
// and Q3_K (or Q2_K) down, with strata.expert_pack.{version,source,calibrated,hash.<tensor>} and the same source
// fingerprint (file size, mtime and header bytes of every shard). One extra key, strata.expert_pack.architecture
// = "mimo2", keeps a GLM loader from mistaking it for one of its own packs.
//
// MiMo differs from GLM in the layer range (layer 0 is dense, routed layers 1..47; the 3 MTP blocks 48..50 are
// dense) and the source type: the checkpoint ships its experts as MXFP4, which this tool dequantizes exactly
// (ggml MXFP4 -> f32) before requantizing. Calibration: --imatrix=FILE takes the per-expert input importance from
// llama.cpp's llama-imatrix run on the same model (it supports mimo2); --uncalibrated quantizes unweighted.
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
using strata::GgufFile;
using strata::GgufModel;
using strata::TensorInfo;

constexpr uint64_t kFnvBasis = 14695981039346656037ULL;
constexpr uint32_t kQ2K = 10, kQ3K = 11, kMXFP4 = 39;

template<class T> void write(std::ostream& out,T value){out.write(reinterpret_cast<const char*>(&value),sizeof value);}
void string(std::ostream& out,const std::string& value){write<uint64_t>(out,value.size());out.write(value.data(),value.size());}
uint64_t hash_bytes(uint64_t hash,const uint8_t* p,size_t n){for(size_t i=0;i<n;++i){hash^=p[i];hash*=1099511628211ULL;}return hash;}

struct Entry {std::string name;const TensorInfo* source;const GgufFile* file;uint32_t type;uint64_t offset,bytes,hash=0;double error=0,norm=0;};

std::string header(const std::vector<Entry>& entries,uint64_t source,bool calibrated) {
    std::ostringstream out(std::ios::binary);write<uint32_t>(out,0x46554747);write<uint32_t>(out,3);write<uint64_t>(out,entries.size());write<uint64_t>(out,entries.size()+4);
    string(out,"strata.expert_pack.version");write<uint32_t>(out,4);write<uint32_t>(out,1);
    string(out,"strata.expert_pack.source");write<uint32_t>(out,10);write<uint64_t>(out,source);
    string(out,"strata.expert_pack.calibrated");write<uint32_t>(out,7);write<uint8_t>(out,calibrated);
    string(out,"strata.expert_pack.architecture");write<uint32_t>(out,8);string(out,"mimo2");
    for(const auto& e:entries){string(out,"strata.expert_pack.hash."+e.name);write<uint32_t>(out,10);write<uint64_t>(out,e.hash);}
    for(const auto& e:entries){string(out,e.name);write<uint32_t>(out,3);for(auto d:e.source->shape)write<uint64_t>(out,d);write<uint32_t>(out,e.type);write<uint64_t>(out,e.offset);}
    auto result=out.str();result.resize((result.size()+31)/32*32,0);return result;
}

// Identical to strata::core::ModelArtifact::source_fingerprint(), so a mimo2-aware loader can bind the pack.
uint64_t source_fingerprint(const GgufModel& model) {
    uint64_t hash=kFnvBasis;
    auto add=[&](const void* data,size_t size){hash=hash_bytes(hash,static_cast<const uint8_t*>(data),size);};
    for(size_t i=0;i<model.size();++i) {
        const GgufFile& f=model.shard(i);
        const auto size=f.file_size();add(&size,sizeof size);
        const auto stamp=std::filesystem::last_write_time(f.path()).time_since_epoch().count();add(&stamp,sizeof stamp);
        std::ifstream input(f.path(),std::ios::binary);std::vector<char> h(std::min(f.data_start(),f.file_size()));
        if(!input.read(h.data(),h.size()))throw std::runtime_error("expert pack: cannot fingerprint source");
        add(h.data(),h.size());
    }
    return hash;
}

uint64_t meta_u(const GgufFile& g,const std::string& key) {
    const auto* v=g.get(key);if(!v)throw std::runtime_error("missing metadata "+key);return v->u;
}

void check_model(const GgufModel& model,uint64_t& experts) {
    const GgufFile& m=model.meta();
    const auto* arch=m.get("general.architecture");
    if(!arch||arch->s!="mimo2")throw std::runtime_error("MiMo-V2 GGUF (general.architecture = mimo2) required");
    experts=meta_u(m,"mimo2.expert_count");
    if(meta_u(m,"mimo2.embedding_length")!=4096||meta_u(m,"mimo2.expert_feed_forward_length")!=2048||experts==0||experts>384||experts%16)
        throw std::runtime_error("unsupported MiMo-V2 geometry (Flash: hidden 4096, expert rows 2048)");
}

// Bounds come from ggml's row size: the shared GGUF reader's block table has no MXFP4 entry, and it is left
// alone so no GLM code path changes.
uint64_t payload_bytes(const TensorInfo& t) {
    uint64_t rows=1;for(size_t i=1;i<t.shape.size();++i)rows*=t.shape[i];
    return t.shape.empty()?0:uint64_t(ggml_row_size(ggml_type(t.type),int64_t(t.shape[0])))*rows;
}
const TensorInfo& at(const GgufModel& model,const std::string& name,const GgufFile** file) {
    size_t shard=0;const auto* t=model.find(name,&shard);
    if(!t)throw std::runtime_error("missing tensor "+name);
    const GgufFile& g=model.shard(shard);const uint64_t bytes=payload_bytes(*t),payload=g.file_size()-g.data_start();
    if(bytes==0||t->offset>payload||bytes>payload-t->offset)throw std::runtime_error("out-of-bounds tensor "+name);
    *file=&g;return *t;
}

// llama.cpp `llama-imatrix` output (GGUF): `<weight>.in_sum2` F32 [columns, experts] (sum of squared inputs per
// expert) and `<weight>.counts` F32 [experts]. The per-expert mean squared input is the importance that
// ggml_quantize_chunk weights the reconstruction error with (the same quantity Strata's GLM calibration records).
// An expert the calibration text never reached gets the layer's mean over all routed tokens.
std::vector<float> imatrix_importance(const GgufFile& im,const std::string& weight,int columns,int experts) {
    const auto* sum=im.find(weight+".in_sum2");const auto* cnt=im.find(weight+".counts");
    if(!sum||!cnt)throw std::runtime_error("imatrix has no entry for "+weight);
    if(sum->type!=0||cnt->type!=0||sum->shape!=std::vector<uint64_t>{uint64_t(columns),uint64_t(experts)}||cnt->shape!=std::vector<uint64_t>{uint64_t(experts)})
        throw std::runtime_error("imatrix geometry mismatch for "+weight);
    const float* s=reinterpret_cast<const float*>(im.tensor_data(*sum));const float* c=reinterpret_cast<const float*>(im.tensor_data(*cnt));
    std::vector<double> pooled(columns,0.);double total=0;
    for(int e=0;e<experts;++e){total+=c[e];for(int k=0;k<columns;++k)pooled[k]+=s[size_t(e)*columns+k];}
    std::vector<float> out(size_t(columns)*experts);
    for(int e=0;e<experts;++e)for(int k=0;k<columns;++k) {
        const double v=c[e]>0?s[size_t(e)*columns+k]/c[e]:(total>0?pooled[k]/total:1.);
        if(!std::isfinite(v)||v<0)throw std::runtime_error("nonfinite or negative imatrix value for "+weight);
        out[size_t(e)*columns+k]=float(std::max(v,1e-12));
    }
    return out;
}

int verify(const std::string& model_path,const std::string& pack_path) {
    const auto model=GgufModel::open(model_path);uint64_t experts=0;check_model(model,experts);
    GgufFile pack(pack_path);
    const auto* version=pack.get("strata.expert_pack.version");const auto* source=pack.get("strata.expert_pack.source");
    const auto* arch=pack.get("strata.expert_pack.architecture");
    if(!version||version->u!=1||!source||source->u!=source_fingerprint(model)||!arch||arch->s!="mimo2")
        throw std::runtime_error("expert pack: version, architecture or source fingerprint mismatch");
    size_t checked=0;
    for(const auto& t:pack.tensors()) {
        const GgufFile* f=nullptr;const auto& src=at(model,t.name,&f);
        const bool down=t.name.find(".ffn_down_exps.")!=std::string::npos;
        if(t.name.find("_exps.weight")==std::string::npos||t.shape!=src.shape||(t.type!=kQ2K&&!(down&&t.type==kQ3K)))
            throw std::runtime_error("expert pack: unexpected tensor "+t.name);
        const uint64_t bytes=t.elements()/256*(t.type==kQ2K?84:110);
        if(t.offset>pack.file_size()-pack.data_start()||bytes>pack.file_size()-pack.data_start()-t.offset)
            throw std::runtime_error("expert pack: payload out of bounds "+t.name);
        const auto* expected=pack.get("strata.expert_pack.hash."+t.name);
        if(!expected||hash_bytes(kFnvBasis,pack.tensor_data(t),bytes)!=expected->u)throw std::runtime_error("payload checksum mismatch: "+t.name);
        pack.discard_tensor_pages(t,bytes);++checked;
    }
    std::cout<<"PACK verified architecture=mimo2 experts="<<experts<<" projections="<<checked<<'\n';
    return 0;
}
}  // namespace

int main(int argc,char** argv){try {
    if(argc>=2&&std::string(argv[1])=="--verify") {
        if(argc!=4)throw std::runtime_error("usage: strata-mimo-q2-pack --verify MODEL PACK");
        return verify(argv[2],argv[3]);
    }
    if(argc<3)throw std::runtime_error("usage: strata-mimo-q2-pack MODEL OUTPUT (--imatrix=FILE | --uncalibrated) [--first-layer=1] [--last-layer=47] [--threads=16] [--down=q3|q2]");
    bool uncalibrated=false;int first=1,last=47,threads=16;uint32_t down_type=kQ3K;std::string imatrix_path;
    for(int i=3;i<argc;++i){std::string arg=argv[i];if(arg=="--down=q2")down_type=kQ2K;else if(arg=="--down=q3")down_type=kQ3K;else if(arg=="--uncalibrated")uncalibrated=true;else if(arg.starts_with("--imatrix="))imatrix_path=arg.substr(10);else if(arg.starts_with("--first-layer="))first=std::stoi(arg.substr(14));else if(arg.starts_with("--last-layer="))last=std::stoi(arg.substr(13));else if(arg.starts_with("--threads="))threads=std::stoi(arg.substr(10));else throw std::runtime_error("unknown argument "+arg);}
    // Calibration must be explicit: an imatrix from llama-imatrix on the same model, or --uncalibrated.
    if(uncalibrated==!imatrix_path.empty()||first<1||first>last||threads<1||threads>32)throw std::runtime_error("invalid settings: choose --imatrix=FILE or --uncalibrated; layers >= 1, threads 1..32");
    std::unique_ptr<GgufFile> imatrix;
    if(!imatrix_path.empty()) {
        imatrix=std::make_unique<GgufFile>(imatrix_path);
        const auto* type=imatrix->get("general.type");
        if(!type||type->s!="imatrix")throw std::runtime_error("not a llama-imatrix GGUF: "+imatrix_path);
    }
    const auto model=GgufModel::open(argv[1]);uint64_t experts=0;check_model(model,experts);
    const uint64_t blocks=meta_u(model.meta(),"mimo2.block_count");
    const std::filesystem::path output(argv[2]),partial=output.string()+".partial";
    if(std::filesystem::exists(output)||std::filesystem::exists(partial))throw std::runtime_error("refusing to overwrite existing pack or partial file");
    std::vector<Entry> entries;uint64_t offset=0;
    for(int l=first;l<=last&&uint64_t(l)<blocks;++l)for(const char* part:{"gate","up","down"}) {
        const auto name="blk."+std::to_string(l)+".ffn_"+part+"_exps.weight";
        size_t shard=0;const auto* t=model.find(name,&shard);
        if(!t)continue;  // dense layer (0) or MTP block
        const GgufFile* f=nullptr;at(model,name,&f);
        const bool down=std::string(part)=="down";const uint32_t type=down?down_type:kQ2K;
        // MXFP4 (the checkpoint's own format) or a K-quant source.
        if(t->type!=kMXFP4&&t->type!=12&&t->type!=13&&t->type!=14)throw std::runtime_error("unsupported source type "+std::to_string(t->type)+" for "+name);
        if(t->shape!=std::vector<uint64_t>{down?2048u:4096u,down?4096u:2048u,experts})throw std::runtime_error("unsupported expert geometry: "+name);
        const uint64_t bytes=t->elements()/256*(type==kQ2K?84:110);entries.push_back({name,t,f,type,offset,bytes});offset+=bytes;
    }
    if(entries.empty())throw std::runtime_error("no routed expert projections in the layer range");
    if(std::filesystem::space(output.parent_path().empty()?".":output.parent_path()).available<offset+(uint64_t(2)<<30))throw std::runtime_error("insufficient disk capacity for pack plus 2 GiB reserve");
    const auto fingerprint=source_fingerprint(model);auto h=header(entries,fingerprint,!imatrix_path.empty());
    std::cerr<<"PACK plan projections="<<entries.size()<<" bytes="<<offset+h.size()<<" source_fingerprint="<<fingerprint<<'\n';
    std::ofstream out(partial,std::ios::binary|std::ios::trunc);out.exceptions(std::ios::failbit|std::ios::badbit);out.write(h.data(),h.size());
    ggml_cpu_init();
    auto drop_written=[&] {
#ifdef __linux__
        const int fd=::open(partial.c_str(),O_RDONLY);
        if(fd>=0){::fdatasync(fd);::posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);::close(fd);}
#endif
    };
    const auto t0=std::chrono::steady_clock::now();
    for(size_t ei=0;ei<entries.size();++ei) {
        auto& e=entries[ei];
        const auto& t=*e.source;const int cols=t.shape[0],rows=t.shape[1],nexp=t.shape[2];
        const size_t slice=e.bytes/nexp,source_slice=ggml_row_size(ggml_type(t.type),cols)*rows;
        e.hash=kFnvBasis;
        const std::vector<float> importance=imatrix?imatrix_importance(*imatrix,e.name,cols,nexp):std::vector<float>{};
        for(int base=0;base<nexp;base+=threads) {
            const int count=std::min(threads,nexp-base);std::vector<std::vector<uint8_t>> packed(count,std::vector<uint8_t>(slice));
            std::vector<std::jthread> workers;std::vector<std::exception_ptr> errors(count);std::vector<double> err(count),norm(count);
            for(int j=0;j<count;++j)workers.emplace_back([&,j] {try {
                const int expert=base+j;std::vector<float> weights(size_t(cols)*rows),decoded(cols);
                ggml_get_type_traits(ggml_type(t.type))->to_float(e.file->tensor_data(t)+size_t(expert)*source_slice,weights.data(),weights.size());
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
            for(int j=0;j<count;++j){out.write(reinterpret_cast<const char*>(packed[j].data()),slice);e.hash=hash_bytes(e.hash,packed[j].data(),slice);e.error+=err[j];e.norm+=norm[j];}
        }
        out.flush();e.file->discard_tensor_pages(t,source_slice*nexp);drop_written();
        const double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        std::cerr<<"PACK tensor="<<e.name<<" bytes="<<e.bytes<<" normalized_mse="<<std::setprecision(9)<<e.error/std::max(e.norm,1e-30)
                 <<" done="<<ei+1<<"/"<<entries.size()<<" elapsed_s="<<std::setprecision(5)<<s<<'\n';
    }
    if(source_fingerprint(model)!=fingerprint)throw std::runtime_error("source identity changed during conversion");
    h=header(entries,fingerprint,!imatrix_path.empty());out.seekp(0);out.write(h.data(),h.size());out.close();
    std::filesystem::rename(partial,output);
    std::ofstream report(output.string()+".quality.json");report.exceptions(std::ios::failbit|std::ios::badbit);report<<"{\"architecture\":\"mimo2\",\"source_fingerprint\":\""<<fingerprint<<"\",\"calibrated\":"<<(imatrix?"true":"false")<<",\"projections\":[";
    for(size_t i=0;i<entries.size();++i){const auto& e=entries[i];if(i)report<<',';report<<"{\"name\":\""<<e.name<<"\",\"bytes\":"<<e.bytes<<",\"normalized_mse\":"<<std::setprecision(12)<<e.error/std::max(e.norm,1e-30)<<'}';}report<<"]}\n";
    std::cerr<<"PACK complete path="<<output<<" bytes="<<offset+h.size()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
