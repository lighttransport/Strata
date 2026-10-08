#pragma once
#include "strata/core/model.hpp"
#include "strata/artifact/json.hpp"
#include "strata/kernels/cpu/expert_observer.hpp"
#include "strata/kernels/cpu/expert_block_mask.hpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace strata::artifact {
inline uint64_t block_hash(uint64_t hash,const void* data,size_t bytes) {
    const auto* p=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<bytes;++i){hash^=p[i];hash*=1099511628211ULL;}
    return hash;
}
inline std::string block_identity(uint64_t value) {
    std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<value;return out.str();
}
struct ExpertBlockNorms {
    uint64_t identity=0;
    int layers=0,experts=0,hidden=0;
    std::vector<uint8_t> eligible;
    std::vector<float> values;
    const float* at(int layer,int expert)const {
        if(layer<0||layer>=layers||expert<0||expert>=experts)throw std::invalid_argument("block norm index outside model");
        return values.data()+(size_t(layer)*experts+expert)*2048;
    }
    uint64_t payload_hash()const {
        return block_hash(block_hash(14695981039346656037ULL,eligible.data(),eligible.size()),values.data(),values.size()*sizeof(float));
    }
    void validate()const {
        if(layers<1||layers>64||experts<1||experts>512||hidden<1||hidden>16384||
           eligible.size()!=size_t(layers)||values.size()!=size_t(layers)*experts*2048)
            throw std::runtime_error("invalid block norm geometry");
        for(auto flag:eligible)if(flag>1)throw std::runtime_error("invalid block norm eligibility");
        for(float v:values)if(!std::isfinite(v)||v<0)throw std::runtime_error("invalid block norm value");
    }
    void save(const std::filesystem::path& path)const {
        validate();const auto partial=path.string()+".partial";
        if(std::filesystem::exists(path)||std::filesystem::exists(partial))throw std::runtime_error("refusing to overwrite block norms");
        const uint32_t header[]={0x314e4247,1,uint32_t(layers),uint32_t(experts),uint32_t(hidden),2048};
        const uint64_t sum=payload_hash();std::ofstream out(partial,std::ios::binary);out.exceptions(std::ios::failbit|std::ios::badbit);
        out.write(reinterpret_cast<const char*>(header),sizeof header);out.write(reinterpret_cast<const char*>(&identity),8);
        out.write(reinterpret_cast<const char*>(&sum),8);out.write(reinterpret_cast<const char*>(eligible.data()),eligible.size());
        out.write(reinterpret_cast<const char*>(values.data()),values.size()*4);out.close();std::filesystem::rename(partial,path);
    }
    static ExpertBlockNorms load(const std::filesystem::path& path,uint64_t expected,int layers,int experts,int hidden) {
        std::ifstream in(path,std::ios::binary);uint32_t h[6]{};uint64_t identity=0,sum=0;
        if(!in.read(reinterpret_cast<char*>(h),sizeof h)||!in.read(reinterpret_cast<char*>(&identity),8)||!in.read(reinterpret_cast<char*>(&sum),8)||
           h[0]!=0x314e4247||h[1]!=1||h[2]!=uint32_t(layers)||h[3]!=uint32_t(experts)||h[4]!=uint32_t(hidden)||h[5]!=2048||identity!=expected)
            throw std::runtime_error("block norms version, geometry or source identity mismatch");
        ExpertBlockNorms result{identity,layers,experts,hidden,{},{}};
        result.eligible.resize(layers);result.values.resize(size_t(layers)*experts*2048);
        if(!in.read(reinterpret_cast<char*>(result.eligible.data()),result.eligible.size())||
           !in.read(reinterpret_cast<char*>(result.values.data()),result.values.size()*4)||in.peek()!=std::ifstream::traits_type::eof())
            throw std::runtime_error("truncated or oversized block norms");
        result.validate();if(result.payload_hash()!=sum)throw std::runtime_error("block norms checksum mismatch");return result;
    }
};

// Full gate/up and full down still run. This is a quality screen and data collector,
// deliberately restricted by the CLI to ordinary, all-CPU expert evaluation.
class ExpertBlockScreen final:public kernels::cpu::ExpertHiddenTransform {
    struct Route {int position=0;float weight=0;bool seen=false;kernels::cpu::ExpertBlockMask mask;};
    struct Stats {uint64_t rows=0,kept=0,full=0,union_blocks=0,jobs=0;double energy=0,removed=0,bytes=0,ideal_bytes=0;};
    ExpertBlockNorms norms_;
    std::vector<int> keep_;
    std::vector<size_t> bytes_;
    std::vector<Stats> stats_;
    std::map<int,std::vector<Route>> pending_;
    int layer_=-1,sequence_=0;
    std::filesystem::path capture_;
    std::map<int,std::ofstream> files_;
    std::ofstream& file(int layer) {
        auto [it,inserted]=files_.try_emplace(layer);
        if(inserted){it->second.open(capture_/ ("layer-"+std::to_string(layer)+".bin"),std::ios::binary);it->second.exceptions(std::ios::failbit|std::ios::badbit);}
        return it->second;
    }
    void finish_layer() {
        if(layer_<0)return;
        for(const auto& [expert,routes]:pending_) {
            (void)expert;kernels::cpu::ExpertBlockMask joined;
            for(const auto& route:routes){if(!route.seen)throw std::runtime_error("block screen missed a routed expert");for(int w=0;w<2;++w)joined.words[w]|=route.mask.words[w];}
            int blocks=0;for(int b=0;b<128;++b)blocks+=joined.contains(b);
            auto& s=stats_[layer_];++s.jobs;s.union_blocks+=blocks;s.bytes+=bytes_[layer_];s.ideal_bytes+=double(bytes_[layer_])*blocks/128;
        }
        pending_.clear();layer_=-1;
    }
public:
    ExpertBlockScreen(const core::ModelArtifact& model,const std::filesystem::path& profile,const std::filesystem::path& capture={}) {
        const auto& m=model.descriptor();const auto config=read_json(profile);
        if(config.at("version").integer()!=1||config.at("mode").string()!="oracle"||config.at("block_channels").integer()!=16||
           m.architecture!="glm5next"||m.hidden!=4096||m.layers.size()!=45||!model.has_expert_pack())
            throw std::invalid_argument("block screen requires oracle v1, 16 channels and a GLM Q23 pack");
        const auto identity=model.source_fingerprint();
        if(config.at("source_identity").string()!=block_identity(identity))throw std::runtime_error("block profile source identity mismatch");
        auto norm_path=std::filesystem::path(config.at("norms").string());if(norm_path.is_relative())norm_path=profile.parent_path()/norm_path;
        norms_=ExpertBlockNorms::load(norm_path,identity,int(m.layers.size()),int(m.experts),int(m.hidden));
        const auto keep=config.at("retained_blocks").integer();if(keep<1||keep>128)throw std::invalid_argument("retained blocks outside 1..128");
        keep_.assign(m.layers.size(),int(keep));bytes_.resize(m.layers.size());stats_.resize(m.layers.size());
        if(const auto* layers=config.find("layers")) {
            if(layers->kind!=Json::Object)throw std::invalid_argument("block profile layers must be an object");
            for(const auto& [key,value]:layers->object){size_t used=0;int layer=std::stoi(key,&used);auto count=value.integer();
                if(used!=key.size()||layer<3||layer>44||count<1||count>128)throw std::invalid_argument("invalid block layer budget");
                keep_[layer]=int(count);}
        }
        for(size_t l=3;l<m.layers.size();++l) {
            const auto p="blk."+std::to_string(l)+".ffn_";const auto& g=model.at(p+"gate_exps.weight");const auto& u=model.at(p+"up_exps.weight");const auto& d=model.at(p+"down_exps.weight");
            const bool eligible=(g.tensor->type==10||g.tensor->type==11)&&u.tensor->type==g.tensor->type&&(d.tensor->type==10||d.tensor->type==11);
            if(norms_.eligible[l]!=eligible||g.tensor->shape!=std::vector<uint64_t>{4096,2048,uint64_t(m.experts)}||u.tensor->shape!=g.tensor->shape||d.tensor->shape!=std::vector<uint64_t>{2048,4096,uint64_t(m.experts)})
                throw std::runtime_error("block norm projection mismatch");
            if(!eligible)keep_[l]=128;
            bytes_[l]=(g.bytes+u.bytes+d.bytes)/m.experts;
        }
        capture_=capture;
        if(!capture_.empty()) {
            if(std::filesystem::exists(capture_))throw std::runtime_error("refusing to overwrite block capture directory");
            std::filesystem::create_directories(capture_);std::ofstream manifest(capture_/"manifest.json");
            manifest<<"{\"version\":1,\"source_identity\":\""<<block_identity(identity)<<"\",\"block_channels\":16,\"hidden\":4096,\"experts\":"<<m.experts<<",\"targets\":\"full pre-mask squared hidden times down-column norm; unweighted by route\"}\n";
            if(!manifest)throw std::runtime_error("block capture manifest write failed");
        }
    }
    void begin_sequence(){finish_layer();++sequence_;}
    void uniform_budget(int keep) {
        if(keep<1||keep>128||!capture_.empty())throw std::invalid_argument("budget sweeps require 1..128 blocks and capture disabled");
        flush();for(size_t l=0;l<keep_.size();++l)keep_[l]=norms_.eligible[l]?keep:128;
        std::fill(stats_.begin(),stats_.end(),Stats{});
    }
    void layer_budget(int layer,int keep) {
        if(layer<3||layer>=norms_.layers)throw std::invalid_argument("invalid block sensitivity layer");
        uniform_budget(128);if(norms_.eligible[layer])keep_[layer]=keep;
    }
    void input(int layer,int position,int nt,int hidden,int topk,const int* ids,const float* weights,const float* values) {
        finish_layer();if(layer<3||layer>=norms_.layers||nt<1||nt>8||hidden!=norms_.hidden||topk<1||topk>32||!ids||!weights||!values)
            throw std::invalid_argument("block capture input geometry");
        layer_=layer;
        for(int t=0;t<nt;++t){std::set<int> unique;for(int k=0;k<topk;++k){int e=ids[t*topk+k];float w=weights[t*topk+k];
            if(e<0||e>=norms_.experts||!unique.insert(e).second||!std::isfinite(w)||w<=0)throw std::runtime_error("invalid block screen route");
            pending_[e].push_back(Route{position+t,w,false,{}});}}
        if(!capture_.empty()) {
            auto& out=file(layer);const uint32_t header[]={0x31434247,1,uint32_t(sequence_),uint32_t(layer),uint32_t(position),uint32_t(nt),uint32_t(hidden),uint32_t(topk)};
            out.write(reinterpret_cast<const char*>(header),sizeof header);out.write(reinterpret_cast<const char*>(ids),nt*topk*4);
            out.write(reinterpret_cast<const char*>(weights),nt*topk*4);out.write(reinterpret_cast<const char*>(values),nt*hidden*4);
        }
    }
    void apply(int layer,int expert,int token,float* hidden,int count)override {
        if(layer!=layer_||!pending_.contains(expert)||token<0||token>=int(pending_.at(expert).size()))throw std::runtime_error("block screen hidden record without matching input");
        auto& route=pending_.at(expert)[token];if(route.seen)throw std::runtime_error("duplicate block screen hidden record");route.seen=true;
        const auto scores=kernels::cpu::expert_block_energy(hidden,norms_.at(layer,expert),count);
        route.mask=kernels::cpu::expert_block_topk(scores,keep_[layer]);auto& stats=stats_[layer];++stats.rows;stats.kept+=keep_[layer];stats.full+=128;
        for(int b=0;b<128;++b){const double energy=scores[b]*double(route.weight)*route.weight;stats.energy+=energy;if(!route.mask.contains(b))stats.removed+=energy;}
        if(!capture_.empty()) {
            auto& out=file(layer);const uint32_t header[]={0x31434247,2,uint32_t(sequence_),uint32_t(layer),uint32_t(route.position),uint32_t(expert),128,0};
            std::array<float,128> target{};for(int b=0;b<128;++b){target[b]=float(scores[b]);if(!std::isfinite(target[b]))throw std::runtime_error("block capture energy exceeds float range");}
            out.write(reinterpret_cast<const char*>(header),sizeof header);out.write(reinterpret_cast<const char*>(target.data()),sizeof target);
        }
        kernels::cpu::expert_block_zero(hidden,count,route.mask);
    }
    void flush(){finish_layer();for(auto& [layer,out]:files_){(void)layer;out.flush();}}
    void report(std::ostream& out) {
        flush();uint64_t jobs=0,rows=0;double bytes=0,ideal=0;
        for(size_t l=3;l<stats_.size();++l){const auto& s=stats_[l];if(!s.rows)continue;jobs+=s.jobs;rows+=s.rows;bytes+=s.bytes;ideal+=s.ideal_bytes;
            out<<"BLOCK_SCREEN layer="<<l<<" rows="<<s.rows<<" retained_blocks="<<keep_[l]<<" weighted_energy_removed="<<(s.energy?s.removed/s.energy:0)<<" union_blocks="<<s.union_blocks<<" jobs="<<s.jobs
               <<" full_expert_bytes="<<std::setprecision(15)<<s.bytes<<" ideal_union_bytes="<<s.ideal_bytes<<'\n';}
        out<<"BLOCK_SCREEN_TOTAL rows="<<rows<<" jobs="<<jobs<<" full_expert_bytes="<<std::setprecision(15)<<bytes<<" ideal_union_bytes="<<ideal<<" ideal_byte_reduction="<<(bytes?1-ideal/bytes:0)<<" diagnostic_only=1\n";
    }
};
}
