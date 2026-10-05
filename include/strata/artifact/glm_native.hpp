#pragma once
#include "strata/artifact/glm_exl3.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <functional>
#include <cstring>
namespace strata::artifact {
static_assert(std::endian::native == std::endian::little, "Native safetensors requires a little-endian host");
// Canonical engine names over native tensors. Small scalar/norm tensors are promoted;
// matrices preserve source BF16. Materialization is lazy and owned by the artifact.
class GlmNative {
public:
    GlmExl3 model;
    struct Binding {
        TensorInfo info; const GlmExl3* model;
        std::string source;bool promote=false;int kv_part=-1;
        mutable std::vector<uint8_t> storage;
        const uint8_t* data()const {
            if(!storage.empty())return storage.data();
            const auto& t=model->tensors.at(source);std::vector<uint8_t> raw(t.bytes);
            model->tensors.read(t,0,raw.data(),raw.size());
            if(kv_part>=0) {
                storage.resize(64*256*512*2);
                for(size_t h=0;h<64;++h)for(size_t r=0;r<256;++r)for(size_t c=0;c<512;++c) {
                    size_t src=((h*512+kv_part*256+r)*512+c)*2;
                    size_t dst=(h*256*512+(kv_part? r*512+c:c*256+r))*2;
                    storage[dst]=raw[src];storage[dst+1]=raw[src+1];
                }
            }else if(promote && t.dtype=="BF16") {
                storage.resize(raw.size()*2);
                for(size_t i=0;i<raw.size()/2;++i) { uint32_t v=(uint32_t(raw[i*2])|(uint32_t(raw[i*2+1])<<8))<<16;std::memcpy(storage.data()+i*4,&v,4); }
            }else storage=std::move(raw);
            if(info.name.ends_with(".ssm_a")) {
                for(size_t i=0;i<storage.size()/4;++i) {float value;std::memcpy(&value,storage.data()+4*i,4);value=-std::exp(value);std::memcpy(storage.data()+4*i,&value,4);}
            }
            return storage.data();
        }
    };
    std::map<std::string,Binding> bindings;
    explicit GlmNative(const std::filesystem::path& root):model(root) {
        add("token_embd.weight","model.language_model.embed_tokens.weight");
        add("output.weight","lm_head.weight");add("output_norm.weight","model.language_model.norm.weight",true);
        const std::map<std::string,std::string> names={
            {"input_layernorm.weight","attn_norm.weight"},{"post_attention_layernorm.weight","ffn_norm.weight"},
            {"hc_attn_fn","hc_attn_fn.weight"},{"hc_attn_base","hc_attn_base.weight"},{"hc_attn_scale","hc_attn_scale.weight"},
            {"hc_ffn_fn","hc_ffn_fn.weight"},{"hc_ffn_base","hc_ffn_base.weight"},{"hc_ffn_scale","hc_ffn_scale.weight"},
            {"mlp.gate.weight","ffn_gate_inp.weight"},{"mlp.gate.e_score_correction_bias","exp_probs_b.bias"},
            {"self_attn.A_log","ssm_a"},{"self_attn.dt_bias","ssm_dt.bias"},{"self_attn.b_proj.weight","ssm_beta.weight"},
            {"self_attn.o_norm.weight","ssm_norm.weight"},{"self_attn.o_proj.weight","attn_output.weight"},
            {"self_attn.q_a_proj.weight","attn_q_a.weight"},{"self_attn.q_a_layernorm.weight","attn_q_a_norm.weight"},
            {"self_attn.q_b_proj.weight","attn_q_b.weight"},{"self_attn.kv_a_proj_with_mqa.weight","attn_kv_a_mqa.weight"},
            {"self_attn.kv_a_layernorm.weight","attn_kv_a_norm.weight"},
            {"self_attn.indexer.wq_b.weight","indexer.attn_q_b.weight"},{"self_attn.indexer.wk.weight","indexer.attn_k.weight"},
            {"self_attn.indexer.weights_proj.weight","indexer.proj.weight"},{"self_attn.indexer.k_norm.weight","indexer.k_norm.weight"},
            {"self_attn.indexer.k_norm.bias","indexer.k_norm.bias"},{"self_attn.indexer.index_kpool_compress_gate","indexer_compressor_gate.weight"},
            {"self_attn.indexer.index_kpool_compress_ape","indexer_compressor_ape.weight"}};
        for(unsigned l=0;l<45;++l) {
            auto src="model.language_model.layers."+std::to_string(l)+".",dst="blk."+std::to_string(l)+".";
            for(const auto& [from,to]:names)if(model.tensors.tensors().contains(src+from)) {
                bool scalar=to.find("norm.")!=std::string::npos||to.find("_base.")!=std::string::npos||to.find("_scale.")!=std::string::npos||to=="ssm_a"||to=="ssm_dt.bias"||to=="exp_probs_b.bias"||to=="indexer_compressor_ape.weight";
                add(dst+to,src+from,scalar);
            }
            for(const auto* p:{"gate","up","down"})add(dst+"ffn_"+p+(l<3?"":"_shexp")+".weight",src+"mlp."+(l<3?"":"shared_experts.")+p+"_proj.weight");
            if(l%4!=3) {
                for(const auto* p:{"q","k","v"}) {
                    add(dst+"attn_"+p+".weight",src+"self_attn."+p+"_proj.weight");
                    add(dst+"ssm_conv1d_"+p+".weight",src+"self_attn."+p+"_conv1d.weight",true);
                }
                for(const auto* p:{"f","g"})for(const auto* q:{"a","b"})add(dst+"ssm_"+p+"_"+q+".weight",src+"self_attn."+p+"_"+q+"_proj.weight");
            }else for(int part=0;part<2;++part) {
                model.tensors.expect(src+"self_attn.kv_b_proj.weight","BF16",{32768,512});
                auto name=dst+(part?"attn_v_b.weight":"attn_k_b.weight");add(name,src+"self_attn.kv_b_proj.weight");
                auto& b=bindings.at(name);b.kv_part=part;b.info.shape=part?std::vector<uint64_t>{512,256,64}:std::vector<uint64_t>{256,512,64};
            }
        }
        validate();
    }
    void validate()const {
        auto shape=[&](const std::string& name,std::initializer_list<uint64_t> dims){const auto& t=bindings.at(name).info;if(t.shape!=std::vector<uint64_t>(dims))throw std::runtime_error("GLM native: incompatible shape "+name);};
        shape("token_embd.weight",{4096,154880});shape("output.weight",{4096,154880});shape("output_norm.weight",{4096});
        for(unsigned l=0;l<45;++l) {
            auto p="blk."+std::to_string(l)+".";
            for(const auto* part:{"attn","ffn"}) {
                shape(p+part+"_norm.weight",{4096});shape(p+"hc_"+part+"_fn.weight",{16384,24});
                shape(p+"hc_"+part+"_base.weight",{24});shape(p+"hc_"+part+"_scale.weight",{3});
            }
            uint64_t ff=l<3?12288:2048;std::string suffix=l<3?"":"_shexp";
            for(const auto* part:{"gate","up"})shape(p+"ffn_"+part+suffix+".weight",{4096,ff});
            shape(p+"ffn_down"+suffix+".weight",{ff,4096});
            if(l>=3){shape(p+"ffn_gate_inp.weight",{4096,288});shape(p+"exp_probs_b.bias",{288});}
            if(l%4!=3) {
                for(const auto* part:{"q","k","v"}){shape(p+"attn_"+part+".weight",{4096,8192});shape(p+"ssm_conv1d_"+part+".weight",{4,1,8192});}
                shape(p+"attn_output.weight",{8192,4096});shape(p+"ssm_a",{64});shape(p+"ssm_dt.bias",{8192});shape(p+"ssm_beta.weight",{4096,64});shape(p+"ssm_norm.weight",{128});
                for(const auto* part:{"f","g"}){shape(p+"ssm_"+part+"_a.weight",{4096,128});shape(p+"ssm_"+part+"_b.weight",{128,8192});}
            }else {
                shape(p+"attn_q_a.weight",{4096,1536});shape(p+"attn_q_a_norm.weight",{1536});shape(p+"attn_q_b.weight",{1536,16384});
                shape(p+"attn_kv_a_mqa.weight",{4096,512});shape(p+"attn_kv_a_norm.weight",{512});shape(p+"attn_output.weight",{16384,4096});
                shape(p+"indexer.attn_q_b.weight",{1536,4096});shape(p+"indexer.attn_k.weight",{4096,128});shape(p+"indexer.proj.weight",{4096,32});
                shape(p+"indexer.k_norm.weight",{128});shape(p+"indexer.k_norm.bias",{128});shape(p+"indexer_compressor_gate.weight",{4096,128});shape(p+"indexer_compressor_ape.weight",{128,4});
            }
        }
    }
    void add(const std::string& name,const std::string& source,bool promote=false) {
        const auto& t=model.tensors.at(source);Binding b;b.model=&model;b.source=source;b.promote=promote;
        b.info.name=name;b.info.type=promote||t.dtype=="F32"?0:30;b.info.shape.assign(t.shape.rbegin(),t.shape.rend());
        bindings.emplace(name,std::move(b));
    }
};
}
