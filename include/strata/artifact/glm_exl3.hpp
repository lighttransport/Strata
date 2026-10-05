#pragma once
#include "strata/artifact/safetensors.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
namespace strata::artifact {
// This reader accepts only the inspected K2/MCG pack contract.
class GlmExl3 {
public:
    Safetensors tensors;
    Json config;
    std::vector<bool> token_ids;
    uint64_t experts_bytes=0, fixed_bytes=0, draft_bytes=0, vision_bytes=0;
    explicit GlmExl3(const std::filesystem::path& root):tensors(root),config(read_json((root/"config.json").string())) {
        const auto& q=config.at("quantization_config");
        if(q.at("quant_method").string()!="exl3" || q.at("bits").integer()!=2 || q.at("codebook").string()!="mcg" ||
           q.at("scope").string()!="glm53_routed_experts_only")throw std::runtime_error("GLM EXL3: requires routed-only K2 MCG pack");
        const auto& c=config.at("text_config");
        if(c.at("vocab_size").integer()!=154880||c.at("hidden_size").integer()!=4096||c.at("moe_intermediate_size").integer()!=2048||
           c.at("num_hidden_layers").integer()!=45||c.at("n_routed_experts").integer()!=288||
           c.at("num_experts_per_tok").integer()!=8)throw std::runtime_error("GLM EXL3: unsupported model geometry");
        token_ids.resize(c.at("vocab_size").integer(),false);
        const auto tokenizer=read_json((root/"tokenizer.json").string());
        auto add_token=[&](uint64_t id){if(id>=token_ids.size())throw std::runtime_error("GLM EXL3: tokenizer ID exceeds vocabulary");token_ids[id]=true;};
        for(const auto& [token,id]:tokenizer.at("model").at("vocab").object)add_token(id.integer());
        for(const auto& token:tokenizer.at("added_tokens").array)add_token(token.at("id").integer());
        const auto& linear=c.at("linear_attn_config");
        if(c.at("layer_types").array.size()!=45 || c.at("first_k_dense_replace").integer()!=3 ||
           c.at("qk_rope_head_dim").integer()!=0 || c.at("qk_nope_head_dim").integer()!=256 || c.at("v_head_dim").integer()!=256 ||
           c.at("num_attention_heads").integer()!=64 || c.at("kv_lora_rank").integer()!=512 || c.at("q_lora_rank").integer()!=1536 ||
           c.at("hc_mult").integer()!=4 || c.at("n_group").integer()!=1 || c.at("topk_group").integer()!=1 || c.at("n_shared_experts").integer()!=1 ||
           c.at("scoring_func").string()!="sigmoid" || !c.at("norm_topk_prob").flag() ||
           c.at("index_n_heads").integer()!=32 || c.at("index_head_dim").integer()!=128 || c.at("index_kpool").integer()!=4 ||
           c.at("index_topk").integer()!=2048 || !c.at("index_kpool_compress").flag() || !c.at("index_kpool_always_select_tail").flag() ||
           linear.at("num_heads").integer()!=64 || linear.at("head_dim").integer()!=128 || linear.at("short_conv_kernel_size").integer()!=4)
            throw std::runtime_error("GLM EXL3: unsupported attention/routing geometry");
        if(c.at("rms_norm_eps").real()<=0 || c.at("hc_eps").real()<=0 || c.at("hc_sinkhorn_iters").integer()<1 ||
           c.at("hc_sinkhorn_iters").integer()>128 || c.at("swiglu_limit").real()<=0 || c.at("routed_scaling_factor").real()<=0 ||
           linear.at("gate_lower_bound").real()>=0 || !c.at("mla_use_nope").flag() || !c.at("mhc").flag())
            throw std::runtime_error("GLM EXL3: invalid normalization/gating configuration");
        for(unsigned l=0;l<45;++l)if(c.at("layer_types").array[l].string()!=(l%4==3?"deepseek_sparse_attention":"linear_attention"))throw std::runtime_error("GLM EXL3: unsupported layer schedule");
        for(unsigned l=3;l<46;++l)for(unsigned e=0;e<288;++e)for(const auto* projection:{"gate_proj","up_proj","down_proj"}) {
            const auto p=prefix(l,e,projection);bool down=std::string(projection)=="down_proj";
            const uint64_t in=down?2048:4096,out=down?4096:2048;
            tensors.expect(p+".trellis","I16",{in/16,out/16,32});
            tensors.expect(p+".suh","F16",{in});tensors.expect(p+".svh","F16",{out});tensors.expect(p+".mcg","I32",{1});
            uint8_t marker[4];tensors.read(p+".mcg",0,marker,4);
            if(cpu::exl3::word(marker,0)!=0xcbac1fedu)throw std::runtime_error("GLM EXL3: invalid MCG marker "+p);
        }
        for(const auto& [name,t]:tensors.tensors()) {
            if(name.starts_with("model.visual."))vision_bytes+=t.bytes;
            else if(name.starts_with("model.language_model.layers.45."))draft_bytes+=t.bytes;
            else if(name.find(".mlp.experts.")!=std::string::npos)experts_bytes+=t.bytes;
            else { if(t.dtype!="BF16"&&t.dtype!="F32")throw std::runtime_error("GLM EXL3: unsupported fixed dtype "+name);fixed_bytes+=t.bytes; }
        }
    }
    static std::string prefix(unsigned layer,unsigned expert,const std::string& projection) {
        return "model.language_model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(expert)+"."+projection;
    }
    struct Matrix {
        size_t in,out;std::vector<uint8_t> trellis;std::vector<uint16_t>suh,svh;
        void apply(const float* x,float* y)const { cpu::exl3::linear(trellis.data(),suh.data(),svh.data(),in,out,x,y); }
    };
    Matrix load(unsigned layer,unsigned expert,const std::string& projection)const {
        auto p=prefix(layer,expert,projection);const auto& t=tensors.at(p+".trellis");
        Matrix m{size_t(t.shape[0]*16),size_t(t.shape[1]*16),std::vector<uint8_t>(t.bytes),{}, {}};
        m.suh.resize(m.in);m.svh.resize(m.out);tensors.read(t,0,m.trellis.data(),m.trellis.size());
        tensors.read(p+".suh",0,m.suh.data(),m.in*2);tensors.read(p+".svh",0,m.svh.data(),m.out*2);
        for(const auto& scale:{&m.suh,&m.svh})for(uint16_t value:*scale)if(!std::isfinite(cpu::exl3::half_value(value)))throw std::runtime_error("GLM EXL3: nonfinite scale "+p);
        return m;
    }
};
}
