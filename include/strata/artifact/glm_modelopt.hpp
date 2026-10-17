#pragma once
#include "strata/artifact/safetensors.hpp"
#include "strata/host/glm_fp4.h"
#include <cmath>
#include <cstring>

namespace strata::artifact {
// These are native checkpoint formats, not GGML type numbers. Scales retain
// their original semantics; in particular MXFP8's UE8M0 is not E4M3.
enum class ModelOptFormat { Nvfp4, Mxfp8, BlockFp8, Bf16, F32 };
struct ModelOptMatrix {
    ModelOptFormat format;
    const TensorView* weights;
    size_t rows, columns;
    std::string name;
};
struct ModelOptPacked {
    ModelOptFormat format;
    size_t rows, columns;
    float global = 1;
    std::vector<uint8_t> weights, scales;
    const uint8_t* mapped = nullptr;
    const uint8_t* data()const {return mapped?mapped:weights.data();}
};
struct ModelOptCsf32 {
    float global;
    const uint8_t *bases,*codes;
    size_t codes_count;
    std::vector<uint16_t> offsets,exceptions;
    std::vector<uint32_t> pages;
};
class GlmModelOpt {
    const Safetensors& tensors_;
    const Json& config_;
    template<class T> std::vector<T> read(const std::string& name, const char* dtype) const {
        const auto& t=tensors_.at(name);
        if(t.dtype!=dtype || t.bytes%sizeof(T))throw std::runtime_error("GLM ModelOpt: invalid scale dtype: "+name);
        std::vector<T> values(t.bytes/sizeof(T));
        tensors_.read(t,0,values.data(),t.bytes);
        return values;
    }
public:
    GlmModelOpt(const Safetensors& tensors,const Json& config):tensors_(tensors),config_(config) {}
    bool has_csf32(const ModelOptMatrix& m)const {
        return m.format==ModelOptFormat::Nvfp4 && !tensors_.tensors().contains(m.name+"_scale");
    }
    ModelOptCsf32 csf32(const ModelOptMatrix& m)const {
        if(!has_csf32(m))throw std::invalid_argument("GLM ModelOpt: expected compressed CSF32 scales");
        const auto s=m.name+"_scale.csf32_";
        const auto g=read<float>(m.name+"_scale_2","F32");
        if(g.size()!=1 || !std::isfinite(g[0]) || g[0]<=0)throw std::runtime_error("GLM ModelOpt: invalid global scale: "+m.name);
        ModelOptCsf32 c{g[0],tensors_.data(s+"bases"),tensors_.data(s+"codes"),size_t(tensors_.at(s+"codes").bytes),
                       read<uint16_t>(s+"offsets","U16"),read<uint16_t>(s+"exceptions","U16"),read<uint32_t>(s+"page_bases","U32")};
        if(strata_fp4_csf32_validate(m.rows,m.columns/16,c.bases,m.rows,c.codes,c.codes_count,
             c.offsets.data(),c.offsets.size(),c.pages.data(),c.pages.size(),c.exceptions.data(),c.exceptions.size()))
            throw std::runtime_error("GLM ModelOpt: malformed CSF32 scales: "+m.name);
        return c;
    }
    ModelOptMatrix matrix(const std::string& name)const {
        const auto& t=tensors_.at(name);
        if(t.shape.size()!=2 || !t.shape[0] || !t.shape[1])throw std::runtime_error("GLM ModelOpt: invalid matrix: "+name);
        ModelOptMatrix m{ModelOptFormat::F32,&t,size_t(t.shape[0]),size_t(t.shape[1]),name};
        if(t.dtype=="U8") {
            if(m.columns>SIZE_MAX/2)throw std::runtime_error("GLM ModelOpt: matrix size overflow");
            m.format=ModelOptFormat::Nvfp4;m.columns*=2;
            tensors_.expect(name+"_scale_2","F32",{});
            if(m.columns%16)throw std::runtime_error("GLM ModelOpt: FP4 group width");
            if(tensors_.tensors().contains(name+"_scale")) {
                tensors_.expect(name+"_scale","F8_E4M3",{m.rows,m.columns/16});
            }else {
                const auto s=name+"_scale.csf32_";
                tensors_.expect(s+"bases","U8",{m.rows});
                tensors_.expect(s+"codes","U8",{m.rows*m.columns/32});
                if(m.rows%128 || m.columns%64)throw std::runtime_error("GLM ModelOpt: CSF32 matrix alignment");
                const size_t tiles=(m.rows/128)*(m.columns/64),pages=(tiles+31)/32;
                tensors_.expect(s+"page_bases","U32",{pages});
                tensors_.expect(s+"offsets","U16",{m.rows/32*(m.columns/32)+pages});
                if(tensors_.at(s+"exceptions").dtype!="U16" || tensors_.at(s+"exceptions").shape.size()!=1)
                    throw std::runtime_error("GLM ModelOpt: CSF32 exceptions: "+name);
            }
        }else if(t.dtype=="F8_E4M3") {
            const auto& s=tensors_.at(name+"_scale");
            if(s.dtype=="U8") {
                m.format=ModelOptFormat::Mxfp8;
                if(m.columns%32 || s.shape!=std::vector<uint64_t>{m.rows,m.columns/32})
                    throw std::runtime_error("GLM ModelOpt: MXFP8 scale grid: "+name);
            }else if(s.dtype=="F32") {
                m.format=ModelOptFormat::BlockFp8;
                const auto r=(m.rows+127)/128,c=(m.columns+127)/128;
                if(s.shape!=std::vector<uint64_t>{r,c} && s.shape!=std::vector<uint64_t>{r,1,c,1})
                    throw std::runtime_error("GLM ModelOpt: block FP8 scale grid: "+name);
            }else throw std::runtime_error("GLM ModelOpt: unknown FP8 scale codec: "+name);
        }else if(t.dtype=="BF16")m.format=ModelOptFormat::Bf16;
        else if(t.dtype!="F32")throw std::runtime_error("GLM ModelOpt: unknown matrix dtype: "+name);
        // Quantized matrices must agree with the module's published policy.
        // Draft experts have direct scales and may omit the parent policy.
        std::string module=name.ends_with(".weight")?name.substr(0,name.size()-7):name;
        const auto expert=module.find(".mlp.experts.");
        if(expert!=std::string::npos)module.resize(expert+12);
        const auto& q=config_.at("quantization_config").at("quantized_layers");
        if(const auto* policy=q.find(module)) {
            const auto algo=policy->at("quant_algo").string();
            const bool valid=(m.format==ModelOptFormat::Nvfp4&&(algo=="NVFP4"||algo=="W4A16_NVFP4")) ||
                (m.format==ModelOptFormat::Mxfp8&&algo=="MXFP8") ||
                (m.format==ModelOptFormat::BlockFp8&&algo=="FP8_PB_WO");
            if(!valid)throw std::runtime_error("GLM ModelOpt: module policy disagrees with weights: "+name);
            if(m.format==ModelOptFormat::Nvfp4 && !tensors_.tensors().contains(name+"_scale") &&
                policy->at("weight_scale_encoding").string()!="csf_32x2_v1")
                throw std::runtime_error("GLM ModelOpt: unsupported compressed scale encoding");
        }
        return m;
    }
    ModelOptPacked load(const ModelOptMatrix& m,bool map_weights=false)const {
        ModelOptPacked p{m.format,m.rows,m.columns,1,{},{}};
        if(map_weights)p.mapped=tensors_.data(*m.weights);
        else {p.weights.resize(m.weights->bytes);tensors_.read(*m.weights,0,p.weights.data(),p.weights.size());}
        if(m.format==ModelOptFormat::Nvfp4) {
            const auto g=read<float>(m.name+"_scale_2","F32");
            if(g.size()!=1 || !std::isfinite(g[0]) || g[0]<=0)throw std::runtime_error("GLM ModelOpt: invalid global scale: "+m.name);
            p.global=g[0];
            if(tensors_.tensors().contains(m.name+"_scale"))p.scales=read<uint8_t>(m.name+"_scale","F8_E4M3");
            else {
                const auto s=m.name+"_scale.csf32_";
                const auto b=read<uint8_t>(s+"bases","U8"),c=read<uint8_t>(s+"codes","U8");
                const auto o=read<uint16_t>(s+"offsets","U16"),e=read<uint16_t>(s+"exceptions","U16");
                const auto pages=read<uint32_t>(s+"page_bases","U32");
                p.scales.resize(m.rows*m.columns/16);
                if(strata_fp4_csf32_decode(m.rows,m.columns/16,b.data(),b.size(),c.data(),c.size(),
                    o.data(),o.size(),pages.data(),pages.size(),e.data(),e.size(),p.scales.data(),p.scales.size()))
                    throw std::runtime_error("GLM ModelOpt: malformed CSF32 scales: "+m.name);
            }
            for(uint8_t s:p.scales)if(s>=127)
                throw std::runtime_error("GLM ModelOpt: invalid FP4 group scale: "+m.name);
        }else if(m.format==ModelOptFormat::Mxfp8 || m.format==ModelOptFormat::BlockFp8) {
            const auto& s=tensors_.at(m.name+"_scale");p.scales.resize(s.bytes);tensors_.read(s,0,p.scales.data(),s.bytes);
            if(m.format==ModelOptFormat::Mxfp8) {
                for(auto value:p.scales)if(value==255)throw std::runtime_error("GLM ModelOpt: NaN MXFP8 scale");
            }else {
                for(size_t i=0;i<p.scales.size();i+=4) {float v;std::memcpy(&v,p.scales.data()+i,4);
                    if(!std::isfinite(v)||v<0)throw std::runtime_error("GLM ModelOpt: invalid block FP8 scale");}
            }
        }
        return p;
    }
    ModelOptPacked load(const std::string& name,bool map_weights=false)const {return load(matrix(name),map_weights);}
    static float value(const ModelOptPacked& m,size_t row,size_t column) {
        const size_t i=row*m.columns+column;
        if(m.format==ModelOptFormat::Nvfp4) {
            const auto q=(m.data()[i/2]>>((i%2)*4))&15;
            return strata_fp4_e2m1(q)*strata_fp4_e4m3(m.scales[row*(m.columns/16)+column/16])*m.global;
        }
        if(m.format==ModelOptFormat::Mxfp8)return strata_fp4_e4m3(m.data()[i])*strata_fp4_e8m0(m.scales[row*(m.columns/32)+column/32]);
        if(m.format==ModelOptFormat::BlockFp8) {float scale;std::memcpy(&scale,m.scales.data()+4*((row/128)*((m.columns+127)/128)+column/128),4);return strata_fp4_e4m3(m.data()[i])*scale;}
        if(m.format==ModelOptFormat::Bf16) {uint32_t bits=(uint32_t(m.data()[2*i])|(uint32_t(m.data()[2*i+1])<<8))<<16;float v;std::memcpy(&v,&bits,4);return v;}
        float v;std::memcpy(&v,m.data()+4*i,4);return v;
    }
};
}
