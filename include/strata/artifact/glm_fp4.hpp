#pragma once
#include "strata/artifact/safetensors.hpp"
#include "strata/host/glm_fp4.h"
#include <cmath>
#include <cstring>

namespace strata::artifact {
// Lossless checkpoint inputs. Global scale remains separate from local E2M1*E4M3.
struct GlmFp4Matrix {
    size_t rows=0,cols=0;
    float global=1;
    std::vector<uint8_t> codes,scales;
    size_t bytes()const {return codes.size()+scales.size()+sizeof(global);}
    std::vector<float> dequantize()const {
        std::vector<float> out(rows*cols);
        for(size_t i=0;i<out.size();++i)
            out[i]=strata_glm_e2m1((codes[i/2]>>((i%2)*4))&15)*strata_glm_e4m3(scales[i/16])*global;
        return out;
    }
};
inline std::vector<uint8_t> glm_tensor_bytes(const Safetensors& ts,const std::string& name) {
    const auto& t=ts.at(name);std::vector<uint8_t> out(t.bytes);ts.read(t,0,out.data(),out.size());return out;
}
inline GlmFp4Matrix glm_load_fp4(const Safetensors& ts,const std::string& prefix) {
    const auto& w=ts.at(prefix+".weight");
    if(w.dtype!="U8"||w.shape.size()!=2||w.shape[0]==0||w.shape[1]==0||w.shape[1]%8)
        throw std::runtime_error("GLM FP4: invalid packed matrix "+prefix);
    GlmFp4Matrix m;m.rows=w.shape[0];m.cols=w.shape[1]*2;m.codes=glm_tensor_bytes(ts,prefix+".weight");
    const auto& g=ts.at(prefix+".weight_scale_2");
    if(g.dtype!="F32"||g.bytes!=4)throw std::runtime_error("GLM FP4: global scale must be scalar F32");
    ts.read(g,0,&m.global,4);
    if(!(m.global>0)||!std::isfinite(m.global))throw std::runtime_error("GLM FP4: invalid global scale");
    const std::string base=prefix+".weight_scale";
    if(ts.tensors().contains(base)) {
        ts.expect(base,"F8_E4M3",{m.rows,m.cols/16});m.scales=glm_tensor_bytes(ts,base);
    }else {
        const size_t r=m.rows,c=m.cols/16,groups=((r+127)/128)*((c+3)/4),pages=(groups+31)/32;
        ts.expect(base+".csf32_bases","U8",{r});ts.expect(base+".csf32_codes","U8",{r*c/2});
        ts.expect(base+".csf32_page_bases","U32",{pages});ts.expect(base+".csf32_offsets","U16",{r/32*((c+1)/2)+pages});
        const auto& e=ts.at(base+".csf32_exceptions");
        if(e.dtype!="U16"||e.shape.size()!=1)throw std::runtime_error("GLM FP4: invalid exception tensor");
        auto b=glm_tensor_bytes(ts,base+".csf32_bases"),v=glm_tensor_bytes(ts,base+".csf32_codes"),
             p=glm_tensor_bytes(ts,base+".csf32_page_bases"),o=glm_tensor_bytes(ts,base+".csf32_offsets"),x=glm_tensor_bytes(ts,base+".csf32_exceptions");
        m.scales.resize(r*c);
        int rc=strata_glm_csf32_decode(r,c,b.data(),b.size(),v.data(),v.size(),p.data(),p.size(),o.data(),o.size(),x.data(),x.size(),m.scales.data(),m.scales.size());
        if(rc)throw std::runtime_error("GLM FP4: invalid CSF scale stream, code="+std::to_string(rc));
    }
    for(auto s:m.scales)if(s>=127)throw std::runtime_error("GLM FP4: scale must be finite and nonnegative");
    return m;
}
// MXFP8: E4M3 data, one UE8M0 byte per 32 consecutive K values.
inline std::vector<float> glm_load_mxfp8(const Safetensors& ts,const std::string& prefix) {
    const auto& w=ts.at(prefix+".weight");const auto& s=ts.at(prefix+".weight_scale");
    if(w.dtype!="F8_E4M3"||w.shape.size()!=2||!w.shape[0]||!w.shape[1]||w.shape[1]%32 ||
       s.dtype!="U8"||s.shape!=std::vector<uint64_t>{w.shape[0],w.shape[1]/32})
        throw std::runtime_error("GLM MXFP8: invalid matrix/scale geometry "+prefix);
    auto data=glm_tensor_bytes(ts,prefix+".weight"),scale=glm_tensor_bytes(ts,prefix+".weight_scale");
    std::vector<float> out(data.size());
    for(size_t i=0;i<data.size();++i) {
        out[i]=strata_glm_e4m3(data[i])*strata_glm_ue8m0(scale[i/32]);
        if(!std::isfinite(out[i]))throw std::runtime_error("GLM MXFP8: nonfinite decoded weight");
    }
    return out;
}
}
