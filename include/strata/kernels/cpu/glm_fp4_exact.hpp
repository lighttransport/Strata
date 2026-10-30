#pragma once
#include "strata/artifact/glm_modelopt.hpp"
#include "strata/kernels/cpu/glm_fp4.hpp"
#include "strata/kernels/cpu/exl3/pool.hpp"
#include <array>
#include "strata/host/glm_env.h"

namespace strata::kernels::cpu {
// The native executor's unquantized W4A16 sequence: FP16-rounded inputs,
// separately evaluated gate/up, clamped SwiGLU, FP16-rounded hidden and down.
// Packed code pointers borrow immutable Safetensors storage for their lifetime.
struct GlmFp4ExactExpert {
    std::array<artifact::ModelOptPacked,3> projection;
    std::array<std::array<float,glm_fp4_weight_table_floats>,3> table;
    std::array<std::unique_ptr<GlmFp4Csf32SparseRows>,3> sparse;
    const bool local_scales=strata_env_is_one("STRATA_GLM_FP4_CPU_CSF32_LOCAL");
    size_t scale_bytes=0;
    GlmFp4ExactExpert(const artifact::GlmModelOpt& model, unsigned layer, unsigned expert, bool qpn, bool compact=strata_env_is_one("STRATA_GLM_FP4_CPU_SPARSE")) {
        unsigned p=0;
        for(const char* part:{"gate_proj","up_proj","down_proj"}) {
            auto& matrix=projection[p];
            const auto name="model.language_model.layers."+std::to_string(layer)+".mlp.experts."+
                            std::to_string(expert)+"."+part+".weight";
            const auto descriptor=model.matrix(name);
            matrix=model.load(descriptor,true);
            if(matrix.format!=artifact::ModelOptFormat::Nvfp4)throw std::runtime_error("FP4 worker requires native NVFP4 experts");
            if(compact && model.has_csf32(descriptor)) {
                const auto csf=model.csf32(descriptor);
                sparse[p]=std::make_unique<GlmFp4Csf32SparseRows>(glm_fp4_csf32_sparse_rows(
                    matrix.scales.data(),csf.bases,csf.codes,matrix.rows,matrix.columns/16,true));
                scale_bytes+=sparse[p]->bytes();std::vector<uint8_t>().swap(matrix.scales);
            }else scale_bytes+=matrix.scales.size();
            glm_fp4_weight_table(matrix.global,qpn,table[p++].data());
        }
        if(projection[0].rows!=projection[1].rows || projection[0].columns!=projection[1].columns ||
           projection[2].columns!=projection[0].rows || projection[2].rows!=projection[0].columns)
            throw std::runtime_error("FP4 worker expert geometry mismatch");
    }
    void product(unsigned p,const float* x,float* y,int tokens,int first,int last)const {
        const auto& m=projection[p];
        if(sparse[p]&&local_scales)glm_fp4_rows_csf32_sparse_local_table(m.data(),*sparse[p],table[p].data(),x,y,m.columns,m.rows,tokens,first,last);
        else if(sparse[p])glm_fp4_rows_csf32_sparse_table(m.data(),*sparse[p],table[p].data(),x,y,m.columns,m.rows,tokens,first,last);
        else glm_fp4_rows_table(m.data(),m.scales.data(),table[p].data(),x,y,m.columns,m.rows,tokens,first,last);
    }
    // Experimental activation quantization. Callers must select this explicitly;
    // run() remains the exact W4A16 reference used by the wire protocol.
    std::vector<float> run_q8(strata::cpu::exl3::Pool& pool,const std::vector<float>& x,int tokens,float limit)const {
        const size_t width=projection[0].columns,hidden=projection[0].rows;
        if(tokens<1||tokens>8||x.size()!=size_t(tokens)*width||!std::isfinite(limit)||limit<=0)
            throw std::invalid_argument("FP4 Q8 invalid activation geometry or clamp");
        for(unsigned p=0;p<3;++p)if(!sparse[p])throw std::invalid_argument("FP4 Q8 requires compact CSF32 scales");
        for(float v:x)if(!std::isfinite(v))throw std::invalid_argument("FP4 Q8 nonfinite input");
        auto input=x;glm_fp4_round_f16(input.data(),input.size());
        auto qi=glm_fp4_quantize_q8(input.data(),width,tokens);
        std::vector<float> gate(size_t(tokens)*hidden),up(gate.size()),h(gate.size()),out(size_t(tokens)*width);
        auto multiply=[&](unsigned p,const GlmFp4Q8& q,float* y,int first,int last) {
            const auto& m=projection[p];
            glm_fp4_rows_csf32_q8(m.data(),*sparse[p],table[p].data(),q,y,m.columns,m.rows,tokens,first,last,1,true,false);
        };
        pool.run([&](size_t rank) {
            const int first=hidden*rank/pool.size(),last=hidden*(rank+1)/pool.size();
            multiply(0,qi,gate.data(),first,last);multiply(1,qi,up.data(),first,last);
            for(int t=0;t<tokens;++t)for(int r=first;r<last;++r) {
                const size_t at=size_t(t)*hidden+r;
                const float g=std::min(gate[at],limit),u=std::clamp(up[at],-limit,limit);
                h[at]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round((g/(1.f+std::exp(-g)))*u));
            }
        });
        auto qh=glm_fp4_quantize_q8(h.data(),hidden,tokens);
        pool.run([&](size_t rank){multiply(2,qh,out.data(),width*rank/pool.size(),width*(rank+1)/pool.size());});
        for(float v:out)if(!std::isfinite(v))throw std::runtime_error("FP4 Q8 nonfinite output");
        return out;
    }
    std::vector<float> run(strata::cpu::exl3::Pool& pool,const std::vector<float>& x,int tokens,float limit)const {
        const size_t width=projection[0].columns,hidden=projection[0].rows;
        if(tokens<1||tokens>8||x.size()!=size_t(tokens)*width||!std::isfinite(limit)||limit<=0)
            throw std::invalid_argument("FP4 worker invalid activation geometry or clamp");
        for(float v:x)if(!std::isfinite(v))throw std::invalid_argument("FP4 worker nonfinite input");
        auto input=x;glm_fp4_round_f16(input.data(),input.size());
        std::vector<float> gate(size_t(tokens)*hidden),up(gate.size()),h(gate.size()),out(size_t(tokens)*width);
        pool.run([&](size_t rank) {
            const int first=hidden*rank/pool.size(),last=hidden*(rank+1)/pool.size();
            for(int p=0;p<2;++p)product(p,input.data(),p?up.data():gate.data(),tokens,first,last);
            for(int t=0;t<tokens;++t)for(int r=first;r<last;++r) {
                const size_t at=size_t(t)*hidden+r;
                const float g=std::min(gate[at],limit),u=std::clamp(up[at],-limit,limit);
                h[at]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round((g/(1.f+std::exp(-g)))*u));
            }
        });
        pool.run([&](size_t rank) {
            product(2,h.data(),out.data(),tokens,width*rank/pool.size(),width*(rank+1)/pool.size());
        });
        for(float v:out)if(!std::isfinite(v))throw std::runtime_error("FP4 worker nonfinite output");
        return out;
    }
};
}
