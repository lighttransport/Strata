#include "strata/core/model.hpp"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc<2 || argc>3)throw std::runtime_error("usage: strata-glm-fp4-check MODEL_DIR [TENSOR_WEIGHT_NAME]");
        strata::core::ModelArtifact artifact(argv[1]);
        if(!artifact.is_modelopt())throw std::runtime_error("expected a native ModelOpt GLM checkpoint");
        const auto& model=artifact.exl3();
        std::array<size_t,5> formats{};
        for(const auto& [name,t]:artifact.tensors())if(t.native && t.tensor->shape.size()==2) {
            const auto m=model.modelopt->matrix(t.native->source);
            ++formats.at(size_t(m.format));
        }
        std::cout<<"MODEL_OPT tensors="<<model.tensors.tensors().size()<<" payload_bytes="<<model.tensors.bytes()
                 <<" main_expert_bytes="<<model.experts_bytes<<" fixed_bytes="<<model.fixed_bytes
                 <<" draft_bytes="<<model.draft_bytes<<" vision_bytes="<<model.vision_bytes<<'\n';
        std::cout<<"FIXED_FORMATS nvfp4="<<formats[0]<<" mxfp8="<<formats[1]<<" block_fp8="<<formats[2]
                 <<" bf16="<<formats[3]<<" f32="<<formats[4]<<'\n';
        const std::string name=argc==3?argv[2]:"model.language_model.layers.3.mlp.experts.0.gate_proj.weight";
        const auto m=model.modelopt->load(name);
        double sum=0;float maximum=0;
        for(size_t r=0;r<m.rows;++r)for(size_t c=0;c<m.columns;++c) {
            const float v=strata::artifact::GlmModelOpt::value(m,r,c);
            if(!std::isfinite(v))throw std::runtime_error("nonfinite reconstructed weight");
            maximum=std::max(maximum,std::abs(v));sum+=double(v)*v;
        }
        std::cout<<"MATRIX rows="<<m.rows<<" columns="<<m.columns<<" stored_bytes="<<m.weights.size()
                 <<" scale_bytes="<<m.scales.size()<<" global="<<m.global<<" max_abs="<<maximum
                 <<" sum_squares="<<sum<<" PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
