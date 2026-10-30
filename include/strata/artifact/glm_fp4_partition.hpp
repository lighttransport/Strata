#pragma once
#include "strata/artifact/glm_modelopt.hpp"
#include "strata/host/glm_fp4_layout.h"
#include <cstdio>

namespace strata::artifact {
// A compact local partition has no sparse placeholders and no implicit remote reads.
// The worker opens this directly; the full transformer catalog is a separate concern.
class GlmFp4Partition {
public:
    Json report, config;
    Safetensors tensors;
    GlmModelOpt model;
    unsigned remote_count;
    std::vector<const TensorView*> ordered;
    explicit GlmFp4Partition(const std::filesystem::path& root,bool ram_fds=false)
        :report(read_json((root/"partition.json").string())),config(read_json((root/"config.json").string())),
         tensors(root,true,false,ram_fds),model(tensors,config),remote_count(report.at("remote_experts_per_layer").integer()) {
        if(report.at("schema").string()!="strata.glm.fp4.partition.v1" || report.at("owner").string()!="worker" || report.at("remote_experts_per_layer").integer()>288 ||
           report.at("payload_bytes").integer()!=tensors.bytes())throw std::runtime_error("invalid FP4 worker partition");
        if(const auto* fixed=report.find("gpu_fixed_shards");fixed&&!fixed->array.empty())
            throw std::runtime_error("FP4 CPU worker cannot own GPU-fixed disk files");
        for(const char* field:{"layout_sha256","index_sha256"}) {
            const auto& digest=report.at(field).string();
            if(digest.size()!=64 || digest.find_first_not_of("0123456789abcdef")!=std::string::npos)
                throw std::runtime_error("invalid FP4 partition fingerprint");
        }
        uint64_t sum=0;
        for(const auto& [name,t]:tensors.tensors()) {
            unsigned layer=0,expert=0;int length=0;
            if(std::sscanf(name.c_str(),"model.language_model.layers.%u.mlp.experts.%u.%n",&layer,&expert,&length)!=2 || !length ||
               strata_glm_fp4_layout_owner(layer,expert,remote_count)!=1)
                throw std::runtime_error("FP4 partition contains an unowned tensor: "+name);
            if(t.bytes>32*1024*1024 || t.bytes>UINT64_MAX-sum)throw std::runtime_error("FP4 partition tensor size overflow");
            sum+=t.bytes;ordered.push_back(&t);
        }
        for(unsigned layer=3;layer<45;++layer)for(unsigned expert=0;expert<remote_count;++expert)
            for(const char* part:{"gate_proj","up_proj","down_proj"}) {
                const auto matrix=model.matrix("model.language_model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(expert)+"."+part+".weight");
                if(matrix.rows!=unsigned(part==std::string("down_proj")?4096:2048) ||
                   matrix.columns!=unsigned(part==std::string("down_proj")?2048:4096))
                    throw std::runtime_error("FP4 partition expert geometry mismatch");
            }
    }
};
}
