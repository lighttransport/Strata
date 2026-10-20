// Offline calibration operands. The checkpoint and inference path are read-only.
#include "strata/core/model.hpp"
#include "strata/kernels/glm_fp4.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include <cuda_runtime.h>
#include <bit>
#include <algorithm>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace k=strata::kernels;
static void check(cudaError_t code) {
    if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
struct Buffer {
    void* p=nullptr;
    explicit Buffer(size_t bytes){check(cudaMalloc(&p,bytes));}
    ~Buffer(){cudaFree(p);}
};
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::invalid_argument("usage: strata-glm-fp4-export-head MODEL_DIR NEW_OUTPUT_DIR");
        if(std::endian::native!=std::endian::little)throw std::runtime_error("little-endian host required");
        strata::core::ModelArtifact artifact(argv[1]);
        if(!artifact.is_modelopt())throw std::runtime_error("native ModelOpt checkpoint required");
        const auto& binding=artifact.at("output.weight");
        if(!binding.native)throw std::runtime_error("native head binding required");
        const auto& model=*artifact.exl3().modelopt;
        const auto matrix=model.matrix(binding.native->source);
        const auto& allowed=artifact.exl3().token_ids;
        if(matrix.format!=strata::artifact::ModelOptFormat::Mxfp8||matrix.rows!=allowed.size()||
           matrix.rows%32||matrix.columns%64||matrix.rows>INT_MAX||matrix.columns>INT_MAX||
           matrix.columns>SIZE_MAX/matrix.rows/2)
            throw std::runtime_error("aligned MXFP8 head required");
        const int N=int(matrix.rows),K=int(matrix.columns);
        const size_t count=matrix.rows*matrix.columns;
        const std::filesystem::path out=argv[2];
        if(!std::filesystem::create_directory(out))throw std::runtime_error("output directory already exists");
        auto host=model.load(matrix);
        Buffer raw(count),scales(host.scales.size()),packed(count),packed_scales(count/16);
        Buffer cached(count*2),row_major(count*2),roundtrip(count*2);
        check(cudaMemcpy(raw.p,host.data(),count,cudaMemcpyHostToDevice));
        check(cudaMemcpy(scales.p,host.scales.data(),host.scales.size(),cudaMemcpyHostToDevice));
        k::glm_fp8_prepack(k::GlmPackedFormat::Mxfp8,static_cast<uint8_t*>(raw.p),scales.p,
            static_cast<uint8_t*>(packed.p),packed_scales.p,N,K,nullptr);
        k::glm_fp8_cache_f16_qpn(k::GlmPackedFormat::Mxfp8Qpn,packed.p,packed_scales.p,
            static_cast<uint16_t*>(cached.p),N,K,nullptr);
        k::glm_fp4_dequant_f16(k::GlmPackedFormat::F16Qpn,cached.p,nullptr,1,
            static_cast<uint16_t*>(row_major.p),N,K,nullptr);
        k::glm_f16_prepack_qpn(static_cast<uint16_t*>(row_major.p),
            static_cast<uint16_t*>(roundtrip.p),N,K,nullptr);
        // Compare the cached operands to the actual compressed-head kernel,
        // using its identical K partitioning at single and block widths.
        Buffer input(size_t(8)*K*2),reference(size_t(8)*N*4),result(size_t(8)*N*4);
        std::vector<uint16_t> x(size_t(8)*K);
        for(size_t i=0;i<x.size();++i)x[i]=strata::cpu::exl3::half_round(std::sin(float(i)*.17f));
        check(cudaMemcpy(input.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
        std::vector<float> want(size_t(8)*N),got(want.size());
        for(int M:{1,8}) {
            k::glm_fp4_gemm(k::GlmPackedFormat::Mxfp8Qpn,packed.p,packed_scales.p,1,
                static_cast<uint16_t*>(input.p),static_cast<float*>(reference.p),N,K,M,true,nullptr);
            k::glm_fp4_gemm(k::GlmPackedFormat::F16Qpn,roundtrip.p,nullptr,1,
                static_cast<uint16_t*>(input.p),static_cast<float*>(result.p),N,K,M,true,nullptr);
            check(cudaMemcpy(want.data(),reference.p,size_t(M)*N*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(got.data(),result.p,size_t(M)*N*4,cudaMemcpyDeviceToHost));
            for(int row=0;row<M;++row)for(int id=0;id<N;++id)if(allowed[id]) {
                const size_t at=size_t(row)*N+id;
                if(!std::isfinite(want[at])||std::bit_cast<uint32_t>(want[at])!=std::bit_cast<uint32_t>(got[at]))
                    throw std::runtime_error("exported operands change valid head output bits");
            }
        }
        std::ofstream weights(out/"weights.f16.partial",std::ios::binary);
        std::ofstream mask(out/"allowed.u8.partial",std::ios::binary);
        weights.exceptions(std::ios::failbit|std::ios::badbit);mask.exceptions(std::ios::failbit|std::ios::badbit);
        std::vector<uint16_t> rows(size_t(256)*K);size_t valid=0;
        for(int first=0;first<N;first+=256) {
            const int n=std::min(256,N-first);
            check(cudaMemcpy(rows.data(),static_cast<uint16_t*>(row_major.p)+size_t(first)*K,
                size_t(n)*K*2,cudaMemcpyDeviceToHost));
            for(int r=0;r<n;++r) {
                const uint8_t keep=allowed[first+r];mask.write(reinterpret_cast<const char*>(&keep),1);valid+=keep;
                for(int col=0;col<K;++col) {
                    auto& value=rows[size_t(r)*K+col];
                    if(!keep)value=0; // Invalid vocabulary rows never participate in native selection.
                    else if((value&0x7c00)==0x7c00)throw std::runtime_error("nonfinite valid head operand");
                }
            }
            weights.write(reinterpret_cast<const char*>(rows.data()),size_t(n)*K*2);
        }
        weights.close();mask.close();
        std::filesystem::rename(out/"weights.f16.partial",out/"weights.f16");
        std::filesystem::rename(out/"allowed.u8.partial",out/"allowed.u8");
        std::ofstream metadata(out/"metadata.json");metadata.exceptions(std::ios::failbit|std::ios::badbit);
        metadata<<"{\"complete\":true,\"rows\":"<<N<<",\"columns\":"<<K<<",\"valid_rows\":"<<valid
            <<",\"weights_bytes\":"<<count*2<<",\"dtype\":\"little_endian_float16\","
            <<"\"valid_head_bitwise_widths\":[1,8],\"invalid_rows_zeroed\":true,\"performance_anchor\":false}\n";
        metadata.close();
        std::cout<<"HEAD_EXPORT rows="<<N<<" columns="<<K<<" valid_rows="<<valid<<" bytes="<<count*2
                 <<" bitwise_widths=1,8 PASS\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
