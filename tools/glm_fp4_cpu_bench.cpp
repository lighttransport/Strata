// Rotating native-FP4 code working set; excludes activation quantization/model loading.
#include "strata/kernels/cpu/glm_fp4.hpp"
#include "strata/kernels/cpu/exl3/pool.hpp"
#include "strata/kernels/cpu/exl3/math.hpp"
#include "strata/host/glm_fp4.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
namespace c=strata::kernels::cpu;
int main(int argc,char** argv) {try {
    const int workers=argc>1?std::stoi(argv[1]):1;
    if(workers<1||workers>72)throw std::invalid_argument("workers must be 1..72");
    strata::cpu::exl3::Pool pool(workers,true,true);
    for(int N:{2048,4096}) {
        const int K=N==2048?4096:2048,G=K/16,copies=16,repeats=64;
        const size_t wb=size_t(N)*K/2,sb=size_t(N)*G;
        std::vector<uint8_t>w(wb*copies),bases(N),codes(sb/2),scales(sb);
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(i*53+7);
        for(int i=0;i<N;++i)bases[i]=30+i%20;
        for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*59+19);
        const size_t tiles=size_t(N/128)*(G/4),pages=(tiles+31)/32;
        std::vector<uint16_t> offsets(size_t(N/32)*(G/2)+pages);std::vector<uint32_t> page_bases(pages);
        if(strata_fp4_csf32_decode(N,G,bases.data(),bases.size(),codes.data(),codes.size(),offsets.data(),offsets.size(),page_bases.data(),pages,nullptr,0,scales.data(),scales.size()))throw std::runtime_error("CSF32 fixture");
        for(size_t i=0;i<scales.size();i+=1021)scales[i]=uint8_t(8+i%110);
        auto sparse=c::glm_fp4_csf32_sparse_rows(scales.data(),bases.data(),codes.data(),N,G,true),cached=sparse;
        c::glm_fp4_cache_scale_quartets(cached);
        std::vector<float>x(K*8),reference(N*8),out(N*8);
        for(size_t i=0;i<x.size();++i)x[i]=strata::cpu::exl3::half_value(strata::cpu::exl3::half_round(std::sin(float(i)*.37f)));
        const auto q=c::glm_fp4_quantize_q8(x.data(),K,8);auto prepared=q;c::glm_fp4_prepare_q8(prepared);
        alignas(64)float table[c::glm_fp4_weight_table_floats];c::glm_fp4_weight_table(.00061f,true,table);
        struct Variant {const char* name;int tile;bool prepare,cache;};
        const Variant variants[]={{"baseline",1,false,false},{"prepared",1,true,false},{"tile2",2,false,false},
            {"tile4",4,false,false},{"tile2_prepared",2,true,false},{"quartets",1,false,true}};
        auto run=[&](const Variant& v,int width,int copy) {
            pool.run([&](size_t rank) {
                c::glm_fp4_rows_csf32_q8(w.data()+wb*copy,v.cache?cached:sparse,table,v.prepare?prepared:q,
                    out.data(),K,N,width,N*rank/pool.size(),N*(rank+1)/pool.size(),v.tile);
            });
        };
        std::cout<<"{\"kind\":\"configuration\",\"rows\":"<<N<<",\"columns\":"<<K
            <<",\"workers\":"<<workers<<",\"code_working_set_bytes\":"<<w.size()<<",\"vnni\":"<<c::glm_fp4_q8_vnni_available()<<"}\n";
        for(int width:{1,2,4,8}) {
            run(variants[0],width,0);reference=out;
            for(const auto& v:variants) {
                run(v,width,0);
                if(std::memcmp(reference.data(),out.data(),size_t(width)*N*4))throw std::runtime_error("variant changes Q8 outputs");
            }
            for(int sample=0;sample<3;++sample)for(int order=0;order<6;++order) {
                const auto& v=variants[(order+sample)%6];
                for(int i=0;i<copies;++i)run(v,width,i);
                const auto start=std::chrono::steady_clock::now();
                for(int i=0;i<repeats;++i)run(v,width,i%copies);
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
                std::cout<<"{\"kind\":\"timing\",\"rows\":"<<N<<",\"columns\":"<<K<<",\"workers\":"<<workers
                    <<",\"width\":"<<width<<",\"variant\":\""<<v.name<<"\",\"sample\":"<<sample<<",\"us\":"<<us<<"}\n";
            }
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
