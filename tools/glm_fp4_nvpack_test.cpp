#include "strata/artifact/glm_fp4.hpp"
#include "ggml.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
using strata::artifact::GlmFp4Matrix;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void check(const GlmFp4Matrix& m) {
    const size_t bytes=m.rows*(m.cols/64)*36;
    require(ggml_row_size(GGML_TYPE_NVFP4,m.cols)*m.rows==bytes,"GGML NVFP4 extent changed");
    std::vector<uint8_t> packed(bytes+2,0xa5);
    require(!strata_glm_fp4_pack_nvfp4(m.rows,m.cols,m.codes.data(),m.codes.size(),m.scales.data(),m.scales.size(),packed.data()+1,bytes),"packing failed");
    require(packed.front()==0xa5&&packed.back()==0xa5,"output guard changed");
    std::vector<uint8_t> cpu(bytes+2,0xa5);
    require(!strata_glm_nvfp4_repack_k32(packed.data()+1,cpu.data()+1,bytes),"native K32 conversion failed");
    require(cpu.front()==0xa5&&cpu.back()==0xa5,"native K32 guard changed");
    std::vector<float> decoded(m.rows*m.cols);
    ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(packed.data()+1,decoded.data(),decoded.size());
    for(size_t i=0;i<decoded.size();++i) {
        const unsigned q=(m.codes[i/2]>>((i%2)*4))&15;
        const auto* block=packed.data()+1+(i/64)*36;
        const unsigned j=i%16,sub=(i%64)/16;
        const unsigned back=(block[4+sub*8+j%8]>>(j/8*4))&15;
        require(back==q&&block[sub]==m.scales[i/16],"code or scale payload changed");
        const auto* cpu_block=cpu.data()+1+(i/64)*36;
        const unsigned k=i%64;
        const unsigned cpu_code=(cpu_block[4+(k/32)*16+k%16]>>((k%32)/16*4))&15;
        require(cpu_code==q&&cpu_block[sub]==m.scales[i/16],"native K32 code or scale changed");
        // GGML's integer E2M1 table collapses signed zero; numeric values are exact.
        const float local=strata_glm_e2m1(q)*strata_glm_e4m3(m.scales[i/16]);
        require(decoded[i]==local,"GGML decoded value differs");
        require(decoded[i]*m.global==local*m.global,"global scale mismatch");
    }
    require(!strata_glm_nvfp4_repack_k32(cpu.data()+1,cpu.data()+1,bytes),"native K32 inverse failed");
    require(cpu==packed,"native K32 roundtrip changed payload or guards");
    std::cout<<"PASS rows="<<m.rows<<" cols="<<m.cols<<" weights="<<decoded.size()<<" exact_payload=1 exact_numeric_values=1\n";
}
static void rejects() {
    std::vector<uint8_t> codes(32,0x19),scales(4,56),out(36,0xa5);
    auto bad=[&](size_t r,size_t k,const uint8_t* q,size_t nq,const uint8_t* s,size_t ns,uint8_t* y,size_t ny) {
        const auto q0=codes,s0=scales,y0=out;
        require(strata_glm_fp4_pack_nvfp4(r,k,q,nq,s,ns,y,ny)!=0,"invalid packing accepted");
        require(codes==q0&&scales==s0&&out==y0,"rejected packing wrote output");
    };
    bad(0,64,codes.data(),32,scales.data(),4,out.data(),36);
    bad(1,32,codes.data(),32,scales.data(),4,out.data(),36);
    bad(std::numeric_limits<size_t>::max(),64,codes.data(),32,scales.data(),4,out.data(),36);
    bad(1,64,codes.data(),31,scales.data(),4,out.data(),36);
    bad(1,64,codes.data(),32,scales.data(),3,out.data(),36);
    bad(1,64,codes.data(),32,scales.data(),4,out.data(),35);
    bad(1,64,nullptr,32,scales.data(),4,out.data(),36);
    bad(1,64,codes.data(),32,nullptr,4,out.data(),36);
    bad(1,64,codes.data(),32,scales.data(),4,nullptr,36);
    bad(1,64,codes.data(),32,scales.data(),4,codes.data(),36);
    bad(1,64,codes.data(),32,scales.data(),4,codes.data()+1,36);
    bad(1,64,codes.data(),32,scales.data(),4,scales.data(),36);
    for(int invalid=127;invalid<256;++invalid) {
        scales.back()=uint8_t(invalid);
        bad(1,64,codes.data(),32,scales.data(),4,out.data(),36);
    }
    std::cout<<"PASS rejection without writes: dimensions, overflow, extents, nulls, overlap, invalid scales\n";
}
int main(int argc,char** argv){try {
    require(argc<=2,"usage: glm-fp4-nvpack-test [PARTIAL_MODEL_DIR]");
    GlmFp4Matrix m;m.rows=127;m.cols=64;m.global=.375f;
    m.codes.resize(m.rows*m.cols/2);m.scales.resize(m.rows*m.cols/16);
    for(size_t i=0;i<m.codes.size();++i)m.codes[i]=uint8_t((2*i%16)|((2*i+1)%16)<<4);
    for(size_t i=0;i<m.scales.size();++i)m.scales[i]=uint8_t(i/4);
    check(m);rejects();
    m.rows=5;m.cols=4096;m.codes.resize(m.rows*m.cols/2);m.scales.resize(m.rows*m.cols/16);
    for(size_t i=0;i<m.codes.size();++i)m.codes[i]=uint8_t(i*73+19);
    for(size_t i=0;i<m.scales.size();++i)m.scales[i]=uint8_t(i%127);
    check(m);
    if(argc==2) {
        strata::artifact::Safetensors ts(argv[1]);
        for(int layer:{3,4})for(const auto* projection:{"gate_proj","up_proj","down_proj"})
            check(strata::artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+projection));
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
