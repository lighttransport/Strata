#include "strata/host/glm_fp4.h"
#include "strata/artifact/glm_fp4.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace strata::kernels::cpu;
static void require(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
static void check_real_gu(const char* directory) {
    strata::artifact::Safetensors ts(directory);
    for(int layer:{3,4}) {
        const std::string prefix="model.language_model.layers."+std::to_string(layer)+".mlp.experts.0.";
        const auto g=strata::artifact::glm_load_fp4(ts,prefix+"gate_proj");
        const auto u=strata::artifact::glm_load_fp4(ts,prefix+"up_proj");
        require(g.rows==u.rows&&g.cols==u.cols,"incompatible real GU shape");
        const size_t stride=g.cols/64*36;
        std::vector<uint8_t>wg(g.rows*stride),wu(wg.size()),pg(wg.size()),pu(wg.size());
        auto pack=[&](const auto& m,auto& original,auto& packed){
            require(!strata_glm_fp4_pack_nvfp4(m.rows,m.cols,m.codes.data(),m.codes.size(),m.scales.data(),m.scales.size(),original.data(),original.size()),"real GU original packing failed");
            require(!strata_glm_nvfp4_repack_k32(original.data(),packed.data(),packed.size()),"real GU K32 packing failed");
        };
        pack(g,wg,pg);pack(u,wu,pu);
        for(int nt:{1,4,9})for(float amplitude:{1.f,4096.f,-4096.f}) {
            std::vector<block_q8_0>q(nt*g.cols/32);
            std::vector<float>x(g.cols),out(nt*g.rows),expected(out.size());
            std::vector<const void*>ap(nt);std::vector<float*>op(nt);
            size_t high_g=0,high_u=0,low_u=0;
            for(int t=0;t<nt;++t) {
                for(size_t j=0;j<g.cols;++j)x[j]=amplitude*(1.f+float((j+t)%7)/16);
                ap[t]=q.data()+t*g.cols/32;op[t]=out.data()+t*g.rows;
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(),q.data()+t*g.cols/32,g.cols);
                for(size_t r=0;r<g.rows;++r) {
                    float gv,uv;
                    const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_NVFP4)->vec_dot;
                    dot(g.cols,&gv,0,wg.data()+r*stride,0,ap[t],0,1);dot(g.cols,&uv,0,wu.data()+r*stride,0,ap[t],0,1);
                    gv*=g.global;uv*=u.global;high_g+=gv>10;high_u+=uv>10;low_u+=uv<-10;
                    gv=std::fmin(gv,10.f);uv=std::fmax(-10.f,std::fmin(uv,10.f));
                    expected[t*g.rows+r]=(gv/(1.f+std::exp(-gv)))*uv;
                }
            }
            nvfp4_256_k32_gu_rows_clamped(pg.data(),stride,0,int(g.cols),ap.data(),nt,op.data(),0,int(g.rows),g.global,u.global,10.f,pu.data());
            require(!std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)),"real K32 GU mismatch");
            if(amplitude!=1)require(high_g&&high_u&&low_u,"real GU stress missed a clamp branch");
            std::cout<<"PASS real GU layer="<<layer<<" tokens="<<nt<<" amplitude="<<amplitude<<" gate_high="<<high_g<<" up_high="<<high_u<<" up_low="<<low_u<<" bit_exact=1\n";
        }
    }
}
static void check_real(const char* directory) {
    strata::artifact::Safetensors ts(directory);
    for(int layer:{3,4})for(const char* name:{"gate_proj","up_proj","down_proj"}) {
        const auto m=strata::artifact::glm_load_fp4(ts,"model.language_model.layers."+std::to_string(layer)+".mlp.experts.0."+name);
        const size_t stride=m.cols/64*36;
        std::vector<uint8_t>original(m.rows*stride),packed(original.size());
        require(!strata_glm_fp4_pack_nvfp4(m.rows,m.cols,m.codes.data(),m.codes.size(),m.scales.data(),m.scales.size(),original.data(),original.size()),"real NVFP4 packing failed");
        require(!strata_glm_nvfp4_repack_k32(original.data(),packed.data(),packed.size()),"real K32 packing failed");
        for(int nt:{1,4,9}) {
            std::vector<block_q8_0>q(nt*m.cols/32);
            std::vector<float>x(m.cols),out(nt*m.rows),expected(out.size());
            std::vector<const void*>ap(nt);std::vector<float*>op(nt);
            for(int t=0;t<nt;++t) {
                for(size_t j=0;j<m.cols;++j)x[j]=float(int((j*19+t*37)%257)-128)/128;
                ap[t]=q.data()+t*m.cols/32;op[t]=out.data()+t*m.rows;
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(),q.data()+t*m.cols/32,m.cols);
                for(size_t r=0;r<m.rows;++r){float v;ggml_get_type_traits_cpu(GGML_TYPE_NVFP4)->vec_dot(m.cols,&v,0,original.data()+r*stride,0,ap[t],0,1);expected[t*m.rows+r]=v*m.global;}
            }
            nvfp4_256_k32_rows(packed.data(),stride,int(m.cols),ap.data(),nt,op.data(),0,int(m.rows),m.global);
            require(!std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)),"real K32 dot mismatch");
            std::cout<<"PASS real layer="<<layer<<" projection="<<name<<" tokens="<<nt<<" rows="<<m.rows<<" K="<<m.cols<<" bit_exact=1\n";
        }
    }
}
int main(int argc,char** argv){try {
    require(argc<=2,"usage: glm_nvfp4_k32_dot_test [PARTIAL_MODEL_DIR]");
    if(!cpu_avx2_ok())return 77;
    ggml_cpu_init();size_t cases=0;
    for(int k:{64,128,640,4096})for(int nt:{1,2,3,4,5,6,7,8,9,16,32})
    for(uint16_t scale:{0x0000,0x0001,0x03ff,0x0400,0x3c00,0x7bff,0xbc00})for(int pattern=0;pattern<4;++pattern) {
        constexpr int rows=16;const int nb=k/64;
        const size_t stride=size_t(nb)*sizeof(block_nvfp4)+7;
        std::vector<block_nvfp4>w(rows*nb);
        for(size_t b=0;b<w.size();++b){for(int j=0;j<4;++j)w[b].d[j]=(b*4+j)%127;for(int j=0;j<32;++j)w[b].qs[j]=(b*73+j*19)&255;}
        std::vector<uint8_t>packed(rows*stride,0xa5);
        for(int r=0;r<rows;++r)require(!strata_glm_nvfp4_repack_k32(reinterpret_cast<const uint8_t*>(w.data()+r*nb),packed.data()+r*stride,nb*sizeof(block_nvfp4)),"packing failed");
        std::vector<block_q8_0>q(nt*k/32);
        for(size_t b=0;b<q.size();++b){q[b].d=scale;for(int j=0;j<32;++j)q[b].qs[j]=pattern==3?-128:pattern==0?(j%2?127:-127):int((b*32+j)%255)-127;}
        if(pattern==2)for(int t=0;t<nt;++t)q[(t+1)*k/32-1].qs[31]=-128;
        std::vector<float>out(nt*rows),expected(nt*rows);std::vector<const void*>ap(nt);std::vector<float*>op(nt);
        const float global=pattern%2?-.375f:1.25f;
        for(int t=0;t<nt;++t){ap[t]=q.data()+t*k/32;op[t]=out.data()+t*rows;for(int r=0;r<rows;++r){float value;ggml_get_type_traits_cpu(GGML_TYPE_NVFP4)->vec_dot(k,&value,0,w.data()+r*nb,0,ap[t],0,1);expected[t*rows+r]=value*global;}}
        nvfp4_256_k32_rows(packed.data(),stride,k,ap.data(),nt,op.data(),0,rows,global);
        require(!std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)),"full rows mismatch");
        std::fill(out.begin(),out.end(),-12345.f);
        nvfp4_256_k32_rows(packed.data(),stride,k,ap.data(),nt,op.data(),1,rows-2,global);
        for(int t=0;t<nt;++t){require(out[t*rows]==-12345.f&&out[t*rows+rows-2]==-12345.f&&out[t*rows+rows-1]==-12345.f,"output guard changed");require(!std::memcmp(out.data()+t*rows+1,expected.data()+t*rows+1,(rows-3)*sizeof(float)),"partial rows mismatch");}
        nvfp4_256_rows(reinterpret_cast<const uint8_t*>(w.data()),nb*sizeof(block_nvfp4),k,ap.data(),nt,op.data(),0,rows,global);
        require(!std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)),"ordinary full rows mismatch");
        std::fill(out.begin(),out.end(),-12345.f);
        nvfp4_256_rows(reinterpret_cast<const uint8_t*>(w.data()),nb*sizeof(block_nvfp4),k,ap.data(),nt,op.data(),1,rows-2,global);
        for(int t=0;t<nt;++t){require(out[t*rows]==-12345.f&&out[t*rows+rows-2]==-12345.f&&out[t*rows+rows-1]==-12345.f,"ordinary row guard changed");require(!std::memcmp(out.data()+t*rows+1,expected.data()+t*rows+1,(rows-3)*sizeof(float)),"ordinary partial rows mismatch");}
        const auto gate_values=expected;
        std::vector<float>up_values(expected.size());
        std::vector<uint8_t>combined=packed;combined.resize(packed.size()*2);
        for(int r=0;r<rows;++r)std::memcpy(combined.data()+packed.size()+r*stride,packed.data()+(rows-1-r)*stride,stride);
        // Ordinary-layout GU shares this independent GGML-dot oracle with K32.
        std::vector<uint8_t> ordinary(combined.size(), 0xa5);
        for(int r=0;r<rows;++r) {
            std::memcpy(ordinary.data()+r*stride,w.data()+r*nb,nb*sizeof(block_nvfp4));
            std::memcpy(ordinary.data()+packed.size()+r*stride,w.data()+(rows-1-r)*nb,nb*sizeof(block_nvfp4));
        }
        const float up_global=-.5f*global;
        for(int t=0;t<nt;++t)for(int r=0;r<rows;++r){float v;ggml_get_type_traits_cpu(GGML_TYPE_NVFP4)->vec_dot(k,&v,0,w.data()+(rows-1-r)*nb,0,ap[t],0,1);up_values[t*rows+r]=v*up_global;}
        for(float limit:{0.f,10.f}) {
            for(size_t i=0;i<expected.size();++i) {
                float g=gate_values[i],u=up_values[i];
                if(limit>0){g=std::fmin(g,limit);u=std::fmax(-limit,std::fmin(u,limit));}
                expected[i]=(g/(1.f+std::exp(-g)))*u;
            }
            nvfp4_256_gu_rows_clamped(ordinary.data(),stride,packed.size(),k,ap.data(),nt,op.data(),0,rows,global,up_global,limit);
            require(!std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)),"ordinary GU mismatch");
            if(limit==0) {
                nvfp4_256_gu_rows(ordinary.data(),stride,packed.size(),k,ap.data(),nt,op.data(),0,rows,global,up_global);
                require(!std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)),"ordinary unclamped GU mismatch");
            }
            std::fill(out.begin(),out.end(),-12345.f);
            nvfp4_256_gu_rows_clamped(ordinary.data(),stride,0,k,ap.data(),nt,op.data(),1,rows-2,global,up_global,limit,ordinary.data()+packed.size());
            for(int t=0;t<nt;++t){require(out[t*rows]==-12345.f&&out[t*rows+rows-2]==-12345.f&&out[t*rows+rows-1]==-12345.f,"ordinary GU guard changed");require(!std::memcmp(out.data()+t*rows+1,expected.data()+t*rows+1,(rows-3)*sizeof(float)),"ordinary separate GU mismatch");}
            nvfp4_256_k32_gu_rows_clamped(combined.data(),stride,packed.size(),k,ap.data(),nt,op.data(),0,rows,global,up_global,limit);
            require(!std::memcmp(out.data(),expected.data(),out.size()*sizeof(float)),"combined K32 GU mismatch");
            std::fill(out.begin(),out.end(),-12345.f);
            nvfp4_256_k32_gu_rows_clamped(packed.data(),stride,0,k,ap.data(),nt,op.data(),1,rows-2,global,up_global,limit,combined.data()+packed.size());
            for(int t=0;t<nt;++t){require(out[t*rows]==-12345.f&&out[t*rows+rows-2]==-12345.f&&out[t*rows+rows-1]==-12345.f,"GU guard changed");require(!std::memcmp(out.data()+t*rows+1,expected.data()+t*rows+1,(rows-3)*sizeof(float)),"separate K32 GU mismatch");}
        }
        ++cases;
    }
    float sentinel=123;float* output[]={&sentinel};
    auto rejects=[&](int n,int nt,size_t stride,int first,int last){bool caught=false;try{nvfp4_256_k32_rows(nullptr,stride,n,nullptr,nt,output,first,last);}catch(const std::invalid_argument&){caught=true;}require(caught&&sentinel==123,"invalid request wrote output or succeeded");};
    rejects(0,1,36,0,1);rejects(65,1,36,0,1);rejects(4160,1,36,0,1);
    rejects(64,0,36,0,1);rejects(64,33,36,0,1);rejects(64,1,35,0,1);
    rejects(64,1,36,-1,1);rejects(64,1,36,2,1);rejects(64,1,36,0,1);
    nvfp4_256_k32_rows(nullptr,36,64,nullptr,1,nullptr,0,0);
    NativeFmt format;std::string error;
    require(native_fmt(10,11,256,256,format,error),"Q2/Q3 format unavailable");
    format.nvfp4_k32=true;
    bool rejected_gu=false,rejected_down=false;
    try{native_gu_rows(format,nullptr,nullptr,1,output,0,1);}catch(const std::invalid_argument&){rejected_gu=true;}
    try{native_down_rows(format,nullptr,nullptr,1,output,0,1);}catch(const std::invalid_argument&){rejected_down=true;}
    require(rejected_gu&&rejected_down&&sentinel==123,"incompatible K32 format accepted or wrote output");
    require(native_fmt(kNvfp4Type,kNvfp4Type,64,64,format,error)&&!format.nvfp4_k32,"native_fmt retained packed-layout selection");
    uint8_t zero_weights[36]={};block_q8_0 zero_act[2]={};const void* zero_ap[]={zero_act};
    bool offset_rejected=false;
    try{nvfp4_256_k32_gu_rows_clamped(zero_weights,36,SIZE_MAX,64,zero_ap,1,output,0,1,1,1,10);}
    catch(const std::invalid_argument&){offset_rejected=true;}
    require(offset_rejected&&sentinel==123,"GU offset overflow was not rejected before writes");
    nvfp4_256_k32_gu_rows_clamped(zero_weights,36,SIZE_MAX,64,zero_ap,1,output,0,1,1,1,10,zero_weights);
    require(sentinel==0,"separate GU incorrectly used combined offset");
    std::cout<<"PASS native K32 dot cases="<<cases<<" tokens=1..9,16,32 raw_min=1 finite_scales=1 padded_stride=1 partial_rows=1 clamped_gu=1 separate_up=1 bit_exact=1\n";
    if(argc==2){check_real(argv[1]);check_real_gu(argv[1]);}
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
