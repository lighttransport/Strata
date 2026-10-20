#include "strata/kernels/cpu/q23_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/host/q3_layout.h"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <vector>
#include <random>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>
using namespace strata::kernels::cpu;
static void require(bool b) {if(!b)throw std::runtime_error("expanded Q3 parity failed");}
static void bounds() {
    uint8_t w[228]={};block_q8_K a{};float output[4]={-9876,-9876,-9876,-9876};
    const void* ap[]={&a,nullptr};float* op[]={output,output};
    auto reject=[&](auto call) {
        bool threw=false;try{call();}catch(const std::invalid_argument&){threw=true;}
        require(threw);for(float v:output)require(v==-9876);
    };
    auto dot=[&](size_t stride,int n,int nt,int lo,int hi) {q3k_expanded_rows(w,stride,n,ap,nt,op,lo,hi);};
    for(int n:{0,255,257,4352})reject([&]{dot(114,n,1,0,1);});
    for(int nt:{0,9})reject([&]{dot(114,256,nt,0,1);});
    reject([&]{dot(113,256,1,0,1);});
    reject([&]{dot(114,256,1,-1,1);});
    reject([&]{dot(114,256,1,2,1);});
    reject([&]{dot(SIZE_MAX,256,1,0,2);});
    reject([&]{dot(114,256,2,0,1);}); // late null token must precede writes
    reject([&]{q3k_expanded_rows(nullptr,114,256,ap,1,op,0,1);});
    reject([&]{q3k_expanded_rows(w,114,256,nullptr,1,op,0,1);});
    reject([&]{q3k_expanded_rows(w,114,256,ap,1,nullptr,0,1);});
    ap[1]=&a;op[1]=nullptr;
    reject([&]{dot(114,256,2,0,1);});
    reject([&]{q3k_expanded_gu_rows(w,nullptr,114,256,ap,1,op,0,1,7);});
    q3k_expanded_rows(nullptr,114,256,nullptr,1,nullptr,2,2);
    q3k_expanded_gu_rows(nullptr,nullptr,114,256,nullptr,1,nullptr,2,2,7);
}
static void native_format_checks() {
    NativeFmt f;std::string err;require(native_fmt(10,11,256,256,f,err));
    f.q3_expanded_down=true;f.d_row=114;f.bytes=f.down_off+f.d_row*256;
    const auto valid=f;
    for(int test=0;test<12;++test) {
        f=valid;
        switch(test) {
        case 0:f.d_type=10;break;case 1:f.gu_type=11;break;
        case 2:f.gu_act=0;break;case 3:f.lossless=true;break;
        case 4:f.q23_layout=1;break;case 5:f.nvfp4_k32=true;break;
        case 6:f.d_row=110;break;case 7:++f.bytes;break;
        case 8:++f.down_off;break;case 9:f.tail_off=1;break;
        case 10:f.d_act=0;break;case 11:f.n_ff=257;break;
        }
        for(bool down:{false,true}) {
            float output=-9876;float* op[]={&output};bool threw=false;
            try {
                if(down)native_down_rows(f,nullptr,nullptr,1,op,0,1);
                else native_gu_rows(f,nullptr,nullptr,1,op,0,1);
            }catch(const std::invalid_argument&){threw=true;}
            require(threw&&output==-9876);
        }
    }
    f=valid;require(native_fmt(10,11,256,256,f,err));require(!f.q3_expanded_down);
}
static void native_down_parity() {
    NativeFmt f;std::string err;require(native_fmt(10,11,256,256,f,err));
    const size_t original_stride=f.d_row;
    std::vector<uint8_t> original(original_stride*256);
    std::mt19937 rng(519);
    for(size_t b=0;b<256;++b) {
        for(size_t j=0;j<108;++j)original[b*110+j]=uint8_t(rng());
        uint16_t h=0x2400;std::memcpy(original.data()+b*110+108,&h,2);
    }
    std::vector<uint8_t> blob(f.bytes);
    std::memcpy(blob.data()+f.down_off,original.data(),original.size());
    const auto original_blob=blob;const size_t old_bytes=f.bytes;
    bool wrong_size=false;
    try { native_prepare_q3_expanded_down(f,blob.data(),blob.size()-1,f,blob); }
    catch(const std::invalid_argument&) { wrong_size=true; }
    require(wrong_size && blob==original_blob && f.bytes==old_bytes && !f.q3_expanded_down);
    // Source/destination aliasing is intentional: conversion commits after success.
    native_prepare_q3_expanded_down(f,blob.data(),blob.size(),f,blob);
    require(f.q3_expanded_down && f.d_row==114 && blob.size()==f.bytes);
    const auto preserved=blob;const auto bytes=f.bytes;
    bool rejected=false;
    try { native_prepare_q3_expanded_down(f,blob.data(),blob.size(),f,blob); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected && blob==preserved && f.bytes==bytes && f.q3_expanded_down);
    block_q8_K act[4]={};const void* ap[4];float* refp[4];float* gotp[4];
    float ref[4][256],got[4][256];
    for(int t=0;t<4;++t) {
        act[t].d=.03125f;for(auto& v:act[t].qs)v=int8_t(rng());
        ap[t]=&act[t];refp[t]=ref[t];gotp[t]=got[t];
    }
    for(bool canon:{false,true})for(int nt:{1,4})for(bool separate:{false,true}) {
        f.canon=canon;
        for(int t=0;t<4;++t)for(int r=0;r<256;++r)ref[t][r]=got[t][r]=-9876;
        q23_rows(11,original.data(),original_stride,256,ap,nt,refp,1,255);
        native_down_rows(f,separate?nullptr:blob.data(),ap,nt,gotp,1,255,separate?blob.data()+f.down_off:nullptr);
        require(!std::memcmp(ref,got,sizeof ref));
    }
}
int main() {try {
    if(!cpu_avx2_ok())return 77;
    bounds();native_format_checks();native_down_parity();
    ggml_cpu_init(); std::mt19937 rng(90217); size_t checks=0;
    for(int k:{256,512,768,2048,4096})for(int pattern=0;pattern<6;++pattern) {
        constexpr int rows=9;
        const size_t oldstride=k/256*110, stride=k/256*114+7;
        std::vector<uint8_t> gate(rows*oldstride),up(gate.size());
        // Arbitrary codes cover the full representable scale/code range, not
        // only the subset selected by a quantizer. Half scales stay finite.
        for(auto* w:{&gate,&up})for(size_t b=0;b<w->size()/110;++b) {
            for(size_t j=0;j<108;++j)(*w)[b*110+j]=uint8_t(rng());
            const uint16_t h[]={0,0x0001,0x1400,0x3c00,0xbc00,0x4000};
            std::memcpy(w->data()+b*110+108,&h[(b+pattern)%6],2);
        }
        std::vector<uint8_t> eg(rows*stride,0xa5),eu(rows*stride,0x5a);
        for(int r=0;r<rows;++r) {
            require(!strata_q3k_expand_scales(gate.data()+r*oldstride,oldstride,eg.data()+r*stride,stride-7));
            require(!strata_q3k_expand_scales(up.data()+r*oldstride,oldstride,eu.data()+r*stride,stride-7));
        }
        std::vector<std::vector<block_q8_K>> acts(8,std::vector<block_q8_K>(k/256));
        const void* ap[8];
        for(int t=0;t<8;++t) {
            for(auto& a:acts[t]) {
                a.d=.03125f;
                for(int j=0;j<256;++j) a.qs[j]=pattern==0?0:pattern==1?-128:pattern==2?127:pattern==3?(j%2?-128:127):int8_t(rng());
                for(int j=0;j<16;++j){int sum=0;for(int i=0;i<16;++i)sum+=a.qs[j*16+i];a.bsums[j]=int16_t(sum);}
            }
            ap[t]=acts[t].data();
        }
        std::vector<float> decoded(rows*k);
        ggml_get_type_traits(GGML_TYPE_Q3_K)->to_float(gate.data(),decoded.data(),decoded.size());
        for(int nt=1;nt<=8;++nt)for(int mode=0;mode<5;++mode) {
            std::vector<std::vector<float>> ref(nt,std::vector<float>(rows,-9876)),got=ref;
            float* rp[8];float* gp[8];for(int t=0;t<nt;++t){rp[t]=ref[t].data();gp[t]=got[t].data();}
            if(mode==0) {
                q23_rows(11,gate.data(),oldstride,k,ap,nt,rp,1,rows-1);
                q3k_expanded_rows(eg.data(),stride,k,ap,nt,gp,1,rows-1);
                for(int t=0;t<nt;++t)for(int r=1;r<rows-1;++r) {
                    double value=0,norm=0;
                    for(int j=0;j<k;++j){double v=double(decoded[r*k+j])*acts[t][j/256].d*acts[t][j/256].qs[j%256];value+=v;norm+=std::abs(v);}
                    require(std::isfinite(got[t][r])&&std::abs(got[t][r]-value)<=1e-5*norm+1e-5);
                }
            } else {
                const float clamp=mode==1?0.f:7.f; const bool canon=mode>=3; const float skip=mode==4?.1f:0.f;
                q23_gu_rows(11,gate.data(),up.data(),oldstride,k,ap,nt,rp,1,rows-1,clamp,canon,skip);
                q3k_expanded_gu_rows(eg.data(),eu.data(),stride,k,ap,nt,gp,1,rows-1,clamp,canon,skip);
            }
            for(int t=0;t<nt;++t)require(!std::memcmp(ref[t].data(),got[t].data(),rows*sizeof(float)));
            ++checks;
        }
    }
    std::cout<<"PASS expanded Q3 configurations="<<checks<<" K=256/512/768/2048/4096 nt=1..8, raw Q8 extremes, padded strides, partial guards, GU clamp/canonical/skip\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
