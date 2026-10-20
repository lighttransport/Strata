// CPU-only scratch-kernel qualification. Link expanded q23_avx2.cpp with
// -Dq23_rows=q3_expanded_rows -Dq23_gu_rows=q3_expanded_gu_rows alongside baseline.
#include "strata/kernels/cpu/q23_avx2.hpp"
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
namespace strata::kernels::cpu {
void q3_expanded_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int);
void q3_expanded_gu_rows(int,const uint8_t*,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int,float,bool,float);
}
using namespace strata::kernels::cpu;
static void require(bool b) {if(!b)throw std::runtime_error("expanded Q3 parity failed");}
int main() {try {
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
                q3_expanded_rows(11,eg.data(),stride,k,ap,nt,gp,1,rows-1);
                for(int t=0;t<nt;++t)for(int r=1;r<rows-1;++r) {
                    double value=0,norm=0;
                    for(int j=0;j<k;++j){double v=double(decoded[r*k+j])*acts[t][j/256].d*acts[t][j/256].qs[j%256];value+=v;norm+=std::abs(v);}
                    require(std::isfinite(got[t][r])&&std::abs(got[t][r]-value)<=1e-5*norm+1e-5);
                }
            } else {
                const float clamp=mode==1?0.f:7.f; const bool canon=mode>=3; const float skip=mode==4?.1f:0.f;
                q23_gu_rows(11,gate.data(),up.data(),oldstride,k,ap,nt,rp,1,rows-1,clamp,canon,skip);
                q3_expanded_gu_rows(11,eg.data(),eu.data(),stride,k,ap,nt,gp,1,rows-1,clamp,canon,skip);
            }
            for(int t=0;t<nt;++t)require(!std::memcmp(ref[t].data(),got[t].data(),rows*sizeof(float)));
            ++checks;
        }
    }
    std::cout<<"PASS expanded Q3 configurations="<<checks<<" K=256/512/768/2048/4096 nt=1..8, raw Q8 extremes, padded strides, partial guards, GU clamp/canonical/skip\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
