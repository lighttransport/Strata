#include "strata/host/glm_fp4.h"
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
static void require(bool ok) {if(!ok)throw std::runtime_error("FP4 codec assertion failed");}
struct Plane {
    size_t rows,cols;
    std::vector<uint8_t> bases,codes,want,out;
    std::vector<uint16_t> offsets,exceptions;
    std::vector<uint32_t> pages;
    Plane(size_t r,size_t c):rows(r),cols(c),bases(r),want(r*c),out(r*c) {
        for(size_t i=0;i<r;++i)bases[i]=uint8_t(i*7);
        for(size_t i=0;i<r*c;++i)want[i]=uint8_t(bases[i/c]+((i/c+3*(i%c))&15));
        size_t tile=0,field=0;
        for(size_t r0=0;r0<r;r0+=128)for(size_t c0=0;c0<c;c0+=4,++tile) {
            if(tile%32==0){pages.push_back(exceptions.size()*sizeof(uint16_t));offsets.push_back(0);}
            for(size_t col=c0;col<c && col<c0+4;col+=2)for(size_t row=r0;row<r && row<r0+128;row+=32) {
                const size_t width=std::min(size_t(2),c-col);
                for(size_t rr=0;rr<32;++rr)for(size_t cc=0;cc<width;++cc,++field) {
                    if(field%2==0)codes.push_back(0);
                    codes.back()|=uint8_t(((row+rr+3*(col+cc))&15)<<((field%2)*4));
                }
                exceptions.push_back(uint16_t(233u<<8));want[row*c+col]=233;
                const unsigned last=width==2?63:62;
                exceptions.push_back(uint16_t(last|(17u<<8)));want[(row+31)*c+col+width-1]=17;
                offsets.push_back(uint16_t(exceptions.size()-pages.back()/sizeof(uint16_t)));
            }
        }
    }
    int decode(){return strata_fp4_csf32_decode(rows,cols,bases.data(),bases.size(),codes.data(),codes.size(),offsets.data(),offsets.size(),pages.data(),pages.size(),exceptions.data(),exceptions.size(),out.data(),out.size());}
    int validate(){return strata_fp4_csf32_validate(rows,cols,bases.data(),bases.size(),codes.data(),codes.size(),offsets.data(),offsets.size(),pages.data(),pages.size(),exceptions.data(),exceptions.size());}
};
int main(){try {
    require(std::bit_cast<uint32_t>(strata_fp4_e2m1(8))==0x80000000u);
    const float expected[8]={0,.5,1,1.5,2,3,4,6};
    for(unsigned i=0;i<16;++i)require(strata_fp4_e2m1(i)==(i&8?-expected[i&7]:expected[i&7]));
    for(unsigned i=0;i<256;++i) {
        const unsigned e=(i>>3)&15,m=i&7;
        const double ref=e?std::ldexp(1.+m/8.,int(e)-7):std::ldexp(m/8.,-6);
        if((i&127)==127)require(std::isnan(strata_fp4_e4m3(i)));
        else require(double(strata_fp4_e4m3(i))==(i&128?-ref:ref));
        if(i==255)require(std::isnan(strata_fp4_e8m0(i)));
        else require(strata_fp4_e8m0(i)==std::ldexp(1.f,int(i)-127));
    }
    for(auto geometry: {std::pair<size_t,size_t>{128,4},{160,67},{128,132},{4096,128}}) {
        Plane p(geometry.first,geometry.second);require(!p.validate());require(!p.decode());require(p.out==p.want);
        p.exceptions[1]=p.exceptions[0];require(p.decode()!=0);require(p.validate()!=0);
    }
    Plane p(128,132);++p.pages[1];require(p.decode()!=0);require(p.validate()!=0);--p.pages[1];
    p.offsets.pop_back();require(p.decode()!=0);require(p.validate()!=0);
    require(strata_fp4_csf32_decode(SIZE_MAX,32,nullptr,0,nullptr,0,nullptr,0,nullptr,0,nullptr,0,nullptr,0)!=0);
    std::cout<<"FP4 codecs PASS (all encodings, clipped tiles, page resets, malformed metadata)\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
