#pragma once
#include "strata/kernels/cpu/exl3/math.hpp"
#include <algorithm>
namespace strata::cpu::exl3 {
// Private runtime W8A32 layout: one FP16 scale and 32 signed INT8 weights.
// Source BF16 tensors and EXL3 expert files are never modified.
struct DenseQ8Block { uint16_t scale; int8_t values[32]; };
static_assert(sizeof(DenseQ8Block)==34);
inline void dense_q8_row(const uint16_t* src,DenseQ8Block* dst,size_t in) {
    if(in%32)throw std::runtime_error("EXL3 Q8: row width must be divisible by 32");
    for(size_t b=0;b<in/32;++b) {
        float values[32],peak=0;
        for(unsigned i=0;i<32;++i) {
            values[i]=std::bit_cast<float>(uint32_t(src[b*32+i])<<16);
            if(!std::isfinite(values[i]))throw std::runtime_error("EXL3 Q8: nonfinite source weight");
            peak=std::max(peak,std::abs(values[i]));
        }
        auto& block=dst[b];block.scale=half_round(peak/127.f);
        if(peak && !block.scale)block.scale=1;
        float scale=half_value(block.scale);
        if(!std::isfinite(scale))throw std::runtime_error("EXL3 Q8: scale exceeds FP16 range");
        for(unsigned i=0;i<32;++i)block.values[i]=scale?int8_t(std::clamp(std::round(values[i]/scale),-127.f,127.f)):0;
    }
}
}
namespace strata::cpu::exl3 {
struct DenseQ6Block { uint16_t scale; uint8_t values[24]; };
static_assert(sizeof(DenseQ6Block)==26);
inline void dense_q6_row(const uint16_t* src,DenseQ6Block* dst,size_t in) {
    if(in%32)throw std::runtime_error("EXL3 Q6: row width must be divisible by 32");
    for(size_t b=0;b<in/32;++b) {
        float values[32],peak=0;
        for(unsigned i=0;i<32;++i){values[i]=std::bit_cast<float>(uint32_t(src[b*32+i])<<16);if(!std::isfinite(values[i]))throw std::runtime_error("EXL3 Q6: nonfinite source weight");peak=std::max(peak,std::abs(values[i]));}
        auto& block=dst[b];block.scale=half_round(peak/31.f);if(peak&&!block.scale)block.scale=1;
        float scale=half_value(block.scale);if(!std::isfinite(scale))throw std::runtime_error("EXL3 Q6: scale exceeds FP16 range");
        std::fill(std::begin(block.values),std::end(block.values),0);
        for(unsigned i=0;i<32;++i) {
            int q=scale?int(std::clamp(std::round(values[i]/scale),-31.f,31.f)):0;unsigned bits=unsigned(q)&63,byte=i*6/8,shift=i*6%8;
            block.values[byte]|=uint8_t(bits<<shift);if(shift>2)block.values[byte+1]|=uint8_t(bits>>(8-shift));
        }
    }
}
inline float dense_q6_value(const DenseQ6Block& block,unsigned i) {
    unsigned byte=i*6/8,shift=i*6%8,value=block.values[byte];if(shift>2)value|=unsigned(block.values[byte+1])<<8;
    int q=(value>>shift)&63;if(q&32)q-=64;return half_value(block.scale)*q;
}
}
