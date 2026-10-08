#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels::cpu {
inline constexpr int kExpertBlockChannels=16;
inline constexpr int kExpertBlockCount=128;
struct ExpertBlockMask {
    std::array<uint64_t,2> words{};
    bool contains(int block)const {return block>=0&&block<128&&(words[block/64]&(uint64_t(1)<<(block%64)));}
    void add(int block) {
        if(block<0||block>=128)throw std::invalid_argument("expert block outside mask");
        words[block/64]|=uint64_t(1)<<(block%64);
    }
};
// Contribution energy is a ranking heuristic, not an error bound: it omits
// cross-channel cancellation and downstream amplification.
inline std::array<double,128> expert_block_energy(const float* hidden,const float* down_norm,int count) {
    if(!hidden||!down_norm||count!=2048)throw std::invalid_argument("expert blocks require 2048 channels");
    std::array<double,128> scores{};
    for(int i=0;i<count;++i) {
        if(!std::isfinite(hidden[i])||!std::isfinite(down_norm[i])||down_norm[i]<0)
            throw std::runtime_error("nonfinite activation or invalid down norm");
        scores[i/16]+=double(hidden[i])*hidden[i]*down_norm[i];
    }
    return scores;
}
inline ExpertBlockMask expert_block_topk(const std::array<double,128>& scores,int keep) {
    if(keep<1||keep>128)throw std::invalid_argument("retained expert blocks must be 1..128");
    std::array<int,128> order{};
    for(int i=0;i<128;++i){if(!std::isfinite(scores[i])||scores[i]<0)throw std::runtime_error("invalid expert block energy");order[i]=i;}
    std::partial_sort(order.begin(),order.begin()+keep,order.end(),[&](int a,int b){return scores[a]!=scores[b]?scores[a]>scores[b]:a<b;});
    ExpertBlockMask mask;for(int i=0;i<keep;++i)mask.add(order[i]);return mask;
}
inline void expert_block_zero(float* hidden,int count,const ExpertBlockMask& mask) {
    if(!hidden||count!=2048)throw std::invalid_argument("expert block mask geometry mismatch");
    for(int b=0;b<128;++b)if(!mask.contains(b))std::fill_n(hidden+b*16,16,0.f);
}
}
