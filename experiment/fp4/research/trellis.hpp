#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace research {
// Exact Viterbi for a fixed initial/final state. This is NOT globally optimal
// tail biting over all 65536 possible boundary states.
inline std::array<uint8_t,256> viterbi(const std::array<float,256>& target,
                                     const std::array<float,256>& weight,
                                     const std::vector<float>& book,uint16_t boundary) {
    constexpr unsigned states=65536;
    if(book.size()!=states)throw std::runtime_error("bad trellis codebook");
    std::vector<float> prev(states,INFINITY),next(states);
    std::vector<uint8_t> parent(size_t(256)*states);
    prev[boundary]=0;
    for(unsigned t=0;t<256;++t){
        if(!std::isfinite(target[t])||!std::isfinite(weight[t])||weight[t]<0)throw std::runtime_error("invalid trellis objective");
        int required=t>=248?int((boundary>>(2*(255-t)))&3):-1;
        for(unsigned s=0;s<states;++s){
            if(required>=0&&int(s&3)!=required){next[s]=INFINITY;continue;}
            unsigned base=s>>2,best=0;float cost=prev[base];
            for(unsigned b=1;b<4;++b)if(prev[base+(b<<14)]<cost){best=b;cost=prev[base+(b<<14)];}
            float error=book[s]-target[t];next[s]=cost+weight[t]*error*error;
            parent[size_t(t)*states+s]=uint8_t(best);
        }
        prev.swap(next);
    }
    if(!std::isfinite(prev[boundary]))throw std::runtime_error("no closed trellis path");
    std::array<uint8_t,256> branches{};unsigned s=boundary;
    for(int t=255;t>=0;--t){branches[t]=s&3;s=(s>>2)|(unsigned(parent[size_t(t)*states+s])<<14);}
    if(s!=boundary)throw std::runtime_error("trellis traceback boundary mismatch");
    return branches;
}
inline void pack_ring(const uint8_t* branches,uint8_t* p) {
    std::fill_n(p,64,0);
    for(unsigned t=0;t<256;++t){if(branches[t]>3)throw std::runtime_error("invalid branch");
        unsigned wi=t/16,shift=30-2*(t%16);uint32_t v=uint32_t(branches[t])<<shift;
        for(unsigned b=0;b<4;++b)p[wi*4+b]|=uint8_t(v>>(8*b));}
}
inline std::array<uint8_t,256> search_boundaries(const std::array<float,256>& target,
                                              const std::array<float,256>& weight,
                                              const std::vector<float>& book,uint16_t initial,unsigned trials) {
    if(trials<1||trials>8)throw std::runtime_error("boundary trials must be 1..8");
    std::array<uint8_t,256> best{};double best_cost=INFINITY;
    for(unsigned trial=0;trial<trials;++trial){
        uint16_t boundary=trial==0?initial:trial==1?0:uint16_t(initial^(0x9e37u*trial));
        auto path=viterbi(target,weight,book,boundary);double cost=0;uint16_t state=boundary;
        for(unsigned t=0;t<256;++t){state=uint16_t((state<<2)|path[t]);double error=double(book[state])-target[t];cost+=weight[t]*error*error;}
        if(state!=boundary)throw std::runtime_error("searched ring does not close");
        if(cost<best_cost){best_cost=cost;best=path;}
    }return best;
}
} // namespace research
