#include "native_finite.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <stdexcept>
int main() {
    using strata::experiment::native_all_finite;
    size_t checks=0;
    auto check=[&](const float* x,size_t n) {
        bool expected=true;
        for(size_t i=0;i<n;++i)expected &= std::isfinite(x[i]);
        if(native_all_finite(x,n)!=expected)throw std::runtime_error("finite classification mismatch");
        ++checks;
    };
    check(nullptr,0);
    std::mt19937 rng(831);
    for(size_t n:{0,1,3,4,5,7,8,15,16,17,4096,65536}) {
        std::vector<float> x(n+1);
        for(int repeat=0;repeat<40;++repeat) {
            for(auto& v:x)v=std::bit_cast<float>(uint32_t(rng()));
            check(x.data(),n);check(x.data()+1,n);
        }
        if(!n)continue;
        for(uint32_t exponent=0;exponent<256;++exponent)
            for(uint32_t sign:{0u,0x80000000u})for(uint32_t mantissa:{0u,1u,0x400000u,0x7fffffu}) {
                std::fill(x.begin(),x.end(),1.f);
                const float v=std::bit_cast<float>(sign|(exponent<<23)|mantissa);
                for(size_t pos:{size_t(0),n/2,n-1}) {
                    x[pos]=v;check(x.data(),n);x[pos]=1.f;
                }
            }
    }
    std::cout<<"PASS finite_checks="<<checks<<"\n";
}
