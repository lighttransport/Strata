#include "strata/kernels/cpu/input_quant_ready.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace strata::kernels::cpu;
static void require(bool ok) {
    if (!ok) throw std::runtime_error("input quantization readiness mismatch");
}
int main() {
    try {
        for (bool requested : {false,true}) for (bool remote : {false,true})
            for (bool check : {false,true})
                require(defer_input_quantization(requested,remote,check) ==
                        (requested && !remote && !check));
        for (int invalid : {0,65}) {
            bool caught=false;
            try { InputQuantReady ready(invalid); } catch(const std::invalid_argument&) { caught=true; }
            require(caught);
        }
        InputQuantReady edge(64);
        int calls=0;
        for (int token : {63,0,63,0}) edge.ensure(token,[&](int){++calls;});
        require(calls==2);
        for (int invalid : {-1,64}) {
            bool caught=false;
            try { edge.ensure(invalid,[](int){}); } catch(const std::out_of_range&) { caught=true; }
            require(caught);
        }
        InputQuantReady retry(1);
        try { retry.ensure(0,[](int){throw std::runtime_error("expected");}); } catch(const std::runtime_error&) {}
        calls=0;
        retry.ensure(0,[&](int){++calls;});
        require(calls==1);

        int cases=0;
        for (int type : {10,40}) for (int nt : {1,2,8}) {
            NativeFmt f; std::string error;
            require(native_fmt(type,type==10?11:40,4096,2048,f,error));
            std::vector<std::vector<uint8_t>> actual(nt,std::vector<uint8_t>(f.act_bytes));
            for (int mask=0;mask<(1<<nt);++mask) for(int layer=0;layer<2;++layer) {
                // New readiness with reused buffers must never inherit the preceding layer.
                InputQuantReady ready(nt);
                std::vector<int> counts(nt);
                std::vector<float> input(nt*4096);
                std::vector<std::vector<uint8_t>> expected(nt,std::vector<uint8_t>(f.act_bytes));
                for(int t=0;t<nt;++t) {
                    for(int j=0;j<4096;++j) input[t*4096+j]=float((j*19+t*31+layer*71)%257-128)/128;
                    native_quant_act(f,input.data()+t*4096,expected[t].data());
                    std::fill(actual[t].begin(),actual[t].end(),0xa5);
                }
                for(int repeat=0;repeat<3;++repeat) for(int t=nt-1;t>=0;--t)
                    if(mask&(1<<t)) ready.ensure(t,[&](int token) {
                        ++counts[token];
                        native_quant_act(f,input.data()+token*4096,actual[token].data());
                    });
                for(int t=0;t<nt;++t) {
                    require(counts[t]==bool(mask&(1<<t)));
                    if(mask&(1<<t)) require(actual[t]==expected[t]);
                    else require(std::all_of(actual[t].begin(),actual[t].end(),[](uint8_t v){return v==0xa5;}));
                }
                ++cases;
            }
        }
        std::cout<<"Input quantization readiness PASS "<<cases<<" demand/layer cases\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
